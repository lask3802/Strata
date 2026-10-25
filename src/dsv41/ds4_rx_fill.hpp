// src/dsv41/ds4_rx_fill.hpp - the remote experts' fills (EP U5): the expert server's set follows the main's heat.
//
// U4 sends the server the profile's hottest CPU-lane experts once, at the load.  U5 swaps them during decode: at a
// token boundary, the main's cache tick (cache_tick_swaps) picks, for some layers, a CPU-lane expert hotter than the
// coldest one the server holds (the 2080 Ti cache's own rule: heat[in] > heat[out] x margin + bonus), takes `out` off
// the server's list at once (the lane computes it from the next token), and queues a fill: in's bytes into out's slot.
// `in` is the server's only after the server says the bytes are there (FillOk), at the next token boundary.  So no
// request ever names an expert whose slot is being loaded (evict-then-load, no spare slot; ep-u5-plan.md 4-5).
//
// This file is the pure part (no socket, no GPU, no model), so it is tested alone (tests/dsv41/rx_fill_test.cpp):
//   - remote_pick / remote_choose: which (layer, in, out) to fill, over the layers with the most lane hits first;
//     a server-held expert that is also in this GPU's VRAM (a duplicate) leaves first, whatever its heat;
//   - FillBook: the fills a token (R), pending (Q), the bulk pace (bytes a second) and the pause while the link idles;
//   - ram_in: the RAM tier's next expert, never one the server holds (the RAM tier would save the lane nothing there).
// The server's list (the main's rx_has row of a layer) has three states here: 0 not held, kHeld, kFilling.  cache_in
// (ds4_rx.hpp) and ram_in treat any non-zero as the server's, so neither cache takes an expert on its way there.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::ds4::rx {

constexpr uint8_t kHeld = 1;      // the server holds it: requests may name it
constexpr uint8_t kFilling = 2;   // its fill is queued or under way: requests do not name it, no cache takes it

/// The RAM tier's next expert from a layer's CPU-lane hits: the hottest not in VRAM (slot < 0), not in RAM (ram < 0),
/// not `vin` (the VRAM tier's pick this tick) and not one the expert server holds or is filling (remote[e] != 0;
/// null: none, or a window the server does not serve - rx_row's rule, as cache_in).  -1: none
inline int ram_in(const std::vector<int>& hits, const int* slot, const int* ram, const float* heat, const uint8_t* remote,
                  int vin) {
    int best = -1;
    for (int e : hits)
        if (e != vin && slot[e] < 0 && ram[e] < 0 && (remote == nullptr || remote[e] == 0) &&
            (best < 0 || heat[e] > heat[best]))
            best = e;
    return best;
}

struct Fill {
    int layer = -1, in = -1, out = -1;
};

/// One layer's fill: `in` the hottest of the CPU-lane hits that is not in VRAM (slot < 0), not the VRAM tier's pick
/// this tick (vin) and not on the server's list (has[e] == 0); `out` a held expert (has[e] == kHeld, never kFilling):
/// the coldest one that is also in VRAM if there is one (the server's copy saves the lane nothing; any `in` beats it),
/// else the coldest held, and then only when heat[in] > heat[out] x margin + bonus.  {-1, -1}: none
inline Fill remote_pick(const std::vector<int>& hits, const int* slot, const float* heat, const uint8_t* has, int n_exp,
                        int vin, float margin, float bonus) {
    Fill f;
    int in = -1;
    for (int e : hits)
        if (e >= 0 && e < n_exp && e != vin && slot[e] < 0 && has[e] == 0 && (in < 0 || heat[e] > heat[in])) in = e;
    if (in < 0) return f;
    int dup = -1, cold = -1;
    for (int e = 0; e < n_exp; ++e) {
        if (has[e] != kHeld) continue;
        if (slot[e] >= 0 && (dup < 0 || heat[e] < heat[dup])) dup = e;
        if (cold < 0 || heat[e] < heat[cold]) cold = e;
    }
    if (dup >= 0) {
        f.in = in;
        f.out = dup;
    } else if (cold >= 0 && heat[in] > heat[cold] * margin + bonus) {
        f.in = in;
        f.out = cold;
    }
    return f;
}

/// The fill budget: at most R fills leave a token (tick), at most Q are pending (queued, not yet FillOk), the bytes
/// queued keep to the bulk pace (a bucket of pace_bps that holds `burst` bytes at most; a fill needs its bytes in it),
/// and none leaves while the link idles (rx::Backoff's spell: the server is late already).  pace_bps <= 0: no pace
class FillBook {
public:
    FillBook(int r, int q, double pace_bps, double burst) : r_(r), q_(q), pace_(pace_bps), burst_(burst) {}
    /// a token boundary at now_ms: the token's R again, the bucket refilled for the time since the last tick
    void tick(double now_ms, bool idle) {
        if (last_ms_ >= 0.0 && pace_ > 0.0)
            bucket_ = std::min(burst_, bucket_ + pace_ * std::max(0.0, now_ms - last_ms_) / 1000.0);
        last_ms_ = now_ms;
        left_ = r_;
        idle_ = idle;
    }
    /// whether a fill of `bytes` may leave now
    bool can(double bytes) const {
        return !idle_ && left_ > 0 && pending_ < q_ && (pace_ <= 0.0 || bytes <= bucket_);
    }
    /// whether no further fill may leave this token, whatever its size (R spent, Q full, idle)
    bool spent() const { return idle_ || left_ <= 0 || pending_ >= q_; }
    void take(double bytes) {
        --left_;
        ++pending_;
        if (pace_ > 0.0) bucket_ -= bytes;
    }
    /// a pending fill ended: landed (FillOk), refused, or dropped with the link
    void done() {
        if (pending_ > 0) --pending_;
    }
    int pending() const { return pending_; }
    int left() const { return left_; }
    double bucket() const { return bucket_; }

private:
    int r_, q_;
    double pace_, burst_;
    double bucket_ = 0.0, last_ms_ = -1.0;
    int left_ = 0, pending_ = 0;
    bool idle_ = false;
};

/// One layer as the remote tier sees it at a tick (pointers into the engine's Layer and rx_has row)
struct TierLayer {
    int layer = -1;
    const std::vector<int>* hits = nullptr;   // its CPU-lane hits since the last tick
    const int* slot = nullptr;                // [n_exp] VRAM slot, -1: not in VRAM
    const float* heat = nullptr;              // [n_exp]
    uint8_t* has = nullptr;                   // [n_exp] the server's list (rx_has row): 0, kHeld, kFilling
    int n_exp = 0;
    int vin = -1;                             // the VRAM tier's pick this tick (-1: none)
    double bytes = 0.0;                       // a fill's bytes in this layer (its blob)
};

/// A tick's fills: the layers with the most hits first (ties: the lower index), at most one a layer, while the book
/// lets them leave.  Each fill takes `out` off the server's list (0) and marks `in` kFilling at once: the requests of
/// the next token name neither (out is the lane's from then on; in is the server's only at fill_landed)
inline std::vector<Fill> remote_choose(std::vector<TierLayer>& ls, FillBook& book, float margin, float bonus) {
    std::vector<Fill> out;
    std::vector<size_t> order(ls.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const size_t na = ls[a].hits ? ls[a].hits->size() : 0, nb = ls[b].hits ? ls[b].hits->size() : 0;
        return na > nb;
    });
    for (size_t i : order) {
        if (book.spent()) break;
        TierLayer& t = ls[i];
        if (t.hits == nullptr || t.hits->empty() || t.has == nullptr || !book.can(t.bytes)) continue;
        Fill f = remote_pick(*t.hits, t.slot, t.heat, t.has, t.n_exp, t.vin, margin, bonus);
        if (f.in < 0) continue;
        f.layer = t.layer;
        t.has[f.out] = 0;
        t.has[f.in] = kFilling;
        book.take(t.bytes);
        out.push_back(f);
    }
    return out;
}

/// A fill's end at a token boundary: ok (FillOk) - `in` is the server's; not ok (refused) - `in` back to the lane.
/// Only a kFilling entry changes (false: it was not one - a protocol error)
inline bool fill_landed(uint8_t* has, int in, bool ok) {
    if (has[in] != kFilling) return false;
    has[in] = ok ? kHeld : 0;
    return true;
}

}  // namespace strata::ds4::rx
