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
//     `in` only an expert the main computes on its CPU lane (not in VRAM, the RAM tier or the zero-copy store), and a
//     server-held expert that stays in this GPU's VRAM (a duplicate) leaves first, whatever its heat;
//   - FillBook: the fills a token (R), pending (Q), the bulk pace (bytes a second) and the pause while the link idles;
//   - ram_in: the RAM tier's next expert, never one the server holds (the RAM tier would save the lane nothing there);
//   - PromptSettle: one cache settle a prompt that ends on windows;
//   - StageGate: the expert server's copies of a fill into its staging blob, in pieces, each between requests.
// The server's list (the main's rx_has row of a layer) has three states here: 0 not held, kHeld, kFilling.  cache_in
// (ds4_rx.hpp) and ram_in treat any non-zero as the server's, so neither cache takes an expert on its way there.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

/// One layer as the remote tier sees it at a tick (pointers into the engine's Layer and rx_has row)
struct TierLayer {
    int layer = -1;
    const std::vector<int>* hits = nullptr;   // its CPU-lane hits since the last tick
    const int* slot = nullptr;                // [n_exp] VRAM slot, -1: not in VRAM
    const int* ram = nullptr;                 // [n_exp] RAM-tier slot, -1: not there (null: no RAM tier)
    const uint8_t* zc = nullptr;              // [n_exp] != 0: the zero-copy store's, computed on this GPU (null: none)
    const float* heat = nullptr;              // [n_exp]
    uint8_t* has = nullptr;                   // [n_exp] the server's list (rx_has row): 0, kHeld, kFilling
    int n_exp = 0;
    int vin = -1;                             // the VRAM tier's pick this tick (-1: none)
    int vout = -1;                            // the VRAM expert the layer's pending swap takes out (this tick's or one
                                              // still under way; -1: none) - leaving VRAM, so no duplicate
    double bytes = 0.0;                       // a fill's bytes in this layer (its blob)
};

/// One layer's fill: `in` the hottest of the CPU-lane hits the main computes on its lane - not in VRAM (slot < 0), not
/// in the RAM tier or the zero-copy store, not the VRAM tier's pick this tick (vin) - and not on the server's list
/// (has[e] == 0); `out` a held expert (has[e] == kHeld, never kFilling): the coldest one that stays in VRAM if there
/// is one (vout is leaving it; the server's copy of such a duplicate saves the lane nothing, so any `in` beats it),
/// else the coldest held, and then only when heat[in] > heat[out] x margin + bonus.  {-1, -1}: none
inline Fill remote_pick(const TierLayer& t, float margin, float bonus) {
    Fill f;
    const int* slot = t.slot;
    const float* heat = t.heat;
    const uint8_t* has = t.has;
    int in = -1;
    for (int e : *t.hits)
        if (e >= 0 && e < t.n_exp && e != t.vin && slot[e] < 0 && (t.ram == nullptr || t.ram[e] < 0) &&
            (t.zc == nullptr || t.zc[e] == 0) && has[e] == 0 && (in < 0 || heat[e] > heat[in]))
            in = e;
    if (in < 0) return f;
    int dup = -1, cold = -1;
    for (int e = 0; e < t.n_exp; ++e) {
        if (has[e] != kHeld) continue;
        if (slot[e] >= 0 && e != t.vout && (dup < 0 || heat[e] < heat[dup])) dup = e;
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
/// and none leaves while the link idles (rx::Backoff's spell: the server is late already).  pace_bps <= 0: no pace.
/// A burst below the largest fill (max_fill bytes) would never let that fill go: raised to it (clamped(), said once)
class FillBook {
public:
    FillBook(int r, int q, double pace_bps, double burst, double max_fill)
        : r_(r), q_(q), pace_(pace_bps), burst_(burst) {
        if (pace_ > 0.0 && burst_ < max_fill) {
            std::fprintf(stderr, "ds4: remote experts: the fills' burst %.1f MB is below one fill (%.1f MB): raised to it\n",
                         burst_ / 1e6, max_fill / 1e6);
            burst_ = max_fill;
            clamped_ = true;
        }
    }
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
    double burst() const { return burst_; }
    bool clamped() const { return clamped_; }

private:
    int r_, q_;
    double pace_, burst_;
    double bucket_ = 0.0, last_ms_ = -1.0;
    int left_ = 0, pending_ = 0;
    bool idle_ = false, clamped_ = false;
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
        Fill f = remote_pick(t, margin, bonus);
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

/// The cache settle after a prompt that ends on windows (feed_windows), with the expert server on: once a prompt.
/// A prompt comes in several feeds (the server's 16-token units with --prefill-chunk 0, feed_mark's run then its last
/// token, a chunked run then a last window); only the window feed that asks for logits ends it, and a settle already
/// made in the prompt (the chunked path's own) counts.  settled(): any cache_settle; next(): the next feed is another
/// prompt's - after a decode window, a public feed (Model::feed / feed_mark) that asked for logits, reset or restore
struct PromptSettle {
    bool done = false;
    /// at a window feed's end: settle now?
    bool after_feed(bool rx_on, int logits_rows) const { return rx_on && logits_rows > 0 && !done; }
    void settled() { done = true; }
    void next() { done = false; }
};

/// (DS4_RX_FILL_GATE_SRV=1, the expert server's fill_one) a fill's copies into its staging blob - the main's [0, k)
/// from the payload, the parts' [k, nb) - in pieces of at most `piece` bytes, each begun once no request is in flight
/// (idle()).  A request's slice picks read the parts on the server's CPU lane at most of its DRAM's bandwidth
/// (fg-run, pve-5900x: 27.7 GB/s alone, 26.1 beside the gated main's fills, 23.9 beside ungated ones), and the copies
/// beside them took that bandwidth.  `left` carries a piece's bytes over a fill's copies (a down row's [k, nb) is a
/// few hundred bytes), so the wait is once a piece, not once a copy; a piece under way when a request comes is not
/// stopped.  stop(): the wait gives up (the copy still goes); relax(): a spin-wait's pause.  piece 0: one memcpy a copy
struct StageGate {
    size_t piece = 0;
    size_t left = 0;                 // (the bytes the piece under way may still copy)
    int64_t pieces = 0, waits = 0;   // (pieces begun; those that found a request in flight)
    double wait_ms = 0.0;
    explicit StageGate(size_t p) : piece(p) {}
    template <class Idle, class Stop, class Relax>
    void copy(uint8_t* d, const uint8_t* s, size_t n, Idle&& idle, Stop&& stop, Relax&& relax) {
        if (piece == 0) {
            std::memcpy(d, s, n);
            return;
        }
        while (n > 0) {
            if (left == 0) {
                ++pieces;
                if (!idle()) {
                    ++waits;
                    const auto t0 = std::chrono::steady_clock::now();
                    while (!idle() && !stop()) relax();
                    wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                }
                left = piece;
            }
            const size_t c = std::min(n, left);
            std::memcpy(d, s, c);
            d += c;
            s += c;
            n -= c;
            left -= c;
        }
    }
};

}  // namespace strata::ds4::rx
