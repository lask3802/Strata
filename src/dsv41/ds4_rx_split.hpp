// src/dsv41/ds4_rx_split.hpp - which lane computes which n_ff blocks of a routed pick under expert parallelism H'
// (ep-design.md 1.2, ep-u6-plan.md 0 and 3a): the main process's CPU lane [0, k) of a slice pick, the expert server's
// CPU lane [k, nb) of it, the server's GPU (the experts it holds) a pick whole, the main's own GPU and every pick of a
// request not sent whole on the main.  The invariant that keeps a pick from being added twice or not at all: the main
// computes only [0, k) when the slice tag went out in the same request; a pick on the main's GPU (a VRAM tier, a store
// slot the GPU reads, a drafter layer, a CPU lane that cannot slice) is never tagged, since its kernels compute it whole.
// A reply late or lost: the main adds what the server would have (fallback_range).  Pure functions, no state.
#pragma once

#include <cstddef>

namespace strata::ds4::rx {

enum class PickLane : int {
    kMain = 0,         // the main process's GPU or a CPU lane that cannot slice: whole, not sent
    kServerWhole = 1,  // the server's GPU holds it: sent whole, computed there whole
    kSplit = 2,        // sent tagged slice: the main [0, k), the server's CPU lane [k, nb)
    kMainWhole = 3,    // a CPU-lane pick of a request not sent (kept local, a backoff, no link): whole on the main
};

/// A range of n_ff blocks [b0, b1) (b0 == b1: none).
struct BlockRange {
    int b0 = 0, b1 = 0;
    bool empty() const { return b1 <= b0; }
};

/// on_main: the pick is computed by the main's GPU (l.slot[e] >= 0, a zc slot, l.mtp) or a CPU lane that has no slice
/// (not the i-quant one); server_has: the server's GPU holds the expert (rx_has); split_ok: the layer is sliced (k < nb:
/// DS4_EP_K, the server's lo <= k, the layer's experts the i-quant CPU lane's); sent: this request goes to the server.
inline PickLane pick_lane(bool on_main, bool server_has, bool split_ok, bool sent) {
    if (on_main) return PickLane::kMain;
    if (!sent) return PickLane::kMainWhole;
    if (server_has) return PickLane::kServerWhole;
    return split_ok ? PickLane::kSplit : PickLane::kMainWhole;
}
/// whether the pick goes in the request, and whether tagged slice
inline bool pick_sent(PickLane p) { return p == PickLane::kServerWhole || p == PickLane::kSplit; }
inline bool pick_slice(PickLane p) { return p == PickLane::kSplit; }
/// the blocks the main computes before the reply (nb: the layer's n_ff blocks, k the slice's start)
inline BlockRange main_range(PickLane p, int k, int nb) {
    switch (p) {
        case PickLane::kSplit: return {0, k};
        case PickLane::kServerWhole: return {0, 0};
        default: return {0, nb};
    }
}
/// the blocks the server's reply holds for the pick
inline BlockRange server_range(PickLane p, int k, int nb) {
    switch (p) {
        case PickLane::kSplit: return {k, nb};
        case PickLane::kServerWhole: return {0, nb};
        default: return {0, 0};
    }
}
/// the blocks the main adds when the reply is late or lost: exactly the server's share (the late reply is dropped)
inline BlockRange fallback_range(PickLane p, int k, int nb) { return server_range(p, k, nb); }

/// serve_lane's H' choices for one layer's request, the engine's and rx_split_test's (so the wiring is the one tested):
/// spl - the layer splits (k < nb, an EP window, a layer rx_split took); out - the request goes out (the gate and the
/// link allowing); ok - it did (rx_send).  tagged: the CPU-lane picks go with it tagged slice (decided before the send:
/// split_plan(spl, out, true, ..)); sent: a reply to wait for; here: the lane's range for its own picks before the
/// reply; late_whole / late_slice: what the lane adds for the server's whole picks and for the tagged ones when the
/// reply is late or lost.  A request that did not go: nothing tagged, the lane's picks whole (the server's whole picks
/// among them), no fallback
struct SplitPlan {
    bool tagged = false, sent = false;
    BlockRange here, late_whole, late_slice;
};
inline SplitPlan split_plan(bool spl, bool out, bool ok, int k, int nb) {
    SplitPlan s;
    s.sent = out && ok;
    s.tagged = spl && s.sent;
    s.here = main_range(pick_lane(false, false, spl, s.tagged), k, nb);
    if (s.sent) {
        s.late_whole = fallback_range(PickLane::kServerWhole, k, nb);
        if (s.tagged) s.late_slice = fallback_range(PickLane::kSplit, k, nb);
    }
    return s;
}
/// n slice picks as picks of the lane: their [k, nb) share, rounded up - what the server computes of them, the gate's
/// local cost of them (ep-u6-plan.md 3a: a slice pick counts (nb - k) / nb of a pick) and the deadline's count
inline size_t slice_share(size_t n, int k, int nb) {
    return nb > 0 && k < nb ? (n * (size_t) (nb - k) + (size_t) nb - 1) / (size_t) nb : 0;
}
/// the host-routed path's range for its CPU-lane picks (the decode's layers routed on the host, prefill): it sends no
/// request, so a split layer's picks too are whole on the main
inline BlockRange host_range(bool spl, int k, int nb) { return split_plan(spl, false, false, k, nb).here; }
/// the hybrid's balance (U5c, DS4_EP_POOL_BAL): how many of a request's nc W candidates go whole to the server.  A W
/// pick sent moves its [0, k) (wk bytes) from this lane to the server's; the reply should come in before this lane's own
/// part is done.  The estimates, EWMAs of past replies: rm - this lane's ms a byte from the send to its part done; rs -
/// the server's lane ms a byte; fix - the rest of a reply's time (send to its arrival in the kernel less rs x its bytes).
/// n: the largest that keeps fix + rs (N wt + n wk) + margin <= rm (N - n) wk, N the request's CPU-lane picks (all
/// slice picks, the candidates among them), wt a pick's [k, nb) bytes; 0..nc
inline int pool_keep(int nc, int N, double wk, double wt, double rm, double rs, double fix, double margin) {
    if (nc <= 0 || wk <= 0 || rm <= 0 || rs <= 0) return 0;
    const double num = rm * N * wk - fix - rs * N * wt - margin, den = wk * (rs + rm);
    if (num <= 0) return 0;
    const double n = num / den;
    return n >= nc ? nc : (int) n;
}
/// ... in blocks (DS4_EP_POOL_CUT, on with the balance): a whole W pick moves k blocks at once, more than a layer's slack
/// mostly holds, so the server may take a W pick's [cut, k) alone and this lane its [0, cut).  A block (wk / k bytes)
/// moved: this lane rm less a byte, the server rs more.  M: the blocks the slack (pool_keep's) takes, nc k at most;
/// n = ceil(M / k) candidates go, all cut at the same block, k - floor(M / n) (0: whole there, as pool_keep's) - or
/// n - 1 whole where they move as many blocks
struct PoolCut {
    int n = 0, cut = 0;   // (the candidates sent; their first block on the server)
};
inline PoolCut pool_cut(int nc, int N, int k, double wk, double wt, double rm, double rs, double fix, double margin) {
    PoolCut pc;
    if (nc <= 0 || k <= 0 || wk <= 0 || rm <= 0 || rs <= 0) return pc;
    const double num = rm * N * wk - fix - rs * N * wt - margin, den = wk / k * (rs + rm);
    if (num <= 0) return pc;
    const double mf = num / den;
    const int M = mf >= (double) nc * k ? nc * k : (int) mf;
    if (M <= 0) return pc;
    pc.n = (M + k - 1) / k;
    pc.cut = k - M / pc.n;
    if (pc.n > 1 && pc.n * (k - pc.cut) <= (pc.n - 1) * k) {   // (n - 1 whole move as many: fewer picks)
        --pc.n;
        pc.cut = 0;
    }
    return pc;
}
/// one EWMA step (a: the new sample's weight; v < 0: none yet - the sample)
inline double ewma(double v, double x, double a) { return v < 0 ? x : v + a * (x - v); }
/// blocks [b0, b1) as cpu_experts' n_ff values [ff0, ff0 + ffn) of n_ff (the last of nb blocks runs to n_ff)
inline void ff_range(BlockRange br, int nb, int n_ff, int& ff0, int& ffn) {
    ff0 = 256 * br.b0 < n_ff ? 256 * br.b0 : n_ff;
    const int ff1 = br.b1 >= nb ? n_ff : 256 * br.b1;
    ffn = ff1 > ff0 ? ff1 - ff0 : 0;
}

}  // namespace strata::ds4::rx
