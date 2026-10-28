// src/dsv41/ds4_flow_batches.hpp - the prompt flow's expert stage in batches (flow_layer): a (chunk, layer)'s streamed
// items [lb, le) are computed in batches of at most bmax items that never cross the VRAM ring's end (R slots), their
// bounds fixed by the stream alone - so a layer's whole partition, and every batch's compacted bounds, are known once
// the layer routed (flow_rest).  Each batch's bounds go into region n_batch % bc_n of the pinned bc_pin (and of
// bc_dev): bc_n regions of bc_w ints.  DS4_BC_ONCE (1: the layer's regions written up front and sent in one copy, two
// where they wrap; 2: read by the kernels from mapped host memory, no copy; 3: pulled from mapped host memory into
// bc_dev by one small kernel a layer) uses the same partition, the same regions and the same values as the per-batch
// copies (unset / 0), so the sums are the same bits.  Pure functions, no state
// (tests/dsv41/flow_batches_test.cpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::ds4 {

/// DS4_BC_ONCE's text: 0 off (unset, empty or "0"), 1 one copy a layer, 2 mapped host memory, 3 pulled from mapped
/// host memory into bc_dev by one small kernel a layer (ds4_bc_pull.cuh); -1 anything else
inline int bc_once_mode(const char* env) {
    if (env == nullptr || *env == '\0' || std::strcmp(env, "0") == 0) return 0;
    if (std::strcmp(env, "1") == 0) return 1;
    if (std::strcmp(env, "2") == 0) return 2;
    if (std::strcmp(env, "3") == 0) return 3;
    return -1;
}

/// the end of the batch that starts at item i0 of a layer whose items end at le: bmax items at most, never across the
/// ring's end (R slots, item i in slot i % R)
inline size_t flow_batch_end(size_t i0, size_t le, size_t R, size_t bmax) {
    return std::min({le, i0 + bmax, (i0 / R + 1) * R});
}

struct FlowBatch {
    size_t i0 = 0, i1 = 0;
};

/// the layer's batches over [lb, le), in the order flow_layer computes them (none when lb >= le)
inline void flow_batches(size_t lb, size_t le, size_t R, size_t bmax, std::vector<FlowBatch>& out) {
    out.clear();
    for (size_t i0 = lb; i0 < le;) {
        const size_t i1 = flow_batch_end(i0, le, R, bmax);
        out.push_back({i0, i1});
        i0 = i1;
    }
}

/// a batch's compacted bounds into bc[0 .. n]: the items [i0, i1) (virtual slots v(i), ascending, with no rows in the
/// slots between them) - bc[m] the first sorted row of item i0 + m, bc[n] the row after the last item's; b: the
/// layer's bounds by virtual slot.  Rows bc[n] - bc[0] (0: a batch of unpicked likely ones, nothing to compute)
template <class V>
inline void flow_batch_bounds(const int* b, size_t i0, size_t i1, V v, int* bc) {
    const int n = (int) (i1 - i0);
    for (int m = 0; m < n; ++m) bc[m] = b[(size_t) v(i0 + (size_t) m)];
    bc[n] = b[(size_t) v(i1 - 1) + 1];
}

/// the region of batch n_batch (bc_n regions, used in turn)
inline size_t bc_region(uint64_t n_batch, int bc_n) { return (size_t) (n_batch % (uint64_t) bc_n); }

/// nb batches from n0 on take regions bc_region(n0) .. bc_region(n0 + nb - 1): as runs of adjacent regions, out[0]
/// then out[1] where they wrap past region bc_n - 1; how many runs (0: nb 0).  nb must be at most bc_n
/// (bc_layer_fits): more and a run would cover a region twice - a batch's bounds written over another's of the layer
struct BcRun {
    size_t first = 0, n = 0;
};
inline int bc_runs(uint64_t n0, size_t nb, int bc_n, BcRun out[2]) {
    if (nb == 0) return 0;
    const size_t r0 = bc_region(n0, bc_n), to_end = (size_t) bc_n - r0;
    out[0] = {r0, std::min(nb, to_end)};
    if (nb <= to_end) return 1;
    out[1] = {0, nb - to_end};
    return 2;
}

/// DS4_BC_ONCE's invariant: a layer's batches written up front each take a region of their own - at most bc_n of them
/// (a layer streams at most its n_exp experts, one item each, and bc_n = n_exp_max + 2).  The regions the layers
/// before used are read by nothing once those layers' work on the compute stream is done - which the layer's bounds
/// wait (layer_chunk's cudaStreamSynchronize before flow_layer) already made sure of; flow_bc_upfront waits for
/// bc_done too, so the rule does not hang on that wait alone - and the host never writes a region a copy or a kernel
/// still reads
inline bool bc_layer_fits(size_t nb, int bc_n) { return bc_n > 0 && nb <= (size_t) bc_n; }

}  // namespace strata::ds4
