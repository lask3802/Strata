// src/dsv41/ds4_bc_pull.cuh - DS4_BC_ONCE=3: a layer's batch bounds pulled into bc_dev by one small kernel on the
// compute stream, reading them from the mapped (pinned) bc_pin - the SMs do the copy, not the H2D copy engine, so it
// never queues behind the copy stream's expert DMAs, and host memory is read once a layer (a few hundred ints), not by
// every block of every batch's kernels (DS4_BC_ONCE=2).  Header only: the engine's flow_bc_upfront and
// tests/dsv41/bc_pull_test.cu.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "ds4_flow_batches.hpp"

namespace strata::ds4 {

// dst[j] = src[j] for j in [a0, a0 + n0) then [a1, a1 + n1): src mapped host memory (its device address), dst device
// memory; one block, its threads striding over both - a warp reads 32 adjacent ints
static __global__ void bc_pull_kernel(const int* __restrict__ src, int* __restrict__ dst, size_t a0, size_t n0,
                                      size_t a1, size_t n1) {
    for (size_t i = threadIdx.x; i < n0 + n1; i += blockDim.x) {
        const size_t j = i < n0 ? a0 + i : a1 + (i - n0);
        dst[j] = src[j];
    }
}

/// a layer's runs of regions (bc_runs: nr of them, regions of w ints) from the table src (mapped host memory, its
/// device address) to the table dst (device memory) on stream s, in one launch; nr 0: nothing launched.  The launch's
/// error, if any
inline cudaError_t bc_pull(const int* src, int* dst, size_t w, const BcRun* run, int nr, cudaStream_t s) {
    if (nr <= 0) return cudaSuccess;
    const size_t a1 = nr > 1 ? run[1].first * w : 0, n1 = nr > 1 ? run[1].n * w : 0;
    bc_pull_kernel<<<1, 256, 0, s>>>(src, dst, run[0].first * w, run[0].n * w, a1, n1);
    return cudaGetLastError();
}

}  // namespace strata::ds4
