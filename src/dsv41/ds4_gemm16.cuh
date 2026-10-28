// src/dsv41/ds4_gemm16.cuh - the chunked prompt's dense GEMM in its pieces (ds4_model.cu's gemm()): W reconstructed to
// FP16 in whole 128-column blocks that fit the w16 scratch, then cuBLAS over each.  DS4_DQ_ONCE reconstructs the
// attention slabs' weights once a (chunk, layer) in the same pieces (dequant_whole, at places()) and runs every slab's
// GEMMs over them with the same calls - the same FP16 operands, shapes and parameters, so the same bits.  Header-only:
// the test target (dq_test.cu) runs both orders through these on random weights without the model.
#pragma once

#include "ds4_gguf.cuh"
#include "strata/dsv41/exl3.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::ds4::gemm16 {

/// A piece's columns: the most whole 128-column blocks whose k FP16 values a column fit w16_elems (at least one block).
inline int cols(const strata::dsv41k::exl3::Mat& m, size_t w16_elems) {
    return std::max(128, (int) std::min<size_t>((size_t) m.n, w16_elems / (size_t) m.k) / 128 * 128);
}
/// W's columns [c0, c0 + nc) as FP16 rows: dst[c][r] (nc rows of k values) - a GGUF's (m.fmt != 0) or an EXL3 one's.
inline void dequant(const strata::dsv41k::exl3::Mat& m, int c0, int nc, uint16_t* dst, cudaStream_t s) {
    namespace x3 = strata::dsv41k::exl3;
    namespace gg = strata::ds4::gguf;
    if (m.fmt != 0) gg::rows(nc == m.n ? m : gg::view(m, c0, nc), dst, false, s);
    else x3::reconstruct_rows(nc == m.n ? m : x3::view(m, c0, nc), dst, false, s);
}
/// y[r][c] = x16[r] . w16[c] (+ beta y) for r < rows, c < nc: a piece's GEMM (w16: nc rows of k FP16 values).
inline cublasStatus_t mm(cublasHandle_t blas, const uint16_t* w16, int k, int nc, const uint16_t* x16, int ldx, int rows,
                         float* y, int ldy, float beta) {
    const float one = 1.0f;
    return cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, k, &one, w16, CUDA_R_16F, k, x16, CUDA_R_16F, ldx, &beta,
                        y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

/// A matrix reconstructed whole (dequant_whole): its piece at column c0, c0 k values in.
inline const uint16_t* piece(const uint16_t* whole, int c0, int k) { return whole + (size_t) c0 * (size_t) k; }
/// W reconstructed whole into dst in gemm()'s pieces (piece c0 at piece(dst, c0, k)); before(c0, nc) / after() around
/// each piece's reconstruction (DS4_PROF's marks).
template <class Before, class After>
inline void dequant_whole(const strata::dsv41k::exl3::Mat& m, size_t w16_elems, uint16_t* dst, cudaStream_t s,
                          Before&& before, After&& after) {
    const int cs = cols(m, w16_elems);
    for (int c0 = 0; c0 < m.n; c0 += cs) {
        const int nc = std::min(cs, m.n - c0);
        before(c0, nc);
        dequant(m, c0, nc, dst + (size_t) c0 * (size_t) m.k, s);
        after();
    }
}
/// FP16 values of m reconstructed whole, its room in a DS4_DQ_ONCE buffer (a multiple of 128: every piece 256-byte
/// aligned, as the cudaMalloc'd w16 scratch is)
inline size_t whole_elems(const strata::dsv41k::exl3::Mat& m) { return ((size_t) m.n * (size_t) m.k + 127) / 128 * 128; }
/// The attention slab's matrices in a DS4_DQ_ONCE buffer from base: wq_b, wo_a's og groups, wo_b one after another;
/// elems their values in all (base null: the size alone)
struct Places {
    uint16_t* qb = nullptr;
    uint16_t* oa[16] = {};
    uint16_t* ob = nullptr;
    size_t elems = 0;
};
inline Places places(uint16_t* base, const strata::dsv41k::exl3::Mat& wq_b, const strata::dsv41k::exl3::Mat* wo_a, int og,
                     const strata::dsv41k::exl3::Mat& wo_b) {
    Places p;
    auto at = [&](const strata::dsv41k::exl3::Mat& m) {
        uint16_t* q = base != nullptr ? base + p.elems : nullptr;
        p.elems += whole_elems(m);
        return q;
    };
    p.qb = at(wq_b);
    for (int gi = 0; gi < og && gi < 16; ++gi) p.oa[gi] = at(wo_a[gi]);
    p.ob = at(wo_b);
    return p;
}

}  // namespace strata::ds4::gemm16
