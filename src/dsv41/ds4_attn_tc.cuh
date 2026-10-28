// Tensor-core sparse attention for the chunked prompt: Strata (dsv41), used with the DeepSeek attention of Copyright (c)
// 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17) (the
// surrounding types, AttnRows and the packed-row formats, are theirs; see third_party/project-maya/NOTICE.md).
// src/dsv41/ds4_attn_tc.cuh - DS4_ATTN_TC: the prompt slab's sparse attention (AttnRows gen 1, ckv kKvFp4, d 512) with
// FP16 mma (Turing's m16n8k8, Ampere's m16n8k16; ldmatrix), MQA (K = V) so a query's heads are the M dimension.
// A block: 32 heads of one query; warp (m-tile, d slice) holds q * scale's A fragments in registers and its 16 heads x
// slice of O (F32).  Per tile of 16 rows: the rows staged once as FP16 in shared memory (the packed rows decoded:
// exact; the window rows, BF16 on the ring's FP8 grid: exact from about 2^-21 to 65504, clamped above; the next
// tile's rows prefetched to L2), S = q K^T a warp's d slice into shared memory, one online softmax a block over the
// slices' sum (F32, base 2), P back as A fragments, O += P V.  The roundings to FP16 of q * scale and of P are what
// the modes split: hi + lo FP16 pairs give F32-like scores and weights at twice the mma.  Not
// sparse_attn_packed_kernel's arithmetic: other bits (tests/dsv41/attn_tc_test.cu measures how far).  CUDA only (PTX).
#pragma once
#include "ds4_kernels.cuh"
#include "ds4_numerics.cuh"
#include <cuda_fp16.h>

#include <atomic>
#include <cstdio>

// (sm_60 / sm_70 builds, STRATA_EXPERIMENTAL_SM60: no ldmatrix or m16n8k8 - the kernel compiles to a trap there and
// sparse_attn_tc never launches it below compute capability 7.5)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#define DS4_ATTN_TC_NONE 1
#endif

namespace strata::ds4 {
namespace attntc {

constexpr int kD = 512, kNT = 16, kLd = kD + 8;   // a staged row 520 halves apart (1040 B: ldmatrix conflict-free)

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *(uint32_t*) &h; }
__device__ __forceinline__ float e2m1_x2p14(uint32_t c) {
    return __half2float(__ushort_as_half((unsigned short) (((c & 8u) << 12) | ((c & 7u) << 9))));
}
__device__ __forceinline__ float clamp16(float v) { return fabsf(v) > 65504.0f ? copysignf(65504.0f, v) : v; }   // (a NaN stays)
__device__ __forceinline__ __half2 bf2h2(uint32_t w) {   // two BF16 -> FP16, clamped to +-65504
    return __floats2half2_rn(clamp16(__uint_as_float(w << 16)), clamp16(__uint_as_float(w & 0xffff0000u)));
}

// c += a b: m16 x k16 (row) times k16 x n8 (col), FP16 in, F32 accumulate (Turing: two m16n8k8)
__device__ __forceinline__ void mma16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(DS4_ATTN_TC_NONE)
    __trap();
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(DS4_ATTN_TC_K8)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const __half* p) {
#if defined(DS4_ATTN_TC_NONE)
    r[0] = r[1] = r[2] = r[3] = 0u;
    __trap();
#else
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const __half* p) {
#if defined(DS4_ATTN_TC_NONE)
    r[0] = r[1] = r[2] = r[3] = 0u;
    __trap();
#else
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}

// FP16 hi and lo of two F32s: hi + lo is x to about 2^-22 relative (below about 2^-3, lo subnormal: to about 6e-8)
// - the split modes' operands
__device__ __forceinline__ void split2(float x, float y, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(x, y);
    const float2 f = __half22float2(h);
    hi = h2u(h);
    lo = h2u(__floats2half2_rn(x - f.x, y - f.y));
}

// 32 heads (two m-tiles) of query blockIdx.y a block, 2 NQ warps: warp (mi, dq) holds q and O of heads hb + 16 mi ..
// + 16 over d 512 / NQ dq .. + 512 / NQ.  SP bit 1: q * scale as FP16 hi + lo, bit 2: P as well (each twice the mma;
// both: F32-like scores and weights); 0: plain FP16.  The scores in base 2 (log2(e) in q's scale and the sink's: exp2).
// AttnRows gen 1 only
constexpr int kHeads = 32, kMT = 2;
constexpr float kLog2e = 1.4426950408889634f;
template <int NQ, int SP>
__global__ void __launch_bounds__(64 * NQ, NQ == 4 ? 2 : 1)
    attn_tc_kernel(const float* __restrict__ q, const uint16_t* __restrict__ win, const uint8_t* __restrict__ ckv,
                   size_t crow, const int* __restrict__ sel, int sel_stride, AttnRows ar, const float* __restrict__ sink,
                   int heads, float scale, float* __restrict__ out) {
    constexpr int NTH = 64 * NQ, PP = NTH / kNT, VP = kD / PP;   // the staging: PP threads a row, VP values each
    constexpr int DQ = kD / NQ, KS = DQ / 16, NO = DQ / 8;       // a warp's d, its k16 steps (S), its n8 tiles (O)
    constexpr int EV = kHeads * kNT / NTH;                        // the softmax: S values a thread (2 or 1)
    static_assert(VP == 32 || VP == 16, "a thread stages 16 or 32 values");
    static_assert(EV == 2 || EV == 1, "a thread's softmax: 2 or 1 values");
    __shared__ __align__(16) __half kv[kNT * kLd];
    // S's partial sums: d slice, m-tile, lane, the lane's two n8 tiles' accumulators; then [0] P, the same layout
    __shared__ __align__(16) float part[NQ][kMT][32][8];
    __shared__ float cs_s[kHeads];   // each head's rescale of the tile; at the end 1 / its denominator
    const int hb = blockIdx.x * kHeads, t = blockIdx.y;
    const long long ta = ar.t0 + t, pa = ar.p0 + ta;
    const long long first = max((long long) ar.win - ar.p0, ta + 1);   // (as sparse_attn_packed_kernel)
    const int w0 = (int) first, n_win = (int) (ar.win + ta - first + 1);
    const int n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    const int total = n_win + n_sel;
    const uint16_t* wbase = win + (size_t) w0 * kD;
    const int* sl = sel + (size_t) t * sel_stride;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, mi = wp % kMT, dq = wp / kMT;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    // q * scale * log2(e) of heads mi * 16 + g, + 8 over the warp's d: KS k16 steps' A fragments (SP & 1: and the lo)
    uint32_t qa[KS][4], ql[(SP & 1) ? KS : 1][4];
    {
        const float qs = scale * kLog2e;
        const float* q0 = q + ((size_t) t * heads + hb + mi * 16 + g) * kD + dq * DQ + c2;
        const float* q1 = q0 + 8 * kD;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            const float2 x[4] = {*(const float2*) (q0 + kk * 16), *(const float2*) (q1 + kk * 16),
                                 *(const float2*) (q0 + kk * 16 + 8), *(const float2*) (q1 + kk * 16 + 8)};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if constexpr ((SP & 1) != 0) split2(x[r].x * qs, x[r].y * qs, qa[kk][r], ql[kk][r]);
                else qa[kk][r] = h2u(__floats2half2_rn(x[r].x * qs, x[r].y * qs));
            }
        }
    }
    float o[NO][4];   // heads g / g + 8 x d dq * DQ + 8 n .. + 8: an m16n8 accumulator each
#pragma unroll
    for (int n = 0; n < NO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the softmax's values: m-tile smi, lane slane's accumulators sv .. sv + EV (row g + 8 (sv % 4 / 2) of its lane's
    // g, column (sv / 4) 8 + its c2 + sv % 2) - the 16 columns of a row over 8 / EV threads, a running max and sum each
    const int smi = threadIdx.x / (32 * 8 / EV), slane = (threadIdx.x / (8 / EV)) % 32, sv = (threadIdx.x % (8 / EV)) * EV;
    const int shead = smi * 16 + (slane >> 2) + 8 * ((sv >> 1) & 1);
    const bool slead = (slane & 3) == 0 && sv < 4 && (sv & 1) == 0;   // (the row's first thread)
    float m_r = -1e30f, l_r = 0.0f;
    // the staging: a thread's row of the tile and its values sp * VP ..; the selection a tile ahead (its rows
    // prefetched to L2 while this tile's arithmetic runs)
    const int sr = threadIdx.x / PP, sp = threadIdx.x % PP;
    auto sel_at = [&](int j) { return j >= n_win && j < total ? __ldg(sl + (j - n_win)) : 0; };
    int ix = sel_at(sr), ix_n = sel_at(kNT + sr);
    // ldmatrix rows: K's x4 (non-trans) and V's (trans) for this lane
    const __half* kp = kv + ((lane >> 4) * 8 + (lane & 7)) * kLd + dq * DQ + ((lane >> 3) & 1) * 8;
    const __half* vp = kv + (((lane >> 3) & 1) * 8 + (lane & 7)) * kLd + dq * DQ + (lane >> 4) * 8;
    for (int t0 = 0; t0 < total; t0 += kNT) {
        const int nt = min(kNT, total - t0);
        {   // rows t0 .. t0 + nt as FP16 (past nt: zeros)
            const int j = t0 + sr;
            uint4* dst = (uint4*) (kv + sr * kLd + sp * VP);
            uint32_t h[VP / 2];
            if (sr >= nt) {
#pragma unroll
                for (int i = 0; i < VP / 2; ++i) h[i] = 0u;
            } else if (j < n_win) {
                const uint4* src = (const uint4*) (wbase + (size_t) j * kD + sp * VP);
#pragma unroll
                for (int i = 0; i < VP / 8; ++i) {
                    const uint4 w = src[i];
                    h[4 * i] = h2u(bf2h2(w.x));
                    h[4 * i + 1] = h2u(bf2h2(w.y));
                    h[4 * i + 2] = h2u(bf2h2(w.z));
                    h[4 * i + 3] = h2u(bf2h2(w.w));
                }
            } else {   // codes sp * VP / 2 .. (value 2i the low nibble), an E4M3 scale per 16 values
                const uint8_t* rr = ckv + (size_t) ix * crow;
                uint32_t cw[VP / 8];
#pragma unroll
                for (int i = 0; i < VP / 8; ++i) cw[i] = ((const uint32_t*) (rr + sp * (VP / 2)))[i];
                float ks[VP / 16];
#pragma unroll
                for (int i = 0; i < VP / 16; ++i) ks[i] = e4m3_value(rr[kD / 2 + sp * (VP / 16) + i]) * 16384.0f;
#pragma unroll
                for (int i = 0; i < VP / 2; ++i) {   // values 2i, 2i + 1
                    const uint32_t b = cw[i >> 2] >> (8 * (i & 3));
                    h[i] = h2u(__floats2half2_rn(e2m1_x2p14(b) * ks[i / 8], e2m1_x2p14(b >> 4) * ks[i / 8]));   // (exact)
                }
            }
#pragma unroll
            for (int i = 0; i < VP / 8; ++i) dst[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
        }
        {   // the next tile's rows towards L2, the one after's selection
            const int jn = t0 + kNT + sr;
            if (jn < total) {
                const void* p = jn < n_win ? (const void*) (wbase + (size_t) jn * kD + sp * VP)
                                           : (const void*) (ckv + (size_t) ix_n * crow + sp * (VP / 2));
                asm volatile("prefetch.global.L2 [%0];\n" ::"l"(__cvta_generic_to_global(p)));
                if (jn >= n_win && sp == 0)   // (the row's scales)
                    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(__cvta_generic_to_global(ckv + (size_t) ix_n * crow + kD / 2)));
            }
            ix = ix_n;
            ix_n = sel_at(t0 + 2 * kNT + sr);
        }
        __syncthreads();
        {   // S's slice dq: 16 heads x 16 rows
            float s[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                uint32_t b[4];
                ldsm4(b, kp + kk * 16);
                mma16(s[0], qa[kk], b[0], b[1]);
                mma16(s[1], qa[kk], b[2], b[3]);
                if constexpr ((SP & 1) != 0) {
                    mma16(s[0], ql[kk], b[0], b[1]);
                    mma16(s[1], ql[kk], b[2], b[3]);
                }
            }
            float4* pp = (float4*) part[dq][mi][lane];
            pp[0] = make_float4(s[0][0], s[0][1], s[0][2], s[0][3]);
            pp[1] = make_float4(s[1][0], s[1][1], s[1][2], s[1][3]);
        }
        __syncthreads();
        {   // the online softmax, once a block: S = the slices' sum (in slice order), columns past nt out; P into part[0]
            float sf[EV];
#pragma unroll
            for (int e = 0; e < EV; ++e) {
                float v = part[0][smi][slane][sv + e];
#pragma unroll
                for (int k = 1; k < NQ; ++k) v += part[k][smi][slane][sv + e];
                sf[e] = ((sv + e) >> 2) * 8 + (slane & 3) * 2 + ((sv + e) & 1) < nt ? v : -INFINITY;
            }
            float mx = EV == 2 ? fmaxf(sf[0], sf[EV - 1]) : sf[0];
            if (EV == 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, EV == 2 ? 2 : 4));   // (the other n8 tile)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8 / EV));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16 / EV));
            const float mn = fmaxf(m_r, mx), cs = exp2f(m_r - mn);
            m_r = mn;
            float ls = 0.0f;
#pragma unroll
            for (int e = 0; e < EV; ++e) {
                sf[e] = exp2f(sf[e] - mn);
                ls += sf[e];
                part[0][smi][slane][sv + e] = sf[e];
            }
            l_r = l_r * cs + ls;
            if (slead) cs_s[shead] = cs;
        }
        __syncthreads();
        uint32_t pa_[4], pl_[(SP & 2) ? 4 : 1];
        {   // the warp's rows' rescale; P as A fragments (rows g, g + 8 of k 0 .. 8, then of k 8 .. 16)
            const float cs0 = cs_s[mi * 16 + g], cs1 = cs_s[mi * 16 + g + 8];
#pragma unroll
            for (int n = 0; n < NO; ++n) {
                o[n][0] *= cs0;
                o[n][1] *= cs0;
                o[n][2] *= cs1;
                o[n][3] *= cs1;
            }
            const float4 p0 = *(const float4*) &part[0][mi][lane][0], p1 = *(const float4*) &part[0][mi][lane][4];
            const float pv[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float x = pv[(r >> 1) * 4 + (r & 1) * 2], y = pv[(r >> 1) * 4 + (r & 1) * 2 + 1];
                if constexpr ((SP & 2) != 0) split2(x, y, pa_[r], pl_[r]);
                else pa_[r] = h2u(__floats2half2_rn(x, y));
            }
        }
#pragma unroll
        for (int n2 = 0; n2 < NO / 2; ++n2) {   // O += P V over d dq * DQ + 16 n2 .. + 16
            uint32_t b[4];
            ldsm4t(b, vp + n2 * 16);
            mma16(o[2 * n2], pa_, b[0], b[1]);
            mma16(o[2 * n2 + 1], pa_, b[2], b[3]);
            if constexpr ((SP & 2) != 0) {
                mma16(o[2 * n2], pl_, b[0], b[1]);
                mma16(o[2 * n2 + 1], pl_, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    {   // each head's sum over its row's threads, the sink, 1 / the denominator
        float l = l_r;
        if (EV == 1) l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, EV == 2 ? 2 : 4);
        l += __shfl_xor_sync(0xffffffffu, l, 8 / EV);
        l += __shfl_xor_sync(0xffffffffu, l, 16 / EV);
        if (slead) cs_s[shead] = 1.0f / (l + exp2f(sink[hb + shead] * kLog2e - m_r));
    }
    __syncthreads();
    const int h0 = hb + mi * 16 + g;
    const float r0 = cs_s[mi * 16 + g], r1 = cs_s[mi * 16 + g + 8];
    float* ov0 = out + ((size_t) t * heads + h0) * kD + dq * DQ + c2;
    float* ov1 = ov0 + 8 * kD;
#pragma unroll
    for (int n = 0; n < NO; ++n) {
        *(float2*) (ov0 + n * 8) = make_float2(o[n][0] * r0, o[n][1] * r0);
        *(float2*) (ov1 + n * 8) = make_float2(o[n][2] * r1, o[n][3] * r1);
    }
}

// the current device's state for the kernels, cached a device (4: cached): bit 0 its compute capability 7.5 or more;
// the first call sets the 4-slice kernels' shared-memory carveout on it
inline int device_state() {
    static std::atomic<int> cached[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    int v = cached[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int ma = 0, mi = 0;
        cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
        v = 4 | (ma * 10 + mi >= 75 ? 1 : 0);
        if (v & 1) {   // (two blocks a multiprocessor: Turing's 64 KB of shared memory)
            bool ok = cudaFuncSetAttribute(attn_tc_kernel<4, 0>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                           cudaSharedmemCarveoutMaxShared) == cudaSuccess;
#if defined(DS4_ATTN_TC_TEST)
            ok = cudaFuncSetAttribute(attn_tc_kernel<4, 3>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                      cudaSharedmemCarveoutMaxShared) == cudaSuccess && ok;
#endif
            if (!ok) cudaGetLastError();   // (a hint only: not this call's error)
        }
        cached[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

// mode 1: q and P as FP16 hi + lo, 2: q hi + lo and P plain (both 8 d slices, 16 warps a block), 3: plain FP16 (4
// slices, 8 warps; two blocks a multiprocessor); 4 (a test's, DS4_ATTN_TC_TEST builds): mode 1's arithmetic in mode
// 3's 4-slice layout.  False: not taken - heads not a multiple of 32, a GPU below compute capability 7.5 (each said
// once), another mode
inline bool sparse_attn_tc(int mode, const float* q, const uint16_t* win, const uint8_t* ckv, size_t crow,
                           const int* sel, int sel_stride, const AttnRows& a, const float* sink, int heads, int T,
                           float scale, float* out, cudaStream_t s) {
    if (mode < 1 || mode > 4 || T <= 0) return false;
    static std::atomic<bool> told_heads{false}, told_cc{false};
    if (heads % kHeads != 0) {
        if (!told_heads.exchange(true))
            std::fprintf(stderr, "ds4: DS4_ATTN_TC: %d heads a device, not a multiple of %d - the packed kernel\n", heads,
                         kHeads);
        return false;
    }
    if ((device_state() & 1) == 0) {
        if (!told_cc.exchange(true))
            std::fprintf(stderr, "ds4: DS4_ATTN_TC: a GPU below compute capability 7.5 - the packed kernel\n");
        return false;
    }
    const dim3 grid(heads / kHeads, T);
    if (mode == 1)
        attn_tc_kernel<8, 3><<<grid, 512, 0, s>>>(q, win, ckv, crow, sel, sel_stride, a, sink, heads, scale, out);
    else if (mode == 2)
        attn_tc_kernel<8, 1><<<grid, 512, 0, s>>>(q, win, ckv, crow, sel, sel_stride, a, sink, heads, scale, out);
    else if (mode == 3)
        attn_tc_kernel<4, 0><<<grid, 256, 0, s>>>(q, win, ckv, crow, sel, sel_stride, a, sink, heads, scale, out);
    else {
#if defined(DS4_ATTN_TC_TEST)
        attn_tc_kernel<4, 3><<<grid, 256, 0, s>>>(q, win, ckv, crow, sel, sel_stride, a, sink, heads, scale, out);
#else
        return false;
#endif
    }
    return true;
}

}  // namespace attntc
}  // namespace strata::ds4
