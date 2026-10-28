// tests/dsv41/dq_test.cu - the chunked prompt's dense dequant (ds4_iq.cu's dequant_rows, gg::rows' K-quant / i-quant
// path) with each kernel DS4_DQ_NEW selects against the row-per-lane one it is unset, byte for byte: random blocks of
// every type dequant_rows takes (any code bits, F16 scales random and at the edges: zeros, subnormals, the largest),
// several row counts, first rows (gemm()'s column pieces: a view's rows c0 ..) and widths, F16 and BF16, and the bytes
// around the output untouched.  Then DS4_DQ_ONCE's order (ds4_gemm16.cuh): the attention slab's matrices at their
// shapes, reconstructed once and every slab's GEMMs over them, against gemm()'s (each slab reconstructing each piece into
// the w16 scratch, then its GEMM), the outputs byte for byte - with each dequant kernel.  Needs a GPU, no model.
//   dq_test [--bench: the attention slab's pieces timed, each kernel]
#include "../../src/dsv41/ds4_gemm16.cuh"
#include "../../src/dsv41/ds4_gguf.cuh"
#include "../../src/dsv41/ds4_iq.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
namespace gg = strata::ds4::gguf;
namespace x3 = strata::dsv41k::exl3;
namespace g16 = strata::ds4::gemm16;

namespace {

int g_fail = 0, g_cases = 0;
std::mt19937 rng(1028);

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

uint16_t f16(float f) { return __half_as_ushort(__float2half_rn(f)); }

// an F16 scale: a typical one either sign, or (one in four) an edge value - +-0, the smallest subnormal, the largest
// subnormal, the smallest normal, 1, the largest finite (no Inf / NaN: not a valid scale)
uint16_t scale16(bool edges = true) {
    static const uint16_t edge[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0xfbff, 0x1400};
    if (edges && rng() % 4 == 0) return edge[rng() % (sizeof(edge) / sizeof(edge[0]))];
    const float m = 0.0005f + 0.05f * (float) (rng() % 10000) / 10000.0f;
    return f16(rng() % 8 == 0 ? -m : m);
}
void put16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }

// where each type's F16 scales lie in its block (ds4_iq.cu's decode_b): d, and dmin for the K-quants with one
struct TypeInfo {
    int type, d, dmin;   // byte offsets (-1: none); IQ1_M: its d in the top nibbles of its four u16 scales
};
const TypeInfo kTypes[] = {
    {iq::kIQ2_XXS, 0, -1}, {iq::kIQ2_XS, 0, -1}, {iq::kIQ2_S, 0, -1}, {iq::kIQ3_XXS, 0, -1}, {iq::kIQ3_S, 0, -1},
    {iq::kIQ4_XS, 0, -1},  {iq::kIQ1_S, 0, -1},  {iq::kIQ1_M, -1, -1}, {iq::kQ4_K, 0, 2},    {iq::kQ5_K, 0, 2},
    {iq::kQ6_K, 208, -1},  {iq::kQ2_K, 80, 82},  {iq::kQ3_K, 108, -1},
};

// rows x k of random blocks of the type (edges false: typical scales only)
std::vector<uint8_t> blocks(const TypeInfo& t, int rows, int k, bool edges = true) {
    const int bb = iq::block_bytes(t.type);
    std::vector<uint8_t> w(iq::row_bytes(t.type, k) * (size_t) rows);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    for (size_t o = 0; o + bb <= w.size(); o += (size_t) bb) {
        uint8_t* p = &w[o];
        if (t.type == iq::kIQ1_M) {
            const uint16_t d = scale16(edges);
            p[49] = (uint8_t) ((p[49] & 0x0f) | ((d & 0x000f) << 4));
            p[51] = (uint8_t) ((p[51] & 0x0f) | (d & 0x00f0));
            p[53] = (uint8_t) ((p[53] & 0x0f) | ((d & 0x0f00) >> 4));
            p[55] = (uint8_t) ((p[55] & 0x0f) | ((d & 0xf000) >> 8));
            continue;
        }
        put16(p + t.d, scale16(edges));
        if (t.dmin >= 0) put16(p + t.dmin, scale16(edges));
    }
    return w;
}

constexpr size_t kGuard = 4096;   // bytes of 0xA5 before and after each output
constexpr int kKernels = 3;       // dq_kernel(): 0 the row-per-lane one (DS4_DQ_NEW unset), 1, 2

// rows [c0, c0 + n) of a (c0 + n + 2) x k matrix of the type, at dst + dst_off (bytes past a 256-aligned base): every
// kernel's output and guard bytes against kernel 0's
void check(const TypeInfo& t, int k, int n, int c0, size_t dst_off, bool bf16, cudaStream_t s) {
    const int rows = c0 + n + 2;
    const std::vector<uint8_t> h = blocks(t, rows, k);
    const size_t ld = iq::row_bytes(t.type, k);
    uint8_t* raw = nullptr;
    CU(cudaMalloc(&raw, h.size()));
    // (on s and waited for: a pageable cudaMemcpy may return before its DMA lands, and s does not wait for the legacy stream)
    CU(cudaMemcpyAsync(raw, h.data(), h.size(), cudaMemcpyHostToDevice, s));
    CU(cudaStreamSynchronize(s));
    const size_t out = (size_t) n * k * 2, span = kGuard + dst_off + out + kGuard;
    uint8_t* buf = nullptr;
    CU(cudaMalloc(&buf, span));
    std::vector<uint8_t> r[kKernels];
    const int keep = iq::dq_kernel();
    bool refused = false;
    for (int kern = 0; kern < kKernels; ++kern) {
        iq::dq_kernel() = kern;
        CU(cudaMemsetAsync(buf, 0xA5, span, s));
        uint16_t* dst = (uint16_t*) (buf + kGuard + dst_off);
        refused = refused || !iq::dequant_rows(t.type, raw + (size_t) c0 * ld, ld, k, n, dst, bf16, s);
        CU(cudaStreamSynchronize(s));
        CU(cudaGetLastError());
        r[kern].resize(span);
        CU(cudaMemcpy(r[kern].data(), buf, span, cudaMemcpyDeviceToHost));
    }
    iq::dq_kernel() = keep;
    // kernel 0 must have written the output (not the fill everywhere) and left the guards
    size_t filled = 0, guard_bad = 0;
    for (size_t i = kGuard + dst_off; i < kGuard + dst_off + out; ++i) filled += r[0][i] == 0xA5;
    for (size_t i = 0; i < span; ++i)
        if ((i < kGuard + dst_off || i >= kGuard + dst_off + out) && r[0][i] != 0xA5) ++guard_bad;
    char what[160];
    std::snprintf(what, sizeof(what), "%-7s k %5d n %5d c0 %3d dst+%-2zu %s", iq::name(t.type), k, n, c0, dst_off,
                  bf16 ? "bf16" : "f16 ");
    for (int kern = 1; kern < kKernels; ++kern) {
        size_t diff = 0, first = 0;
        for (size_t i = 0; i < span; ++i)
            if (r[kern][i] != r[0][i] && diff++ == 0) first = i;
        if (diff) {   // (the first few values that differ)
            int shown = 0;
            for (size_t i = kGuard + dst_off; i + 1 < kGuard + dst_off + out && shown < 4; i += 2) {
                uint16_t a, b;
                std::memcpy(&a, &r[0][i], 2);
                std::memcpy(&b, &r[kern][i], 2);
                if (a == b) continue;
                const size_t v = (i - kGuard - dst_off) / 2;
                std::printf("  val %zu row %zu col %zu (sb %zu j %zu): k0 %04x k%d %04x\n", v, v / k, v % k, (v % k) / 32,
                            v % 32, a, kern, b);
                ++shown;
            }
        }
        const bool ok = !refused && diff == 0 && filled < out && guard_bad == 0;
        ++g_cases;
        if (!ok) ++g_fail;
        if (!ok || std::getenv("DQ_TEST_VERBOSE") != nullptr)
            std::printf("%-4s %s kernel %d: %zu bytes, %zu differ%s%s%s\n", ok ? "ok" : "FAIL", what, kern, out, diff,
                        diff ? (" (first at " + std::to_string((long long) first - (long long) (kGuard + dst_off)) + ")").c_str() : "",
                        refused ? ", refused" : "", guard_bad ? ", kernel 0 wrote past its output" : "");
    }
    CU(cudaFree(buf));
    CU(cudaFree(raw));
}

void test_types(cudaStream_t s) {
    struct Shape {
        int k, n, c0;
        size_t dst_off;
    };
    // (k 256 / 1280: 8 / 40 sub-blocks a row - not whole warps of a row, kernel 2 unstaged; 2048 .. 8192: staged for Q4_K,
    // Q5_K, Q2_K; c0 > 0: a view's rows, as gemm()'s pieces; dst + 2: not 16-byte aligned, kernel 2 hands it to kernel 1)
    const Shape shapes[] = {{256, 1, 0, 0},   {256, 37, 5, 0},    {1280, 300, 128, 0}, {1280, 77, 1, 0},
                            {4096, 129, 3, 0}, {4096, 64, 0, 2},   {8192, 33, 0, 0},    {8192, 260, 7, 16},
                            {2048, 1000, 2, 0}};
    for (const TypeInfo& t : kTypes) {
        const int g0 = g_fail, c0 = g_cases;
        for (const Shape& sh : shapes)
            for (int bf = 0; bf < 2; ++bf) check(t, sh.k, sh.n, sh.c0, sh.dst_off, bf != 0, s);
        std::printf("%-4s %-7s %d cases (%zu shapes x F16 / BF16 x kernels 1, 2 against 0), %d differ\n",
                    g_fail == g0 ? "ok" : "FAIL", iq::name(t.type), g_cases - c0, sizeof(shapes) / sizeof(shapes[0]),
                    g_fail - g0);
    }
    // the attention slab's Q5_K pieces as gemm() cuts them (wq_b's 6528 of 32768 columns, wo_a's group whole, wo_b's 1024
    // of 5120 at its second piece)
    const TypeInfo& q5 = kTypes[9];
    const int g0 = g_fail;
    check(q5, 1280, 6528, 0, 0, false, s);
    check(q5, 4096, 1024, 0, 0, false, s);
    check(q5, 8192, 1024, 1024, 0, false, s);
    std::printf("%-4s Q5_K    the slab's pieces (wq_b 1280 x 6528, wo_a 4096 x 1024, wo_b 8192 x 1024)\n",
                g_fail == g0 ? "ok" : "FAIL");
}

// ---- DS4_DQ_ONCE: the slab loop's GEMMs over the matrix reconstructed once against gemm()'s per-slab order
// a k x n weight of the engine's formats (as mat_gg / mat_bf16 put them on the device): a GGUF's Q5_K, Q8_0 or BF16
struct W {
    x3::Mat m;
    uint8_t* raw = nullptr;
    W(int fmt, int k, int n) {
        std::vector<uint8_t> h;
        if (fmt == gg::kFmtGG) {
            h = blocks(kTypes[9], n, k, false);
            m.K = iq::kQ5_K;
            m.ld = (int) gg::type_row_bytes(iq::kQ5_K, k);
        } else if (fmt == gg::kFmtQ8_0) {
            h.resize((size_t) n * (k / 32) * gg::kQ8Bytes);
            for (auto& v : h) v = (uint8_t) (rng() & 0xff);
            for (size_t o = 0; o < h.size(); o += gg::kQ8Bytes) put16(&h[o], scale16(false));
            m.ld = (int) gg::q8_row_bytes(k);
        } else {
            h.resize((size_t) n * k * 2);
            for (size_t o = 0; o < h.size(); o += 2) {
                const float f = 0.05f * ((float) (rng() % 20001) / 10000.0f - 1.0f);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                const uint16_t b = (uint16_t) (u >> 16);
                std::memcpy(&h[o], &b, 2);
            }
            m.ld = k;
        }
        CU(cudaMalloc(&raw, h.size()));
        CU(cudaMemcpy(raw, h.data(), h.size(), cudaMemcpyHostToDevice));
        CU(cudaDeviceSynchronize());
        m.fmt = fmt;
        m.raw = raw;
        m.k = k;
        m.n = n;
    }
    ~W() { cudaFree(raw); }
};

// a device buffer of n random FP16 values in [-1, 1]
uint16_t* rand16(size_t n) {
    std::vector<uint16_t> h(n);
    for (auto& v : h) v = f16((float) (rng() % 20001) / 10000.0f - 1.0f);
    uint16_t* p = nullptr;
    CU(cudaMalloc(&p, n * 2));
    CU(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice));
    return p;
}

// layer_chunk's attention slab projections for T rows in slabs of A (wq_b, the og wo_a groups, wo_b at the engine's
// shapes, x and y strides), in its two orders: gemm()'s (each slab's pieces reconstructed into the w16 scratch, then
// their GEMMs) and DS4_DQ_ONCE's (the three in one buffer at gemm16::places, each reconstructed whole by dequant_whole
// once, every slab's pieces GEMMs over gemm16::piece).  The inputs are random (not chained through the attention) and
// each slab's outputs land at its own rows; every dequant kernel in the once order against kernel 0 in gemm()'s
void check_once(const char* what, int fmt, int T, int A, cublasHandle_t blas, cudaStream_t s) {
    const size_t w16_elems = (size_t) 8 << 20;   // (alloc_chunks' w16 scratch)
    const int QL = 1280, HQ = 32768, OG = 8, OL = 1024, D = 5120, per = HQ / OG;
    W wq_b(fmt, QL, HQ), wo_b(fmt, OG * OL, D);
    std::vector<std::unique_ptr<W>> wa;
    x3::Mat wo_a[16];
    for (int gi = 0; gi < OG; ++gi) {
        wa.emplace_back(new W(fmt, per, OL));
        wo_a[gi] = wa.back()->m;
    }
    uint16_t* qr16 = rand16((size_t) T * QL);
    uint16_t* o16 = rand16((size_t) T * HQ);
    uint16_t* oa16 = rand16((size_t) T * OG * OL);
    const g16::Places sz = g16::places(nullptr, wq_b.m, wo_a, OG, wo_b.m);
    uint16_t *w16 = nullptr, *once = nullptr;
    CU(cudaMalloc(&w16, w16_elems * 2));
    CU(cudaMalloc(&once, sz.elems * 2));
    struct Out {
        float* p;
        size_t n;
    };
    Out q{nullptr, (size_t) T * HQ}, oa{nullptr, (size_t) T * OG * OL}, acc{nullptr, (size_t) T * D};
    for (Out* o : {&q, &oa, &acc}) CU(cudaMalloc(&o->p, o->n * 4));
    CU(cudaDeviceSynchronize());
    // gemm()'s loop: pre null - each piece reconstructed into w16; else its piece of pre
    auto gemm = [&](const x3::Mat& m, const uint16_t* pre, const uint16_t* x16, int ldx, int rows, float* y, int ldy) {
        const int cs = g16::cols(m, w16_elems);
        for (int c0 = 0; c0 < m.n; c0 += cs) {
            const int nc = std::min(cs, m.n - c0);
            if (pre == nullptr) g16::dequant(m, c0, nc, w16, s);
            const uint16_t* wp = pre != nullptr ? g16::piece(pre, c0, m.k) : w16;
            if (g16::mm(blas, wp, m.k, nc, x16, ldx, rows, y + c0, ldy, 0.0f) != CUBLAS_STATUS_SUCCESS) {
                std::printf("cublasGemmEx failed\n");
                std::exit(1);
            }
        }
    };
    auto run = [&](bool once_on) {
        for (Out* o : {&q, &oa, &acc}) CU(cudaMemsetAsync(o->p, 0xA5, o->n * 4, s));
        const g16::Places wp = once_on ? g16::places(once, wq_b.m, wo_a, OG, wo_b.m) : g16::Places{};
        if (once_on) {
            auto none = [](int, int) {};
            auto none0 = [] {};
            g16::dequant_whole(wq_b.m, w16_elems, wp.qb, s, none, none0);
            for (int gi = 0; gi < OG; ++gi) g16::dequant_whole(wo_a[gi], w16_elems, wp.oa[gi], s, none, none0);
            g16::dequant_whole(wo_b.m, w16_elems, wp.ob, s, none, none0);
        }
        for (int a0 = 0; a0 < T; a0 += A) {
            const int na = std::min(A, T - a0);
            gemm(wq_b.m, wp.qb, qr16 + (size_t) a0 * QL, QL, na, q.p + (size_t) a0 * HQ, HQ);
            for (int gi = 0; gi < OG; ++gi)
                gemm(wo_a[gi], wp.oa[gi], o16 + (size_t) a0 * HQ + (size_t) gi * per, HQ, na,
                     oa.p + (size_t) a0 * OG * OL + (size_t) gi * OL, OG * OL);
            gemm(wo_b.m, wp.ob, oa16 + (size_t) a0 * OG * OL, OG * OL, na, acc.p + (size_t) a0 * D, D);
        }
        CU(cudaStreamSynchronize(s));
        CU(cudaGetLastError());
        std::vector<uint8_t> r;
        for (Out* o : {&q, &oa, &acc}) {
            std::vector<uint8_t> h(o->n * 4);
            CU(cudaMemcpy(h.data(), o->p, h.size(), cudaMemcpyDeviceToHost));
            r.insert(r.end(), h.begin(), h.end());
        }
        return r;
    };
    const int keep = iq::dq_kernel();
    iq::dq_kernel() = 0;
    const std::vector<uint8_t> ref = run(false);
    size_t filled = 0;   // (outputs gemm()'s order left at the fill: every one must be written)
    for (size_t i = 0; i + 3 < ref.size(); i += 4) filled += std::memcmp(&ref[i], "\xA5\xA5\xA5\xA5", 4) == 0;
    for (int kern = 0; kern < (fmt == gg::kFmtGG ? kKernels : 1); ++kern) {
        iq::dq_kernel() = kern;
        const std::vector<uint8_t> got = run(true);
        size_t diff = 0, first = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (got[i] != ref[i] && diff++ == 0) first = i;
        const bool ok = diff == 0 && filled == 0 && gg::launch_errors() == 0;
        ++g_cases;
        if (!ok) ++g_fail;
        std::printf("%-4s once %-4s wq_b + %d wo_a + wo_b (%.1f MB as FP16), T %d in slabs of %d, dequant kernel %d: %zu "
                    "bytes, %zu differ%s\n",
                    ok ? "ok" : "FAIL", what, OG, (double) sz.elems * 2 / 1e6, T, A, kern, ref.size(), diff,
                    diff ? (" (first at " + std::to_string(first) + ")").c_str() : filled ? " (not all written)" : "");
    }
    iq::dq_kernel() = keep;
    for (Out* o : {&q, &oa, &acc}) CU(cudaFree(o->p));
    CU(cudaFree(once));
    CU(cudaFree(w16));
    CU(cudaFree(oa16));
    CU(cudaFree(o16));
    CU(cudaFree(qr16));
}

void test_once(cudaStream_t s) {
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, s) != CUBLAS_STATUS_SUCCESS) {
        std::printf("FAIL cuBLAS\n");
        ++g_fail;
        return;
    }
    // (gemm()'s pieces at these shapes: wq_b 1280 x 32768 6, a wo_a group 4096 x 1024 one, wo_b 8192 x 5120 5) T = 2
    // slabs of 256 and a short one; Q5_K (the dense K-quants' path, each DS4_DQ_NEW kernel), Q8_0 and BF16
    check_once("Q5_K", gg::kFmtGG, 700, 256, blas, s);
    check_once("Q8_0", gg::kFmtQ8_0, 520, 256, blas, s);
    check_once("BF16", gg::kFmtBF16, 520, 256, blas, s);
    cublasDestroy(blas);
}

// --bench: the pieces timed, each kernel (Q5_K, F16)
void bench(cudaStream_t s) {
    struct P {
        const char* name;
        int k, n;
    };
    const P ps[] = {{"wq_b piece", 1280, 6528}, {"wo_a group", 4096, 1024}, {"wo_b piece", 8192, 1024}};
    for (const TypeInfo& t : {kTypes[9], kTypes[8], kTypes[10]})
        for (const P& p : ps) {
            const std::vector<uint8_t> h = blocks(t, p.n, p.k);
            uint8_t* raw = nullptr;
            uint16_t* dst = nullptr;
            CU(cudaMalloc(&raw, h.size()));
            CU(cudaMemcpyAsync(raw, h.data(), h.size(), cudaMemcpyHostToDevice, s));
            CU(cudaStreamSynchronize(s));
            CU(cudaMalloc(&dst, (size_t) p.n * p.k * 2));
            const size_t ld = iq::row_bytes(t.type, p.k);
            cudaEvent_t e0, e1;
            CU(cudaEventCreate(&e0));
            CU(cudaEventCreate(&e1));
            const int keep = iq::dq_kernel();
            double us[kKernels];
            for (int kern = 0; kern < kKernels; ++kern) {
                iq::dq_kernel() = kern;
                for (int w = 0; w < 3; ++w) iq::dequant_rows(t.type, raw, ld, p.k, p.n, dst, false, s);
                const int reps = 50;
                CU(cudaEventRecord(e0, s));
                for (int i = 0; i < reps; ++i) iq::dequant_rows(t.type, raw, ld, p.k, p.n, dst, false, s);
                CU(cudaEventRecord(e1, s));
                CU(cudaEventSynchronize(e1));
                float ms = 0;
                CU(cudaEventElapsedTime(&ms, e0, e1));
                us[kern] = 1e3 * ms / reps;
            }
            iq::dq_kernel() = keep;
            const double bytes = (double) h.size() + (double) p.n * p.k * 2;   // (read + written)
            std::printf("     bench %-5s %-10s k %5d n %5d: %8.1f us %6.1f GB/s | 1: %8.1f us %6.1f GB/s | 2: %8.1f us %6.1f GB/s\n",
                        iq::name(t.type), p.name, p.k, p.n, us[0], bytes / us[0] / 1e3, us[1], bytes / us[1] / 1e3, us[2],
                        bytes / us[2] / 1e3);
            CU(cudaEventDestroy(e0));
            CU(cudaEventDestroy(e1));
            CU(cudaFree(dst));
            CU(cudaFree(raw));
        }
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop, 0));
    std::printf("info %s (sm_%d%d); DS4_DQ_NEW in the environment: %s (the test sets the kernel itself)\n", prop.name,
                prop.major, prop.minor, std::getenv("DS4_DQ_NEW") ? std::getenv("DS4_DQ_NEW") : "unset");
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    if (do_bench) bench(s);
    else {
        test_types(s);
        test_once(s);
    }
    std::printf("%d of %d cases failed\n", g_fail, g_cases);
    return g_fail ? 1 : 0;
}
