// tests/dsv41/dq_test.cu - the chunked prompt's dense dequant (ds4_iq.cu's dequant_rows, gg::rows' K-quant / i-quant
// path) with each kernel DS4_DQ_NEW selects against the row-per-lane one it is unset, byte for byte: random blocks of
// every type dequant_rows takes (any code bits, F16 scales random and at the edges: zeros, subnormals, the largest),
// several row counts, first rows (gemm()'s column pieces: a view's rows c0 ..) and widths, F16 and BF16, and the bytes
// around the output untouched.  Needs a GPU, no model.  dq_test [--bench: the attention slab's pieces timed, each kernel]
#include "../../src/dsv41/ds4_iq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace iq = strata::ds4::gguf::iq;

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
uint16_t scale16() {
    static const uint16_t edge[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0xfbff, 0x1400};
    if (rng() % 4 == 0) return edge[rng() % (sizeof(edge) / sizeof(edge[0]))];
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

// rows x k of random blocks of the type
std::vector<uint8_t> blocks(const TypeInfo& t, int rows, int k) {
    const int bb = iq::block_bytes(t.type);
    std::vector<uint8_t> w(iq::row_bytes(t.type, k) * (size_t) rows);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    for (size_t o = 0; o + bb <= w.size(); o += (size_t) bb) {
        uint8_t* p = &w[o];
        if (t.type == iq::kIQ1_M) {
            const uint16_t d = scale16();
            p[49] = (uint8_t) ((p[49] & 0x0f) | ((d & 0x000f) << 4));
            p[51] = (uint8_t) ((p[51] & 0x0f) | (d & 0x00f0));
            p[53] = (uint8_t) ((p[53] & 0x0f) | ((d & 0x0f00) >> 4));
            p[55] = (uint8_t) ((p[55] & 0x0f) | ((d & 0xf000) >> 8));
            continue;
        }
        put16(p + t.d, scale16());
        if (t.dmin >= 0) put16(p + t.dmin, scale16());
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
    else test_types(s);
    std::printf("%d of %d cases failed\n", g_fail, g_cases);
    return g_fail ? 1 : 0;
}
