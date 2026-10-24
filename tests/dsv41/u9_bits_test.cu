// tests/dsv41/u9_bits_test.cu - the decode GPU-part changes (src/dsv41/ds4_u9.hpp) against the path they replace, bit
// for bit: each case runs the same inputs with the change off (the 747981ba path) and on and compares every output
// byte.  Needs a GPU, no model.  u9_bits_test
#include "../../src/dsv41/ds4_gguf.cuh"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_u9.hpp"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace x3 = strata::dsv41k::exl3;
namespace u9 = strata::ds4::u9;

namespace {

int g_fail = 0, g_cases = 0;
std::mt19937 rng(925);

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, h.size() * sizeof(T)));
    CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
std::vector<uint8_t> host_bytes(const void* p, size_t n) {
    std::vector<uint8_t> h(n);
    CU(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost));
    return h;
}
void put_half(uint8_t* at, float f) {
    const __half h = __float2half(f);
    std::memcpy(at, &h, 2);
}
// random rows of a Q5_K (d, dmin small) / Q8_0 (d small) / BF16 matrix, n rows of k
std::vector<uint8_t> rows_q5k(int n, int k) {
    std::vector<uint8_t> w(gg::type_row_bytes(gg::iq::kQ5_K, k) * n);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    for (size_t o = 0; o < w.size(); o += 176) {
        put_half(&w[o], 0.001f + 0.002f * std::fabs(frand()));
        put_half(&w[o + 2], 0.0005f + 0.001f * std::fabs(frand()));
    }
    return w;
}
std::vector<uint8_t> rows_q8(int n, int k) {
    std::vector<uint8_t> w((size_t) n * (k / 32) * 34);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    for (size_t o = 0; o < w.size(); o += 34) put_half(&w[o], 0.001f + 0.002f * std::fabs(frand()));
    return w;
}
std::vector<uint8_t> rows_bf16(int n, int k) {
    std::vector<uint8_t> w((size_t) n * k * 2);
    for (size_t o = 0; o < w.size(); o += 2) {
        const float f = frand(0.05f);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        const uint16_t b = (uint16_t) (u >> 16);
        std::memcpy(&w[o], &b, 2);
    }
    return w;
}

// a set of device buffers the case writes (compared off vs on) after a run
struct Outs {
    std::vector<std::pair<void*, size_t>> b;
    void* add(size_t bytes) {
        void* p = nullptr;
        CU(cudaMalloc(&p, bytes));
        b.push_back({p, bytes});
        return p;
    }
    void clear() {
        for (auto& x : b) CU(cudaMemset(x.first, 0xA5, x.second));
        CU(cudaDeviceSynchronize());   // (the legacy stream's memset does not order against a non-blocking stream)
    }
    std::vector<uint8_t> get() const {
        std::vector<uint8_t> all;
        for (auto& x : b) {
            std::vector<uint8_t> h = host_bytes(x.first, x.second);
            all.insert(all.end(), h.begin(), h.end());
        }
        return all;
    }
};
// runs `go` with `item` off then on; the outputs must match byte for byte (and differ from the 0xA5 fill somewhere)
void ab(const char* what, unsigned item, Outs& o, const std::function<void()>& go, cudaStream_t s) {
    const unsigned keep = u9::mask();
    std::vector<uint8_t> r[2];
    for (int on = 0; on < 2; ++on) {
        u9::mask() = on ? (keep | item) : (keep & ~item);
        o.clear();
        go();
        CU(cudaStreamSynchronize(s));
        CU(cudaGetLastError());
        r[on] = o.get();
    }
    u9::mask() = keep;
    ++g_cases;
    size_t diff = 0, first = 0, filled = 0;
    for (size_t i = 0; i < r[0].size(); ++i) {
        if (r[0][i] != r[1][i] && diff++ == 0) first = i;
        filled += r[0][i] == 0xA5;
    }
    const bool ok = diff == 0 && filled < r[0].size() && gg::launch_errors() == 0;
    if (!ok) ++g_fail;
    std::printf("%-4s %-58s %zu bytes, %zu differ%s\n", ok ? "ok" : "FAIL", what, r[0].size(), diff,
                diff ? (" (first at " + std::to_string(first) + ")").c_str() : "");
}

struct M {
    x3::Mat m;
    uint8_t* w = nullptr;
    M(int fmt, int k, int n) {
        std::vector<uint8_t> h = fmt == gg::kFmtGG ? rows_q5k(n, k) : fmt == gg::kFmtQ8_0 ? rows_q8(n, k) : rows_bf16(n, k);
        w = dev(h);
        m.fmt = fmt;
        m.K = fmt == gg::kFmtGG ? gg::iq::kQ5_K : 0;
        m.raw = w;
        m.k = k;
        m.n = n;
    }
};
x3::MvJob job(const M& m, const float* x, float* y, int T = 1, size_t xcs = 0, size_t ycs = 0) {
    x3::MvJob j;
    j.m = &m.m;
    j.xf = x;
    j.y = y;
    j.ncol = T;
    j.xf_cs = xcs;
    j.y_cs = ycs;
    return j;
}
float* xbuf(size_t n) {
    std::vector<float> h(n);
    for (auto& v : h) v = frand();
    return dev(h);
}

// ---- item quant: gg::mv's int8 inputs
void test_quant(cudaStream_t s) {
    const int D = 4096, Q = 1024, HD = 512, HQ = 8 * 4096, OL = 1024, FF = 2048;
    M wq_a(gg::kFmtQ8_0, D, Q), wkv(gg::kFmtQ8_0, D, HD), cw(gg::kFmtBF16, D, HD);
    M s1(gg::kFmtGG, D, FF), s3(gg::kFmtGG, D, FF), wq_b(gg::kFmtGG, Q, 768);
    std::vector<M*> wo;
    for (int g = 0; g < 8; ++g) wo.push_back(new M(gg::kFmtGG, HQ / 8, OL));
    for (int T : {1, 3}) {
        float* x = xbuf((size_t) T * D);
        float* o = xbuf((size_t) T * HQ);
        Outs out;
        float* qr = (float*) out.add((size_t) T * Q * 4);
        float* kv = (float*) out.add((size_t) T * HD * 4);
        float* ck = (float*) out.add((size_t) T * HD * 4);
        float* oa = (float*) out.add((size_t) T * 8 * OL * 4);
        float* sg = (float*) out.add((size_t) T * FF * 4);
        float* su = (float*) out.add((size_t) T * FF * 4);
        float* q2 = (float*) out.add((size_t) T * 4 * 768 * 4);
        char what[96];
        std::snprintf(what, sizeof(what), "quant: wq_a + wkv + compressor, the same x (T %d)", T);
        ab(what, u9::kQuant, out, [&] {
            x3::MvJob j[3] = {job(wq_a, x, qr, T, D, Q), job(wkv, x, kv, T, D, HD), job(cw, x, ck, T, D, HD)};
            gg::mv(j, 3, s);
        }, s);
        std::snprintf(what, sizeof(what), "quant: wo_a's 8 slices of o (T %d)", T);
        ab(what, u9::kQuant, out, [&] {
            x3::MvJob j[8];
            for (int g = 0; g < 8; ++g) j[g] = job(*wo[g], o + (size_t) g * (HQ / 8), oa + (size_t) g * OL, T, HQ, 8 * OL);
            gg::mv(j, 8, s);
        }, s);
        std::snprintf(what, sizeof(what), "quant: s1 + s3, the same x (T %d)", T);
        ab(what, u9::kQuant, out, [&] {
            x3::MvJob j[2] = {job(s1, x, sg, T, D, FF), job(s3, x, su, T, D, FF)};
            gg::mv(j, 2, s);
        }, s);
        // mixed: Q5_K x, Q8_0 x (shares), Q5_K x + k (joins the first launch), x + 3k (a gap: its own), x + 3k again
        // (shares), then x + 4k (joins that one)
        std::snprintf(what, sizeof(what), "quant: mixed types, strides and repeats (T %d)", T);
        float* xs = xbuf((size_t) 8 * Q * T);
        ab(what, u9::kQuant, out, [&] {
            const size_t cs = (size_t) 8 * Q;
            x3::MvJob j[6] = {job(wq_b, xs, q2, T, cs, 4 * 768),
                              job(wq_b, xs + Q, q2 + 768, T, cs, 4 * 768),
                              job(wq_b, xs + 3 * Q, q2 + 2 * 768, T, cs, 4 * 768),
                              job(wq_b, xs + 4 * Q, q2 + 3 * 768, T, cs, 4 * 768),
                              job(wq_b, xs + 3 * Q, sg, T, cs, FF),
                              job(wq_b, xs, su, T, cs, FF)};
            gg::mv(j, 6, s);
        }, s);
        for (auto& b : out.b) CU(cudaFree(b.first));
        CU(cudaFree(x));
        CU(cudaFree(o));
        CU(cudaFree(xs));
    }
    for (M* m : wo) delete m;
}

}  // namespace

int main() {
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    if (!gg::prepare(s)) return 1;
    test_quant(s);
    std::printf("%d of %d cases failed\n", g_fail, g_cases);
    return g_fail ? 1 : 0;
}
