// tests/dsv41/u9_bits_test.cu - the decode GPU-part changes (src/dsv41/ds4_u9.hpp) against the path they replace, bit
// for bit: each case runs the same inputs with the change off (the 747981ba path) and on and compares every output
// byte.  Needs a GPU, no model.  u9_bits_test [--bench: the Q5_K GEMV shapes timed, off and on]
#include "../../src/dsv41/ds4_gguf.cuh"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_kernels.cuh"
#include "../../src/dsv41/ds4_u9.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace x3 = strata::dsv41k::exl3;
namespace u9 = strata::ds4::u9;
namespace sd = strata::ds4;

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
    M s1(gg::kFmtGG, D, FF), s3(gg::kFmtGG, D, FF), wq_b(gg::kFmtGG, Q, 768), q8b(gg::kFmtQ8_0, Q, 768);
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
        // Q8_0 jobs whose x follow at a stride (one launch; the Q8_0 kernel reads the scales in pairs) and a repeat
        std::snprintf(what, sizeof(what), "quant: Q8_0 slices at a stride and a repeat (T %d)", T);
        ab(what, u9::kQuant, out, [&] {
            const size_t cs = (size_t) 8 * Q;
            x3::MvJob j[4] = {job(q8b, xs, q2, T, cs, 4 * 768), job(q8b, xs + 2 * Q, q2 + 768, T, cs, 4 * 768),
                              job(q8b, xs + 4 * Q, q2 + 2 * 768, T, cs, 4 * 768),
                              job(q8b, xs + 2 * Q, q2 + 3 * 768, T, cs, 4 * 768)};
            gg::mv(j, 4, s);
        }, s);
        for (auto& b : out.b) CU(cudaFree(b.first));
        CU(cudaFree(x));
        CU(cudaFree(o));
        CU(cudaFree(xs));
    }
    for (M* m : wo) delete m;
}

// ---- item route: the gate GEMVs over BF16 rows in one launch, the lookahead + routing + publish in one launch
uint16_t bf_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
float f_of_bf(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
void test_route(cudaStream_t s) {
    for (int D : {4096, 5120}) {
        const int n = 384;
        std::vector<uint16_t> wb[3];
        std::vector<float> wf[3];
        uint16_t* dwb[3];
        float* dwf[3];
        for (int m = 0; m < 3; ++m) {
            wb[m].resize((size_t) n * D);
            wf[m].resize((size_t) n * D);
            for (size_t i = 0; i < wb[m].size(); ++i) wf[m][i] = f_of_bf(wb[m][i] = bf_of(frand(0.05f)));
            dwb[m] = dev(wb[m]);
            dwf[m] = dev(wf[m]);
        }
        for (int T : {1, 3}) {
            float* x = xbuf((size_t) T * D);
            Outs out;
            float* y[3];
            for (int m = 0; m < 3; ++m) y[m] = (float*) out.add((size_t) T * 1024 * 4);
            char what[96];
            std::snprintf(what, sizeof(what), "route: 3 gates (BF16, one launch) vs gemv_f32, D %d T %d", D, T);
            ab(what, u9::kRoute, out, [&] {
                if (!u9::on(u9::kRoute)) {
                    for (int m = 0; m < 3; ++m) sd::gemv_f32(dwf[m], x, n, D, T, y[m], 1024, s);
                    return;
                }
                sd::GemvBf16 g = {};
                for (int m = 0; m < 3; ++m) g.w[m] = dwb[m], g.y[m] = y[m], g.n[m] = n;
                sd::gemv_bf16(g, 3, x, D, T, 1024, s);
            }, s);
            for (auto& b : out.b) CU(cudaFree(b.first));
            CU(cudaFree(x));
        }
        for (int m = 0; m < 3; ++m) {
            CU(cudaFree(dwb[m]));
            CU(cudaFree(dwf[m]));
        }
    }
    // the routing: scores with ties (logits and biases from a few values: many experts at the best score) and without;
    // image rows (vl) with bias_vl; experts on the CPU lane (vram 0) so the x rows go into the request; 0, 1 and 2
    // lookahead layers, a request or none
    const int n = 384, k = 6, D = 5120;
    std::vector<unsigned long long> vram(n);
    for (int e = 0; e < n; ++e) vram[e] = (e % 3 == 0) ? 0ull : 0x1000ull + (unsigned long long) e * 64;
    unsigned long long* dvram = dev(vram);
    for (int ties = 0; ties < 2; ++ties) {
        std::vector<float> bias(n), bias_vl(n), bias1(n), bias2(n);
        for (auto* b : {&bias, &bias_vl, &bias1, &bias2})
            for (auto& v : *b) v = ties ? (float) (rng() % 2) * 0.25f : frand(0.2f);
        float *dbias = dev(bias), *dbias_vl = dev(bias_vl), *dbias1 = dev(bias1), *dbias2 = dev(bias2);
        for (int T : {1, 3}) {
            std::vector<float> lg((size_t) 3 * sd::kWin * 1024);
            for (size_t i = 0; i < lg.size(); ++i) lg[i] = ties ? (float) ((int) (rng() % 9) - 4) * 0.5f : frand(4.0f);
            float* dlg = dev(lg);   // [own rows (T x 1024)][lookahead 1 (kWin x 1024)][lookahead 2]
            float* x = xbuf((size_t) T * D);
            for (int nah = 0; nah <= 2; ++nah)
                for (int with_req = 0; with_req < 2; ++with_req) {
                    if (!with_req && nah > 0) continue;
                    Outs out;
                    int* idx = (int*) out.add((size_t) T * sd::kPicks * 4);
                    float* w = (float*) out.add((size_t) T * sd::kPicks * 4);
                    unsigned long long* plan = (unsigned long long*) out.add((size_t) T * sd::kPicks * 8);
                    sd::Req* req = with_req ? (sd::Req*) out.add(sizeof(sd::Req)) : nullptr;
                    const float* l1 = dlg + (size_t) sd::kWin * 1024;
                    const float* l2 = l1 + (size_t) sd::kWin * 1024;
                    const int m1 = 9, m2 = 6;
                    char what[110];
                    std::snprintf(what, sizeof(what), "route: fused routing, %s, T %d, %d lookahead%s", ties ? "ties" : "no ties",
                                  T, nah, with_req ? ", request" : ", no request");
                    ab(what, u9::kRoute, out, [&] {
                        const unsigned vl = T > 1 ? 0x2u : 0u;   // (token 1: an image row)
                        if (!u9::on(u9::kRoute)) {
                            if (nah >= 1) sd::route_ahead(l1, dbias1, n, m1, T, req, 0, s);
                            if (nah >= 2) sd::route_ahead(l2, dbias2, n, m2, T, req, sd::kAhead1, s);
                            sd::route_publish(dlg, dbias, n, k, T, 2.5f, dvram, x, D, idx, w, plan, req, 77u, s, dbias_vl,
                                              vl, nah >= 1 ? m1 : 0, nah >= 2 ? m2 : 0);
                            return;
                        }
                        sd::RouteAhead ra = {};
                        ra.logits[0] = l1, ra.bias[0] = dbias1, ra.n[0] = n, ra.m[0] = m1, ra.off[0] = 0;
                        ra.logits[1] = l2, ra.bias[1] = dbias2, ra.n[1] = n, ra.m[1] = m2, ra.off[1] = sd::kAhead1;
                        sd::route_fused(dlg, dbias, n, k, T, 2.5f, dvram, x, D, idx, w, plan, req, 77u, s, dbias_vl, vl,
                                        nah >= 1 ? m1 : 0, nah >= 2 ? m2 : 0, ra, nah);
                    }, s);
                    for (auto& b : out.b) CU(cudaFree(b.first));
                }
            CU(cudaFree(dlg));
            CU(cudaFree(x));
        }
        for (float* p : {dbias, dbias_vl, dbias1, dbias2}) CU(cudaFree(p));
    }
}

// ---- the split-KV decode attention (DS4_ATTN_SPLIT, on with the U9 defaults): its merge sums in another order, so it
// is checked against a double reference (with the old kernel beside it), at its edges: a partial last chunk, a verify
// window's queries each with its own rows, a short window, no selected rows, the 960-row cap; and where it must fall
// back to the old kernel (above the cap, its scratch not allocated) the old kernel's bytes
float e4m3_val(uint8_t q) {
    const int e = (q >> 3) & 15, m = q & 7;
    const float v = e == 0 ? m * (1.f / 512.f) : std::ldexp(1.f + m / 8.f, e - 7);
    return (q & 0x80) ? -v : v;
}
float e2m1_val(int c) {
    static const float v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}
struct SplitCase {
    const char* name;
    int heads, T, nwin[5], nsel[5];
    bool fallback;   // (above the cap: the old kernel's bytes)
};
const SplitCase kSplitCases[] = {   // (64 heads first: the split's scratch is sized by a stream's first call)
    {"T 1, win 128 + sel 512 (16 chunks)", 64, 1, {128}, {512}, false},
    {"T 1, win 128 + sel 301 (a partial last chunk)", 64, 1, {128}, {301}, false},
    {"T 1, win 37, no sel", 64, 1, {37}, {0}, false},
    {"T 4, win 5..128, sel 0..832 (960: the cap)", 64, 4, {5, 60, 128, 128}, {0, 77, 512, 832}, false},
    {"T 5, the first tokens (win 1..5, no sel)", 64, 5, {1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, false},
    {"T 2, 16 heads (TP 4), win 128 + sel 200 / 512", 16, 2, {128, 128}, {200, 512}, false},
    {"T 3, sel to 833 (961: above the cap)", 64, 3, {128, 128, 128}, {100, 400, 833}, true},
};
void test_split(cudaStream_t s) {
    const int d = 512, ncache = 2048, stride = 1024;
    const size_t rb = sd::kv_row_bytes(sd::kKvFp4, d);
    const float scale = 1.f / std::sqrt((float) d);
    std::normal_distribution<float> nrand(0.f, 1.f);
    std::vector<uint8_t> ckv((size_t) ncache * rb, 0);
    std::vector<float> cdec((size_t) ncache * d);
    for (int p = 0; p < ncache; ++p) {
        uint8_t* r = &ckv[(size_t) p * rb];
        for (int j = 0; j < d / 2; ++j) r[j] = (uint8_t) (rng() & 0xff);
        for (int g = 0; g < d / 16; ++g) r[d / 2 + g] = (uint8_t) ((7 << 3) | (rng() & 7)) - (uint8_t) (rng() % 3 * 8);
        for (int e = 0; e < d; ++e)
            cdec[(size_t) p * d + e] = e2m1_val((r[e / 2] >> (4 * (e & 1))) & 15) * e4m3_val(r[d / 2 + e / 16]);
    }
    uint8_t* dck = dev(ckv);
    sd::AttnScratch scr;
    CU(cudaMalloc((void**) &scr.rows, (size_t) sd::kAttnScratchT * stride * d * 2));
    scr.max_t = sd::kAttnScratchT;
    scr.cap = stride;
    const bool keep = sd::attn_split();
    // (the last round: the first case on a new stream with the device's memory taken for a moment, so the scratch
    // fails - for an idle GPU: U9_TEST_STARVE=0 leaves it out where another process allocates on the same card)
    const int ncase = (int) (sizeof(kSplitCases) / sizeof(kSplitCases[0]));
    const char* sv = std::getenv("U9_TEST_STARVE");
    const int rounds = ncase + (sv != nullptr && std::atoi(sv) == 0 ? 0 : 1);
    for (int ci = 0; ci < rounds; ++ci) {
        const bool starve = ci == ncase;
        const SplitCase& c = kSplitCases[starve ? 0 : ci];
        const int T = c.T, heads = c.heads;
        sd::AttnRows ar = {};
        int wrows = 0;
        for (int t = 0; t < T; ++t) {
            ar.win0[t] = t;
            ar.nwin[t] = c.nwin[t];
            ar.nsel[t] = c.nsel[t];
            wrows = std::max(wrows, t + c.nwin[t]);
        }
        std::vector<uint16_t> win((size_t) wrows * d);
        std::vector<float> wdec(win.size());
        for (size_t i = 0; i < win.size(); ++i) {
            const float f = nrand(rng);
            uint32_t u;
            std::memcpy(&u, &f, 4);
            win[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
            const uint32_t b = (uint32_t) win[i] << 16;
            std::memcpy(&wdec[i], &b, 4);
        }
        std::vector<float> q((size_t) T * heads * d), sink(heads);
        for (auto& v : q) v = nrand(rng);
        for (auto& v : sink) v = frand();
        std::vector<int> sel((size_t) T * stride, 0);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < c.nsel[t]; ++i) sel[(size_t) t * stride + i] = (int) (rng() % ncache);
        uint16_t* dwin = dev(win);
        float *dq = dev(q), *dsink = dev(sink);
        int* dsel = dev(sel);
        Outs out;
        float* o = (float*) out.add((size_t) T * heads * d * 4);
        cudaStream_t st = s;
        if (starve) CU(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        std::vector<uint8_t> r[2];
        void* hog = nullptr;
        unsigned long long ran[2] = {0, 0};   // (the split kernel's launches in each run)
        for (int on = 0; on < 2; ++on) {
            if (starve && on == 1) {   // (all but a few MiB taken, less than the split's ~16 MB scratch)
                size_t fr = 0, tot = 0;
                CU(cudaMemGetInfo(&fr, &tot));
                for (size_t keep_mib = 2; hog == nullptr && keep_mib <= 12 && fr > (keep_mib << 20); keep_mib += 2)
                    if (cudaMalloc(&hog, fr - (keep_mib << 20)) != cudaSuccess) {
                        cudaGetLastError();
                        hog = nullptr;
                    }
            }
            sd::attn_split() = on == 1;
            out.clear();
            const unsigned long long n0 = sd::attn_split_runs();
            sd::sparse_attn(dq, dwin, dck, dsel, stride, ar, dsink, heads, d, T, scale, o, st, sd::kKvFp4, 0, &scr);
            CU(cudaStreamSynchronize(st));
            CU(cudaGetLastError());
            ran[on] = sd::attn_split_runs() - n0;
            r[on] = out.get();
        }
        if (hog != nullptr) CU(cudaFree(hog));
        sd::attn_split() = keep;
        // the reference: every query, every 5th head, in double
        double err[2] = {0, 0}, mx = 0;
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < heads; h += 5) {
                std::vector<const float*> rows;
                for (int j = 0; j < c.nwin[t]; ++j) rows.push_back(&wdec[(size_t) (ar.win0[t] + j) * d]);
                for (int i = 0; i < c.nsel[t]; ++i) rows.push_back(&cdec[(size_t) sel[(size_t) t * stride + i] * d]);
                const float* qh = &q[((size_t) t * heads + h) * d];
                std::vector<double> sc(rows.size());
                double m = sink[h];
                for (size_t j = 0; j < rows.size(); ++j) {
                    double a = 0;
                    for (int e = 0; e < d; ++e) a += (double) qh[e] * rows[j][e];
                    sc[j] = a * scale;
                    m = std::max(m, sc[j]);
                }
                double den = std::exp(sink[h] - m);
                for (double v : sc) den += std::exp(v - m);
                std::vector<double> acc(d, 0.0);
                for (size_t j = 0; j < rows.size(); ++j) {
                    const double p = std::exp(sc[j] - m) / den;
                    for (int e = 0; e < d; ++e) acc[e] += p * rows[j][e];
                }
                for (int e = 0; e < d; ++e) {
                    mx = std::max(mx, std::fabs(acc[e]));
                    for (int on = 0; on < 2; ++on) {
                        float g;
                        std::memcpy(&g, &r[on][(((size_t) t * heads + h) * d + e) * 4], 4);
                        err[on] = std::max(err[on], std::fabs(g - acc[e]));
                    }
                }
            }
        for (double& e : err) e /= mx + 1e-30;
        size_t diff = 0;
        for (size_t i = 0; i < r[0].size(); ++i) diff += r[0][i] != r[1][i];
        const bool same = c.fallback || starve;   // (the old kernel must run: its bytes)
        // (starved, the split ran anyway: the scratch was allocated after all - a driver that spills to host memory)
        const bool skip = starve && (hog == nullptr || ran[1] != 0);
        const bool ok = err[0] < 1e-4 && err[1] < 1e-4 && ran[0] == 0 && ran[1] == (same ? 0u : 1u) &&
                        (!same || diff == 0) && (hog != nullptr) == starve;
        if (!skip) {
            ++g_cases;
            if (!ok) ++g_fail;
        }
        std::printf("%-4s split: %s%s   rel err old %.1e split %.1e, split launches %llu, %zu bytes differ%s\n",
                    skip ? "skip" : ok ? "ok" : "FAIL", c.name, starve ? " - no scratch" : "", err[0], err[1], ran[1],
                    diff, same ? " (must be 0: the old kernel)" : "");
        if (starve) CU(cudaStreamDestroy(st));
        for (void* p : std::initializer_list<void*>{dwin, dq, dsink, dsel}) CU(cudaFree(p));
        for (auto& b : out.b) CU(cudaFree(b.first));
    }
    CU(cudaFree(scr.rows));
    CU(cudaFree(dck));
}

// ---- item q5k: the staged Q5_K GEMV (the decode's dense shapes: wq_b, wo_a's slices, wo_b, the shared expert, the engram's
// wkv (layers 1 and 14), the head)
struct Shape {
    const char* name;
    int n, k, jobs;
};
const Shape kQ5Shapes[] = {{"wq_b", 32768, 1280, 1}, {"wo_a x 8", 1024, 4096, 8}, {"wo_b", 5120, 8192, 1},
                           {"s1 + s3", 2304, 5120, 2}, {"s2", 5120, 2304, 1}, {"engram wkv", 25600, 6144, 1},
                           {"head", 129280, 5120, 1}, {"odd rows", 1003, 768, 3}};
void test_q5k(cudaStream_t s, bool bench) {
    for (const Shape& sh : kQ5Shapes)
        for (int T : {1, 3, 16}) {
            if (bench && T != 1) continue;
            std::vector<M*> m;
            for (int j = 0; j < sh.jobs; ++j) m.push_back(new M(gg::kFmtGG, sh.k, sh.n));
            float* x = xbuf((size_t) T * sh.k * sh.jobs);
            Outs out;
            float* y = (float*) out.add((size_t) T * sh.n * sh.jobs * 4);
            auto go = [&] {
                x3::MvJob j[8];
                for (int i = 0; i < sh.jobs; ++i)
                    j[i] = job(*m[i], x + (size_t) i * sh.k, y + (size_t) i * sh.n, T, (size_t) sh.k * sh.jobs,
                               (size_t) sh.n * sh.jobs);
                gg::mv(j, sh.jobs, s);
            };
            char what[96];
            std::snprintf(what, sizeof(what), "q5k: %s (n %d k %d) T %d", sh.name, sh.n, sh.k, T);
            ab(what, u9::kQ5k, out, go, s);
            if (bench) {   // us a launch, off / on (the int8 rounding included), and the weights' GB/s
                const double bytes = (double) gg::type_row_bytes(gg::iq::kQ5_K, sh.k) * sh.n * sh.jobs;
                const unsigned keep = u9::mask();
                double us[2];
                for (int on = 0; on < 2; ++on) {
                    u9::mask() = on ? (keep | u9::kQ5k) : (keep & ~u9::kQ5k);
                    cudaEvent_t e0, e1;
                    CU(cudaEventCreate(&e0));
                    CU(cudaEventCreate(&e1));
                    for (int r = 0; r < 5; ++r) go();
                    const int reps = 50;
                    CU(cudaEventRecord(e0, s));
                    for (int r = 0; r < reps; ++r) go();
                    CU(cudaEventRecord(e1, s));
                    CU(cudaEventSynchronize(e1));
                    float ms = 0;
                    CU(cudaEventElapsedTime(&ms, e0, e1));
                    us[on] = 1000.0 * ms / reps;
                    CU(cudaEventDestroy(e0));
                    CU(cudaEventDestroy(e1));
                }
                u9::mask() = keep;
                std::printf("     bench %-10s %9.1f us -> %9.1f us   %6.1f -> %6.1f GB/s\n", sh.name, us[0], us[1],
                            bytes / us[0] / 1e3, bytes / us[1] / 1e3);
            }
            for (auto& b : out.b) CU(cudaFree(b.first));
            CU(cudaFree(x));
            for (M* p : m) {
                CU(cudaFree(p->w));
                delete p;
            }
        }
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    if (std::getenv("DS4_U9_Q5K") == nullptr) u9::q5k_arch() = 1;   // (the staged kernel on any device: checked here)
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    if (!gg::prepare(s)) return 1;
    if (!bench) test_quant(s);
    if (!bench) test_route(s);
    if (!bench) test_split(s);
    if (!bench) {   // (U9 hc's side stream and events are made after free0: what they take of the free VRAM, if any)
        size_t f0 = 0, f1 = 0, tot = 0;
        CU(cudaMemGetInfo(&f0, &tot));
        cudaStream_t s2;
        cudaEvent_t e[2];
        CU(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        for (auto& ev : e) CU(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CU(cudaMemGetInfo(&f1, &tot));
        std::printf("info a non-blocking stream and two events took %lld bytes of the free VRAM\n",
                    (long long) f0 - (long long) f1);
        for (auto& ev : e) CU(cudaEventDestroy(ev));
        CU(cudaStreamDestroy(s2));
    }
    test_q5k(s, bench);
    std::printf("%d of %d cases failed\n", g_fail, g_cases);
    return g_fail ? 1 : 0;
}
