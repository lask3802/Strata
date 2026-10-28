// tests/dsv41/attn_tc_test.cu - DS4_ATTN_TC (ds4_attn_tc.cuh): the chunked prompt's sparse attention on tensor cores
// against the packed rows' kernel it replaces (sparse_attn_packed_kernel, what the slab runs with DS4_ATTN_TC unset)
// and against a double reference on a few queries of each shape.  The inputs as the prompt path makes them: 64 heads x
// d 512, a slab of c.A = 256 queries (8 MB / (64 x 512 x 4 B)), the window buffer's rows (BF16 on the window ring's FP8
// grid: finish quant 1, as the engine's fk) and the compressed rows packed by finish's kKvFp4 packing (as the engine's
// fc), RMS-normed rows with outlier channels, q of per-head scales 0.5 .. 16 (some peaky heads), sinks, each query's
// selection min(index_topk 512, (p + 1) / R) distinct rows (neighbours sharing most of theirs, as top-k's do), at
// positions about 0, 1K, 2K, 8K and 30K, ratios 0 (window only), 1, 2, 4, a slab not the chunk's first, a short slab, 32
// heads (two members) and 48 (not a multiple of 32: the packed kernel, the same bits).  Needs a GPU, no model.
// What it does not see: the engine's real q / K values (the KL window on the hosts checks those) and the wiring in
// layer_chunk (the same sparse_attn call; the switch is read inside sparse_attn).
//   attn_tc_test [--bench [--runs N]: each shape's time, the packed kernel and the three modes] [--qscale X: every
//   head's q scale X]
#include "../../src/dsv41/ds4_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace dk = strata::ds4;

namespace {

int g_fail = 0;
float g_qscale = 0.0f;   // (--qscale: every head's)
std::mt19937 rng(1104);

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

float urand() { return (float) ((rng() >> 8) * (1.0 / 16777216.0)); }
float grand() {   // a standard normal (Box-Muller)
    const float u = std::max(urand(), 1e-7f), v = urand();
    return std::sqrt(-2.0f * std::log(u)) * std::cos(6.2831853f * v);
}

template <class T>
T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    CU(cudaMalloc(&p, v.size() * sizeof(T)));
    CU(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    CU(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

// n rows of d values as the KV paths feed finish: normal values, then (finish) rmsnorm times w - its channels 0.3 ..
// 1.5, one in 50 eight times that
std::vector<float> norm_weights(int d) {
    std::vector<float> w((size_t) d);
    for (float& v : w) v = (0.3f + 1.2f * urand()) * (rng() % 50 == 0 ? 8.0f : 1.0f);
    return w;
}
// finish(rmsnorm w, quant) of n rows: BF16 rows (quant 1, the window buffer) or packed rows (kKvFp4, the cache's)
void kv_rows(int n, int d, const float* w, int quant, uint16_t* bf, uint8_t* pk, cudaStream_t s) {
    std::vector<float> x((size_t) n * d);
    for (float& v : x) v = grand();
    float* dx = dev(x);
    dk::Finish f;
    f.w = w;
    f.eps = 1e-6f;
    f.quant = quant;
    if (pk != nullptr) {
        f.pdst = pk;
        f.pdst_stride = (long long) dk::kv_row_bytes(quant, d);
    } else {
        f.dst = bf;
        f.dst_stride = d;
    }
    dk::finish(dx, n, d, d, f, s);
    CU(cudaStreamSynchronize(s));
    CU(cudaFree(dx));
}

double h_e4m3(uint8_t q) {
    const int e = (q >> 3) & 15, m = q & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return (q & 0x80) ? -v : v;
}
double h_e2m1(uint32_t c) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (c & 8u) ? -mag[c & 7u] : mag[c & 7u];
}
double h_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Shape {
    const char* what;
    int heads;
    long long p0;
    int t0, na, R;
};

struct Case {
    Shape sh;
    int C = 0, NC = 0, SEL = 512;
    dk::AttnRows ar;
    std::vector<float> q, sink, hs;   // (hs: each head's q scale)
    std::vector<int> sel;
    std::vector<uint16_t> ext;   // (host copies for the reference)
    std::vector<uint8_t> ckv;
    float *d_q = nullptr, *d_sink = nullptr, *d_w = nullptr;
    int* d_sel = nullptr;
    uint16_t* d_ext = nullptr;
    uint8_t* d_ckv = nullptr;
    size_t crow = dk::kv_row_bytes(dk::kKvFp4, 512);
    // query t's rows, as the kernels see them
    void rows(int t, int& w0, int& n_win, int& n_sel) const {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = std::max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) std::min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    }
};

constexpr int HD = 512, WIN = 128;

void make_case(Case& c, cudaStream_t s) {
    const Shape& sh = c.sh;
    c.C = sh.t0 + sh.na;
    c.NC = sh.R > 0 ? (int) ((sh.p0 + c.C) / sh.R) + 1 : 1;
    c.ar.gen = 1;
    c.ar.p0 = sh.p0;
    c.ar.t0 = sh.t0;
    c.ar.win = WIN;
    c.ar.R = sh.R;
    c.ar.kcap = c.SEL;
    const std::vector<float> w = norm_weights(HD);
    c.d_w = dev(w);
    CU(cudaMalloc(&c.d_ext, (size_t) (WIN + c.C) * HD * 2));
    CU(cudaMalloc(&c.d_ckv, (size_t) c.NC * c.crow));
    kv_rows(WIN + c.C, HD, c.d_w, dk::kKvFp8, c.d_ext, nullptr, s);
    kv_rows(c.NC, HD, c.d_w, dk::kKvFp4, nullptr, c.d_ckv, s);
    c.ext = host(c.d_ext, (size_t) (WIN + c.C) * HD);
    c.ckv = host(c.d_ckv, (size_t) c.NC * c.crow);
    // q: head h's scale 0.5 .. 16 (log-uniform), a few channels larger
    c.q.resize((size_t) sh.na * sh.heads * HD);
    std::vector<float>& hs = c.hs;
    hs.resize((size_t) sh.heads);
    for (float& v : hs) v = g_qscale > 0.0f ? g_qscale : 0.5f * std::pow(32.0f, urand());
    for (int t = 0; t < sh.na; ++t)
        for (int h = 0; h < sh.heads; ++h)
            for (int e = 0; e < HD; ++e) c.q[((size_t) t * sh.heads + h) * HD + e] = hs[(size_t) h] * grand();
    c.sink.resize((size_t) sh.heads);
    for (float& v : c.sink) v = -2.0f + 8.0f * urand();
    // the selections: all rows while they fit, else 512 distinct of them, each query keeping 92% of its neighbour's
    c.sel.assign((size_t) sh.na * c.SEL, 0);
    std::vector<int> cur;
    for (int t = 0; t < sh.na; ++t) {
        int w0, nw, ns;
        c.rows(t, w0, nw, ns);
        const int lim = sh.R > 0 ? (int) ((sh.p0 + sh.t0 + t + 1) / sh.R) : 0;
        int* out = c.sel.data() + (size_t) t * c.SEL;
        if (ns == 0) continue;
        if (ns >= lim) {
            for (int i = 0; i < ns; ++i) out[i] = i;
            cur.clear();
            continue;
        }
        std::vector<char> in((size_t) lim, 0);
        std::vector<int> nxt;
        for (int v : cur)
            if (v < lim && urand() < 0.92f && !in[(size_t) v]) {
                in[(size_t) v] = 1;
                nxt.push_back(v);
            }
        while ((int) nxt.size() < ns) {
            const int v = (int) (rng() % (unsigned) lim);
            if (!in[(size_t) v]) {
                in[(size_t) v] = 1;
                nxt.push_back(v);
            }
        }
        std::shuffle(nxt.begin(), nxt.end(), rng);
        std::copy(nxt.begin(), nxt.end(), out);
        cur = nxt;
    }
    c.d_q = dev(c.q);
    c.d_sink = dev(c.sink);
    c.d_sel = dev(c.sel);
}

void free_case(Case& c) {
    CU(cudaFree(c.d_q));
    CU(cudaFree(c.d_sink));
    CU(cudaFree(c.d_w));
    CU(cudaFree(c.d_sel));
    CU(cudaFree(c.d_ext));
    CU(cudaFree(c.d_ckv));
}

// the slab's attention with the switch at mode (0: the packed kernel; 1, 2, 3 the tensor-core modes, 4 the test's)
void run(const Case& c, int mode, float* out, cudaStream_t s) {
    dk::attn_tc() = mode;
    dk::sparse_attn(c.d_q, c.d_ext, c.d_ckv, c.d_sel, c.SEL, c.ar, c.d_sink, c.sh.heads, HD, c.sh.na,
                    1.0f / std::sqrt((float) HD), out, s, dk::kKvFp4, 0, nullptr);
}

// query t's outputs in double
void reference(const Case& c, int t, std::vector<double>& o) {
    const int H = c.sh.heads;
    int w0, nw, ns;
    c.rows(t, w0, nw, ns);
    const int n = nw + ns;
    std::vector<double> kv((size_t) n * HD);
    for (int j = 0; j < n; ++j)
        for (int e = 0; e < HD; ++e) {
            double v;
            if (j < nw) v = h_bf16(c.ext[(size_t) (w0 + j) * HD + e]);
            else {
                const uint8_t* rr = c.ckv.data() + (size_t) c.sel[(size_t) t * c.SEL + (j - nw)] * c.crow;
                v = h_e2m1((rr[e / 2] >> (4 * (e & 1))) & 15u) * h_e4m3(rr[HD / 2 + e / 16]);
            }
            kv[(size_t) j * HD + e] = v;
        }
    o.assign((size_t) H * HD, 0.0);
    const double scale = 1.0 / std::sqrt((double) HD);
    std::vector<double> sc((size_t) n);
    for (int h = 0; h < H; ++h) {
        const float* qh = c.q.data() + ((size_t) t * H + h) * HD;
        double m = c.sink[(size_t) h];
        for (int j = 0; j < n; ++j) {
            double d = 0.0;
            for (int e = 0; e < HD; ++e) d += (double) qh[e] * kv[(size_t) j * HD + e];
            sc[(size_t) j] = d * scale;
            m = std::max(m, sc[(size_t) j]);
        }
        double den = std::exp((double) c.sink[(size_t) h] - m);
        for (int j = 0; j < n; ++j) {
            const double p = std::exp(sc[(size_t) j] - m);
            den += p;
            for (int e = 0; e < HD; ++e) o[(size_t) h * HD + e] += p * kv[(size_t) j * HD + e];
        }
        for (int e = 0; e < HD; ++e) o[(size_t) h * HD + e] /= den;
    }
}

struct Err {
    double max = 0.0;
    int t = -1, h = -1, e = -1;
    double a = 0.0, b = 0.0;
    void add(double got, double want, int t_, int h_, int e_) {
        const double r = std::fabs(got - want) / (1.0 + std::fabs(want));
        if (!(r <= max)) {   // (a NaN counts)
            max = std::isnan(r) ? INFINITY : r;
            t = t_, h = h_, e = e_, a = got, b = want;
        }
    }
};

double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int runs = 25;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--qscale") == 0 && i + 1 < argc) g_qscale = (float) std::atof(argv[++i]);
        else {
            std::printf("usage: %s [--bench [--runs N]]\n", argv[0]);
            return 2;
        }
    }
    cudaDeviceProp pr{};
    CU(cudaGetDeviceProperties(&pr, 0));
    std::printf("device: %s (sm_%d%d, %d SMs)\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount);
    cudaStream_t s;
    CU(cudaStreamCreate(&s));
    // (c.A: 256 queries a slab at 64 heads; 2 members' 32 heads: 512 - a slab of 256 here as well)
    const Shape shapes[] = {
        {"p 2K, ratio 1 (selection 512)", 64, 2048, 0, 256, 1},
        {"p 8K, ratio 1", 64, 8192, 0, 256, 1},
        {"p 30K, ratio 1", 64, 30000, 0, 256, 1},
        {"p 30K, ratio 4", 64, 30000, 0, 256, 4},
        {"p 1K, ratio 4 (tiles part full)", 64, 1000, 0, 256, 4},
        {"p 8K, ratio 2, second slab, 100 queries", 64, 8192, 256, 100, 2},
        {"p 2K, ratio 0 (window only)", 64, 2048, 0, 256, 0},
        {"p 0, ratio 1 (the prompt's start)", 64, 0, 0, 256, 1},
        {"p 8K, ratio 1, 32 heads", 32, 8192, 0, 256, 1},
        {"p 8K, ratio 1, 48 heads (packed kernel)", 48, 8192, 0, 256, 1},
        {"p 30K, ratio 1, 8 queries", 64, 30000, 0, 8, 1},
    };
    // the bounds of max |new - old| / (1 + |old|) and of the same against the double reference: mode 1 F32-like (the
    // packed kernel's own distance to the double one is about 1e-5); modes 2 and 3 FP16's 2^-11 relative rounding of P,
    // of q * scale as well (3: the error growing with q's scale, the peaky heads' - up to 1.6e-1 here; its layout is
    // mode 4's, which holds it to mode 1's bound: mode 1's arithmetic in mode 3's 4 slices, this test's build only)
    const double kTol[5] = {0.0, 3e-4, 1e-2, 5e-1, 3e-4};
    const char* const kMode[5] = {"packed", "TC 1 (q, P hi + lo)", "TC 2 (q hi + lo)   ", "TC 3 (plain FP16)  ",
                                  "TC 4 (1, 4 slices) "};
    for (const Shape& sh : shapes) {
        Case c;
        c.sh = sh;
        make_case(c, s);
        const size_t n = (size_t) sh.na * sh.heads * HD;
        float *o_old = nullptr, *o_new = nullptr;
        CU(cudaMalloc(&o_old, n * 4));
        CU(cudaMalloc(&o_new, n * 4));
        CU(cudaMemset(o_old, 0xff, n * 4));
        run(c, 0, o_old, s);
        CU(cudaStreamSynchronize(s));
        CU(cudaGetLastError());
        const std::vector<float> old = host(o_old, n);
        // the reference's queries: the first two, a third and half in, the last two
        std::vector<int> qs = {0, 1, sh.na / 3, sh.na / 2, sh.na - 2, sh.na - 1};
        std::sort(qs.begin(), qs.end());
        qs.erase(std::unique(qs.begin(), qs.end()), qs.end());
        std::vector<std::vector<double>> ref(qs.size());
        for (size_t i = 0; i < qs.size(); ++i) reference(c, qs[i], ref[i]);
        Err e_old;
        for (size_t i = 0; i < qs.size(); ++i)
            for (int h = 0; h < sh.heads; ++h)
                for (int e = 0; e < HD; ++e)
                    e_old.add(old[((size_t) qs[i] * sh.heads + h) * HD + e], ref[i][(size_t) h * HD + e], qs[i], h, e);
        const bool taken = sh.heads % 32 == 0 && sh.na > dk::kAttnScratchT;
        std::printf("%-44s packed kernel vs double %.2e\n", sh.what, e_old.max);
        for (int mode : {1, 2, 3, 4}) {
            CU(cudaMemset(o_new, 0xff, n * 4));
            run(c, mode, o_new, s);
            CU(cudaStreamSynchronize(s));
            CU(cudaGetLastError());
            const std::vector<float> got = host(o_new, n);
            bool ok;
            if (!taken) {   // the packed kernel itself: the same bits
                ok = std::memcmp(got.data(), old.data(), n * 4) == 0;
                std::printf("  %s: not taken, %s\n", kMode[mode], ok ? "the packed kernel's bits" : "DIFFER");
            } else {
                Err e_new, e_ref, e_by[3];   // (by the head's q scale: below 2, 2 .. 8, 8 and up)
                for (int t = 0; t < sh.na; ++t)
                    for (int h = 0; h < sh.heads; ++h) {
                        const float hsc = c.hs[(size_t) h];
                        Err& eb = e_by[hsc < 2.0f ? 0 : hsc < 8.0f ? 1 : 2];
                        for (int e = 0; e < HD; ++e) {
                            const size_t k = ((size_t) t * sh.heads + h) * HD + e;
                            e_new.add(got[k], old[k], t, h, e);
                            eb.add(got[k], old[k], t, h, e);
                        }
                    }
                for (size_t i = 0; i < qs.size(); ++i)
                    for (int h = 0; h < sh.heads; ++h)
                        for (int e = 0; e < HD; ++e)
                            e_ref.add(got[((size_t) qs[i] * sh.heads + h) * HD + e], ref[i][(size_t) h * HD + e], qs[i],
                                      h, e);
                ok = e_new.max <= kTol[mode] && e_ref.max <= kTol[mode];
                std::printf("  %s: %s  vs packed %.2e (worst query %d head %d value %d: %.6g vs %.6g), vs double %.2e\n",
                            kMode[mode], ok ? "ok  " : "FAIL", e_new.max, e_new.t, e_new.h, e_new.e, e_new.a, e_new.b,
                            e_ref.max);
                std::printf("      vs packed by the head's q scale: below 2 %.2e, 2 .. 8 %.2e, 8 .. 16 %.2e (the worst "
                            "head's %.2f)\n",
                            e_by[0].max, e_by[1].max, e_by[2].max, c.hs[(size_t) e_new.h]);
            }
            if (!ok) ++g_fail;
        }
        if (bench) {   // each way's median of runs calls (events around each), after three to warm up
            cudaEvent_t e0, e1;
            CU(cudaEventCreate(&e0));
            CU(cudaEventCreate(&e1));
            double us[4];
            for (int w = 0; w < 4; ++w) {
                for (int i = 0; i < 3; ++i) run(c, w, o_new, s);
                std::vector<float> ms((size_t) runs);
                for (int i = 0; i < runs; ++i) {
                    CU(cudaEventRecord(e0, s));
                    run(c, w, o_new, s);
                    CU(cudaEventRecord(e1, s));
                    CU(cudaEventSynchronize(e1));
                    CU(cudaEventElapsedTime(&ms[(size_t) i], e0, e1));
                }
                us[w] = median(ms) * 1000.0;
            }
            CU(cudaEventDestroy(e0));
            CU(cudaEventDestroy(e1));
            const double nq = sh.na;
            std::printf("  time (median of %d, us a query-layer): packed %.2f, TC 1 %.2f (x%.2f), TC 2 %.2f (x%.2f), "
                        "TC 3 %.2f (x%.2f)\n",
                        runs, us[0] / nq, us[1] / nq, us[0] / us[1], us[2] / nq, us[0] / us[2], us[3] / nq,
                        us[0] / us[3]);
        }
        dk::attn_tc() = 0;
        CU(cudaFree(o_old));
        CU(cudaFree(o_new));
        free_case(c);
    }
    std::printf("%s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    CU(cudaStreamDestroy(s));
    return g_fail == 0 ? 0 : 1;
}
