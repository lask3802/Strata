// tests/dsv41/defer_test.cu - DS4_EP_DEFER's late add (hc_late_add, src/dsv41/ds4_kernels.cuh) against hc_post of the
// summed output: a layer's FFN hc_post with y = y_dev + y_late, then the next layer's attention hc_post, must equal the
// FFN hc_post with y_dev alone, the attention hc_post, then hc_late_add of y_late with the FFN's post and the
// attention's comb - up to the order of the float sums (the coefficients held fixed: in the engine the attention's
// own coefficients and output differ, the deferral's error, which this does not test).  The head's case (comb null:
// nothing ran since) the same against one hc_post.  A token whose flag is 0 keeps its streams whatever its part row
// holds.  First on the host (the algebra), then the device kernels (hc_post, hc_late_add) against the host's; then
// DS4_EP_DEFER_BELOW's split (run_split: the light part late from the mailbox's row 1); last max_rel_diff,
// DS4_EP_DEFER_VERIFY's comparison.  Needs a GPU, no model.  defer_test
#include "../../src/dsv41/ds4_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace sd = strata::ds4;

namespace {

int g_fail = 0, g_cases = 0;
std::mt19937 rng(717);

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }
std::vector<float> rnd(size_t n, float a = 1.0f) {
    std::vector<float> v(n);
    for (float& x : v) x = frand(a);
    return v;
}

// the host's hc_post (ds4_kernels.cuh): out[t][k][e] = post[t][k] y[t][e] + sum_j comb[t][j][k] res[t][j][e]
void post_ref(const std::vector<float>& y, const std::vector<float>& res, const std::vector<float>& post,
              const std::vector<float>& comb, int D, int H, int T, std::vector<float>& out) {
    out.assign((size_t) T * H * D, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < H; ++k)
            for (int e = 0; e < D; ++e) {
                float a = post[(size_t) t * 8 + k] * y[(size_t) t * D + e];
                for (int j = 0; j < H; ++j)
                    a += comb[(size_t) t * 64 + j * H + k] * res[((size_t) t * H + j) * D + e];
                out[((size_t) t * H + k) * D + e] = a;
            }
}
// the host's late add (hc_late_add): part rows ps floats apart, comb empty: the head's (c = post)
void late_ref(std::vector<float>& h, const std::vector<float>& part, size_t ps, const std::vector<int>& flag,
              const std::vector<float>& post, const std::vector<float>& comb, int D, int H, int T) {
    for (int t = 0; t < T; ++t) {
        if (flag[(size_t) t] == 0) continue;
        for (int k = 0; k < H; ++k) {
            float c = 0.0f;
            if (comb.empty()) c = post[(size_t) t * 8 + k];
            else
                for (int j = 0; j < H; ++j) c += comb[(size_t) t * 64 + j * H + k] * post[(size_t) t * 8 + j];
            for (int e = 0; e < D; ++e) h[((size_t) t * H + k) * D + e] += c * part[(size_t) t * ps + e];
        }
    }
}

// max |a - b| / (1 + |a|)
double diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = a.size() == b.size() ? 0.0 : 1e30;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, (double) std::fabs(a[i] - b[i]) / (1.0 + std::fabs(a[i])));
    return m;
}
void check(const char* what, double d, double tol) {
    ++g_cases;
    const bool ok = d <= tol;
    if (!ok) ++g_fail;
    std::printf("%-58s max rel diff %.2e (<= %.0e) -> %s\n", what, d, tol, ok ? "ok" : "FAIL");
}

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, h.size() * sizeof(T)));
    CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
std::vector<float> host(const float* p, size_t n) {
    std::vector<float> h(n);
    CU(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost));
    return h;
}

// one shape: T tokens of D values, H streams; token 1 (when there is one) has no part (flag 0, its row garbage)
void run(int D, int H, int T) {
    const size_t ps = (size_t) D + 40;   // (the mailbox's rows are kMaxDim apart, not D)
    std::vector<float> h0 = rnd((size_t) T * H * D), y_dev = rnd((size_t) T * D), y_att = rnd((size_t) T * D);
    std::vector<float> f_post = rnd((size_t) T * 8, 2.0f), a_post = rnd((size_t) T * 8, 2.0f);
    std::vector<float> f_comb = rnd((size_t) T * 64), a_comb = rnd((size_t) T * 64);   // (not symmetric: j, k apart)
    std::vector<float> part = rnd((size_t) T * ps, 0.5f);
    std::vector<int> flag((size_t) T, 1);
    if (T > 1) flag[1] = 0;
    std::vector<float> y_sum = y_dev;   // (y_dev + the part where the flag is set: the exact path's FFN output)
    for (int t = 0; t < T; ++t)
        if (flag[(size_t) t] != 0)
            for (int e = 0; e < D; ++e) y_sum[(size_t) t * D + e] += part[(size_t) t * ps + e];
    char nm[96];

    // the host: exact (FFN hc_post of the sum, then the attention's) vs deferred (without the part, then the late add)
    std::vector<float> f_ex, a_ex, f_df, a_df, head_df;
    post_ref(y_sum, h0, f_post, f_comb, D, H, T, f_ex);
    post_ref(y_att, f_ex, a_post, a_comb, D, H, T, a_ex);
    post_ref(y_dev, h0, f_post, f_comb, D, H, T, f_df);
    post_ref(y_att, f_df, a_post, a_comb, D, H, T, a_df);
    late_ref(a_df, part, ps, flag, f_post, a_comb, D, H, T);
    std::snprintf(nm, sizeof nm, "host: through the next attention, D %d H %d T %d", D, H, T);
    check(nm, diff(a_ex, a_df), 1e-5);
    head_df = f_df;
    late_ref(head_df, part, ps, flag, f_post, {}, D, H, T);
    std::snprintf(nm, sizeof nm, "host: the head (comb null), D %d H %d T %d", D, H, T);
    check(nm, diff(f_ex, head_df), 1e-5);

    // the device: hc_post + hc_late_add against the host's late add (the part in host-mapped memory, as the mailbox)
    float *d_h0 = dev(h0), *d_ydev = dev(y_dev), *d_yatt = dev(y_att), *d_fp = dev(f_post), *d_ap = dev(a_post),
          *d_fc = dev(f_comb), *d_ac = dev(a_comb);
    int* d_flag = dev(flag);
    float* part_m = nullptr;
    CU(cudaHostAlloc((void**) &part_m, part.size() * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    std::copy(part.begin(), part.end(), part_m);
    float *d_f = nullptr, *d_a = nullptr;
    CU(cudaMalloc(&d_f, h0.size() * 4));
    CU(cudaMalloc(&d_a, h0.size() * 4));
    cudaStream_t s = nullptr;
    CU(cudaStreamCreate(&s));
    sd::hc_post(d_ydev, d_h0, d_fp, d_fc, D, H, T, d_f, s);
    sd::hc_post(d_yatt, d_f, d_ap, d_ac, D, H, T, d_a, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<float> a_dev_pre = host(d_a, h0.size());
    sd::hc_late_add(d_a, part_m, ps, d_flag, d_fp, d_ac, D, H, T, s);
    sd::hc_late_add(d_f, part_m, ps, d_flag, d_fp, nullptr, D, H, T, s);
    CU(cudaStreamSynchronize(s));
    CU(cudaGetLastError());
    std::vector<float> a_ref = a_dev_pre;   // (the late add alone: the device's hc_posts as they came)
    late_ref(a_ref, part, ps, flag, f_post, a_comb, D, H, T);
    std::snprintf(nm, sizeof nm, "device: hc_late_add after the attention, D %d H %d T %d", D, H, T);
    check(nm, diff(a_ref, host(d_a, h0.size())), 1e-6);
    std::snprintf(nm, sizeof nm, "device: deferred vs exact (hc_post of the sum), D %d H %d T %d", D, H, T);
    check(nm, diff(a_ex, host(d_a, h0.size())), 1e-5);
    std::snprintf(nm, sizeof nm, "device: the head (comb null) vs exact, D %d H %d T %d", D, H, T);
    check(nm, diff(f_ex, host(d_f, h0.size())), 1e-5);
    if (T > 1) {   // (the flag-0 token: its streams exactly what the hc_posts left)
        const std::vector<float> got = host(d_a, h0.size());
        bool same = true;
        for (size_t i = (size_t) H * D; i < (size_t) 2 * H * D; ++i) same = same && got[i] == a_dev_pre[i];
        ++g_cases;
        if (!same) ++g_fail;
        std::printf("%-58s %s\n", "device: a token without a part left as it was", same ? "ok" : "FAIL");
    }
    cudaFree(d_h0), cudaFree(d_ydev), cudaFree(d_yatt), cudaFree(d_fp), cudaFree(d_ap), cudaFree(d_fc), cudaFree(d_ac);
    cudaFree(d_flag), cudaFree(d_f), cudaFree(d_a), cudaFreeHost(part_m);
    cudaStreamDestroy(s);
}

// DS4_EP_DEFER_BELOW / TOP (a split layer, T 1): the mailbox's own Resp in host-mapped memory, the heavy part in row 0
// (cpu[0]), the light one in row 1 (cpu[1]).  The layer adds the heavy part through the experts' sum (here: in y), the
// next layer's late_flush the light one after its attention - hc_late_add on &part[1][0] with &cpu[1] - which must equal
// the exact path (both parts in the FFN's hc_post); the head's form (comb null) and DS4_EP_DEFER_VERIFY's in-place add
// on the exact streams (comb null right after the FFN hc_post) likewise.  The wrong row (part[0] / cpu[0], the heavy
// part again: a late_row left at 0) must show; a light row with its flag 0 must leave the streams as they were
void run_split(int D, int H) {
    const int T = 1;
    std::vector<float> h0 = rnd((size_t) H * D), y_dev = rnd((size_t) D), y_att = rnd((size_t) D);
    std::vector<float> f_post = rnd(8, 2.0f), a_post = rnd(8, 2.0f), f_comb = rnd(64), a_comb = rnd(64);
    std::vector<float> heavy = rnd((size_t) D, 0.5f), light = rnd((size_t) D, 0.5f);
    std::vector<float> y_sum = y_dev, y_dh = y_dev;   // (the exact path's FFN output; the split layer's: the heavy part in it)
    for (int e = 0; e < D; ++e) {
        y_sum[(size_t) e] += heavy[(size_t) e] + light[(size_t) e];
        y_dh[(size_t) e] += heavy[(size_t) e];
    }
    std::vector<float> f_ex, a_ex;
    post_ref(y_sum, h0, f_post, f_comb, D, H, T, f_ex);
    post_ref(y_att, f_ex, a_post, a_comb, D, H, T, a_ex);
    sd::Resp* rp = nullptr;
    CU(cudaHostAlloc((void**) &rp, sizeof(sd::Resp), cudaHostAllocMapped | cudaHostAllocPortable));
    std::fill((char*) rp, (char*) rp + sizeof(sd::Resp), (char) 0x7f);   // (garbage past what the host writes)
    std::copy(heavy.begin(), heavy.end(), rp->part[0]);
    std::copy(light.begin(), light.end(), rp->part[1]);
    rp->cpu[0] = rp->cpu[1] = 1;
    float *d_h0 = dev(h0), *d_ydh = dev(y_dh), *d_yatt = dev(y_att), *d_fp = dev(f_post), *d_ap = dev(a_post),
          *d_fc = dev(f_comb), *d_ac = dev(a_comb);
    float *d_f = nullptr, *d_a = nullptr;
    CU(cudaMalloc(&d_f, h0.size() * 4));
    CU(cudaMalloc(&d_a, h0.size() * 4));
    cudaStream_t s = nullptr;
    CU(cudaStreamCreate(&s));
    char nm[96];
    // the layer (heavy in y), the next attention, then late_flush's add of row 1
    auto layer_then_attention = [&]() {
        sd::hc_post(d_ydh, d_h0, d_fp, d_fc, D, H, T, d_f, s);
        sd::hc_post(d_yatt, d_f, d_ap, d_ac, D, H, T, d_a, s);
    };
    layer_then_attention();
    sd::hc_late_add(d_a, &rp->part[1][0], (size_t) sd::kMaxDim, &rp->cpu[1], d_fp, d_ac, D, H, T, s);
    CU(cudaStreamSynchronize(s));
    std::snprintf(nm, sizeof nm, "split: row 1 a layer late vs exact, D %d H %d", D, H);
    check(nm, diff(a_ex, host(d_a, h0.size())), 1e-5);
    // the head's form, and VERIFY's exact streams: row 1 with comb null right after the FFN hc_post
    sd::hc_late_add(d_f, &rp->part[1][0], (size_t) sd::kMaxDim, &rp->cpu[1], d_fp, nullptr, D, H, T, s);
    CU(cudaStreamSynchronize(s));
    std::snprintf(nm, sizeof nm, "split: row 1, comb null (head, VERIFY) vs exact, D %d H %d", D, H);
    check(nm, diff(f_ex, host(d_f, h0.size())), 1e-5);
    // the wrong row: row 0 (the heavy part again) late instead of row 1 - the comparison must see it
    layer_then_attention();
    sd::hc_late_add(d_a, &rp->part[0][0], (size_t) sd::kMaxDim, &rp->cpu[0], d_fp, d_ac, D, H, T, s);
    CU(cudaStreamSynchronize(s));
    {
        const double d = diff(a_ex, host(d_a, h0.size()));
        ++g_cases;
        const bool seen = d > 1e-3;
        if (!seen) ++g_fail;
        std::snprintf(nm, sizeof nm, "split: row 0 late instead (the wrong row) seen, D %d H %d", D, H);
        std::printf("%-58s max rel diff %.2e (> 1e-3) -> %s\n", nm, d, seen ? "ok" : "FAIL");
    }
    // a light row with its flag 0 (no light pick): the streams exactly as the hc_posts left them
    layer_then_attention();
    CU(cudaStreamSynchronize(s));
    const std::vector<float> pre = host(d_a, h0.size());
    rp->cpu[1] = 0;
    sd::hc_late_add(d_a, &rp->part[1][0], (size_t) sd::kMaxDim, &rp->cpu[1], d_fp, d_ac, D, H, T, s);
    CU(cudaStreamSynchronize(s));
    CU(cudaGetLastError());
    {
        const std::vector<float> got = host(d_a, h0.size());
        bool same = true;
        for (size_t i = 0; i < got.size(); ++i) same = same && got[i] == pre[i];
        ++g_cases;
        if (!same) ++g_fail;
        std::snprintf(nm, sizeof nm, "split: light flag 0 leaves the streams, D %d H %d", D, H);
        std::printf("%-58s %s\n", nm, same ? "ok" : "FAIL");
    }
    cudaFree(d_h0), cudaFree(d_ydh), cudaFree(d_yatt), cudaFree(d_fp), cudaFree(d_ap), cudaFree(d_fc), cudaFree(d_ac);
    cudaFree(d_f), cudaFree(d_a), cudaFreeHost(rp);
    cudaStreamDestroy(s);
}

// max_rel_diff (DS4_EP_DEFER_VERIFY's comparison): 0 for the same rows, the largest |a - b| / (1 + |a|) otherwise, kept
// across calls into one host-mapped value; a NaN counts as inf
void run_diff() {
    const size_t n = 5000;
    std::vector<float> a = rnd(n, 4.0f), b = a;
    float *da = dev(a), *db = dev(b), *out = dev(std::vector<float>(1, 0.0f));
    auto got = [&](const std::vector<float>& bb) {
        CU(cudaMemcpy(db, bb.data(), n * 4, cudaMemcpyHostToDevice));
        sd::max_rel_diff(da, db, n, out, nullptr);
        CU(cudaDeviceSynchronize());
        return host(out, 1)[0];
    };
    const float z = got(b);
    b[1234] = a[1234] + 0.01f * (1.0f + std::fabs(a[1234]));   // (relative 1e-2 at one value)
    const float r = got(b);
    std::vector<float> c = a;
    c[7] = a[7] + 1e-4f * (1.0f + std::fabs(a[7]));   // (smaller: the kept value stays r)
    const float r2 = got(c);
    c[9] = NAN;
    const float rn = got(c);
    ++g_cases;
    const bool ok = z == 0.0f && std::fabs(r - 1e-2f) < 1e-5f && r2 == r && std::isinf(rn);
    if (!ok) ++g_fail;
    std::printf("%-58s same %.1e, one at 1e-2 %.3e, then a smaller one %.3e, a NaN %g -> %s\n", "device: max_rel_diff", z,
                r, r2, rn, ok ? "ok" : "FAIL");
    // add_diff: y += a - b (an in-place change of one buffer carried onto another)
    std::vector<float> y = rnd(n), c2 = rnd(n), want(n);
    for (size_t i = 0; i < n; ++i) want[i] = y[i] + (a[i] - c2[i]);
    float *dy = dev(y), *dc = dev(c2);
    sd::add_diff(dy, da, dc, n, nullptr);
    CU(cudaDeviceSynchronize());
    check("device: add_diff", diff(want, host(dy, n)), 0.0);
    cudaFree(da), cudaFree(db), cudaFree(out), cudaFree(dy), cudaFree(dc);
}

// DS4_EP_DEFER_VERIFY's scale (hc_mag, add_abs2, max_mag_diff): a decode token whose residual streams carry a massive
// activation (M at a few values, every stream) that the layer's FFN output cancels - the combine's m + part + shared
// expert, m cancelling the shared expert at others - through the engine's chain: the fork's two hc_posts (exact: the
// sum with the part; deferred: without it), an Engram-like in-place change of the exact streams carried by add_diff, the
// next attention's hc_post, the late add.  |d| / (1 + |exact|) grows with M (the r10 Hvc1 / Hvc2 lift at the last
// layers: ~6e-8 M); |d| / S must stay under float order's bound, 31 roundings of S (1.9e-6), whatever M - and a wiring
// fault (the late add left out, the wrong comb, a split layer's heavy row late instead of its light one) must still be
// far above the engine's bound (1e-5) against S.  comb: positive, doubly stochastic (Sinkhorn's), as hc_split makes it
std::vector<float> sinkhorn(int H) {
    std::vector<float> c(64, 0.0f);
    for (int j = 0; j < H; ++j)
        for (int k = 0; k < H; ++k) c[(size_t) j * H + k] = 0.1f + std::fabs(frand());
    for (int it = 0; it < 30; ++it)
        for (int pass = 0; pass < 2; ++pass)
            for (int a = 0; a < H; ++a) {
                float sum = 0.0f;
                for (int b = 0; b < H; ++b) sum += pass == 0 ? c[(size_t) a * H + b] : c[(size_t) b * H + a];
                for (int b = 0; b < H; ++b) (pass == 0 ? c[(size_t) a * H + b] : c[(size_t) b * H + a]) /= sum;
            }
    return c;
}
struct MagOut {
    double rel = 0, mag = 0, mag_head = 0, miss = 0, wrong = 0, row0 = 0;
};
// split: a split layer (the heavy part in both sides' FFN output, the light one in place on the exact side, late on the
// deferred); engram: the in-place change between the fork and the attention; at_attn: the next attention's output
// cancels the streams, not the FFN's (a split layer's two sides share the FFN hc_post's bits: only that one shows there)
MagOut run_mag_case(int D, int H, float M, bool split, bool engram, bool at_attn) {
    const int T = 1;
    std::vector<float> h0 = rnd((size_t) H * D), ao = rnd((size_t) D);
    std::vector<float> f_post = rnd(8, 1.0f), a_post = rnd(8, 1.0f);
    for (int k = 0; k < H; ++k) {   // (positive, as hc_split's)
        f_post[(size_t) k] = std::fabs(f_post[(size_t) k]) * 1.25f + 0.5f;
        a_post[(size_t) k] = std::fabs(a_post[(size_t) k]) + 0.5f;
    }
    std::vector<float> f_comb = sinkhorn(H), a_comb = sinkhorn(H);
    // x with sum_j comb[j][k] x[j] = rhs[k] (Gauss-Jordan, partial pivoting): streams M x that an output of -M cancels
    // in every stream of the hc_post with that comb and post
    auto solve = [H](const std::vector<float>& comb, const float* rhs, float* x) {
        double A[8][9];
        for (int k = 0; k < H; ++k) {
            for (int j = 0; j < H; ++j) A[k][j] = comb[(size_t) j * H + k];
            A[k][H] = rhs[k];
        }
        for (int c = 0; c < H; ++c) {
            int pv = c;
            for (int r = c + 1; r < H; ++r)
                if (std::fabs(A[r][c]) > std::fabs(A[pv][c])) pv = r;
            for (int q = 0; q <= H; ++q) std::swap(A[c][q], A[pv][q]);
            for (int r = 0; r < H; ++r) {
                if (r == c) continue;
                const double f = A[r][c] / A[c][c];
                for (int q = 0; q <= H; ++q) A[r][q] -= f * A[c][q];
            }
        }
        for (int j = 0; j < H; ++j) x[j] = (float) (A[j][H] / A[j][j]);
    };
    float xs[8], zs[8];
    if (at_attn) {   // (the FFN's streams M z, z for the attention's comb and post: h0 = M w, w for the FFN's comb and z)
        solve(a_comb, a_post.data(), zs);
        solve(f_comb, zs, xs);
    } else {
        solve(f_comb, f_post.data(), xs);
    }
    std::vector<float> sh = rnd((size_t) D), heavy = rnd((size_t) D, 0.5f), light = rnd((size_t) D, 0.5f);
    std::vector<float> m((size_t) D);   // (the combine's routed sum, m + part + sh its output)
    for (int e = 0; e < D; ++e) m[(size_t) e] = frand();
    for (int q = 0; q < 16; ++q) {   // (16 massive values, M x[j] in stream j; the FFN output -M there, or the attention's)
        const int e = (q * 257 + 11) % D;
        for (int j = 0; j < H; ++j) h0[(size_t) j * D + e] = M * xs[j];
        if (at_attn) ao[(size_t) e] = -M;
        else m[(size_t) e] = -M - sh[(size_t) e] - heavy[(size_t) e] - light[(size_t) e];
    }
    for (int q = 0; q < 16; ++q) {   // (16 more where the routed sum cancels the shared expert's M)
        const int e = (q * 263 + 101) % D;
        sh[(size_t) e] = M;
        m[(size_t) e] = -M + frand();
    }
    // the combine's two outputs (ds4_gguf.cu combine: m, + the part when flagged, + sh - each add rounded): the exact
    // side's FFN output with the part (a split layer's: the heavy one alone - both sides take it), the deferred side's
    std::vector<float> y_ex((size_t) D), y_df((size_t) D);
    const std::vector<float>& p_in = split ? heavy : light;   // (the part inside the exact side's combine)
    for (int e = 0; e < D; ++e) {
        volatile float a = m[(size_t) e] + p_in[(size_t) e];
        y_ex[(size_t) e] = a + sh[(size_t) e];
        volatile float b = m[(size_t) e] + sh[(size_t) e];
        y_df[(size_t) e] = split ? y_ex[(size_t) e] : (float) b;
    }
    sd::Resp* rp = nullptr;   // (the mailbox: the heavy part in row 0, the light one in row 1; unsplit: row 0 the part)
    CU(cudaHostAlloc((void**) &rp, sizeof(sd::Resp), cudaHostAllocMapped | cudaHostAllocPortable));
    std::fill((char*) rp, (char*) rp + sizeof(sd::Resp), (char) 0x7f);
    std::copy((split ? heavy : light).begin(), (split ? heavy : light).end(), rp->part[0]);
    std::copy(light.begin(), light.end(), rp->part[1]);
    rp->cpu[0] = rp->cpu[1] = 1;
    const int lr = split ? 1 : 0;   // (late_row)
    float *d_h0 = dev(h0), *d_ao = dev(ao), *d_yex = dev(y_ex), *d_ydf = dev(y_df), *d_sh = dev(sh), *d_fp = dev(f_post),
          *d_ap = dev(a_post), *d_fc = dev(f_comb), *d_ac = dev(a_comb);
    const size_t nh = (size_t) H * D;
    float *d_e1 = nullptr, *d_v1 = nullptr, *d_a = nullptr, *d_x = nullptr, *d_y = nullptr, *d_t = nullptr;
    for (float** p : {&d_e1, &d_v1, &d_a, &d_x, &d_y, &d_t}) CU(cudaMalloc(p, nh * 4));
    float* d_out = dev(std::vector<float>(6, 0.0f));
    cudaStream_t s = nullptr;
    CU(cudaStreamCreate(&s));
    // the fork (run_layer's FFN): the deferred streams, the scale, the exact streams (+ a split layer's light part in place)
    sd::hc_post(d_ydf, d_h0, d_fp, d_fc, D, H, T, d_v1, s);
    sd::hc_mag(d_yex, split ? nullptr : d_ydf, d_sh, &rp->part[lr][0], (size_t) sd::kMaxDim, &rp->cpu[lr], d_h0, d_fp,
               d_fc, D, H, T, d_a, s);
    sd::hc_post(d_yex, d_h0, d_fp, d_fc, D, H, T, d_e1, s);
    if (split) sd::hc_late_add(d_e1, &rp->part[1][0], (size_t) sd::kMaxDim, &rp->cpu[1], d_fp, nullptr, D, H, T, s);
    CU(cudaStreamSynchronize(s));
    MagOut o;
    {   // the head's form: the deferred streams + the part (comb null) against the exact ones, S the fork's scale
        CU(cudaMemcpyAsync(d_y, d_v1, nh * 4, cudaMemcpyDeviceToDevice, s));
        sd::hc_late_add(d_y, &rp->part[lr][0], (size_t) sd::kMaxDim, &rp->cpu[lr], d_fp, nullptr, D, H, T, s);
        sd::max_mag_diff(d_e1, d_y, d_a, d_ao, d_ap, nullptr, D, H, T, d_out + 1, s);
    }
    if (engram) {   // (an in-place change of the exact streams, carried onto the deferred ones: run_layer's Engram)
        std::vector<float> e1 = host(d_e1, nh), e1b = e1;
        for (size_t i = 0; i < nh; ++i) e1b[i] = e1[i] * 0.75f + 0.3f * frand();
        CU(cudaMemcpy(d_t, e1.data(), nh * 4, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_e1, e1b.data(), nh * 4, cudaMemcpyHostToDevice));
        sd::add_diff(d_v1, d_e1, d_t, nh, s);
        sd::add_abs2(d_a, d_e1, d_t, nh, s);
    }
    // the next attention's hc_post on both, late_flush's add; the comparison both ways
    auto deferred = [&](const float* comb, const float* part, const int* flag, bool add) {
        sd::hc_post(d_ao, d_v1, d_ap, d_ac, D, H, T, d_y, s);
        if (add) sd::hc_late_add(d_y, part, (size_t) sd::kMaxDim, flag, d_fp, comb, D, H, T, s);
    };
    auto measure = [&](int at) {
        CU(cudaMemsetAsync(d_out + at, 0, 4, s));
        sd::max_mag_diff(d_x, d_y, d_a, d_ao, d_ap, d_ac, D, H, T, d_out + at, s);
        CU(cudaStreamSynchronize(s));
        return (double) host(d_out + at, 1)[0];
    };
    sd::hc_post(d_ao, d_e1, d_ap, d_ac, D, H, T, d_x, s);
    deferred(d_ac, &rp->part[lr][0], &rp->cpu[lr], true);
    sd::max_rel_diff(d_x, d_y, nh, d_out, s);
    o.mag = measure(2);
    o.rel = host(d_out, 1)[0];
    o.mag_head = host(d_out + 1, 1)[0];
    deferred(d_ac, &rp->part[lr][0], &rp->cpu[lr], false);   // (the late add left out)
    o.miss = measure(3);
    deferred(d_fc, &rp->part[lr][0], &rp->cpu[lr], true);   // (the FFN's comb: VERIFY=2's wrong one)
    o.wrong = measure(4);
    if (split) {
        deferred(d_ac, &rp->part[0][0], &rp->cpu[0], true);   // (the heavy row late instead of the light one)
        o.row0 = measure(5);
    }
    CU(cudaGetLastError());
    cudaFree(d_h0), cudaFree(d_ao), cudaFree(d_yex), cudaFree(d_ydf), cudaFree(d_sh), cudaFree(d_fp), cudaFree(d_ap);
    cudaFree(d_fc), cudaFree(d_ac), cudaFree(d_e1), cudaFree(d_v1), cudaFree(d_a), cudaFree(d_x), cudaFree(d_y);
    cudaFree(d_t), cudaFree(d_out), cudaFreeHost(rp);
    cudaStreamDestroy(s);
    return o;
}
void run_mag() {
    const double fl_bound = 31.0 * std::ldexp(1.0, -24), eng = 1e-5;   // (float order's, the engine's)
    for (int q = 0; q < 16; ++q) {   // (attn cancels or FFN, split or whole, engram or not, 4 sizes)
        const int at = q >> 3, sp = (q >> 2) & 1, eg = (q >> 1) & 1;
        for (float M : (q & 1) ? std::vector<float>{6000.0f, 30000.0f} : std::vector<float>{1.0f, 1000.0f}) {
            const MagOut o = run_mag_case(4096, 4, M, sp != 0, eg != 0, at != 0);
            char nm[96];
            std::snprintf(nm, sizeof nm, "scale: %s%s, %s, M %g", sp ? "split" : "whole", eg ? " + engram" : "",
                          at ? "attn cancels" : "FFN cancels", (double) M);
            const bool ok = o.mag <= fl_bound && o.mag_head <= fl_bound && o.miss > 100 * eng && o.wrong > 100 * eng &&
                            (sp == 0 || o.row0 > 100 * eng);
            ++g_cases;
            if (!ok) ++g_fail;
            std::printf("%-46s |d|/(1+|x|) %.1e; |d|/S %.1e, head %.1e (<= %.1e); faults |d|/S: no add %.1e, wrong "
                        "comb %.1e, row 0 %.1e (> %.0e) -> %s\n", nm, o.rel, o.mag, o.mag_head, fl_bound, o.miss,
                        o.wrong, o.row0, 100 * eng, ok ? "ok" : "FAIL");
            if (M >= 6000.0f && eg == 0 && (sp == 0 || at != 0)) {   // (the old measure's lift: reproduced)
                ++g_cases;
                const bool lift = o.rel > eng;
                if (!lift) ++g_fail;
                std::printf("%-46s |d|/(1+|x|) %.1e above %.0e (the false MISMATCH reproduced) -> %s\n", nm, o.rel, eng,
                            lift ? "ok" : "FAIL");
            }
        }
    }
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no GPU\n");
        return 1;
    }
    run(64, 4, 1);
    run(4096, 4, 1);   // (a model-sized row, 4 streams, a decode token)
    run(300, 4, 4);
    run(96, 2, 3);
    run_split(64, 4);
    run_split(4096, 4);
    run_diff();
    run_mag();
    std::printf("%d of %d cases failed\n", g_fail, g_cases);
    return g_fail != 0;
}
