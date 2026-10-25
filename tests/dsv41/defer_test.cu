// tests/dsv41/defer_test.cu - DS4_EP_DEFER's late add (hc_late_add, src/dsv41/ds4_kernels.cuh) against hc_post of the
// summed output: a layer's FFN hc_post with y = y_dev + y_late, then the next layer's attention hc_post, must equal the
// FFN hc_post with y_dev alone, the attention hc_post, then hc_late_add of y_late with the FFN's post and the
// attention's comb - up to the order of the float sums (the coefficients held fixed: in the engine the attention's
// own coefficients and output differ, the deferral's error, which this does not test).  The head's case (comb null:
// nothing ran since) the same against one hc_post.  A token whose flag is 0 keeps its streams whatever its part row
// holds.  First on the host (the algebra), then the device kernels (hc_post, hc_late_add) against the host's; last
// max_rel_diff, DS4_EP_DEFER_VERIFY's comparison.  Needs a GPU, no model.  defer_test
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
    run_diff();
    std::printf("%d of %d cases failed\n", g_fail, g_cases);
    return g_fail != 0;
}
