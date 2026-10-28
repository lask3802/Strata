// Strata. The prompt flow's expert-stage batches (ds4_flow_batches.hpp): the batch partition flow_layer walks (bmax
// items at most, never across the ring's end; none for an empty layer), DS4_BC_ONCE's up-front list against that walk,
// each batch's compacted bounds written up front against written a batch at a time (zero-row batches included), the
// regions a layer's batches take (one run, or two where they wrap) and the region-reuse invariant over a run of
// layers; DS4_BC_ONCE's text.  No GPU, no model.
#include "../../src/dsv41/ds4_flow_batches.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace ds4 = strata::ds4;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

// flow_layer's loop at ece02640, as it walked a layer: i1 = min(le, i0 + bmax, the ring's next end)
static std::vector<ds4::FlowBatch> walk(size_t lb, size_t le, size_t R, size_t bmax) {
    std::vector<ds4::FlowBatch> v;
    for (size_t i0 = lb; i0 < le;) {
        const size_t i1 = std::min({le, i0 + bmax, (i0 / R + 1) * R});
        v.push_back({i0, i1});
        i0 = i1;
    }
    return v;
}
static bool same(const std::vector<ds4::FlowBatch>& a, const std::vector<ds4::FlowBatch>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].i0 != b[i].i0 || a[i].i1 != b[i].i1) return false;
    return true;
}

int main() {
    // ---- the partition
    check(ds4::flow_batch_end(0, 100, 40, 20) == 20, "end: bmax");
    check(ds4::flow_batch_end(30, 100, 40, 20) == 40, "end: the ring's end before bmax");
    check(ds4::flow_batch_end(40, 100, 40, 20) == 60, "end: from the ring's start");
    check(ds4::flow_batch_end(95, 100, 40, 20) == 100, "end: the layer's end");
    check(ds4::flow_batch_end(79, 81, 40, 20) == 80, "end: the ring's end before the layer's");
    std::vector<ds4::FlowBatch> bs;
    ds4::flow_batches(30, 105, 40, 20, bs);
    check(same(bs, {{30, 40}, {40, 60}, {60, 80}, {80, 100}, {100, 105}}), "a layer across two ring ends");
    ds4::flow_batches(77, 77, 40, 20, bs);
    check(bs.empty(), "an empty layer: no batch");
    ds4::flow_batches(80, 81, 40, 20, bs);
    check(same(bs, {{80, 81}}), "one item at the ring's start");
    ds4::flow_batches(0, 7, 1, 1, bs);
    check(bs.size() == 7 && bs[6].i0 == 6 && bs[6].i1 == 7, "a ring of 1: a batch an item");
    {   // against the walk, and the partition's rules, over many layers
        std::mt19937 rng(7);
        bool eq = true, rules = true;
        for (int t = 0; t < 20000; ++t) {
            const size_t R = 1 + rng() % 96, bmax = std::max<size_t>(1, R / 2), lb = rng() % 5000, le = lb + rng() % 400;
            ds4::flow_batches(lb, le, R, bmax, bs);
            eq = eq && same(bs, walk(lb, le, R, bmax));
            size_t at = lb;
            for (const ds4::FlowBatch& x : bs) {
                rules = rules && x.i0 == at && x.i1 > x.i0 && x.i1 - x.i0 <= bmax && (x.i1 - 1) / R == x.i0 / R;
                at = x.i1;
            }
            rules = rules && at == std::max(lb, le);
        }
        check(eq, "the up-front list is what the loop walks (20000 layers)");
        check(rules, "batches side by side over [lb, le), bmax at most, none across the ring's end");
    }

    // ---- the bounds: up front against a batch at a time, zero-row batches among them
    {
        std::mt19937 rng(11);
        bool eq = true, zero_seen = false, rows_seen = false;
        for (int t = 0; t < 2000; ++t) {
            const int n_slots = (int) (rng() % 8), n_host = 20 + (int) (rng() % 236), n_vs = n_slots + n_host;
            std::vector<int> b((size_t) n_vs + 1, 0);   // a layer's bounds by virtual slot, many slots empty
            for (int s = 0; s < n_vs; ++s) b[(size_t) s + 1] = b[(size_t) s] + (rng() % 3 == 0 ? (int) (rng() % 40) : 0);
            std::vector<int> v;   // the streamed items' virtual slots: ascending, the CPU lane's slots only
            for (int s = n_slots; s < n_vs; ++s)
                if (rng() % 2 == 0) v.push_back(s);
            const size_t R = 4 + rng() % 60, bmax = std::max<size_t>(1, R / 2), lb = rng() % 1000, le = lb + v.size();
            const int bc_n = 256 + 2, bc_w = (int) R + 1;
            const uint64_t n0 = rng() % 100000;
            auto vf = [&](size_t i) { return v[i - lb]; };
            std::vector<int> per((size_t) bc_n * bc_w, -7), up((size_t) bc_n * bc_w, -7);
            uint64_t nb = n0;   // a batch at a time, as the loop wrote them at ece02640
            for (size_t i0 = lb; i0 < le;) {
                const size_t i1 = ds4::flow_batch_end(i0, le, R, bmax);
                int* bc = per.data() + (size_t) (nb++ % (uint64_t) bc_n) * bc_w;
                const int n = (int) (i1 - i0);
                for (int m = 0; m < n; ++m) bc[m] = b[(size_t) v[i0 + (size_t) m - lb]];   // (flow_layer at ece02640)
                bc[n] = b[(size_t) v[i1 - 1 - lb] + 1];
                (bc[n] > bc[0] ? rows_seen : zero_seen) = true;
                i0 = i1;
            }
            ds4::flow_batches(lb, le, R, bmax, bs);   // up front (flow_bc_upfront)
            for (size_t j = 0; j < bs.size(); ++j)
                ds4::flow_batch_bounds(b.data(), bs[j].i0, bs[j].i1, vf,
                                       up.data() + ds4::bc_region(n0 + j, bc_n) * (size_t) bc_w);
            eq = eq && per == up && nb - n0 == bs.size();
        }
        check(eq, "every region's bounds up front as written a batch at a time (2000 layers)");
        check(zero_seen && rows_seen, "zero-row batches (only unpicked likely ones) among them, and batches with rows");
        // one batch by hand: slots 3, 4, 7 (rows 5, 0, 2), the slots between them empty
        const int b[9] = {0, 0, 0, 0, 5, 5, 5, 5, 7};
        const int v3[3] = {3, 4, 7};
        int bc[4] = {-1, -1, -1, -1};
        ds4::flow_batch_bounds(b, 10, 13, [&](size_t i) { return v3[i - 10]; }, bc);
        check(bc[0] == 0 && bc[1] == 5 && bc[2] == 5 && bc[3] == 7, "a batch's bounds by hand");
        const int bz[9] = {0, 0, 0, 0, 4, 4, 4, 4, 4};
        ds4::flow_batch_bounds(bz, 0, 2, [&](size_t i) { return (int) i + 5; }, bc);
        check(bc[0] == 4 && bc[1] == 4 && bc[2] == 4, "a zero-row batch: bc[n] == bc[0]");
    }

    // ---- the regions
    {
        ds4::BcRun r[2];
        check(ds4::bc_runs(5, 0, 258, r) == 0, "no batches: no copy");
        check(ds4::bc_runs(258 * 3 + 5, 10, 258, r) == 1 && r[0].first == 5 && r[0].n == 10, "one run");
        check(ds4::bc_runs(250, 20, 258, r) == 2 && r[0].first == 250 && r[0].n == 8 && r[1].first == 0 && r[1].n == 12,
              "two where the regions wrap");
        check(ds4::bc_runs(248, 10, 258, r) == 1 && r[0].first == 248 && r[0].n == 10, "up to the last region: one run");
        check(ds4::bc_runs(1, 258, 258, r) == 2 && r[0].n == 257 && r[1].first == 0 && r[1].n == 1, "all bc_n of them");
        std::mt19937 rng(3);
        bool cover = true;
        for (int t = 0; t < 20000; ++t) {
            const int bc_n = 1 + (int) (rng() % 300);
            const uint64_t n0 = (uint64_t) rng() * 977u;
            const size_t nb = rng() % (size_t) (bc_n + 1);
            const int nr = ds4::bc_runs(n0, nb, bc_n, r);
            std::vector<size_t> got, want;
            for (int i = 0; i < nr; ++i)
                for (size_t q = 0; q < r[i].n; ++q) got.push_back(r[i].first + q);
            for (size_t j = 0; j < nb; ++j) want.push_back(ds4::bc_region(n0 + j, bc_n));
            cover = cover && got == want && nr <= 2 && (nb == 0) == (nr == 0);
            for (int i = 0; i < nr; ++i) cover = cover && r[i].first + r[i].n <= (size_t) bc_n;
        }
        check(cover, "the runs are the batches' regions in order, within the buffer (20000 layers)");
    }

    // ---- the invariant: over a run of layers, a layer's up-front writes never touch a region a batch of the same
    // layer took (they are only waited apart from the layers before - bc_done), nor more than bc_n
    {
        check(ds4::bc_layer_fits(0, 258) && ds4::bc_layer_fits(258, 258) && !ds4::bc_layer_fits(259, 258) &&
                  !ds4::bc_layer_fits(1, 0),
              "a layer fits: at most bc_n batches");
        std::mt19937 rng(5);
        const int n_exp = 256, bc_n = n_exp + 2;
        uint64_t n_batch = 0;
        std::vector<int64_t> last(bc_n, -1);   // region -> the layer that last wrote it
        bool ok = true;
        for (int64_t layer = 0; layer < 5000; ++layer) {
            const size_t items = rng() % (size_t) (n_exp + 1);   // a layer streams at most its n_exp experts
            const size_t R = 2 + rng() % 80;
            ds4::flow_batches(0, items, R, std::max<size_t>(1, R / 2), bs);
            ok = ok && ds4::bc_layer_fits(bs.size(), bc_n);
            for (size_t j = 0; j < bs.size(); ++j) {
                const size_t reg = ds4::bc_region(n_batch + j, bc_n);
                ok = ok && last[reg] < layer;   // (never a region this layer wrote already)
                last[reg] = layer;
            }
            n_batch += bs.size();
        }
        check(ok, "5000 layers: each batch of a layer its own region, every layer within bc_n");
        std::set<size_t> regs;   // one batch more than bc_n: two batches share a region
        for (size_t j = 0; j < (size_t) bc_n + 1; ++j) regs.insert(ds4::bc_region(17 + j, bc_n));
        check(regs.size() == (size_t) bc_n, "bc_n + 1 batches: a region twice (what bc_layer_fits refuses)");
    }

    // ---- DS4_BC_ONCE's text
    check(ds4::bc_once_mode(nullptr) == 0 && ds4::bc_once_mode("") == 0 && ds4::bc_once_mode("0") == 0, "off");
    check(ds4::bc_once_mode("1") == 1 && ds4::bc_once_mode("2") == 2, "1, 2");
    check(ds4::bc_once_mode("3") == -1 && ds4::bc_once_mode("on") == -1 && ds4::bc_once_mode(" 1") == -1 &&
              ds4::bc_once_mode("01") == -1,
          "anything else: refused");
    std::printf(fails == 0 ? "ALL OK\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
