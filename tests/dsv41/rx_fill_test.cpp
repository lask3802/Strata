// Strata. The remote experts' fills (src/dsv41/ds4_rx_fill.hpp, EP U5): which CPU-lane expert goes to the expert
// server in place of which held one, the budget a fill leaves under (R a token, Q pending, the bulk pace, the idle
// pause), the server's list's states, and the RAM tier's rule that it never takes a server's expert.  No GPU, no
// socket.  Heat as the engine's (profile-scaled counts), the rule's margin 1.25 and bonus 1 (DS4_CACHE_MARGIN / _BONUS).
#include "../../src/dsv41/ds4_rx.hpp"
#include "../../src/dsv41/ds4_rx_fill.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace rx = strata::ds4::rx;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static const float M = 1.25f, B = 1.0f;

// one layer's pick with no RAM tier and no zero-copy store unless given, and no VRAM swap under way unless vout
static rx::Fill pick(const std::vector<int>& hits, const int* slot, const float* heat, uint8_t* has, int n, int vin,
                     const int* ram = nullptr, const uint8_t* zc = nullptr, int vout = -1) {
    rx::TierLayer t;
    t.hits = &hits;
    t.slot = slot;
    t.ram = ram;
    t.zc = zc;
    t.heat = heat;
    t.has = has;
    t.n_exp = n;
    t.vin = vin;
    t.vout = vout;
    return rx::remote_pick(t, M, B);
}

int main() {
    {
        // the rule: in's heat past the coldest held one's x 1.25 + 1
        const int n = 6;
        std::vector<int> slot(n, -1);
        std::vector<float> heat = {8.0f, 3.0f, 20.0f, 0.0f, 11.0f, 11.1f};
        std::vector<uint8_t> has = {rx::kHeld, rx::kHeld, 0, 0, 0, 0};   // (held: 0 (8), 1 (3): the coldest is 1)
        // 3 x 1.25 + 1 = 4.75
        heat[4] = 4.75f;
        rx::Fill f = pick({4}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in < 0, "in's heat equal to the coldest's x 1.25 + 1 (4.75): no fill");
        heat[4] = 4.76f;
        f = pick({4}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in == 4 && f.out == 1, "past it (4.76): in 4, out the coldest held (1)");
        heat[4] = 4.0f;   // (past 3 x 1.25 = 3.75, not past 4.75: the bonus decides)
        f = pick({4}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in < 0, "past the margin alone (4.0 > 3.75), not the bonus: no fill");
        f = pick({4, 2, 5}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in == 2 && f.out == 1, "several hits: the hottest (2, heat 20) goes in");
        f = pick({0, 1}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in < 0, "hits the server holds already: no fill");
        std::vector<uint8_t> none(n, 0);
        f = pick({2}, slot.data(), heat.data(), none.data(), n, -1);
        check(f.in < 0, "the server holds nothing in the layer: nothing to swap out");
        slot[2] = 0;
        f = pick({2}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in < 0, "a hit in this GPU's VRAM: not a candidate");
        slot[2] = -1;
        f = pick({2}, slot.data(), heat.data(), has.data(), n, 2);
        check(f.in < 0, "the VRAM tier's pick this tick (vin): not a candidate");
        f = pick({-1, 9}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in < 0, "hit ids outside the layer: ignored");
    }
    {
        // a server-held expert also in this GPU's VRAM leaves first, whatever its heat
        const int n = 5;
        std::vector<int> slot = {3, -1, -1, -1, -1};          // (0 in VRAM)
        std::vector<float> heat = {50.0f, 2.0f, 9.0f, 1.0f, 1.5f};
        std::vector<uint8_t> has = {rx::kHeld, rx::kHeld, rx::kHeld, 0, 0};
        rx::Fill f = pick({3}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in == 3 && f.out == 0, "a duplicate (0, hot 50) leaves before the coldest held (1), and a cold in (1.0) beats it");
        slot[2] = 1;
        f = pick({3}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in == 3 && f.out == 2, "two duplicates: the colder one (2, 9) leaves");
        f = pick({3}, slot.data(), heat.data(), has.data(), n, -1, nullptr, nullptr, 2);
        check(f.in == 3 && f.out == 0, "the colder one (2) is the VRAM tier's pending swap-out: not a duplicate, 0 leaves");
        slot[2] = -1;
        heat[3] = 4.0f;
        f = pick({3}, slot.data(), heat.data(), has.data(), n, -1, nullptr, nullptr, 0);
        check(f.in == 3 && f.out == 1,
              "the only duplicate (0) leaving VRAM this round: no duplicate; the coldest held (1, 2) leaves by the rule");
        has[0] = rx::kFilling;
        f = pick({3}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in == 3 && f.out == 1, "a filling expert in VRAM (0) is no duplicate to evict: the coldest held (1) leaves");
    }
    {
        // in: only an expert the main computes on its CPU lane (not the RAM tier's, not the zero-copy store's)
        const int n = 4;
        std::vector<int> slot(n, -1), ram = {-1, 0, -1, -1};
        std::vector<uint8_t> zc = {0, 0, 1, 0};
        std::vector<float> heat = {1.0f, 30.0f, 20.0f, 10.0f};
        std::vector<uint8_t> has = {rx::kHeld, 0, 0, 0};
        rx::Fill f = pick({1, 2, 3}, slot.data(), heat.data(), has.data(), n, -1, ram.data(), zc.data());
        check(f.in == 3 && f.out == 0, "1 in the RAM tier, 2 in the zero-copy store: in is 3");
        f = pick({1, 2, 3}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in == 1, "no RAM tier, no store: in is the hottest, 1");
    }
    {
        // the filling state: neither in nor out, and no cache takes it
        const int n = 4;
        std::vector<int> slot(n, -1), ram(n, -1);
        std::vector<float> heat = {1.0f, 30.0f, 40.0f, 5.0f};
        std::vector<uint8_t> has = {rx::kFilling, rx::kHeld, rx::kFilling, 0};
        rx::Fill f = pick({2, 3}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in < 0, "a filling hit (2, heat 40) is not a candidate; the other (3, 5) is not past the held 1 (30)");
        heat[1] = 2.0f;
        f = pick({3}, slot.data(), heat.data(), has.data(), n, -1);
        check(f.in == 3 && f.out == 1, "a filling one (0, heat 1) never leaves: the coldest held (1) does");
        check(rx::cache_in({2, 3}, slot.data(), heat.data(), has.data(), nullptr) == 3,
              "cache_in: a filling expert is the server's (not into this GPU's VRAM)");
        check(rx::ram_in({2, 3}, slot.data(), ram.data(), heat.data(), has.data(), -1) == 3,
              "ram_in: a filling expert is the server's (not into the RAM tier)");
    }
    {
        // the RAM tier's next expert
        const int n = 6;
        std::vector<int> slot = {-1, 0, -1, -1, -1, -1}, ram = {-1, -1, 2, -1, -1, -1};
        std::vector<float> heat = {5.0f, 90.0f, 80.0f, 70.0f, 60.0f, 4.0f};
        std::vector<uint8_t> has = {0, 0, 0, rx::kHeld, 0, 0};
        const std::vector<int> hits = {0, 1, 2, 3, 4, 5};
        check(rx::ram_in(hits, slot.data(), ram.data(), heat.data(), has.data(), -1) == 4,
              "the hottest not in VRAM (1), not in RAM (2), not the server's (3): 4");
        check(rx::ram_in(hits, slot.data(), ram.data(), heat.data(), nullptr, -1) == 3,
              "no server's list (a window it does not serve): its expert 3 may come in");
        check(rx::ram_in(hits, slot.data(), ram.data(), heat.data(), has.data(), 4) == 0,
              "the VRAM tier's pick this tick (4) left out: 0");
        check(rx::ram_in({1, 2, 3}, slot.data(), ram.data(), heat.data(), has.data(), -1) < 0, "nothing left: -1");
    }
    {
        // the book: R a token, Q pending, the pace's bucket, the idle pause
        rx::FillBook b(2, 3, 0.0, 0.0, 8.6e6);
        b.tick(0.0, false);
        check(b.can(8.6e6), "no pace: a fill may leave");
        b.take(8.6e6);
        b.take(8.6e6);
        check(!b.can(8.6e6) && b.spent(), "R=2: two a token, then none");
        b.tick(10.0, false);
        b.take(8.6e6);
        check(!b.can(8.6e6) && b.pending() == 3, "Q=3: the third pending, then none (R left 1)");
        b.done();
        check(b.can(8.6e6), "one landed: one may leave again");
        b.tick(20.0, true);
        check(!b.can(8.6e6) && b.spent(), "a tick while the link idles: none");
        b.tick(30.0, false);
        check(b.can(8.6e6), "the spell over: they may again");
        b.done();
        b.done();
        b.done();
        b.done();
        check(b.pending() == 0, "done past zero stays at 0");

        rx::FillBook p(8, 8, 0.6e9, 2 * 8.6e6, 8.6e6);   // (0.6 GB/s, room for two fills of 8.6 MB)
        p.tick(0.0, false);
        check(!p.can(8.6e6), "the pace's bucket starts empty");
        p.tick(10.0, false);   // (10 ms: 6 MB)
        check(near(p.bucket(), 6.0e6) && !p.can(8.6e6) && p.can(5.0e6), "10 ms at 0.6 GB/s: 6 MB, not a fill of 8.6 MB");
        p.tick(20.0, false);
        check(near(p.bucket(), 12.0e6) && p.can(8.6e6), "20 ms: 12 MB, one fill");
        p.take(8.6e6);
        check(near(p.bucket(), 3.4e6) && !p.can(8.6e6), "taken: 3.4 MB left");
        p.tick(1000.0, false);
        check(near(p.bucket(), 17.2e6), "a long quiet: the bucket holds two fills at most (burst)");
        check(!p.spent(), "spent: only R, Q or idle, not the bucket");
        p.tick(999.0, false);
        check(near(p.bucket(), 17.2e6), "a clock that went back adds nothing");

        rx::FillBook e(8, 8, 0.86e9, 2 * 8.6e6, 8.6e6);
        e.tick(0.0, false);
        e.tick(10.0, false);   // (10 ms at 0.86 GB/s: 8.6 MB exactly)
        check(near(e.bucket(), 8.6e6) && e.can(8.6e6), "the bucket holding exactly one fill's bytes: it may leave");
        rx::FillBook c(8, 8, 1.0e9, 1.0e6, 8.6e6);
        check(c.clamped() && near(c.burst(), 8.6e6), "a burst (1 MB) below one fill (8.6 MB): raised to the fill");
        c.tick(0.0, false);
        c.tick(100.0, false);
        check(near(c.bucket(), 8.6e6) && c.can(8.6e6), "so a fill of that size can leave after all");
        rx::FillBook z(8, 8, 1.0e9, 0.0, 8.6e6);
        check(z.clamped() && near(z.burst(), 8.6e6), "a burst of 0: raised too");
        rx::FillBook u(8, 8, 0.0, 0.0, 8.6e6), w(8, 8, 1.0e9, 2.0e7, 8.6e6);
        check(!u.clamped() && !w.clamped() && near(w.burst(), 2.0e7), "no pace, or a burst above one fill: as given");
    }
    {
        // the layers' order on equal hits, and one fill a layer at most
        const int n = 4;
        std::vector<int> s0(n, -1), s1(n, -1);
        std::vector<float> h = {1.0f, 2.0f, 10.0f, 12.0f};
        std::vector<uint8_t> a0 = {rx::kHeld, rx::kHeld, 0, 0}, a1 = {rx::kHeld, rx::kHeld, 0, 0};
        std::vector<int> k = {2, 3};
        std::vector<rx::TierLayer> ls(2);
        for (int i = 0; i < 2; ++i) {
            ls[(size_t) i].layer = 20 + i;
            ls[(size_t) i].hits = &k;
            ls[(size_t) i].slot = (i == 0 ? s0 : s1).data();
            ls[(size_t) i].heat = h.data();
            ls[(size_t) i].has = (i == 0 ? a0 : a1).data();
            ls[(size_t) i].n_exp = n;
            ls[(size_t) i].bytes = 8.6e6;
        }
        rx::FillBook one(1, 4, 0.0, 0.0, 8.6e6);
        one.tick(0.0, false);
        std::vector<rx::Fill> f = rx::remote_choose(ls, one, M, B);
        check(f.size() == 1 && f[0].layer == 20, "equal hits (2 each), R=1: the lower layer (20) first");
        rx::FillBook four(4, 4, 0.0, 0.0, 8.6e6);
        four.tick(0.0, false);
        std::vector<rx::TierLayer> only(ls.begin() + 1, ls.end());
        f = rx::remote_choose(only, four, M, B);
        check(f.size() == 1 && f[0].in == 3 && f[0].out == 0 && a1[2] == 0 && a1[1] == rx::kHeld,
              "two held and two ins in one layer, R=4: one fill (3 for 0); the other pair waits for the next tick");
    }
    {
        // a window prompt's settle, once a prompt (rx::PromptSettle), over the feeds the engine sees
        auto run = [](rx::PromptSettle& ps, bool rx_on, const std::vector<int>& feeds) {
            std::vector<int> at;   // (the feeds that settled; feeds: a window feed's logits rows; -1: a chunked run,
                                   // which settles; -2: a decode window)
            for (int i = 0; i < (int)feeds.size(); ++i) {
                const int lr = feeds[i];
                if (lr == -1) {
                    ps.settled();
                    at.push_back(i);
                } else if (lr == -2) {
                    ps.decoded();
                } else if (ps.after_feed(rx_on, lr)) {
                    ps.settled();
                    at.push_back(i);
                }
            }
            return at;
        };
        using V = std::vector<int>;
        rx::PromptSettle ps;
        std::vector<int> unit16(256, 0);   // (--prefill-chunk 0: a 4K prompt in 16-token feeds, the last with logits)
        unit16.back() = 1;
        check(run(ps, true, unit16) == V{255}, "a 4K prompt in 256 feeds of 16 tokens: one settle, at the last");
        check(run(ps, true, {-2, 0, 1}) == V{2}, "a decode, then feed_mark's windows (n - 1 rows, then the last): one settle, at the last");
        check(run(ps, true, {-2, -1, 1}) == V{1}, "a decode, then a chunked run and its last window: the chunked run's settle only");
        check(run(ps, true, {-2, 1, -2, 1}) == (V{1, 3}), "two prompts with a decode between: two");
        check(run(ps, true, {-2, 4, 4, 4}) == V{1}, "scoring windows (logits each) after a decode: one, at the first");
        rx::PromptSettle off;
        check(run(off, false, {1, -2, 1}).empty(), "no expert server: none");
    }
    {
        // a tick over the layers: the most hits first, one a layer, the list's states set at once
        const int n = 4;
        std::vector<int> s0(n, -1), s1(n, -1), s2(n, -1);
        std::vector<float> h0 = {1.0f, 10.0f, 0.0f, 0.0f}, h1 = {1.0f, 10.0f, 9.0f, 0.0f}, h2 = {1.0f, 10.0f, 9.0f, 8.0f};
        std::vector<uint8_t> a0 = {rx::kHeld, 0, 0, 0}, a1 = {rx::kHeld, 0, 0, 0}, a2 = {rx::kHeld, 0, 0, 0};
        std::vector<int> k0 = {1}, k1 = {1, 2}, k2 = {1, 2, 3};
        auto layers = [&]() {
            std::vector<rx::TierLayer> v(3);
            std::vector<int>* k[3] = {&k0, &k1, &k2};
            std::vector<int>* s[3] = {&s0, &s1, &s2};
            std::vector<float>* h[3] = {&h0, &h1, &h2};
            std::vector<uint8_t>* a[3] = {&a0, &a1, &a2};
            for (int i = 0; i < 3; ++i) {
                v[(size_t) i].layer = 10 + i;
                v[(size_t) i].hits = k[i];
                v[(size_t) i].slot = s[i]->data();
                v[(size_t) i].heat = h[i]->data();
                v[(size_t) i].has = a[i]->data();
                v[(size_t) i].n_exp = n;
                v[(size_t) i].bytes = 8.6e6;
            }
            return v;
        };
        rx::FillBook b(2, 2, 0.0, 0.0, 8.6e6);
        b.tick(0.0, false);
        auto ls = layers();
        std::vector<rx::Fill> f = rx::remote_choose(ls, b, M, B);
        check(f.size() == 2 && f[0].layer == 12 && f[1].layer == 11, "R=2: the two layers with the most hits (12: 3, 11: 2)");
        check(f.size() == 2 && f[0].in == 1 && f[0].out == 0, "layer 12: in 1 (the hottest hit), out 0");
        check(a2[0] == 0 && a2[1] == rx::kFilling && a1[0] == 0 && a1[1] == rx::kFilling && a0[0] == rx::kHeld,
              "out off the list (0) and in filling at once; the third layer untouched");
        b.tick(1.0, false);
        ls = layers();
        f = rx::remote_choose(ls, b, M, B);
        check(f.empty(), "Q=2 pending: no fill the next token");
        check(rx::fill_landed(a2.data(), 1, true) && a2[1] == rx::kHeld, "FillOk: in held");
        check(rx::fill_landed(a1.data(), 1, false) && a1[1] == 0, "a refused fill: in back to the lane (0)");
        check(!rx::fill_landed(a1.data(), 1, true) && a1[1] == 0, "a FillOk for an expert not filling: refused, unchanged");
        b.done();
        b.done();
        b.tick(2.0, false);
        ls = layers();
        f = rx::remote_choose(ls, b, M, B);
        check(f.size() == 1 && f[0].layer == 10 && f[0].in == 1 && f[0].out == 0,
              "layer 12's next (2, heat 9) not past its held 1 (10), 11 holds nothing: layer 10's fill");
        rx::FillBook p(8, 8, 1.0e9, 9.0e6, 8.6e6);
        p.tick(0.0, false);
        p.tick(9.0, false);   // (9 MB: one fill)
        a0 = a1 = a2 = {rx::kHeld, 0, 0, 0};
        ls = layers();
        f = rx::remote_choose(ls, p, M, B);
        check(f.size() == 1 && f[0].layer == 12, "the pace: one fill's bytes in the bucket, one fill");
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
