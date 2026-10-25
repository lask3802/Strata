// Strata. DS4_LANE_FLOW's batch (strata::ds4::LaneFlow, src/dsv41/ds4_flow.hpp): an entry's h quantized once, after
// every node's gate / up jobs of it and before any of its down jobs, and every output the same as the two batches with
// a barrier between them give.  Synthetic jobs on the engine's own Pool, one node and two (two pools, the first one's
// caller helping), some jobs late on purpose (a straggler's sleep).  No GPU, no model.
//   ds4_flow_test [rounds]
#include "../../src/dsv41/ds4_flow.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace ds4 = strata::ds4;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}
static uint64_t mix(uint64_t a) {
    a ^= a >> 33;
    a *= 0xff51afd7ed558ccdull;
    a ^= a >> 33;
    return a;
}
static void late(std::mt19937& rng, int one_in) {   // (a straggler now and then: a job that takes much longer)
    if ((int) (rng() % (unsigned) one_in) == 0) std::this_thread::sleep_for(std::chrono::microseconds(50 + rng() % 200));
}

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 300;
    std::mt19937 top(12345);
    int bad_h = 0, bad_order = 0, bad_y = 0, bad_once = 0;
    int64_t waits0 = 0;
    for (int NH = 1; NH <= 2; ++NH)
        for (int nw : {0, 1, 3, 7}) {
            std::vector<std::unique_ptr<ds4::Pool>> pools;
            for (int X = 0; X < NH; ++X) pools.push_back(std::make_unique<ds4::Pool>(X == 0 ? nw : std::max(1, nw), std::vector<int>(), 2000));
            ds4::LaneFlow flow;
            for (int r = 0; r < rounds; ++r) {
                const int ne = 1 + (int) (top() % 12), GB = 1 + (int) (top() % 30), DB = 1 + (int) (top() % 80);
                // h[q][X][b]: a gate / up job's output; hs[q]: its h quantized (every part's bands, in order); y[q][X][b]
                std::vector<uint64_t> h((size_t) ne * NH * GB, 0), hs((size_t) ne, 0), y((size_t) ne * NH * DB, 0), yref(y.size());
                std::vector<std::atomic<int>> once((size_t) ne);
                for (auto& o : once) o.store(0);
                std::atomic<int> e_h{0}, e_order{0};
                const uint64_t salt = top();
                auto gu = [&](int X, int q, int b) {
                    thread_local std::mt19937 rng((unsigned) std::hash<std::thread::id>()(std::this_thread::get_id()));
                    late(rng, 40);
                    h[((size_t) q * NH + X) * GB + b] = mix(salt ^ ((uint64_t) q << 40) ^ ((uint64_t) X << 32) ^ (uint64_t) b) | 1;
                };
                auto hq = [&](int q) {
                    thread_local std::mt19937 rng((unsigned) std::hash<std::thread::id>()(std::this_thread::get_id()) + 7);
                    late(rng, 4);   // (often slow: a down job that ran before it would read hs unset)
                    uint64_t s = 0;
                    for (int i = 0; i < NH * GB; ++i) {
                        const uint64_t v = h[(size_t) q * NH * GB + i];
                        if (v == 0) e_h.fetch_add(1);   // (a gate / up job not done)
                        s = mix(s ^ v);
                    }
                    hs[(size_t) q] = s;
                    once[(size_t) q].fetch_add(1);
                };
                auto dn = [&](int X, int q, int b) {
                    thread_local std::mt19937 rng((unsigned) std::hash<std::thread::id>()(std::this_thread::get_id()) + 13);
                    late(rng, 200);
                    if (once[(size_t) q].load() != 1) e_order.fetch_add(1);   // (its h not quantized yet)
                    y[((size_t) q * NH + X) * DB + b] = mix(hs[(size_t) q] ^ ((uint64_t) X << 32) ^ (uint64_t) b);
                };
                flow.run(pools, NH, ne, GB, DB, gu, hq, dn);
                // the reference: the barrier's order, on this thread
                std::vector<uint64_t> hr(h.size()), hsr((size_t) ne);
                for (int q = 0; q < ne; ++q)
                    for (int X = 0; X < NH; ++X)
                        for (int b = 0; b < GB; ++b)
                            hr[((size_t) q * NH + X) * GB + b] = mix(salt ^ ((uint64_t) q << 40) ^ ((uint64_t) X << 32) ^ (uint64_t) b) | 1;
                for (int q = 0; q < ne; ++q) {
                    uint64_t s = 0;
                    for (int i = 0; i < NH * GB; ++i) s = mix(s ^ hr[(size_t) q * NH * GB + i]);
                    hsr[(size_t) q] = s;
                    for (int X = 0; X < NH; ++X)
                        for (int b = 0; b < DB; ++b) yref[((size_t) q * NH + X) * DB + b] = mix(s ^ ((uint64_t) X << 32) ^ (uint64_t) b);
                }
                bad_h += e_h.load();
                bad_order += e_order.load();
                bad_y += y != yref || hs != hsr;
                for (auto& o : once) bad_once += o.load() != 1;
            }
            std::printf("NH %d, %d workers + the caller (a second node's pool: %d): %d rounds, %lld of %lld down jobs "
                        "waited for their entry's h\n", NH, nw, std::max(1, nw), rounds, (long long) flow.waits.load(),
                        (long long) flow.downs.load());
            waits0 += flow.waits.load();
            (void) flow;
        }
    check(bad_h == 0, "every entry's h quantized after all of its gate / up jobs (every node's)");
    check(bad_order == 0, "no down job before its entry's h");
    check(bad_once == 0, "each entry's h quantized exactly once");
    check(bad_y == 0, "every output the same as the barrier's order gives");
    check(waits0 > 0, "some down jobs did wait (the test reached the case it is about)");
    std::printf("%s\n", fails == 0 ? "ALL OK" : "FAILED");
    return fails == 0 ? 0 : 1;
}
