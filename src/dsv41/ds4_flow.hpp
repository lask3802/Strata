// src/dsv41/ds4_flow.hpp - DS4_LANE_FLOW=1: a CPU-lane call's gate / up and down jobs as one batch a node, without the
// barrier between them.  An entry's down jobs need its whole h quantized (every node's gate / up rows of it), so each
// entry counts its gate / up jobs down; the thread that does an entry's last one quantizes its h (hq) and marks it
// ready, and a down job of an entry not ready yet waits for it.  The gate / up jobs come first in each pool's batch (the
// pool hands out its jobs in index order), so every one of them is taken before any down job waits: no wait can block
// the jobs it waits for.  Each output is still written by the same job from the same inputs as with the barrier, so
// the results are the same bits; only the order the jobs run in changes.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "ds4_pool.hpp"

namespace strata::ds4 {

struct LaneFlow {
    // (the entries' counters: one call at a time, as the lane's buffers are)
    std::unique_ptr<std::atomic<int>[]> left, ready;
    int cap = 0;
    std::atomic<int64_t> waits{0}, downs{0}, calls{0};   // down jobs that found their entry not ready, of all; calls

    /// ne entries: on each of NH nodes GB gate / up jobs an entry (gu(X, q, b), b < GB), then DB down jobs an entry
    /// (dn(X, q, b), b < DB); hq(q) once entry q's NH GB gate / up jobs are done, before any of its down jobs.
    /// pools[0]'s caller helps it, as run_nodes does.
    void run(const std::vector<std::unique_ptr<Pool>>& pools, int NH, int ne, int GB, int DB,
             const std::function<void(int, int, int)>& gu, const std::function<void(int)>& hq,
             const std::function<void(int, int, int)>& dn) {
        if (ne <= 0) return;
        if (ne > cap) {
            left.reset(new std::atomic<int>[(size_t) ne]);
            ready.reset(new std::atomic<int>[(size_t) ne]);
            cap = ne;
        }
        for (int q = 0; q < ne; ++q) {
            left[(size_t) q].store(NH * GB, std::memory_order_relaxed);
            ready[(size_t) q].store(0, std::memory_order_relaxed);
        }
        calls.fetch_add(1, std::memory_order_relaxed);
        downs.fetch_add((int64_t) NH * ne * DB, std::memory_order_relaxed);
        const int ng = ne * GB, n = ng + ne * DB;
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                if (j < ng) {
                    const int q = j / GB;
                    gu(X, q, j % GB);
                    if (left[(size_t) q].fetch_sub(1, std::memory_order_acq_rel) == 1) {   // (the entry's last: its h)
                        hq(q);
                        ready[(size_t) q].store(1, std::memory_order_release);
                    }
                    return;
                }
                const int q = (j - ng) / DB;
                if (ready[(size_t) q].load(std::memory_order_acquire) == 0) {
                    waits.fetch_add(1, std::memory_order_relaxed);
                    while (ready[(size_t) q].load(std::memory_order_acquire) == 0) cpu_relax();
                }
                dn(X, q, (j - ng) % DB);
            };
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    }
};

}  // namespace strata::ds4
