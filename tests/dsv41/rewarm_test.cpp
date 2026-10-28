// Strata. DS4_REWARM's runner (ds4_rewarm.hpp): one re-warm at a time - a start while one runs stops that one first,
// and the work never runs twice at once; a cancel stops the running one and waits for it (none running: nothing to
// wait for); work that ends by itself leaves nothing running; the destructor cancels.  No GPU, no model.
#include "../../src/dsv41/ds4_rewarm.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace ds4 = strata::ds4;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

struct Counts {
    std::atomic<int> live{0}, max_live{0}, started{0}, stopped{0}, ranges{0};
};
// a re-warm's shape: ranges one after another, the stop flag looked at between them
static ds4::Rewarm::Work work(Counts& n, int ranges) {
    return [&n, ranges](const std::atomic<bool>& stop) {
        const int now = n.live.fetch_add(1) + 1;
        int m = n.max_live.load();
        while (now > m && !n.max_live.compare_exchange_weak(m, now)) {}
        n.started.fetch_add(1);
        for (int i = 0; i < ranges; ++i) {
            if (stop.load(std::memory_order_acquire)) {
                n.stopped.fetch_add(1);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));   // (a range read in)
            n.ranges.fetch_add(1);
        }
        n.live.fetch_sub(1);
    };
}

int main() {
    {
        ds4::Rewarm rw;
        rw.cancel();
        check(!rw.running(), "a cancel with none running: nothing, returns");
    }
    {   // prompts one after another: each prompt's end starts one while the last may still run
        Counts n;
        ds4::Rewarm rw;
        for (int i = 0; i < 6; ++i) {
            rw.start(work(n, 1000000));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            check(rw.running(), "started: running");
        }
        rw.cancel();
        check(n.max_live.load() == 1, "never two at once (6 starts, each while one ran)");
        check(n.started.load() == 6 && n.stopped.load() == 6, "each one stopped by the next start or the cancel");
        check(!rw.running() && n.live.load() == 0, "after the cancel: none running");
    }
    {   // the next prompt's cancel stops it within a range, long before its end
        Counts n;
        ds4::Rewarm rw;
        rw.start(work(n, 1000000));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto t0 = std::chrono::steady_clock::now();
        rw.cancel();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check(ms < 500 && n.stopped.load() == 1 && n.ranges.load() < 1000000, "a cancel stops it between ranges");
    }
    {   // one that ends by itself
        Counts n;
        ds4::Rewarm rw;
        rw.start(work(n, 3));
        for (int i = 0; i < 500 && rw.running(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        check(!rw.running() && n.ranges.load() == 3 && n.stopped.load() == 0, "work that ends: all its ranges, none running");
        rw.start(work(n, 3));   // (a start after one that ended: joins it, runs)
        rw.cancel();
        check(n.started.load() == 2 && n.live.load() == 0, "a start after one that ended");
    }
    {   // the destructor cancels the running one
        Counts n;
        {
            ds4::Rewarm rw;
            rw.start(work(n, 1000000));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        check(n.live.load() == 0 && n.stopped.load() == 1, "the destructor: stopped and waited for");
    }
    std::printf(fails == 0 ? "ALL OK\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
