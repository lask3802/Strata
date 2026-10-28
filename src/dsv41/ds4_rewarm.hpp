// src/dsv41/ds4_rewarm.hpp - DS4_REWARM=1: once a layer-major prompt is done, the warm set (DS4_IQ_WARM=split's ranges)
// read in again on a thread of its own, the pages the prompt cost the page cache brought back for the decode after it.
// One at a time: a start cancels the one still running and waits for it to stop first; the next prompt cancels it at
// its start (Model::feed / feed_mark), so its reads never compete with a prompt's.  The work is given the stop flag
// and checks it between ranges (a range is a few MB: a cancel waits for one).  It reads pages only: the model's
// state and sums never depend on it (only timings do - and pick_path, which learns from them, may choose otherwise).
// No GPU, no model (tests/dsv41/rewarm_test.cpp).
#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include <utility>

namespace strata::ds4 {

class Rewarm {
public:
    using Work = std::function<void(const std::atomic<bool>& stop)>;
    Rewarm() = default;
    Rewarm(const Rewarm&) = delete;
    Rewarm& operator=(const Rewarm&) = delete;
    ~Rewarm() { cancel(); }
    /// work on a thread of its own, after the one still running (if any) was told to stop and has
    void start(Work work) {
        cancel();
        stop_.store(false, std::memory_order_release);
        running_.store(true, std::memory_order_release);
        th_ = std::thread([this, w = std::move(work)] {
            w(stop_);
            running_.store(false, std::memory_order_release);
        });
    }
    /// the running one told to stop, and waited for (none: nothing); the caller's thread only, as start
    void cancel() {
        stop_.store(true, std::memory_order_release);
        if (th_.joinable()) th_.join();
    }
    bool running() const { return running_.load(std::memory_order_acquire); }

private:
    std::atomic<bool> stop_{false}, running_{false};
    std::thread th_;
};

}  // namespace strata::ds4
