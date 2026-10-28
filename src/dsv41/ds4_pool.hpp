// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_pool.hpp - the DeepSeek engine's CPU workers: run(n, fn) runs fn(0..n-1) on the workers and the caller and
// returns when every job is done.  The design is glmfast::Workers' (src/core/glm_fast_state.hpp): one 64-bit ticket
// (generation << 32 | next index) claimed by CAS, so a worker that wakes late for an older batch can never take a job
// of the current one; idle workers spin for a while, then sleep on a condition variable.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "ds4_os.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strata::ds4 {

inline void cpu_relax() {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

class Pool {
public:
    explicit Pool(int n, const std::vector<int>& cpus = {}, int spin_us = 200) : spin_us_(spin_us) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this] { loop(0); });   // (generation 0: a batch started before the thread runs is still its)
            if (!cpus.empty()) os::pin_thread(th_.back(), cpus[(size_t) i % cpus.size()]);
        }
    }
    ~Pool() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }

    void run(int n, const std::function<void(int)>& fn) {
        start(n, fn);
        help();
        wait();
    }
    /// The batch goes to the workers; the caller may help() and must wait() before fn goes out of scope.  (Two pools
    /// - one a NUMA node - run at once this way: start both, help one, wait for both.)
    void start(int n, const std::function<void(int)>& fn) {
        n_ = n;
        if (n <= 0) return;
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
        ticket_.store(g << 32, std::memory_order_release);
        // (seq_cst, both: the other half is a worker's sleeping_ add, then its gen_ read - with release / acquire the store
        // and the load may pass each other, the worker sleeps through this batch and nobody wakes it)
        gen_.store(g, std::memory_order_seq_cst);
        if (sleeping_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void help() {
        if (n_ > 0) work(gen_.load(std::memory_order_acquire));
    }
    /// help() a job at a time: at most one job of the batch run here; false when none was left to take (the caller
    /// polls something between its jobs, DS4_EP_DEFER_BELOW's speed path the expert server's link)
    bool help_one() {
        if (n_ <= 0) return false;
        const uint64_t g = gen_.load(std::memory_order_acquire);
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return false;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return false;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            done_.fetch_add(1, std::memory_order_acq_rel);
            return true;
        }
    }
    /// every job of the batch done (wait() returns at once)
    bool done() const { return n_ <= 0 || done_.load(std::memory_order_acquire) >= n_; }
    void wait() {
        if (n_ <= 0) return;
        while (done_.load(std::memory_order_acquire) < n_) cpu_relax();
        n_ = 0;
    }

private:
    void work(uint64_t g) {
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void loop(uint64_t seen) {
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t g;
            for (;;) {
                g = gen_.load(std::memory_order_acquire);
                if (g != seen || quit_.load(std::memory_order_relaxed)) break;
                cpu_relax();
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return gen_.load() != seen || quit_.load(); });
                    sleeping_.fetch_sub(1);
                }
            }
            if (quit_.load()) return;
            seen = g;
            work(g);
        }
    }

    std::vector<std::thread> th_;
    int spin_us_;
    int n_ = 0;   // the current batch's size (the caller's side)
    const std::function<void(int)>* fn_ = nullptr;
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> total_{0};
    alignas(64) std::atomic<int> done_{0};
    std::atomic<int> sleeping_{0};
    std::atomic<bool> quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

/// File reads on threads of their own, in the order they were pushed, the pusher never waiting: the prompt path's
/// streamed experts (each part of an expert a job), unbuffered where the job has a direct descriptor and its range is
/// not all in the page cache already (the disk straight into dst, the cache neither used nor disturbed), buffered
/// otherwise or when the unbuffered read fails - the bytes land at dst + off % os::kDirectAlign either way.  A job's `left` is
/// counted down when it is done (`bad` set if it failed); wait_any() sleeps until some job finished or the timeout.
/// split (DS4_IO_SPLIT): a job whose range has a resident lead but not all of it - the lane's warm reads the first
/// blocks of an expert's gate / up rows, the pack's first 2/3 of each matrix - copies that lead from the mapping and
/// reads the rest from the disk unbuffered, from the page that starts it (the two parts of dst apart, one thread),
/// instead of all of it from the disk; split_check: each split job read again buffered and the bytes compared.
class IoQueue {
public:
    struct Job {
        os::fd_t fd = os::kBadFd, dfd = os::kBadFd;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        std::atomic<int>* left = nullptr;
        std::atomic<bool>* bad = nullptr;
        const uint8_t* mem = nullptr;   // non-null: a copy of len bytes from memory to dst instead of a read
        const void* map = nullptr;      // the range in the file's mapping: all in the page cache, it is read buffered
        uint8_t* keep = nullptr;        // non-null (a read): the len bytes read copied here as well
        std::atomic<bool>* cancel = nullptr;   // set before the job starts: it is dropped unread (bad is set, left counted down)
    };
    explicit IoQueue(int n, bool split = false, bool split_check = false) : split_(split), split_check_(split_check) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~IoQueue() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    /// low: a read that can wait (a guess) - the jobs of the first queue are all taken before any of this one's
    void push(const Job* jobs, int n, bool low = false) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; ++i) (low ? ql_ : q_).push_back(jobs[i]);
        }
        if (n == 1) cv_.notify_one();
        else cv_.notify_all();
    }
    void wait_any(int us) {
        std::unique_lock<std::mutex> lk(done_mu_);
        const uint64_t seen = done_;
        done_cv_.wait_for(lk, std::chrono::microseconds(us), [&] { return done_ != seen; });
    }
    double busy_ms() const { return busy_us_.load(std::memory_order_relaxed) / 1000.0; }
    uint64_t direct_bytes() const { return direct_.load(std::memory_order_relaxed); }
    uint64_t buffered_bytes() const { return buffered_.load(std::memory_order_relaxed); }
    bool split() const { return split_; }
    struct Split {   // (the split jobs: how many, their leads' bytes, their rests' read unbuffered; checked, not the same)
        uint64_t jobs = 0, lead = 0, direct = 0, checked = 0, bad = 0;
    };
    Split split_stats() const {
        return {sp_jobs_.load(std::memory_order_relaxed), sp_lead_.load(std::memory_order_relaxed),
                sp_direct_.load(std::memory_order_relaxed), sp_checked_.load(std::memory_order_relaxed),
                sp_bad_.load(std::memory_order_relaxed)};
    }

private:
    // a split read (split above) of j, its bytes from dst + lead: 1 read, 0 failed, -1 not split (no resident lead, all
    // of it resident, the mapping not page-aligned with the file) - the usual read then
    int split_read(const Job& j, uint64_t lead) {
        if (((uintptr_t) j.map - j.off) % os::kDirectAlign != 0) return -1;
        const uint64_t h = os::resident_lead(j.map, (size_t) j.len);
        if (h == 0 || h >= j.len) return -1;
        std::memcpy(j.dst + lead, j.map, (size_t) h);
        const uint64_t at = j.off + h, n = j.len - h;   // (at: a page's start, so the rest lands at d itself)
        uint8_t* d = j.dst + lead + h;                    // (j.dst + (at - j.off rounded down): aligned as j.dst)
        const bool dir = os::pread_direct(j.dfd, d, n, at);
        if (!dir && !os::pread_par(j.fd, d, n, at)) return 0;
        sp_jobs_.fetch_add(1, std::memory_order_relaxed);
        sp_lead_.fetch_add(h, std::memory_order_relaxed);
        buffered_.fetch_add(dir ? h : j.len, std::memory_order_relaxed);
        if (dir) {
            direct_.fetch_add(n, std::memory_order_relaxed);
            sp_direct_.fetch_add(n, std::memory_order_relaxed);
        }
        if (split_check_) {
            thread_local std::vector<uint8_t> v;
            if (v.size() < j.len) v.resize((size_t) j.len);
            const bool same = os::pread_par(j.fd, v.data(), j.len, j.off) &&
                              std::memcmp(v.data(), j.dst + lead, (size_t) j.len) == 0;
            sp_checked_.fetch_add(1, std::memory_order_relaxed);
            if (!same) sp_bad_.fetch_add(1, std::memory_order_relaxed);
        }
        return 1;
    }
    void loop() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty() || !ql_.empty(); });
                if (quit_ && q_.empty() && ql_.empty()) return;
                std::deque<Job>& qq = q_.empty() ? ql_ : q_;
                j = qq.front();
                qq.pop_front();
            }
            if (j.cancel != nullptr && j.cancel->load(std::memory_order_acquire)) {   // (dropped unread)
                if (j.bad != nullptr) j.bad->store(true, std::memory_order_release);
                if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
                {
                    std::lock_guard<std::mutex> lk(done_mu_);
                    ++done_;
                }
                done_cv_.notify_all();
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t lead = j.off % os::kDirectAlign;
            bool ok = false;
            const int sp = split_ && j.mem == nullptr && j.dfd != os::kBadFd && j.map != nullptr ? split_read(j, lead) : -1;
            if (sp >= 0) {
                ok = sp == 1;
            } else if (j.mem != nullptr) {
                std::memcpy(j.dst, j.mem, (size_t) j.len);
                ok = true;
            } else if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                       os::pread_direct(j.dfd, j.dst, j.len, j.off)) {   // (the page cache's ones: buffered, below)
                ok = true;
                direct_.fetch_add(j.len, std::memory_order_relaxed);
            } else if (os::pread_par(j.fd, j.dst + lead, j.len, j.off)) {
                ok = true;
                buffered_.fetch_add(j.len, std::memory_order_relaxed);
            }
            if (ok && j.keep != nullptr && j.mem == nullptr) std::memmove(j.keep, j.dst + lead, (size_t) j.len);   // (keep may overlap dst: the store reads in place and moves up)
            busy_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - t0).count(),
                               std::memory_order_relaxed);
            if (!ok && j.bad != nullptr) j.bad->store(true, std::memory_order_release);
            if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(done_mu_);
                ++done_;
            }
            done_cv_.notify_all();
        }
    }
    std::vector<std::thread> th_;
    std::mutex mu_, done_mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> q_, ql_;
    uint64_t done_ = 0;
    bool quit_ = false;
    std::atomic<uint64_t> busy_us_{0}, direct_{0}, buffered_{0};
    bool split_ = false, split_check_ = false;
    std::atomic<uint64_t> sp_jobs_{0}, sp_lead_{0}, sp_direct_{0}, sp_checked_{0}, sp_bad_{0};
};

}  // namespace strata::ds4
