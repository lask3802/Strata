// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_os.hpp - the few OS services maya-ds4 needs, on Linux and on Windows: files read at an offset and
// mapped read-only (the experts read in place), page-cache hints, anonymous host memory on a NUMA node, thread pinning,
// each NUMA node's physical cores, the RAM this process can still get.  (On Windows as src/core/expert_source.cpp
// does it for GLM: no FILE_FLAG_RANDOM_ACCESS - the page cache must keep what it read.)
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>   // (K32QueryWorkingSetEx: kernel32's, no psapi.lib)
#include <timeapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "winmm.lib")
#endif
#else
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::ds4::os {

/// Windows sleeps in 15.6 ms ticks unless a process asks for finer ones: the CPU lane's service thread naps (sleep_for 20 us)
/// between tokens, and the device's first wait of the next token then waits out the tick.  1 ms for the process's life
/// (Project Maya #358).  Off by default (owner PC, shipped profile, 3 pairs: not faster in every pair); DS4_TIMER_MS=1 turns it on.
/// Elsewhere nothing.
inline void fine_timer() {
#if defined(_WIN32)
    const char* v = std::getenv("DS4_TIMER_MS");
    const int ms = v != nullptr ? std::atoi(v) : 0;
    if (ms > 0) timeBeginPeriod((UINT) ms);
#endif
}

using fd_t = intptr_t;
constexpr fd_t kBadFd = -1;

#if defined(_WIN32)
inline std::wstring wide(const std::string& s) {   // UTF-8 -> UTF-16 (a path with non-ASCII characters)
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w((size_t) (n > 0 ? n : 1), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}
#endif

/// A file opened for reading (kBadFd: it could not be).
inline fd_t open_ro(const std::string& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    return h == INVALID_HANDLE_VALUE ? kBadFd : (fd_t) h;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    return fd < 0 ? kBadFd : (fd_t) fd;
#endif
}
/// The same file opened for unbuffered reads (Linux: O_DIRECT - no page cache, the disk straight into the buffer;
/// kBadFd where the filesystem refuses it, or on Windows, whose unbuffered reads of a mapped file take a slow path)
inline fd_t open_direct(const std::string& path) {
#if defined(_WIN32)
    // FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED: the disk straight into an aligned buffer, any number of reads in flight on
    // one handle (DS4_WIN_DIRECT=0: none, the page cache's buffered reads as before)
    static const bool off = [] {
        const char* v = std::getenv("DS4_WIN_DIRECT");
        return v != nullptr && std::atoi(v) == 0;
    }();
    if (off) return kBadFd;
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    return h == INVALID_HANDLE_VALUE ? kBadFd : (fd_t) h;
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    return fd < 0 ? kBadFd : (fd_t) fd;
#endif
}
/// [off, off + n) through an unbuffered descriptor into dst, a kDirectAlign-aligned buffer of at least
/// direct_span(off, n) bytes: the file read from off rounded down, so the bytes land at dst + (off % kDirectAlign).
/// False if a read fails (the caller reads buffered instead).
constexpr uint64_t kDirectAlign = 4096;
inline uint64_t direct_span(uint64_t off, uint64_t n) {
    const uint64_t a0 = off & ~(kDirectAlign - 1), a1 = (off + n + kDirectAlign - 1) & ~(kDirectAlign - 1);
    return a1 - a0;
}
inline bool pread_direct(fd_t f, void* dst, uint64_t n, uint64_t off) {
#if defined(_WIN32)
    struct Ev {   // (each thread its own, closed when it ends)
        HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ~Ev() {
            if (h != nullptr) CloseHandle(h);
        }
    };
    static thread_local Ev ev;
    if (ev.h == nullptr) return false;
    const uint64_t a0 = off & ~(kDirectAlign - 1), need = direct_span(off, n), want = (off - a0) + n;
    uint8_t* p = (uint8_t*) dst;
    uint64_t got = 0;
    while (got < need) {
        const DWORD ask = (DWORD) std::min<uint64_t>(need - got, 1ull << 30);   // (a multiple of 4096)
        OVERLAPPED ov{};
        const uint64_t at = a0 + got;
        ov.Offset = (DWORD) (at & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (at >> 32);
        ov.hEvent = ev.h;
        ResetEvent(ev.h);
        DWORD r = 0;
        if (!ReadFile((HANDLE) f, p + got, ask, nullptr, &ov)) {
            const DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            if (e != ERROR_IO_PENDING) return false;
        }
        if (!GetOverlappedResult((HANDLE) f, &ov, &r, TRUE)) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            return false;
        }
        if (r == 0) break;   // (the file's end: the last aligned read is short)
        got += r;
        if (r % kDirectAlign != 0) break;
    }
    return got >= want;
#else
    const uint64_t a0 = off & ~(kDirectAlign - 1), need = direct_span(off, n), want = (off - a0) + n;
    uint8_t* p = (uint8_t*) dst;
    uint64_t got = 0;
    while (got < need) {
        const ssize_t r = ::pread((int) f, p + got, (size_t) std::min<uint64_t>(need - got, 1ull << 30), (off_t) (a0 + got));
        if (r < 0) return false;
        if (r == 0) break;   // (the file's end: the last aligned read is short)
        got += (uint64_t) r;
        if ((uint64_t) r % kDirectAlign != 0) break;
    }
    return got >= want;
#endif
}
/// Windows: the file opened for unbuffered reads (FILE_FLAG_NO_BUFFERING), for a file no view of this process maps -
/// unbuffered reads of a mapped file take a slow path there (open_direct); kBadFd elsewhere
inline fd_t open_unbuffered(const std::string& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
    return h == INVALID_HANDLE_VALUE ? kBadFd : (fd_t) h;
#else
    (void) path;
    return kBadFd;
#endif
}
/// [off, off + n) through open_unbuffered's handle (one thread's) into buf, kDirectAlign-aligned and at least
/// direct_span(off, n) bytes: the bytes land at buf + off % kDirectAlign
inline bool pread_unbuffered(fd_t f, void* buf, uint64_t n, uint64_t off) {
#if defined(_WIN32)
    const uint64_t a0 = off & ~(kDirectAlign - 1), span = direct_span(off, n);
    OVERLAPPED ov{};
    ov.Offset = (DWORD) a0;
    ov.OffsetHigh = (DWORD) (a0 >> 32);
    DWORD got = 0;
    return ReadFile((HANDLE) f, buf, (DWORD) span, &got, &ov) && got >= (off - a0) + n;
#else
    (void) f, (void) buf, (void) n, (void) off;
    return false;
#endif
}
/// The same file opened for reads from many threads at once (Windows runs the reads of a synchronous handle one at a
/// time, so an overlapped one; Linux: a plain descriptor, pread is concurrent anyway)
inline fd_t open_par(const std::string& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    return h == INVALID_HANDLE_VALUE ? kBadFd : (fd_t) h;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    return fd < 0 ? kBadFd : (fd_t) fd;
#endif
}
/// n bytes at off through open_par's handle (any thread, concurrently)
inline bool pread_par(fd_t f, void* dst, uint64_t n, uint64_t off) {
#if defined(_WIN32)
    struct Ev {   // (each thread its own, closed when it ends)
        HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ~Ev() {
            if (h != nullptr) CloseHandle(h);
        }
    };
    static thread_local Ev ev;
    if (ev.h == nullptr) return false;
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const DWORD want = (DWORD) std::min<uint64_t>(n, 1ull << 30);
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        ov.hEvent = ev.h;
        ResetEvent(ev.h);
        DWORD got = 0;
        if (!ReadFile((HANDLE) f, p, want, nullptr, &ov) && GetLastError() != ERROR_IO_PENDING) return false;
        if (!GetOverlappedResult((HANDLE) f, &ov, &got, TRUE) || got == 0) return false;
        p += got, off += got, n -= got;
    }
    return true;
#else
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const ssize_t r = ::pread((int) f, p, (size_t) std::min<uint64_t>(n, 1ull << 30), (off_t) off);
        if (r <= 0) return false;
        p += r, off += (uint64_t) r, n -= (uint64_t) r;
    }
    return true;
#endif
}
/// `from` renamed to `to`, replacing `to` if it exists (rename(2) does; Windows' rename fails on an existing file, so
/// MoveFileExW with MOVEFILE_REPLACE_EXISTING)
inline bool replace_file(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    return MoveFileExW(wide(from).c_str(), wide(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}
inline void close_fd(fd_t f) {
    if (f == kBadFd) return;
#if defined(_WIN32)
    CloseHandle((HANDLE) f);
#else
    ::close((int) f);
#endif
}
inline uint64_t file_size(fd_t f) {
#if defined(_WIN32)
    LARGE_INTEGER sz{};
    return GetFileSizeEx((HANDLE) f, &sz) ? (uint64_t) sz.QuadPart : 0;
#else
    struct stat sb;
    return ::fstat((int) f, &sb) == 0 && sb.st_size > 0 ? (uint64_t) sb.st_size : 0;
#endif
}
/// n bytes at offset off into dst (any thread, concurrently); false if the file ends first or a read fails.
inline bool pread_full(fd_t f, void* dst, uint64_t n, uint64_t off) {
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const uint64_t want = std::min<uint64_t>(n, 1ull << 30);
#if defined(_WIN32)
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        if (!ReadFile((HANDLE) f, p, (DWORD) want, &got, &ov) || got == 0) return false;
        const uint64_t r = got;
#else
        const ssize_t rr = ::pread((int) f, p, (size_t) want, (off_t) off);
        if (rr <= 0) return false;
        const uint64_t r = (uint64_t) rr;
#endif
        p += r, off += r, n -= r;
    }
    return true;
}
/// The whole file mapped read-only (nullptr: it could not be).
inline void* map_ro(fd_t f, uint64_t bytes) {
    if (bytes == 0) return nullptr;
#if defined(_WIN32)
    HANDLE m = CreateFileMappingW((HANDLE) f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) return nullptr;
    void* v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);   // (the view keeps the mapping)
    return v;
#else
    void* p = ::mmap(nullptr, (size_t) bytes, PROT_READ, MAP_SHARED, (int) f, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}
inline void unmap(void* p, uint64_t bytes) {
    if (p == nullptr) return;
#if defined(_WIN32)
    (void) bytes;
    UnmapViewOfFile(p);
#else
    ::munmap(p, (size_t) bytes);
#endif
}
/// A hint that [off, off + n) of the file is needed soon (need) or not again soon (!need); nothing where the OS has
/// no such hint for a file (Windows' page cache reads ahead on its own).
inline void advise(fd_t f, uint64_t off, uint64_t n, bool need) {
#if defined(_WIN32)
    (void) f, (void) off, (void) n, (void) need;
#else
    posix_fadvise((int) f, (off_t) off, (off_t) n, need ? POSIX_FADV_WILLNEED : POSIX_FADV_DONTNEED);
#endif
}

/// Ranges of a file's mapping about to be read: the OS reads them in now, with large requests side by side, instead of
/// a page fault at a time later (Windows: PrefetchVirtualMemory - cold expert-sized pieces came in 5x faster on an NVMe;
/// Linux: madvise WILLNEED, the kernel's read-ahead of the whole range).  Best effort: a failure changes nothing.
inline void prefetch(const std::vector<std::pair<const void*, size_t>>& ranges) {
    if (ranges.empty()) return;
#if defined(_WIN32)
    std::vector<WIN32_MEMORY_RANGE_ENTRY> r(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        r[i].VirtualAddress = const_cast<void*>(ranges[i].first);
        r[i].NumberOfBytes = ranges[i].second;
    }
    PrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) r.size(), r.data(), 0);
#else
    const uintptr_t pg = 4096;
    for (const auto& x : ranges) {
        const uintptr_t a = (uintptr_t) x.first, b = a & ~(pg - 1);
        madvise((void*) b, x.second + (a - b), MADV_WILLNEED);
    }
#endif
}

/// [p, p + n) of a file's mapping read in and mapped now (Linux 5.14+ madvise MADV_POPULATE_READ: the page cache
/// filled from the file where it has not the pages and this process's page tables set, so the first read of each page
/// is no fault): 1 done, 0 not had here (EINVAL: a kernel without it; Windows), -1 failed otherwise (errno: a part of
/// the range not mapped, an I/O error) - the caller reads it then (pread: the page cache, not the page tables)
inline int populate(const void* p, size_t n) {
#if defined(_WIN32)
    (void) p, (void) n;
    return 0;
#else
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    int r;
    while ((r = madvise((void*) b, n + (a - b), MADV_POPULATE_READ)) != 0 && errno == EINTR) {}
    return r == 0 ? 1 : errno == EINVAL ? 0 : -1;
#endif
}

/// Whether every page of [p, p + n) of a file's mapping is in the page cache (Linux mincore; elsewhere false: the
/// caller reads buffered anyway, which the cache serves)
inline bool resident(const void* p, size_t n) {
#if defined(_WIN32)
    (void) p, (void) n;
    return false;
#else
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    const size_t len = n + (a - b), np = (len + pg - 1) / pg;
    thread_local std::vector<unsigned char> v;
    if (v.size() < np) v.resize(np);
    if (mincore((void*) b, len, v.data()) != 0) return false;
    for (size_t i = 0; i < np; ++i)
        if ((v[i] & 1) == 0) return false;
    return true;
#endif
}
/// The bytes of [p, p + n) of a file's mapping not in the page cache, by whole 4 KiB pages (Linux mincore; elsewhere
/// or on an error 0: not known)
inline size_t cold_bytes(const void* p, size_t n) {
#if defined(_WIN32)
    (void) p, (void) n;
    return 0;
#else
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    const size_t len = n + (a - b), np = (len + pg - 1) / pg;
    thread_local std::vector<unsigned char> v;
    if (v.size() < np) v.resize(np);
    if (mincore((void*) b, len, v.data()) != 0) return 0;
    size_t cold = 0;
    for (size_t i = 0; i < np; ++i) cold += (v[i] & 1) == 0 ? 1 : 0;
    return cold * pg;
#endif
}
/// The bytes of [p, p + n) of a file's mapping before its first page not in the page cache: n when every page is in, 0
/// when p's own page is not, else up to that page's start (Linux mincore; elsewhere or on an error 0: not known)
inline size_t resident_lead(const void* p, size_t n) {
#if defined(_WIN32)
    (void) p, (void) n;
    return 0;
#else
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    const size_t len = n + (a - b), np = (len + pg - 1) / pg;
    thread_local std::vector<unsigned char> v;
    if (v.size() < np) v.resize(np);
    if (mincore((void*) b, len, v.data()) != 0) return 0;
    size_t i = 0;
    while (i < np && (v[i] & 1) != 0) ++i;
    return i == np ? n : i == 0 ? 0 : (size_t) (b + i * pg - a);
#endif
}

/// Whether a range of a file's mapping looks resident, so that prefetch() would only cost CPU time (it walks every page:
/// ~50 ns a 4 KiB page on Windows - 0.34 ms for a 6.3 MB expert matrix on a Strix Halo): `samples` pages spread over
/// it all resident (Linux: mincore, the page cache; Windows: QueryWorkingSetEx, the process's working set - a page on
/// the standby list counts as not resident, and prefetch() is what brings those back cheaply).  An expert's pages are
/// read together, so they come and go together: a wrong "yes" costs page faults, a wrong "no" a prefetch.
inline bool resident_sampled(const void* p, size_t bytes, int samples = 8) {
    if (bytes == 0) return true;
    const uintptr_t pg = 4096, a = (uintptr_t) p, span = bytes - 1;
    samples = std::max(1, std::min(samples, 16));
#if defined(_WIN32)
    PSAPI_WORKING_SET_EX_INFORMATION w[16] = {};
    for (int i = 0; i < samples; ++i)
        w[i].VirtualAddress = (PVOID) ((a + (samples > 1 ? span / (samples - 1) * i : 0)) & ~(pg - 1));
    if (!K32QueryWorkingSetEx(GetCurrentProcess(), w, (DWORD) (samples * sizeof w[0]))) return false;
    for (int i = 0; i < samples; ++i)
        if (!w[i].VirtualAttributes.Valid) return false;
    return true;
#else
    for (int i = 0; i < samples; ++i) {
        unsigned char v = 0;
        const uintptr_t at = (a + (samples > 1 ? span / (samples - 1) * i : 0)) & ~(pg - 1);
        if (mincore((void*) at, pg, &v) != 0 || !(v & 1)) return false;
    }
    return true;
#endif
}
/// prefetch() of the ranges that do not look resident (resident_sampled) - on Windows, where PrefetchVirtualMemory walks
/// every page of a resident range (0.34 ms a 6.3 MB expert matrix on a Strix Halo); Linux's madvise(WILLNEED) on cached
/// pages costs ~1 us a matrix (a Core i5-12600T), less than the residency check (2.5 us): every range there
inline void prefetch_cold(const std::vector<std::pair<const void*, size_t>>& ranges) {
#if defined(_WIN32)
    static const bool all = [] {   // (DS4_PF_ALL=1: every range, as before)
        const char* v = std::getenv("DS4_PF_ALL");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (all) return prefetch(ranges);
    std::vector<std::pair<const void*, size_t>> cold;
    for (const auto& x : ranges)
        if (!resident_sampled(x.first, x.second)) cold.push_back(x);
    prefetch(cold);
#else
    prefetch(ranges);
#endif
}

/// [p, p + n) of a file's mapping kept in RAM (Linux mlock: every page read in and pinned until unmapped); false where
/// it was not (RLIMIT_MEMLOCK, Windows: the caller touches the pages instead, and the page cache may drop them)
inline bool lock_range(const void* p, size_t n) {
#if defined(_WIN32)
    (void) p, (void) n;
    return false;
#else
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    return mlock((const void*) b, n + (a - b)) == 0;
#endif
}
/// lock_range's range let go (the page cache may drop it again)
inline void unlock_range(const void* p, size_t n) {
#if defined(_WIN32)
    (void) p, (void) n;
#else
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    munlock((const void*) b, n + (a - b));
#endif
}

/// Whether [p, p + n) of a file's mapping is in RAM now: every page (Linux, one mincore), sampled pages of the working
/// set (Windows; a page on the standby list counts as not in RAM)
inline bool in_ram(const void* p, size_t n) {
#if defined(_WIN32)
    return resident_sampled(p, n, 8);
#else
    return resident(p, n);
#endif
}

/// Anonymous memory on NUMA node `node` of `nodes` (node < 0 or nodes < 2: wherever the first touch puts it), on huge
/// pages where the OS gives them (huge false: no MADV_HUGEPAGE - 4 KiB pages, so a fault never compacts memory for one);
/// nullptr if it cannot be had.
inline void* anon_alloc(size_t bytes, int node, int nodes, bool huge = true) {
#if defined(_WIN32)
    (void) huge;
    if (node >= 0 && nodes > 1)
        return VirtualAllocExNuma(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
                                  (DWORD) node);
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* m = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    if (huge) madvise(m, bytes, MADV_HUGEPAGE);
    if (node >= 0 && nodes > 1) {
        unsigned long mask[4] = {0, 0, 0, 0};
        mask[node / 64] = 1ul << (node % 64);
        syscall(SYS_mbind, m, bytes, 2 /* MPOL_BIND */, mask, 256, 0);
    }
    return m;
#endif
}
/// [p, p + n) of new anonymous memory faulted in now, for writing (Linux 5.14+ madvise MADV_POPULATE_WRITE; without it,
/// and on Windows, a zero written to every 4 KiB page - new memory reads as zeros either way), so no later write to it
/// faults; what did it
inline const char* populate_write(void* p, size_t n) {
    if (n == 0) return "nothing";
#if !defined(_WIN32)
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    int r;
    while ((r = madvise((void*) b, n + (a - b), MADV_POPULATE_WRITE)) != 0 && errno == EINTR) {}
    if (r == 0) return "MADV_POPULATE_WRITE";
#endif
    volatile uint8_t* q = (volatile uint8_t*) p;
    for (size_t i = 0; i < n; i += 4096) q[i] = 0;
    q[n - 1] = 0;
    return "a zero written to every 4 KiB page";
}
/// An address range of `bytes` reserved (no memory behind it yet), and pieces of it made real: the store's arena is one range so a
/// kernel can tell its pointers by [lo, hi).  nullptr: it could not be had.
inline void* range_reserve(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_READWRITE);
#else
    void* m = ::mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return m == MAP_FAILED ? nullptr : m;
#endif
}
inline bool range_commit(void* p, size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(p, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    return ::mprotect(p, bytes, PROT_READ | PROT_WRITE) == 0;
#endif
}
/// A committed piece of a reserved range given back to the system (its address range stays reserved)
inline void range_decommit(void* p, size_t bytes) {
#if defined(_WIN32)
    VirtualFree(p, bytes, MEM_DECOMMIT);
#else
    ::madvise(p, bytes, MADV_DONTNEED);
    ::mprotect(p, bytes, PROT_NONE);
#endif
}
inline void range_release(void* p, size_t bytes) {
    if (p == nullptr) return;
#if defined(_WIN32)
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    ::munmap(p, bytes);
#endif
}
inline void anon_free(void* p, size_t bytes) {
    if (p == nullptr) return;
#if defined(_WIN32)
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    ::munmap(p, bytes);
#endif
}

#if !defined(_WIN32)
/// The calling thread's mask from before its first pin_self (saved: it pinned itself and has not gone back since)
struct PinSaved {
    bool saved = false;
    cpu_set_t before;
};
inline PinSaved& pin_saved() {
    thread_local PinSaved s;
    return s;
}
#endif
/// The calling thread, or `t`, on logical CPU `cpu` (Windows: the first 64 CPUs, processor group 0).  pin_self(-1):
/// the calling thread's mask from before its first pin_self again (nothing if it was not pinned since).
inline void pin_self(int cpu) {
#if defined(_WIN32)
    thread_local DWORD_PTR before = 0;
    if (cpu >= 0 && cpu < 64) {
        const DWORD_PTR was = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
        if (before == 0) before = was;
    } else if (cpu < 0 && before != 0) {
        SetThreadAffinityMask(GetCurrentThread(), before);
        before = 0;
    }
#else
    bool& saved = pin_saved().saved;
    cpu_set_t& before = pin_saved().before;
    if (cpu >= 0) {
        if (!saved) saved = sched_getaffinity(0, sizeof before, &before) == 0;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof set, &set);
    } else if (saved) {
        sched_setaffinity(0, sizeof before, &before);
        saved = false;
    }
#endif
}
/// Whether the calling thread may run on `cpu` (Linux: its mask - a container's cpuset is in it -, from before its
/// pin_self if it pinned itself, so a pinned thread still sees the others; Windows: the process's)
inline bool cpu_allowed(int cpu) {
#if defined(_WIN32)
    DWORD_PTR proc = 0, sys = 0;
    return cpu >= 0 && cpu < 64 && GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) && ((proc >> cpu) & 1);
#else
    if (pin_saved().saved) return cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &pin_saved().before);
    cpu_set_t s;
    return cpu >= 0 && cpu < CPU_SETSIZE && sched_getaffinity(0, sizeof s, &s) == 0 && CPU_ISSET(cpu, &s);
#endif
}
/// The one logical CPU the calling thread may run on (-1: more than one, or not known)
inline int pinned_cpu() {
    int one = -1;
#if defined(_WIN32)
    GROUP_AFFINITY ga{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &ga) || ga.Group != 0) return -1;
    for (int c = 0; c < 64; ++c)
        if ((ga.Mask >> c) & 1) {
            if (one >= 0) return -1;
            one = c;
        }
#else
    cpu_set_t s;
    if (sched_getaffinity(0, sizeof s, &s) != 0 || CPU_COUNT(&s) != 1) return -1;
    for (int c = 0; c < CPU_SETSIZE && one < 0; ++c)
        if (CPU_ISSET(c, &s)) one = c;
#endif
    return one;
}
/// The logical CPU the calling thread runs on now (-1: not known)
inline int cpu_now() {
#if defined(_WIN32)
    PROCESSOR_NUMBER pn{};
    GetCurrentProcessorNumberEx(&pn);
    return pn.Group == 0 ? (int) pn.Number : -1;
#else
    return sched_getcpu();
#endif
}
/// The calling thread's CPU time, ms (Linux: CLOCK_THREAD_CPUTIME_ID; Windows: its cycles at the base clock are not
/// time - -1)
inline double thread_cpu_ms() {
#if defined(_WIN32)
    return -1.0;
#else
    timespec t{};
    return clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0 ? (double) t.tv_sec * 1e3 + (double) t.tv_nsec / 1e6 : -1.0;
#endif
}
/// A kernel CPU list ("3,19", "3-4", "0-1,8-9") as its CPUs, in its order; it ends at anything else
inline std::vector<int> cpu_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p >= '0' && *p <= '9';) {
        char* q = nullptr;
        const int lo = (int) std::strtol(p, &q, 10);
        int hi = lo;
        if (*q == '-') hi = (int) std::strtol(q + 1, &q, 10);
        for (int c = lo; c <= hi && c - lo < 1024; ++c) v.push_back(c);
        p = *q == ',' ? q + 1 : q;
    }
    return v;
}
/// The logical CPUs of `cpu`'s physical core, `cpu` among them, ascending ({cpu} where the OS does not say)
inline std::vector<int> cpu_siblings(int cpu) {
    std::vector<int> v;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> b(len);
    if (cpu >= 0 && cpu < 64 && len > 0 &&
        GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) b.data(), &len))
        for (size_t o = 0; o < (size_t) len && v.empty();) {
            auto* e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (b.data() + o);
            const KAFFINITY m = e->Processor.GroupCount > 0 && e->Processor.GroupMask[0].Group == 0
                                    ? e->Processor.GroupMask[0].Mask : 0;
            if ((m >> cpu) & 1)
                for (int c = 0; c < 64; ++c)
                    if ((m >> c) & 1) v.push_back(c);
            o += e->Size;
        }
#else
    FILE* f = cpu >= 0 ? std::fopen(("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list").c_str(), "r")
                       : nullptr;
    char s[256] = {};
    if (f != nullptr) {
        if (std::fgets(s, sizeof s, f) == nullptr) s[0] = '\0';
        std::fclose(f);
    }
    v = cpu_list(s);   // "3,19" or "3-4"
#endif
    if (std::find(v.begin(), v.end(), cpu) == v.end()) v.assign(1, cpu);
    std::sort(v.begin(), v.end());
    return v;
}
inline void pin_thread(std::thread& t, int cpu) {
#if defined(_WIN32)
    if (cpu >= 0 && cpu < 64) SetThreadAffinityMask((HANDLE) t.native_handle(), (DWORD_PTR) 1 << cpu);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
#endif
}

#if defined(_WIN32)
/// Each NUMA node's physical cores (the first logical CPU of each), processor group 0 - Windows' answer to
/// /sys/devices/system/node; empty if the OS does not say.
inline std::vector<std::vector<int>> node_cores(bool smt = false) {   // (smt: every logical CPU of a core, not its first)
    std::vector<std::vector<int>> nodes;
    auto query = [](LOGICAL_PROCESSOR_RELATIONSHIP rel, std::vector<uint8_t>& buf) {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(rel, nullptr, &len);
        buf.resize(len);
        return len > 0 && GetLogicalProcessorInformationEx(rel, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len);
    };
    std::vector<uint8_t> nb, cb;
    if (!query(RelationNumaNode, nb) || !query(RelationProcessorCore, cb)) return nodes;
    std::vector<KAFFINITY> masks;
    for (size_t o = 0; o < nb.size();) {
        auto* e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (nb.data() + o);
        if (e->NumaNode.GroupMask.Group == 0) masks.push_back(e->NumaNode.GroupMask.Mask);
        o += e->Size;
    }
    nodes.resize(masks.size());
    for (size_t o = 0; o < cb.size();) {
        auto* e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (cb.data() + o);
        if (e->Processor.GroupCount > 0 && e->Processor.GroupMask[0].Group == 0) {
            const KAFFINITY m = e->Processor.GroupMask[0].Mask;
            int first = -1;
            for (int b = 0; b < 64; ++b)
                if ((m >> b) & 1) {
                    first = b;
                    break;
                }
            for (size_t n = 0; n < masks.size() && first >= 0; ++n)
                if ((masks[n] >> first) & 1) {
                    nodes[n].push_back(first);
                    if (smt)
                        for (int b = first + 1; b < 64; ++b)
                            if ((m >> b) & 1) nodes[n].push_back(b);
                }
        }
        o += e->Size;
    }
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [](const std::vector<int>& v) { return v.empty(); }),
                nodes.end());
    for (auto& v : nodes) std::sort(v.begin(), v.end());
    return nodes;
}
/// The RAM this process can still get (~0: not known).
inline uint64_t ram_avail() {
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? (uint64_t) s.ullAvailPhys : ~uint64_t{0};
}
#endif

/// Windows' commit (RAM + page file, what every private allocation is charged to - on an integrated GPU the GPU's
/// own allocations too): the limit, what this process can still commit, the RAM; all ~0 where there is no such limit
/// (Linux overcommits; its RAM is ram_avail's business).
struct Commit {
    uint64_t limit = ~uint64_t{0}, avail = ~uint64_t{0}, ram = ~uint64_t{0};
};
inline Commit commit() {
    Commit c;
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) {
        c.limit = (uint64_t) s.ullTotalPageFile;
        c.avail = (uint64_t) s.ullAvailPageFile;
        c.ram = (uint64_t) s.ullTotalPhys;
    }
#endif
    return c;
}

}  // namespace strata::ds4::os
