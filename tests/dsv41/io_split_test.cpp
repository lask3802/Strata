// Strata. DS4_IO_SPLIT (IoQueue's split read, src/dsv41/ds4_pool.hpp) against a file of its own: the page cache dropped,
// a lead of a read's range read in again, then the read - its lead copied from the mapping, the rest from the disk
// unbuffered from the page that starts it - lands the file's bytes at dst + off % 4096 as the whole read does; a read
// with no resident lead, or all of it resident, is not split; os::resident_lead on its own.
//   ds4_io_split_test [dir]   exit 0: all passed (dir: the test file's, default /var/tmp - a file system with O_DIRECT;
//                             Linux only: elsewhere, or with no O_DIRECT or no way to drop the page cache there, skipped)
#include "../../src/dsv41/ds4_pool.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace os = strata::ds4::os;
using strata::ds4::IoQueue;

static int checks_bad = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++checks_bad;
}

#if defined(_WIN32)
int main() {
    std::printf("skipped: Linux only (resident_lead knows nothing elsewhere, so nothing is split)\n");
    return 0;
}
#else
static uint8_t byte_at(uint64_t o) { return (uint8_t) ((o * 2654435761u) >> 13); }

struct File {
    std::string path;
    int fd = -1, dfd = -1;
    uint64_t size = 0;
    uint8_t* map = nullptr;
    // the page cache of it dropped (the mapping made again: a page this process maps stays in)
    bool drop() {
        if (map != nullptr) munmap(map, (size_t) size);
        map = nullptr;
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0) return false;
        void* m = mmap(nullptr, (size_t) size, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) return false;
        map = (uint8_t*) m;
        return os::resident_lead(map, (size_t) size) == 0 && os::cold_bytes(map, (size_t) size) >= size - 4096;
    }
    bool warm(uint64_t off, uint64_t n) {   // [off, off + n) read buffered: in the page cache
        std::vector<uint8_t> b((size_t) n);
        return pread(fd, b.data(), (size_t) n, (off_t) off) == (ssize_t) n;
    }
};

// one read of [off, off + len) through q; true when it went and its bytes are the file's
static bool read_one(IoQueue& q, const File& f, uint64_t off, uint64_t len) {
    const size_t room = (size_t) ((len + 3 * 4096) & ~(uint64_t) 4095);
    uint8_t* dst = (uint8_t*) std::aligned_alloc(4096, room);
    std::memset(dst, 0xa5, room);
    std::atomic<int> left{1};
    std::atomic<bool> bad{false};
    IoQueue::Job j;
    j.fd = f.fd;
    j.dfd = f.dfd;
    j.off = off;
    j.len = len;
    j.dst = dst;
    j.left = &left;
    j.bad = &bad;
    j.map = f.map + off;
    q.push(&j, 1);
    while (left.load() != 0) q.wait_any(1000);
    const uint8_t* got = dst + off % 4096;
    bool same = !bad.load();
    for (uint64_t i = 0; same && i < len; ++i) same = got[i] == byte_at(off + i);
    std::free(dst);
    return same;
}

int main(int argc, char** argv) {
    File f;
    f.path = std::string(argc > 1 ? argv[1] : "/var/tmp") + "/ds4_io_split_test.bin";
    f.size = (16ull << 20) + 777;   // (not whole pages: the last read's tail short)
    {
        const int w = open(f.path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (w < 0) {
            std::printf("FAIL: cannot write %s\n", f.path.c_str());
            return 1;
        }
        std::vector<uint8_t> b(1 << 20);
        for (uint64_t at = 0; at < f.size; at += b.size()) {
            const size_t n = (size_t) std::min<uint64_t>(b.size(), f.size - at);
            for (size_t i = 0; i < n; ++i) b[i] = byte_at(at + i);
            if (write(w, b.data(), n) != (ssize_t) n) {
                std::printf("FAIL: cannot write %s\n", f.path.c_str());
                return 1;
            }
        }
        fsync(w);
        close(w);
    }
    f.fd = open(f.path.c_str(), O_RDONLY);
    f.dfd = open(f.path.c_str(), O_RDONLY | O_DIRECT);
    if (f.fd < 0 || f.dfd < 0) {
        std::printf("skipped: no O_DIRECT descriptor for %s (another dir: its argument)\n", f.path.c_str());
        unlink(f.path.c_str());
        return 0;
    }
    posix_fadvise(f.fd, 0, 0, POSIX_FADV_RANDOM);   // (no read-ahead: warm() puts in what it reads, no more)
    if (!f.drop()) {
        std::printf("skipped: %s's page cache cannot be dropped here\n", f.path.c_str());
        unlink(f.path.c_str());
        return 0;
    }
    const uint64_t M = 1ull << 20;

    // os::resident_lead: [0, 3 MiB) in
    check(f.warm(0, 3 * M), "the file's first 3 MiB read in");
    std::printf("      (resident_lead from byte 1000 over 6 MiB: %zu)\n", os::resident_lead(f.map + 1000, (size_t) (6 * M)));
    check(os::resident_lead(f.map + 1000, (size_t) (6 * M)) == 3 * M - 1000, "resident_lead: from byte 1000, 6 MiB: 3 MiB - 1000");
    check(os::resident_lead(f.map, (size_t) (2 * M)) == 2 * M, "resident_lead: all in: the length");
    check(os::resident_lead(f.map + 3 * M, (size_t) M) == 0, "resident_lead: the first page out: 0");
    check(os::resident_lead(f.map + 3 * M - 10, (size_t) M) == 10, "resident_lead: 10 bytes of an in page, then out: 10");

    // a split read: its lead from the cache, the rest unbuffered, the bytes the file's
    {
        IoQueue q(2, true, true);
        const bool same = read_one(q, f, 1000, 6 * M);
        const IoQueue::Split s = q.split_stats();
        check(same, "split on, [1000, 6 MiB + 1000) with [0, 3 MiB) in: the file's bytes");
        check(s.jobs == 1 && s.lead == 3 * M - 1000, "  split: 1 read, its lead 3 MiB - 1000");
        check(s.direct == 3 * M + 1000, "  split: the rest, 3 MiB + 1000, read unbuffered");
        check(q.direct_bytes() == s.direct && q.buffered_bytes() == s.lead, "  the queue's bytes: the lead buffered, the rest direct");
        check(s.checked == 1 && s.bad == 0, "  DS4_IO_SPLIT_CHECK: read again, the same bytes");
    }
    // the same with split off: the whole from the disk (the default)
    check(f.drop() && f.warm(0, 3 * M), "dropped, [0, 3 MiB) in again");
    {
        IoQueue q(2);
        check(read_one(q, f, 1000, 6 * M), "split off, the same read: the file's bytes");
        check(q.split_stats().jobs == 0 && q.direct_bytes() == 6 * M && q.buffered_bytes() == 0, "  all of it unbuffered, nothing split");
    }
    // no resident lead: not split, all unbuffered
    check(f.drop() && f.warm(5 * M, M), "dropped, [5, 6) MiB in");
    {
        IoQueue q(2, true);
        check(read_one(q, f, 4 * M + 333, 3 * M), "split on, [4 MiB + 333, 7 MiB + 333), its first page out: the file's bytes");
        check(q.split_stats().jobs == 0 && q.direct_bytes() == 3 * M, "  not split: all unbuffered");
    }
    // all of it resident: not split, buffered
    check(f.drop() && f.warm(8 * M, 2 * M), "dropped, [8, 10) MiB in");
    {
        IoQueue q(2, true);
        check(read_one(q, f, 8 * M + 5, M), "split on, [8 MiB + 5, 9 MiB + 5) all in: the file's bytes");
        check(q.split_stats().jobs == 0 && q.buffered_bytes() == M && q.direct_bytes() == 0, "  not split: buffered");
    }
    // the file's end: the unbuffered rest's last page short
    check(f.drop() && f.warm(12 * M, 2 * M), "dropped, [12, 14) MiB in");
    {
        IoQueue q(2, true, true);
        const uint64_t off = 12 * M + 4096 * 3 + 17;
        check(read_one(q, f, off, f.size - off), "split on, to the file's end (not a whole page): the file's bytes");
        const IoQueue::Split s = q.split_stats();
        check(s.jobs == 1 && s.lead == 14 * M - off && s.checked == 1 && s.bad == 0, "  split: its lead to 14 MiB, the check the same");
    }
    munmap(f.map, (size_t) f.size);
    close(f.fd);
    close(f.dfd);
    unlink(f.path.c_str());
    if (checks_bad == 0) std::printf("ALL OK\n");
    else std::printf("%d FAILED\n", checks_bad);
    return checks_bad == 0 ? 0 : 1;
}
#endif
