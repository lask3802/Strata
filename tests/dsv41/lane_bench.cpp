// Strata. The i-quant CPU lane's decode calls (Impl::cpu_experts_iq with one part, NH = 1) replayed on a pack's experts
// without the engine or a GPU: where the lane's time goes inside its jobs.  Each call is the engine's: the prefetch hint
// on the call's ranges and the token's x quantized (prep), the gate / up jobs of RB rows on a Pool (the engine's, the
// caller helping), the h quantized by the caller, the down jobs; each batch timed as DS4_LANE_PROF times it (wall, the
// caller's wait once it ran out of jobs - tail - and the jobs' time in all - busy).  A "token" is one call a layer over
// --layers, --picks experts a call over the [0, k) n_ff blocks of 9 (H': k 6 is the main's share).  The variants change
// one thing each against "file" (the engine's state: the pack's files mapped, every page touched before, madvise
// WILLNEED a call):
//   file-nopf      no hint                                file-fresh     a new mapping, each token's experts touched in
//   file-populate  a new mapping, MADV_POPULATE_READ        it for the first time (the engine's first touches: minor
//                  instead of WILLNEED (Linux 5.14+)        faults); the page cache warm
//   anon4k         the experts copied to anonymous memory, 4 KiB pages (MADV_NOHUGEPAGE): no file mapping
//   anon2m         the same on huge pages (MADV_HUGEPAGE: fewer TLB misses, the prefetchers not stopped at 4 KiB)
//   hot            every call the same experts (layer 0's first picks): from the caches, the kernels' own rate
//                  (hot:k=9:picks=8 is the engine's load probe: 8 whole experts, one token each, the same ones again)
//   :k=K :picks=N :rb=R :split (one job a thread an entry) :thr=T   - any variant, e.g. file:thr=8, hot:picks=1
//   :thr=T past the physical cores takes their second hardware threads (SMT), in the cores' order; :skip=2+18 leaves
//   those CPUs out (the link core's pair); :empty - the same jobs doing nothing (the batches' dispatch and barrier
//   alone); :pool=fix - the bench's FixPool instead of the engine's Pool (its flags on lines of their own, each worker's
//   done count added once a batch, the jobs' busy time in a slot a thread: no shared write a job)
//   :dparts - the down's slice from --down-parts DIR (ep_parts.py --down-only 0:K: each row's [0, K) back to back, as
//   DS4_IQ_DOWN_PARTS reads it) instead of the pack's rows in place (strided); a slice the parts do not end at: the pack's
//   :pf=N - the hint only on the experts the layer's last N calls did not take (DS4_IQ_PF=N)
//   :spin=A+B - a thread on each CPU listed spinning on a load (no pause) the whole variant: a busy neighbour (the
//   engine's CUDA host thread waiting, a polling link) on a lane core or its sibling
//   :trace - the calls of --trace FILE instead of the pool's (file / file-nopf only): the engine's picks a layer, their
//   number varying call to call and their reuse as a session has it, over the pack's every expert (the TLB, the page
//   tables and the caches as cold as the engine's) - FILE is DS4_TRACE's (every pick of a decode token; --trace-hot N
//   leaves out each layer's N most picked, the ones the GPUs would hold) or DS4_LANE_TRACE's (the lane's own calls,
//   their slices too); one replay of --trace-tokens (default all) warms the mapping, a second is timed
//   :cold - (:trace) no warm replay: the timed one on a mapping no page of which is mapped yet (the page cache warm), as an
//   engine process's lane meets its experts - each one's first call faults its pages in (minor faults), later ones not
// Rates: lane GB/s = bytes / (prep + gu + hq + down); in-job GB/s = a batch's bytes over its busy time / threads (the
// rate while the jobs run); a thread's GB/s = bytes / busy.  minflt a token: getrusage (Linux), the page faults (Windows).
// Linux: the core's counters of this process's threads over the timed tokens (perf_event_open; root or
// perf_event_paranoid <= 2 - "pmc: unavailable" otherwise): IPC, the dispatch stalls, the FP pipes, the L1D fills.
//   ds4_lane_bench --pack DIR [--layers 0-39] [--picks 4] [--k 6] [--pool 8] [--tokens 16] [--fresh-tokens 4]
//                  [--threads N (every physical core)] [--variants file,file-nopf,..] [--dim 5120] [--ff 2304]
//                  [--remap FROM=TO (the pack's paths: the host's view of a container's)] [--no-pmc]
//                  [--down-parts DIR] [--trace FILE] [--trace-hot N (0)] [--trace-tokens N (all)]
//   ds4_lane_bench --synthetic DIR [--syn-layers 2] [--syn-experts 16] [--syn-type IQ2_XXS] [the options above]:
//                  a small fake pack (random legal blocks) written to DIR, then benched as --pack DIR
// Read only on a pack; nothing else written.  No GPU.
#include "../../src/dsv41/ds4_epparts.hpp"
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_os.hpp"
#include "../../src/dsv41/ds4_pool.hpp"

#include "ggml.h"   // the block sizes, the type names

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif
#if defined(__linux__)
#include <cerrno>
#include <cpuid.h>
#include <dirent.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace gg = strata::ds4::gguf;
namespace os = strata::ds4::os;
namespace epx = strata::ds4::ep;
using strata::ds4::Pool;

static int E = 5120, FF = 2304;       // the DeepSeek-V4.1-Flash geometry (--dim, --ff)
static constexpr float kLim = 10.0f;  // swiglu_limit

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::vector<int> ints(const std::string& s) {   // "3,7,9" or "0-39" (or both: "0-3,7")
    std::vector<int> v;
    std::istringstream is(s);
    for (std::string p; std::getline(is, p, ',');) {
        const size_t d = p.find('-');
        if (d == std::string::npos) v.push_back(std::atoi(p.c_str()));
        else for (int i = std::atoi(p.substr(0, d).c_str()); i <= std::atoi(p.substr(d + 1).c_str()); ++i) v.push_back(i);
    }
    return v;
}
static int64_t faults() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pc;
    return K32GetProcessMemoryInfo(GetCurrentProcess(), &pc, sizeof pc) ? (int64_t) pc.PageFaultCount : -1;
#else
    rusage u;
    return getrusage(RUSAGE_SELF, &u) == 0 ? (int64_t) u.ru_minflt : -1;
#endif
}
static long anon_huge_kb() {   // (Linux) AnonHugePages of this process, kB; -1 elsewhere
#if defined(_WIN32)
    return -1;
#else
    std::ifstream f("/proc/self/smaps_rollup");
    for (std::string s; std::getline(f, s);)
        if (s.rfind("AnonHugePages:", 0) == 0) return std::atol(s.c_str() + 14);
    return -1;
#endif
}
static int type_id(std::string n) {
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* tn = ggml_type_name((ggml_type) t);
        if (tn != nullptr && n == tn) return t;
    }
    return -1;
}
static size_t rowb(int type, int k) { return ggml_row_size((ggml_type) type, k); }

// ---- the pack: tensors.txt ("file N name", "t name TYPE file offset bytes ..."), ds4.txt's model_dir
struct Ref {
    int type = -1, file = -1;
    uint64_t off = 0, bytes = 0;
};
struct Pack {
    std::vector<std::string> files;          // their paths
    std::map<long long, Ref> t;              // (layer, expert, matrix) -> where
    std::map<int, int> n_exp;                // layer -> experts
    static long long key(int L, int e, int r) { return ((long long) L * 4096 + e) * 4 + r; }
    const Ref* at(int L, int e, int r) const {
        auto it = t.find(key(L, e, r));
        return it == t.end() ? nullptr : &it->second;
    }
};
static std::string g_from, g_to;   // --remap FROM=TO: a path the pack names under FROM is read under TO
static std::string remap(const std::string& p) {
    return !g_from.empty() && p.rfind(g_from, 0) == 0 ? g_to + p.substr(g_from.size()) : p;
}
static bool read_pack(const std::string& dir, Pack& pk) {
    std::string mdir = dir;
    {
        std::ifstream f(dir + "/ds4.txt");
        for (std::string s; std::getline(f, s);)
            if (s.rfind("model_dir ", 0) == 0) mdir = remap(s.substr(10));
    }
    std::ifstream f(dir + "/tensors.txt");
    if (!f) {
        std::printf("pack: no %s/tensors.txt\n", dir.c_str());
        return false;
    }
    const char* w[3] = {"w1", "w3", "w2"};   // gate, up, down
    for (std::string s; std::getline(f, s);) {
        std::istringstream is(s);
        std::string k;
        is >> k;
        if (k == "file") {
            int n;
            std::string nm;
            is >> n >> nm;
            if ((int) pk.files.size() <= n) pk.files.resize((size_t) n + 1);
            pk.files[(size_t) n] = nm[0] == '/' || (nm.size() > 1 && nm[1] == ':') ? remap(nm) : mdir + "/" + nm;
            continue;
        }
        if (k != "t") continue;
        std::string nm, ty;
        is >> nm;
        int L, e;
        char wn[8] = {0};
        if (std::sscanf(nm.c_str(), "layers.%d.ffn.experts.%d.%2s.weight", &L, &e, wn) != 3) continue;
        int r = -1;
        for (int i = 0; i < 3; ++i) if (std::strcmp(wn, w[i]) == 0) r = i;
        if (r < 0) continue;
        Ref x;
        is >> ty >> x.file >> x.off >> x.bytes;
        x.type = type_id(ty);
        pk.t[Pack::key(L, e, r)] = x;
        pk.n_exp[L] = std::max(pk.n_exp[L], e + 1);
    }
    return !pk.t.empty();
}

// ---- --trace: the calls to replay, by token.  A call: its layer (an index into the bench's layers), its n_ff blocks
// [a, b) (a < 0: the variant's [0, k)) and its experts, a token each.  DS4_TRACE (ds4_model.cu: a decode record u8 layer,
// u8 T (bit 7: the guesses follow - u8 na1, u8 na2, T x (na1 + na2) int16), T x 6 int16 picks; a prompt chunk's u8
// layer | 0x80, u16 T, T x 6 int16): one call a decode record of one token, its picks less the layer's --trace-hot most
// picked.  DS4_LANE_TRACE ("LANETRC1", a call u8 layer, u8 a, u8 b, u8 entries, each int16 expert and u8 tokens): the
// lane's calls as they were.  A token: the records from one whose layer is below the one before it.
struct TCall {
    int li = 0, a = -1, b = -1;
    std::vector<int> e;
};
struct Trace {
    std::string kind;
    std::vector<std::vector<TCall>> tok;
    // the calls read, and those left out: a layer not benched, more than one token (a call or an entry; over 16
    // entries), none left once the hot picks are out; the hot picks left out
    long calls = 0, out_layer = 0, out_multi = 0, out_empty = 0, out_hot = 0;
};
static constexpr int kTraceK = 6;   // (DS4_TRACE's picks a token: the model's k_exp)
static bool read_trace(const std::string& path, const std::map<int, int>& li_of, int hot, Trace& tr, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open";
        return false;
    }
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t at = 0;
    auto u8 = [&](int& v) { return at < s.size() ? (v = (uint8_t) s[at++], true) : false; };
    auto i16 = [&](int& v) {
        if (at + 2 > s.size()) return false;
        v = (int16_t) (uint16_t) ((uint8_t) s[at] | (uint8_t) s[at + 1] << 8);
        at += 2;
        return true;
    };
    int prev = 1 << 30;
    // (a token begins at a layer below the record before it - every decode record counts, the ones left out too)
    auto mark = [&](int layer) {
        if (layer < prev || tr.tok.empty()) tr.tok.emplace_back();
        prev = layer;
    };
    if (s.compare(0, 8, "LANETRC1") == 0) {
        tr.kind = "DS4_LANE_TRACE";
        at = 8;
        for (int L, a, b, ne; u8(L) && u8(a) && u8(b) && u8(ne);) {
            TCall c;
            c.a = a, c.b = b;
            bool multi = ne > 16 || b <= a;
            for (int i = 0, e, n; i < ne; ++i) {
                if (!i16(e) || !u8(n)) {
                    err = "a call cut short at byte " + std::to_string(at);
                    return false;
                }
                multi = multi || n != 1;
                c.e.push_back(e);
            }
            ++tr.calls;
            mark(L);
            const auto li = li_of.find(L);
            if (li == li_of.end()) ++tr.out_layer;
            else if (multi) ++tr.out_multi;
            else if (c.e.empty()) ++tr.out_empty;
            else {
                c.li = li->second;
                tr.tok.back().push_back(std::move(c));
            }
        }
    } else {
        tr.kind = "DS4_TRACE";
        for (int L; u8(L);) {
            int T = 0, x = 0, y = 0;
            if (L & 0x80) {   // (a prompt chunk: not the decode's)
                if (!u8(x) || !u8(y)) break;
                at += (size_t) (x | y << 8) * kTraceK * 2;
                continue;
            }
            if (!u8(T)) break;
            const bool ahead = (T & 0x80) != 0;
            T &= 0x7f;
            TCall c;
            for (int i = 0, e; i < T * kTraceK; ++i) {
                if (!i16(e)) {
                    err = "a record cut short at byte " + std::to_string(at);
                    return false;
                }
                c.e.push_back(e);
            }
            if (ahead) {
                if (!u8(x) || !u8(y)) break;
                at += (size_t) T * (size_t) (x + y) * 2;
            }
            ++tr.calls;
            mark(L);
            const auto li = li_of.find(L);
            if (li == li_of.end()) ++tr.out_layer;
            else if (T != 1) ++tr.out_multi;
            else {
                c.li = li->second;
                tr.tok.back().push_back(std::move(c));
            }
        }
        if (at > s.size()) {
            err = "the last record cut short";
            return false;
        }
        if (hot > 0) {   // each layer's `hot` most picked (ties: the lower expert) left out of its calls
            std::map<std::pair<int, int>, long> n;
            for (const auto& t : tr.tok)
                for (const TCall& c : t)
                    for (int e : c.e) ++n[{c.li, e}];
            std::map<int, std::vector<std::pair<long, int>>> by;
            for (const auto& x : n) by[x.first.first].push_back({-x.second, x.first.second});
            std::map<std::pair<int, int>, bool> out;
            for (auto& x : by) {
                std::sort(x.second.begin(), x.second.end());
                for (size_t i = 0; i < x.second.size() && (int) i < hot; ++i) out[{x.first, x.second[i].second}] = true;
            }
            for (auto& t : tr.tok) {
                for (TCall& c : t) {
                    const size_t n0 = c.e.size();
                    c.e.erase(std::remove_if(c.e.begin(), c.e.end(), [&](int e) { return out.count({c.li, e}) != 0; }), c.e.end());
                    tr.out_hot += (long) (n0 - c.e.size());
                }
                const size_t n0 = t.size();
                t.erase(std::remove_if(t.begin(), t.end(), [](const TCall& c) { return c.e.empty(); }), t.end());
                tr.out_empty += (long) (n0 - t.size());
            }
        }
    }
    tr.tok.erase(std::remove_if(tr.tok.begin(), tr.tok.end(), [](const std::vector<TCall>& t) { return t.empty(); }), tr.tok.end());
    if (tr.tok.empty()) {
        err = "no call of a benched layer in it";
        return false;
    }
    return true;
}

// ---- a fake pack: --syn-layers x --syn-experts experts of --syn-type, random legal blocks (iq_expert_test's generator)
static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u, m = u & 0x7fffffu;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t v = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
static void fill(std::mt19937_64& rng, int type, int rows, int k, uint8_t* p) {
    const size_t bb = ggml_type_size((ggml_type) type), nblk = (size_t) rows * (size_t) (k / ggml_blck_size((ggml_type) type));
    for (size_t i = 0; i < nblk * bb; ++i) p[i] = (uint8_t) rng();
    for (size_t b = 0; b < nblk; ++b) {
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* q = p + b * bb;
        if (type == 29) {   // block_iq1_m: its F16 scale is the top nibbles of its four u16 scales
            q[49] = (uint8_t) ((q[49] & 0x0f) | ((d & 0x000f) << 4));
            q[51] = (uint8_t) ((q[51] & 0x0f) | (d & 0x00f0));
            q[53] = (uint8_t) ((q[53] & 0x0f) | ((d & 0x0f00) >> 4));
            q[55] = (uint8_t) ((q[55] & 0x0f) | ((d & 0xf000) >> 8));
        } else {
            std::memcpy(q, &d, 2);
        }
    }
}
static bool write_synthetic(const std::string& dir, int nl, int ne, const std::string& tname) {
    const int ty = type_id(tname);
    if (ty < 0) {
        std::printf("synthetic: no type %s\n", tname.c_str());
        return false;
    }
    std::ofstream bin(dir + "/experts.bin", std::ios::binary), tt(dir + "/tensors.txt"), d4(dir + "/ds4.txt");
    if (!bin || !tt || !d4) {
        std::printf("synthetic: cannot write into %s\n", dir.c_str());
        return false;
    }
    d4 << "model_dir " << dir << "\n";
    tt << "file 0 experts.bin\n";
    std::mt19937_64 rng(20261026);
    const char* w[3] = {"w1", "w3", "w2"};
    uint64_t off = 0;
    std::vector<uint8_t> buf;
    for (int L = 0; L < nl; ++L)
        for (int e = 0; e < ne; ++e)
            for (int r = 0; r < 3; ++r) {
                const int rows = r < 2 ? FF : E, k = r < 2 ? E : FF;
                buf.assign((size_t) rows * rowb(ty, k), 0);
                fill(rng, ty, rows, k, buf.data());
                bin.write((const char*) buf.data(), (std::streamsize) buf.size());
                tt << "t layers." << L << ".ffn.experts." << e << "." << w[r] << ".weight " << tname << " 0 " << off << " "
                   << buf.size() << "\n";
                off += buf.size();
                const uint64_t pad = (4096 - off % 4096) % 4096;   // (each matrix page-aligned, as a pack's are)
                if (pad > 0) {
                    buf.assign((size_t) pad, 0);
                    bin.write((const char*) buf.data(), (std::streamsize) pad);
                    off += pad;
                }
            }
    std::printf("synthetic pack: %d layers x %d experts of %s, %.1f MB in %s\n", nl, ne, tname.c_str(), (double) off / 1e6,
                dir.c_str());
    return true;
}

// ---- where the experts' bytes are read from: the files mapped, or copies in anonymous memory
struct Maps {   // every pack file mapped read-only
    std::vector<os::fd_t> fd;
    std::vector<uint8_t*> p;
    std::vector<uint64_t> n;
    bool open(const Pack& pk) {
        for (const std::string& f : pk.files) {
            const os::fd_t d = f.empty() ? os::kBadFd : os::open_ro(f);
            const uint64_t b = d == os::kBadFd ? 0 : os::file_size(d);
            uint8_t* m = b > 0 ? (uint8_t*) os::map_ro(d, b) : nullptr;
            if (!f.empty() && m == nullptr) {
                std::printf("cannot map %s\n", f.c_str());
                return false;
            }
            fd.push_back(d);
            p.push_back(m);
            n.push_back(b);
        }
        return true;
    }
    ~Maps() {
        for (size_t i = 0; i < p.size(); ++i) {
            os::unmap(p[i], n[i]);
            if (fd[i] != os::kBadFd) os::close_fd(fd[i]);
        }
    }
    const uint8_t* at(const Ref& r) const { return p[(size_t) r.file] + r.off; }
};
struct Anon {   // copies of a set of matrices, each 4 KiB aligned, on 4 KiB or huge pages
    uint8_t* base = nullptr;
    size_t bytes = 0;
    bool make(size_t n, bool huge) {
        bytes = (n + (2u << 20) - 1) / (2u << 20) * (2u << 20) + (2u << 20);
#if defined(_WIN32)
        (void) huge;
        base = (uint8_t*) VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* m = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        base = m == MAP_FAILED ? nullptr : (uint8_t*) m;
        if (base != nullptr) madvise(base, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
#endif
        return base != nullptr;
    }
    uint8_t* aligned() const { return (uint8_t*) (((uintptr_t) base + (2u << 20) - 1) & ~(uintptr_t) ((2u << 20) - 1)); }
    ~Anon() {
#if defined(_WIN32)
        if (base != nullptr) VirtualFree(base, 0, MEM_RELEASE);
#else
        if (base != nullptr) ::munmap(base, bytes);
#endif
    }
};

// ---- the logical CPUs in the order the threads take them: each core's first (Windows: os::node_cores' order; Linux:
// this process's affinity), then the second hardware threads (SMT) in their cores' order
static void cpu_order(std::vector<int>& phys, std::vector<int>& smt) {
    std::vector<int> all;
#if defined(_WIN32)
    for (const auto& n : os::node_cores(false)) phys.insert(phys.end(), n.begin(), n.end());
    for (const auto& n : os::node_cores(true)) all.insert(all.end(), n.begin(), n.end());
#else
    cpu_set_t s;
    if (sched_getaffinity(0, sizeof s, &s) == 0)
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &s)) all.push_back(c);
    for (int c : all) {
        const std::vector<int> sib = os::cpu_siblings(c);   // (sorted: the core's first is the smallest)
        if (sib.empty() || sib[0] == c || std::find(all.begin(), all.end(), sib[0]) == all.end()) phys.push_back(c);
    }
#endif
    if (phys.empty()) phys.push_back(0);
    for (int c : all)
        if (std::find(phys.begin(), phys.end(), c) == phys.end()) smt.push_back(c);
    auto core_at = [&](int c) {
        const std::vector<int> sib = os::cpu_siblings(c);
        const auto it = std::find(phys.begin(), phys.end(), sib.empty() ? c : sib[0]);
        return (int) (it - phys.begin());
    };
    std::stable_sort(smt.begin(), smt.end(), [&](int a, int b) { return core_at(a) < core_at(b); });
}

// ---- :pool=fix - the engine's Pool (ds4_pool.hpp) with what a batch writes to shared lines cut down: every flag on a
// line of its own (the Pool's sleeping_ / quit_ share done_'s line, which every job's fetch_add writes while the idle
// workers poll quit_), each worker's jobs counted locally and added to done_ once when it runs out (one write a worker
// a batch instead of one a job); the ticket claimed by CAS as the Pool claims it (a late worker of an older batch
// must not take a job of this one).  t_slot: the thread's index (the caller 0), for the jobs' busy time a slot each.
static thread_local int t_slot = 0;
class FixPool {
public:
    explicit FixPool(int n, const std::vector<int>& cpus, int spin_us) : spin_us_(spin_us) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this, i] {
                t_slot = i + 1;
                loop(0);
            });
            if (!cpus.empty()) os::pin_thread(th_.back(), cpus[(size_t) i % cpus.size()]);
        }
    }
    ~FixPool() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    void start(int n, const std::function<void(int)>& fn) {
        n_ = n;
        if (n <= 0) return;
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
        ticket_.store(g << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_seq_cst);
        if (sleeping_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void help() {
        if (n_ > 0) work(gen_.load(std::memory_order_acquire));
    }
    void wait() {
        if (n_ <= 0) return;
        while (done_.load(std::memory_order_acquire) < n_) strata::ds4::cpu_relax();
        n_ = 0;
    }

private:
    void work(uint64_t g) {
        int mine = 0;
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) break;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) break;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            ++mine;
        }
        if (mine > 0) done_.fetch_add(mine, std::memory_order_acq_rel);
    }
    void loop(uint64_t seen) {
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t g;
            for (;;) {
                g = gen_.load(std::memory_order_acquire);
                if (g != seen || quit_.load(std::memory_order_relaxed)) break;
                strata::ds4::cpu_relax();
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
    int n_ = 0;
    const std::function<void(int)>* fn_ = nullptr;
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> total_{0};
    alignas(64) std::atomic<int> done_{0};
    alignas(64) std::atomic<int> sleeping_{0};
    alignas(64) std::atomic<bool> quit_{false};
    alignas(64) std::mutex mu_;
    std::condition_variable cv_;
};

// ---- the core's counters (Linux perf_event_open), every thread of this process counted on its own (pid = its tid,
// any CPU), user mode only, from start() to stop().  The events come in groups of at most 5, each led by cycles (a
// group is counted together, so an event's share of its group's cycles is exact; the kernel time-shares the groups on
// the 6 counters - one may be the NMI watchdog's - and a group's counts are scaled by its enabled / running time).  The
// raw events are AMD family 0x19's (Zen 3; the PPR's PMCx numbers - the names after perf's amdzen3 lists, not checked
// against the PPR here); another CPU gets cycles and instructions only.
struct PmcEv {
    const char* name;
    uint32_t type;
    uint64_t cfg;
};
#if defined(__linux__)
static uint64_t amd_raw(unsigned ev, unsigned umask) { return (ev & 0xffu) | ((uint64_t) umask << 8) | ((uint64_t) (ev >> 8) << 32); }
static bool amd_zen3() {
    unsigned a, b, c, d;
    if (!__get_cpuid(0, &a, &b, &c, &d)) return false;
    char v[13];
    std::memcpy(v, &b, 4), std::memcpy(v + 4, &d, 4), std::memcpy(v + 8, &c, 4);
    v[12] = 0;
    if (std::strcmp(v, "AuthenticAMD") != 0 || !__get_cpuid(1, &a, &b, &c, &d)) return false;
    const unsigned base = (a >> 8) & 0xf, fam = base == 0xf ? base + ((a >> 20) & 0xff) : base;
    return fam == 0x19;
}
// the groups (each after its cycles leader)
static std::vector<std::vector<PmcEv>> pmc_groups() {
    std::vector<std::vector<PmcEv>> g = {{{"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS}}};
    if (!amd_zen3()) return g;
    const struct {
        int grp;
        const char* n;
        unsigned ev, um;
    } raw[] = {
        {0, "ret-mops", 0xc1, 0x00},          // retired macro-ops
        {0, "op-src-decoder", 0xaa, 0x01},    // ops dispatched from the decoder
        {0, "op-src-opcache", 0xaa, 0x02},    // .. from the op cache
        {1, "AE.01 int-prf", 0xae, 0x01},     // dispatch stall cycles: integer physical registers
        {1, "AE.02 load-queue", 0xae, 0x02},
        {1, "AE.04 store-queue", 0xae, 0x04},
        {1, "AE.08 int-sched-misc", 0xae, 0x08},
        {2, "AE.10 taken-br-buf", 0xae, 0x10},
        {2, "AE.20 fp-prf", 0xae, 0x20},
        {2, "AE.40 fp-sched", 0xae, 0x40},
        {2, "AE.80 fp-misc", 0xae, 0x80},
        {3, "AF.01 int-sq0", 0xaf, 0x01},     // .. an integer scheduler queue full
        {3, "AF.02 int-sq1", 0xaf, 0x02},
        {3, "AF.04 int-sq2", 0xaf, 0x04},
        {3, "AF.08 int-sq3", 0xaf, 0x08},
        {4, "AF.10 agsq", 0xaf, 0x10},
        {4, "AF.20 retire", 0xaf, 0x20},
        {5, "x000.01 fp-pipe0", 0x00, 0x01},  // FP pipe assignment (Zen 2's PMCx000; on Zen 3 it may read 0)
        {5, "x000.02 fp-pipe1", 0x00, 0x02},
        {5, "x000.04 fp-pipe2", 0x00, 0x04},
        {5, "x000.08 fp-pipe3", 0x00, 0x08},
        {6, "44.01 fill-l2", 0x44, 0x01},     // L1D fills by source: the core's L2
        {6, "44.02 fill-ccx", 0x44, 0x02},    // .. another L2 or the L3 of the CCX
        {6, "44.04 fill-ext", 0x44, 0x04},    // .. another CCX's cache
        {6, "44.08 fill-dram", 0x44, 0x08},   // .. DRAM or IO (the node's)
        {7, "64.08 l2-dc-miss", 0x64, 0x08},  // L2: data cache read misses
        {7, "71 l2pf-l3-hit", 0x71, 0xff},    // L2 prefetches: missed the L2, hit the L3
        {7, "72 l2pf-l3-miss", 0x72, 0xff},   // .. missed the L3 too
    };
    g.resize(8);
    for (const auto& r : raw) g[(size_t) r.grp].push_back({r.n, PERF_TYPE_RAW, amd_raw(r.ev, r.um)});
    return g;
}
struct Pmc {
    std::vector<std::vector<PmcEv>> grp;
    std::vector<std::vector<std::vector<int>>> fd;   // [thread][group][0: the cycles leader, then the group's events]
    std::string err;
    static long open1(uint32_t type, uint64_t cfg, int tid, int leader) {
        perf_event_attr a;
        std::memset(&a, 0, sizeof a);
        a.size = sizeof a;
        a.type = type;
        a.config = cfg;
        a.disabled = leader < 0 ? 1 : 0;   // (the leader starts and stops its group)
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        a.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        return syscall(SYS_perf_event_open, &a, tid, -1, leader, PERF_FLAG_FD_CLOEXEC);
    }
    // every thread of the process now (the pool's are running): false with err if not even the first counter opens; an
    // event this CPU or kernel refuses is left out of its group (-1)
    bool open() {
        grp = pmc_groups();
        std::vector<int> tids;
        if (DIR* d = opendir("/proc/self/task")) {
            while (dirent* x = readdir(d))
                if (x->d_name[0] >= '0' && x->d_name[0] <= '9') tids.push_back(std::atoi(x->d_name));
            closedir(d);
        }
        for (int tid : tids) {
            std::vector<std::vector<int>> t;
            for (const auto& g : grp) {
                std::vector<int> f(g.size() + 1, -1);
                const long l = open1(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, tid, -1);
                if (l < 0 && fd.empty() && t.empty()) {
                    err = "errno " + std::to_string(errno) + ", " + std::strerror(errno);
                    return false;
                }
                f[0] = (int) l;
                for (size_t i = 0; i < g.size() && l >= 0; ++i) f[i + 1] = (int) open1(g[i].type, g[i].cfg, tid, (int) l);
                t.push_back(f);
            }
            fd.push_back(t);
        }
        return !fd.empty();
    }
    void ctl(unsigned long op) {
        for (auto& t : fd)
            for (auto& f : t)
                if (f[0] >= 0) ioctl(f[0], op, PERF_IOC_FLAG_GROUP);
    }
    void start() {
        ctl(PERF_EVENT_IOC_RESET);
        ctl(PERF_EVENT_IOC_ENABLE);
    }
    void stop() { ctl(PERF_EVENT_IOC_DISABLE); }
    // per group: [0] its cycles, then its events, summed over the threads, each thread's scaled by its enabled / running
    // time; running: the share of the groups' enabled time they were counted, never: the thread-groups never counted
    // (their counts lost: a run shorter than a few time-sharing ticks)
    std::vector<std::vector<double>> read(double& running, int& never) {
        std::vector<std::vector<double>> s(grp.size());
        double en = 0, ru = 0;
        never = 0;
        for (size_t k = 0; k < grp.size(); ++k) s[k].assign(grp[k].size() + 1, -1.0);
        for (auto& t : fd)
            for (size_t k = 0; k < t.size(); ++k) {
                const std::vector<int>& f = t[k];
                if (f[0] < 0) continue;
                uint64_t r[3 + 8];   // nr, enabled, running, the values of the events opened (leader first)
                const ssize_t got = ::read(f[0], r, sizeof r);
                if (got < (ssize_t) (3 * sizeof(uint64_t))) continue;
                const double sc = r[2] > 0 ? (double) r[1] / (double) r[2] : 0.0;
                en += (double) r[1];
                ru += (double) r[2];
                if (r[1] > 0 && r[2] == 0) ++never;
                size_t v = 3;
                for (size_t i = 0; i < f.size() && v < 3 + (size_t) r[0]; ++i) {
                    if (f[i] < 0) continue;   // (not opened: not in the group's values)
                    if (s[k][i] < 0) s[k][i] = 0;
                    s[k][i] += (double) r[v++] * sc;
                }
            }
        running = en > 0 ? ru / en : 0.0;
        return s;
    }
    ~Pmc() {
        for (auto& t : fd)
            for (auto& f : t)
                for (size_t i = f.size(); i-- > 0;)
                    if (f[i] >= 0) close(f[i]);
    }
};
#endif

// ---- the bench
struct Layer {
    int L = 0, tg = -1, td = -1;
    gg::IqCpu fm;
    std::vector<int> pool, fresh;   // its experts: the warm pool (rotated), the fresh ones (each used once)
};
struct Exp {
    const uint8_t* m[3];
    int id = -1;
    const uint8_t* dp = nullptr;   // (:dparts) its down part's row 0 at block dpa, the part [dpa, dpb); null: none
    int dpa = 0, dpb = 0;
};
enum Pf { kPfNone, kPfWillneed, kPfPopulate };
struct Acc {
    double prep = 0, gu = 0, gu_tail = 0, gu_busy = 0, hq = 0, dn = 0, dn_tail = 0, dn_busy = 0, gu_bytes = 0, dn_bytes = 0;
    int batches = 0;
    long calls = 0, picks = 0, downs_dp = 0, hinted = 0;   // (the picks' downs from the parts; the experts hinted)
    long hist[17] = {};                                    // calls by their picks
};
// (:spin=) a thread a CPU loading a flag until the variant's timed tokens end - started after the counters open, so
// they count the lane's threads alone
struct Spin {
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    explicit Spin(const std::vector<int>& cpus) {
        for (int c : cpus) {
            th.emplace_back([this] {
                while (!stop.load(std::memory_order_relaxed)) {
                }
            });
            os::pin_thread(th.back(), c);
        }
    }
    ~Spin() {
        stop.store(true);
        for (auto& t : th) t.join();
    }
};
struct alignas(64) Slot {   // (:pool=fix) a thread's jobs' busy time, a line each
    int64_t ns = 0;
};
struct Lane {   // the lane's pool and buffers (Impl's c_xh / c_xhd / c_yd)
    std::unique_ptr<Pool> pool;
    std::unique_ptr<FixPool> fpool;   // (:pool=fix: this one instead)
    std::vector<Slot> slot;
    int threads = 1;
    bool empty = false;   // (:empty: the jobs return at once)
    int pf_age = 0;       // (:pf=N) the hint on the experts the layer's last N calls did not take only
    std::map<long long, uint32_t> seen;   // (layer, expert) -> the layer's call count at its last call
    std::map<int, uint32_t> calls;        // layer -> its calls
    std::vector<float> x, h, y;
    std::vector<uint8_t> xq, hq;
    // run_nodes_prof's batch: wall, tail (the caller's wait once it ran out of jobs), busy (the jobs' time in all)
    void batch(int n, const std::function<void(int)>& f, double& wall, double& tail, double& busy, int& batches) {
        std::atomic<int64_t> ns{0};
        const std::function<void(int)> g = [&](int j) {
            const auto a = std::chrono::steady_clock::now();
            if (!empty) f(j);
            ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count(),
                         std::memory_order_relaxed);
        };
        const std::function<void(int)> gs = [&](int j) {   // (FixPool: no shared write a job)
            const auto a = std::chrono::steady_clock::now();
            if (!empty) f(j);
            slot[(size_t) t_slot].ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count();
        };
        for (Slot& s : slot) s.ns = 0;
        const double t0 = now_ms();
        double th;
        if (fpool) {
            fpool->start(n, gs);
            fpool->help();
            th = now_ms();
            fpool->wait();
        } else {
            pool->start(n, g);
            pool->help();
            th = now_ms();
            pool->wait();
        }
        const double t1 = now_ms();
        wall += t1 - t0;
        tail += t1 - th;
        int64_t sum = ns.load();
        for (const Slot& s : slot) sum += s.ns;
        busy += (double) sum / 1e6;
        ++batches;
    }
    // one call: ne experts (one token each, ne <= 16) over n_ff [ff0, ff0 + ffn); dparts: each expert's down from its
    // part when the part ends where the slice ends (Impl's DS4_IQ_DOWN_PARTS), else the pack's
    void call(const Layer& ly, const Exp* ex, int ne, int ff0, int ffn, int rb, bool split, Pf pf, bool dparts, Acc& a) {
        const gg::IqCpu& fm = ly.fm;
        gg::IqCpu fs;
        size_t d_off = 0;
        std::string err;
        const bool sliced = ff0 != 0 || ffn != FF;
        if (sliced) gg::iq_cpu_slice(fm, ff0, ffn, fs, d_off, err);
        const gg::IqCpu& fd = sliced ? fs : fm;
        const size_t ub = fm.d_row / (size_t) (FF / 256);   // (a down row's bytes a block)
        const uint8_t* dn[16];
        bool usep[16];
        gg::IqCpu fq[16];
        for (int q = 0; q < ne; ++q) {
            const Exp& x = ex[q];
            const bool p = dparts && x.dp != nullptr && ff0 / 256 >= x.dpa && ff0 + ffn == 256 * x.dpb;
            fq[q] = fd;
            usep[q] = p;
            dn[q] = x.m[2] + d_off;
            if (p) {
                fq[q].d_stride = (size_t) (x.dpb - x.dpa) * ub;
                dn[q] = x.dp + (size_t) (ff0 / 256 - x.dpa) * ub;
                ++a.downs_dp;
            }
        }
        ++a.calls;
        a.picks += ne;
        ++a.hist[std::min(ne, 16)];
        const double t0 = now_ms();
        if (pf != kPfNone) {   // (Impl's: a slice's gate / up rows, the down's whole - its rows hold every slice - or its part's)
            const uint32_t tick = pf_age > 0 ? ++calls[ly.L] : 0;
            std::vector<std::pair<const void*, size_t>> want;
            for (int q = 0; q < ne; ++q) {
                if (pf_age > 0) {   // (:pf=N: only an expert the layer's last N calls did not take)
                    uint32_t& s = seen[(long long) ly.L * 65536 + ex[q].id];
                    const bool stale = s == 0 || tick - s > (uint32_t) pf_age;
                    s = tick;
                    if (!stale) continue;
                }
                ++a.hinted;
                for (int r = 0; r < 2; ++r) want.push_back({ex[q].m[r] + (size_t) ff0 * fm.gu_row, (size_t) ffn * fm.gu_row});
                if (usep[q]) want.push_back({dn[q], (size_t) (E - 1) * fq[q].d_stride + fd.d_row});
                else want.push_back({ex[q].m[2], (size_t) E * fm.d_row});
            }
            if (pf == kPfWillneed) os::prefetch(want);
#if !defined(_WIN32)
            else   // MADV_POPULATE_READ (22, Linux 5.14): the pages mapped now, as a read fault would map them
                for (const auto& w : want) {
                    const uintptr_t b = (uintptr_t) w.first & ~(uintptr_t) 4095;
                    madvise((void*) b, w.second + ((uintptr_t) w.first - b), 22);
                }
#endif
        }
        gg::iq_cpu_quant(fm.gu_act, x.data(), E, xq.data());
        a.prep += now_ms() - t0;
        const int GB = split ? threads : (ffn + rb - 1) / rb, DB = split ? threads : (E + rb - 1) / rb;
        const int grs = (ffn + GB - 1) / GB, drs = (E + DB - 1) / DB;   // (rows a job)
        const void* ap[1] = {xq.data()};
        batch(ne * GB, [&](int j) {   // (rows of n_ff, h's indices: [ff0, ff0 + ffn))
            const int q = j / GB, r0 = ff0 + (j % GB) * grs, r1 = std::min(ff0 + ffn, r0 + grs);
            if (r0 >= r1) return;
            float* hp[1] = {h.data() + (size_t) q * FF};
            gg::iq_cpu_gate_up(fm, ex[q].m[0], ex[q].m[1], ap, 1, hp, r0, r1, kLim);
        }, a.gu, a.gu_tail, a.gu_busy, a.batches);
        const double th = now_ms();
        for (int q = 0; q < ne; ++q)
            gg::iq_cpu_quant(fm.d_act, h.data() + (size_t) q * FF + ff0, ffn, hq.data() + (size_t) q * E * 4);
        a.hq += now_ms() - th;
        batch(ne * DB, [&](int j) {
            const int q = j / DB, r0 = (j % DB) * drs, r1 = std::min(E, r0 + drs);
            if (r0 >= r1) return;
            const void* hp[1] = {hq.data() + (size_t) q * E * 4};
            float* yp[1] = {y.data() + (size_t) q * E};
            gg::iq_cpu_down(fq[q], dn[q], hp, 1, yp, r0, r1);
        }, a.dn, a.dn_tail, a.dn_busy, a.batches);
        if (empty) return;   // (nothing read)
        a.gu_bytes += (double) ne * 2.0 * (double) ffn * (double) fm.gu_row;
        a.dn_bytes += (double) ne * (double) E * (double) fd.d_row;
    }
};

struct Variant {
    std::string name, mode;
    int k = 6, picks = 4, rb = 64, thr = 0;
    bool split = false, empty = false, fix = false, dparts = false, trace = false, cold = false;
    int pf_age = 0;          // (:pf=N)
    std::vector<int> skip;   // (:skip=2+18: those CPUs not used)
    std::vector<int> spin;   // (:spin=16+18: a spinning thread on each)
};
static void plus_list(const std::string& s, std::vector<int>& v) {   // "2+18"
    std::istringstream ss(s);
    for (std::string c; std::getline(ss, c, '+');)
        if (!c.empty()) v.push_back(std::atoi(c.c_str()));
}
static bool parse_variant(const std::string& s, const Variant& def, Variant& v) {
    v = def;
    v.name = s;
    std::istringstream is(s);
    std::string p;
    std::getline(is, v.mode, ':');
    while (std::getline(is, p, ':')) {
        if (p.rfind("k=", 0) == 0) v.k = std::atoi(p.c_str() + 2);
        else if (p.rfind("picks=", 0) == 0) v.picks = std::atoi(p.c_str() + 6);
        else if (p.rfind("rb=", 0) == 0) v.rb = std::atoi(p.c_str() + 3);
        else if (p.rfind("thr=", 0) == 0) v.thr = std::atoi(p.c_str() + 4);
        else if (p == "split") v.split = true;
        else if (p == "empty") v.empty = true;
        else if (p == "pool=fix") v.fix = true;
        else if (p == "dparts") v.dparts = true;
        else if (p == "trace") v.trace = true;
        else if (p == "cold") v.cold = true;
        else if (p.rfind("pf=", 0) == 0) v.pf_age = std::atoi(p.c_str() + 3);
        else if (p.rfind("skip=", 0) == 0) plus_list(p.substr(5), v.skip);
        else if (p.rfind("spin=", 0) == 0) plus_list(p.substr(5), v.spin);
        else return false;
    }
    const char* modes[] = {"file", "file-nopf", "file-fresh", "file-populate", "anon4k", "anon2m", "hot"};
    bool known = false;
    for (const char* m : modes) known = known || v.mode == m;
    return known && v.k >= 1 && v.k <= 9 && v.picks >= 1 && v.rb >= 1;
}

// (:dparts) expert e's down part in `p` - the first that holds a block, of the layer's type and row layout (one row's
// bytes a block the pack's) - into x.dp / dpa / dpb; none: x.dp stays null (its downs read from the pack)
static void down_part(const epx::Parts& p, const Layer& ly, int e, Exp& x) {
    epx::View v;
    for (int k = 0; k < FF / 256; ++k)
        if (p.view(ly.L, e, epx::kDown, k, v)) {   // (the first block found is the part's a: v.data its row 0 there)
            if (v.type == ly.td && v.rows == (uint32_t) E && v.stride == (uint64_t) (v.b - v.a) * (ly.fm.d_row / (size_t) (FF / 256)))
                x.dp = v.data, x.dpa = v.a, x.dpb = v.b;
            return;
        }
}

int main(int argc, char** argv) {
    std::string pack, syn, syn_type = "IQ2_XXS", vlist =
        "file,file:k=9,file-nopf,file-fresh,file-populate,anon4k,anon2m,file:picks=1,hot:picks=1,file:k=9:picks=8,"
        "hot:k=9:picks=8,file:rb=128,file:rb=256,file:split,file:thr=1,file:thr=2,file:thr=4,file:thr=8,file:thr=12,"
        "file:thr=24,file:thr=30:skip=2+18,file:thr=32,hot,hot:thr=1,hot:thr=8,hot:thr=24,hot:thr=30:skip=2+18,hot:thr=32,"
        "file:empty,file:empty:pool=fix,file:pool=fix,hot:pool=fix";
    std::string layers_s, dparts_dir, trace_path;
    int syn_l = 2, syn_e = 16, pool_n = 8, tokens = 16, fresh_tokens = 4, threads = 0, trace_hot = 0, trace_tokens = 0;
    bool want_pmc = true;
    Variant def;
    def.mode = "file";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool v = i + 1 < argc;
        if (a == "--pack" && v) pack = argv[++i];
        else if (a == "--synthetic" && v) syn = argv[++i];
        else if (a == "--syn-layers" && v) syn_l = std::atoi(argv[++i]);
        else if (a == "--syn-experts" && v) syn_e = std::atoi(argv[++i]);
        else if (a == "--syn-type" && v) syn_type = argv[++i];
        else if (a == "--layers" && v) layers_s = argv[++i];
        else if (a == "--picks" && v) def.picks = std::atoi(argv[++i]);
        else if (a == "--k" && v) def.k = std::atoi(argv[++i]);
        else if (a == "--pool" && v) pool_n = std::atoi(argv[++i]);
        else if (a == "--tokens" && v) tokens = std::atoi(argv[++i]);
        else if (a == "--fresh-tokens" && v) fresh_tokens = std::atoi(argv[++i]);
        else if (a == "--threads" && v) threads = std::atoi(argv[++i]);
        else if (a == "--variants" && v) vlist = argv[++i];
        else if (a == "--dim" && v) E = std::atoi(argv[++i]);
        else if (a == "--ff" && v) FF = std::atoi(argv[++i]);
        else if (a == "--remap" && v && std::strchr(argv[i + 1], '=') != nullptr) {
            const std::string m = argv[++i];
            g_from = m.substr(0, m.find('='));
            g_to = m.substr(m.find('=') + 1);
        } else if (a == "--no-pmc") want_pmc = false;
        else if (a == "--down-parts" && v) dparts_dir = argv[++i];
        else if (a == "--trace" && v) trace_path = argv[++i];
        else if (a == "--trace-hot" && v) trace_hot = std::atoi(argv[++i]);
        else if (a == "--trace-tokens" && v) trace_tokens = std::atoi(argv[++i]);
        else {
            std::printf("usage: ds4_lane_bench --pack DIR | --synthetic DIR [--layers 0-39] [--picks 4] [--k 6] [--pool 8] "
                        "[--tokens 16] [--fresh-tokens 4] [--threads N] [--variants a,b,..] [--dim 5120] [--ff 2304] "
                        "[--remap FROM=TO] [--no-pmc] [--down-parts DIR] [--trace FILE] [--trace-hot N] [--trace-tokens N]\n");
            return 2;
        }
    }
#if defined(__linux__)
    if (want_pmc) {   // (a counter a thread an event: 32 threads x 30 events need more than the usual 1024 descriptors)
        rlimit r;
        if (getrlimit(RLIMIT_NOFILE, &r) == 0 && r.rlim_cur < r.rlim_max) {
            r.rlim_cur = r.rlim_max;
            setrlimit(RLIMIT_NOFILE, &r);
        }
    }
#endif
    if (!syn.empty()) {
        if (!write_synthetic(syn, syn_l, syn_e, syn_type)) return 2;
        pack = syn;
    }
    Pack pk;
    if (pack.empty() || !read_pack(pack, pk)) {
        std::printf("no pack (--pack DIR or --synthetic DIR)\n");
        return 2;
    }
    std::vector<Variant> vs;
    {
        std::istringstream is(vlist);
        for (std::string s; std::getline(is, s, ',');) {
            Variant v;
            if (!parse_variant(s, def, v)) {
                std::printf("variant %s: a mode (file file-nopf file-fresh file-populate anon4k anon2m hot), then :k= :picks= "
                            ":rb= :thr= :split :skip=A+B :empty :pool=fix :dparts :pf=N :spin=A+B :trace :cold\n", s.c_str());
                return 2;
            }
            vs.push_back(v);
        }
    }
    int max_picks = 1;
    for (const Variant& v : vs) max_picks = std::max(max_picks, v.picks);
    pool_n = std::max(pool_n, max_picks);
    // the layers: every one with experts (--layers: those), their types, the format, the experts used
    std::vector<int> want = layers_s.empty() ? std::vector<int>() : ints(layers_s);
    if (want.empty())
        for (const auto& x : pk.n_exp) want.push_back(x.first);
    std::vector<Layer> ly;
    auto exp_ok = [&](const Layer& x, int e) {   // its three matrices in the pack, of the layer's types and sizes
        for (int r = 0; r < 3; ++r) {
            const Ref* t = pk.at(x.L, e, r);
            const uint64_t want_b = r < 2 ? (uint64_t) FF * rowb(x.tg, E) : (uint64_t) E * rowb(x.td, FF);
            if (t == nullptr || t->bytes != want_b || t->type != (r < 2 ? x.tg : x.td)) {
                std::printf("layer %d expert %d matrix %d: missing, or not %llu bytes of the layer's type\n", x.L, e, r,
                            (unsigned long long) want_b);
                return false;
            }
        }
        return true;
    };
    for (int L : want) {
        const auto ne = pk.n_exp.find(L);
        if (ne == pk.n_exp.end()) {
            std::printf("layer %d: no experts in the pack\n", L);
            return 2;
        }
        Layer x;
        x.L = L;
        const Ref* g = pk.at(L, 0, 0);
        const Ref* d = pk.at(L, 0, 2);
        if (g == nullptr || d == nullptr) {
            std::printf("layer %d: expert 0 not whole\n", L);
            return 2;
        }
        x.tg = g->type;
        x.td = d->type;
        std::string err;
        if (!gg::iq_cpu_fmt(x.tg, x.td, E, FF, x.fm, err)) {
            std::printf("layer %d: %s\n", L, err.c_str());
            return 2;
        }
        const int n = ne->second, fn = std::max(0, std::min(fresh_tokens * max_picks, n - pool_n));
        if (n < pool_n) {
            std::printf("layer %d: %d experts, fewer than the pool's %d\n", L, n, pool_n);
            return 2;
        }
        int step = 29;   // (spread over the layer's experts: a step prime to their count)
        while (std::gcd(step, n) != 1) ++step;
        for (int j = 0; j < pool_n + fn; ++j) {
            const int e = (L * 7 + j * step) % n;
            if (!exp_ok(x, e)) return 2;
            (j < pool_n ? x.pool : x.fresh).push_back(e);
        }
        ly.push_back(x);
    }
    for (const Layer& x : ly) fresh_tokens = std::min(fresh_tokens, (int) (x.fresh.size() / (size_t) max_picks));
    if (max_picks > 16) {
        std::printf("--picks / :picks= up to 16\n");
        return 2;
    }
    // --trace: its calls (each expert checked against the pack), the first --trace-tokens tokens of them
    Trace tr;
    int trace_k = 0;   // (the largest :k= of the :trace variants: a DS4_TRACE call's [0, k))
    for (const Variant& v : vs)
        if (v.trace) trace_k = std::max(trace_k, v.k);
    if (!trace_path.empty()) {
        std::map<int, int> li_of;
        for (size_t i = 0; i < ly.size(); ++i) li_of[ly[i].L] = (int) i;
        std::string err;
        if (!read_trace(trace_path, li_of, trace_hot, tr, err)) {
            std::printf("--trace %s: %s\n", trace_path.c_str(), err.c_str());
            return 2;
        }
        if (trace_tokens > 0 && (int) tr.tok.size() > trace_tokens) tr.tok.resize((size_t) trace_tokens);
        std::map<long long, bool> ok;
        for (const auto& t : tr.tok)
            for (const TCall& c : t) {
                const Layer& x = ly[(size_t) c.li];
                if (c.a >= 0 && (c.b > FF / 256 || c.a >= c.b)) {
                    std::printf("--trace: layer %d: a call over the n_ff blocks [%d, %d) of %d\n", x.L, c.a, c.b, FF / 256);
                    return 2;
                }
                for (int e : c.e) {
                    if (e < 0 || e >= pk.n_exp[x.L]) {
                        std::printf("--trace: layer %d: expert %d, the pack has %d\n", x.L, e, pk.n_exp[x.L]);
                        return 2;
                    }
                    const long long key = (long long) c.li * 65536 + e;
                    if (ok.count(key) == 0 && !exp_ok(x, e)) return 2;
                    ok[key] = true;
                }
            }
    }
    // --down-parts: checked here and read by the warm read; each :dparts variant maps its own (a new mapping, as Maps)
    epx::Parts gdp;
    if (!dparts_dir.empty()) {
        std::string err;
        if (!gdp.open(dparts_dir, err)) {
            std::printf("--down-parts %s: %s\n", dparts_dir.c_str(), err.c_str());
            return 2;
        }
        if (gdp.dim() != E || gdp.n_ff() != FF || gdp.unit() != 256) {
            std::printf("--down-parts %s: dim %d, n_ff %d, unit %d (the bench's %d, %d, 256)\n", dparts_dir.c_str(), gdp.dim(),
                        gdp.n_ff(), gdp.unit(), E, FF);
            return 2;
        }
        // each layer's first pool expert's down over the parts' slices [a, b) and [a + 1, b): from the parts as :dparts
        // (and DS4_IQ_DOWN_PARTS) read them, and from the pack in place - the same bytes, so the same bits
        Maps cm;
        if (!cm.open(pk)) return 2;
        std::mt19937 crng(11);
        std::vector<float> hv((size_t) FF), y0((size_t) E), y1((size_t) E);
        std::vector<uint8_t> hq((size_t) FF * 4);
        int checked = 0;
        for (const Layer& x : ly) {
            Exp d;
            down_part(gdp, x, x.pool[0], d);
            if (d.dp == nullptr) continue;
            const size_t ub = x.fm.d_row / (size_t) (FF / 256);
            for (int k = d.dpa; k < d.dpb && k <= d.dpa + 1; ++k) {
                gg::IqCpu fs, fp;
                size_t d_off = 0;
                std::string err;
                if (!gg::iq_cpu_slice(x.fm, 256 * k, 256 * (d.dpb - k), fs, d_off, err)) continue;
                fp = fs;
                fp.d_stride = (size_t) (d.dpb - d.dpa) * ub;
                for (float& f : hv) f = ((float) (crng() % 20001) - 10000.0f) * 1e-4f;
                gg::iq_cpu_quant(x.fm.d_act, hv.data(), 256 * (d.dpb - k), hq.data());
                const void* hp[1] = {hq.data()};
                float* p0[1] = {y0.data()};
                float* p1[1] = {y1.data()};
                gg::iq_cpu_down(fs, cm.at(*pk.at(x.L, x.pool[0], 2)) + d_off, hp, 1, p0, 0, E);
                gg::iq_cpu_down(fp, d.dp + (size_t) (k - d.dpa) * ub, hp, 1, p1, 0, E);
                if (std::memcmp(y0.data(), y1.data(), (size_t) E * 4) != 0) {
                    std::printf("--down-parts: layer %d expert %d, the slice [%d, %d): the parts' down is not the pack's\n", x.L,
                                x.pool[0], k, d.dpb);
                    return 2;
                }
                ++checked;
            }
        }
        std::printf("down parts check: %d slices of the layers' first pool experts, the parts' down bit for bit the pack's\n", checked);
    }
    std::vector<int> cores, smt;   // (cores: the order the threads take; the first `phys` one a core)
    cpu_order(cores, smt);
    const int phys = (int) cores.size();
    cores.insert(cores.end(), smt.begin(), smt.end());
    if (threads <= 0) threads = phys;
    // the page cache warm for every expert used: each page read once through a mapping of its own (its PTEs dropped
    // with it), so file-fresh's mapping finds the pages cached but not mapped
    // (--trace: its experts' gate / up rows up to the largest slice end of their calls and their downs whole; with
    // --down-parts every expert's down part too - the parts' pages are those of gdp's mapping, dropped at the end too)
    double cold_before = 0, cold_after = 0, used = 0;
    {
        Maps wm;
        if (!wm.open(pk)) return 2;
        volatile uint8_t sink = 0;
        auto touch = [&](const uint8_t* p, size_t n) {
            cold_before += (double) os::cold_bytes(p, n);
            for (size_t o = 0; o < n; o += 4096) sink = (uint8_t) (sink + p[o]);
            cold_after += (double) os::cold_bytes(p, n);
            used += (double) n;
        };
        auto touch_part = [&](const Layer& x, int e) {
            Exp d;
            if (gdp.is_open()) down_part(gdp, x, e, d);
            if (d.dp != nullptr) touch(d.dp, (size_t) E * (size_t) (d.dpb - d.dpa) * (x.fm.d_row / (size_t) (FF / 256)));
        };
        for (const Layer& x : ly)
            for (int pass = 0; pass < 2; ++pass)
                for (int e : pass ? x.fresh : x.pool) {
                    for (int r = 0; r < 3; ++r) {
                        const Ref& t = *pk.at(x.L, e, r);
                        touch(wm.at(t), (size_t) t.bytes);
                    }
                    touch_part(x, e);
                }
        std::map<long long, int> tb;   // (--trace) (layer index, expert) -> the largest slice end of its calls, in blocks
        for (const auto& t : tr.tok)
            for (const TCall& c : t)
                for (int e : c.e) {
                    int& b = tb[(long long) c.li * 65536 + e];
                    b = std::max(b, c.a >= 0 ? c.b : trace_k);
                }
        for (const auto& y : tb) {
            const Layer& x = ly[(size_t) (y.first / 65536)];
            const int e = (int) (y.first % 65536);
            for (int r = 0; r < 3; ++r) {
                const Ref& t = *pk.at(x.L, e, r);
                touch(wm.at(t), r < 2 ? (size_t) y.second * 256 * x.fm.gu_row : (size_t) t.bytes);
            }
            touch_part(x, e);
        }
        (void) sink;
    }
    std::printf("ds4_lane_bench: %s; %zu layers (%d..%d), %s / %s, a pool of %d experts a layer and %d fresh; %d threads on "
                "cores", pack.c_str(), ly.size(), ly.front().L, ly.back().L, ggml_type_name((ggml_type) ly[0].tg),
                ggml_type_name((ggml_type) ly[0].td), pool_n, (int) ly[0].fresh.size(), threads);
    for (int i = 0; i < threads && i < (int) cores.size(); ++i) std::printf("%s%d", i ? "," : " ", cores[(size_t) i]);
    std::printf("; then (:thr= past %d, SMT)", phys);
    for (size_t i = 0; i < smt.size(); ++i) std::printf("%s%d", i ? "," : " ", smt[i]);
    std::printf("\nthe experts used: %.0f MB, %.0f MB of it not in the page cache before the warm read, %.0f MB after it\n",
                used / 1e6, cold_before / 1e6, cold_after / 1e6);
    if (!tr.tok.empty()) {
        long n = 0, p = 0;
        for (const auto& t : tr.tok)
            for (const TCall& c : t) ++n, p += (long) c.e.size();
        std::printf("trace: %s %s, %zu tokens, %.1f calls a token, %.2f picks a call; of its %ld calls left out: %ld of a layer not "
                    "benched, %ld of more than one token, %ld with no pick left (%ld picks of the layers' %d most picked out)\n",
                    tr.kind.c_str(), trace_path.c_str(), tr.tok.size(), (double) n / (double) tr.tok.size(),
                    (double) p / (double) std::max(1L, n), tr.calls, tr.out_layer, tr.out_multi, tr.out_empty, tr.out_hot, trace_hot);
    }
    if (gdp.is_open()) {
        int lay = 0;
        for (const Layer& x : ly) lay += gdp.covers(x.L) ? 1 : 0;
        std::printf("down parts: %s, %zu files, %d of the %zu layers covered\n", dparts_dir.c_str(), gdp.files().size(), lay, ly.size());
    }
#if defined(__linux__)
    if (want_pmc) {   // (once: the counters open here or not at all)
        Pmc t;
        if (!t.open()) {
            std::printf("pmc: unavailable (%s)\n", t.err.empty() ? "no threads" : t.err.c_str());
            want_pmc = false;
        } else {
            std::printf("pmc: %zu groups a thread (%s)\n", t.grp.size(), amd_zen3() ? "AMD family 0x19 raw events" : "cycles, instructions");
        }
    }
#else
    if (want_pmc) std::printf("pmc: unavailable (Linux only)\n");
    want_pmc = false;
#endif
    std::printf("%-26s %3s %5s %2s %4s %6s %7s %5s %6s %5s %4s %5s %6s %5s %4s %7s %6s %6s %6s %6s %6s %6s %8s\n", "variant",
                "thr", "picks", "k", "rb", "tokens", "lane-ms", "prep", "gu", "gu-tl", "gu%", "hq", "down", "dn-tl", "dn%",
                "MB/tok", "lane", "gu-job", "dn-job", "gu-thr", "dn-thr", "bat-us", "minflt");
    std::mt19937 rng(7);
    std::vector<float> xv((size_t) E);
    for (float& v : xv) v = ((float) (rng() % 20001) - 10000.0f) * 1e-4f;
    for (const Variant& v : vs) {
        std::vector<int> cpus;   // (this variant's CPUs, in the threads' order: the caller on the first)
        for (int c : cores)
            if (std::find(v.skip.begin(), v.skip.end(), c) == v.skip.end()) cpus.push_back(c);
        if (cpus.empty()) cpus.push_back(cores[0]);
        const int thr = std::max(1, std::min(v.thr > 0 ? v.thr : threads, (int) cpus.size()));
        const bool fresh = v.mode == "file-fresh" || v.mode == "file-populate";
        const bool anon = v.mode == "anon4k" || v.mode == "anon2m";
        const Pf pf = v.mode == "file-nopf" || anon || v.mode == "hot" ? kPfNone : v.mode == "file-populate" ? kPfPopulate : kPfWillneed;
#if defined(_WIN32)
        if (v.mode == "file-populate") {
            std::printf("%-26s (MADV_POPULATE_READ: Linux only)\n", v.name.c_str());
            continue;
        }
#endif
        if (fresh && fresh_tokens < 1) {
            std::printf("%-26s (no fresh experts: the layers have too few beyond the pool)\n", v.name.c_str());
            continue;
        }
        if (v.trace && (tr.tok.empty() || (v.mode != "file" && v.mode != "file-nopf"))) {
            std::printf("%-26s (:trace: --trace FILE, and the modes file / file-nopf only)\n", v.name.c_str());
            continue;
        }
        if (v.cold && !v.trace) {
            std::printf("%-26s (:cold: a :trace variant only)\n", v.name.c_str());
            continue;
        }
        if (v.dparts && (!gdp.is_open() || anon)) {
            std::printf("%-26s (:dparts: --down-parts DIR, and not the anonymous copies)\n", v.name.c_str());
            continue;
        }
        os::pin_self(cpus[0]);
        Lane ln;
        ln.threads = thr;
        ln.empty = v.empty;
        const std::vector<int> wcpus(cpus.begin() + 1, cpus.begin() + thr);
        if (v.fix) {
            ln.fpool = std::make_unique<FixPool>(thr - 1, wcpus, 20000);
            ln.slot.assign((size_t) thr, Slot{});
        } else {
            ln.pool = std::make_unique<Pool>(thr - 1, wcpus, 20000);
        }
        ln.pf_age = v.pf_age;
        ln.x = xv;
        ln.h.assign((size_t) 16 * FF, 0.0f);   // (a call's entries: up to 16)
        ln.y.assign((size_t) 16 * E, 0.0f);
        ln.xq.assign((size_t) E * 4, 0);
        ln.hq.assign((size_t) 16 * E * 4, 0);
        Maps mp;
        if (!mp.open(pk)) return 2;
        epx::Parts vdp;   // (:dparts: the parts mapped anew, as mp)
        if (v.dparts) {
            std::string err;
            if (!vdp.open(dparts_dir, err)) {
                std::printf("--down-parts %s: %s\n", dparts_dir.c_str(), err.c_str());
                return 2;
            }
        }
        Anon an;
        // each layer's experts' pointers: pool then fresh (:trace: every expert of the layer by its number, those the
        // trace calls set)
        std::vector<std::vector<Exp>> ex(ly.size());
        auto exp_at = [&](const Layer& x, int e) {
            Exp y;
            y.id = e;
            for (int r = 0; r < 3; ++r) y.m[r] = mp.at(*pk.at(x.L, e, r));
            if (v.dparts) down_part(vdp, x, e, y);
            return y;
        };
        if (v.trace) {
            for (size_t i = 0; i < ly.size(); ++i) ex[i].resize((size_t) pk.n_exp[ly[i].L]);
            for (const auto& t : tr.tok)
                for (const TCall& c : t)
                    for (int e : c.e)
                        if (ex[(size_t) c.li][(size_t) e].id < 0) ex[(size_t) c.li][(size_t) e] = exp_at(ly[(size_t) c.li], e);
        } else {
            size_t need = 0;
            if (anon)
                for (const Layer& x : ly)
                    for (int e : x.pool)
                        for (int r = 0; r < 3; ++r) need += (size_t) ((pk.at(x.L, e, r)->bytes + 4095) / 4096 * 4096);
            if (anon && !an.make(need, v.mode == "anon2m")) {
                std::printf("%-26s (no %.0f MB of anonymous memory)\n", v.name.c_str(), (double) need / 1e6);
                continue;
            }
            uint8_t* w = anon ? an.aligned() : nullptr;
            for (size_t i = 0; i < ly.size(); ++i) {
                for (int pass = 0; pass < 2; ++pass)
                    for (int e : pass ? ly[i].fresh : ly[i].pool) {
                        Exp x = exp_at(ly[i], e);
                        for (int r = 0; r < 3; ++r) {
                            const Ref& t = *pk.at(ly[i].L, e, r);
                            if (anon && pass == 0) {
                                std::memcpy(w, x.m[r], (size_t) t.bytes);
                                x.m[r] = w;
                                w += (size_t) ((t.bytes + 4095) / 4096 * 4096);
                            }
                        }
                        ex[i].push_back(x);
                    }
            }
        }
        const long huge = anon ? anon_huge_kb() : -1;
        const int P = (int) ly[0].pool.size(), ffn = v.k * 256;
        auto token = [&](int t, Acc& a) {
            if (v.trace) {   // (the trace's token t: its calls as they were; a DS4_TRACE call over [0, k))
                for (const TCall& c : tr.tok[(size_t) t]) {
                    Exp e[16];
                    const int np = (int) c.e.size();
                    for (int q = 0; q < np; ++q) e[q] = ex[(size_t) c.li][(size_t) c.e[(size_t) q]];
                    const int s0 = c.a >= 0 ? 256 * c.a : 0, sn = c.a >= 0 ? 256 * (c.b - c.a) : ffn;
                    ln.call(ly[(size_t) c.li], e, np, s0, sn, v.rb, v.split, pf, v.dparts, a);
                }
                return;
            }
            for (size_t i = 0; i < ly.size(); ++i) {
                Exp e[16];
                const int np = std::min(v.picks, 16);
                for (int q = 0; q < np; ++q) {
                    if (v.mode == "hot") e[q] = ex[0][(size_t) q];
                    else if (fresh) e[q] = ex[i][(size_t) (P + t * v.picks + q)];
                    else e[q] = ex[i][(size_t) ((t * v.picks + q) % P)];
                }
                ln.call(v.mode == "hot" ? ly[0] : ly[i], e, np, 0, ffn, v.rb, v.split, pf, v.dparts, a);
            }
        };
        Acc warm;
        if (v.trace && !v.cold)   // (the trace once through this mapping: its PTEs as the engine's, the caches its reuse's)
            for (int t = 0; t < (int) tr.tok.size(); ++t) token(t, warm);
        else if (!fresh)   // (every pool expert once through this mapping, then once more: PTEs and caches as in steady state)
            for (int t = 0; t < 2 * ((P + v.picks - 1) / v.picks); ++t) token(t, warm);
        const int nt = v.trace ? (int) tr.tok.size() : fresh ? fresh_tokens : tokens;
        Acc a;
#if defined(__linux__)
        Pmc pm;   // (the pool's threads are up: each counted)
        const bool pmc_on = want_pmc && pm.open();
        if (pmc_on) pm.start();
#endif
        std::unique_ptr<Spin> spin;   // (after the counters opened: they count the lane's threads alone)
        if (!v.spin.empty()) spin = std::make_unique<Spin>(v.spin);
        const int64_t f0 = faults();
        for (int t = 0; t < nt; ++t) token(t, a);
        const int64_t f1 = faults();
        spin.reset();
#if defined(__linux__)
        std::vector<std::vector<double>> pc;
        double pc_run = 0;
        int pc_never = 0;
        if (pmc_on) {
            pm.stop();
            pc = pm.read(pc_run, pc_never);
        }
#endif
        const double n = nt, lane = (a.prep + a.gu + a.hq + a.dn) / n, mb = (a.gu_bytes + a.dn_bytes) / n / 1e6;
        auto use = [&](double busy, double wall) { return 100.0 * busy / std::max(1e-9, wall * thr); };
        auto gbs = [](double bytes, double ms) { return bytes / 1e6 / std::max(1e-9, ms); };   // (MB / ms = GB/s)
        std::printf("%-26s %3d %5d %2d %4s %6d %7.2f %5.2f %6.2f %5.2f %4.0f %5.2f %6.2f %5.2f %4.0f %7.1f %6.1f %6.1f %6.1f "
                    "%6.2f %6.2f %6.1f %8.0f%s\n",
                    v.name.c_str(), thr, v.picks, v.k, v.split ? "thr" : std::to_string(v.rb).c_str(), nt, lane, a.prep / n,
                    a.gu / n, a.gu_tail / n, use(a.gu_busy, a.gu), a.hq / n, a.dn / n, a.dn_tail / n, use(a.dn_busy, a.dn), mb,
                    gbs(a.gu_bytes + a.dn_bytes, a.prep + a.gu + a.hq + a.dn), gbs(a.gu_bytes, a.gu_busy / thr),
                    gbs(a.dn_bytes, a.dn_busy / thr), gbs(a.gu_bytes, a.gu_busy), gbs(a.dn_bytes, a.dn_busy),
                    1000.0 * (a.gu + a.dn) / std::max(1, a.batches), f0 >= 0 ? (double) (f1 - f0) / n : -1.0,
                    huge >= 0 ? (" (AnonHugePages " + std::to_string(huge / 1024) + " MB)").c_str() : "");
        if (v.trace || v.dparts || v.pf_age > 0) {   // (what the row's calls were: their picks, the downs' source, the hints)
            std::printf("  %s: %.1f calls a token, %.2f picks a call (calls by picks:", v.name.c_str(), (double) a.calls / n,
                        (double) a.picks / (double) std::max(1L, a.calls));
            for (int i = 1; i <= 16; ++i)
                if (a.hist[i] > 0) std::printf(" %d:%ld", i, a.hist[i]);
            std::printf("); downs from the parts %.0f%%; experts hinted %.0f%%\n", 100.0 * (double) a.downs_dp / (double) std::max(1L, a.picks),
                        pf == kPfNone ? 0.0 : 100.0 * (double) a.hinted / (double) std::max(1L, a.picks));
        }
#if defined(__linux__)
        if (!pc.empty()) {   // (the counters over the timed tokens, every thread: spin waits included)
            // an event's total, and its group's cycles (for its share: both counted over the same time)
            auto at = [&](const char* nm, double* gcyc) {
                for (size_t k = 0; k < pm.grp.size(); ++k)
                    for (size_t i = 0; i < pm.grp[k].size(); ++i)
                        if (std::strcmp(pm.grp[k][i].name, nm) == 0) {
                            if (gcyc != nullptr) *gcyc = pc[k][0];
                            return pc[k][i + 1];
                        }
                return -1.0;
            };
            double cyc = -1;
            const double ins = at("instructions", &cyc), kb = (a.gu_bytes + a.dn_bytes) / 1024.0;
            std::printf("  pmc %s: IPC %.2f, %.0f cycles a token a thread, %.1f cycles a KB read; the groups counted %.0f%% "
                        "of their time, %d thread-groups never", v.name.c_str(), cyc > 0 && ins >= 0 ? ins / cyc : -1.0,
                        cyc / n / thr, kb > 0 ? cyc / kb : -1.0, 100.0 * pc_run, pc_never);
            const double mops = at("ret-mops", nullptr), dec = at("op-src-decoder", nullptr), opc = at("op-src-opcache", nullptr);
            if (mops >= 0) std::printf("; %.2f macro-ops an instruction, %.0f%% of ops from the op cache", ins > 0 ? mops / ins : -1.0,
                                       dec + opc > 0 ? 100.0 * opc / (dec + opc) : -1.0);
            std::printf("\n");
            // pct: an event's share of its group's cycles; else a KB read (or of the FP ops in all: fp)
            auto line = [&](const char* title, const char* pre, int how, const char* unit) {
                double fp = 0;
                for (size_t k = 0; k < pm.grp.size(); ++k)
                    for (size_t i = 0; i < pm.grp[k].size(); ++i)
                        if (std::strncmp(pm.grp[k][i].name, pre, std::strlen(pre)) == 0 && pc[k][i + 1] > 0) fp += pc[k][i + 1];
                bool any = false;
                for (size_t k = 0; k < pm.grp.size(); ++k)
                    for (size_t i = 0; i < pm.grp[k].size(); ++i) {
                        if (std::strncmp(pm.grp[k][i].name, pre, std::strlen(pre)) != 0) continue;
                        const double x = pc[k][i + 1], over = how == 0 ? pc[k][0] / 100.0 : how == 1 ? kb : fp / 100.0;
                        if (!any) std::printf("    %s:", title);
                        any = true;
                        if (x < 0) std::printf(" %s n/a", pm.grp[k][i].name);
                        else std::printf(" %s %.2f", pm.grp[k][i].name, over > 0 ? x / over : 0.0);
                    }
                if (any) std::printf(" (%s)\n", unit);
            };
            line("dispatch stalls", "AE.", 0, "% of cycles");
            line("dispatch stalls", "AF.", 0, "% of cycles");
            line("FP pipes", "x000.", 2, "% of the FP ops; all 0: not counted on this CPU");
            line("L1D fills", "44.", 1, "a KB read");
            line("L2", "64.", 1, "a KB read");
            line("L2 prefetches", "7", 1, "a KB read");
        }
#endif
        std::fflush(stdout);
    }
    std::printf("(ms a token: lane = prep + gu + hq + down; prep: the hint and the x quant; gu / down: each batch's wall, -tl\n"
                " its tail, %% its jobs' time over threads x wall; MB/tok: the bytes the jobs read; GB/s: lane over lane-ms,\n"
                " -job over the busy time / threads, -thr a thread's own; bat-us: a batch's wall, us (gate / up and down\n"
                " alike); minflt: page faults a token)\n");
    return 0;
}
