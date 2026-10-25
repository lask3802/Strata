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
// Rates: lane GB/s = bytes / (prep + gu + hq + down); in-job GB/s = a batch's bytes over its busy time / threads (the
// rate while the jobs run); a thread's GB/s = bytes / busy.  minflt a token: getrusage (Linux), the page faults (Windows).
//   ds4_lane_bench --pack DIR [--layers 0-39] [--picks 4] [--k 6] [--pool 8] [--tokens 16] [--fresh-tokens 4]
//                  [--threads N (every physical core)] [--variants file,file-nopf,..] [--dim 5120] [--ff 2304]
//   ds4_lane_bench --synthetic DIR [--syn-layers 2] [--syn-experts 16] [--syn-type IQ2_XXS] [the options above]:
//                  a small fake pack (random legal blocks) written to DIR, then benched as --pack DIR
// Read only on a pack; nothing else written.  No GPU.
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

namespace gg = strata::ds4::gguf;
namespace os = strata::ds4::os;
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
static bool read_pack(const std::string& dir, Pack& pk) {
    std::string mdir = dir;
    {
        std::ifstream f(dir + "/ds4.txt");
        for (std::string s; std::getline(f, s);)
            if (s.rfind("model_dir ", 0) == 0) mdir = s.substr(10);
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
            pk.files[(size_t) n] = nm[0] == '/' || (nm.size() > 1 && nm[1] == ':') ? nm : mdir + "/" + nm;
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

// ---- the bench
struct Layer {
    int L = 0, tg = -1, td = -1;
    gg::IqCpu fm;
    std::vector<int> pool, fresh;   // its experts: the warm pool (rotated), the fresh ones (each used once)
};
struct Exp {
    const uint8_t* m[3];
};
enum Pf { kPfNone, kPfWillneed, kPfPopulate };
struct Acc {
    double prep = 0, gu = 0, gu_tail = 0, gu_busy = 0, hq = 0, dn = 0, dn_tail = 0, dn_busy = 0, gu_bytes = 0, dn_bytes = 0;
};
struct Lane {   // the lane's pool and buffers (Impl's c_xh / c_xhd / c_yd)
    std::unique_ptr<Pool> pool;
    int threads = 1;
    std::vector<float> x, h, y;
    std::vector<uint8_t> xq, hq;
    // run_nodes_prof's batch: wall, tail (the caller's wait once it ran out of jobs), busy (the jobs' time in all)
    void batch(int n, const std::function<void(int)>& f, double& wall, double& tail, double& busy) {
        std::atomic<int64_t> ns{0};
        const std::function<void(int)> g = [&](int j) {
            const auto a = std::chrono::steady_clock::now();
            f(j);
            ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count(),
                         std::memory_order_relaxed);
        };
        const double t0 = now_ms();
        pool->start(n, g);
        pool->help();
        const double th = now_ms();
        pool->wait();
        const double t1 = now_ms();
        wall += t1 - t0;
        tail += t1 - th;
        busy += (double) ns.load() / 1e6;
    }
    // one call: ne experts (one token each) over n_ff [0, ffn)
    void call(const Layer& ly, const Exp* ex, int ne, int ffn, int rb, bool split, Pf pf, Acc& a) {
        const gg::IqCpu& fm = ly.fm;
        gg::IqCpu fs;
        size_t d_off = 0;
        std::string err;
        const bool sliced = ffn != FF;
        if (sliced) gg::iq_cpu_slice(fm, 0, ffn, fs, d_off, err);
        const gg::IqCpu& fd = sliced ? fs : fm;
        const double t0 = now_ms();
        if (pf != kPfNone) {   // (Impl's: a slice's gate / up rows, the down's whole - its rows hold every slice)
            std::vector<std::pair<const void*, size_t>> want;
            for (int q = 0; q < ne; ++q)
                for (int r = 0; r < 3; ++r)
                    want.push_back({ex[q].m[r], r < 2 ? (size_t) ffn * fm.gu_row : (size_t) E * fm.d_row});
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
        batch(ne * GB, [&](int j) {
            const int q = j / GB, r0 = (j % GB) * grs, r1 = std::min(ffn, r0 + grs);
            if (r0 >= r1) return;
            float* hp[1] = {h.data() + (size_t) q * FF};
            gg::iq_cpu_gate_up(fm, ex[q].m[0], ex[q].m[1], ap, 1, hp, r0, r1, kLim);
        }, a.gu, a.gu_tail, a.gu_busy);
        const double th = now_ms();
        for (int q = 0; q < ne; ++q) gg::iq_cpu_quant(fm.d_act, h.data() + (size_t) q * FF, ffn, hq.data() + (size_t) q * E * 4);
        a.hq += now_ms() - th;
        batch(ne * DB, [&](int j) {
            const int q = j / DB, r0 = (j % DB) * drs, r1 = std::min(E, r0 + drs);
            if (r0 >= r1) return;
            const void* hp[1] = {hq.data() + (size_t) q * E * 4};
            float* yp[1] = {y.data() + (size_t) q * E};
            gg::iq_cpu_down(fd, ex[q].m[2] + d_off, hp, 1, yp, r0, r1);
        }, a.dn, a.dn_tail, a.dn_busy);
        a.gu_bytes += (double) ne * 2.0 * (double) ffn * (double) fm.gu_row;
        a.dn_bytes += (double) ne * (double) E * (double) fd.d_row;
    }
};

struct Variant {
    std::string name, mode;
    int k = 6, picks = 4, rb = 64, thr = 0;
    bool split = false;
};
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
        else return false;
    }
    const char* modes[] = {"file", "file-nopf", "file-fresh", "file-populate", "anon4k", "anon2m", "hot"};
    bool known = false;
    for (const char* m : modes) known = known || v.mode == m;
    return known && v.k >= 1 && v.k <= 9 && v.picks >= 1 && v.rb >= 1;
}

int main(int argc, char** argv) {
    std::string pack, syn, syn_type = "IQ2_XXS", vlist =
        "file,file:k=9,file-nopf,file-fresh,file-populate,anon4k,anon2m,file:picks=1,hot:picks=1,file:k=9:picks=8,"
        "hot:k=9:picks=8,file:rb=128,file:rb=256,file:split,file:thr=1,file:thr=2,file:thr=4,file:thr=8,file:thr=12";
    std::string layers_s;
    int syn_l = 2, syn_e = 16, pool_n = 8, tokens = 16, fresh_tokens = 4, threads = 0;
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
        else {
            std::printf("usage: ds4_lane_bench --pack DIR | --synthetic DIR [--layers 0-39] [--picks 4] [--k 6] [--pool 8] "
                        "[--tokens 16] [--fresh-tokens 4] [--threads N] [--variants a,b,..] [--dim 5120] [--ff 2304]\n");
            return 2;
        }
    }
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
                            ":rb= :thr= :split\n", s.c_str());
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
            for (int r = 0; r < 3; ++r) {
                const Ref* t = pk.at(L, e, r);
                const uint64_t want_b = r < 2 ? (uint64_t) FF * rowb(x.tg, E) : (uint64_t) E * rowb(x.td, FF);
                if (t == nullptr || t->bytes != want_b || t->type != (r < 2 ? x.tg : x.td)) {
                    std::printf("layer %d expert %d matrix %d: missing, or not %llu bytes of the layer's type\n", L, e, r,
                                (unsigned long long) want_b);
                    return 2;
                }
            }
            (j < pool_n ? x.pool : x.fresh).push_back(e);
        }
        ly.push_back(x);
    }
    for (const Layer& x : ly) fresh_tokens = std::min(fresh_tokens, (int) (x.fresh.size() / (size_t) max_picks));
    if (max_picks > 16) {
        std::printf("--picks / :picks= up to 16\n");
        return 2;
    }
    std::vector<int> cores;
    for (const auto& n : os::node_cores()) cores.insert(cores.end(), n.begin(), n.end());
    if (cores.empty()) cores.push_back(0);
    if (threads <= 0) threads = (int) cores.size();
    // the page cache warm for every expert used: each page read once through a mapping of its own (its PTEs dropped
    // with it), so file-fresh's mapping finds the pages cached but not mapped
    double cold_before = 0, cold_after = 0, used = 0;
    {
        Maps wm;
        if (!wm.open(pk)) return 2;
        volatile uint8_t sink = 0;
        for (const Layer& x : ly)
            for (int pass = 0; pass < 2; ++pass)
                for (int e : pass ? x.fresh : x.pool)
                    for (int r = 0; r < 3; ++r) {
                        const Ref& t = *pk.at(x.L, e, r);
                        const uint8_t* p = wm.at(t);
                        cold_before += (double) os::cold_bytes(p, (size_t) t.bytes);
                        for (uint64_t o = 0; o < t.bytes; o += 4096) sink = (uint8_t) (sink + p[o]);
                        cold_after += (double) os::cold_bytes(p, (size_t) t.bytes);
                        used += (double) t.bytes;
                    }
        (void) sink;
    }
    std::printf("ds4_lane_bench: %s; %zu layers (%d..%d), %s / %s, a pool of %d experts a layer and %d fresh; %d threads on "
                "cores", pack.c_str(), ly.size(), ly.front().L, ly.back().L, ggml_type_name((ggml_type) ly[0].tg),
                ggml_type_name((ggml_type) ly[0].td), pool_n, (int) ly[0].fresh.size(), threads);
    for (int i = 0; i < threads && i < (int) cores.size(); ++i) std::printf("%s%d", i ? "," : " ", cores[(size_t) i]);
    std::printf("\nthe experts used: %.0f MB, %.0f MB of it not in the page cache before the warm read, %.0f MB after it\n",
                used / 1e6, cold_before / 1e6, cold_after / 1e6);
    std::printf("%-20s %3s %5s %2s %4s %6s %7s %5s %6s %5s %4s %5s %6s %5s %4s %7s %6s %6s %6s %6s %6s %8s\n", "variant", "thr",
                "picks", "k", "rb", "tokens", "lane-ms", "prep", "gu", "gu-tl", "gu%", "hq", "down", "dn-tl", "dn%", "MB/tok",
                "lane", "gu-job", "dn-job", "gu-thr", "dn-thr", "minflt");
    os::pin_self(cores[0]);
    std::mt19937 rng(7);
    std::vector<float> xv((size_t) E);
    for (float& v : xv) v = ((float) (rng() % 20001) - 10000.0f) * 1e-4f;
    for (const Variant& v : vs) {
        const int thr = std::max(1, std::min(v.thr > 0 ? v.thr : threads, (int) cores.size()));
        const bool fresh = v.mode == "file-fresh" || v.mode == "file-populate";
        const bool anon = v.mode == "anon4k" || v.mode == "anon2m";
        const Pf pf = v.mode == "file-nopf" || anon || v.mode == "hot" ? kPfNone : v.mode == "file-populate" ? kPfPopulate : kPfWillneed;
#if defined(_WIN32)
        if (v.mode == "file-populate") {
            std::printf("%-20s (MADV_POPULATE_READ: Linux only)\n", v.name.c_str());
            continue;
        }
#endif
        if (fresh && fresh_tokens < 1) {
            std::printf("%-20s (no fresh experts: the layers have too few beyond the pool)\n", v.name.c_str());
            continue;
        }
        Lane ln;
        ln.threads = thr;
        ln.pool = std::make_unique<Pool>(thr - 1, std::vector<int>(cores.begin() + 1, cores.begin() + thr), 20000);
        ln.x = xv;
        ln.h.assign((size_t) v.picks * FF, 0.0f);
        ln.y.assign((size_t) v.picks * E, 0.0f);
        ln.xq.assign((size_t) E * 4, 0);
        ln.hq.assign((size_t) v.picks * E * 4, 0);
        Maps mp;
        if (!mp.open(pk)) return 2;
        Anon an;
        // each layer's experts' pointers: pool then fresh
        std::vector<std::vector<Exp>> ex(ly.size());
        {
            size_t need = 0;
            if (anon)
                for (const Layer& x : ly)
                    for (int e : x.pool)
                        for (int r = 0; r < 3; ++r) need += (size_t) ((pk.at(x.L, e, r)->bytes + 4095) / 4096 * 4096);
            if (anon && !an.make(need, v.mode == "anon2m")) {
                std::printf("%-20s (no %.0f MB of anonymous memory)\n", v.name.c_str(), (double) need / 1e6);
                continue;
            }
            uint8_t* w = anon ? an.aligned() : nullptr;
            for (size_t i = 0; i < ly.size(); ++i) {
                for (int pass = 0; pass < 2; ++pass)
                    for (int e : pass ? ly[i].fresh : ly[i].pool) {
                        Exp x;
                        for (int r = 0; r < 3; ++r) {
                            const Ref& t = *pk.at(ly[i].L, e, r);
                            x.m[r] = mp.at(t);
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
            for (size_t i = 0; i < ly.size(); ++i) {
                Exp e[16];
                const int np = std::min(v.picks, 16);
                for (int q = 0; q < np; ++q) {
                    if (v.mode == "hot") e[q] = ex[0][(size_t) q];
                    else if (fresh) e[q] = ex[i][(size_t) (P + t * v.picks + q)];
                    else e[q] = ex[i][(size_t) ((t * v.picks + q) % P)];
                }
                ln.call(v.mode == "hot" ? ly[0] : ly[i], e, np, ffn, v.rb, v.split, pf, a);
            }
        };
        Acc warm;
        if (!fresh)   // (every pool expert once through this mapping, then once more: PTEs and caches as in steady state)
            for (int t = 0; t < 2 * ((P + v.picks - 1) / v.picks); ++t) token(t, warm);
        const int nt = fresh ? fresh_tokens : tokens;
        Acc a;
        const int64_t f0 = faults();
        for (int t = 0; t < nt; ++t) token(t, a);
        const int64_t f1 = faults();
        const double n = nt, lane = (a.prep + a.gu + a.hq + a.dn) / n, mb = (a.gu_bytes + a.dn_bytes) / n / 1e6;
        auto use = [&](double busy, double wall) { return 100.0 * busy / std::max(1e-9, wall * thr); };
        auto gbs = [](double bytes, double ms) { return bytes / 1e6 / std::max(1e-9, ms); };   // (MB / ms = GB/s)
        std::printf("%-20s %3d %5d %2d %4s %6d %7.2f %5.2f %6.2f %5.2f %4.0f %5.2f %6.2f %5.2f %4.0f %7.1f %6.1f %6.1f %6.1f "
                    "%6.2f %6.2f %8.0f%s\n",
                    v.name.c_str(), thr, v.picks, v.k, v.split ? "thr" : std::to_string(v.rb).c_str(), nt, lane, a.prep / n,
                    a.gu / n, a.gu_tail / n, use(a.gu_busy, a.gu), a.hq / n, a.dn / n, a.dn_tail / n, use(a.dn_busy, a.dn), mb,
                    gbs(a.gu_bytes + a.dn_bytes, a.prep + a.gu + a.hq + a.dn), gbs(a.gu_bytes, a.gu_busy / thr),
                    gbs(a.dn_bytes, a.dn_busy / thr), gbs(a.gu_bytes, a.gu_busy), gbs(a.dn_bytes, a.dn_busy),
                    f0 >= 0 ? (double) (f1 - f0) / n : -1.0,
                    huge >= 0 ? (" (AnonHugePages " + std::to_string(huge / 1024) + " MB)").c_str() : "");
        std::fflush(stdout);
    }
    std::printf("(ms a token: lane = prep + gu + hq + down; prep: the hint and the x quant; gu / down: each batch's wall, -tl\n"
                " its tail, %% its jobs' time over threads x wall; MB/tok: the bytes the jobs read; GB/s: lane over lane-ms,\n"
                " -job over the busy time / threads, -thr a thread's own; minflt: page faults a token)\n");
    return 0;
}
