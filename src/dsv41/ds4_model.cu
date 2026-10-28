// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_model.cu - DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp).
//
// The layer, for one token at position p (DeepSeek's inference/model.py, Block.forward with the lagged pre-mix):
//   [Engram, layers 1 and 14]  h += gate(h, key) * value from 24 n-gram rows of the FP8 tables
//   a_pre, a_post, a_comb = hc_mixes(h);  x = rmsnorm(sum_s pre[s] h[s])   (pre: the previous layer's f_pre)
//   attention(x): q from the low-rank wq_a / wq_b, one shared K = V row per token in a 128-slot window ring, plus
//                 (compress ratio > 0) up to 512 compressed rows the indexer picks; a learned sink per head; the
//                 output's rotary part rotated back, rounded to FP8, through the grouped wo_a and wo_b
//   h = post * attn + comb^T h;  f_* = hc_mixes(h);  x = rmsnorm(sum_s a_pre[s] h[s])
//   moe(x): 6 of 384 routed experts (sqrt-softplus scores, biased top-k) + the shared expert
//   h = post * moe + comb^T h;  pre = f_pre
// and after the last layer logits = head(rmsnorm(sum_s pre[s] h[s])).
#include "ds4_model.hpp"
#include "ds4_engram_io.hpp"
#include "ds4_epparts.hpp"

#include "ds4_gguf.cuh"
#include "ds4_iq.cuh"
#include "ds4_gguf_cpu.hpp"
#include "ds4_ffslice.hpp"
#include "ds4_flow_batches.hpp"
#include "ds4_gemm16.cuh"
#include "ds4_kernels.cuh"
#include "ds4_os.hpp"
#include "ds4_pool.hpp"
#include "ds4_rx_fill.hpp"
#include "ds4_rx_fill_link.hpp"
#include "ds4_rx.hpp"
#include "ds4_rx_split.hpp"
#include "ds4_small_prompt.hpp"
#include "ds4_stage.hpp"
#include "ds4_stage_slots.hpp"
#include "ds4_store.hpp"
#include "ds4_u9.hpp"
#include "ds4_vision.cuh"
#include "strata/dsv41/exl3_cpu.hpp"
#include "strata/dsv41/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/dsv41/moe_win.hpp"
#include "strata/dsv41/sampler.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_MSC_VER)
#define strtok_r strtok_s
#endif

namespace strata::ds4 {
namespace x3 = strata::dsv41k::exl3;
namespace kc = strata::dsv41k::cpu;
namespace gf = strata::dsv41k::glmf;
namespace gg = strata::ds4::gguf;

#define CK(call)                                                                            \
    do {                                                                                    \
        cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                       \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// printf onto the end of `out`, as long as it comes out (a stats line: no fixed buffer to cut it)
#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
void add_f(std::string& out, const char* f, ...) {
    va_list a, a2;
    va_start(a, f);
    va_copy(a2, a);
    const int n = std::vsnprintf(nullptr, 0, f, a);
    va_end(a);
    if (n > 0) {
        const size_t at = out.size();
        out.resize(at + (size_t) n + 1);
        std::vsnprintf(&out[at], (size_t) n + 1, f, a2);
        out.resize(at + (size_t) n);
    }
    va_end(a2);
}

// part p of np of `whole` (a multiple of 128) in 128-blocks, as even as they go (whole / 128 % np parts a block more:
// those from part rot on, cyclically - so a group's layers, each its own rot, even out what each member holds):
// tensor parallel's columns of an FFN or of the head - an even split is what whole / np gave before
void split128(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<int> cpu_list(const std::string& s) {   // "0-3,8,10-11"
    std::vector<int> v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty() || part == "\n") continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
// each NUMA node's physical cores (the first logical CPU of each core), in order
std::vector<std::vector<int>> node_cores() {
#if defined(_WIN32)
    static const bool smt = [] {   // DS4_CPU_SMT=1: the second hardware thread of every core as well (memory-bound rows gain from it)
        const char* v = std::getenv("DS4_CPU_SMT");
        return v != nullptr && std::atoi(v) != 0;
    }();
    std::vector<std::vector<int>> nodes = os::node_cores(smt);
#else
    std::vector<std::vector<int>> nodes;
    for (int n = 0; n < 64; ++n) {
        const std::string cl = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cl.empty()) break;
        std::vector<int> cores;
        for (int c : cpu_list(cl)) {
            const std::vector<int> sib = cpu_list(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            if (sib.empty() || sib[0] == c) cores.push_back(c);
        }
        if (!cores.empty()) nodes.push_back(cores);
    }
#endif
    if (nodes.empty()) {
        std::vector<int> all;
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency() / 2); ++c) all.push_back((int) c);
        nodes.push_back(all);
    }
    return nodes;
}
// anonymous memory on one NUMA node (node < 0: wherever the first touch puts it), transparent huge pages
void* node_map(size_t bytes, int node, int n_nodes) { return os::anon_alloc(bytes, node, n_nodes); }
void pin_self(int cpu) { os::pin_self(cpu); }
// the RAM this process can still get: MemAvailable, and below it the tightest cgroup v2 limit on the way to the root
// (its limit less what is charged, the page cache counted as free: the kernel takes it back); ~0: not known
uint64_t ram_avail() {
#if defined(_WIN32)
    return os::ram_avail();
#endif
    uint64_t avail = ~uint64_t{0};
    {
        std::istringstream mi(read_file("/proc/meminfo"));
        std::string k;
        uint64_t kb = 0;
        while (mi >> k >> kb) {
            if (k == "MemAvailable:") avail = kb << 10;
            mi.ignore(64, '\n');
        }
    }
    std::string cg = read_file("/proc/self/cgroup");   // "0::/user.slice/..."
    const size_t at = cg.find("0::");
    if (at == std::string::npos) return avail;
    cg = cg.substr(at + 3);
    cg = cg.substr(0, cg.find('\n'));
    for (;;) {
        const std::string dir = "/sys/fs/cgroup" + (cg == "/" ? std::string() : cg);
        const std::string mx = read_file(dir + "/memory.max");
        if (!mx.empty() && mx.compare(0, 3, "max") != 0) {
            const uint64_t lim = std::strtoull(mx.c_str(), nullptr, 10);
            const uint64_t cur = std::strtoull(read_file(dir + "/memory.current").c_str(), nullptr, 10);
            uint64_t file = 0;
            std::istringstream st(read_file(dir + "/memory.stat"));
            std::string k;
            uint64_t v = 0;
            while (st >> k >> v)
                if (k == "file") file = v;
            const uint64_t used = cur > file ? cur - file : 0;
            avail = std::min(avail, lim > used ? lim - used : 0);
        }
        if (cg.empty() || cg == "/") break;
        const size_t sl = cg.rfind('/');
        cg = sl == 0 || sl == std::string::npos ? "/" : cg.substr(0, sl);
    }
    return avail;
}

// the remote experts' upload, before and after (Linux): the time (UTC), the cgroup's anon / file (this container's:
// the namespace's root, /sys/fs/cgroup) and this process's RssAnon / RssFile, MB; -1 not there
#if defined(__linux__)
std::string rx_mem_note() {
    long long an = -1, fi = -1, ra = -1, rf = -1;
    {
        std::istringstream st(read_file("/sys/fs/cgroup/memory.stat"));
        std::string k;
        long long v = 0;
        while (st >> k >> v) {
            if (k == "anon") an = v / 1000000;
            if (k == "file") fi = v / 1000000;
        }
    }
    {
        std::istringstream st(read_file("/proc/self/status"));
        std::string k;
        long long kb = 0;
        while (st >> k) {
            if (k == "RssAnon:" && st >> kb) ra = kb * 1024 / 1000000;
            else if (k == "RssFile:" && st >> kb) rf = kb * 1024 / 1000000;
            st.ignore(256, '\n');
        }
    }
    const std::time_t s = std::time(nullptr);
    std::tm u{};
    gmtime_r(&s, &u);
    char b[160];
    std::snprintf(b, sizeof b, "%02d:%02d:%02d UTC: the cgroup's anon %lld file %lld MB, this process's RssAnon %lld "
                               "RssFile %lld MB", u.tm_hour, u.tm_min, u.tm_sec, an, fi, ra, rf);
    return b;
}
#endif

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            u = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ------------------------------------------------------------------ the pack
struct Tensor {
    std::string dtype;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Pack {
    std::string dir, model_dir;
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> files;
    std::vector<os::fd_t> fds;
    std::unordered_map<std::string, Tensor> t;
    int eg_vocab = 0, eg_layers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<int64_t> eg_mult, eg_primes, eg_off;
    // format gguf (tools/ds4/pack_gguf.py): the files are a GGUF's shards, every one mapped at load (map()) - the
    // routed experts are always read in place (the OS page cache their RAM tier, the disk the rest)
    bool gguf = false;

    std::vector<std::pair<uint8_t*, size_t>> maps;   // file -> its read-only mapping (map(): the experts in place)
    std::vector<os::fd_t> dfds;   // (gguf) file -> unbuffered (os::open_direct; kBadFd: buffered reads only)
    std::vector<os::fd_t> pfds;   // (gguf) file -> for reads from many threads at once (os::open_par)

    ~Pack() {
        for (auto& m : maps) os::unmap(m.first, m.second);
        for (os::fd_t fd : fds) os::close_fd(fd);
        for (os::fd_t fd : dfds) os::close_fd(fd);
        for (os::fd_t fd : pfds) os::close_fd(fd);
    }
    // file i mapped whole, read-only (once; not thread-safe: called while loading)
    const uint8_t* map(int file) {
        if (maps.size() < fds.size()) maps.resize(fds.size(), {nullptr, 0});
        auto& m = maps[(size_t) file];
        if (m.first == nullptr) {
            const uint64_t size = os::file_size(fds[(size_t) file]);
            void* p = os::map_ro(fds[(size_t) file], size);
            if (p == nullptr) return nullptr;
            m = {(uint8_t*) p, (size_t) size};
        }
        return m.first;
    }

    bool load(const std::string& d, std::string& err) {
        dir = d;
        {
            std::ifstream f(d + "/ds4.txt");
            if (!f) {
                err = d + "/ds4.txt: not found (run tools/ds4/pack.py --model <model>)";
                return false;
            }
            std::string line;
            while (std::getline(f, line)) {
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) continue;
                kv[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        model_dir = str("model_dir");
        std::FILE* f = std::fopen((d + "/tensors.txt").c_str(), "r");
        if (f == nullptr) {
            err = d + "/tensors.txt: not found";
            return false;
        }
        std::vector<char> buf(1 << 16);
        while (std::fgets(buf.data(), (int) buf.size(), f) != nullptr) {
            char* s = buf.data();
            if (s[0] == 'f') {   // file <i> <name>
                char name[4096];
                int i = 0;
                if (std::sscanf(s, "file %d %4095s", &i, name) == 2) {
                    if ((int) files.size() <= i) files.resize((size_t) i + 1);
                    files[(size_t) i] = name;
                }
                continue;
            }
            if (s[0] != 't') continue;
            char* save = nullptr;
            ::strtok_r(s, " \n", &save);
            const char* name = ::strtok_r(nullptr, " \n", &save);
            const char* dt = ::strtok_r(nullptr, " \n", &save);
            const char* fi = ::strtok_r(nullptr, " \n", &save);
            const char* off = ::strtok_r(nullptr, " \n", &save);
            const char* nb = ::strtok_r(nullptr, " \n", &save);
            if (name == nullptr || nb == nullptr) continue;
            Tensor x;
            x.dtype = dt;
            x.file = std::atoi(fi);
            x.off = std::strtoull(off, nullptr, 10);
            x.bytes = std::strtoull(nb, nullptr, 10);
            while (const char* dim = ::strtok_r(nullptr, " \n", &save)) x.shape.push_back(std::atoll(dim));
            t.emplace(name, std::move(x));
        }
        std::fclose(f);
        for (const std::string& fn : files) {
            const std::string p = model_dir + "/" + fn;
            const os::fd_t fd = os::open_ro(p);
            if (fd == os::kBadFd) {
                err = p + ": cannot open";
                return false;
            }
            fds.push_back(fd);
        }
        gguf = str("format") == "gguf";
        if (gguf)
            for (size_t i = 0; i < fds.size(); ++i) {
                if (map((int) i) == nullptr) {
                    err = model_dir + "/" + files[i] + ": cannot map";
                    return false;
                }
                dfds.push_back(os::open_direct(model_dir + "/" + files[i]));
                pfds.push_back(os::open_par(model_dir + "/" + files[i]));
            }
        // the Engram tables
        std::FILE* e = std::fopen((d + "/engram.bin").c_str(), "rb");
        if (e == nullptr) {
            err = d + "/engram.bin: not found";
            return false;
        }
        char magic[4];
        int32_t hdr[6];
        bool ok = std::fread(magic, 1, 4, e) == 4 && std::memcmp(magic, "DSE1", 4) == 0 &&
                  std::fread(hdr, 4, 6, e) == 6;
        if (ok) {
            eg_vocab = hdr[0], eg_layers = hdr[1], eg_ngram = hdr[2], eg_heads = hdr[3], eg_hd = hdr[4], eg_pad = hdr[5];
            const int cols = (eg_ngram - 1) * eg_heads;
            eg_map.resize((size_t) eg_vocab);
            eg_mult.resize((size_t) eg_layers * eg_ngram);
            eg_primes.resize((size_t) eg_layers * cols);
            eg_off.resize((size_t) eg_layers * cols);
            ok = std::fread(eg_map.data(), 4, eg_map.size(), e) == eg_map.size() &&
                 std::fread(eg_mult.data(), 8, eg_mult.size(), e) == eg_mult.size() &&
                 std::fread(eg_primes.data(), 8, eg_primes.size(), e) == eg_primes.size() &&
                 std::fread(eg_off.data(), 8, eg_off.size(), e) == eg_off.size();
        }
        std::fclose(e);
        if (!ok) {
            err = d + "/engram.bin: unreadable";
            return false;
        }
        return true;
    }

    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v;
        std::stringstream ss(str(k));
        std::string x;
        while (std::getline(ss, x, ','))
            if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    bool read_at(int file, uint64_t off, uint64_t n, void* dst) const {   // (any thread; side by side where pfds is)
        if ((size_t) file < pfds.size() && pfds[(size_t) file] != os::kBadFd)
            return os::pread_par(pfds[(size_t) file], dst, n, off);
        return os::pread_full(fds[(size_t) file], dst, n, off);
    }
    bool read(const Tensor& x, void* dst) const { return read_at(x.file, x.off, x.bytes, dst); }
    const uint8_t* at(int file, uint64_t off) const { return maps[(size_t) file].first + off; }   // (gguf: mapped)
};

// ------------------------------------------------------------------ geometry
struct Geo {
    int vocab = 0, dim = 0, nl = 0, heads = 0, hd = 0, rd = 0, q_lora = 0, o_lora = 0, o_groups = 0, n_exp = 0,
        k_exp = 0, ff = 0, win = 0, hc = 0, sinkhorn = 0, ix_heads = 0, ix_hd = 0, ix_topk = 0, cand_src = -1,
        cand_topk = 0, cand_blk = 0, orig_ctx = 0, eos = 1;
    float eps = 1e-20f, hc_eps = 1e-6f, route_scale = 1.0f, lim = 0.0f;
    double rope_theta = 1e4, comp_theta = 1e4, factor = 1.0, beta_fast = 32, beta_slow = 1;
    std::vector<int> ratios, kv_src, ix_src, eg_layers;
    // the DSpark drafter (the checkpoint's mtp.* blocks): a block of [seed, noise x (block - 1)] through n_mtp blocks
    int n_mtp = 0, d_block = 0, d_noise = 0, d_rank = 0, d_n_exp = 0, d_k = 0;
    std::vector<int> d_targets;

    void from(const Pack& p) {
        vocab = (int) p.num("vocab_size", 0), dim = (int) p.num("hidden_size", 0);
        nl = (int) p.num("num_hidden_layers", 0), heads = (int) p.num("num_attention_heads", 0);
        hd = (int) p.num("head_dim", 0), rd = (int) p.num("qk_rope_head_dim", 0);
        q_lora = (int) p.num("q_lora_rank", 0), o_lora = (int) p.num("o_lora_rank", 0);
        o_groups = (int) p.num("o_groups", 0), n_exp = (int) p.num("n_routed_experts", 0);
        k_exp = (int) p.num("num_experts_per_tok", 0), ff = (int) p.num("moe_intermediate_size", 0);
        win = (int) p.num("sliding_window", 0), hc = (int) p.num("hc_mult", 0);
        sinkhorn = (int) p.num("hc_sinkhorn_iters", 0), ix_heads = (int) p.num("index_n_heads", 0);
        ix_hd = (int) p.num("index_head_dim", 0), ix_topk = (int) p.num("index_topk", 0);
        cand_src = (int) p.num("candidate_source_layer_id", -1), cand_topk = (int) p.num("candidate_topk_blocks", 0);
        cand_blk = (int) p.num("candidate_block_size", 0);
        orig_ctx = (int) p.num("rope_scaling.original_max_position_embeddings", 0);
        eos = (int) p.num("eos_token_id", 1);
        eps = (float) p.num("rms_norm_eps", 1e-20), hc_eps = (float) p.num("hc_eps", 1e-6);
        route_scale = (float) p.num("routed_scaling_factor", 1.0), lim = (float) p.num("swiglu_limit", 0.0);
        rope_theta = p.num("rope_theta", 1e4), comp_theta = p.num("compress_rope_theta", 1e4);
        factor = p.num("rope_scaling.factor", 1.0), beta_fast = p.num("rope_scaling.beta_fast", 32);
        beta_slow = p.num("rope_scaling.beta_slow", 1);
        ratios = p.list("compress_ratios"), kv_src = p.list("kv_source_layer_ids");
        ix_src = p.list("index_source_layer_ids"), eg_layers = p.list("engram_layer_ids");
        d_block = (int) p.num("dspark_block_size", 0), d_noise = (int) p.num("dspark_noise_token_id", 0);
        d_rank = (int) p.num("dspark_markov_rank", 0), d_n_exp = (int) p.num("dspark_n_routed_experts", 0);
        d_k = (int) p.num("dspark_num_experts_per_tok", 0), d_targets = p.list("dspark_target_layer_ids");
        // the DSpark blocks: an EXL3 pack's, or a GGUF pack's from DeepSeek's checkpoint (tools/ds4/drafter_pack.py)
        n_mtp = d_block > 0 && (int) ratios.size() > nl &&
                        (p.get("mtp.0.attn.wq_a.trellis") != nullptr || p.get("mtp.0.attn.wq_a.weight") != nullptr)
                    ? (int) ratios.size() - nl : 0;
    }
};

// DeepSeek's precompute_freqs_cis frequencies (YaRN when original_seq_len > 0), in F32 as torch computes them
std::vector<float> rope_freqs(int dim, int original, double base, double factor, double beta_fast, double beta_slow) {
    std::vector<float> f((size_t) dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float e = (float) (2 * i) / (float) dim;
        f[(size_t) i] = 1.0f / std::pow((float) base, e);
    }
    if (original > 0) {
        auto corr = [&](double rot) {
            return dim * std::log(original / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base));   // (M_PI: not MSVC's)
        };
        const double low = std::max(std::floor(corr(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corr(beta_slow)), (double) dim - 1);
        const float den = (float) std::max(high - low, 1e-3);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / den;
            ramp = std::min(1.0f, std::max(0.0f, ramp));
            const float smooth = 1.0f - ramp;
            f[(size_t) i] = f[(size_t) i] / (float) factor * (1.0f - smooth) + f[(size_t) i] * smooth;
        }
    }
    return f;
}

// ------------------------------------------------------------------ devices, layers
struct Dev {
    int id = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    // U9 hc: a sublayer's hc_mix / hc_split beside st (forked after the residual is final, joined before its hc_post)
    cudaStream_t st2 = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    size_t free0 = 0, total = 0, weights = 0, experts = 0, caches = 0, scratch = 0;
    std::vector<void*> allocs;
    // a window's scratch, rows of up to kWin tokens (the routed experts' rows: kWin * kPicks)
    float *h = nullptr, *h2 = nullptr, *x = nullptr, *qr = nullptr, *q = nullptr, *kvr = nullptr, *ckr = nullptr,
          *cgr = nullptr, *latent = nullptr, *ikr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr,
          *bm = nullptr, *o = nullptr, *oa = nullptr, *ao = nullptr, *mix = nullptr, *a_pre = nullptr,
          *a_post = nullptr, *a_comb = nullptr, *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr,
          *glog = nullptr, *rw = nullptr, *moe_out = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *sh_out = nullptr, *cpu_part = nullptr, *eg_x = nullptr, *eg_kv = nullptr, *logits = nullptr,
          *freqs_win = nullptr, *freqs_comp = nullptr, *xe = nullptr, *xg = nullptr, *xu = nullptr, *xhh = nullptr,
          *ed = nullptr;
    float* glog_a = nullptr;   // lookahead: the next two layers' gate logits on this layer's FFN input (rows 1024 apart)
    float* late_post = nullptr;   // DS4_EP_DEFER: the FFN post of the layer whose mailbox part is still out (rows 8 apart)
    // DS4_EP_DEFER_VERIFY: the deferred path beside the exact one - its FFN output without the part, its streams after
    // that FFN, and after the next attention
    float *v_moe = nullptr, *v_h = nullptr, *v_h2 = nullptr;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *tok = nullptr;
    unsigned long long* plan_ptr = nullptr;           // the decode path's VRAM experts (0: on the CPU lane)
    uint8_t* eg_rows = nullptr;
    uint16_t *emb_rows = nullptr, *ext = nullptr;   // the window's embedding rows; the window attention's rows
    AttnScratch attn_scr;   // a few queries' selected compressed rows decoded (sparse_attn)
    // the hand-off to the next device through pinned host memory (these cards copy peer to peer only through the
    // driver's staging, ~1.7 ms a copy): h | pre | topk | cand
    uint8_t* bounce = nullptr;
    cudaEvent_t ev_out = nullptr;
    float* tap = nullptr;   // a window's stream means before a DSpark target layer
    // threads / tensor parallelism: its group and its index in it, the group's all-reduces it took part in, xfer_put's
    // block counter, the hand-off to the next group (host-mapped: h | pre | topk | cand, then its flag), the window
    // MoE's grouping scratch
    int grp = 0, mi = 0;
    unsigned ar_seq = 0;
    unsigned* xcnt = nullptr;
    // the head group's members (the last layer's group): each its columns (split128 of V) of the head and the norms;
    // logits_part: a member's share of the logits (the lead's: where the others' land before they join its rows)
    x3::Mat head;
    float *norm_w = nullptr, *d_norm = nullptr, *logits_part = nullptr;
    unsigned head_seq = 0;
    // the drafter's markov bias over the head group (markov_share): this member's rows [mk_v0, mk_v0 + mk_vn) of the
    // markov head, the previous token's E row, the drafted tokens (the lead's: d_ints)
    uint16_t *mk_share = nullptr, *mk_erow = nullptr;
    int* mk_ints = nullptr;
    int mk_v0 = 0, mk_vn = 0;
    uint8_t* hand = nullptr;
    unsigned* hand_flag = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
};

// the chunked prompt path's buffers on a device: a chunk of C tokens through a layer at once (the attention in slabs of
// A queries, the indexer in slabs of AI), its GEMMs through FP16 reconstructions of the weights and cuBLAS
// DS4_PRED_DUMP: a fixed +-1 random projection of the router input (D -> 256), one block a token
__global__ void pred_proj_k(const float* x, int D, float* out) {
    const int t = blockIdx.x, j = threadIdx.x;   // 256 threads
    const float* xr = x + (size_t) t * D;
    float a = 0.0f;
    for (int d = 0; d < D; ++d) {
        unsigned h = (unsigned) d * 2654435761u ^ ((unsigned) j + 1u) * 40503u;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        a += (h & 1u) ? xr[d] : -xr[d];
    }
    out[(size_t) t * 256 + j] = a * rsqrtf((float) D);
}

struct Chunk {
    int C = 0, A = 0, AI = 0;
    float *h = nullptr, *x = nullptr, *acc = nullptr, *qr = nullptr, *kvr = nullptr, *ckr = nullptr, *cgr = nullptr,
          *latent = nullptr, *ikr = nullptr, *q = nullptr, *o = nullptr, *oa = nullptr, *iq = nullptr, *iw = nullptr,
          *scores = nullptr, *bm = nullptr, *mix = nullptr, *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr,
          *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr, *glog = nullptr, *rw = nullptr,
          *sg = nullptr, *su = nullptr, *y = nullptr, *eg_x = nullptr, *eg_kv = nullptr;
    uint16_t *x16 = nullptr, *qr16 = nullptr, *o16 = nullptr, *oa16 = nullptr, *lat16 = nullptr, *sh16 = nullptr,
             *h16 = nullptr, *eg_x16 = nullptr, *ext = nullptr, *w16 = nullptr, *emb = nullptr;
    size_t w16_elems = 0;
    // DS4_DQ_ONCE: a (chunk, layer)'s attention slab weights (wq_b, wo_a's groups, wo_b) as FP16, w16_once_elems of them
    // (null: off, or not had - the slabs reconstruct them each into w16)
    uint16_t* w16_once = nullptr;
    size_t w16_once_elems = 0;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *bounds = nullptr, *row_tok = nullptr, *pos = nullptr;
    uint8_t* eg_rows = nullptr;
    int* bounds_pin = nullptr;   // pinned: the sort's bounds, read back
    // the CPU lane's experts streamed in: ring_n VRAM slots of ring_stride bytes in kParts parts, a batch a part,
    // kParts batches in flight on the copy stream (filled / freed: each part's last copy, its last use)
    static constexpr int kParts = 4;
    uint8_t* ring = nullptr;
    size_t ring_stride = 0;
    int ring_n = 0;
    cudaStream_t cs = nullptr;
    cudaEvent_t filled[kParts] = {}, freed[kParts] = {};
    // in place (file_experts): each streamed expert's file region read into pinned staging (S slots - ring_n, more on
    // the flow with DS4_STAGE_N - of stage_stride bytes, a part's slots free again once its `filled` copies are done) by the readers, in parallel,
    // and DMA'd from there - a copy from the files' pages goes through the driver's staging, one thread at ~1.3 GB/s
    uint8_t* stage = nullptr;
    size_t stage_stride = 0;
    std::unique_ptr<Pool> readers;
    cublasHandle_t blas = nullptr;
    // host-mapped: a chunk's hand-off to the next group (h | pre | topk | cand), two in flight (slot = chunk % 2), each
    // with its flag, and the next group's members' acknowledgements [slot][member] (32 apart)
    uint8_t* hand[2] = {nullptr, nullptr};
    unsigned* hand_flag[2] = {nullptr, nullptr};
    unsigned* hand_ack = nullptr;
    unsigned arb_seq = 0;        // the group's chunk all-reduces
    // pinned, two chunks in flight: the embedding rows (the first layer's device), the Engram rows (an Engram layer's)
    uint16_t* emb_host[2] = {nullptr, nullptr};
    uint8_t* eg_host[2] = {nullptr, nullptr};
    cudaEvent_t emb_ev[2] = {nullptr, nullptr}, eg_ev[2] = {nullptr, nullptr};
    double stream_bytes = 0;     // the experts streamed in, in all
    // the flow (a GGUF's in-place experts, every layer of the device on it alone; DS4_FLOW=0: the batches above): a
    // (chunk, layer)'s streamed experts are those it routes to - its likely ones (the profile's chance of a pick by the
    // chunk's tokens) queued once the layer before it routed, read while that one computes, the rest of its picks once
    // it routed itself; never an expert no token of the chunk picked unless it was a likely one.  Each item read by io
    // (unbuffered: the disk straight into its staging slot, every part of every item in flight at once - none waits
    // for a batch) or taken from its RAM-tier slot as it lies, then DMA'd into the ring in order, item i into slot
    // i % ring_n (its read into staging slot n_file % S; S = ring_n unless DS4_STAGE_N); the compute takes the items as
    // they land, in runs of what has.
    struct Item {
        int il = 0, k = 0, e = 0, v = 0, stage = -1;   // layer, chunk, expert, virtual slot, staging slot (-1: RAM)
        int mi = 0;                                    // the device's member index in the layer's group
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};   // where its gate / up / down (its parts) lie once read
        const uint8_t* ram = nullptr;                  // its RAM-tier blob, copied into staging (not pinned there)
        uint8_t* keep = nullptr;                       // (layer-major) its layer-cache slot: what is read kept there
        int64_t after = -1;                            // (layer-major) the items up to this one read first
        std::atomic<int> left{0};                      // parts still being read
        std::atomic<bool> bad{false};
    };
    bool flow = false;
    int dev = -1;                            // the device (index into devs) this chunk state belongs to
    std::unique_ptr<IoQueue> io;
    std::deque<Item> items;                  // items[0] is item `base`; the indices below are absolute
    size_t base = 0, n_read = 0, n_dma = 0, n_comp = 0;
    uint64_t n_file = 0;                     // file items so far (staging slot = ordinal % S)
    int S = 0;                               // staging slots, stage_stride apart
    std::vector<int64_t> stage_item;         // staging slot -> the item last read into it (-1: none)
    std::vector<cudaEvent_t> stage_ev, ring_fill, ring_free;   // a staging slot's DMA out; a ring slot's DMA in, its use
    int spec_k = -1, spec_il = -1, n_spec = 0;   // the likely items queued last: (chunk, layer), how many, from where
    size_t spec_begin = 0;
    std::vector<int> fl_layers, fl_T;        // the device's trunk layers in order; the run's chunk sizes
    int fl_yoco = -1, fl_cut_k = 0, fl_cut_r = 0;   // Run::yoco / cut_k / cut_r: what a layer's chunk streams for
    int bc_n = 0, bc_w = 0;                  // a batch's compacted bounds: bc_n regions of bc_w, by batch count
    int *bc_pin = nullptr, *bc_dev = nullptr;
    uint64_t n_batch = 0;
    // DS4_BC_ONCE (Impl::bc_once, ds4_flow_batches.hpp): a layer's batches and their bounds known up front (bc_list), in
    // one copy (1) or read through bc_map, bc_pin's device address (2: bc_pin mapped); bc_done - the last layer's
    // batches done on the compute stream, waited for before their regions are written again; vs_ev - the virtual
    // slots flow_spec decided for (chunk vs_k, layer vs_il, member vs_mi) copied to that member's vs_dev on cs (-1: none)
    int* bc_map = nullptr;
    std::vector<FlowBatch> bc_list;          // (batch bc_n0 + j: bc_list[j])
    uint64_t bc_n0 = 0;
    cudaEvent_t bc_done = nullptr, vs_ev = nullptr;
    int vs_k = -1, vs_il = -1, vs_mi = -1;
    double wait_ms = 0;                      // the compute waiting for a read
    uint64_t it_file = 0, it_ram = 0, it_spec = 0, it_spec_unused = 0;
    uint64_t io_d0 = 0, io_b0 = 0;           // io's unbuffered / buffered bytes when the run started
    double io_busy0 = 0;                     // io's readers' busy ms then (summed over the readers)
    IoQueue::Split io_s0;                    // io's split jobs then (DS4_IO_SPLIT)
    // layer-major (a run of several chunks, every layer on this device): the layers one after another, each through
    // all the chunks; a chunk's state between its layers (h | pre | topk | cand, chunk k at k chunk_hand_bytes()) in
    // host memory, and the layer cache - RAM slots where the experts a layer's chunk reads from the files are kept for
    // its later chunks, so each is read once a run
    bool lm = false, lm_pinned = false;
    uint8_t* lm_state = nullptr;
    size_t lm_state_bytes = 0;
    uint8_t* lc = nullptr;                   // lc_bytes of slots, a layer's experts' blobs side by side
    size_t lc_bytes = 0;
    // DS4_LM_PRE=1 (lm_prealloc; after DS4_LC_ONCE, 9aae6eee): the layer cache (lc_cap bytes) and the chunks' states
    // (st_cap, pinned or not) allocated and faulted in at the load and kept for the process - lc / lm_state point
    // into them while a run is on; freed with the model
    uint8_t *lc_kept = nullptr, *st_kept = nullptr;
    size_t lc_cap = 0, st_cap = 0;
    bool st_kept_pinned = false;
    int lc_il = -1, lc_used = 0, lc_n = 0;   // the layer the cache holds, its slots taken, its slots in all
    size_t lc_first = 0, lc_chk = 0;         // the layer's first item; the items before lc_chk all read
    std::vector<int> lc_of;                  // expert -> its slot (-1: none)
    std::vector<int64_t> lc_writer;          // slot -> the item reading its expert into it
    uint64_t it_lc = 0;                      // items taken from the layer cache
    // DS4_LM_STATE_OVERLAP (layer-major, ls_begin): the states' moves on two copy streams of their own, ls_out (to host
    // memory) and ls_in (back), off d.st - ls_mode 1: ping-pong, two state sets (ls_h[0] .. the chunk's own buffers,
    // [1] a second set in VRAM for the run), a unit's state loaded into the other set while the unit before computes;
    // 2: one set, a state's save and the next one's load piece by piece, both directions at once; 0: off (both on d.st)
    static constexpr int kLsPieces = 8;      // (h in this many row slices; then pre, topk, cand)
    int ls_mode = 0;
    float *ls_h[2] = {nullptr, nullptr}, *ls_pre[2] = {nullptr, nullptr};
    int *ls_tk[2] = {nullptr, nullptr}, *ls_cd[2] = {nullptr, nullptr};
    size_t ls_bytes[4] = {};                 // a set's h, pre, topk, cand bytes
    cudaStream_t ls_out = nullptr, ls_in = nullptr;
    // a set's last unit done on d.st, its last save out, its last load in; the last save out; mode 2's pieces out
    cudaEvent_t ls_comp[2] = {}, ls_save[2] = {}, ls_load[2] = {}, ls_saved = nullptr, ls_pc[kLsPieces + 3] = {};
};

struct Layer {
    int il = 0, dev = 0, ratio = 0;
    // its group (tensor parallel: nm members, each a copy of the device state in mates, member mi's share of the
    // heads, of wo_a's groups, of wo_b's rows and of the FFNs: gate / up columns, down rows; its part of an expert
    // laid out as hlay, hblob bytes - the host state, the CPU lane and the full blob layout stay with member 0)
    int grp = 0, nm = 1, mi = 0;
    std::vector<Layer> mates;
    x3::ExpertLayout hlay;
    size_t hblob = 0;
    int ff0 = 0, ffn = 0;     // this member's FFN columns [ff0, ff0 + ffn) (split128 of n_ff; the whole one alone)
    // expert parallelism: the routed experts' n_ff slice [ep_ff0, ep_ff0 + ep_ffn) this process's CPU lane computes of a
    // pick it sent tagged slice (H', DS4_EP_K: rx_split sets [0, 256 k); each cpu_experts call is given its range by the
    // pick's lane, ds4_rx_split.hpp - these fields are what ep_slices_ok checks) - ep_ffn -1: the whole n_ff (resolved at
    // load, ep_resolve); a slice on a layer whose experts another path computes is refused there (ep_slices_ok)
    int ep_ff0 = 0, ep_ffn = -1;
    // this member's heads [ah0, ah0 + ahn) and wo_a groups [og0, og0 + ogn) - all of them alone; in a group of more
    // members than the heads split among evenly (head_members: 9, 7 or 6 over 64 heads), the first ones none: their
    // attention adds nothing to the sum
    int ah0 = 0, ahn = 0, og0 = 0, ogn = 0;
    // the chunked prompt path: each expert's virtual slot (its VRAM slot, or n_slots + its place among the CPU lane's
    // experts, which a chunk streams in that order) - pinned on member 0, a device copy on each member
    int* vs_pin = nullptr;
    int* vs_dev = nullptr;
    std::vector<int> host_list;
    std::string name;         // "layers.N." or, for the DSpark drafter's blocks, "mtp.N."
    int n_exp = 0, k_exp = 0; // routed experts, and how many a token takes (the drafter's: 128, 3)
    bool mtp = false;         // a DSpark block (index stage = il - n_layers)
    bool kv_src = false, ix_src = false, owns_k = false, cand_src = false, uses_cand = false;
    int src = -1;      // the kv source whose compressed KV this layer reads (index into Impl::srcs)
    int ik_src = -1;   // the kv source whose index keys its indexer reads
    int eg = -1;       // engram layer index, or -1
    x3::Mat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_bias = nullptr, *eg_w = nullptr,
          *gate_bias_vl = nullptr;   // (the image tokens' routing bias, with the vision tower)
    uint16_t* i_wproj = nullptr;
    float* gate_w = nullptr;   // the router's weights, F32 (DeepSeek's inference computes the gate in F32)
    uint16_t* gate_bf = nullptr;   // U9 route: instead, a BF16 file's rows as they are (gemv_bf16 widens them: the same F32)
    uint8_t* win = nullptr;                     // [win] rows of kKvFp8 (kv_row_bytes)
    float *c_kv = nullptr, *c_score = nullptr;  // the compressor's partial group (ratio > 1)
    // a run of chunks' mid-run checkpoint (feed_mark): the ring and the partial group as they were before a row
    uint8_t* snap_win = nullptr;
    float *snap_kv = nullptr, *snap_sc = nullptr;
    // routed experts
    x3::ExpertLayout lay;
    size_t blob = 0;                            // bytes of one expert's blob
    int n_vram = 0;
    bool cpu_only = false;   // a role's type has no device kernel: every expert of this layer is the CPU lane's
    uint8_t* vram_base = nullptr;
    std::vector<int> slot;                      // expert -> VRAM slot or -1
    std::vector<int> ram;                       // expert -> RAM-tier slot or -1 (a GGUF's in-place experts; empty: none)
    std::vector<uint8_t*> host;                 // expert -> host (non-null: on the CPU lane; its halves in hh)
    // the CPU lane keeps an expert as NH column parts, part X on NUMA node X: gate / up columns [X n_ff / NH, ..),
    // down columns [X dim / NH, ..), each a standalone EXL3 matrix (its own trellis, the full suh, its part of svh)
    struct Half {
        size_t trellis[3], suh[3], svh[3], bytes;
        int k[3], n[3];
    } hl;
    std::vector<std::array<uint8_t*, 4>> hh;     // expert -> its parts on the nodes
    // the expert cache: one VRAM slot and one set of host parts kept free, so an expert can move in while the one it
    // replaces moves out; heat is how much each expert was used lately (the profile's share to start with)
    int n_slots = 0, spare_vram = -1;
    std::array<uint8_t*, 4> spare_host = {nullptr, nullptr, nullptr, nullptr};
    std::vector<float> heat;
    std::vector<int> cpu_hits;                  // CPU-lane experts used since the cache last looked
    bool swap_pending = false;
    unsigned long long* tab_pin = nullptr;      // pinned: the table entries a swap writes
    // a speculative window's backups (trunk layers): the ring rows it replaced, and a compressor's state before it and
    // the window's raw rows (kv, then score) to replay the accepted ones
    uint8_t* ring_bak = nullptr;
    float *c_snap = nullptr, *c_rows = nullptr;
    unsigned long long* vram_tab = nullptr;     // device [n_exp]
    std::vector<uint8_t> zc;                    // (the RAM store) expert -> 1: its store slot is GPU-visible, the device computes it over PCIe
    unsigned long long* zc_tab = nullptr;       // device [n_exp]: those slots' device addresses (0: none) - the decode token's zc_stage reads it
    std::vector<std::array<x3::Mat, 3>> vm;     // expert -> its gate, up, down as EXL3 views of its VRAM blob
    std::vector<uint64_t> region_off;           // expert -> file region (offset, bytes, file)
    std::vector<uint64_t> region_bytes;
    std::vector<int> region_file;
    size_t piece_src[3][3];                     // part (gate, up, down) x (trellis, suh, svh): offset in the region
    size_t piece_bytes[3][3];
    size_t dense_bytes = 0;
    // a GGUF's MXFP4 experts (lay.fmt == gg::kFmtMXFP4): matrix r (gate, up, down) of expert e lies at mx_off[r] +
    // e mx_stride[r] in file mx_file[r] - loaded from there, and read in place (the file's mapping) by the CPU lane
    int mx_file[3] = {-1, -1, -1};
    uint64_t mx_off[3] = {0, 0, 0}, mx_stride[3] = {0, 0, 0};
};

struct Src {
    int layer = 0, ratio = 1;
    int64_t cap = 0;
    // per device (nullptr where no layer reads it): rows of kKvFp4 (the compressed KV), kKvFp4p (the index keys)
    std::vector<uint8_t*> ckv, ik;
    uint8_t *h_ckv = nullptr, *h_ik = nullptr;   // pinned: a window's new rows, on their way to the other devices
    cudaEvent_t ev_ckv = nullptr, ev_ik = nullptr;
    // threads: whether devices outside the source's group read it, and the host-mapped hand-off of a window's new rows
    // (ckv rows, then ik rows at kWin * hd) and its flag
    bool remote = false;
    uint8_t* x_stage = nullptr;
    unsigned* x_flag = nullptr;
    // pipelined chunks: a second staging slot (slot 1; slot 0 is x_stage), its flag, and the readers' acknowledgements
    // [slot][device] (32 apart); readers: the devices outside the source's group that read it
    uint8_t* x_stage1 = nullptr;
    unsigned* x_flag1 = nullptr;
    unsigned* x_ack = nullptr;
    std::vector<int> readers;
    uint8_t* stage(int slot) const { return slot ? x_stage1 : x_stage; }
    unsigned* flag(int slot) const { return slot ? x_flag1 : x_flag; }
};

// a host thread that enqueues one device's work (threads mode)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    bool idle() {
        std::lock_guard<std::mutex> lk(mu);
        return !busy;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};

struct Group {
    std::vector<int> mem;   // device indices, member order (member 0 the lead)
    ArGroup ar, arb;        // a window's all-reduces, a chunk's
    // the head group: the members' logits shares on their way to the lead, two slots (by seq), their flags and the
    // lead's acknowledgements (host-mapped)
    float* lg_stage = nullptr;
    unsigned *lg_flag = nullptr, *lg_ack = nullptr;
};

constexpr int kMaxL = 64;   // layers (the trunk's and the drafter's) a window plan covers

// what a window's layers need from the host, decided before they run (threads enqueue the layers concurrently): the
// mailbox seq of each layer (async: the device publishes its routing), the indexer's selection sizes and the candidate
// blocks as each layer sees them, the window's hand-off seq
struct WinPlan {
    // (no default: whoever makes a plan says whether it is an EP window)
    explicit WinPlan(bool ep_window) : ep(ep_window) {}
    bool par = false, async = false;
    bool ep;                // an EP window (decode, a draft's verification): the layers' n_ff slices (Layer::ep_ff0)
    unsigned win = 0;
    int k = 0;              // (a run of chunks) the chunk's index: its hand-off slot k % 2
    int snap_t = -1;        // (a run of chunks) the chunk's row whose state before it the layers keep (-1: none)
    int eg_slot = -1;       // (a run of chunks, layer-major) the Engram rows' slot (-1: k % 2)
    unsigned vl = 0;        // (a window) bit t: token t is inside a picture's span (routed with the image bias)
    bool defer = false;     // DS4_EP_DEFER: a decode window's mailbox parts each added a layer late (Impl::late_*)
    unsigned seq[kMaxL] = {};
    int topk[kMaxL][kWin] = {}, cand[kMaxL][kWin] = {};
};

}  // namespace

// ------------------------------------------------------------------ the model
struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    std::vector<Dev> devs;
    std::vector<Layer> L;
    // a layer range (Options::stage_begin / stage_end): the trunk's layers [lo, hi) here.  The others keep their Layer
    // and its roles (the window plan's selection sizes follow every layer) but no device (dev -1), experts, weights or
    // memory; every loop that runs or allocates layers passes them by.  ranged: not the whole trunk - [0, nl) asked
    // for outright is the default run
    int lo = 0, hi = 0;
    bool ranged = false;
    bool local(int il) const { return il >= g.nl || (il >= lo && il < hi); }
    static bool elsewhere(const Layer& l) { return l.dev < 0; }
    // a tensor of the pack this process loads (a layer range: not the other layers' "layers.N." ones; a remote stage's
    // worker: no head)
    bool tensor_here(const std::string& n) const {
        if (o.stage_worker && n.rfind("head.", 0) == 0) return false;
        return !ranged || n.rfind("layers.", 0) != 0 || local(std::atoi(n.c_str() + 7));
    }
    // (a layer range) an Engram layer among its layers
    bool eg_here() const {
        for (int il : g.eg_layers)
            if (local(il)) return true;
        return false;
    }
    // a remote stage (Options::remote_stage / stage_worker): the main process's link after its last layer, and the rows
    // that cross it, pinned - link_tx out, link_rx in (kWin rows of full_floats each, Model::StageGeo's layout)
    Model::StageRun stage_hop;
    std::function<bool(int, std::string&)> stage_commit;
    std::function<bool(std::string&)> stage_reset;
    Model::StageCkpt stage_ck;        // (the main process) the link's checkpoints and slots (Options::stage_ckpt)
    bool stage_lost = false;          // (the main process) a hop, a commit or the worker's reset failed: only position 0
                                      // goes on
    int stage_t = 0;                  // (the worker) the last window's rows, while its backups stand (2 or more; 0: none)
    std::string stage_err;
    float *link_tx = nullptr, *link_rx = nullptr;
    // (stage_bf16) the same rows as the link carries them, compact: wire_tx / wire_rx, kWin rows of handoff_floats()
    // words, packed from link_tx after a window's last sync and unpacked into link_rx before its first copy
    float *wire_tx = nullptr, *wire_rx = nullptr;
    double conv_in_ms = 0, conv_out_ms = 0;   // (stage_conv)
    int64_t conv_in_n = 0, conv_out_n = 0;
    bool linked() const { return o.remote_stage || o.stage_worker; }
    bool bf16_rows() const { return o.stage_bf16 && linked(); }
    stage::RowGeo row_geo() const {
        stage::RowGeo r;
        r.h = (int64_t) g.hc * g.dim;
        r.pre = 8;
        r.topk = (int64_t) topk_stride();
        r.cand = (int64_t) cand_stride();
        return r;
    }
    // checkpoints (checkpoint / restore / pin): not in a remote stage, except a worker (the main process asks, over the
    // link, for what it keeps itself) and a main process that keeps them across the link
    bool ckpt_ok() const {
        return !linked() || o.stage_worker || (o.stage_ckpt && stage_ck.save != nullptr && stage_ck.restore != nullptr);
    }
    // a slot: the whole trunk in one process, or the two ends of a remote stage that keeps checkpoints (each its layers')
    bool slot_ok() const { return !ranged || (linked() && ckpt_ok()); }
    const char* ckpt_link_note() const {   // (the load line's)
        if (o.stage_worker) return ", checkpoints and slots as the main process asks";
        if (o.remote_stage && o.stage_ckpt) return ", checkpoints and slots across the link";
        return linked() ? ", no slots, no checkpoints" : ", no slots";
    }
    // a row in the fp32 block layout, and the link's (the compact one with stage_bf16)
    int64_t full_floats() const { return row_geo().full(); }
    int64_t handoff_floats() const { return bf16_rows() ? row_geo().compact() : full_floats(); }
    // a remote stage's prompt chunks (the chunked path, chunk > 0): the main process's sends (Model::StageChunks) from
    // two slots of chunk rows of full_floats taken in turn (link_cx; the compact rows: ChunkLink packs them); the
    // worker's four buffers the link lends (link_pf: two in, two out, chunk rows of handoff_floats - the link's row),
    // with bf16 rows two fp32 ones the chunk runs on (link_st: in, out - stage_prefill unpacks into the first and packs
    // the second's reply rows), and the chunk it runs - its rows in (stage_in, block by block) and out (stage_out, its
    // rows from stage_skip on, row by row, full_floats each).  All pageable, committed as written on Linux (at the
    // allocation on Windows; link_maps: freed at the end)
    Model::StageChunks stage_chunks;
    float* link_cx[2] = {nullptr, nullptr};
    float* link_pf[4] = {nullptr, nullptr, nullptr, nullptr};
    float* link_st[2] = {nullptr, nullptr};
    std::vector<std::pair<void*, size_t>> link_maps;
    const float* stage_in = nullptr;
    float* stage_out = nullptr;
    int stage_skip = 0;
    bool stage_last = true;           // (the worker) the chunk is its main process's run's last
    // (the main process) a run's chunks sent so far, for its replies' thread (each reply read once its chunk is out:
    // a read waits at most the link's time-out); link_stop: the run ended
    std::mutex link_mu;
    std::condition_variable link_cv;
    int link_sent = 0;
    bool link_stop = false;
    std::vector<Src> srcs;
    std::vector<Group> groups;
    std::vector<std::unique_ptr<Worker>> workers;   // threads mode: one per device
    bool par = false;                 // threads mode (the windows: decode)
    bool prompt_par = false;          // the prompt path's chunks (a thread a GPU), as --tp / --threads asked
    unsigned win_seq = 0;             // the windows the threads ran (the hand-offs' seq)
    // the chunked prompt path (threads mode): chunk tokens a pass, runs shorter than chunk_min in windows
    int chunk = 0, chunk_min = 512;
    // DS4_BC_ONCE (bc_once_mode, ds4_flow_batches.hpp; the load reads it): the flow's expert stage's batch bounds 0 a
    // copy a batch on the compute stream, as before; 1 written up front once a layer routed and sent in one copy; 2 read
    // by the kernels from mapped host memory.  1 and 2 also send a layer's virtual slots on the copy stream when
    // flow_spec decides them, not on the compute stream before the sort
    int bc_once = 0;
    // DS4_LC_THP=0 (the load reads it): the prompt's host buffers - the layer-major prompt's layer cache and pageable
    // chunk states, a remote stage's replies - mapped without MADV_HUGEPAGE, on 4 KiB pages: a fault near the cgroup's
    // memory.high then reclaims a page, never compacts memory for a huge one.  Unset, empty or 1: huge pages advised, as before
    bool lc_thp = true;
    // DS4_LM_PRE=1 (the load reads it; lm_prealloc): the layer-major prompt's host memory allocated once at the load
    bool lm_pre = false;
    int chunk_own = -1;             // (a remote stage's main process, once set_stage_chunk ran) the chunk of its load
    static constexpr bool kChunkedPromptDefault = true;   // the layer-major prompt is the default (gate: KL 0.1685 vs 0.1686, 8K prompt 6x, decode within 2% - ds4 notes-dsv41-r97)
    static constexpr int kFlowChunkMin = 2;   // chunk_min with every device's layers on the flow (see load)
    // --prefill-chunk at most (route() runs in kVlRows-row slabs).  The default stays kVlRows: a 16384-token chunk
    // reads a 16K prompt's experts once instead of twice (V100: 461.8 -> 321.3 s, 736 -> 506 GB read) but its buffers
    // take ~1.9 GB of VRAM experts (745 -> 644 on a 32 GB V100) from every decoded token
    static constexpr int kMaxChunk = 4 * kVlRows;
    std::vector<std::unique_ptr<Chunk>> chunks;   // per device
    size_t eg_cap = kWin;             // tokens h_eg holds
    int tap_cap = kWin;               // rows h_taps holds (a chunk seeds the drafter with its last 128)
    static constexpr int kYocoMin = 1024;   // a long prompt's skip (forward_chunks): the fewest rows worth skipping
    bool logits_all = false;          // the last feed was a chunk with logits for every row (chunk_logits)
    std::vector<float> chunk_logits;
    std::function<void(int64_t)> progress;   // a run of chunks: the tokens done so far (the main thread, while it waits)
    // a Stop while a run of chunks goes through the layers (Model::set_cancel, asked while forward_chunks waits): one
    // GPU stops after the layer (or the layer's chunk) it is on (halt); several stop at a chunk every one of them
    // reaches - none starts chunk halt_k or later, halt_k one past the latest any started - so no device waits on a
    // hand-off that never comes (Project Maya v1.1.20)
    std::function<bool()> cancel;
    std::atomic<bool> halt{false};
    std::mutex halt_mu;
    int halt_k = INT_MAX, halt_seen = -1;
    bool halted = false;                      // the last run ended early on a Stop
    static constexpr const char* kHalted = "stopped";
    bool chunk_go(int k) {   // chunk k may start on this device (several GPUs)
        std::lock_guard<std::mutex> lk(halt_mu);
        if (k >= halt_k) return false;
        halt_seen = std::max(halt_seen, k);
        return true;
    }
    bool halt_now() const { return devs.size() == 1 && halt.load(std::memory_order_acquire); }
    // a run's early end on device di: the expert reads still in flight land first (they count down items the next run
    // clears), then its streams are idle
    bool halt_end(int di, std::string& err) {
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(devs[(size_t) di].id);
        if (c.flow)
            for (size_t i = c.base; i < c.n_read; ++i)
                while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
        if (c.cs != nullptr) cudaStreamSynchronize(c.cs);
        cudaStreamSynchronize(devs[(size_t) di].st);
        err = kHalted;
        return false;
    }
    std::vector<cudaEvent_t> chunk_done;     // on the last layer's device: chunk k through every layer
    std::unique_ptr<std::atomic<int>[]> chunk_rec;   // chunk k's event recorded
    // a prompt the flow could take: the flow or windows of 16, whichever this PC runs it faster (pick_path) - this
    // PC's speeds measured at load (calibrate_paths), each path's predictions scaled by how long it really took
    struct PathModel {
        bool on = false;
        double dma = 0, copy = 0, disk = 0, cpu = 0, cpu_cold = 0;   // bytes/s (disk, cpu_cold 0: not measured)
        // the flow's taken / predicted so far [0] by the prompt's size class (2-31, 32-127, 128-511, 512 tokens and
        // more: a short one's fixed costs weigh more), and how many prompts set it ([1] unused)
        double k[2][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}};
        int64_t kn[2][4] = {};
        double win_s = 0;                                              // windows: seconds a window took, so far
        int64_t win_n = 0;
        int last_cls = 0, last_nw = 0;                                 // the last pick's size class, its windows
        double m = 1;                                                  // the picks' spread, against the profile's
        int64_t n_m = 0;
        int last = -1;                                                 // the last pick: 1 the flow, 0 windows
        double last_pred = 0;                                          // its prediction (s, before its k)
    } pm;
    static constexpr int kPickMax = 2048;   // (from here on the flow: windows' work grows with every 16 tokens)
    static constexpr double kPickRes = 0.9;  // (this share of the CPU-lane experts in RAM: windows if predicted faster)
    std::atomic<int> lm_layers{0};           // (layer-major) the layers every chunk went through
    WinPlan wplan{false};                    // (plan_window sets each window's kind)
    int head_dev = 0;
    x3::Mat head;
    float* norm_w = nullptr;          // on the head device
    // the drafter: main_proj / main_norm (on its first block's device), its final norm, the markov head and the
    // confidence head (on the head device: the drafts' logits come from the trunk's head)
    x3::Mat d_main_proj;
    float *d_main_norm = nullptr, *d_norm = nullptr, *d_conf = nullptr;
    uint16_t *d_mk_embed = nullptr, *d_mk_head = nullptr;   // BF16 [vocab][rank], F16 [vocab][rank]
    // a head group of two or more: the markov steps over its members (each its rows of the bias and an argmax of its
    // share, the global one through host-mapped slots [step][member] {value, index, seq}) in place of the lead's whole
    // rows - E host-mapped (each member reads the picked token's row), a draft's seq in mk_seq
    bool mk_split = false;
    uint16_t* h_mk_embed = nullptr;
    unsigned* mk_slots = nullptr;
    unsigned mk_seq = 0;
    int d_dev = 0;                    // the drafter's first block's device
    bool spec_win = false;            // the window being run is a draft's verification (layer() keeps backups)
    float* h_taps = nullptr;          // pinned [kWin][targets][dim]: the last window's taps
    cudaEvent_t ev_seed = nullptr;    // the drafter's seeding read h_taps: the next window's taps wait for it
    float* d_confv = nullptr;         // on the head device
    int* d_ints = nullptr;            // on the head device: drafts in, picks out
    float conf_threshold = std::getenv("DS4_DSPARK_CONF") ? (float) std::atof(std::getenv("DS4_DSPARK_CONF")) : 0.5f;
    // a sampled round's drafts: the distribution q each one was drawn from (its kept list, on the head device) for
    // verify()'s rule - a point at the draft for the drafter's argmax; and the chain they were drawn with (verify()
    // checks it is its own)
    strata::dsv41k::SpecList* d_specq = nullptr;
    std::vector<strata::dsv41k::SpecList> h_specq = std::vector<strata::dsv41k::SpecList>(kWin);
    bool drafts_q = false;
    float dq_temp = 0.0f, dq_top_p = 1.0f, dq_min_p = 0.0f;
    int dq_top_k = 0;
    uint64_t dq_seed = 0, dq_counter = 0;
    // (tests) DS4_DSPARK_SAMPLE=1: a sampled request's drafts drawn from the drafter's distribution through the
    // request's chain, as DeepSeek's reference does (else its argmax: the same acceptance on Uranus, 31-32% of 5
    // drafts a round either way); DS4_DSPARK_QTEMP=f: those draws at f x the request's temperature (any q keeps the
    // rule exact); DS4_DSPARK_MAX=n: at most n drafts a round; DS4_DSPARK_DUMP=file: the drafter's inputs and outputs
    // (tools/ds4/dspark_ref.py checks them against DeepSeek's checkpoint)
    bool sample_drafts = std::getenv("DS4_DSPARK_SAMPLE") != nullptr && std::atoi(std::getenv("DS4_DSPARK_SAMPLE")) != 0;
    float q_temp_scale = std::getenv("DS4_DSPARK_QTEMP") ? (float) std::atof(std::getenv("DS4_DSPARK_QTEMP")) : 1.0f;
    int max_drafts = std::getenv("DS4_DSPARK_MAX") ? std::atoi(std::getenv("DS4_DSPARK_MAX")) : 0;
    std::FILE* dump_d = std::getenv("DS4_DSPARK_DUMP") ? std::fopen(std::getenv("DS4_DSPARK_DUMP"), "wb") : nullptr;
    float* h_dump = nullptr;          // (the dump's pinned copies: each block's stream, each block's ring)
    uint16_t* h_dump_ring = nullptr;

    uint16_t* embed_pinned = nullptr;
    std::vector<float> freqs_win, freqs_comp;
    std::vector<std::unique_ptr<Pool>> pools;   // one per NUMA node, its workers on that node's cores
    // DS4_LANE_PROF=1: the lane's phases timed (Stats::lp_*, the decode's "decode lane phases" line) - clocks read,
    // nothing computed otherwise
    const bool lane_prof = std::getenv("DS4_LANE_PROF") != nullptr && std::atoi(std::getenv("DS4_LANE_PROF")) != 0;
    // DS4_IQ_PF: the i-quant CPU lane's hint on its experts' ranges each call (madvise WILLNEED; on pages the cache holds
    // it only costs the walk - 1.7 ms a token of 160 slices on CT 580, lane-bench file vs file-nopf): unset as before
    // (every range; DS4_IQ_PF_COLD=1 prefetch_cold), 0 none, 1 every range, N > 1 the experts the layer's last N calls did
    // not take only (pf_seen: the layer's call count at an expert's last call, pf_calls: the layer's calls)
    const int iq_pf = std::getenv("DS4_IQ_PF") != nullptr ? std::max(0, std::atoi(std::getenv("DS4_IQ_PF"))) : -1;
    std::vector<uint32_t> pf_seen, pf_calls;
    size_t pf_ne = 0;
    // DS4_IQ_DOWN_PARTS=dir: the CPU lane's down over a slice [ff0, ff0 + ffn) read from a parts pack of the down
    // matrices alone (ep_parts.py --down-only A:B: each expert's dim rows cut to the blocks [A, B), back to back) instead
    // of the pack's rows in place - there a row's [0, k) is strided (594 bytes a row, 396 of them read at k 6 on IQ2_XXS)
    // and the prefetchers bring the rest of the row too: the down's DRAM time at k 6 is the whole row's (lane-bench file
    // 11.10 ms at k 6 vs 11.40 at k 9).  A slice the parts end at (ff0 + ffn = 256 B, A <= ff0 / 256) of an expert read
    // from the files (not a RAM-tier copy, no store) takes them; any other the pack's.  init_down_parts checks the parts:
    // the pack's (pack_hash), every expert of a covered i-quant layer one part of the layer's down type and row layout.
    ::strata::ds4::ep::Parts dparts;
    struct DownParts {
        int a = 0, b = 0;
        size_t ub = 0, rb = 0;                // a row's bytes a block, and the part's row (b - a) ub
        std::vector<const uint8_t*> at;       // expert -> its part's row 0, block a (empty: the layer has none)
    };
    std::vector<DownParts> dpl;               // by layer index
    bool dp_pack = false;                     // (the cut check's W call: the pack's down, the parts' left out)
    // DS4_IQ_WARM=split: the load's read-ahead of the CPU lane's experts (load_experts_mx: each one whole, read into a
    // buffer - the page cache warm, the lane's mapping not) left out, warm_split after the expert server's upload
    // instead: of each CPU-lane expert only what the lane reads of a pick - its layer's slice [ep_ff0, ep_ff0 + ep_ffn)
    // of the gate / up rows, the down from the parts where they serve that slice (DS4_IQ_DOWN_PARTS), else the pack's
    // down whole - read in through the lane's own mapping (os::populate: the page cache and the page tables; pread where
    // the kernel has no MADV_POPULATE_READ), the most-routed up to warm_bytes (DS4_IQ_WARM_ORDER: in what order).  At k 6 with the parts that is 2/3
    // of an expert's bytes: lane-bench-report.md section 13 - the whole read-ahead's 97 GB held 12559 of CT 580's 15229
    // CPU-lane experts, and the parts, outside it, were pushed out by the arms before.  Unset or empty: as before; any
    // other value fails the load
    const char* iq_warm_env = std::getenv("DS4_IQ_WARM");
    const bool iq_warm_split = iq_warm_env != nullptr && std::strcmp(iq_warm_env, "split") == 0;
    // DS4_IQ_WARM_TOUCH (with split): 1, warm_split reads a byte of every 4 KiB page it read in, through the lane's
    // mapping, so each one's page table entry has its accessed bit when the kernel next looks for pages to evict (the
    // guess of lane-bench-report.md section 17, that MADV_POPULATE_READ maps them without it on x86). Section 18: the
    // populated pages were promoted to the youngest generation within the populate, with no aging - they had the bit;
    // the wt-run A/B (slot-20261025T2329Z): touched or not, the same speed (28.2-28.8 tok/s cold, 30.7-31.3 hot). Unset,
    // empty or 0: no touch; any other value fails the load
    const char* iq_warm_touch_env = std::getenv("DS4_IQ_WARM_TOUCH");
    const bool iq_warm_touch = iq_warm_touch_env != nullptr && std::strcmp(iq_warm_touch_env, "1") == 0;
    // DS4_IQ_WARM_ORDER (with split): cold, warm_split reads its experts (still the most-routed up to warm_bytes) the
    // least-routed first, so what the read-ahead pushes out of a full page cache itself is its coldest pages, not its
    // hottest (section 17's guess). The wt-run A/B: either order the same speed. Unset, empty or hot: the most-routed
    // first; any other value fails the load
    const char* iq_warm_order_env = std::getenv("DS4_IQ_WARM_ORDER");
    const bool iq_warm_cold_first = iq_warm_order_env != nullptr && std::strcmp(iq_warm_order_env, "cold") == 0;
    // DS4_IQ_WARM_SRV (with split): 1, warm_split reads the expert server's experts too (their slices as the CPU
    // lane's: the gate / up rows, the down from the parts) - the fills hand ~500 of them back to the lane in a decode
    // (ds4_rx_fill.hpp: the evicted one is the lane's from then on), and a config before that left their pages out of
    // the page cache made the first arm's down read them in by faults in its decode (wt-run, slot-20261025T2329Z: dn
    // 10.3-10.6 ms a token against 7.5-7.6, the decode's disk reads 1.2-2.5 MB a token more). On by default (srv-run,
    // the first arm after a drop: 30.0 tok/s, dn 8.3 ms a token, against 25.5-28.3 and 10.8-12.4 without): unset, empty
    // or 1 reads them; 0 leaves them out, as before; any other value fails the load
    const char* iq_warm_srv_env = std::getenv("DS4_IQ_WARM_SRV");
    const bool iq_warm_srv = iq_warm_srv_env == nullptr || *iq_warm_srv_env == '\0' || std::strcmp(iq_warm_srv_env, "1") == 0;
    // DS4_LANE_TRACE=file: the i-quant CPU lane's calls as cpu_experts_iq gets them, for lane-bench --trace (its picks,
    // their n_ff slice and tokens): "LANETRC1", then a record a call - u8 layer, u8 ff0 / 256, u8 (ff0 + ffn) / 256,
    // u8 entries, then each entry's int16 expert and u8 tokens (flushed every 4096 calls and at the end)
    std::FILE* lane_tr = open_lane_trace();
    uint32_t lane_tr_n = 0;
    static std::FILE* open_lane_trace() {
        const char* p = std::getenv("DS4_LANE_TRACE");
        std::FILE* f = p != nullptr && *p != '\0' ? std::fopen(p, "wb") : nullptr;
        if (f != nullptr) std::fwrite("LANETRC1", 1, 8, f);
        return f;
    }
    std::vector<std::vector<int>> cores;        // each node's physical cores
    int NH = 1;                                 // the CPU lane's parts of an expert (the NUMA nodes)
    // the CPU lane's experts read in place from the model's files (one part, the files' layout; the page cache keeps
    // what fits) instead of copied into RAM, and how much of them the load reads ahead (the most-routed first)
    bool file_experts = false;
    uint64_t warm_bytes = 0;
    static constexpr uint64_t kRamMargin = 8ull << 30;   // RAM left to the rest (the system, the page cache's churn)
    uint8_t* file_part(const Layer& l, int e) const {
        return pk.maps[(size_t) l.region_file[(size_t) e]].first + l.region_off[(size_t) e];
    }
    // host side of the CPU lane, a window's rows
    float *h_x = nullptr, *h_w = nullptr, *h_part = nullptr;
    int* h_idx = nullptr;
    uint8_t* h_eg = nullptr;          // the window's Engram rows: [engram layer][token] of [cols][hd] then the scales
    // the Engram rows' reader (its own threads and file handles), the rows already fetched, the window's batches (one
    // an Engram layer: asked for when the window's tokens are known, collected right before that layer)
    EgReader eg_io;
    EgRowCache eg_cache;
    std::mutex eg_cache_mu;
    std::vector<std::unique_ptr<EgBatch>> eg_win;
    // Engram layer li's embed rows, as the pack stores them: 0 MXFP8 ([scale, 32 codes] blocks, 33 bytes a 32), 1
    // ggml Q6_K (210 bytes a 256), 2 ggml IQ4_NL (18 bytes a 32: an fp16 d and 16 nibble bytes), 3 an old pack's FP8
    // codes with their E8M0 scales in a tensor of their own (see eg_issue)
    std::vector<uint8_t> eg_fmt;
    std::atomic<int64_t> eg_rows_n{0}, eg_hits_n{0}, eg_reads_n{0}, eg_wait_ns{0}, eg_io_ns{0};
    uint16_t* h_emb = nullptr;        // the window's embedding rows
    std::vector<float> c_xh, c_xsb, c_acc, c_xhd, c_xsd, c_accd, c_yd;
    // the state
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    struct Check {                    // the state at a position: every window ring, the compressors' partial groups
        int64_t pos = -1;
        std::vector<std::vector<uint8_t>> win;
        std::vector<std::vector<float>> comp;
    };
    static constexpr int kChecks = 4;
    static_assert(kChecks == stage::kCkRoles, "the link's checkpoint ids are by role");
    Check ck[kChecks];                // the end of the last prompt, the position before it, the anchor (the position
                                      // before the end of the first prompt of the run of prompts that each continued
                                      // the last in place: a chat template that drops earlier turns' reasoning at a new
                                      // user message re-renders the history from about there) and the prefix (where the
                                      // server said the prompt's shared start - system prompt and tools - ends: another
                                      // agent behind the same start goes back to it)
    bool slot_held = false;           // slot_save took this state and nothing has changed it since
    std::vector<uint32_t> prof;       // [nl][n_exp] routed-expert hits
    Stats st;
    // the head's wait (a token's long host wait on the GPU, decode and prompt): its wall time, the waiting thread's CPU
    // time in it and the CPU it began on (Stats::head_*) - whether the wait spins (DS4_CUDA_SYNC) and on which CPU
    bool head_wait(cudaStream_t s, std::string& err) {
        const int c = os::cpu_now();
        ++st.head_on[c >= 0 && c < 128 ? c : 128];
        ++st.head_waits;
        const double th = now_ms(), tc = os::thread_cpu_ms();
        CK(cudaStreamSynchronize(s));
        st.head_ms += now_ms() - th;
        const double tc1 = os::thread_cpu_ms();
        if (tc >= 0 && tc1 >= 0) st.head_cpu_ms += tc1 - tc;
        return true;
    }
    int cur_topk_n[kWin] = {}, cur_cand_n[kWin] = {};   // what the latest indexer / candidate source left, per query
                                                        // (plan_window's: the windows' plans are made in order)
    int n_logits = 0;                 // the last forward's logit rows
    // the decode path (windows of one): the device publishes each layer's routing into reqs, the service thread
    // answers in resps with the CPU lane's part, and the device waits for the answer only before the expert sum
    Req* reqs = nullptr;
    Resp* resps = nullptr;
    unsigned seq_issued = 0;
    std::atomic<int> layer_of[kRing];
    // the seq's window an EP window (WinPlan::ep).  Written with its layer_of before the layer's kernels are launched;
    // serve_lane reads both only once the device has written that request's seq - the launch, not layer_of's release /
    // acquire (the slot's previous store may hold the same layer), orders them, as it does layer_of
    bool ep_of[kRing] = {};
    // ... and a small prompt's (small_feed when the seq was issued): the decode-only statistics (the lookahead's, the
    // hybrid's decode-request tables) skip it.  Written and read as ep_of is
    bool small_of[kRing] = {};
    std::thread service;
    std::atomic<bool> quit{false};
    // the link core (--remote-experts, ds4_rx.hpp): the service thread's CPU (DS4_EP_CPU; -1: none) and that core's
    // logical CPUs, which the CPU lane's pools leave out; a window's start (forward), for serve_lane's Wake;
    // DS4_EP_WAKE=0: no Wake and the lane's 2 ms naps as without the link (A/B)
    int ep_cpu = -1;
    bool ep_freed = false;   // (no server held any expert: the link core went back to the CPU lane at the load)
    std::vector<int> ep_sibs;
    std::atomic<bool> rx_wake{false};
    bool ep_wake = true;
    // remote experts (--remote-experts, ds4_rx.hpp): the link (the service thread's after the load), the experts the
    // server holds ([layer][n_exp]), the per-request times (ms) for the percentiles
    std::unique_ptr<rx::Main> rx_link;
    std::vector<uint8_t> rx_has;
    std::vector<uint8_t> rx_pool;   // (the hybrid, U5c: the server's W, [layer][n_exp]; empty: none - rx_split)
    // the hybrid's balance (DS4_EP_POOL_BAL, on with W; rx::pool_keep): EWMAs of the T=1 replies' times - this lane's
    // ms a byte (send to its part done), the server lane's, the rest of a reply's (ms); -1: none yet.  The samples,
    // and the margin (DS4_EP_POOL_MARGIN, ms)
    // (DS4_EP_POOL_CUT, on with the balance: rx::pool_cut - a W pick's [cut, k) there, its [0, cut) here)
    bool rx_bal = false, rx_cut = false, rx_cut_chk = false;   // (rx_cut_chk: DS4_EP_POOL_CUT_CHECK=1, not for speed)
    double rx_bal_rm = -1, rx_bal_rs = -1, rx_bal_fix = -1, rx_bal_margin = 0.1;
    int64_t rx_bal_nm = 0, rx_bal_ns = 0;
    std::atomic<bool> rx_on{false};   // (rx_lost on the service thread clears it; the main thread's cache reads it)
    double rx_timeout_ms = 2.0;   // (counted from when the lane's own picks are done: until the lane's time a pick is
    bool rx_wait_fixed = false;   //  known, or always with DS4_RX_TIMEOUT_MS; rx::Gate::wait_ms)
    double rx_seed_ms = -1.0;     // (the load's warm-up requests: the round trip at 1 row the gate starts from; -1 none)
    rx::Backoff rx_back;   // the last 64 replies' lateness (8 late: the link idle for DS4_RX_BACKOFF requests)
    rx::Gate rx_gate;   // the send gate and the deadline's floor (ds4_rx.hpp)
    rx::LateBook rx_late_book;   // the late requests, for their replies' round trips if they come
    bool rx_kts = false;         // the link has given a kernel arrival time (then a late request waits for its reply's)
    static_assert(rx::kMaxRows == kWin && rx::kMaxPicks == kPicks, "a request's rows and picks are a window's");
    std::vector<rx::ReqRow> rx_rows;
    // expert parallelism H' (DS4_EP_K, ds4_rx_split.hpp): the n_ff blocks a layer's CPU-lane picks split at (rx_nb: none)
    // - a request of an EP window that goes to the server takes them tagged slice, the server's CPU lane computes their
    // [k, nb) from its parts and this one [0, k) - and the layers that split (rx_split_l[il]; checked at rx_setup).
    // DS4_EP_SPLIT_VERIFY=1: the server's reply in two sets (rx::kReqSets), the slice picks' sum checked against this
    // lane's [k, nb) of them for x in bf16 (the server's x); =2 the server's slice wrong on purpose (the check's check)
    int rx_k = 1, rx_nb = 1;   // (rx_setup: nb = n_ff / 256)
    std::vector<uint8_t> rx_split_l;
    int rx_verify = 0;
    bool rx_tagged = false;   // (H' on: a layer's slice computed only for picks sent tagged - ep_slices_ok)
    std::vector<float> rx_vx, rx_vy;   // (the verify's x in bf16 and its sums: kWin rows)
    // the server's set following the heat (DS4_RX_FILL=1, EP U5b: ds4_rx_fill.hpp, ds4_rx_fill_link.hpp): the fills'
    // socket, their budget (R a token, Q pending; the socket paces the bytes), each layer's blob as a fill ships it, the
    // VRAM expert a layer's swap under way takes out (rx_vout: no duplicate kept on the server), a tick's layers
    std::unique_ptr<rx::FillLink> rx_fill;
    bool rx_zc_noted = false;   // (DS4_RX_FILL_ZC=2: sendfile's refusal said)
    std::unique_ptr<rx::FillBook> rx_book;
    std::vector<rx::FillGeo> rx_fgeo;
    std::vector<int> rx_vout;
    std::vector<std::vector<int>> rx_thits;
    std::vector<rx::TierLayer> rx_tier;
    double rx_fill_max = 0.0;
    uint32_t rx_fill_seq = 0;
    mutable std::mutex rx_mu;
    std::vector<float> rx_rtt, rx_wait, rx_arr;
    std::vector<std::pair<float, int>> rx_late_arr;   // the late replies that came: send to arrival in the kernel, layer
    std::vector<std::pair<float, float>> rx_first;    // the gate's first 8 samples at 1 row: the sample, the estimate after
    int rx_first_n = 0;                                // (the service thread's count of them)
    bool sync_decode = std::getenv("DS4_SYNC") != nullptr;
    // a window of one through moe_gate_up / moe_down (DS4_MOE_TOKEN=0: win_experts, whose down rows run
    // before the wait for the CPU lane - the same bits, no faster)
    bool moe_token_path = std::getenv("DS4_MOE_TOKEN") == nullptr || std::atoi(std::getenv("DS4_MOE_TOKEN")) != 0;
    // DS4_EP_DEFER=1 (n+1 deferral, plain decode only; defer_setup says when it is off): layer l does not wait for its
    // mailbox part (the CPU lane's, and the remote experts' the lane adds to it); layer l + 1 waits for it after its
    // attention and adds it before its FFN (hc_late_add), the last layer before the head.  Only the head's add is exact
    // (up to the order of the float sums): layer l + 1's attention - its output, the KV it writes, and the hc
    // coefficients computed from h - and an Engram layer's gate see h without layer l's part.  That is the deferral's
    // error, not a bug.  (DS4_DUMP_H / DS4_DUMP_L rows then lack the part still out.)
    // late_slot: the mailbox slot whose part is still out (-1: none), late_seq its seq; never across a window
    bool ep_defer = false;
    int late_slot = -1;
    unsigned late_seq = 0;
    // DS4_EP_DEFER_FROM=L: only layers L and later deferred (nl - 1: the last alone, whose add before the head is exact).
    // DS4_EP_DEFER_VERIFY=1: a wiring check, not a speed mode - every layer waits and adds its part in place as without
    // the deferral, and beside it the deferred streams are built from the same attention outputs and hc coefficients
    // (Dev::v_*: hc_post without the part, the next attention's hc_post, late_flush's add) and compared with the exact
    // streams after that attention (or before the head): up to the float sums' order they are the same, so a larger
    // difference is a wiring bug.  An Engram layer's in-place add to the exact streams (its gates from them) goes onto
    // the deferred ones too (add_diff).  v_err[il]: the largest relative difference for layer il's part (device; Stats
    // gets a copy after each window)
    int defer_from = 0, late_il = -1;
    bool ep_verify = false;
    bool v_wrong = false;   // DS4_EP_DEFER_VERIFY=2: the check's own check - the deferred side given a wrong comb (layer
                            // l's FFN one) on purpose, which the comparison must show; decode stays exact
    float* v_err = nullptr;
    int async_max_t = std::getenv("DS4_ASYNC_T") ? std::atoi(std::getenv("DS4_ASYNC_T")) : 8;   // windows up to this
    bool win_prompt = false;   // (feed_windows: a prompt's windows - their cache ticks are the decode's, serve_T 0)
    rx::PromptSettle psettle;  // (a window prompt's settle: once a prompt, ds4_rx_fill.hpp)
    // DS4_SMALL_PROMPT=N (0 off), DS4_SMALL_WIN=T: prompts of 2-N tokens in windows of T rows (small_setup,
    // ds4_small_prompt.hpp); small_ms: their ms a token so far (a running mean, for --verbose), small_k how many
    int small_n = 0, small_t = 1;
    double small_ms = 0;
    int64_t small_k = 0;
    // feed_small's windows are on (set only there): they are a prompt's, though each is an EP window of one row, so the
    // statistics that mean the decode's windows (DS4_PROF's, the lookahead's, ep_late, the defer checks' count, the
    // hybrid's decode-request tables) leave them out.  Not routing, the caches, the settle or the expert server's lane
    bool small_feed = false;
    // lookahead: while layer L runs, the device guesses the next layer's picks (its gate on L's FFN input, the same
    // choice) and publishes them with L's routing; the CPU lane asks for the guessed experts it will read from the files
    // at once (a thread of its own, os::prefetch), so the SSD reads them while L computes instead of on the next
    // layer's critical path.  Only the page cache sees it - the outputs are the same.  DS4_AHEAD=0: off, 2: also the
    // layer after (from L's input too); DS4_AHEAD_EXTRA: guesses past the next layer's top k (< 0: fewer - the guess's
    // picks by rank on Sun: 97 91 81 70 57 41 %, past the top 6 29 20 %: reads wasted on an SSD-bound PC).
    int ahead_mode = std::getenv("DS4_AHEAD") ? std::atoi(std::getenv("DS4_AHEAD")) : 1;
    int ahead_extra = std::getenv("DS4_AHEAD_EXTRA") ? std::atoi(std::getenv("DS4_AHEAD_EXTRA")) : 0;
    // (DS4_AHEAD_ORDER=0: the CPU lane computes a layer's experts in expert order, not the ones asked for early first)
    bool ahead_order = std::getenv("DS4_AHEAD_ORDER") == nullptr || std::atoi(std::getenv("DS4_AHEAD_ORDER")) != 0;
    std::vector<uint8_t> ahead_fl;     // [layer][1024]: 1 the top-k guess, 2 any guess, a layer before; 4 two layers before;
                                       // 8 asked for (prefetch)
    std::vector<uint8_t> ahead_rank;   // [layer][1024]: 1 + the expert's rank in the guess a layer before (0: not guessed)
    std::vector<unsigned> ahead_for;   // [layer]: the request (seq) its flags are for
    std::thread pf_th;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    std::vector<std::pair<const void*, size_t>> pf_q;
    // the expert cache: swaps prepared on threads of their own (DS4_CACHE_THREADS, 1 by default - 4 measured no faster
    // on one V100 - more for a PC to try: each its staging buffers and
    // copy streams), switched between windows
    struct Swap {
        int layer = 0, in = -1, out = -1;
        int ram = -1;   // (copy_mode, `in` held in RAM) the spare slot `out` is copied into from VRAM
    };
    static constexpr int kMaxSwaps = 8;
    // the swap rule: a CPU-lane expert in when its heat passes the coldest VRAM one's x margin + bonus; heat x decay
    // every 256 tokens; at most `inflight` swaps under way (DS4_CACHE_MARGIN / _BONUS / _DECAY / _INFLIGHT)
    static float env_f(const char* k, float d) { return std::getenv(k) ? (float) std::atof(std::getenv(k)) : d; }
    const float c_margin = env_f("DS4_CACHE_MARGIN", 1.25f), c_bonus = env_f("DS4_CACHE_BONUS", 1.0f),
                c_decay = env_f("DS4_CACHE_DECAY", 0.9f);
    const int c_inflight = std::max(1, (int) env_f("DS4_CACHE_INFLIGHT", (float) kMaxSwaps));
    // the RAM tier (a GGUF's in-place experts, as GLM's: the hottest in VRAM, the next hottest here, the coldest in the
    // files): RAM slots of one expert blob each, filled at load with the hottest experts VRAM does not hold, so the
    // CPU lane reads them from RAM; between windows (cache_tick) a hotter expert from the files takes the place of the
    // layer's coldest held one - read into a free slot by ram_th, switched while nothing reads them.  Opt-in
    // (STRATA_DS4_RAM_TIER=1): on a V100 with 64 GB of RAM it was slower than the page cache alone (1.4 against 2.2-2.4
    // tokens/s sampled) - the cache follows a conversation's experts at once, the tier's moves lag behind.
    struct RamSwap {
        int layer = 0, in = -1, out = -1, slot = -1;
    };
    std::vector<uint8_t*> ram_slots, ram_chunks;
    std::vector<size_t> ram_chunk_bytes;
    bool ram_pinned = false;   // the slots are pinned memory (the prompt path DMAs them from where they lie)
    size_t flow_part[3] = {0, 0, 0};   // the flow's staging room for each part: its largest over the trunk's layers
    std::vector<int> ram_free;
    std::deque<RamSwap> ram_queue;
    std::vector<RamSwap> ram_ready;
    std::condition_variable ram_cv;
    std::thread ram_th;
    int ram_inflight = 0, ram_held = 0;
    bool ram_tier = std::getenv("STRATA_DS4_RAM_TIER") != nullptr && std::atoi(std::getenv("STRATA_DS4_RAM_TIER")) != 0;
    // STRATA_DS4_DISK_SKIP_W (experimental opt-in, CHANGES THE ANSWERS; 0 = off): on the small windows (decode, a
    // draft's check - never the prompt path) the CPU lane leaves out a routed expert whose share of its token's route
    // weights is below this and whose pages are not in RAM now (the page cache or the RAM tier; os::in_ram - Linux every
    // page, Windows sampled pages of the working set), so a token waits on the SSD only for the experts that weigh; the
    // weight it carried is lost (no renormalisation).  On layers from STRATA_DS4_DISK_SKIP_FROM (0) on.  An expert left
    // out still counts as used (the expert cache, the profile).
    const float disk_skip_w = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_W");
        return v != nullptr ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    const int disk_skip_from = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_FROM");
        return v != nullptr ? std::max(0, std::atoi(v)) : 0;
    }();
    // the RAM store (--ssd-cache): the CPU lane's experts in slots a policy owns, read unbuffered (null: off)
    std::unique_ptr<ExpertStore> store;
    uint8_t* zc_stage_buf = nullptr;   // device: k staging slots for the store's experts of a decode token
    size_t zc_stride = 0;
    bool zc_copy = true;               // DS4_STORE_STAGE=0: the expert kernels read the arena over PCIe themselves
    bool in_ram(const Layer& l, int e) const {   // (DISK_SKIP_W) expert e's pages in RAM now
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) return true;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r)
                if (!os::in_ram(mx_mat(l, e, r), mx_part_bytes(l, r))) return false;
            return true;
        }
        return !file_experts || os::in_ram(file_part(l, e), l.blob);
    }
    // a GGUF's CPU-lane experts copied into RAM (copy_experts, after VRAM took its share: --mmap-experts off - auto reads
    // them in place; DS4_COPY_GB=<n>: the hottest n GB, the rest in place - static, no moves between
    // them): RAM slots (l.ram, as the tier's) of NH parts, part X on NUMA node X with rows [X n / NH, (X + 1) n / NH) of
    // each matrix at cp_lay[layer][r] - whole rows, so the split never cuts a GGUF block (MXFP4's 32, the IQ / K types'
    // 256 values: blocks run along a row); one spare slot a layer for the cache's swaps (the expert VRAM gives back takes
    // the RAM place of the one it takes in)
    // Windows' commit on an integrated GPU: its allocations are charged to the commit (RAM + page file) like any private
    // memory - measured on a Radeon 8065S with 160 GiB of its own: the system's free commit fell by each one, whatever
    // its size (one above ~3 GiB also counts as the process's private bytes, a smaller one does not, the commit the
    // same), so a page file at least the GPU's share is what lets it hold experts at all.  The experts it holds
    // (place_experts) and the copies stay within what Windows can still commit, less max(8 GB, 10% of the RAM) for the
    // other programs - commit_kept: the experts that kept off the GPU, pagefile_hint: the page file (GB) that would hold
    // them (the log's WARNING, a NOTE and INFO for the server)
    double commit_kept = 0, pagefile_hint = 0;
    double commit_margin() const {
        const os::Commit c = os::commit();
        return std::max(8.0 * (double) (1ull << 30), c.ram == ~uint64_t{0} ? 0.0 : 0.1 * (double) c.ram);
    }
    // what may still be committed beside the margin (~inf: nothing limits it; DS4_COMMIT_CAP_GB: at most that, tests)
    double commit_room() const {
        const os::Commit c = os::commit();
        double avail = c.avail == ~uint64_t{0} ? 1e300 : (double) c.avail;
        if (const char* v = std::getenv("DS4_COMMIT_CAP_GB")) avail = std::min(avail, std::atof(v) * 1e9);
        return avail >= 1e300 ? 1e300 : std::max(0.0, avail - commit_margin());
    }
    // a device whose allocations Windows charges to the commit: an integrated GPU (DS4_COMMIT_CAP_GB: any, tests)
    bool commit_charged(const Dev& d) const {
        if (std::getenv("DS4_COMMIT_CAP_GB") != nullptr) return true;
#if defined(_WIN32)
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, d.id) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        return v != 0;
#else
        (void) d;
        return false;
#endif
    }
    bool copy_mode = false;
    std::vector<std::array<uint8_t*, 2>> ram_part;   // slot -> its parts
    std::vector<std::array<size_t, 4>> cp_lay;       // layer -> part offsets of gate, up, down rows; a part's bytes
    std::vector<int> copy_spare;                      // layer -> its spare slot (-1: none)
    std::mutex sw_mu;
    std::condition_variable sw_cv;
    std::deque<Swap> sw_queue;
    std::vector<Swap> sw_ready;
    struct SwapBufs {
        uint8_t *a = nullptr, *b = nullptr, *h = nullptr;
        std::vector<cudaStream_t> st;   // a copy stream a device
    };
    std::vector<SwapBufs> sw_bufs;
    std::vector<std::thread> cache_ths;
    int sw_inflight = 0;
    int64_t decay_tokens = 0;
    bool check_swaps = std::getenv("DS4_CHECK_SWAPS") != nullptr;
    std::vector<float> host_logits;
    std::vector<void*> host_allocs;
    std::FILE* dump_h = std::getenv("DS4_DUMP_H") ? std::fopen(std::getenv("DS4_DUMP_H"), "wb") : nullptr;
    // DS4_DUMP_L: (threads) every layer's residual streams as records [layer, p0, T, T x hc x dim floats], window or
    // chunk path (tests: where the two part)
    std::FILE* dump_l = std::getenv("DS4_DUMP_L") ? std::fopen(std::getenv("DS4_DUMP_L"), "wb") : nullptr;
    // DS4_DUMP_R: the chunk path's routing, a record a layer [layer, p0, T, k, n_exp, n_vs, n_slots, kPicks] (int64),
    // then the picks' experts [T kPicks] (the first k of each token's), each expert's virtual slot [n_exp], the
    // slots' row bounds [n_vs + 1], the sorted rows' tokens [T k] (int32), the gate's logits [T n_exp] and its bias
    // [n_exp] (F32)
    std::FILE* dump_r = std::getenv("DS4_DUMP_R") ? std::fopen(std::getenv("DS4_DUMP_R"), "wb") : nullptr;
    // DS4_DUMP_Y=<layer>:<file> (layer -1: every layer): its expert sets on the chunk path, a record a piece [layer,
    // row0, rows, n_exp, wbase kind (0 VRAM, 1 ring), max_rows] (int64), its bounds [n_exp + 1], its h rows [rows ff]
    // (F16) and y rows [rows dim] (F32)
    int dump_y_layer = std::getenv("DS4_DUMP_Y") ? std::atoi(std::getenv("DS4_DUMP_Y")) : -1;
    std::FILE* dump_y = [] {
        const char* v = std::getenv("DS4_DUMP_Y");
        const char* f = v != nullptr ? std::strchr(v, ':') : nullptr;
        return f != nullptr ? std::fopen(f + 1, "wb") : nullptr;
    }();
    std::mutex dump_mu;
    // DS4_PROF: the GPU's time a decoded token by stage - events in stream order at a layer's start, after its
    // attention, after the routing and the shared expert, after the experts' gate / up, after the wait for the CPU
    // lane, after the down pass and the post - read after the token's sync, each the time since the mark before it.
    // (U9 tail moves the device experts' down into the gate / up stage: compare such runs by their totals, not by stage)
    bool prof_on = std::getenv("DS4_PROF") != nullptr;
    // DS4_PRED_DUMP=file (predictor data, the chunked prompt path): per layer and chunk u8 layer, u16 T, T x 384 fp16 router
    // logits, T x 256 fp16 projection of the router input.  DS4_XIN_DUMP=file (expert similarity inputs): u8 layer, u16 T,
    // T x D fp32 router inputs, the first DS4_XIN_N (1024) tokens of each layer.
    std::FILE* pred_f = std::getenv("DS4_PRED_DUMP") != nullptr ? std::fopen(std::getenv("DS4_PRED_DUMP"), "wb") : nullptr;
    std::FILE* xin_f = std::getenv("DS4_XIN_DUMP") != nullptr ? std::fopen(std::getenv("DS4_XIN_DUMP"), "wb") : nullptr;
    int xin_n = std::getenv("DS4_XIN_N") ? std::atoi(std::getenv("DS4_XIN_N")) : 1024;
    std::array<int, 64> xin_got{};
    std::FILE* trace_f = std::getenv("DS4_TRACE") != nullptr ? std::fopen(std::getenv("DS4_TRACE"), "wb") : nullptr;
    struct ProfMark {
        const Dev* d;
        cudaEvent_t ev;
        int stage;
    };
    std::mutex prof_mu;
    std::vector<ProfMark> prof_marks;
    std::map<int, std::vector<cudaEvent_t>> prof_pool;   // by device: an event belongs to the device it was made on
    bool prof_window = false;   // (decode windows only: T == 1, and not a small prompt's)
    // ... and a run of chunks' (prof_chunk): a layer's start, after its attention, after the routing and the shared
    // expert, after its experts (the streamed ones' waits for their reads in it); printed after the run
    bool prof_chunk = false;
    // DS4_PROF stages of the chunked prompt (finer ones than the decode's 0..5): 0 between layers, 1 attention rest (wq_b, wo,
    // hc_post), 2 router-side tail (shared expert), 3 experts (+ waits), 4 Engram, 5 attn mHC (mix, split, norm), 6 attention
    // projections (wq_a, wkv, compressor), 7 compressor + indexer, 8 q_b + rope, 9 sparse attention, 10 ffn mHC + router + sort
    double cprof_ms[16] = {};
    // DS4_PROF: the experts' kernels alone, by type (events around the gate/up and down calls of run_set)
    struct EkMark {
        cudaEvent_t e0, e1, e2;
        int tg, td, nrows, nexp, maxrows, dev, ff;
    };
    std::vector<EkMark> ek_marks;
    // DS4_PROF (a run of chunks): gemm()'s pieces - its FP16 reconstruction of W (e0 - e1) apart from cuBLAS's GEMM
    // (e1 - e2), by the matrix (k x n); nc the piece's columns, first: a call's first piece.  DS4_DQ_ONCE: a piece of
    // dq_whole's reconstruction is a mark of its own (no GEMM: rows 0, first 0 - not a call), the slabs' pieces GEMMs alone
    struct GmMark {
        cudaEvent_t e0, e1, e2;
        int k, n, nc, rows, dev, first;
    };
    std::vector<GmMark> gm_marks;
    std::map<int, std::vector<cudaEvent_t>> ek_pool;   // by device
    cudaEvent_t ek_ev() {
        int cur = 0;
        cudaGetDevice(&cur);
        std::vector<cudaEvent_t>& pl = ek_pool[cur];
        if (!pl.empty()) {
            cudaEvent_t e = pl.back();
            pl.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        cudaEventCreate(&e);
        return e;
    }
    void prof_mark(const Dev& d, int stage) {
        if (!prof_window && !prof_chunk) return;
        std::lock_guard<std::mutex> lk(prof_mu);
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(d.id);   // (events of another device are an "invalid resource handle")
        cudaEvent_t e = nullptr;
        std::vector<cudaEvent_t>& pool = prof_pool[d.id];
        if (pool.empty()) cudaEventCreate(&e);
        else {
            e = pool.back();
            pool.pop_back();
        }
        cudaEventRecord(e, d.st);
        cudaSetDevice(cur);
        prof_marks.push_back({&d, e, stage});
    }
    void prof_collect() {
        if (!prof_window && !prof_chunk) return;
        std::lock_guard<std::mutex> lk(prof_mu);
        std::vector<std::pair<const Dev*, cudaEvent_t>> last;
        for (const ProfMark& m : prof_marks) {
            auto it = std::find_if(last.begin(), last.end(), [&](const auto& x) { return x.first == m.d; });
            if (it != last.end()) {
                float ms = 0.0f;
                cudaSetDevice(m.d->id);
                if (cudaEventElapsedTime(&ms, it->second, m.ev) == cudaSuccess)
                    (prof_chunk ? cprof_ms : st.prof_ms)[m.stage] += ms;
                it->second = m.ev;
            } else {
                last.push_back({m.d, m.ev});
            }
        }
        for (const ProfMark& m : prof_marks) prof_pool[m.d->id].push_back(m.ev);
        prof_marks.clear();
        if (!prof_chunk) ++st.prof_tokens;
    }
    void dump_layer(Dev& d, const float* h, int il, int64_t p0, int T) {
        std::vector<float> v((size_t) T * g.hc * g.dim);
        cudaSetDevice(d.id);
        cudaMemcpyAsync(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        std::lock_guard<std::mutex> lk(dump_mu);
        const int64_t hdr[3] = {il, p0, T};
        std::fwrite(hdr, 8, 3, dump_l);
        std::fwrite(v.data(), 4, v.size(), dump_l);
        std::fflush(dump_l);
    }

    ~Impl() {
        eg_io.stop();   // (its threads write into h_eg)
        for (auto& w : workers) w->stop();
        for (size_t i = 0; i < chunks.size(); ++i)
            if (chunks[i] && chunks[i]->blas) {
                cudaSetDevice(devs[i].id);
                cublasDestroy(chunks[i]->blas);
            }
        quit.store(true);
        if (service.joinable()) service.join();
        if (lane_tr != nullptr) std::fclose(lane_tr);   // (the lane's calls are over: its service thread is)
        lane_tr = nullptr;
        {
            std::lock_guard<std::mutex> lk(pf_mu);
        }
        pf_cv.notify_all();
        if (pf_th.joinable()) pf_th.join();
        {
            std::lock_guard<std::mutex> lk(sw_mu);
        }
        sw_cv.notify_all();
        ram_cv.notify_all();
        for (std::thread& t : cache_ths)
            if (t.joinable()) t.join();
        if (ram_th.joinable()) ram_th.join();
        for (auto& cp : chunks) {   // (DS4_LM_PRE's layer cache and chunks' states)
            if (cp == nullptr) continue;
            if (cp->lc_kept != nullptr) os::anon_free(cp->lc_kept, cp->lc_cap);
            if (cp->st_kept != nullptr) {
                if (cp->st_kept_pinned) cudaFreeHost(cp->st_kept);
                else os::anon_free(cp->st_kept, cp->st_cap);
            }
        }
        for (size_t i = 0; i < ram_chunks.size(); ++i) os::anon_free(ram_chunks[i], ram_chunk_bytes[i]);
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            for (void* p : d.allocs) cudaFree(p);
            if (d.st) cudaStreamDestroy(d.st);
            if (d.ev) cudaEventDestroy(d.ev);
            if (d.st2) cudaStreamDestroy(d.st2);
            if (d.ev_fork) cudaEventDestroy(d.ev_fork);
            if (d.ev_join) cudaEventDestroy(d.ev_join);
        }
        for (void* p : host_allocs) cudaFreeHost(p);
        for (Layer& l : L)
            for (uint8_t* b : l.host)
                if (b != nullptr) {
                    (void) b;   // host blobs live in one mapping per layer, unmapped below
                }
        for (auto& m : host_maps) {
            if (chunk > 0) cudaHostUnregister(m.first);
            os::anon_free(m.first, m.second);
        }
        for (auto& m : link_maps) os::anon_free(m.first, m.second);
    }
    std::vector<std::pair<void*, size_t>> host_maps;

    template <class T>
    bool dalloc(Dev& d, T*& p, size_t n, std::string& err, size_t* acct = nullptr) {
        void* v = nullptr;
        cudaSetDevice(d.id);
        const size_t b = align256(n * sizeof(T));
        if (cudaMalloc(&v, b) != cudaSuccess) {
            err = "cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        // zeroed before anything lands in it: the loads copy in on the default stream, which the non-blocking d.st
        // does not order against - a zeroing still queued there could wipe the weights just copied
        if (cudaMemsetAsync(v, 0, b, d.st) != cudaSuccess || cudaStreamSynchronize(d.st) != cudaSuccess) {
            cudaFree(v);
            err = "zeroing " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        d.allocs.push_back(v);
        if (acct) *acct += b;
        else d.scratch += b;
        p = (T*) v;
        return true;
    }
    static Layer& member(Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    static const Layer& member(const Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    // bytes of a VRAM slot of l's experts (a member's part of each, in a group)
    static size_t unit(const Layer& l) { return l.nm > 1 ? l.hblob : l.blob; }
    // the members of a group of nm that take heads: the most the heads and the wo_a groups split among evenly (64
    // heads and 8 groups: 9 -> 8, 7 or 6 -> 4, 5 -> 4, 3 -> 2); the first members the rest, only their FFN share
    int head_members(int nm) const {
        int na = nm;
        while (na > 1 && (g.heads % na != 0 || g.o_groups % na != 0)) --na;
        return na;
    }
    // host-mapped memory (the same address on the host and every device), zeroed
    template <class T>
    bool hmap(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "host-mapped memory (" + std::to_string((n * sizeof(T)) >> 20) + " MiB)";
            return false;
        }
        std::memset(v, 0, n * sizeof(T) + 64);
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocPortable) != cudaSuccess) {
            err = "cudaHostAlloc " + std::to_string((n * sizeof(T)) >> 20) + " MiB failed";
            return false;
        }
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }

    // ---- weights
    bool upload(Dev& d, const Tensor& t, void* dst, std::string& err) {
        std::vector<uint8_t> tmp(t.bytes);
        if (!pk.read(t, tmp.data())) {
            err = "read failed";
            return false;
        }
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dst, tmp.data(), t.bytes, cudaMemcpyHostToDevice));
        return true;
    }
    // a dense matrix in a ggml block type (x3::Mat fmt kFmtGG, row = k / 256 blocks): whole, or part of its rows / inputs
    bool mat_gg(Dev& d, const std::string& pre, const Tensor& t, int ggt, x3::Mat& m, std::string& err, size_t* acct, int part,
                int np, bool cols) {
        if (t.shape.size() != 2 || t.shape[1] % 256 != 0) {
            err = pre + ": a " + t.dtype + " tensor that is not a matrix of whole blocks";
            return false;
        }
        int n = (int) t.shape[0], k = (int) t.shape[1];
        const size_t rb = gg::type_row_bytes(ggt, k), bb = rb / ((size_t) k / 256);
        std::vector<uint8_t> w((size_t) n * rb);
        if (t.bytes != w.size() || !pk.read(t, w.data())) {
            err = pre + ": read failed";
            return false;
        }
        if (np > 1) {
            const int whole = cols ? n : k, wd = whole / np;
            if (whole % np != 0 || wd % (cols ? 1 : 256) != 0) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " + std::to_string(np) + " parts";
                return false;
            }
            std::vector<uint8_t> p;
            if (cols) {
                p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
                n = wd;
            } else {
                const size_t pb = (size_t) (wd / 256) * bb, off = (size_t) (part * wd / 256) * bb;
                p.resize((size_t) n * pb);
                for (size_t r = 0; r < (size_t) n; ++r) std::memcpy(p.data() + r * pb, w.data() + r * rb + off, pb);
                k = wd;
            }
            w.swap(p);
        }
        uint8_t* dp = nullptr;
        if (!dalloc(d, dp, w.size(), err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.fmt = gg::kFmtGG;
        m.K = ggt;
        m.raw = dp;
        m.k = k;
        m.n = n;
        m.ld = (int) gg::type_row_bytes(ggt, k);
        return true;
    }
    // a GGUF's BF16 or Q8_0 matrix ([out][in]) onto the device, as mat() takes an EXL3 one: whole, or part `part` of its
    // columns (outputs: a range of its rows) or of its rows (inputs: a range of each row, whole Q8_0 blocks) - a
    // standalone [n][k] matrix
    bool mat_bf16(Dev& d, const std::string& pre, const Tensor& t, x3::Mat& m, std::string& err, size_t* acct, int part,
                  int np, bool cols) {
        const bool q8 = t.dtype == "Q8_0";
        // a ggml K-quant dense matrix (Q4_K / Q5_K / Q6_K: 256 values a block; the GEMV path of ds4_iq)
        const int ggt = (t.dtype == "Q4_K" || t.dtype == "Q5_K" || t.dtype == "Q6_K") ? gg_type_of(t.dtype) : -1;
        if (ggt > 0) return mat_gg(d, pre, t, ggt, m, err, acct, part, np, cols);
        if (t.shape.size() != 2 || (q8 && t.shape[1] % 32 != 0)) {
            err = pre + ": a " + t.dtype + " tensor that is not a matrix";
            return false;
        }
        int n = (int) t.shape[0], k = (int) t.shape[1];
        // a row's bytes, and the bytes of v of its values (BF16: 2 each; Q8_0: 34 every 32)
        auto bytes_of = [&](size_t v) { return q8 ? v / 32 * gg::kQ8Bytes : v * 2; };
        std::vector<uint8_t> w((size_t) n * bytes_of((size_t) k));
        if (t.bytes != w.size() || !pk.read(t, w.data())) {
            err = pre + ": read failed";
            return false;
        }
        if (np > 1) {
            const int whole = cols ? n : k, wd = whole / np;
            if (whole % np != 0 || wd % (q8 ? 32 : 8) != 0) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " +
                      std::to_string(np) + " parts";
                return false;
            }
            const size_t rb = bytes_of((size_t) k);
            std::vector<uint8_t> p;
            if (cols) {
                p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
                n = wd;
            } else {
                const size_t pb = bytes_of((size_t) wd);
                p.resize((size_t) n * pb);
                for (size_t r = 0; r < (size_t) n; ++r)
                    std::memcpy(p.data() + r * pb, w.data() + r * rb + bytes_of((size_t) part * wd), pb);
                k = wd;
            }
            w.swap(p);
        }
        uint8_t* dp = nullptr;
        if (!dalloc(d, dp, w.size(), err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.fmt = q8 ? gg::kFmtQ8_0 : gg::kFmtBF16;
        m.raw = dp;
        m.k = k;
        m.n = n;
        m.ld = q8 ? (int) bytes_of((size_t) k) : k;
        return true;
    }
    // a Q8_0 tensor's values as F32 (host)
    static std::vector<float> q8_values(const uint8_t* q, size_t n) {
        std::vector<float> v(n);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = q + b * gg::kQ8Bytes;
            const float d = f16_to_f32((uint16_t) (p[0] | p[1] << 8));
            for (int j = 0; j < 32; ++j) v[b * 32 + j] = d * (float) (int8_t) p[2 + j];
        }
        return v;
    }
    // an EXL3 matrix onto the device: whole, or (np > 1) part `part` of its columns (cols) or of its rows - a standalone
    // matrix of its own (the rotations' 128-blocks stay whole)
    // (rot: split128's - an EXL3 matrix's uneven 128-block shares going round the members; a GGUF's BF16 / Q8_0 matrix
    // splits evenly, its groups only as wide as that goes)
    bool mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
             bool cols = true, int rot = 0) {
        if (const Tensor* tw = pk.get(pre + ".weight"); tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0" || tw->dtype == "Q4_K" || tw->dtype == "Q5_K" || tw->dtype == "Q6_K"))
            return mat_bf16(d, pre, *tw, m, err, acct, part, np, cols);
        const Tensor* tr = pk.get(pre + ".trellis");
        const Tensor* su = pk.get(pre + ".suh");
        const Tensor* sv = pk.get(pre + ".svh");
        if (tr == nullptr || su == nullptr || sv == nullptr || tr->shape.size() != 3) {
            err = pre + ": not an EXL3 matrix in the pack";
            return false;
        }
        if (pk.get(pre + ".mul1") == nullptr) {
            err = pre + ": only the mul1 codebook is covered";
            return false;
        }
        m.k = (int) tr->shape[0] * 16;
        m.n = (int) tr->shape[1] * 16;
        m.K = (int) tr->shape[2] / 16;
        m.cb = 2;
        m.ld = 0;
        if (np > 1) {
            const int whole = cols ? m.n : m.k;
            int at = 0, w = 0;
            if (whole % 128 != 0 || whole / 128 < np) {
                err = pre + ": " + std::to_string(whole) + (cols ? " columns" : " rows") + " do not split in " +
                      std::to_string(np) + " parts of 128-blocks";
                return false;
            }
            split128(whole, np, part, at, w, rot);
            std::vector<uint8_t> t0(tr->bytes), u0(su->bytes), v0(sv->bytes);
            if (!pk.read(*tr, t0.data()) || !pk.read(*su, u0.data()) || !pk.read(*sv, v0.data())) {
                err = pre + ": read failed";
                return false;
            }
            const size_t tile = tr->bytes / ((size_t) (m.k / 16) * (m.n / 16));
            std::vector<uint8_t> t1, u1, v1;
            if (cols) {   // the tiles [c0 / 16, (c0 + w) / 16) of each 16-row slice, svh [c0, c0 + w)
                const size_t c0 = (size_t) at, nt = (size_t) m.n / 16, nth = (size_t) w / 16;
                t1.resize((size_t) (m.k / 16) * nth * tile);
                for (size_t sl = 0; sl < (size_t) m.k / 16; ++sl)
                    std::memcpy(t1.data() + sl * nth * tile, t0.data() + (sl * nt + c0 / 16) * tile, nth * tile);
                u1 = u0;
                v1.assign(v0.begin() + (long) (c0 * 2), v0.begin() + (long) ((c0 + w) * 2));
                m.n = w;
            } else {      // the 16-row slices [r0 / 16, (r0 + w) / 16), suh [r0, r0 + w)
                const size_t r0 = (size_t) at, row = (size_t) (m.n / 16) * tile;
                t1.assign(t0.begin() + (long) (r0 / 16 * row), t0.begin() + (long) ((r0 + w) / 16 * row));
                u1.assign(u0.begin() + (long) (r0 * 2), u0.begin() + (long) ((r0 + w) * 2));
                v1 = v0;
                m.k = w;
            }
            uint32_t* dt = nullptr;
            uint16_t *ds = nullptr, *dv = nullptr;
            if (!dalloc(d, dt, t1.size() / 4, err, acct) || !dalloc(d, ds, u1.size() / 2, err, acct) ||
                !dalloc(d, dv, v1.size() / 2, err, acct))
                return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(dt, t1.data(), t1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ds, u1.data(), u1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv, v1.data(), v1.size(), cudaMemcpyHostToDevice));
            m.trellis = dt;
            m.suh = ds;
            m.svh = dv;
            return x3::supported(m) || (err = pre + ": a part not covered by the EXL3 kernels", false);
        }
        uint32_t* dt = nullptr;
        uint16_t *ds = nullptr, *dv = nullptr;
        if (!dalloc(d, dt, tr->bytes / 4, err, acct) || !dalloc(d, ds, su->bytes / 2, err, acct) ||
            !dalloc(d, dv, sv->bytes / 2, err, acct))
            return false;
        if (!upload(d, *tr, dt, err) || !upload(d, *su, ds, err) || !upload(d, *sv, dv, err)) return false;
        m.trellis = dt;
        m.suh = ds;
        m.svh = dv;
        if (!x3::supported(m)) {
            err = pre + ": k " + std::to_string(m.k) + " n " + std::to_string(m.n) + " K " + std::to_string(m.K) +
                  " not covered by the EXL3 kernels";
            return false;
        }
        return true;
    }
    // a float tensor (F32 / F16 / BF16) as F32 on the device
    bool f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want = 0) {
        const Tensor* t = pk.get(name);
        if (t == nullptr) {
            err = name + ": not in the pack";
            return false;
        }
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        size_t n = 0;
        std::vector<float> v;
        if (t->dtype == "F32") {
            n = t->bytes / 4;
            v.resize(n);
            std::memcpy(v.data(), raw.data(), t->bytes);
        } else if (t->dtype == "F16" || t->dtype == "BF16") {
            n = t->bytes / 2;
            v.resize(n);
            const uint16_t* s = (const uint16_t*) raw.data();
            for (size_t i = 0; i < n; ++i) v[i] = t->dtype == "F16" ? f16_to_f32(s[i]) : bf16_to_f32(s[i]);
        } else if (t->dtype == "Q8_0") {   // (a GGUF's router may come so)
            n = t->bytes / gg::kQ8Bytes * 32;
            v = q8_values(raw.data(), n);
        } else {
            err = name + ": dtype " + t->dtype + " is not a float";
            return false;
        }
        if (want != 0 && n != want) {
            err = name + ": " + std::to_string(n) + " values, expected " + std::to_string(want);
            return false;
        }
        if (!dalloc(d, p, n, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
        return true;
    }
    // the router's weights: F32 on the device (f32), or with U9 route a BF16 file's rows as they are (half the bytes
    // read a token; gemv_bf16 gives gemv_f32's bits)
    bool gate_load(Dev& d, const std::string& name, Layer& l, std::string& err, size_t* acct) {
        const size_t want = (size_t) l.n_exp * g.dim;
        const Tensor* t = pk.get(name);
        if (t == nullptr || t->dtype != "BF16" || t->bytes != want * 2 || g.dim % 128 != 0 || !u9::on(u9::kRoute))
            return f32(d, name, l.gate_w, err, acct, want);
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        // (allocated at the F32 copy's size: the same cudaMalloc sizes as the F32 path, so the same VRAM is left for the
        // experts - the driver's granules make a smaller buffer free more than its bytes; the second half stays unused)
        if (!dalloc(d, l.gate_bf, want * 2, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(l.gate_bf, raw.data(), t->bytes, cudaMemcpyHostToDevice));
        return true;
    }
    // the router's logits of layer l's gate on x (T rows, D apart) into y (rows 1024 apart)
    void gate_gemv(const Layer& l, const float* x, int T, float* y, cudaStream_t s) {
        if (l.gate_bf == nullptr) {
            gemv_f32(l.gate_w, x, l.n_exp, g.dim, T, y, 1024, s);
            return;
        }
        GemvBf16 gb = {};
        gb.w[0] = l.gate_bf;
        gb.y[0] = y;
        gb.n[0] = l.n_exp;
        gemv_bf16(gb, 1, x, g.dim, T, 1024, s);
    }
    bool f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct) {
        const Tensor* t = pk.get(name);
        if (t != nullptr && (t->dtype == "F32" || t->dtype == "BF16" || t->dtype == "Q8_0")) {   // (a GGUF's)
            const size_t n = t->dtype == "Q8_0" ? t->bytes / gg::kQ8Bytes * 32 : t->bytes / (t->dtype == "F32" ? 4 : 2);
            std::vector<uint8_t> raw(t->bytes);
            if (!pk.read(*t, raw.data())) {
                err = name + ": read failed";
                return false;
            }
            const std::vector<float> q = t->dtype == "Q8_0" ? q8_values(raw.data(), n) : std::vector<float>();
            std::vector<uint16_t> h(n);
            for (size_t i = 0; i < n; ++i) {
                float v;
                if (t->dtype == "F32") std::memcpy(&v, raw.data() + i * 4, 4);
                else if (t->dtype == "Q8_0") v = q[i];
                else v = bf16_to_f32(((const uint16_t*) raw.data())[i]);
                h[i] = strata::kernels::f16_from_f32(v);
            }
            if (!dalloc(d, p, n, err, acct)) return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice));
            return true;
        }
        if (t == nullptr || t->dtype != "F16") {
            err = name + ": not an F16 tensor in the pack";
            return false;
        }
        if (!dalloc(d, p, t->bytes / 2, err, acct)) return false;
        return upload(d, *t, p, err);
    }

    size_t mat_bytes(const std::string& pre) const {
        if (const Tensor* t = pk.get(pre + ".weight")) return align256(t->bytes);
        size_t b = 0;
        for (const char* s : {".trellis", ".suh", ".svh"})
            if (const Tensor* t = pk.get(pre + s)) b += align256(t->bytes);
        return b;
    }
    size_t tensor_bytes(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return t ? align256(t->bytes * (t->dtype == "F32" ? 1 : 2)) : 0;
    }

    // ---- expert regions and blob layout
    // a GGUF's MXFP4 experts: the blob is gate | up | down ([out][in / 32] blocks each, as the file holds them), every
    // expert's matrix r at the same stride from expert 0's in one file (the GGUF's [n_exp][out][in] tensor)
    // (the other block types: lay.fmt stays kFmtMXFP4 - read in place as these - and lay.K[r] is role r's ggml type)
    static int gg_type_of(const std::string& dtype) {
        for (int t : {gg::kTypeMXFP4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 145, 146, 224, 225, 226, 227})
            if (dtype == gg::type_name(t)) return t;
        return -1;
    }
    bool expert_layout_mx(Layer& l, std::string& err) {
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
            const Tensor* t0 = pk.get(P + "0." + parts[r] + ".weight");
            const Tensor* t1 = l.n_exp > 1 ? pk.get(P + "1." + parts[r] + ".weight") : t0;
            const int ty = t0 != nullptr ? gg_type_of(t0->dtype) : -1;
            const size_t bytes = ty > 0 ? (size_t) n * gg::type_row_bytes(ty, k) : 0;
            if (t0 == nullptr || t1 == nullptr || bytes == 0 || t0->bytes != bytes) {
                err = P + "0." + parts[r] + ": not an MXFP4 or low-bit [" + std::to_string(n) + "][" + std::to_string(k) +
                      "] matrix the engine reads (" + (t0 != nullptr ? t0->dtype : std::string("missing")) + ")";
                return false;
            }
            l.mx_file[r] = t0->file;
            l.mx_off[r] = t0->off;
            l.mx_stride[r] = l.n_exp > 1 ? t1->off - t0->off : bytes;
            for (int e = 0; e < l.n_exp; ++e) {
                const Tensor* t = pk.get(P + std::to_string(e) + "." + parts[r] + ".weight");
                if (t == nullptr || t->dtype != t0->dtype || t->bytes != bytes || t->file != l.mx_file[r] ||
                    t->off != l.mx_off[r] + (uint64_t) e * l.mx_stride[r]) {
                    err = P + std::to_string(e) + "." + parts[r] + ": missing, or not where the layer's others are";
                    return false;
                }
            }
            l.lay.trellis[r] = off;
            l.lay.suh[r] = l.lay.svh[r] = off;
            l.lay.K[r] = ty;
            off += align256(bytes) + (store_gb() > 0 ? ExpertStore::kSlack : 0);   // (the store's aligned reads spill into the gaps)
        }
        l.lay.fmt = gg::kFmtMXFP4;
        if (!gg::mx_experts(l.lay) && (gg::expert_type(l.lay, 0) != gg::expert_type(l.lay, 1) ||
                                       gg::expert_type(l.lay, 0) == gg::kTypeMXFP4 ||
                                       gg::expert_type(l.lay, 2) == gg::kTypeMXFP4)) {
            err = P + ": gate / up / down of types " + gg::type_name(l.lay.K[0]) + " / " + gg::type_name(l.lay.K[1]) +
                  " / " + gg::type_name(l.lay.K[2]) + " (gate and up must match; MXFP4 does not mix with the others)";
            return false;
        }
        l.blob = off;
        // a type the device kernels do not take (none of the i-quants now: gg::type_on_device lists them): the layer stays on the CPU lane, where
        // ggml-cpu's row dots read it - one warning, then the placement code keeps it out of VRAM
        for (int r = 0; r < 3; ++r)
            if (!gg::type_on_device(l.lay.K[r])) {
                l.cpu_only = true;
                std::fprintf(stderr,
                             "ds4: layer %d %s is %s: no GPU kernel for it, its experts stay on the CPU lane "
                             "(ggml-cpu's rows; VRAM tier: none)\n", l.il, r == 2 ? "down" : "gate/up",
                             gg::type_name(l.lay.K[r]));
                break;
            }
        return true;
    }
    const uint8_t* mx_mat(const Layer& l, int e, int r) const {   // expert e's matrix r: its RAM-tier slot, else the file's
        if (store) {
            const int s = store->find(l.il, e);
            if (s >= 0) return store->mat(s, r);
        }
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0)   // (a copy with NH > 1: part 0's rows only - whole: read_mx)
            return copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] + cp_lay[(size_t) l.il][(size_t) r]
                             : ram_slots[(size_t) l.ram[(size_t) e]] + l.lay.trellis[r];
        return pk.at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r]);
    }
    size_t mx_part_bytes(const Layer& l, int r) const {   // (its type's)
        return r < 2 ? (size_t) g.ff * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                     : (size_t) g.dim * gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff);
    }
    // (copy_mode) matrix r's rows, and the rows of one NUMA part of it
    int mx_n(int r) const { return r < 2 ? g.ff : g.dim; }
    size_t mx_row_b(const Layer& l, int r) const { return mx_part_bytes(l, r) / (size_t) mx_n(r); }
    // part X's rows [X n / NH, ..) of expert e's matrix r: its copy's part on node X, else where the file has them
    const uint8_t* mx_part_ptr(const Layer& l, int e, int r, int X) const {
        if (copy_mode && l.ram[(size_t) e] >= 0)
            return ram_part[(size_t) l.ram[(size_t) e]][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
        return mx_mat(l, e, r) + (size_t) X * (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
    }
    // a copy's NH parts -> the blob, and back (each matrix's rows of part X at X n / NH rows into the blob's matrix)
    void ram_to_blob(const Layer& l, int s, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.trellis[r] + (size_t) X * nb, ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], nb);
        }
    }
    void blob_to_ram(const Layer& l, const uint8_t* blob, int s) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], blob + l.lay.trellis[r] + (size_t) X * nb, nb);
        }
    }
    // expert e (from the file) -> its blob; and member m of nm's part of a blob (gate / up rows [m n_ff / nm, ..), down
    // inputs of the same range: in each row the blocks [m n_ff / nm / 32, ..)) and back
    bool read_mx(const Layer& l, int e, uint8_t* blob) const {   // (from its RAM-tier slot or copy when it has one)
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) {
            if (copy_mode) ram_to_blob(l, l.ram[(size_t) e], blob);
            else std::memcpy(blob, ram_slots[(size_t) l.ram[(size_t) e]], l.blob);
            return true;
        }
        return read_mx_file(l, e, blob);
    }
    bool read_mx_file(const Layer& l, int e, uint8_t* blob) const {   // (the movers: never a slot that may change)
        for (int r = 0; r < 3; ++r)
            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r], mx_part_bytes(l, r),
                            blob + l.lay.trellis[r]))
                return false;
        return true;
    }
    void mx_blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        const int ffh = g.ff / l.nm;
        mx_blob_to_slice(l, blob, m * ffh, ffh, l.hlay.trellis, dst);
    }
    void mx_half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        const int ffh = g.ff / l.nm;
        mx_slice_to_blob(l, src, m * ffh, ffh, l.hlay.trellis, blob);
    }
    // expert parallelism's n_ff slices: every layer with one must be a trunk layer with its experts on the i-quant CPU lane
    // alone (no VRAM tier, no GPU-visible store slots), in a process that is no remote stage's worker and skips no expert
    // by its RAM (STRATA_DS4_DISK_SKIP_W) - false (err) otherwise.  At load, and again by whatever sets a slice later
    // DS4_IQ_DOWN_PARTS (dparts above): the parts opened and checked against this pack and its layers; false (err): set
    // but not usable - a layer the parts cover with an expert missing or of another type or layout, no layer at all
    bool init_down_parts(std::string& err) {
        const char* dir = std::getenv("DS4_IQ_DOWN_PARTS");
        if (dir == nullptr || *dir == '\0') return true;
        namespace epp = ::strata::ds4::ep;
        auto refuse = [&](const std::string& why) {
            err = std::string("DS4_IQ_DOWN_PARTS=") + dir + ": " + why;
            dparts.close();
            dpl.clear();
            return false;
        };
        std::string perr;
        if (!dparts.open(dir, perr)) return refuse(perr);
        if (dparts.pack_hash() != stage::pack_hash(o.pack)) return refuse("cut from another pack than " + o.pack + " (pack_hash)");
        if (dparts.dim() != g.dim || dparts.n_ff() != g.ff || dparts.unit() != 256)
            return refuse("the parts' geometry (dim " + std::to_string(dparts.dim()) + ", n_ff " + std::to_string(dparts.n_ff()) +
                          ") is not the model's");
        const int nb = dparts.blocks();
        dpl.assign(L.size(), DownParts{});
        int on = 0, a0 = 0, b0 = 0;
        for (size_t li = 0; li < L.size(); ++li) {
            const Layer& l = L[li];
            if (l.mtp || (size_t) l.il != li || l.il >= dparts.layers() || !dparts.covers(l.il) || l.lay.fmt != gg::kFmtMXFP4 ||
                gg::mx_experts(l.lay))
                continue;   // (dpl and down_parts index by il: L's own order)
            const std::string ln = "layer " + std::to_string(l.il);
            const int td = gg::expert_type(l.lay, 2);
            const size_t ub = gg::type_row_bytes(td, g.ff) / (size_t) nb;
            epp::View v;
            int k = 0;
            while (k < nb && !dparts.view(l.il, 0, epp::kDown, k, v)) ++k;
            if (k == nb) return refuse(ln + ": covered, but no down part of expert 0");
            DownParts& d = dpl[li];
            d.a = v.a, d.b = v.b, d.ub = ub, d.rb = (size_t) (v.b - v.a) * ub;
            d.at.assign((size_t) l.n_exp, nullptr);
            for (int e = 0; e < l.n_exp; ++e) {
                if (!dparts.view(l.il, e, epp::kDown, d.a, v) || v.a != d.a || v.b != d.b || v.type != td || v.rows != (uint32_t) g.dim ||
                    v.stride != d.rb)
                    return refuse(ln + " expert " + std::to_string(e) + ": no down part [" + std::to_string(d.a) + ", " +
                                  std::to_string(d.b) + ") of the layer's type " + std::to_string(td) + " and row layout");
                d.at[(size_t) e] = v.data;
            }
            // (the parts' bytes against the pack's, read through dpart_at as the lane reads them: experts 0 and the last,
            // the slices from a and from b - 1, every row - an offset or a stride wrong fails the load, not the tokens)
            const size_t rowb = gg::type_row_bytes(td, g.ff);
            if (l.mx_file[2] < 0 || (size_t) l.mx_file[2] >= pk.maps.size() || pk.maps[(size_t) l.mx_file[2]].first == nullptr)
                return refuse(ln + ": its downs are not in a mapped file of the pack (the lane reads no expert in place)");
            for (int e : {0, l.n_exp - 1})
                for (int f : {d.a, d.b - 1}) {
                    const uint8_t* p = dpart_at(d, e, 256 * f);
                    const uint8_t* q = pk.at(l.mx_file[2], l.mx_off[2] + (uint64_t) e * l.mx_stride[2]) + (size_t) f * ub;
                    for (int r = 0; r < g.dim; ++r)
                        if (std::memcmp(p + (size_t) r * d.rb, q + (size_t) r * rowb, (size_t) (d.b - f) * ub) != 0)
                            return refuse(ln + " expert " + std::to_string(e) + ": the parts' row " + std::to_string(r) +
                                          " of blocks [" + std::to_string(f) + ", " + std::to_string(d.b) + ") is not the pack's");
                }
            ++on;
            a0 = d.a, b0 = d.b;
        }
        if (on == 0) return refuse("it covers none of the i-quant layers of the CPU lane");
        std::fprintf(stderr, "ds4: DS4_IQ_DOWN_PARTS: %d layers' down slices [%d, %d) (and [k, %d) of them) from %s, rows contiguous "
                             "(each layer's first and last expert byte for byte the pack's)\n",
                     on, a0, b0, b0, dir);
        return true;
    }
    // expert e's down rows from n_ff value ff0 (a block's start in [a, b)) in its parts: row 0's, the rows d.rb apart
    static const uint8_t* dpart_at(const DownParts& d, int e, int ff0) {
        return d.at[(size_t) e] + (size_t) (ff0 / 256 - d.a) * d.ub;
    }
    // the layer's down parts when they serve the slice [ff0, ff0 + ffn) (whole blocks inside [a, b): a row's slice is
    // its blocks from ff0 / 256 - a, the rows rb apart - a cut W pick's [0, cut) with cut < b too), else null
    const DownParts* down_parts(const Layer& l, int ff0, int ffn) const {
        if (dpl.empty() || l.il < 0 || (size_t) l.il >= dpl.size() || L[(size_t) l.il].il != l.il) return nullptr;
        const DownParts& d = dpl[(size_t) l.il];
        return !d.at.empty() && ff0 % 256 == 0 && ffn > 0 && ffn % 256 == 0 && ff0 / 256 >= d.a && ff0 + ffn <= 256 * d.b
                   ? &d : nullptr;
    }
    // DS4_IQ_WARM=split (iq_warm_split above), at the end of the load (the server's upload, rx_split's slices and the
    // parts known): what the CPU lane reads of each of its experts' picks read in and mapped, the most-routed up to
    // warm_bytes (in what order: DS4_IQ_WARM_ORDER), then with DS4_IQ_WARM_TOUCH=1 each of its pages read once through that mapping.  Neither read ahead nor dropped from the page cache (no POSIX_FADV_DONTNEED): a CPU-lane expert's
    // rows past its layer's slice (its gate / up [k, 9); with the parts the pack's down), and the expert server's experts
    // (rx_has; DS4_IQ_WARM_SRV=1: read as the lane's own, since a fill hands them back to it).  The lane reads those only when it computes a pick's [k, 9), or a pick whole, after all - a late or lost
    // reply, a pick the send gate keeps here, the link given up - and then they come in by page faults through the same
    // mapping: from the page cache where something left them, else from the disk (such an expert's first fallback pick
    // waits for its reads: ~2.9 MB of the pack's down, 1.9 MB of gate / up [6, 9) at k 6 on this pack).  No range of a
    // CPU-lane expert is one the lane never reads, so none is dropped; the VRAM experts' pages are, as before
    // (load_experts_mx)
    void warm_split() {
        if (!iq_warm_split) return;
        if (!file_experts || ram_tier || warm_bytes == 0) {
            std::fprintf(stderr, "ds4: DS4_IQ_WARM=split: nothing read ahead (%s)\n",
                         !file_experts ? "the CPU lane's experts are not read in place"
                         : ram_tier    ? "the RAM tier holds the CPU lane's hottest experts"
                                       : "no RAM to keep them: warm_bytes 0");
            return;
        }
        struct Rg {   // a range of the lane's mapping and where a pread reads it from: the pack's file, or the parts'
            const uint8_t* p;
            size_t n;
            int file;     // (the pack's; -1: the parts', fd)
            uint64_t off;
            os::fd_t fd;
        };
        struct W {
            double share;
            int li, e;
        };
        std::vector<W> ws;
        size_t n_rx = 0;
        for (const Layer& l : L) {
            if (l.lay.fmt != gg::kFmtMXFP4) continue;   // (load_experts_mx's layers: a slice on the i-quant ones, whole on the others)
            bool mapped = true;
            for (int r = 0; r < 3; ++r)
                mapped = mapped && l.mx_file[r] >= 0 && (size_t) l.mx_file[r] < pk.maps.size() && pk.maps[(size_t) l.mx_file[r]].first != nullptr;
            if (!mapped) continue;
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            double sum = 0;
            for (int e = 0; e < l.n_exp; ++e) sum += p[e];
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0 || (!l.ram.empty() && l.ram[(size_t) e] >= 0)) continue;   // (VRAM, a RAM copy)
                if (!rx_has.empty() && rx_has[(size_t) l.il * g.n_exp + e]) {
                    ++n_rx;
                    if (!iq_warm_srv) continue;   // (DS4_IQ_WARM_SRV: read as the lane's own)
                }
                ws.push_back({sum > 0 ? p[e] / sum : 0.0, l.il, e});
            }
        }
        std::stable_sort(ws.begin(), ws.end(), [](const W& a, const W& b) { return a.share > b.share; });
        const auto& pf = dparts.files();
        auto ranges = [&](const W& w, std::vector<Rg>& out) {
            const Layer& l = L[(size_t) w.li];
            const int ff0 = l.ep_ff0, ffn = l.ep_ffn < 0 ? g.ff - ff0 : l.ep_ffn;
            const uint64_t eo[3] = {l.mx_off[0] + (uint64_t) w.e * l.mx_stride[0], l.mx_off[1] + (uint64_t) w.e * l.mx_stride[1],
                                    l.mx_off[2] + (uint64_t) w.e * l.mx_stride[2]};
            for (int r = 0; r < 2; ++r) {
                const size_t rb = mx_row_b(l, r), o = (size_t) ff0 * rb;
                out.push_back({pk.at(l.mx_file[r], eo[r]) + o, (size_t) ffn * rb, l.mx_file[r], eo[r] + o, os::kBadFd});
            }
            if (const DownParts* d = down_parts(l, ff0, ffn)) {   // (the expert's part whole: its dim rows d->rb apart)
                const uint8_t* q = d->at[(size_t) w.e];
                for (const auto& f : pf)
                    if (f.map != nullptr && q >= f.map && q < f.map + f.bytes) {
                        out.push_back({q, (size_t) g.dim * d->rb, -1, (uint64_t) (q - f.map), f.fd});
                        return;
                    }
            }
            out.push_back({pk.at(l.mx_file[2], eo[2]), mx_part_bytes(l, 2), l.mx_file[2], eo[2], os::kBadFd});
        };
        // the budget, the most-routed first; the ranges by expert, read side by side by 12 threads
        std::vector<Rg> rg;
        double gu = 0, dpb = 0, dwb = 0, b = 0;
        size_t nw = 0;
        for (; nw < ws.size(); ++nw) {
            const size_t at = rg.size();
            ranges(ws[nw], rg);
            double eb = 0;
            for (size_t i = at; i < rg.size(); ++i) eb += (double) rg[i].n;
            if (b + eb > (double) warm_bytes) {
                rg.resize(at);
                break;
            }
            b += eb;
            gu += (double) (rg[at].n + rg[at + 1].n);
            (rg[at + 2].file < 0 ? dpb : dwb) += (double) rg[at + 2].n;
        }
        if (iq_warm_cold_first) std::reverse(rg.begin(), rg.end());   // (DS4_IQ_WARM_ORDER: the budget's coldest first)
        const double t0 = now_ms();
        std::atomic<size_t> ri{0};
        std::atomic<int> no_pop{0};
        std::atomic<long long> by_read{0}, failed{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 12 && !rg.empty(); ++t)
            th.emplace_back([&] {
                std::vector<uint8_t> buf;
                for (size_t i; (i = ri.fetch_add(1)) < rg.size();) {
                    const Rg& x = rg[i];
                    if (no_pop.load(std::memory_order_relaxed) == 0) {
                        const int r = os::populate(x.p, x.n);
                        if (r == 1) continue;
                        if (r == 0) no_pop.store(1, std::memory_order_relaxed);   // (not had here: pread from now on)
                    }
                    // (pread: the page cache filled, the page tables not - the lane's first read of a page faults)
                    if (buf.size() < x.n) buf.resize(x.n);
                    const bool ok = x.file >= 0 ? pk.read_at(x.file, x.off, x.n, buf.data()) : os::pread_full(x.fd, buf.data(), x.n, x.off);
                    by_read.fetch_add(1);
                    if (!ok) failed.fetch_add(1);
                }
            });
        for (auto& t : th) t.join();
        std::string how = no_pop.load() ? "read by pread (MADV_POPULATE_READ not had here): in the page cache, not mapped"
                                         : "read in and mapped (MADV_POPULATE_READ)";
        if (!no_pop.load() && by_read.load() > 0)
            how += ", " + std::to_string(by_read.load()) + " ranges by pread after it failed on them (not mapped)";
        if (failed.load() > 0) how += "; " + std::to_string(failed.load()) + " reads failed (those ranges left to the lane's faults)";
        how += iq_warm_cold_first ? "; the least-routed first (DS4_IQ_WARM_ORDER=cold)" : "; the most-routed first (DS4_IQ_WARM_ORDER)";
        if (iq_warm_srv) how += "; the expert server's " + std::to_string(n_rx) + " among them (DS4_IQ_WARM_SRV=1)";
        std::fprintf(stderr, "ds4: DS4_IQ_WARM=split: %zu of the CPU lane's experts read ahead in %.1f s (%.1f GB: the gate / up "
                             "of their slices %.1f GB, downs %.1f GB from the parts and %.1f GB the pack's whole ones) - %s; "
                             "left out: %zu past warm_bytes (%.1f GB), the expert server's %zu, and each one's rows past its "
                             "slice (read only by the lane's fallback: neither read ahead nor dropped)\n",
                     nw, (now_ms() - t0) / 1000.0, b / 1e9, gu / 1e9, dpb / 1e9, dwb / 1e9, how.c_str(), ws.size() - nw,
                     warm_bytes / 1e9, iq_warm_srv ? (size_t) 0 : n_rx);
        if (!iq_warm_touch) {
            std::fprintf(stderr, "ds4: DS4_IQ_WARM_TOUCH unset or 0: the pages read ahead not touched\n");
            return;
        }
        // (DS4_IQ_WARM_TOUCH) a byte of every 4 KiB page of the ranges read through the lane's mapping, by 12 threads
        // in the read's order: each page's entry gets its accessed bit (a range read by pread instead comes in here by
        // a fault, mapped like the others)
        const double t1 = now_ms();
        std::atomic<size_t> ti{0};
        std::atomic<uint64_t> pages{0};
        std::vector<std::thread> tt;
        for (int t = 0; t < 12 && !rg.empty(); ++t)
            tt.emplace_back([&] {
                uint64_t np = 0;
                for (size_t i; (i = ti.fetch_add(1)) < rg.size();) {
                    const uintptr_t a = (uintptr_t) rg[i].p, e = a + rg[i].n;
                    for (uintptr_t q = a & ~(uintptr_t) 4095; q < e; q += 4096, ++np)
                        (void) *(const volatile uint8_t*) (q < a ? a : q);
                }
                pages.fetch_add(np);
            });
        for (auto& t : tt) t.join();
        std::fprintf(stderr, "ds4: DS4_IQ_WARM_TOUCH: %llu pages of the read-ahead touched through the lane's mapping in %.1f s\n",
                     (unsigned long long) pages.load(), (now_ms() - t1) / 1000.0);
    }
    bool ep_slices_ok(std::string& err) const {
        if (!rx_slices_ok(err)) return false;
        for (const Layer& l : L)
            if (const char* why = ep_slice_refusal(l.ep_ff0, l.ep_ffn, g.ff, l.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(l.lay),
                                                   l.n_slots > 0, !l.zc.empty(), l.mtp, o.stage_worker,
                                                   disk_skip_w > 0.0f, rx_tagged)) {
                err = "expert parallelism: layer " + std::to_string(l.il) + "'s n_ff slice [" + std::to_string(l.ep_ff0) + ", " +
                      std::to_string(l.ep_ff0 + l.ep_ffn) + "): " + why;
                return false;
            }
        return true;
    }
    // --remote-experts beside n_ff slices: refused (rx::slice_refusal) - at load before the server's upload, and with the
    // slices' own checks (ep_slices_ok) for whatever sets a slice later
    bool rx_slices_ok(std::string& err) const {
        for (const Layer& l : L)
            if (const char* why = rx::slice_refusal(!o.remote_experts.empty(), l.ep_ff0, l.ep_ffn, g.ff, rx_tagged)) {
                err = "expert parallelism: layer " + std::to_string(l.il) + "'s n_ff slice [" + std::to_string(l.ep_ff0) + ", " +
                      std::to_string(l.ep_ff0 + l.ep_ffn) + "): " + why;
                return false;
            }
        return true;
    }
    // the same for any slice [ff0, ff0 + ffn) of n_ff in whole blocks of the experts' types (ds4_ffslice.hpp), the slice's
    // matrices at so[3]: a tensor parallel member's even share, an expert parallel side's part of n_ff
    FfSlice mx_slice(const Layer& l, int ff0, int ffn) const {
        const int td = gg::expert_type(l.lay, 2);
        FfSlice s;
        s.dim = g.dim;
        s.ff0 = ff0;
        s.ffn = ffn;
        s.gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim);
        s.dr = gg::type_row_bytes(td, g.ff);
        s.ds = gg::type_row_bytes(td, ffn);
        s.d_off = gg::type_row_bytes(td, ff0);
        return s;
    }
    void mx_blob_to_slice(const Layer& l, const uint8_t* blob, int ff0, int ffn, const size_t so[3], uint8_t* dst) const {
        ff_blob_to_slice(mx_slice(l, ff0, ffn), blob, l.lay.trellis, dst, so);
    }
    void mx_slice_to_blob(const Layer& l, const uint8_t* src, int ff0, int ffn, const size_t so[3], uint8_t* blob) const {
        ff_slice_to_blob(mx_slice(l, ff0, ffn), src, so, blob, l.lay.trellis);
    }

    bool expert_layout(Layer& l, std::string& err) {
        if (const Tensor* t = pk.get(l.name + "ffn.experts.0.w1.weight"); t != nullptr && gg_type_of(t->dtype) > 0)
            return expert_layout_mx(l, err);
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        const char* pieces[3] = {".trellis", ".suh", ".svh"};
        l.region_off.resize((size_t) l.n_exp);
        l.region_bytes.resize((size_t) l.n_exp);
        l.region_file.resize((size_t) l.n_exp);
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const Tensor* tr = pk.get(P + "0." + parts[r] + ".trellis");
            if (tr == nullptr || pk.get(P + "0." + parts[r] + ".mul1") == nullptr) {
                err = P + "0." + parts[r] + ": not an EXL3 (mul1) matrix";
                return false;
            }
            l.lay.K[r] = (int) tr->shape[2] / 16;
            l.lay.trellis[r] = off;
            off += align256(tr->bytes);
            l.lay.suh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".suh")->bytes);
            l.lay.svh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".svh")->bytes);
        }
        l.lay.cb = 2;
        l.blob = off;
        for (int e = 0; e < l.n_exp; ++e) {
            const std::string E = P + std::to_string(e) + ".";
            uint64_t lo = UINT64_MAX, hi = 0;
            int file = -1;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const Tensor* t = pk.get(E + parts[r] + pieces[q]);
                    if (t == nullptr) {
                        err = E + parts[r] + pieces[q] + ": missing";
                        return false;
                    }
                    if (file < 0) file = t->file;
                    if (t->file != file) {
                        err = E + ": split across files";
                        return false;
                    }
                    lo = std::min(lo, t->off);
                    hi = std::max(hi, t->off + t->bytes);
                    if (e == 0) l.piece_bytes[r][q] = t->bytes;
                    else if (t->bytes != l.piece_bytes[r][q]) {
                        err = E + parts[r] + ": not the shape of expert 0's";
                        return false;
                    }
                }
            l.region_off[(size_t) e] = lo;
            l.region_bytes[(size_t) e] = hi - lo;
            l.region_file[(size_t) e] = file;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const size_t src = pk.get(E + parts[r] + pieces[q])->off - lo;
                    if (e == 0) l.piece_src[r][q] = src;
                    else if (src != l.piece_src[r][q]) {
                        err = E + ": the pieces lie differently from expert 0's";
                        return false;
                    }
                }
        }
        return true;
    }
    // the region of expert e (read from the file) -> its blob
    void repack(const Layer& l, const uint8_t* region, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t dst[3] = {l.lay.trellis[r], l.lay.suh[r], l.lay.svh[r]};
            for (int q = 0; q < 3; ++q) std::memcpy(blob + dst[q], region + l.piece_src[r][q], l.piece_bytes[r][q]);
        }
    }

    // the region of expert e -> its CPU-lane part X (columns [X n / NH, (X + 1) n / NH) of each matrix)
    void repack_part(const Layer& l, const uint8_t* region, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = region + l.piece_src[r][0];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], region + l.piece_src[r][1], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], region + l.piece_src[r][2] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }

    // ---- loading
    bool load(const Options& opt, std::string& err) {
        o = opt;
        const double t0 = now_ms();
        if (const char* v = std::getenv("DS4_LC_THP"); v != nullptr && *v != '\0' && std::strcmp(v, "1") != 0) {
            if (std::strcmp(v, "0") != 0) {
                err = "DS4_LC_THP: 0 (the prompt's host buffers on 4 KiB pages); unset, empty or 1: huge pages advised";
                return false;
            }
            lc_thp = false;
            std::fprintf(stderr, "ds4: DS4_LC_THP=0: the layer-major prompt's layer cache and chunk states, and the "
                                 "prompt's other host buffers, on 4 KiB pages (no MADV_HUGEPAGE)\n");
        }
        if (const char* v = std::getenv("DS4_LM_PRE"); v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0) {
            if (std::strcmp(v, "1") != 0) {
                err = "DS4_LM_PRE: 1 (the layer-major prompt's host memory allocated at the load); unset, empty or 0: off";
                return false;
            }
            lm_pre = true;
        }
        if (!pk.load(o.pack, err)) return false;
        g.from(pk);
        if (g.dim == 0 || g.nl == 0 || (int) g.ratios.size() < g.nl) {
            err = o.pack + "/ds4.txt: incomplete geometry";
            return false;
        }
        if (g.hd != 512 || g.ix_hd != 128 || g.k_exp > 8 || g.hc > 8) {
            err = "geometry not covered (head_dim " + std::to_string(g.hd) + ", index_head_dim " +
                  std::to_string(g.ix_hd) + ", experts per token " + std::to_string(g.k_exp) + ")";
            return false;
        }
        lo = o.stage_begin;
        hi = o.stage_end < 0 ? g.nl : o.stage_end;
        if (lo < 0 || lo >= hi || hi > g.nl) {
            err = "layers [" + std::to_string(lo) + ", " + std::to_string(hi) + "): not a range of the trunk's " +
                  std::to_string(g.nl) + " layers";
            return false;
        }
        if (o.remote_stage && o.stage_worker) {
            err = "a remote stage's main process or its worker, not both";
            return false;
        }
        if (o.stage_worker && (lo == 0 || hi != g.nl)) {   // (v1: no relay - the worker runs the trunk to its end)
            err = "a remote stage's worker runs layers K to the last (--stage-begin K, 1 or more; no --stage-end)";
            return false;
        }
        if (o.remote_stage && (lo != 0 || hi == g.nl)) {
            err = "a remote stage's main process runs layers 0 to K - 1 (--stage-end K, below " + std::to_string(g.nl) +
                  "; no --stage-begin)";
            return false;
        }
        if (bf16_rows() && !row_geo().compact_ok()) {
            err = "bf16 rows: a row's streams (" + std::to_string(row_geo().h) + " floats) are not a whole number of words";
            return false;
        }
        ranged = lo > 0 || hi < g.nl || linked();
        if (ranged) {   // (a slot, the drafter, a picture or the prompt path's chunks would need the other layers - a
                        // remote stage's chunks cross the link as they leave this process's last layer, its slots and
                        // checkpoints are each side's part of them)
            // (a remote stage's main process keeps the drafter: the worker's reply carries the taps of its layers; its
            // blocks' rings are this process's layers, in its checkpoints and slots as in one process's)
            const bool d_here = o.remote_stage && o.dspark &&
                                stage::reply_fits(g.dim, (int) g.d_targets.size(), (int64_t) g.hc * g.dim);
            std::fprintf(stderr, "ds4: layers [%d, %d) of %d here%s: prompts in %s of %d, %s, no pictures%s\n", lo, hi, g.nl,
                         o.remote_stage ? " (the rest: the remote stage's worker)" : o.stage_worker ? " (a remote stage's worker)" : "",
                         !linked() ? "windows"
                         : d_here ? "chunks over the link (the chunked path's; windows if the drafter loads), or windows"
                                  : "chunks over the link (the chunked path's), or windows",
                         kWin, d_here ? "DSpark as --dspark says (the worker's taps come back with its rows)" : "no DSpark",
                         ckpt_link_note());
            if (!d_here) o.dspark = false;
            o.vision = false;
        }
        freqs_win = rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow);
        freqs_comp = rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow);

        // devices
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            err = "no CUDA device";
            return false;
        }
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < ndev; ++i) ids.push_back(i);
        devs.resize(ids.size());
        u9::mask();       // (both print their state at load)
        attn_split();
        // DS4_CUDA_SYNC=spin|yield|block: how this process's host threads wait on its GPUs (cudaStreamSynchronize and the
        // like: a decode token's two - the head's, then sample()'s short one; the layers do not wait on the host, their
        // routing goes through mapped memory the service thread polls). Unset: the driver's default,
        // auto, which spins while the process's contexts are fewer than the CPUs - here the head's wait spins its whole
        // time (Stats::head_cpu_ms), a busy thread on a CPU no one pinned it to, likely a lane core's sibling (the
        // lane-bench's :spin=, slot-20261025T1948Z). block sleeps in the driver until the GPU's interrupt: the wait's CPU
        // free, its return later by the wake-up (measured on an RTX 3070, Windows: +43 us a wait, the thread's cycles
        // 0.1 M against spin's 6 M over 2 ms). yield spins too, giving the CPU to a ready thread between its polls. The
        // flags are the primary context's (cudaInitDevice, CUDA 12+), read back after; not on HIP. The fill thread
        // (ds4_rx_fill_link.cpp) has no switch: it does not spin (Conn spin_ms 0 - a poll() wait -, sleep_for between
        // its pieces)
        unsigned sched = 0;
        const char* sync_env = std::getenv("DS4_CUDA_SYNC");
        if (sync_env != nullptr) {
            const std::string v = sync_env;
            sched = v == "spin" ? cudaDeviceScheduleSpin : v == "yield" ? cudaDeviceScheduleYield
                                                         : v == "block" ? cudaDeviceScheduleBlockingSync : 0u;
            if (sched == 0) {
                err = "DS4_CUDA_SYNC=" + v + ": spin, yield, block or unset";
                return false;
            }
#if defined(STRATA_USE_HIP)
            err = "DS4_CUDA_SYNC: CUDA only";
            return false;
#endif
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            Dev& d = devs[i];
            d.id = ids[i];
#if !defined(STRATA_USE_HIP)
            // (cudaInitDeviceFlagsAreValid: without it the call ignores sched and returns success - measured on CUDA 13)
            if (sched != 0) CK(cudaInitDevice(d.id, sched, cudaInitDeviceFlagsAreValid));
#endif
            CK(cudaSetDevice(d.id));
#if !defined(STRATA_USE_HIP)
            if (sched != 0) {
                unsigned f = 0;
                CK(cudaGetDeviceFlags(&f));
                if ((f & cudaDeviceScheduleMask) != sched) {
                    err = "DS4_CUDA_SYNC=" + std::string(sync_env) + ": CUDA" + std::to_string(d.id) + "'s flags are " +
                          std::to_string(f) + ": the schedule was not applied";
                    return false;
                }
                std::fprintf(stderr, "ds4: DS4_CUDA_SYNC=%s: CUDA%d's host waits %s (device flags 0x%x)\n", sync_env, d.id,
                             sched == cudaDeviceScheduleSpin    ? "spin"
                             : sched == cudaDeviceScheduleYield ? "spin, yielding the CPU between polls"
                                                                : "block until the GPU's interrupt",
                             f);
            }
#endif
            CK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
            if (!x3::prepare(d.st) || (pk.gguf && !gg::prepare(d.st))) {
                err = "exl3 scratch on CUDA" + std::to_string(d.id);
                return false;
            }
            CK(cudaMemGetInfo(&d.free0, &d.total));
        }
        // the groups' width: o.tp, less while the experts' blocks do not split in it (a member takes n_ff / tp of each
        // expert: the down's inputs in whole blocks - a GGUF's 256-value types split n_ff 2304 in 1, 3 or 9 parts)
        int tp = o.tp;
        if (pk.gguf && tp > 1) {
            int blk = 32;
            for (const auto& kv : pk.t) {
                const std::string& n = kv.first;
                if (n.find(".ffn.experts.") == std::string::npos || n.size() < 9 ||
                    n.compare(n.size() - 9, 9, "w2.weight") != 0)
                    continue;
                const int ty = gg_type_of(kv.second.dtype);
                if (ty >= 0) blk = std::max(blk, gg::type_block(ty));
            }
            while (tp > 1 && (g.ff % tp != 0 || (g.ff / tp) % blk != 0 || (g.ff / tp) % (std::getenv("DS4_TP_ALIGN") ? std::atoi(std::getenv("DS4_TP_ALIGN")) : 128) != 0 || g.heads % tp != 0 ||
                              g.o_groups % tp != 0))
                --tp;
            if (tp != o.tp)
                std::fprintf(stderr, "ds4: tensor parallel %d -> %d: the experts' %d-value blocks split n_ff %d no other way "
                                     "(each GPU its own layers)\n", o.tp, tp, blk, g.ff);
        }
        // the groups: runs of tp devices of one kind (their memory), a device unlike the next alone
        // - or, tp the number of GPUs, one group of all of them whatever their kind (its tightest member sets the
        // experts' room); threads mode for the windows (decode) where a group is 2 wide or more or --threads on, the
        // prompt path (its own thread a GPU) as asked for - also where the experts' blocks took the groups' width down
        // to 1 (a 256-value type's: Maya's quants), which left every prompt in windows of 16
        par = o.threads || tp > 1;
        // the chunked (layer-major) prompt path: with --tp / --threads as before; --prefill-chunk N (N > 0) or
        // kChunkedPromptDefault bring it to a single card's default run too (a host thread a GPU for the prompt only;
        // decode stays as asked).  --prefill-chunk 0 keeps the windows of 16 (DS4_WINDOWS=1 the same).  Off by default
        // until a prompt costs the decode nothing: its buffers hold VRAM the windowed path gives to experts
        // (a layer range: windows, said above; a remote stage: chunks across, its rows at the link's width - the
        // compact ones packed by the engine, ds4_stage.hpp)
        prompt_par = (!ranged || linked()) &&
                     (par || o.tp > 1 ||
                      ((o.prefill_chunk > 0 || (o.prefill_chunk < 0 && kChunkedPromptDefault)) &&
                       std::getenv("DS4_WINDOWS") == nullptr));
        const bool tp_all = tp > 1 && tp == (int) devs.size();
        for (size_t i = 0; i < devs.size();) {
            Group G;
            G.mem.push_back((int) i);
            size_t j = i + 1;
            while ((int) G.mem.size() < tp && j < devs.size() && (tp_all || devs[j].total == devs[i].total))
                G.mem.push_back((int) j++);
            if ((int) G.mem.size() < tp && G.mem.size() > 1) {   // a short run at the end: devices of their own
                for (int m : G.mem) {
                    Group one;
                    one.mem.push_back(m);
                    groups.push_back(one);
                }
            } else groups.push_back(G);
            i = j;
        }
        for (size_t gi = 0; gi < groups.size(); ++gi)
            for (size_t m = 0; m < groups[gi].mem.size(); ++m) {
                devs[(size_t) groups[gi].mem[m]].grp = (int) gi;
                devs[(size_t) groups[gi].mem[m]].mi = (int) m;
            }
        // whether the CPU lane's experts are read in place (--mmap-experts; auto: the pack's expert bytes beyond what the
        // GPUs can take - their free memory less the other weights and the prompt path's buffers - do not fit the RAM
        // this process can get beside kRamMargin), decided here because it sets the prompt chunk's default: in place
        // every chunk streams every expert VRAM does not hold from the files, so the larger the chunk the fewer times
        if (pk.gguf) file_experts = true;   // (a GGUF's: the files stay their source - copy_experts copies them into RAM
                                            // after VRAM's are in, by --mmap-experts)
        else if (o.mmap_experts >= 0) file_experts = o.mmap_experts > 0;
        else {
            double exp_b = 0, other_b = 0, room = 0;
            for (const auto& kt : pk.t) {
                const std::string& n = kt.first;
                if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                    (!o.dspark && n.rfind("mtp.", 0) == 0) || !tensor_here(n))
                    continue;
                (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
            }
            for (const Dev& d : devs)
                room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) -
                                          (prompt_par ? 1.5 : 0.4) * (1 << 30)) *
                        o.vram_experts;
            file_experts = exp_b - std::max(0.0, room - other_b) + (double) kRamMargin > (double) ram_avail();
        }
        const bool own_default = !(par || o.tp > 1);   // (the single-card default run the chunked path newly takes: chunks of 1024 - 8K prompt 370 tok/s on 4 x R9700, 214 at 8192; the threads / tp runs keep theirs)
        // (a remote stage, either side: stage::kLinkChunk)
        chunk = prompt_par ? (o.prefill_chunk >= 0 ? o.prefill_chunk
                              : linked()          ? stage::kLinkChunk
                              : own_default       ? 1024
                              : file_experts      ? kVlRows
                                                  : 2048)
                           : 0;
        const char* dc = std::getenv("DS4_CHUNK");   // (tests: down to 16, a short prompt in several chunks)
        if (dc != nullptr) chunk = prompt_par ? std::atoi(dc) : 0;
        if (const char* v = std::getenv("DS4_CHUNK_MIN")) chunk_min = std::max(1, std::atoi(v));
        if (chunk > 0) {
            chunk = std::min(kMaxChunk, std::max(dc != nullptr ? 16 : 256, chunk / 16 * 16));
            eg_cap = (size_t) chunk;
            tap_cap = std::max(kWin, 128);
        } else if (ranged) {   // (said with the range)
        } else   // (only as asked for: --prefill-chunk 0, or --tp 1 with --threads off)
            std::fprintf(stderr,
                         "ds4: prompts run in windows of %d tokens (%s): long prompts prefill far slower than in\n"
                         "     chunks (a 7-GPU RTX 3090 rig, Maya-S read in place: 8K / 32K tokens 115 / 99 tokens/s,\n"
                         "     461 / 1010 in chunks of %d) - %s\n",
                         kWin, o.prefill_chunk == 0 ? "--prefill-chunk 0" : "--tp 1 with --threads off", kVlRows,
                         o.prefill_chunk == 0 ? "leave --prefill-chunk out for chunks"
                                              : "leave --tp 1 out, or pass --threads on, for chunks");
        // the vision tower on the first GPU, before the split: the layers get what it leaves
        if (o.vision && !load_vision(err)) {   // (a tower that cannot load leaves the model reading text: said loudly,
            if (o.vision_only) return false;   // and the server is told pictures are off - info_lines)
            std::fprintf(stderr, "ds4: WARNING: the vision tower did not load (%s): pictures are OFF, the model reads "
                                 "text only\n", err.c_str());
            vision_err = err.empty() ? "unknown reason" : err;
            err.clear();
            vt.release();
            img_tok = -1;
        }
        if (o.vision_only) return true;

        // layers: roles and the sources they read
        if (o.dspark_auto && o.dspark && g.n_mtp > 0) {   // (see Options::dspark_auto)
            double exp_b = 0, room_b = 0;
            for (const auto& kt : pk.t)
                if (kt.first.rfind("layers.", 0) == 0 && kt.first.find(".ffn.experts.") != std::string::npos) exp_b += (double) kt.second.bytes;
            for (const Dev& d : devs) room_b += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20));
            if (exp_b > 1.5 * room_b) {
                o.dspark = false;
                std::fprintf(stderr, "ds4: DSpark not loaded: the routed experts (%.0f GB) are far more than the GPU holds (%.0f GB), where its verify "
                                     "rounds read 3.3x a token's bytes and gain nothing (--dspark on loads it anyway)\n", exp_b / 1e9, room_b / 1e9);
            }
        }
        if (!o.dspark) g.n_mtp = 0;
        if (o.remote_stage && g.n_mtp > 0 && chunk > 0) {   // (the drafter here: its seeding after a prompt reads the
            // taps of its target layers, the worker's - a window's reply carries them, a prompt chunk's does not)
            chunk = 0;
            prompt_par = false;
            eg_cap = (size_t) kWin;
            tap_cap = kWin;
            std::fprintf(stderr, "ds4: the drafter on a remote stage's main process: prompts in windows of %d (a prompt "
                                 "chunk's reply carries no taps of the worker's layers)\n", kWin);
        }
        if (o.dspark_block > 0 && std::getenv("DS4_DSPARK_MAX") == nullptr) max_drafts = o.dspark_block;   // (--dspark-block N: at most N drafts a round)
        L.resize((size_t) (g.nl + g.n_mtp));
        int last_src = -1;
        int far_src = -1;   // (a layer range) the latest KV source, when it is a layer outside the range
        for (int il = 0; il < g.nl + g.n_mtp; ++il) {
            Layer& l = L[(size_t) il];
            l.il = il;
            l.mtp = il >= g.nl;
            l.name = l.mtp ? "mtp." + std::to_string(il - g.nl) + "." : "layers." + std::to_string(il) + ".";
            l.n_exp = l.mtp ? g.d_n_exp : g.n_exp;
            l.k_exp = l.mtp ? g.d_k : g.k_exp;
            l.ratio = g.ratios[(size_t) il];
            if (l.mtp) {   // window attention only: no compressor, indexer or Engram
                if (l.ratio != 0 || l.n_exp <= 0 || l.k_exp <= 0 || l.k_exp > kPicks) {
                    err = l.name + ": a DSpark block of compress ratio " + std::to_string(l.ratio) + ", " +
                          std::to_string(l.n_exp) + " experts / " + std::to_string(l.k_exp) + " is not covered";
                    return false;
                }
                if (!expert_layout(l, err)) return false;
                continue;
            }
            l.kv_src = std::find(g.kv_src.begin(), g.kv_src.end(), il) != g.kv_src.end();
            l.ix_src = std::find(g.ix_src.begin(), g.ix_src.end(), il) != g.ix_src.end();
            l.owns_k = l.kv_src;
            l.cand_src = il == g.cand_src;
            l.uses_cand = g.cand_src >= 0 && g.cand_src < il && l.ix_src;
            if (!local(il)) {   // (another process's: its roles stay - the window plan follows every layer's - but no
                l.dev = l.grp = -1;   // device, no experts, no source of its own here)
                l.n_exp = l.k_exp = 0;
                if (l.kv_src) {
                    last_src = -1;
                    far_src = il;
                }
                continue;
            }
            if (l.kv_src) {
                Src s;
                s.layer = il;
                s.ratio = l.ratio;
                s.cap = (o.max_context + l.ratio - 1) / l.ratio + 1;
                srcs.push_back(s);
                last_src = (int) srcs.size() - 1;
                far_src = -1;
            }
            if (l.ratio > 0) {
                if (last_src < 0 && far_src >= 0) {   // (its compressed KV would have to cross too)
                    err = "layer " + std::to_string(il) + " reads the compressed KV of layer " + std::to_string(far_src) +
                          ", outside layers [" + std::to_string(lo) + ", " + std::to_string(hi) +
                          "): the range must start at a KV source";
                    return false;
                }
                if (last_src < 0 || srcs[(size_t) last_src].ratio != l.ratio) {
                    err = "layer " + std::to_string(il) + ": compress ratio " + std::to_string(l.ratio) +
                          " without a KV source of that ratio before it";
                    return false;
                }
                l.src = last_src;
                l.ik_src = last_src;
            }
            auto eg = std::find(g.eg_layers.begin(), g.eg_layers.end(), il);
            if (eg != g.eg_layers.end()) l.eg = (int) (eg - g.eg_layers.begin());
            if (!expert_layout(l, err)) return false;
        }

        // the split: each layer's dense weights + its experts, the head on the last GPU
        for (Layer& l : L) {
            if (elsewhere(l)) continue;   // (another process's: nothing of it here)
            const std::string& P = l.name;
            size_t b = 0;
            if (l.mtp && l.il == g.nl) b += mat_bytes("mtp.0.main_proj");
            for (const char* m : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1",
                                  "ffn.shared_experts.w2", "ffn.shared_experts.w3", "attn.compressor.wkv",
                                  "attn.compressor.wgate", "attn.indexer.wq_b", "attn.indexer.wk", "engram.wkv"})
                b += mat_bytes(P + m);
            for (int gi = 0; gi < g.o_groups; ++gi) b += mat_bytes(P + "attn.wo_a.slice." + std::to_string(gi));
            b += tensor_bytes(P + "hc_attn_fn") + tensor_bytes(P + "hc_ffn_fn") + tensor_bytes(P + "ffn.gate.weight");
            b += (size_t) g.win * win_rb() + (2u << 20);
            l.dense_bytes = b;
        }
        const size_t head_b = o.stage_worker ? 0 : mat_bytes("head") + (size_t) g.vocab * 4 + (16u << 20);   // (a worker: none)
        const size_t per_dev_fixed = (size_t) o.reserve_mib << 20;
        const size_t kv_per_src_dev = 0;   // counted after the split
        (void) kv_per_src_dev;
        std::vector<double> avail(groups.size(), 0.0);
        double avail_sum = 0;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            for (int m : groups[gi].mem)
                avail[gi] += std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed - 400.0 * (1 << 20));
            avail_sum += avail[gi];
        }
        double need_sum = (double) head_b;
        for (const Layer& l : L) need_sum += (double) l.dense_bytes + (double) l.blob * l.n_exp;
        {
            // contiguous layer ranges, each group's share of the total by its free memory
            size_t d = 0;
            double cum = 0, target = avail[0] / avail_sum * need_sum;
            std::vector<Layer*> order;   // the drafter's blocks first (the first GPU), then the trunk's layers
            for (Layer& l : L)
                if (l.mtp) order.push_back(&l);
            for (Layer& l : L)
                if (!l.mtp && !elsewhere(l)) order.push_back(&l);
            for (Layer* lp : order) {
                Layer& l = *lp;
                const double lb = (double) l.dense_bytes + (double) l.blob * l.n_exp;
                if (cum + lb / 2 > target && d + 1 < groups.size()) {
                    ++d;
                    target += avail[d] / avail_sum * need_sum;
                }
                l.grp = (int) d;
                cum += lb;
            }
            // with the chunked prompt path: the contiguous split that streams the least into its busiest member (its
            // VRAM short of its layers' experts, per member) - a chunk's groups run as a pipeline, its slowest stage
            // sets the pace; a token's decode passes every layer either way, the experts in VRAM the same in all
            // (a remote stage's range: the split above - this one spans the whole trunk)
            if (chunk > 0 && !ranged && groups.size() > 1 && groups.size() <= 6 && std::getenv("DS4_SPLIT") == nullptr) {
                const int G = (int) groups.size(), NL = g.nl;
                std::vector<double> cap(groups.size());
                for (int gi = 0; gi < G; ++gi) {
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    double mn = 1e300;
                    for (int m : groups[(size_t) gi].mem)
                        mn = std::min(mn, std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed -
                                                            400.0 * (1 << 20) - 1.1 * (1 << 30)));   // (the chunk's buffers)
                    cap[(size_t) gi] = mn * nm;
                }
                for (const Layer& l : L)
                    if (l.mtp) cap[0] -= (double) l.dense_bytes + (double) l.blob * l.n_exp;
                cap[(size_t) G - 1] -= (double) head_b;
                std::vector<double> pe(NL + 1, 0.0), pd(NL + 1, 0.0);   // prefix sums: experts, dense
                for (int il = 0; il < NL; ++il) {
                    pe[(size_t) il + 1] = pe[(size_t) il] + (double) L[(size_t) il].blob * L[(size_t) il].n_exp;
                    pd[(size_t) il + 1] = pd[(size_t) il] + (double) L[(size_t) il].dense_bytes;
                }
                auto cost = [&](int gi, int a, int b) {   // per-member streaming of layers [a, b) on group gi
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    const double dense = (pd[(size_t) b] - pd[(size_t) a]) * (nm > 1 ? 1.3 : 1.0);
                    return std::max(0.0, pe[(size_t) b] - pe[(size_t) a] - (cap[(size_t) gi] - dense)) / nm;
                };
                // best[gi][b]: the least max cost of layers [0, b) over groups 0..gi (each at least one layer)
                std::vector<std::vector<double>> best((size_t) G, std::vector<double>((size_t) NL + 1, 1e300));
                std::vector<std::vector<int>> cut((size_t) G, std::vector<int>((size_t) NL + 1, 0));
                for (int b = 1; b <= NL; ++b) best[0][(size_t) b] = cost(0, 0, b);
                for (int gi = 1; gi < G; ++gi)
                    for (int b = gi + 1; b <= NL; ++b)
                        for (int a = gi; a < b; ++a) {
                            const double v = std::max(best[(size_t) gi - 1][(size_t) a], cost(gi, a, b));
                            if (v < best[(size_t) gi][(size_t) b]) {
                                best[(size_t) gi][(size_t) b] = v;
                                cut[(size_t) gi][(size_t) b] = a;
                            }
                        }
                int b = NL;
                for (int gi = G - 1; gi >= 0; --gi) {
                    const int a = gi > 0 ? cut[(size_t) gi][(size_t) b] : 0;
                    for (int il = a; il < b; ++il) L[(size_t) il].grp = gi;
                    b = a;
                }
            }
            // DS4_SPLIT=a,b,..: the trunk layers each group takes (the drafter stays with the first)
            if (const char* sp = std::getenv("DS4_SPLIT")) {
                std::vector<int> n;
                for (const char* q = sp; *q;) {
                    n.push_back(std::atoi(q));
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
                int il = lo;   // (a layer range: its layers, from its first)
                for (size_t gi = 0; gi < n.size() && gi < groups.size(); ++gi)
                    for (int j = 0; j < n[gi] && il < hi; ++j) L[(size_t) il++].grp = (int) gi;
                for (; il < hi; ++il) L[(size_t) il].grp = (int) groups.size() - 1;
            }
            for (Layer& l : L) {
                if (elsewhere(l)) continue;
                l.dev = groups[(size_t) l.grp].mem[0];
                l.nm = l.mtp ? 1 : (int) groups[(size_t) l.grp].mem.size();   // (the drafter's blocks: on the lead alone)
            }
            head_dev = L[(size_t) hi - 1].dev;   // (a layer range: after its last layer; a worker: its rows' way out)
            d_dev = g.n_mtp > 0 ? L[(size_t) g.nl].dev : 0;
        }
        // a remote stage: the link's rows go in and out of one GPU at each end (v1: a tensor-parallel group's members
        // would each need them)
        if (linked()) {
            const size_t first = groups[(size_t) L[(size_t) lo].grp].mem.size(), last = groups[(size_t) devs[(size_t) head_dev].grp].mem.size();
            if ((o.stage_worker && first > 1) || (o.remote_stage && last > 1)) {
                err = "remote stage: the GPUs at this end of the link are a tensor-parallel group of " +
                      std::to_string(o.stage_worker ? first : last) + ": v1 hands one GPU's rows over (--tp 1)";
                return false;
            }
        }
        // the groups' members: copies of each layer's device state, a member's part of an expert
        for (Layer& l : L) {
            l.ff0 = 0;
            l.ffn = g.ff;
            ep_resolve(l.ep_ff0, l.ep_ffn, g.ff);
            l.ah0 = l.og0 = 0;
            l.ahn = g.heads;
            l.ogn = g.o_groups;
            if (l.nm == 1) continue;
            const int na = head_members(l.nm);
            if (g.heads % na != 0 || g.o_groups % na != 0 || g.ff % 128 != 0 || g.ff / 128 < l.nm) {
                err = "tensor parallel " + std::to_string(l.nm) + ": the heads, output groups or FFN do not split";
                return false;
            }
            for (int m = 1; m < l.nm; ++m) {
                Layer c = l;
                c.mates.clear();
                c.mi = m;
                c.dev = groups[(size_t) l.grp].mem[(size_t) m];
                l.mates.push_back(std::move(c));
            }
            // each member's FFN columns and its part of an expert's layout: an EXL3 expert's in whole 128-blocks, as even
            // as they go (split128, the odd blocks going round the members layer by layer); a GGUF's in even shares (its
            // groups are only as wide as its experts' blocks split evenly: mx_blob_to_half's n_ff / nm)
            const bool gguf_x = l.lay.fmt == gg::kFmtMXFP4;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const int ha = m - (l.nm - na);   // (its place among the members with heads)
                x.ahn = ha < 0 ? 0 : g.heads / na;
                x.ah0 = ha < 0 ? 0 : ha * x.ahn;
                x.ogn = ha < 0 ? 0 : g.o_groups / na;
                x.og0 = ha < 0 ? 0 : ha * x.ogn;
                if (gguf_x) {
                    x.ffn = g.ff / l.nm;
                    x.ff0 = m * x.ffn;
                } else
                    split128(g.ff, l.nm, m, x.ff0, x.ffn, l.il % l.nm);   // (rot: the odd blocks go round the members)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : x.ffn, n = r < 2 ? x.ffn : g.dim;
                    x.hlay.K[r] = l.lay.K[r];
                    x.hlay.trellis[r] = off;
                    if (gguf_x) {   // gate / up rows, down inputs: [n][k / block] blocks
                        const int ty = gg::expert_type(l.lay, r);
                        if (k % gg::type_block(ty) != 0) {
                            err = "tensor parallel " + std::to_string(l.nm) + ": the experts' " + gg::type_name(ty) +
                                  " blocks do not split";
                            return false;
                        }
                        x.hlay.suh[r] = x.hlay.svh[r] = off;
                        off += align256((size_t) n * gg::type_row_bytes(ty, k));
                        continue;
                    }
                    off += align256(x3::trellis_bytes(k, n, l.lay.K[r]));
                    x.hlay.suh[r] = off;
                    off += align256((size_t) k * 2);
                    x.hlay.svh[r] = off;
                    off += align256((size_t) n * 2);
                }
                x.hlay.cb = l.lay.cb;
                x.hlay.fmt = l.lay.fmt;
                x.hblob = off;
            }
        }
        // which devices hold each source's compressed KV / index keys
        for (Src& s : srcs) {
            s.ckv.assign(devs.size(), nullptr);
            s.ik.assign(devs.size(), nullptr);
        }
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                const Layer& l = member(l0, m);
                if (l.src >= 0) {
                    srcs[(size_t) l.src].ckv[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.src].layer].grp) srcs[(size_t) l.src].remote = true;
                }
                if (l.ix_src && l.ik_src >= 0) {
                    srcs[(size_t) l.ik_src].ik[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.ik_src].layer].grp) srcs[(size_t) l.ik_src].remote = true;
                }
            }

        // per device: the scratch, the caches, the dense weights
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            const int D = g.dim, H = g.hc;
            const size_t W = kWin, P = (size_t) kWin * kPicks, cap = (size_t) o.max_context + 64;
            bool ok = dalloc(d, d.h, W * H * D, err) && dalloc(d, d.h2, W * H * D, err) && dalloc(d, d.x, W * D, err) &&
                      dalloc(d, d.qr, W * g.q_lora, err) && dalloc(d, d.q, W * g.heads * g.hd, err) &&
                      dalloc(d, d.kvr, W * g.hd, err) && dalloc(d, d.ckr, W * g.hd, err) &&
                      dalloc(d, d.cgr, W * g.hd, err) && dalloc(d, d.latent, W * g.hd, err) &&
                      dalloc(d, d.ikr, W * g.ix_hd, err) && dalloc(d, d.iq, W * g.ix_heads * g.ix_hd, err) &&
                      dalloc(d, d.iw, W * 64, err) && dalloc(d, d.scores, W * cap, err) &&
                      dalloc(d, d.bm, W * bm_stride(), err) && dalloc(d, d.o, W * g.heads * g.hd, err) &&
                      dalloc(d, d.oa, W * g.o_groups * g.o_lora, err) && dalloc(d, d.ao, W * D, err) &&
                      dalloc(d, d.mix, W * 32, err) && dalloc(d, d.a_pre, W * 8, err) && dalloc(d, d.a_post, W * 8, err) &&
                      dalloc(d, d.a_comb, W * 64, err) && dalloc(d, d.f_pre, W * 8, err) &&
                      dalloc(d, d.f_post, W * 8, err) && dalloc(d, d.f_comb, W * 64, err) && dalloc(d, d.pre, W * 8, err) &&
                      dalloc(d, d.glog, W * 1024, err) && dalloc(d, d.glog_a, 2 * W * 1024, err) &&
                      dalloc(d, d.rw, W * kPicks, err) &&
                      dalloc(d, d.ridx, W * kPicks, err) && dalloc(d, d.moe_out, W * D, err) &&
                      dalloc(d, d.sg, W * g.ff, err) && dalloc(d, d.su, W * g.ff, err) && dalloc(d, d.sh, W * g.ff, err) &&
                      dalloc(d, d.sh_out, W * D, err) && dalloc(d, d.cpu_part, W * D, err) &&
                      dalloc(d, d.freqs_win, 64, err) && dalloc(d, d.freqs_comp, 64, err) &&
                      dalloc(d, d.topk, W * topk_stride(), err) && dalloc(d, d.cand, W * cand_stride(), err) &&
                      dalloc(d, d.tok, 16, err) && dalloc(d, d.emb_rows, W * D, err) &&
                      dalloc(d, d.ext, ((size_t) g.win + W) * g.hd, err) && dalloc(d, d.xe, P * D, err) &&
                      dalloc(d, d.xg, P * g.ff, err) && dalloc(d, d.xu, P * g.ff, err) && dalloc(d, d.xhh, P * g.ff, err) &&
                      dalloc(d, d.ed, P * D, err) && dalloc(d, d.plan_ptr, P, err);
            if (!ok) return false;
            d.attn_scr.max_t = kAttnScratchT;
            d.attn_scr.cap = (int) topk_stride();
            if (!dalloc(d, d.attn_scr.rows, (size_t) kAttnScratchT * topk_stride() * g.hd, err)) return false;
            if (!g.eg_layers.empty()) {
                const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
                ok = dalloc(d, d.eg_rows, W * cols * (pk.eg_hd + pk.eg_hd / 32), err) &&
                     dalloc(d, d.eg_x, W * cols * pk.eg_hd, err) && dalloc(d, d.eg_kv, W * (g.hc + 1) * D, err);
                if (!ok) return false;
            }
            if ((int) di == head_dev && !o.stage_worker && !dalloc(d, d.logits, (size_t) kWin * g.vocab, err)) return false;
            if (!halloc(d.bounce, bounce_bytes(), err)) return false;
            if (!dalloc(d, d.xcnt, 4, err) || !hmap(d.hand, bounce_bytes(), err) || !hmap(d.hand_flag, 32, err) ||
                !dalloc(d, d.grp_ptr, P, err) || !dalloc(d, d.grp_start, P + 1, err) || !dalloc(d, d.grp_ent, P, err) ||
                !dalloc(d, d.n_groups, 4, err))
                return false;
            if ((g.n_mtp > 0 || o.stage_worker) && !dalloc(d, d.tap, (size_t) tap_cap * D, err)) return false;
            if (g.n_mtp > 0 && (int) di == head_dev &&
                (!dalloc(d, d_confv, kWin, err) || !dalloc(d, d_ints, 64, err) || !dalloc(d, d_specq, kWin, err)))
                return false;
            CK(cudaEventCreateWithFlags(&d.ev_out, cudaEventDisableTiming));
            cudaSetDevice(d.id);
            CK(cudaMemcpy(d.freqs_win, freqs_win.data(), freqs_win.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d.freqs_comp, freqs_comp.data(), freqs_comp.size() * 4, cudaMemcpyHostToDevice));
            for (Src& s : srcs) {
                if (s.h_ckv == nullptr) {   // (its events belong to the source layer's device: it records them)
                    if (!halloc(s.h_ckv, (size_t) kWin * ckv_rb(), err) || !halloc(s.h_ik, (size_t) kWin * ik_rb(), err))
                        return false;
                    CK(cudaSetDevice(devs[(size_t) L[(size_t) s.layer].dev].id));
                    CK(cudaEventCreateWithFlags(&s.ev_ckv, cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&s.ev_ik, cudaEventDisableTiming));
                    CK(cudaSetDevice(d.id));
                }
                if (s.ckv[di] != nullptr && !dalloc(d, s.ckv[di], (size_t) s.cap * ckv_rb(), err, &d.caches))
                    return false;
                if (s.ik[di] != nullptr && !dalloc(d, s.ik[di], (size_t) s.cap * ik_rb(), err, &d.caches))
                    return false;
            }
        }
        for (Src& sr : srcs) {
            if (sr.remote && (!hmap(sr.x_stage, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag, 32, err)))
                return false;
            if (sr.remote && chunk > 0 &&
                (!hmap(sr.x_stage1, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag1, 32, err) ||
                 !hmap(sr.x_ack, (size_t) 2 * devs.size() * 32, err)))
                return false;
            const int sg = L[(size_t) sr.layer].grp;
            for (size_t di = 0; di < devs.size(); ++di)
                if ((sr.ckv[di] != nullptr || sr.ik[di] != nullptr) && devs[di].grp != sg) sr.readers.push_back((int) di);
        }
        for (Group& G : groups) {
            G.ar.nm = (int) G.mem.size();
            G.ar.slots = 4;
            G.ar.stride = (size_t) kWin * g.dim;
            if (G.ar.nm > 1 && (!hmap(G.ar.data, (size_t) G.ar.nm * G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.flags, (size_t) G.ar.nm * G.ar.slots * 32, err)))
                return false;
            // three members or more: the reduce-scatter's sums and its per-block flags (allreduce)
            if (G.ar.nm > 2 && (!hmap(G.ar.sums, (size_t) G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.bflags, (size_t) 2 * G.ar.nm * G.ar.slots * kArBlocks * 8, err)))
                return false;
            if (G.ar.nm > 1 && &G == &groups[(size_t) devs[(size_t) head_dev].grp] &&
                (!hmap(G.lg_stage, (size_t) 2 * (G.ar.nm - 1) * kWin * ((g.vocab / 128 + G.ar.nm - 1) / G.ar.nm * 128), err) ||
                 !hmap(G.lg_flag, (size_t) 2 * (G.ar.nm - 1) * 32, err) || !hmap(G.lg_ack, (size_t) 2 * 32, err)))
                return false;
            G.arb = G.ar;   // (a chunk's: two slots of chunk rows)
            G.arb.slots = 2;
            G.arb.stride = (size_t) chunk * g.dim;
            if (chunk > 0 && G.arb.nm > 1 && (!hmap(G.arb.data, (size_t) G.arb.nm * G.arb.slots * G.arb.stride, err) ||
                                              !hmap(G.arb.flags, (size_t) G.arb.nm * G.arb.slots * 32, err)))
                return false;
            if (chunk > 0 && G.arb.nm > 2 &&
                (!hmap(G.arb.sums, (size_t) G.arb.slots * G.arb.stride, err) ||
                 !hmap(G.arb.bflags, (size_t) 2 * G.arb.nm * G.arb.slots * kArBlocks * 8, err)))
                return false;
        }
        if (g.n_mtp > 0) {
            if (!halloc(h_taps, (size_t) tap_cap * g.d_targets.size() * g.dim, err)) return false;
            CK(cudaSetDevice(devs[(size_t) d_dev].id));
            CK(cudaEventCreateWithFlags(&ev_seed, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_seed, devs[(size_t) d_dev].st));
        }
        // the dense weights, each device's layers by a thread of its own
        {
            std::vector<std::string> errs(devs.size());
            std::vector<std::thread> th;
            for (size_t di = 0; di < devs.size(); ++di)
                th.emplace_back([this, di, &errs] {
                    std::string e;
                    if (!load_dense((int) di, e)) errs[di] = e;
                });
            for (auto& t : th) t.join();
            for (auto& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
        }
        if (o.verbose)
            std::fprintf(stderr, "ds4: dense weights loaded in %.1f s\n", (now_ms() - t0) / 1000.0);
        // the markov steps over a head group of two or more (DS4_MK_SPLIT=0: the lead's whole rows, as before): E
        // host-mapped, the steps' slots
        // (only where every member holds its rows of the markov head: a type load_dense does not take keeps the lead's)
        bool mk_rows = g.n_mtp > 0 && head_nm() > 1;
        for (int m : groups[(size_t) devs[(size_t) head_dev].grp].mem) mk_rows = mk_rows && devs[(size_t) m].mk_share != nullptr;
        if (mk_rows && !(std::getenv("DS4_MK_SPLIT") != nullptr && std::atoi(std::getenv("DS4_MK_SPLIT")) == 0)) {
            const Tensor* te = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.embed.weight");
            if (te == nullptr || te->dtype != "BF16" || te->bytes != (uint64_t) g.vocab * g.d_rank * 2) {
                err = "the markov embed: not a BF16 [vocab][rank] table";
                return false;
            }
            if (!hmap(h_mk_embed, (size_t) g.vocab * g.d_rank, err) || !pk.read(*te, h_mk_embed) ||
                !hmap(mk_slots, (size_t) kWin * head_nm() * 4, err)) {
                if (err.empty()) err = "the markov embed: read failed";
                return false;
            }
            devs[(size_t) head_dev].mk_ints = d_ints;   // (the lead's drafted tokens: where the confidence head reads them)
            mk_split = true;
        }

        // a remote stage's rows, both ways
        if (linked() && (!halloc(link_tx, (size_t) kWin * full_floats(), err) ||
                         !halloc(link_rx, (size_t) kWin * full_floats(), err)))
            return false;
        if (bf16_rows() && (!halloc(wire_tx, (size_t) kWin * handoff_floats(), err) ||
                            !halloc(wire_rx, (size_t) kWin * handoff_floats(), err)))
            return false;
        // the embedding table, pinned on the host (a remote stage's worker: none - its rows come over the link)
        if (!o.stage_worker) {
            const Tensor* t = pk.get("embed.weight");
            const size_t nv = (size_t) g.vocab * g.dim;
            const bool q8 = t != nullptr && t->dtype == "Q8_0" && t->bytes == nv / 32 * gg::kQ8Bytes;
            // other block types (a GGUF whose dense tensors are Q4_K / Q5_K / Q6_K: the IST-DAS sets' Q5_K embed.weight):
            // each row through ggml's to_float, to the same BF16 the host table holds
            const int ety = t != nullptr ? gg_type_of(t->dtype) : -1;
            const size_t erb = ety > 0 ? gg::type_row_bytes(ety, g.dim) : 0;
            const bool qrow = t != nullptr && !(t->dtype == "BF16") && erb > 0 && t->bytes == (size_t) g.vocab * erb;
            if (t == nullptr || (!q8 && !qrow && (t->dtype != "BF16" || t->bytes != nv * 2))) {
                err = "embed.weight: not a BF16 (or Q8_0 / a ggml block type) [vocab][dim] table";
                return false;
            }
            if (!halloc(embed_pinned, nv, err)) return false;
            if (qrow) {   // (ggml's decode of each row, rounded to BF16 as the Q8_0 path rounds)
                std::vector<uint8_t> row(erb);
                for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                    if (!pk.read_at(t->file, t->off + r * erb, erb, row.data())) {
                        err = "embed.weight: read failed";
                        return false;
                    }
                    if (!gg::gg_row_to_bf16(ety, row.data(), g.dim, embed_pinned + r * g.dim)) {
                        err = "embed.weight: ggml has no decode for " + t->dtype;
                        return false;
                    }
                }
            } else if (q8) {   // (a GGUF's: each value rounded to BF16, a row at a time)
                std::vector<uint8_t> row((size_t) g.dim / 32 * gg::kQ8Bytes);
                for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                    if (!pk.read_at(t->file, t->off + r * row.size(), row.size(), row.data())) {
                        err = "embed.weight: read failed";
                        return false;
                    }
                    const std::vector<float> v = q8_values(row.data(), (size_t) g.dim);
                    for (size_t c = 0; c < (size_t) g.dim; ++c) {
                        uint32_t u;
                        std::memcpy(&u, &v[c], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
                        embed_pinned[r * g.dim + c] = (uint16_t) (u >> 16);
                    }
                }
            } else if (!pk.read(*t, embed_pinned)) {
                err = "embed.weight: read failed";
                return false;
            }
        }
        // the host side of the CPU lane and the Engram rows
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
        if (!halloc(h_x, (size_t) kWin * g.dim, err) || !halloc(h_w, (size_t) kWin * kPicks, err) ||
            !halloc(h_part, (size_t) kWin * g.dim, err) || !halloc(h_idx, (size_t) kWin * kPicks, err) ||
            !halloc(h_emb, (size_t) std::max(kWin, chunk) * g.dim, err) ||
            !halloc(h_eg, (size_t) std::max(1, pk.eg_layers) * eg_cap * cols * (pk.eg_hd + pk.eg_hd / 32) + 64, err))
            return false;
        if (pk.eg_layers > 0 && (!ranged || eg_here())) {
            // the Engram rows' reader: threads enough to keep an SSD's queue busy (reads from the page cache need few);
            // the row cache a small share of the RAM free (DS4_ENGRAM_CACHE_MB: its size, 0 off; DS4_ENGRAM_THREADS)
            const size_t row = (size_t) pk.eg_hd + pk.eg_hd / 32;
#if defined(_WIN32)
            // (Windows: each thread's ReadFile is synchronous, so the threads are the queue depth - Project Maya: a 28K-token
            //  prompt in one chunk, 10 threads 17.1 s, 64 13.4 s on a 20-core PC.  Here (6 cores, shipped profile) 64 readers
            //  gave +4.5% on a 4K prompt but no decode gain in every pair: DS4_ENGRAM_THREADS=64 turns them on)
#endif
            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            int nth = (int) std::min(16u, std::max(4u, hw / 2));
            if (const char* e = std::getenv("DS4_ENGRAM_THREADS")) nth = std::max(1, std::atoi(e));
            const uint64_t avail = ram_avail();
            uint64_t mb = avail == ~uint64_t{0} ? 64 : std::min<uint64_t>(256, std::max<uint64_t>(16, (avail >> 20) / 64));
            if (const char* e = std::getenv("DS4_ENGRAM_CACHE_MB")) mb = std::strtoull(e, nullptr, 10);
            eg_cache.init((size_t) ((mb << 20) / row), row);
            std::vector<std::string> paths;
            for (const std::string& fn : pk.files) paths.push_back(pk.model_dir + "/" + fn);
            // Windows: the Engram tables' files that hold nothing else are unmapped and read unbuffered - an SSD's
            // queue kept by the reader threads instead of the page cache's (whose reads of one file from many threads
            // went ~255K rows a second here); unbuffered reads of a mapped file take a slow path, so only unmapped
            // ones (off by default: -8% on a 4K prompt here; DS4_EG_DIRECT=1 turns it on)
            std::vector<char> eg_direct(pk.files.size(), 0);
#if defined(_WIN32)
            if (pk.gguf && std::getenv("DS4_EG_DIRECT") != nullptr && std::atoi(std::getenv("DS4_EG_DIRECT")) != 0) {
                std::vector<char> eg_file(pk.files.size(), 0), other(pk.files.size(), 0);
                for (const auto& kt : pk.t)
                    if (kt.second.file >= 0 && (size_t) kt.second.file < pk.files.size())
                        (kt.first.find(".engram.embed.") != std::string::npos ? eg_file : other)[(size_t) kt.second.file] = 1;
                for (size_t i = 0; i < pk.files.size(); ++i)
                    if (eg_file[i] && !other[i]) {
                        if (i < pk.maps.size() && pk.maps[i].first != nullptr) {
                            os::unmap(pk.maps[i].first, pk.maps[i].second);
                            pk.maps[i] = {nullptr, 0};
                        }
                        eg_direct[i] = 1;
                        if (o.verbose) std::fprintf(stderr, "ds4: %s unmapped, its Engram rows read unbuffered\n",
                                                    pk.files[i].c_str());
                    }
            }
#endif
            eg_io.start(paths, nth, eg_direct);
            for (int li = 0; li < pk.eg_layers; ++li) {
                eg_win.emplace_back(new EgBatch());
                const std::string id = std::to_string(g.eg_layers[(size_t) li]);
                const Tensor* tw = pk.get("layers." + id + ".engram.embed.weight");
                // The row layout the reader and the dequant take.  A type not here (or no tensor at all) would be read
                // as some other format's bytes - the old code then read the scale tensor of a pack that has none and
                // crashed in eg_issue - so refuse the pack once, with the type's name, before any token is read.
                int fmt;
                if (tw == nullptr) {
                    std::fprintf(stderr, "ds4: Engram layer %s: engram.embed.weight is missing from the pack\n", id.c_str());
                    std::exit(2);
                } else if (tw->dtype == "MXFP8") {
                    fmt = 0;
                } else if (tw->dtype == "Q6_K") {
                    fmt = 1;
                } else if (tw->dtype == "IQ4_NL") {
                    fmt = 2;
                } else if (pk.get("layers." + id + ".engram.embed.scale") != nullptr) {
                    fmt = 3;   // (an old pack: FP8 codes in one tensor, their E8M0 scales in another)
                } else {
                    std::fprintf(stderr,
                                 "ds4: Engram layer %s: engram.embed.weight type %s is not one the Engram reader handles "
                                 "(MXFP8, Q6_K, IQ4_NL)\n",
                                 id.c_str(), tw->dtype.c_str());
                    std::exit(2);
                }
                if (fmt == 1 && pk.eg_hd % 256 != 0) {
                    err = "Engram layer " + id + ": Q6_K rows of " + std::to_string(pk.eg_hd) +
                          " values (not whole superblocks of 256)";
                    return false;
                }
                if (fmt == 2 && pk.eg_hd % 32 != 0) {
                    err = "Engram layer " + id + ": IQ4_NL rows of " + std::to_string(pk.eg_hd) +
                          " values (not whole blocks of 32)";
                    return false;
                }
                eg_fmt.push_back((uint8_t) fmt);
            }
            if (o.verbose)
                std::fprintf(stderr, "ds4: Engram rows read on %d threads, issued with the tokens, collected before "
                                     "their layer; row cache %llu MB (%zu rows)\n",
                             nth, (unsigned long long) mb, eg_cache.capacity());
        }
        prof.assign((size_t) (g.nl + g.n_mtp) * g.n_exp, 0u);
        if (!o.profile.empty()) {
            std::FILE* f = std::fopen(o.profile.c_str(), "rb");
            if (f != nullptr) {
                std::vector<uint32_t> p(prof.size());
                const size_t n = std::fread(p.data(), 4, p.size(), f);
                if (n == p.size() || n == (size_t) g.nl * g.n_exp) std::copy(p.begin(), p.begin() + (long) n, prof.begin());
                std::fclose(f);
            }
        }

        // the chunked prompt path's buffers (before the experts: they take what is left)
        if (chunk > 0) {
            // recommend, never force: a chunk the cards cannot hold in buffers is halved (down to 256 rows) and then the
            // prompt runs in windows of 16, said in one line - the load only fails for a reason that is not this
            std::vector<size_t> na0, sc0;
            for (Dev& d : devs) {
                na0.push_back(d.allocs.size());
                sc0.push_back(d.scratch);
            }
            const size_t nh = host_allocs.size();
            std::string aerr;
            while (chunk > 0 && !alloc_chunks(aerr)) {
                for (size_t di = 0; di < devs.size(); ++di) {
                    Dev& d = devs[di];
                    cudaSetDevice(d.id);
                    cudaDeviceSynchronize();
                    if (di < chunks.size() && chunks[di]) {
                        if (chunks[di]->blas) cublasDestroy(chunks[di]->blas);
                        if (chunks[di]->cs) cudaStreamDestroy(chunks[di]->cs);
                    }
                    while (d.allocs.size() > na0[di]) {
                        cudaFree(d.allocs.back());
                        d.allocs.pop_back();
                    }
                    d.scratch = sc0[di];
                }
                while (host_allocs.size() > nh) {
                    cudaFreeHost(host_allocs.back());
                    host_allocs.pop_back();
                }
                chunks.clear();
                const int was = chunk;
                chunk = chunk > 512 ? chunk / 2 / 16 * 16 : 0;
                if (chunk > 0) {
                    eg_cap = (size_t) chunk;
                    std::fprintf(stderr, "ds4: the prompt path's buffers for chunks of %d rows did not fit (%s): trying %d\n",
                                 was, aerr.c_str(), chunk);
                } else {
                    prompt_par = false;
                    eg_cap = (size_t) kWin;
                    tap_cap = kWin;
                    std::fprintf(stderr, "ds4: the prompt path's buffers did not fit (%s): prompts run in windows of %d "
                                         "tokens (slower on long prompts; more free GPU memory or a lower --reserve-mib "
                                         "brings the chunks back)\n", aerr.c_str(), kWin);
                }
                aerr.clear();
            }
            if (chunk > 0 && aerr.empty()) err.clear();
            static const char* dq_once_env = std::getenv("DS4_DQ_ONCE");
            static const bool dq_once = dq_once_env != nullptr && std::atoi(dq_once_env) == 1;
            if (dq_once_env != nullptr && !dq_once)   // (set to anything else: said - a mistyped arm is not silently off)
                std::fprintf(stderr, "ds4: DS4_DQ_ONCE=%s: off - 1 turns it on\n", dq_once_env);
            if (chunk > 0 && dq_once) alloc_dq_once();
        }
        // a remote stage's prompt chunks: the main process's two slots of rows out, the worker's four the link lends
        // (and with bf16 rows its two fp32 ones) - pageable (a chunk's rows cross the network at its pace; on Linux a
        // page is committed once a chunk reaches it, on Windows at the allocation).  Memory short: chunks half as
        // large, as the prompt path's buffers above
        if (linked() && chunk > 0) {
            float** const bufs[6] = {o.remote_stage ? &link_cx[0] : &link_pf[0], o.remote_stage ? &link_cx[1] : &link_pf[1],
                                     &link_pf[2], &link_pf[3], &link_st[0], &link_st[1]};
            const int nb = o.remote_stage ? 2 : bf16_rows() ? 6 : 4;
            // (a row's floats: the main process's slots and the worker's fp32 ones full_floats, the link's handoff_floats)
            auto row_of = [&](int i) { return o.stage_worker && i < 4 ? handoff_floats() : full_floats(); };
            for (;;) {
                size_t all = 0;
                int got = 0;
                for (; got < nb; ++got) {
                    const size_t b = (size_t) chunk * (size_t) row_of(got) * 4;
                    void* m = os::anon_alloc(b, -1, 1);
                    if (m == nullptr) break;
                    link_maps.push_back({m, b});
                    *bufs[got] = (float*) m;
                    all += b;
                }
                if (got == nb) break;
                for (int i = 0; i < nb; ++i) all += i >= got ? (size_t) chunk * (size_t) row_of(i) * 4 : 0;
                for (; got > 0; --got) {
                    os::anon_free(link_maps.back().first, link_maps.back().second);
                    link_maps.pop_back();
                    *bufs[got - 1] = nullptr;
                }
                if (chunk <= 256) {
                    err = "remote stage: " + std::to_string(all >> 20) + " MiB of memory for prompt chunks' rows "
                          "(--prefill-chunk 0 for windows)";
                    return false;
                }
                const int was = chunk;
                chunk = chunk / 2 / 16 * 16;
                eg_cap = (size_t) chunk;
                std::fprintf(stderr, "ds4: remote stage: %zu MiB of memory for chunks of %d rows' hand-off rows: not "
                                     "had - trying %d\n", all >> 20, was, chunk);
            }
        }
        // the CPU lane's experts read in place (decided above): how much of them the load reads ahead, from what VRAM
        // now leaves over
        if (file_experts) {
            double room = 0, all = 0;   // the VRAM place_experts will find for experts; every expert's bytes
            for (Dev& d : devs) {
                cudaSetDevice(d.id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                room += std::max(0.0, (double) fr - (double) ((size_t) o.reserve_mib << 20)) * o.vram_experts;
            }
            for (const Layer& l : L) all += (double) l.blob * l.n_exp;
            const double need = std::max(0.0, all - room);
            const uint64_t avail = ram_avail();
            warm_bytes = avail > kRamMargin ? std::min<uint64_t>(avail - kRamMargin, (uint64_t) need) : 0;
            if (avail == ~uint64_t{0}) warm_bytes = (uint64_t) need;
            if (store_gb() > 0) warm_bytes = 0;   // (the store is the RAM tier: nothing to warm in the OS cache)
            std::fprintf(stderr,
                         "ds4: the experts VRAM leaves over (~%.1f GB) are read in place from the model's files%s "
                         "(%.1f GB of RAM free to keep them: the most-routed %.1f GB read ahead; prompts in chunks of %d)\n",
                         need / 1e9, o.mmap_experts > 0 ? " (--mmap-experts on)"
                                     : pk.gguf ? " (--mmap-experts off copies them into RAM)" : ", not copied into RAM",
                         avail == ~uint64_t{0} ? 0.0 : avail / 1e9, warm_bytes / 1e9, chunk);
        }
        // the CPU lane: one pool of workers per NUMA node, each expert split in NH column parts (part X on node X); in
        // place, one part (the files' layout) and one pool over every node's cores (the pages lie where the cache put them)
        cores = node_cores();
        if (!o.remote_experts.empty()) {   // (before the pools: they leave the link core out)
            if ((ep_cpu = rx::link_cpu()) >= 0) ep_sibs = os::cpu_siblings(ep_cpu);
            const char* wv = std::getenv("DS4_EP_WAKE");
            ep_wake = wv == nullptr || std::atoi(wv) != 0;
        }
        NH = file_experts ? 1 : (int) std::min<size_t>(2, cores.size());
        if (NH > 1 && ((g.ff / NH) % 128 != 0 || (g.dim / NH) % 128 != 0)) NH = 1;
        if (!pk.gguf) make_pools();   // (a GGUF's after its experts: copy_experts decides NH)
        {
            const size_t E = (size_t) g.dim, FF = (size_t) g.ff, GQ = E / 256, DQ = FF / 128, S = kSlots;
            c_xh.resize((size_t) NH * S * 2 * E);
            c_xsb.resize((size_t) NH * S * 2 * (E / 128));
            c_acc.resize((size_t) NH * S * 2 * GQ * (FF / NH));
            c_xhd.resize(S * FF);
            c_xsd.resize(S * (FF / 128));
            c_accd.resize((size_t) NH * S * DQ * (E / NH));
            c_yd.resize(S * E);
        }
        // the experts: VRAM slots in what each device has left, the rest on the host
        if (iq_warm_env != nullptr && *iq_warm_env != '\0' && !iq_warm_split) {
            err = std::string("DS4_IQ_WARM=") + iq_warm_env + ": split or unset";
            return false;
        }
        if (!iq_warm_touch && iq_warm_touch_env != nullptr && *iq_warm_touch_env != '\0' && std::strcmp(iq_warm_touch_env, "0") != 0) {
            err = std::string("DS4_IQ_WARM_TOUCH=") + iq_warm_touch_env + ": 0, 1 or unset";
            return false;
        }
        if (!iq_warm_cold_first && iq_warm_order_env != nullptr && *iq_warm_order_env != '\0' && std::strcmp(iq_warm_order_env, "hot") != 0) {
            err = std::string("DS4_IQ_WARM_ORDER=") + iq_warm_order_env + ": cold, hot or unset";
            return false;
        }
        if (!iq_warm_srv && iq_warm_srv_env != nullptr && *iq_warm_srv_env != '\0' && std::strcmp(iq_warm_srv_env, "0") != 0) {
            err = std::string("DS4_IQ_WARM_SRV=") + iq_warm_srv_env + ": 0, 1 or unset";
            return false;
        }
        if (!place_experts(err)) return false;
        if (!load_experts(err)) return false;
        if (pk.gguf) make_pools();
        if (chunk > 0) {
            // the CPU lane's parts pinned (the chunks stream them to the GPUs by DMA), each layer's virtual slots
            const double tr = now_ms();
            for (auto& m : host_maps)
                if (cudaHostRegister(m.first, m.second, cudaHostRegisterPortable) != cudaSuccess) {
                    err = "pinning the CPU lane's experts (" + std::to_string(m.second >> 20) + " MiB) failed";
                    return false;
                }
            if (o.verbose) std::fprintf(stderr, "ds4: CPU lane's experts pinned in %.1f s\n", (now_ms() - tr) / 1000.0);
            if (file_experts) {   // in place: each device's staging for its ring, and its readers (DS4_READERS, 8)
                size_t mr = 0;
                for (const Layer& l : L)
                    if (!l.mtp) {
                        for (uint64_t r : l.region_bytes) mr = std::max<size_t>(mr, r);
                        if (l.lay.fmt == gg::kFmtMXFP4) {   // (a GGUF's: read as its blob)
                            mr = std::max(mr, l.blob);
                            for (int r = 0; r < 3; ++r) flow_part[r] = std::max(flow_part[r], mx_part_bytes(l, r));
                        }
                    }
                const char* nr = std::getenv("DS4_READERS");
                const int n_readers = std::max(1, nr != nullptr ? std::atoi(nr) : 8);
                bc_once = bc_once_mode(std::getenv("DS4_BC_ONCE"));
                if (bc_once < 0) {
                    err = "DS4_BC_ONCE: 1 (a layer's batch bounds in one copy) or 2 (read from mapped host memory); "
                          "unset, empty or 0: off";
                    return false;
                }
                const bool flow_on = std::getenv("DS4_FLOW") == nullptr || std::atoi(std::getenv("DS4_FLOW")) != 0;
                // (a tensor-parallel group's members on the flow only with DS4_FLOW_TP=1 for now: each reads the
                // whole down of every expert it streams, 25 MB an expert for a pair, 18.8 alone - 3x V100, the
                // 17-token prompt from the page cache: 2.5-2.7 s on the flow, 1.3-1.6 s in the group's windows)
                const bool flow_tp = std::getenv("DS4_FLOW_TP") != nullptr && std::atoi(std::getenv("DS4_FLOW_TP")) != 0;
                for (size_t di = 0; di < chunks.size(); ++di) {
                    auto& cp = chunks[di];
                    if (cp == nullptr || cp->ring == nullptr) continue;
                    cp->stage_stride = (mr + 4095) & ~(size_t) 4095;
                    // the flow: every trunk layer of the device a GGUF's (alone or a tensor-parallel group's member)
                    bool fl = flow_on;
                    int n_exp_max = 0;
                    for (const Layer& l : L)
                        if (!l.mtp)
                            for (int m = 0; m < l.nm; ++m)
                                if (member(l, m).dev == (int) di) {
                                    // (a group's members: each its share of the down's blocks in every row)
                                    fl = fl && l.lay.fmt == gg::kFmtMXFP4 &&
                                         (l.nm == 1 || (flow_tp && (g.ff / l.nm) %
                                                        std::max(1, gg::type_block(gg::expert_type(l.lay, 2))) == 0));
                                    n_exp_max = std::max(n_exp_max, l.n_exp);
                                }
                    if (fl)   // (each part read unbuffered from its offset rounded down: a page more each)
                        cp->stage_stride = std::max(cp->stage_stride, flow_stride());
                    // DS4_STAGE_N=N (unset: as ring_n slots): a flow device's staging has N slots when that is more than
                    // its ring's - the reads then run further ahead of the compute (ds4_stage_slots.hpp); pinned memory
                    // that does not come falls back to ring_n slots
                    cp->S = stage_slots(std::getenv("DS4_STAGE_N"), cp->ring_n, fl);
                    if (!halloc(cp->stage, (size_t) cp->S * cp->stage_stride, err)) {
                        if (cp->S == cp->ring_n) return false;
                        (void) cudaGetLastError();
                        std::fprintf(stderr, "ds4: DS4_STAGE_N=%d: %s - the flow's staging keeps its %d slots\n", cp->S,
                                     err.c_str(), cp->ring_n);
                        cp->S = cp->ring_n;
                        err.clear();
                        if (!halloc(cp->stage, (size_t) cp->S * cp->stage_stride, err)) return false;
                    }
                    if (cp->S > cp->ring_n && o.verbose)
                        std::fprintf(stderr, "ds4: CUDA%d flow staging: %d slots of %.1f MB, %.2f GB pinned (DS4_STAGE_N; the ring: %d slots)\n",
                                     devs[di].id, cp->S, (double) cp->stage_stride / 1e6,
                                     (double) cp->S * (double) cp->stage_stride / 1e9, cp->ring_n);
                    cp->readers = std::make_unique<Pool>(n_readers - 1, std::vector<int>{}, 50);
                    if (!fl) continue;
                    cudaSetDevice(devs[di].id);
                    cp->flow = true;
                    cp->dev = (int) di;
                    // DS4_IO_SPLIT: a read whose resident lead is not all of it - a gate / up the lane's warm read
                    // the first blocks of - takes that lead from the page cache, the rest from the disk unbuffered
                    // (not the whole from the disk); on by default (pf-run, a 4K prompt: 23.3-25.2 -> 20.4-22.0 s with
                    // the down parts, 21.1-21.2 -> 20.7-20.8 without), unset or empty: on, 0: every read whole as
                    // before; DS4_IO_SPLIT_CHECK=1 reads each such one again and compares
                    const char* sp = std::getenv("DS4_IO_SPLIT");
                    const char* spk = std::getenv("DS4_IO_SPLIT_CHECK");
                    cp->io = std::make_unique<IoQueue>(n_readers, sp == nullptr || *sp == '\0' || std::atoi(sp) != 0,
                                                       spk != nullptr && std::atoi(spk) != 0);
                    cp->stage_item.assign((size_t) cp->S, -1);
                    cp->stage_ev.resize((size_t) cp->S);
                    cp->ring_fill.resize((size_t) cp->ring_n);
                    cp->ring_free.resize((size_t) cp->ring_n);
                    for (auto* v : {&cp->stage_ev, &cp->ring_fill, &cp->ring_free})
                        for (cudaEvent_t& e : *v) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                    // (a region is written again only bc_n batches later - more than a layer's items, so after that
                    // layer's bounds wait, which every copy from it precedes.  DS4_BC_ONCE: a layer's regions written
                    // up front, each batch its own (bc_layer_fits), once the layers before are done with theirs on
                    // the compute stream - bc_done; 2: bc_pin mapped, the kernels read it there)
                    cp->bc_n = n_exp_max + 2;
                    cp->bc_w = cp->ring_n + 1;
                    if (!(bc_once == 2 ? hmap(cp->bc_pin, (size_t) cp->bc_n * cp->bc_w, err)
                                       : halloc(cp->bc_pin, (size_t) cp->bc_n * cp->bc_w, err)) ||
                        !dalloc(devs[di], cp->bc_dev, (size_t) cp->bc_n * cp->bc_w, err))
                        return false;
                    if (bc_once == 2) {
                        void* m = nullptr;
                        if (cudaHostGetDevicePointer(&m, cp->bc_pin, 0) != cudaSuccess) {
                            err = "DS4_BC_ONCE=2: no device address for the mapped batch bounds";
                            return false;
                        }
                        cp->bc_map = (int*) m;
                    }
                    if (bc_once != 0) {
                        CK(cudaEventCreateWithFlags(&cp->bc_done, cudaEventDisableTiming));
                        CK(cudaEventCreateWithFlags(&cp->vs_ev, cudaEventDisableTiming));
                    }
                }
                // every device with layers on the flow: a prompt reads only the experts it picks, once, so a short one
                // takes the chunked path too (one pass, where windows of 16 read the CPU lane's picks again for each
                // window, a page fault at a time) - DS4_CHUNK_MIN still overrides
                bool all_flow = true;
                for (const auto& cp : chunks) all_flow = all_flow && (cp == nullptr || cp->ring == nullptr || cp->flow);
                if (all_flow && std::getenv("DS4_CHUNK_MIN") == nullptr) chunk_min = kFlowChunkMin;
            }
            for (Layer& l : L) {
                if (l.mtp || elsewhere(l)) continue;   // (a layer range: its own layers')
                for (int m = 0; m < l.nm; ++m) {   // (each member its own: the flow's members decide them each)
                    Layer& x = member(l, m);
                    if (!halloc(x.vs_pin, (size_t) l.n_exp, err)) return false;
                    if (!dalloc(devs[(size_t) x.dev], x.vs_dev, (size_t) l.n_exp, err)) return false;
                }
            }
        }
        if (!rx_slices_ok(err)) return false;
        rx_setup();   // (--remote-experts: the expert server's blobs, before the service thread takes the link)
        if (!rx_on && ep_cpu >= 0) {   // (no server after all: the link core and its sibling back to the CPU lane - the
                                       // pools are idle until the service thread starts, below)
            std::fprintf(stderr, "ds4: remote experts: none held - the link core (CPU %d) goes back to the CPU lane\n", ep_cpu);
            ep_cpu = -1;
            ep_freed = true;
            ep_sibs.clear();
            pools.clear();
            make_pools();
        }
        if (!defer_setup(err)) return false;
        small_setup();   // (after defer_setup: its line says whether the deferral rides along)
        {
            void* m = nullptr;
            if (cudaHostAlloc(&m, sizeof(Req) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's requests";
                return false;
            }
            host_allocs.push_back(m);
            reqs = (Req*) m;
            if (cudaHostAlloc(&m, sizeof(Resp) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's answers";
                return false;
            }
            host_allocs.push_back(m);
            resps = (Resp*) m;
            std::memset((void*) reqs, 0, sizeof(Req) * kRing);
            std::memset((void*) resps, 0, sizeof(Resp) * kRing);
            for (auto& x : layer_of) x.store(-1);
            ahead_fl.assign(L.size() * 1024, 0);
            ahead_rank.assign(L.size() * 1024, 0);
            ahead_for.assign(L.size(), 0);
            const int core = ep_cpu >= 0 ? ep_cpu : !cores.empty() && !cores[0].empty() ? cores[0][0] : -1;
            service = std::thread([this, core] {
                if (core >= 0) pin_self(core);
                serve_lane();
            });
            if (ahead_mode > 0) pf_th = std::thread([this] { prefetch_loop(); });
        }
        if (o.expert_cache) {
            size_t maxb = 0;
            for (const Layer& l : L) maxb = std::max(maxb, l.blob);
            const int nt = std::max(1, std::min(c_inflight, std::getenv("DS4_CACHE_THREADS") ? std::atoi(std::getenv("DS4_CACHE_THREADS")) : 1));
            sw_bufs.resize((size_t) nt);
            for (SwapBufs& B : sw_bufs) {
                if (!halloc(B.a, maxb, err) || !halloc(B.b, maxb, err) || !halloc(B.h, maxb, err)) return false;
                for (Dev& d : devs) {
                    cudaSetDevice(d.id);
                    cudaStream_t cs = nullptr;
                    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
                    B.st.push_back(cs);
                }
            }
            for (SwapBufs& B : sw_bufs) cache_ths.emplace_back([this, &B] { cache_loop(B); });
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
        }
        if (par || chunk > 0)   // (the windows' threads; the prompt path's, whatever mode the windows take)
            for (size_t di = 0; di < devs.size(); ++di) {
                workers.push_back(std::make_unique<Worker>());
                workers.back()->start();
            }
        calibrate_paths();
        lm_prealloc();     // (DS4_LM_PRE: pinned too, before the store's arena; its RAM out of the warm read's budget)
        init_store();     // (last: the store's arena is registered with the driver - page-locked memory the other pinned allocations must come before)
        preload_store();
        if (!ep_slices_ok(err)) return false;
        if (!init_down_parts(err)) return false;
        warm_split();
        if (o.verbose) std::fprintf(stderr, "ds4: loaded in %.1f s\n%s", (now_ms() - t0) / 1000.0, describe().c_str());
        // U9 hc's side stream and events, made last: the plan, the read-ahead, the placement and the load's report all
        // read the free VRAM before them, as the old path's do
        if (u9::on(u9::kHc))
            for (Dev& d : devs) {
                CK(cudaSetDevice(d.id));
                CK(cudaStreamCreateWithFlags(&d.st2, cudaStreamNonBlocking));
                CK(cudaEventCreateWithFlags(&d.ev_fork, cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&d.ev_join, cudaEventDisableTiming));
            }
        reset();
        return true;
    }


    // the caches' row bytes, packed as DeepSeek keeps them (ds4_kernels.cuh): the window ring's (FP8), the compressed
    // KV's (FP4 + E4M3 scales), the index keys' (FP4 + power-of-two scales)
    size_t win_rb() const { return kv_row_bytes(kKvFp8, g.hd); }
    size_t ckv_rb() const { return kv_row_bytes(kKvFp4, g.hd); }
    size_t ik_rb() const { return kv_row_bytes(kKvFp4p, g.ix_hd); }

    // ---- the chunked prompt path
    size_t x_cap() const { return (size_t) std::max<int>(kWin, chunk + 2); }   // a window's / chunk's new source rows
    size_t chunk_hand_bytes() const {
        return (size_t) chunk * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }

    // the staging ring's size (MB) for the experts a chunk streams from the CPU lane: 400 where a good part of them are
    // streamed; 16 where the cards hold most of the experts (they are within 30% of fitting) (a 4-card box) - the ring's VRAM is then worth more
    // as expert slots, and the decode after a prompt keeps the windowed path's speed (R9700 x4: 29.6 -> 31.7 tok/s)
    size_t ring_mb_auto() const {
        double exp_b = 0, other_b = 0, room = 0;
        for (const auto& kt : pk.t) {
            const std::string& n = kt.first;
            if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                (!o.dspark && n.rfind("mtp.", 0) == 0))
                continue;
            (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
        }
        for (const Dev& d : devs)
            room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) - 1.5 * (1 << 30)) * o.vram_experts;
        const bool fit = exp_b > 0 && exp_b <= 1.3 * std::max(1.0, room - other_b);   // (R9700 x4: 1.16 -> ring 16: decode -1.8% after a prompt, 64: -4.4%, 400: -4.6%)
        if (o.verbose)
            std::fprintf(stderr, "ds4: staging ring: experts %.1f GB, VRAM room %.1f GB, dense %.1f GB -> %d MB\n", exp_b / 1e9,
                         room / 1e9, other_b / 1e9, fit ? 16 : 400);
        return fit ? 16 : 400;
    }
    bool alloc_chunks(std::string& err) {
        chunks.resize(devs.size());
        const int D = g.dim, H = g.hc, HD = g.hd;
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            chunks[di] = std::make_unique<Chunk>();
            Chunk& c = *chunks[di];
            const int nm = (int) groups[(size_t) d.grp].mem.size();
            const int NHL = g.heads / head_members(nm), HQ = NHL * HD, OG = g.o_groups / head_members(nm);
            int FF = 0, ff0_ = 0;
            split128(g.ff, nm, 0, ff0_, FF);   // (the widest member's share: member 0 takes the odd blocks)
            const size_t C = (size_t) chunk;
            c.C = chunk;
            c.A = std::max(64, (int) (((size_t) 8 << 20) / (size_t) HQ) / 64 * 64);
            // (the indexer's slab: its scores rows are the context wide - about 64 MB of them)
            c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
            c.w16_elems = (size_t) 8 << 20;
            bool has_eg = false, has_first = false;
            size_t stride = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di && !l.mtp) {
                        has_eg = has_eg || l.eg >= 0;
                        has_first = has_first || l.il == 0;
                        stride = std::max(stride, align256(unit(member(l, m))));
                    }
            const size_t A = (size_t) c.A, AI = (size_t) c.AI, ss = (size_t) o.max_context + 64;
            bool ok = dalloc(d, c.h, C * H * D, err) && dalloc(d, c.x, C * D, err) && dalloc(d, c.acc, C * D, err) &&
                      dalloc(d, c.x16, C * D, err) && dalloc(d, c.qr, C * g.q_lora, err) &&
                      dalloc(d, c.qr16, C * g.q_lora, err) && dalloc(d, c.kvr, C * HD, err) &&
                      dalloc(d, c.ckr, C * HD, err) && dalloc(d, c.cgr, C * HD, err) &&
                      dalloc(d, c.latent, (C + 2) * HD, err) && dalloc(d, c.lat16, (C + 2) * HD, err) &&
                      dalloc(d, c.ikr, (C + 2) * g.ix_hd, err) && dalloc(d, c.ext, ((size_t) g.win + C) * HD, err) &&
                      dalloc(d, c.q, A * HQ, err) && dalloc(d, c.o, A * HQ, err) && dalloc(d, c.o16, A * HQ, err) &&
                      dalloc(d, c.oa, A * OG * g.o_lora, err) && dalloc(d, c.oa16, A * OG * g.o_lora, err) &&
                      dalloc(d, c.iq, AI * g.ix_heads * g.ix_hd, err) && dalloc(d, c.iw, C * 64, err) &&
                      dalloc(d, c.scores, AI * ss, err) && dalloc(d, c.bm, AI * bm_stride(), err) &&
                      dalloc(d, c.topk, C * topk_stride(), err) && dalloc(d, c.cand, C * cand_stride(), err) &&
                      dalloc(d, c.mix, C * 32, err) && dalloc(d, c.a_pre, C * 8, err) && dalloc(d, c.a_post, C * 8, err) &&
                      dalloc(d, c.a_comb, C * 64, err) && dalloc(d, c.f_pre, C * 8, err) &&
                      dalloc(d, c.f_post, C * 8, err) && dalloc(d, c.f_comb, C * 64, err) && dalloc(d, c.pre, C * 8, err) &&
                      dalloc(d, c.glog, C * 1024, err) && dalloc(d, c.ridx, C * kPicks, err) &&
                      dalloc(d, c.rw, C * kPicks, err) && dalloc(d, c.bounds, 2048, err) &&
                      dalloc(d, c.row_tok, C * kPicks, err) && dalloc(d, c.pos, C * kPicks, err) &&
                      dalloc(d, c.sg, C * FF, err) && dalloc(d, c.su, C * FF, err) && dalloc(d, c.sh16, C * FF, err) &&
                      dalloc(d, c.h16, C * FF, err) && dalloc(d, c.y, C * D, err) && dalloc(d, c.w16, c.w16_elems, err) &&
                      halloc(c.bounds_pin, 2048, err);
            if (!ok) return false;
            if (has_first && !dalloc(d, c.emb, C * D, err)) return false;
            if (has_eg) {
                const size_t ES = 256, per = (size_t) cols * (ehd + ehd / 32);
                if (!dalloc(d, c.eg_x, ES * cols * ehd, err) || !dalloc(d, c.eg_x16, ES * cols * ehd, err) ||
                    !dalloc(d, c.eg_kv, ES * (H + 1) * D, err) || !dalloc(d, c.eg_rows, C * per, err))
                    return false;
            }
            if (stride > 0) {   // DS4_RING_MB (400) of slots in kParts batches
                const size_t mb = std::getenv("DS4_RING_MB") ? (size_t) std::atoi(std::getenv("DS4_RING_MB")) : ring_mb_auto();
                c.ring_stride = stride;
                c.ring_n = std::max(Chunk::kParts, (int) ((mb << 20) / stride) / Chunk::kParts * Chunk::kParts);
                if (!dalloc(d, c.ring, (size_t) c.ring_n * stride, err)) return false;
            }
            CK(cudaStreamCreateWithFlags(&c.cs, cudaStreamNonBlocking));
            for (int i = 0; i < Chunk::kParts; ++i) {
                CK(cudaEventCreateWithFlags(&c.filled[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.freed[i], cudaEventDisableTiming));
            }
            if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(c.blas, d.st) != CUBLAS_STATUS_SUCCESS) {
                err = "cuBLAS on CUDA" + std::to_string(d.id);
                return false;
            }
            for (int sl = 0; sl < 2; ++sl) {
                if (d.mi == 0 && (!hmap(c.hand[sl], chunk_hand_bytes(), err) || !hmap(c.hand_flag[sl], 32, err))) return false;
                if (has_first && !halloc(c.emb_host[sl], C * D, err)) return false;
                if (has_eg && !halloc(c.eg_host[sl], C * pk.eg_layers * (size_t) cols * (ehd + ehd / 32), err)) return false;
                CK(cudaEventCreateWithFlags(&c.emb_ev[sl], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.eg_ev[sl], cudaEventDisableTiming));
            }
            if (d.mi == 0 && !hmap(c.hand_ack, (size_t) 2 * 8 * 32, err)) return false;
        }
        return true;
    }

    // y[r][c] = x16[r] . W[:, c] (+ beta y) for r < rows: W reconstructed to FP16 in whole 128-column blocks, cuBLAS
    // (pre: W reconstructed whole there already by dq_whole, DS4_DQ_ONCE - the same pieces' GEMMs, each over its own)
    bool gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
              std::string& err, const uint16_t* pre = nullptr) {
        if (rows <= 0) return true;
        const int cols = gemm16::cols(m, c.w16_elems);
        const bool gp = prof_chunk;
        for (int c0 = 0; c0 < m.n; c0 += cols) {
            const int nc = std::min(cols, m.n - c0);
            GmMark gm{};
            if (gp) {
                std::lock_guard<std::mutex> lk(prof_mu);
                gm.e0 = ek_ev();
                gm.e1 = ek_ev();
                gm.e2 = ek_ev();
                gm.k = m.k;
                gm.n = m.n;
                gm.nc = nc;
                gm.rows = rows;
                gm.first = c0 == 0;
                cudaGetDevice(&gm.dev);
                cudaEventRecord(gm.e0, d.st);
            }
            if (pre == nullptr) gemm16::dequant(m, c0, nc, c.w16, d.st);
            if (gp) cudaEventRecord(gm.e1, d.st);
            const uint16_t* w16 = pre != nullptr ? gemm16::piece(pre, c0, m.k) : c.w16;
            const cublasStatus_t st = gemm16::mm(c.blas, w16, m.k, nc, x16, ldx, rows, y + c0, ldy, beta);
            if (gp) {
                cudaEventRecord(gm.e2, d.st);
                std::lock_guard<std::mutex> lk(prof_mu);
                gm_marks.push_back(gm);
            }
            if (st != CUBLAS_STATUS_SUCCESS) {
                err = "cublasGemmEx: status " + std::to_string((int) st);
                return false;
            }
        }
        return true;
    }
    // DS4_DQ_ONCE: W reconstructed whole into dst in gemm()'s pieces (piece c0 at dst + c0 k), for gemm(.., pre = dst)
    void dq_whole(Dev& d, Chunk& c, const x3::Mat& m, uint16_t* dst) {
        const bool gp = prof_chunk;
        GmMark gm{};
        gemm16::dequant_whole(
            m, c.w16_elems, dst, d.st,
            [&](int, int nc) {
                if (!gp) return;
                gm = GmMark{};
                std::lock_guard<std::mutex> lk(prof_mu);
                gm.e0 = ek_ev();
                gm.e1 = ek_ev();
                gm.e2 = ek_ev();
                gm.k = m.k;
                gm.n = m.n;
                gm.nc = nc;
                cudaGetDevice(&gm.dev);
                cudaEventRecord(gm.e0, d.st);
            },
            [&] {
                if (!gp) return;
                cudaEventRecord(gm.e1, d.st);
                cudaEventRecord(gm.e2, d.st);
                std::lock_guard<std::mutex> lk(prof_mu);
                gm_marks.push_back(gm);
            });
    }
    // DS4_DQ_ONCE's buffer: l's slab weights' FP16 values at their places (0: l has no heads)
    static size_t once_elems(const Layer& l) {
        return l.ahn == 0 ? 0 : gemm16::places(nullptr, l.wq_b, l.wo_a, l.ogn, l.wo_b).elems;
    }
    // DS4_DQ_ONCE=1: room on each device for its widest layer's attention slab weights (wq_b, wo_a's groups, wo_b) as
    // FP16, reconstructed once a (chunk, layer) - made before the experts take what VRAM is left.  Not had: that
    // device's slabs reconstruct them each, as without it
    void alloc_dq_once() {
        for (size_t di = 0; di < devs.size() && di < chunks.size(); ++di) {
            Chunk& c = *chunks[di];
            size_t need = 0;
            for (const Layer& l0 : L)   // (not the MTP layers: alloc_chunks leaves them out, no layer_chunk runs them)
                for (int m = 0; m < l0.nm; ++m)
                    if (member(l0, m).dev == (int) di && !l0.mtp) need = std::max(need, once_elems(member(l0, m)));
            if (need == 0) continue;
            std::string e;
            if (!dalloc(devs[di], c.w16_once, need, e)) {
                cudaGetLastError();   // (the failed cudaMalloc's error, not left for a kernel's check to find)
                c.w16_once = nullptr;
                std::fprintf(stderr, "ds4: DS4_DQ_ONCE: %s - the attention slabs reconstruct their weights each slab\n",
                             e.c_str());
                continue;
            }
            c.w16_once_elems = need;
            std::fprintf(stderr, "ds4: DS4_DQ_ONCE: the attention slabs' weights as FP16 once a (chunk, layer): %.1f MB on "
                                 "CUDA%d (the w16 scratch's %.1f MB as before)\n",
                         (double) need * 2 / 1e6, devs[di].id, (double) c.w16_elems * 2 / 1e6);
        }
    }
    // y[r][c] = alpha x16[r] . W[c] for F16 weights W [n][k] (the indexer's weights_proj)
    bool gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows, float alpha,
                   float* y, int ldy, std::string& err) {
        const float zero = 0.0f;
        const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F, k,
                                               x16, CUDA_R_16F, ldx, &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            err = "cublasGemmEx (F16 weights): status " + std::to_string((int) st);
            return false;
        }
        return true;
    }
    // hc_mix over a chunk: the dots by an F32 GEMM, then the rms scale
    bool hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes, std::string& err) {
        const int hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, fn, hcd, h, hcd, &zero, mixes, 32) !=
            CUBLAS_STATUS_SUCCESS) {
            err = "cublasSgemm (mHC mixes)";
            return false;
        }
        hc_mix_scale(h, hcd, n_mix, T, g.eps, mixes, d.st);
        return true;
    }

    // expert e of l (on the CPU lane: its NH column parts) -> member m's VRAM copy at dst, in the GPU layout (the full
    // blob, or a group member's part), by DMA from the pinned parts on the copy stream (in place: from `src`, its file
    // region read into pinned staging)
    bool stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                       const uint8_t* src = nullptr) {
        if (l.lay.fmt == gg::kFmtMXFP4) {   // the blob, or member m's rows / blocks: from `src` (the expert read into
            // pinned staging as its blob) or from the file's mapping
            auto mat_r = [&](int r) { return src != nullptr ? src + l.lay.trellis[r] : mx_mat(l, e, r); };
            bool ok = true;
            if (l.nm == 1) {
                if (src != nullptr)
                    ok = cudaMemcpyAsync(dst, src, l.blob, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                else
                    for (int r = 0; r < 3; ++r)
                        ok = ok && cudaMemcpyAsync(dst + l.lay.trellis[r], mat_r(r), mx_part_bytes(l, r),
                                                   cudaMemcpyHostToDevice, cs) == cudaSuccess;
            } else {
                const int ffh = g.ff / l.nm;
                const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                             dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                             dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
                for (int r = 0; r < 2; ++r)
                    ok = ok && cudaMemcpyAsync(dst + l.hlay.trellis[r], mat_r(r) + (size_t) m * ffh * gr,
                                               (size_t) ffh * gr, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                ok = ok && cudaMemcpy2DAsync(dst + l.hlay.trellis[2], dh, mat_r(2) + (size_t) m * dh, dr, dh,
                                             (size_t) g.dim, cudaMemcpyHostToDevice, cs) == cudaSuccess;
            }
            if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
            return ok;
        }
        const auto& parts = l.hh[(size_t) e];
        const int D = g.dim, FF = g.ff;
        auto cp2 = [&](uint8_t* d0, size_t dp, const uint8_t* s0, size_t sp, size_t w, size_t h) {
            return cudaMemcpy2DAsync(d0, dp, s0, sp, w, h, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        auto cp = [&](uint8_t* d0, const uint8_t* s0, size_t n) {
            return cudaMemcpyAsync(d0, s0, n, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        bool ok = true;
        if (NH == 1) {   // one part: the whole expert in exllamav3's layout, its pieces at hl's offsets
            const uint8_t* p = src != nullptr ? src : parts[0];
            if (l.nm == 1) {
                for (int r = 0; r < 3; ++r)
                    ok = ok && cp(dst + l.lay.trellis[r], p + l.hl.trellis[r], l.piece_bytes[r][0]) &&
                         cp(dst + l.lay.suh[r], p + l.hl.suh[r], (size_t) l.hl.k[r] * 2) &&
                         cp(dst + l.lay.svh[r], p + l.hl.svh[r], (size_t) l.hl.n[r] * 2);
            } else {   // member m's part (its FFN columns [f0, f0 + fn)), cut as blob_to_half cuts a blob
                const Layer& x = member(l, m);
                const int f0 = x.ff0, fn = x.ffn;
                for (int r = 0; r < 2; ++r) {
                    const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) FF / 16, nth = (size_t) fn / 16;
                    ok = ok && cp2(dst + x.hlay.trellis[r], nth * tile, p + l.hl.trellis[r] + (size_t) (f0 / 16) * tile,
                                   nt * tile, nth * tile, (size_t) D / 16) &&
                         cp(dst + x.hlay.suh[r], p + l.hl.suh[r], (size_t) D * 2) &&
                         cp(dst + x.hlay.svh[r], p + l.hl.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
                }
                const size_t row = (size_t) (D / 16) * 32 * l.lay.K[2];
                ok = ok && cp(dst + x.hlay.trellis[2], p + l.hl.trellis[2] + (size_t) (f0 / 16) * row,
                              (size_t) (fn / 16) * row) &&
                     cp(dst + x.hlay.suh[2], p + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2) &&
                     cp(dst + x.hlay.svh[2], p + l.hl.svh[2], (size_t) D * 2);
            }
        } else if (l.nm == 1) {   // the full blob: each matrix's columns from the two parts side by side
            for (int r = 0; r < 3; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int k = l.hl.k[r], nh = l.hl.n[r];   // part: k x nh
                for (int X = 0; X < 2; ++X) {
                    ok = ok && cp2(dst + l.lay.trellis[r] + (size_t) X * (nh / 16) * tile, (size_t) (2 * nh / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r], (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile,
                                   (size_t) (k / 16));
                    ok = ok && cp(dst + l.lay.svh[r] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[r], (size_t) nh * 2);
                }
                ok = ok && cp(dst + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            }
        } else {   // member m (FFN columns [f0, f0 + fn)): gate / up from the parts holding them; down rows [f0, ..) of
                   // both parts' columns
            const Layer& x = member(l, m);
            const int f0 = x.ff0, fn = x.ffn;
            for (int r = 0; r < 2; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int hn = l.hl.n[r];   // part X: columns [X hn, (X + 1) hn)
                for (int X = 0; X < 2; ++X) {
                    const int a = std::max(f0, X * hn), b = std::min(f0 + fn, (X + 1) * hn);
                    if (a >= b) continue;
                    ok = ok && cp2(dst + x.hlay.trellis[r] + (size_t) ((a - f0) / 16) * tile, (size_t) (fn / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r] + (size_t) ((a - X * hn) / 16) * tile,
                                   (size_t) (hn / 16) * tile, (size_t) ((b - a) / 16) * tile, (size_t) (D / 16)) &&
                         cp(dst + x.hlay.svh[r] + (size_t) (a - f0) * 2, parts[(size_t) X] + l.hl.svh[r] + (size_t) (a - X * hn) * 2,
                            (size_t) (b - a) * 2);
                }
                ok = ok && cp(dst + x.hlay.suh[r], parts[0] + l.hl.suh[r], (size_t) D * 2);
            }
            const size_t tile = (size_t) 32 * l.lay.K[2];
            const int nh = D / 2;
            for (int X = 0; X < 2; ++X) {
                ok = ok && cp2(dst + x.hlay.trellis[2] + (size_t) X * (nh / 16) * tile, (size_t) (D / 16) * tile,
                               parts[(size_t) X] + l.hl.trellis[2] + (size_t) (f0 / 16) * (nh / 16) * tile,
                               (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile, (size_t) (fn / 16));
                ok = ok && cp(dst + x.hlay.svh[2] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[2], (size_t) nh * 2);
            }
            ok = ok && cp(dst + x.hlay.suh[2], parts[0] + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
        return ok;
    }

    // a device's streamed experts: batches of ring_n / kParts of each of its layers' CPU-lane experts, chunk by chunk
    // in layer order; the copies run at most kParts batches ahead of the compute
    struct Batch {
        int k = 0, il = 0, j0 = 0, n = 0;   // chunk, layer, its host_list range
    };
    struct Streamer {
        std::vector<Batch> b;
        size_t cp = 0, kp = 0;        // the next batch to copy, to compute
        std::atomic<size_t> done{0};  // cp, for read_ahead's thread
    };
    // in place (file_experts): the batches' file regions read ahead of their copies - POSIX_FADV_WILLNEED, the kernel
    // reading them in the background in large requests, where the copies alone fault the pages in a few at a time -
    // at most DS4_READAHEAD_MB (2048) beyond the batch being copied; each region by its layer's first member alone
    void read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const {
        static const double cap = (std::getenv("DS4_READAHEAD_MB") ? std::atof(std::getenv("DS4_READAHEAD_MB")) : 2048.0) *
                                  1048576.0;
        std::deque<std::pair<size_t, double>> issued;   // (batch, bytes) not yet copied
        double ahead = 0;
        for (size_t q = 0; q < S.b.size() && !stop.load(std::memory_order_relaxed);) {
            const size_t done = S.done.load(std::memory_order_relaxed);
            while (!issued.empty() && issued.front().first < done) {
                ahead -= issued.front().second;
                issued.pop_front();
            }
            if (ahead > cap) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            const Batch& bt = S.b[q];
            const Layer& l0 = L[(size_t) bt.il];
            double nb = 0;
            if (l0.dev == di)
                for (int i = 0; i < bt.n; ++i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    if (l0.lay.fmt == gg::kFmtMXFP4) {
                        if (copy_mode && l0.ram[(size_t) e] >= 0) continue;   // (in RAM: nothing to read)
                        for (int r = 0; r < 3; ++r)
                            os::advise(pk.fds[(size_t) l0.mx_file[r]], (l0.mx_off[r] + (uint64_t) e * l0.mx_stride[r]), mx_part_bytes(l0, r), true);
                        nb += (double) l0.blob;
                        continue;
                    }
                    os::advise(pk.fds[(size_t) l0.region_file[(size_t) e]], l0.region_off[(size_t) e], l0.region_bytes[(size_t) e], true);
                    nb += (double) l0.region_bytes[(size_t) e];
                }
            issued.push_back({q++, nb});
            ahead += nb;
        }
    }
    bool copy_ahead(Dev& d, Chunk& c, Streamer& S, int di, std::string& err) {
        const int P = Chunk::kParts;
        while (S.cp < S.b.size() && S.cp < S.kp + P) {
            const Batch& bt = S.b[S.cp];
            const int half = (int) (S.cp % P);
            const Layer& l0 = L[(size_t) bt.il];
            int mi = 0;
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di) mi = m;
            if (S.cp >= (size_t) P) CK(cudaStreamWaitEvent(c.cs, c.freed[half], 0));
            uint8_t* sb = nullptr;   // in place: the batch's regions in the part's staging slots
            if (c.stage != nullptr) {
                if (S.cp >= (size_t) P) CK(cudaEventSynchronize(c.filled[half]));   // (its last batch's copies out)
                sb = c.stage + (size_t) half * (c.ring_n / P) * c.stage_stride;
                std::atomic<bool> bad{false};
                c.readers->run(bt.n, [&](int i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    if (l0.lay.fmt == gg::kFmtMXFP4 ? !read_mx(l0, e, sb + (size_t) i * c.stage_stride)
                                                    : !pk.read_at(l0.region_file[(size_t) e], l0.region_off[(size_t) e],
                                                                  l0.region_bytes[(size_t) e],
                                                                  sb + (size_t) i * c.stage_stride))
                        bad.store(true);
                });
                if (bad.load()) {
                    err = "reading layer " + std::to_string(l0.il) + "'s experts from the model's files failed";
                    return false;
                }
            }
            for (int i = 0; i < bt.n; ++i) {
                uint8_t* dst = c.ring + ((size_t) half * (c.ring_n / P) + i) * c.ring_stride;
                if (!stream_expert(l0, mi, l0.host_list[(size_t) (bt.j0 + i)], dst, c.cs, err,
                                   sb != nullptr ? sb + (size_t) i * c.stage_stride : nullptr))
                    return false;
                c.stream_bytes += (double) unit(member(l0, mi));
            }
            CK(cudaEventRecord(c.filled[half], c.cs));
            ++S.cp;
            S.done.store(S.cp, std::memory_order_relaxed);
        }
        return true;
    }

    // ---- the flow (see Chunk)
    // Windows (WDDM: HIP's runtime and CUDA's driver alike) submits queued work only when the host waits on the GPU:
    // the flow's copies and the compute it queued between its waits are submitted at once (win_submit); elsewhere
    // nothing.  (CUDA too since #303: two RTX 3080s hung before the first layer, one GPU spinning on the other's flow)
    static void kick(cudaStream_t s) { win_submit(s); }
    size_t flow_region(int r) const {   // a part's room in a staging slot (read from its offset rounded down)
        return ((flow_part[r] + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1)) + os::kDirectAlign;
    }
    size_t flow_stride() const { return flow_region(0) + flow_region(1) + flow_region(2); }
    static Chunk::Item& fit(Chunk& c, size_t i) { return c.items[i - c.base]; }
    // what member m of l's group reads of matrix r of an expert: gate / up its rows [m n_ff / nm, ..) - one run, from
    // this offset in the matrix, this long; the down the whole matrix (its part, the blocks [m n_ff / nm / 32, ..) of
    // every row, is DMA'd out of it row by row); alone (nm 1) every matrix whole
    size_t mx_read_off(const Layer& l, int m, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) m * (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : 0;
    }
    size_t mx_read_len(const Layer& l, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : mx_part_bytes(l, r);
    }
    // the device's member index in L0's group (0 alone)
    int flow_mi(const Chunk& c, const Layer& L0) const {
        for (int m = 0; m < L0.nm; ++m)
            if (member(L0, m).dev == c.dev) return m;
        return 0;
    }
    void flow_push(Chunk& c, const Layer& l, int mi, int k, int e, int v) {
        c.items.emplace_back();
        Chunk::Item& it = c.items.back();
        it.il = l.il;
        it.mi = mi;
        it.k = k;
        it.e = e;
        it.v = v;
        bool in_ram = !l.ram.empty() && l.ram[(size_t) e] >= 0;
        const uint8_t* store_blob = nullptr;   // (the RAM store holds it: copied into staging like a RAM-tier slot, no read)
        if (!in_ram && store && l.nm == 1) {
            const int ss = store->find(l.il, e);
            if (ss >= 0) {
                store_blob = store->blob(ss);
                in_ram = true;
            }
        }
        if (in_ram && ram_pinned && store_blob == nullptr) {   // (DMA'd from where it lies)
            const uint8_t* p = ram_slots[(size_t) l.ram[(size_t) e]];
            for (int r = 0; r < 3; ++r) it.src[r] = p + l.lay.trellis[r] + mx_read_off(l, mi, r);
            ++c.it_ram;
            return;
        }
        if (in_ram)   // (copied into staging by io, then DMA'd; copy_mode: from its NH parts, flow_pump)
            it.ram = store_blob != nullptr ? store_blob
                     : copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] : ram_slots[(size_t) l.ram[(size_t) e]];
        bool hit = false;
        if (!in_ram && c.lc != nullptr && l.nm == 1) {   // layer-major: from the layer cache, or kept for the chunks after
            const size_t i = c.base + c.items.size() - 1, stride = (l.blob + 63) & ~(size_t) 63;
            if (c.lc_il != l.il) {   // (a layer's first item: the slots the layer before used are its own again
                c.lc_il = l.il;      // once that layer's items are all read - `after` below)
                c.lc_used = 0;
                c.lc_n = (int) std::min<size_t>(c.lc_of.size(), c.lc_bytes / stride);
                c.lc_first = i;
                std::fill(c.lc_of.begin(), c.lc_of.end(), -1);
            }
            const int j = c.lc_of[(size_t) e];
            if (j >= 0) {   // (copied into staging by io once its reader is done, then DMA'd)
                it.ram = c.lc + (size_t) j * stride;
                it.after = c.lc_writer[(size_t) j];
                hit = true;
            } else if (k + 1 < (int) c.fl_T.size() && c.lc_used < c.lc_n) {   // (a later chunk may pick it)
                const int jn = c.lc_used++;
                c.lc_of[(size_t) e] = jn;
                c.lc_writer[(size_t) jn] = (int64_t) i;
                it.keep = c.lc + (size_t) jn * stride;
                it.after = (int64_t) c.lc_first - 1;
            }
        }
        it.stage = (int) (c.n_file++ % (uint64_t) c.S);
        uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = l.mx_off[r] + (uint64_t) e * l.mx_stride[r] + mx_read_off(l, mi, r);
            it.src[r] = sb + (in_ram || hit ? 0 : off % os::kDirectAlign);
            sb += flow_region(r);
        }
        it.left.store(in_ram && copy_mode && store_blob == nullptr ? 3 * NH : 3, std::memory_order_relaxed);
        ++(in_ram ? c.it_ram : hit ? c.it_lc : c.it_file);
    }
    // the likely experts of (chunk k, layer il) for T tokens, first in its stream: the CPU lane's experts whose chance
    // to be picked at least once, 1 - (1 - share)^(picks), is at least DS4_SPEC_Q (0.9: the reads are what takes the
    // time, an unpicked one costs a whole read, and the routing is known a few ms after the layer before is done -
    // early reads pay where a chunk's long attention keeps the disk otherwise idle, and there nearly every expert is
    // that likely), likeliest first, at most what the staging and the ring hold (no more can be read before the layer
    // routes); the layer's virtual slots put them first, then its other CPU-lane experts in index order (the order the
    // rest of its picks are streamed in)
    // (a tensor-parallel group's members each decide their own stream and virtual slots - the same, from the same
    // profile and the same routing - and each reads only its part of every expert)
    void flow_spec(Chunk& c, int k, int il, int T) {
        static const double spec_q = std::getenv("DS4_SPEC_Q") ? std::atof(std::getenv("DS4_SPEC_Q")) : 0.9;
        Layer& l = L[(size_t) il];
        const int mi = flow_mi(c, l);
        int* vs = member(l, mi).vs_pin;
        if (bc_once != 0) cudaEventSynchronize(c.vs_ev);   // (the last slots sent read from vs_pin no more)
        c.spec_k = k;
        c.spec_il = il;
        c.spec_begin = c.base + c.items.size();
        const uint32_t* pr = prof.data() + (size_t) il * g.n_exp;
        double sum = 0;
        for (int e = 0; e < l.n_exp; ++e) sum += pr[e];
        std::vector<std::pair<double, int>> cand;
        const double picks = (double) l.k_exp * T;
        // (early reads only while they hit: once a run's likely ones went unpicked more often than DS4_SPEC_WASTE
        // (0.2) of the time, the rest of the run reads only what the routing picked - an unpicked one costs a whole
        // read, and how well the profile guesses depends on the PC's history and the prompt: V100, 64 tokens - 376
        // of 921 likely ones unpicked, 7 GB of 84 read for nothing; Radeon 8065S, 128 and 512 tokens - 44 of 945 and
        // 21 of 1568, the early reads 9-12% faster there)
        static const double waste = std::getenv("DS4_SPEC_WASTE") ? std::atof(std::getenv("DS4_SPEC_WASTE")) : 0.2;
        const bool early = c.it_spec < 2u * (uint64_t) c.ring_n || (double) c.it_spec_unused <= waste * (double) c.it_spec;
        for (int e : l.host_list) {
            if (!early) break;
            const double p = sum > 0 ? pr[e] / sum : 0.0;
            const double q = p >= 1.0 ? 1.0 : 1.0 - std::exp(picks * std::log1p(-p));
            if (q >= spec_q && spec_q <= 1.0) cand.push_back({q, e});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first;
        });
        if (cand.size() > (size_t) (c.S + c.ring_n)) cand.resize((size_t) (c.S + c.ring_n));
        std::vector<char> is((size_t) l.n_exp, 0);
        int j = 0;
        for (const auto& x : cand) {
            vs[x.second] = l.n_slots + j++;
            is[(size_t) x.second] = 1;
        }
        for (int e : l.host_list)
            if (!is[(size_t) e]) vs[e] = l.n_slots + j++;
        // DS4_BC_ONCE: the slots on their way to the member's vs_dev now, on cs, off the compute stream - the sort of
        // (k, il) waits for vs_ev instead of copying them itself (layer_chunk).  Nothing still reads that vs_dev: this
        // is called only once the compute stream was waited for (flow_layer, after its layer's bounds wait; the run's
        // start), and the layer's sort before it is there; nothing writes vs_pin again before the sort of (k, il) is
        // done (the next flow_spec waits for vs_ev above; plan_streams).  Not sent: the sort copies them, as before
        if (bc_once != 0) {
            Layer& x = member(l, mi);
            const bool sent =
                cudaMemcpyAsync(x.vs_dev, x.vs_pin, (size_t) l.n_exp * 4, cudaMemcpyHostToDevice, c.cs) == cudaSuccess &&
                cudaEventRecord(c.vs_ev, c.cs) == cudaSuccess;
            kick(c.cs);
            c.vs_k = sent ? k : -1;
            c.vs_il = sent ? il : -1;
            c.vs_mi = sent ? mi : -1;
        }
        for (const auto& x : cand) flow_push(c, l, mi, k, x.second, vs[x.second]);
        c.n_spec = (int) cand.size();
        c.it_spec += cand.size();
    }
    // (chunk k, layer il) routed (bounds b by virtual slot): the rest of its picks after its likely ones
    void flow_rest(Chunk& c, int k, int il, const std::vector<int>& b) {
        const Layer& l = L[(size_t) il];
        const int mi = flow_mi(c, l);
        const int* vs = member(l, mi).vs_pin;
        for (size_t i = c.spec_begin; i < c.spec_begin + (size_t) c.n_spec; ++i) {
            const int v = fit(c, i).v;
            if (b[(size_t) v + 1] == b[(size_t) v]) ++c.it_spec_unused;
        }
        for (int e : l.host_list) {   // (index order: their virtual slots ascending)
            const int v = vs[e];
            if (v >= l.n_slots + c.n_spec && b[(size_t) v + 1] > b[(size_t) v]) flow_push(c, l, mi, k, e, v);
        }
    }
    // the next (chunk, layer) of the device after (k, il) in the run's order that streams experts (false: none) - a
    // long prompt's skip leaves out its early chunks' layers from Run::yoco on (kv_only there: no experts)
    bool flow_next(const Chunk& c, int k, int il, int& k2, int& il2) const {
        while (flow_next1(c, k, il, k2, il2)) {
            if (c.fl_yoco < 0 || k2 >= c.fl_cut_k || il2 < c.fl_yoco) return true;
            k = k2;
            il = il2;
        }
        return false;
    }
    // the rows chunk k runs layer il on (a long prompt's skip: the cut chunk's from cut_r on, from the yoco layer on)
    static int fl_rows(const Chunk& c, int k, int il) {
        const int T = c.fl_T[(size_t) k];
        return c.fl_yoco >= 0 && k == c.fl_cut_k && il >= c.fl_yoco ? T - c.fl_cut_r : T;
    }
    bool flow_next1(const Chunk& c, int k, int il, int& k2, int& il2) const {
        const auto it = std::find(c.fl_layers.begin(), c.fl_layers.end(), il);
        const size_t i = (size_t) (it - c.fl_layers.begin());
        if (c.lm) {   // (layer-major: the layer's next chunk, then the next layer's first)
            if (k + 1 < (int) c.fl_T.size()) {
                k2 = k + 1;
                il2 = il;
                return true;
            }
            if (i + 1 >= c.fl_layers.size()) return false;
            k2 = 0;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (i + 1 < c.fl_layers.size()) {
            k2 = k;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (k + 1 < (int) c.fl_T.size() && !c.fl_layers.empty()) {
            k2 = k + 1;
            il2 = c.fl_layers[0];
            return true;
        }
        return false;
    }
    // the reads (in order, each into its staging slot once that slot's last DMA is done) and the DMAs (in order, each
    // into its ring slot once that slot's last use is queued: the copy stream waits for it) as far as they can go now
    bool flow_pump(Chunk& c, std::string& err) {
        const size_t end = c.base + c.items.size();
        IoQueue::Job jobs[6];
        while (c.n_read < end) {
            Chunk::Item& it = fit(c, c.n_read);
            if (it.after >= 0) {   // layer-major: a slot's expert copied out of it once its reader is done; a slot
                // written again once the layer before's items are all read (none still copies out of it)
                bool wait = false;
                if (it.keep == nullptr)
                    wait = (size_t) it.after >= c.base && fit(c, (size_t) it.after).left.load(std::memory_order_acquire) != 0;
                else {
                    while (c.lc_chk < c.n_read &&
                           (c.lc_chk < c.base || fit(c, c.lc_chk).left.load(std::memory_order_acquire) == 0))
                        ++c.lc_chk;
                    wait = c.lc_chk <= (size_t) it.after;
                }
                if (wait) break;
            }
            if (it.stage >= 0) {
                const int64_t prev = c.stage_item[(size_t) it.stage];
                if (prev >= 0) {
                    if ((size_t) prev >= c.n_dma) break;
                    const cudaError_t q = cudaEventQuery(c.stage_ev[(size_t) it.stage]);
                    if (q == cudaErrorNotReady) break;
                    CK(q);
                }
                c.stage_item[(size_t) it.stage] = (int64_t) c.n_read;
                const Layer& l = L[(size_t) it.il];
                uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
                const int cs = copy_mode && it.ram != nullptr ? l.ram[(size_t) it.e] : -1;   // (a copy's slot)
                int nj = 0;
                for (int r = 0; r < 3; ++r) {
                    if (cs >= 0) {   // a copy: each matrix's NH parts of rows, side by side into staging
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X) {
                            IoQueue::Job& j = jobs[nj++];
                            j = IoQueue::Job{};
                            j.len = nb;
                            j.dst = sb + (size_t) X * nb;
                            j.left = &it.left;
                            j.bad = &it.bad;
                            j.mem = ram_part[(size_t) cs][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
                        }
                        sb += flow_region(r);
                        continue;
                    }
                    IoQueue::Job& j = jobs[nj++];
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) it.e * l.mx_stride[r] + mx_read_off(l, it.mi, r);
                    j.len = mx_read_len(l, r);
                    j.dst = sb;
                    j.left = &it.left;
                    j.bad = &it.bad;
                    j.mem = it.ram != nullptr ? it.ram + l.lay.trellis[r] + mx_read_off(l, it.mi, r) : nullptr;
                    j.map = pk.at(l.mx_file[r], j.off);
                    j.keep = it.keep != nullptr ? it.keep + l.lay.trellis[r] : nullptr;
                    sb += flow_region(r);
                }
                c.io->push(jobs, nj);
            }
            ++c.n_read;
        }
        const size_t R = (size_t) c.ring_n, dma0 = c.n_dma;
        struct Kick {   // (every way out: what was queued is submitted)
            cudaStream_t s;
            const size_t &a, b;
            ~Kick() {
                if (a != b) kick(s);
            }
        } kq{c.cs, c.n_dma, dma0};
        while (c.n_dma < c.n_read && c.n_dma < c.n_comp + R) {
            Chunk::Item& it = fit(c, c.n_dma);
            if (it.left.load(std::memory_order_acquire) != 0) break;
            if (it.bad.load(std::memory_order_acquire)) {
                err = "reading layer " + std::to_string(it.il) + " expert " + std::to_string(it.e) +
                      " from the model's files failed";
                return false;
            }
            const size_t r = c.n_dma % R;
            if (c.n_dma >= R) CK(cudaStreamWaitEvent(c.cs, c.ring_free[r], 0));
            const Layer& l = L[(size_t) it.il];
            uint8_t* dst = c.ring + r * c.ring_stride;
            const x3::ExpertLayout& lay = l.nm > 1 ? l.hlay : l.lay;
            for (int p = 0; p < 2; ++p)
                CK(cudaMemcpyAsync(dst + lay.trellis[p], it.src[p], mx_read_len(l, p), cudaMemcpyHostToDevice, c.cs));
            if (l.nm > 1) {   // (the down's part: in each of its rows the member's blocks)
                const int t2 = gg::expert_type(l.lay, 2);
                const size_t dr = gg::type_row_bytes(t2, g.ff), dh = gg::type_row_bytes(t2, g.ff / l.nm);
                CK(cudaMemcpy2DAsync(dst + lay.trellis[2], dh, it.src[2] + (size_t) it.mi * dh, dr, dh, (size_t) g.dim,
                                     cudaMemcpyHostToDevice, c.cs));
            } else
                CK(cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs));
            CK(cudaEventRecord(c.ring_fill[r], c.cs));
            if (it.stage >= 0) CK(cudaEventRecord(c.stage_ev[(size_t) it.stage], c.cs));
            c.stream_bytes += (double) unit(l);
            ++c.n_dma;
        }
        return true;
    }
    // DS4_BC_ONCE (bc_once 1 / 2): the layer's batches - what flow_layer's loop walks from n_comp to le, the batch
    // partition depending on the stream alone (flow_batches) - and each one's bounds written into its region now, the
    // regions the loop takes (n_batch on), with the values it would write itself; 1: sent to bc_dev in one copy on s
    // (two where the regions wrap) before the first batch, instead of a copy a batch behind that batch's ring_fill -
    // a small copy on s queues behind the copy stream's DMAs on the same copy engine.  The regions: one each
    // (bc_layer_fits), and the layers before are done with theirs once bc_done is (their last work on s; the
    // layer's bounds wait already waited for it)
    bool flow_bc_upfront(Chunk& c, size_t le, const std::vector<int>& b, cudaStream_t s, std::string& err) {
        static std::once_flag said;
        std::call_once(said, [&] {
            std::fprintf(stderr, bc_once == 1
                         ? "ds4: DS4_BC_ONCE=1: the expert stage's batch bounds written once a layer routed and sent to "
                           "the GPU in one copy a layer (two where they wrap), not one a batch; a layer's virtual slots "
                           "sent on the copy stream when the stream decides them, not before the sort\n"
                         : "ds4: DS4_BC_ONCE=2: the expert stage's batch bounds written once a layer routed and read by "
                           "the kernels from mapped host memory, no copy; a layer's virtual slots sent on the copy "
                           "stream when the stream decides them, not before the sort\n");
        });
        const size_t R = (size_t) c.ring_n, bmax = std::max<size_t>(1, R / 2);
        flow_batches(c.n_comp, le, R, bmax, c.bc_list);
        c.bc_n0 = c.n_batch;
        const size_t nb = c.bc_list.size();
        if (!bc_layer_fits(nb, c.bc_n)) {
            err = "DS4_BC_ONCE: " + std::to_string(nb) + " batches in a layer, more than the " + std::to_string(c.bc_n) +
                  " regions of their bounds";
            return false;
        }
        CK(cudaEventSynchronize(c.bc_done));
        for (size_t j = 0; j < nb; ++j)
            flow_batch_bounds(b.data(), c.bc_list[j].i0, c.bc_list[j].i1, [&](size_t i) { return fit(c, i).v; },
                              c.bc_pin + bc_region(c.n_batch + j, c.bc_n) * (size_t) c.bc_w);
        if (bc_once == 1) {
            BcRun run[2];
            const int nr = bc_runs(c.n_batch, nb, c.bc_n, run);
            const size_t w = (size_t) c.bc_w;
            for (int r = 0; r < nr; ++r)
                CK(cudaMemcpyAsync(c.bc_dev + run[r].first * w, c.bc_pin + run[r].first * w, run[r].n * w * 4,
                                   cudaMemcpyHostToDevice, s));
            if (nr > 0) kick(s);
        }
        return true;
    }
    // layer L0 of chunk k routed (bounds b, the device's copy d_bounds): its streamed experts as they land, the VRAM
    // ones in shares between them (the device works while the reads run)
    bool flow_layer(Dev& d, Chunk& c, Layer& L0, int mi, int k, int T, const std::vector<int>& b, std::string& err) {
        cudaStream_t s = d.st;
        const uint8_t* vbase = member(L0, mi).vram_base;   // (a member's: its part of each VRAM expert, L0's slots)
        if (c.spec_k != k || c.spec_il != L0.il) {
            err = "the prompt path's expert stream is out of step (layer " + std::to_string(L0.il) + ")";
            return false;
        }
        flow_rest(c, k, L0.il, b);
        const size_t lb = c.spec_begin, le = c.base + c.items.size(), n_lay = le - lb;
        int k2 = 0, il2 = 0;
        if (flow_next(c, k, L0.il, k2, il2)) flow_spec(c, k2, il2, fl_rows(c, k2, il2));
        const size_t R = (size_t) c.ring_n, bmax = std::max<size_t>(1, R / 2);
        if (bc_once != 0 && !flow_bc_upfront(c, le, b, s, err)) return false;
        int v_done = 0;
        while (c.n_comp < le) {
            // a batch's bounds are fixed by the stream alone (bmax items, never across the ring's end), never by what
            // has landed: the experts' outputs are summed batch after batch (the VRAM ones' shares between them), so
            // the sums come out the same on every run whatever the reads' timing
            const size_t i0 = c.n_comp, i1 = flow_batch_end(i0, le, R, bmax);
            if (!flow_pump(c, err)) return false;
            if (c.n_dma < i1) {   // not all of the batch landed: wait for a read, or for a staging slot's copy
                const double tw = now_ms();
                kick(s);
                if (c.n_read > c.n_dma) c.io->wait_any(500);
                else {
                    const Chunk::Item& it = fit(c, c.n_read);
                    if (it.stage >= 0) CK(cudaEventSynchronize(c.stage_ev[(size_t) it.stage]));
                }
                c.wait_ms += now_ms() - tw;
                continue;
            }
            const int n = (int) (i1 - i0);
            if (bc_once != 0) {   // (the batch the up-front list has at this place, or the bounds there are another's)
                const uint64_t j = c.n_batch - c.bc_n0;
                if (j >= c.bc_list.size() || c.bc_list[(size_t) j].i0 != i0 || c.bc_list[(size_t) j].i1 != i1) {
                    err = "DS4_BC_ONCE: the expert stage's batches out of step with their bounds (layer " +
                          std::to_string(L0.il) + ")";
                    return false;
                }
            }
            const size_t reg = bc_region(c.n_batch, c.bc_n) * (size_t) c.bc_w;
            int* bc = c.bc_pin + reg;
            int* bcd = (bc_once == 2 ? c.bc_map : c.bc_dev) + reg;
            ++c.n_batch;
            // (the items' virtual slots ascend and the slots between them have no rows: their bounds side by side;
            // DS4_BC_ONCE: written there already, flow_bc_upfront)
            if (bc_once == 0) flow_batch_bounds(b.data(), i0, i1, [&](size_t i) { return fit(c, i).v; }, bc);
            if (bc[n] > bc[0]) {
                CK(cudaStreamWaitEvent(s, c.ring_fill[(i1 - 1) % R], 0));
                if (bc_once == 0) CK(cudaMemcpyAsync(bcd, bc, (size_t) (n + 1) * 4, cudaMemcpyHostToDevice, s));
                if (!run_set(d, c, L0, c.ring + (i0 % R) * c.ring_stride, 0, c.ring_stride, bcd, bc, 0, n, T, err))
                    return false;
            }
            for (size_t i = i0; i < i1; ++i) CK(cudaEventRecord(c.ring_free[i % R], s));
            kick(s);
            c.n_comp = i1;
            if (!flow_pump(c, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (int64_t) (c.n_comp - lb) / (int64_t) std::max<size_t>(1, n_lay));
            if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        if (bc_once != 0) CK(cudaEventRecord(c.bc_done, s));   // (the layer's regions read by nothing after this)
        while (c.base < c.n_comp && c.items.size() > 4096) {   // (done with: read, copied and used)
            c.items.pop_front();
            ++c.base;
        }
        return true;
    }

    // the experts [i0, i1) of a sorted set (blobs at wbase + i * stride; bounds b on the host, d_bounds on the device)
    // over the chunk's rows, in pieces of at most C rows, each added into acc in pick order
    bool run_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride, const int* d_bounds,
                 const int* b, int i0, int i1, int T, std::string& err) {
        const int D = g.dim, FF = l.ffn;
        while (i0 < i1) {
            int j = i0 + 1;
            while (j < i1 && b[j + 1] - b[i0] <= c.C) ++j;
            const int a = b[i0], nrows = b[j] - a;
            int max_rows = 0;
            for (int i = i0; i < j; ++i) max_rows = std::max(max_rows, b[i + 1] - b[i]);
            if (nrows > 0) {
                x3::SetArgs sa;
                sa.wbase = wbase + (size_t) (i0 - wb0) * stride;   // (wbase: set member wb0's blob)
                sa.stride = stride;
                sa.lay = l.nm > 1 ? l.hlay : l.lay;
                sa.n_exp = j - i0;
                sa.bounds = d_bounds + i0;
                sa.row0 = a;
                sa.max_rows = max_rows;
                sa.n_embd = D;
                sa.n_ff = FF;
                sa.limit = g.lim;
                EkMark ek{};
                const bool ekp = prof_chunk;
                if (ekp) {
                    std::lock_guard<std::mutex> lk(prof_mu);
                    ek.e0 = ek_ev();
                    ek.e1 = ek_ev();
                    ek.e2 = ek_ev();
                    ek.tg = gg::expert_type(sa.lay, 0);
                    ek.td = gg::expert_type(sa.lay, 2);
                    ek.nrows = nrows;
                    ek.nexp = j - i0;
                    ek.maxrows = max_rows;
                    ek.ff = FF;
                    cudaGetDevice(&ek.dev);
                    cudaEventRecord(ek.e0, d.st);
                }
                if (sa.lay.fmt == gg::kFmtMXFP4) {
                    gg::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                    if (ekp) cudaEventRecord(ek.e1, d.st);
                    gg::set_down(sa, c.h16, FF, c.y, D, d.st);
                } else {
                    x3::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                    if (ekp) cudaEventRecord(ek.e1, d.st);
                    x3::set_down(sa, c.h16, FF, c.y, D, d.st);
                }
                if (ekp) {
                    cudaEventRecord(ek.e2, d.st);
                    std::lock_guard<std::mutex> lk(prof_mu);
                    ek_marks.push_back(ek);
                }
                if (dump_y != nullptr && (dump_y_layer < 0 || l.il == dump_y_layer)) {
                    std::vector<int> bb(b + i0, b + j + 1);
                    std::vector<uint16_t> hv((size_t) nrows * FF);
                    std::vector<float> yv((size_t) nrows * D);
                    CK(cudaStreamSynchronize(d.st));
                    CK(cudaMemcpy(hv.data(), c.h16, hv.size() * 2, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(yv.data(), c.y, yv.size() * 4, cudaMemcpyDeviceToHost));
                    std::lock_guard<std::mutex> lk(dump_mu);
                    const int64_t hdr[6] = {l.il, a, nrows, j - i0, wbase == c.ring || (wbase > c.ring &&
                                            wbase < c.ring + (size_t) c.ring_n * c.ring_stride) ? 1 : 0, max_rows};
                    std::fwrite(hdr, 8, 6, dump_y);
                    std::fwrite(bb.data(), 4, bb.size(), dump_y);
                    std::fwrite(hv.data(), 2, hv.size(), dump_y);
                    std::fwrite(yv.data(), 4, yv.size(), dump_y);
                    std::fflush(dump_y);
                }
                moe_combine(c.acc, c.y, c.pos, c.rw, T, l.k_exp, a, a + nrows, D, d.st);
            }
            i0 = j;
        }
        (void) err;
        return true;
    }

    bool load_dense(int di, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        size_t* acct = &d.weights;
        for (Layer& l0 : L)
          for (int mi = 0; mi < l0.nm; ++mi) {
            Layer& l = member(l0, mi);
            if (l.dev != di) continue;
            // (a group's member: its share of the heads (wq_b columns, the sinks), of wo_a's groups and wo_b's rows,
            // of the shared expert's FFN; the rest whole)
            const int nm = l0.nm, na = g.heads / std::max(1, l.ahn), ha = l.ah0 / std::max(1, l.ahn);
            const std::string& P = l.name;
            const std::string A = P + "attn.";
            bool ok = mat(d, A + "wq_a", l.wq_a, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wq_b", l.wq_b, err, acct, ha, na, true)) &&
                      mat(d, A + "wkv", l.wkv, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wo_b", l.wo_b, err, acct, ha, na, false)) &&
                      f32(d, A + "q_norm.weight", l.q_norm, err, acct, (size_t) g.q_lora) &&
                      f32(d, A + "kv_norm.weight", l.kv_norm, err, acct, (size_t) g.hd) &&
                      f32(d, A + "attn_sink", l.sink, err, acct, (size_t) g.heads) &&
                      f32(d, P + "attn_norm.weight", l.attn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "ffn_norm.weight", l.ffn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "hc_attn_fn", l.hca_fn, err, acct) && f32(d, P + "hc_attn_scale", l.hca_scale, err, acct) &&
                      f32(d, P + "hc_attn_base", l.hca_base, err, acct) && f32(d, P + "hc_ffn_fn", l.hcf_fn, err, acct) &&
                      f32(d, P + "hc_ffn_scale", l.hcf_scale, err, acct) &&
                      f32(d, P + "hc_ffn_base", l.hcf_base, err, acct) &&
                      gate_load(d, P + "ffn.gate.weight", l, err, acct) &&
                      f32(d, P + "ffn.gate.bias", l.gate_bias, err, acct, (size_t) l.n_exp) &&
                      (l.mtp || img_tok < 0 || pk.get(P + "ffn.gate.bias_vl") == nullptr ||
                       f32(d, P + "ffn.gate.bias_vl", l.gate_bias_vl, err, acct, (size_t) l.n_exp)) &&
                      mat(d, P + "ffn.shared_experts.w1", l.s1, err, acct, mi, nm, true, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w2", l.s2, err, acct, mi, nm, false, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w3", l.s3, err, acct, mi, nm, true, l.il % nm);
            if (!ok) return false;
            l.sink += (size_t) l.ah0;
            for (int gi = 0; gi < l.ogn; ++gi)
                if (!mat(d, A + "wo_a.slice." + std::to_string(l.og0 + gi), l.wo_a[gi], err, acct)) return false;
            if (l.kv_src) {
                const std::string C = A + "compressor.";
                if (!mat(d, C + "wkv", l.c_wkv, err, acct) || !f32(d, C + "norm.weight", l.c_norm, err, acct)) return false;
                if (l.ratio > 1 && !mat(d, C + "wgate", l.c_wgate, err, acct)) return false;
                if (l.ratio > 1 && (!dalloc(d, l.c_kv, (size_t) l.ratio * g.hd, err) ||
                                    !dalloc(d, l.c_score, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
            if (l.ix_src) {
                const std::string I = A + "indexer.";
                if (!mat(d, I + "wq_b", l.i_wq_b, err, acct) || !f16raw(d, I + "weights_proj.weight", l.i_wproj, err, acct))
                    return false;
                if (l.owns_k && (!mat(d, I + "wk", l.i_wk, err, acct) || !f32(d, I + "k_norm.weight", l.ik_norm, err, acct)))
                    return false;
            }
            if (l.eg >= 0) {
                if (!mat(d, P + "engram.wkv", l.eg_wkv, err, acct)) return false;
                if (pk.get(P + "engram.gate.weight") != nullptr) {   // a GGUF: q_weight * k_weight, folded already
                    if (!f32(d, P + "engram.gate.weight", l.eg_w, err, acct, (size_t) g.hc * g.dim)) return false;
                } else {
                    float *qw = nullptr, *kw = nullptr;
                    if (!f32(d, P + "engram.q_weight", qw, err, acct) || !f32(d, P + "engram.k_weight", kw, err, acct))
                        return false;
                    std::vector<float> a((size_t) g.hc * g.dim), b(a.size());
                    CK(cudaMemcpy(a.data(), qw, a.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(b.data(), kw, b.size() * 4, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];   // only ever used as a product
                    CK(cudaMemcpy(qw, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
                    l.eg_w = qw;
                }
            }
            if (!dalloc(d, l.win, (size_t) g.win * win_rb(), err, &d.caches)) return false;
            if ((g.n_mtp > 0 || o.stage_worker) && !l.mtp) {   // (a worker: its main process's drafts are checked here too)
                if (!dalloc(d, l.ring_bak, (size_t) kWin * win_rb(), err)) return false;
                if (l.kv_src && l.ratio > 1 && (!dalloc(d, l.c_snap, (size_t) 2 * l.ratio * g.hd, err) ||
                                                !dalloc(d, l.c_rows, (size_t) 2 * kWin * g.hd, err)))
                    return false;
            }
            if (!dalloc(d, l.vram_tab, (size_t) l.n_exp, err, acct)) return false;
          }
        // (a remote stage's worker: the final norm alone - its reply's rows are the head's input)
        if (o.stage_worker && di == head_dev && !f32(d, "norm.weight", d.norm_w, err, acct, (size_t) g.dim)) return false;
        if (d.grp == devs[(size_t) head_dev].grp && !o.stage_worker) {   // the head: each member of its group a share of the columns
            const int hn = head_nm();
            if (!mat(d, "head", d.head, err, acct, d.mi, hn, true) || !f32(d, "norm.weight", d.norm_w, err, acct, (size_t) g.dim))
                return false;
            if (g.n_mtp > 0 && !f32(d, "mtp." + std::to_string(g.n_mtp - 1) + ".norm.weight", d.d_norm, err, acct, (size_t) g.dim))
                return false;
            if (!dalloc(d, d.logits_part, (size_t) kWin * ((g.vocab / 128 + hn - 1) / hn * 128), err)) return false;
            if (g.n_mtp > 0 && hn > 1) {   // the markov steps over the group (mk_split): this member's rows of the head
                const Tensor* th = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.head.weight");
                const int R = g.d_rank;
                head_cols(hn, d.mi, d.mk_v0, d.mk_vn);   // (its share of the head's columns: the same rows)
                // F16 as the pack holds it, or a GGUF drafter's BF16 rounded to F16 as f16raw rounds d_mk_head (the same
                // values as the lead's whole rows); another type: no mk_split (the lead's whole rows)
                if (th != nullptr && (th->dtype == "F16" || th->dtype == "BF16") && th->bytes == (uint64_t) g.vocab * R * 2) {
                    std::vector<uint16_t> rows((size_t) d.mk_vn * R);
                    if (!pk.read_at(th->file, th->off + (uint64_t) d.mk_v0 * R * 2, rows.size() * 2, rows.data())) {
                        err = "the markov head: read failed";
                        return false;
                    }
                    if (th->dtype == "BF16")
                        for (uint16_t& h : rows) h = strata::kernels::f16_from_f32(bf16_to_f32(h));
                    if (!dalloc(d, d.mk_share, rows.size(), err, acct) || !dalloc(d, d.mk_erow, (size_t) R, err) ||
                        !dalloc(d, d.mk_ints, (size_t) kWin + 2, err))
                        return false;
                    CK(cudaMemcpy(d.mk_share, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
                }
            }
        }
        if (di == head_dev) {
            head = d.head;
            norm_w = d.norm_w;
            d_norm = d.d_norm;
            if (g.n_mtp > 0) {
                const std::string Z = "mtp." + std::to_string(g.n_mtp - 1) + ".";
                if (!f32(d, Z + "confidence_head.proj.weight", d_conf, err, acct, (size_t) (g.dim + g.d_rank)))
                    return false;
                const Tensor* te = pk.get(Z + "markov_head.embed.weight");
                const Tensor* th = pk.get(Z + "markov_head.head.weight");
                if (te == nullptr || th == nullptr || te->dtype != "BF16" || (th->dtype != "F16" && th->dtype != "BF16")) {
                    err = Z + "markov_head: not the BF16 embed and F16 / BF16 head expected";
                    return false;
                }
                if (!dalloc(d, d_mk_embed, te->bytes / 2, err, acct) || !upload(d, *te, d_mk_embed, err) ||
                    !f16raw(d, Z + "markov_head.head.weight", d_mk_head, err, acct))
                    return false;
            }
        }
        if (g.n_mtp > 0 && di == d_dev) {
            if (!mat(d, "mtp.0.main_proj", d_main_proj, err, acct) ||
                !f32(d, "mtp.0.main_norm.weight", d_main_norm, err, acct, (size_t) g.dim))
                return false;
        }
        return true;
    }

    void set_vm(Layer& l, int e, int slot) {
        uint8_t* b = l.vram_base + (size_t) slot * l.blob;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r) {
                x3::Mat& m = l.vm[(size_t) e][(size_t) r];
                m = x3::Mat{};
                m.K = gg::expert_type(l.lay, r);   // (a low-bit type: its GEMV, ds4_iq's)
                m.fmt = m.K == gg::kTypeMXFP4 ? gg::kFmtMXFP4 : gg::kFmtGG;
                m.raw = b + l.lay.trellis[r];
                m.k = r < 2 ? g.dim : g.ff;
                m.n = r < 2 ? g.ff : g.dim;
            }
            return;
        }
        for (int r = 0; r < 3; ++r) {
            x3::Mat& m = l.vm[(size_t) e][(size_t) r];
            m.trellis = (const uint32_t*) (b + l.lay.trellis[r]);
            m.suh = (const uint16_t*) (b + l.lay.suh[r]);
            m.svh = (const uint16_t*) (b + l.lay.svh[r]);
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
            m.K = l.lay.K[r];
            m.cb = 2;
            m.ld = 0;
        }
    }
    // an expert's VRAM blob -> its CPU-lane part X, and its NH parts -> the blob (the cache's swaps)
    void blob_to_part(const Layer& l, const uint8_t* blob, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = blob + l.lay.trellis[r];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], blob + l.lay.suh[r], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }
    void parts_to_blob(const Layer& l, const std::array<uint8_t*, 4>& parts, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            uint8_t* tr = blob + l.lay.trellis[r];
            for (int X = 0; X < NH; ++X)
                for (int sl = 0; sl < k / 16; ++sl)
                    std::memcpy(tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                                parts[(size_t) X] + l.hl.trellis[r] + (size_t) sl * nth * tile, (size_t) nth * tile);
            std::memcpy(blob + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, parts[(size_t) X] + l.hl.svh[r],
                            (size_t) l.hl.n[r] * 2);
        }
    }

    // an expert's blob -> member m's part (gate / up columns [ff0, ff0 + ffn), down rows of the same), and back
    void blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_blob_to_half(l, blob, m, dst);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(dst + x.hlay.trellis[r] + sl * nth * tile,
                            blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile, nth * tile);
            std::memcpy(dst + x.hlay.suh[r], blob + l.lay.suh[r], (size_t) g.dim * 2);
            std::memcpy(dst + x.hlay.svh[r], blob + l.lay.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(dst + x.hlay.trellis[2], blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, (size_t) (fn / 16) * row);
        std::memcpy(dst + x.hlay.suh[2], blob + l.lay.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        std::memcpy(dst + x.hlay.svh[2], blob + l.lay.svh[2], (size_t) g.dim * 2);
    }
    void half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_half_to_blob(l, src, m, blob);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile,
                            src + x.hlay.trellis[r] + sl * nth * tile, nth * tile);
            if (m == 0) std::memcpy(blob + l.lay.suh[r], src + x.hlay.suh[r], (size_t) g.dim * 2);
            std::memcpy(blob + l.lay.svh[r] + (size_t) f0 * 2, src + x.hlay.svh[r], (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, src + x.hlay.trellis[2], (size_t) (fn / 16) * row);
        std::memcpy(blob + l.lay.suh[2] + (size_t) f0 * 2, src + x.hlay.suh[2], (size_t) fn * 2);
        if (m == 0) std::memcpy(blob + l.lay.svh[2], src + x.hlay.svh[2], (size_t) g.dim * 2);
    }

    bool place_experts(std::string& err) {
        // (Windows, an integrated GPU: what its experts may commit - commit_room - and what that kept off the GPU)
        double room = 1e300, kept = 0;
        for (const Dev& d : devs)
            if (commit_charged(d)) room = commit_room();
        const double room0 = room;
        // per group: what its tightest member has left (each member holds its part of every VRAM expert of the
        // group's layers, its unit bytes a slot - counted at the members' mean: an uneven split's odd blocks go round
        // the members layer by layer, so each member's total comes to about the mean's)
        auto ucount = [](const Layer& l) {
            double s = 0;
            for (int m = 0; m < l.nm; ++m) s += (double) unit(member(l, m));
            return s / l.nm;
        };
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            double budget = 1e300;
            bool charged = false;
            for (int m : groups[gi].mem) {
                cudaSetDevice(devs[(size_t) m].id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                budget = std::min(budget, std::max(0.0, ((double) fr - (double) ((size_t) o.reserve_mib << 20))) *
                                              o.vram_experts);
                charged = charged || commit_charged(devs[(size_t) m]);
            }
            if (charged && room < 1e300) {   // (each member's slots charged: the group's share of the room)
                double all = 0;
                for (const Layer& l : L)
                    if (l.grp == (int) gi) all += (double) unit(l) * l.n_exp;
                const double nm = (double) groups[gi].mem.size(), capped = std::min(budget, room / nm);
                kept += std::max(0.0, std::min(budget, all) - std::min(capped, all)) * nm;
                budget = capped;
                room = std::max(0.0, room - std::min(budget, all) * nm);
            }
            double budget_t = budget;   // the drafter's experts first, all of them (every draft uses its blocks)
            for (Layer& l : L)
                if (l.grp == (int) gi && l.mtp) {
                    l.n_vram = std::min(l.n_exp, (int) std::floor(budget_t / (double) ucount(l)));
                    budget_t -= (double) l.n_vram * ucount(l);
                }
            double want = 0;
            for (const Layer& l : L)
                if (l.grp == (int) gi && !l.mtp && !l.cpu_only) want += (double) ucount(l) * l.n_exp;
            const double frac = want > 0 ? std::min(1.0, std::max(0.0, budget_t) / want) : 0.0;
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.mtp) continue;
                if (l.cpu_only) {   // (its type has no device kernel: the CPU lane reads every expert of it)
                    l.n_vram = 0;
                    continue;
                }
                l.n_vram = std::min(l.n_exp, (int) std::floor(frac * l.n_exp));
                // never more than fits: the floor of each layer's share can still round up the total
                while (l.n_vram > 0 && (double) ucount(l) * l.n_vram > budget) --l.n_vram;
            }
            double used = 0;
            for (Layer& l : L)
                if (l.grp == (int) gi) used += (double) ucount(l) * l.n_vram;
            // what the floors left: whole experts, layer by layer
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.cpu_only) continue;
                while (l.n_vram < l.n_exp && used + (double) ucount(l) <= budget) {
                    ++l.n_vram;
                    used += (double) ucount(l);
                }
            }
        }
        if (kept > 0) {   // the commit kept experts off the GPU: say so, and the page file that keeps them on it
            const os::Commit c = os::commit();
            const double gib = (double) (1ull << 30);
            const double pf = c.limit != ~uint64_t{0} && c.ram != ~uint64_t{0} && c.limit > c.ram ? (double) (c.limit - c.ram) : 0.0;
            commit_kept = kept;
            pagefile_hint = std::ceil((pf + kept) / gib);
            std::fprintf(stderr,
                         "ds4: WARNING: %.1f GB of experts stay off the GPU - Windows charges the GPU's memory to its "
                         "commit (RAM + page file) and can commit only %.1f GB more for them (%.0f GB kept for the "
                         "other programs). A page file of %.0f GB (%.0f MB; now %.0f GB) keeps them on the GPU: System "
                         "> About > Advanced system settings > Performance > Advanced > Virtual memory, then restart\n",
                         kept / 1e9, room0 / 1e9, commit_margin() / 1e9, pagefile_hint, pagefile_hint * 1024.0, pf / gib);
        } else if (room0 < 1e300 && o.verbose) {
            std::fprintf(stderr, "ds4: Windows' commit has room for the GPU's experts (%.1f GB beside %.0f GB for the other "
                                 "programs)\n", room0 / 1e9, commit_margin() / 1e9);
        }
        for (Layer& l : L) {
            if (elsewhere(l)) continue;
            // the hottest experts of the profile first, then by index
            std::vector<int> order((size_t) l.n_exp);
            std::iota(order.begin(), order.end(), 0);
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
            l.slot.assign((size_t) l.n_exp, -1);
            l.host.assign((size_t) l.n_exp, nullptr);
            // a layer with experts on both sides keeps one VRAM slot free for the cache's swaps
            l.n_slots = std::min(l.n_vram, l.n_exp);
            if (o.expert_cache && l.n_slots >= 2 && l.n_slots < l.n_exp) {   // (every expert fits: nothing to swap)
                l.n_vram = l.n_slots - 1;
                l.spare_vram = l.n_slots - 1;
            } else {
                l.n_vram = l.n_slots;
                l.spare_vram = -1;
            }
            for (int i = 0; i < l.n_vram; ++i) l.slot[(size_t) order[(size_t) i]] = i;
            for (int m = 0; m < l.nm && l.n_slots > 0; ++m) {
                Layer& x = member(l, m);
                Dev& d = devs[(size_t) x.dev];
                cudaSetDevice(d.id);
                void* v = nullptr;
                if (cudaMalloc(&v, (size_t) l.n_slots * unit(x)) != cudaSuccess) {
                    err = "cudaMalloc of layer " + std::to_string(l.il) + "'s " + std::to_string(l.n_slots) +
                          " experts on CUDA" + std::to_string(d.id) + " failed";
                    return false;
                }
                d.allocs.push_back(v);
                d.experts += (size_t) l.n_slots * unit(x);
                x.vram_base = (uint8_t*) v;
            }
            l.heat.assign((size_t) l.n_exp, 0.0f);
            {   // the profile as the starting heat: its counts scaled to ~512 hits a layer
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                const double sc = sum > 0 ? 512.0 / sum : 0.0;
                for (int e = 0; e < l.n_exp; ++e) l.heat[(size_t) e] = (float) (p[e] * sc);
            }
            const int n_host = l.n_exp - l.n_vram;
            {   // the CPU lane's parts of an expert (in place: the whole expert, its pieces where the file has them)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : g.ff, nf = r < 2 ? g.ff : g.dim, n = nf / NH;
                    l.hl.k[r] = k;
                    l.hl.n[r] = n;
                    l.hl.trellis[r] = file_experts ? l.piece_src[r][0] : off;
                    off += align256((size_t) (k / 16) * (n / 16) * 32 * l.lay.K[r]);
                    l.hl.suh[r] = file_experts ? l.piece_src[r][1] : off;
                    off += align256((size_t) k * 2);
                    l.hl.svh[r] = file_experts ? l.piece_src[r][2] : off;
                    off += align256((size_t) n * 2);
                }
                l.hl.bytes = off;
            }
            l.hh.assign((size_t) l.n_exp, {nullptr, nullptr, nullptr, nullptr});
            if (l.lay.fmt == gg::kFmtMXFP4) {   // a GGUF's: read in place from the file's mapping
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = (uint8_t*) mx_mat(l, e, 0);
            } else if (file_experts) {   // every expert's file mapped (the cache's swaps hand VRAM experts back to the files)
                for (int e = 0; e < l.n_exp; ++e)
                    if (pk.map(l.region_file[(size_t) e]) == nullptr) {
                        err = "mapping " + pk.files[(size_t) l.region_file[(size_t) e]] + " failed";
                        return false;
                    }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) {
                        l.hh[(size_t) e][0] = file_part(l, e);
                        l.host[(size_t) e] = l.hh[(size_t) e][0];
                    }
            } else if (n_host > 0) {
                const int n_parts = n_host + (l.spare_vram >= 0 ? 1 : 0);   // (+ the cache's spare)
                const size_t bytes = (size_t) n_parts * l.hl.bytes;
                for (int X = 0; X < NH; ++X) {
                    void* m = node_map(bytes, X, NH);
                    if (m == nullptr) {
                        err = "host memory for layer " + std::to_string(l.il) + "'s " + std::to_string(n_host) +
                              " CPU experts (" + std::to_string(bytes >> 20) + " MiB on node " + std::to_string(X) + ")";
                        return false;
                    }
                    host_maps.emplace_back(m, bytes);
                    int hi = 0;
                    for (int e = 0; e < l.n_exp; ++e)
                        if (l.slot[(size_t) e] < 0) l.hh[(size_t) e][(size_t) X] = (uint8_t*) m + (size_t) (hi++) * l.hl.bytes;
                    if (l.spare_vram >= 0) l.spare_host[(size_t) X] = (uint8_t*) m + (size_t) hi * l.hl.bytes;
                }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = l.hh[(size_t) e][0];
            }
            l.vm.assign((size_t) l.n_exp, {});
            if (l.nm == 1)   // (the host-routed window path's views: a group's members take the device-routed path)
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0) set_vm(l, e, l.slot[(size_t) e]);
            if (!halloc(l.tab_pin, (size_t) 2 * l.nm, err)) return false;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0)
                        tab[(size_t) e] = (unsigned long long) (x.vram_base + (size_t) l.slot[(size_t) e] * unit(x));
                cudaSetDevice(devs[(size_t) x.dev].id);
                CK(cudaMemcpy(x.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice));
            }
        }
        return true;
    }

    // a GGUF's VRAM experts, in file order, by a few threads through pinned staging blobs (the CPU lane's stay in the
    // file: the mapping reads them on first use)
    bool load_experts_mx(std::string& err) {
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        size_t maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < 12; ++t)
            th.emplace_back([&] {
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    bool ok = stage != nullptr && stage2 != nullptr && read_mx(l, e, stage);
                    for (int m = 0; m < l.nm && ok; ++m) {   // (a group: each member its part)
                        const Layer& x = member(l, m);
                        cudaSetDevice(devs[(size_t) x.dev].id);
                        if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                        ok = cudaMemcpy(x.vram_base + (size_t) l.slot[(size_t) e] * unit(l), l.nm > 1 ? stage2 : stage,
                                        unit(l), cudaMemcpyHostToDevice) == cudaSuccess;
                    }
                    if (!ok) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "loading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        // as load_experts in place: the VRAM experts' pages leave the cache, the most-routed of the others (the
        // profile's share of their layer, then by index) are read once, up to warm_bytes, so the cache starts with them
        struct Warm {
            int layer, e;
            double share;
        };
        std::vector<Warm> warm;
        for (const Layer& l : L) {
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            double sum = 0;
            for (int e = 0; e < l.n_exp; ++e) sum += p[e];
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0) {
                    for (int r = 0; r < 3; ++r)
                        os::advise(pk.fds[(size_t) l.mx_file[r]], (l.mx_off[r] + (uint64_t) e * l.mx_stride[r]), mx_part_bytes(l, r), false);
                } else {
                    warm.push_back({l.il, e, sum > 0 ? p[e] / sum : 0.0});
                }
            }
        }
        std::stable_sort(warm.begin(), warm.end(), [](const Warm& a, const Warm& b) { return a.share > b.share; });
        if (!ram_tier && !warm.empty()) {   // copied into RAM (copy_experts decides), the rest read ahead into what is left
            std::vector<std::pair<int, int>> hot;
            for (const Warm& w : warm) hot.push_back({w.layer, w.e});
            if (!copy_experts(hot, err)) return false;
            if (copy_mode) {
                warm.erase(std::remove_if(warm.begin(), warm.end(),
                                          [&](const Warm& w) { return L[(size_t) w.layer].ram[(size_t) w.e] >= 0; }),
                           warm.end());
                const uint64_t av = ram_avail();
                if (av != ~uint64_t{0}) warm_bytes = av > kRamMargin ? std::min<uint64_t>(warm_bytes, av - kRamMargin) : 0;
            }
        }
        if (ram_tier && warm_bytes > 0 && !warm.empty()) {
            // chunks of 32 slots up to warm_bytes (all the RAM the rest leaves, kRamMargin kept), one slot kept free for
            // a move in; the hottest experts read in, their pages then dropped from the page cache.  Plain memory, not
            // pinned: only the CPU reads it (what goes to a GPU passes the pinned staging), and pinned memory is capped on
            // Windows - the engine's own pinned buffers come after this
            const size_t want = std::min(warm.size(), (size_t) (warm_bytes / maxb));
            while (ram_slots.size() < want + 1) {
                const size_t n = std::min<size_t>(32, want + 1 - ram_slots.size());
                uint8_t* c = (uint8_t*) os::anon_alloc(n * maxb, -1, 1);
                if (c == nullptr) break;
                ram_chunks.push_back(c);
                ram_chunk_bytes.push_back(n * maxb);
                for (size_t i = 0; i < n; ++i) ram_slots.push_back(c + i * maxb);
            }
            ram_held = ram_slots.empty() ? 0 : (int) ram_slots.size() - 1;
            for (Layer& l : L)
                if (l.lay.fmt == gg::kFmtMXFP4) l.ram.assign((size_t) l.n_exp, -1);
            std::atomic<size_t> ri{0};
            std::atomic<bool> rfail{false};
            std::vector<std::thread> rt;
            for (int t = 0; t < 12 && ram_held > 0; ++t)
                rt.emplace_back([&] {
                    for (size_t i; (i = ri.fetch_add(1)) < (size_t) ram_held;) {
                        const Layer& l = L[(size_t) warm[i].layer];
                        if (!read_mx_file(l, warm[i].e, ram_slots[i])) rfail.store(true);
                        for (int r = 0; r < 3; ++r)
                            os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) warm[i].e * l.mx_stride[r],
                                       mx_part_bytes(l, r), false);
                    }
                });
            for (auto& t : rt) t.join();
            if (rfail.load()) {
                err = "reading the RAM tier's experts from the model's files failed";
                return false;
            }
            for (int i = 0; i < ram_held; ++i) L[(size_t) warm[(size_t) i].layer].ram[(size_t) warm[(size_t) i].e] = i;
            if (ram_held > 0) ram_free.push_back(ram_held);
            if (o.expert_cache && ram_held > 0) ram_th = std::thread([this] { ram_loop(); });
            if (o.verbose)
                std::fprintf(stderr, "ds4: RAM tier: %d experts (%.1f GB of RAM), the hottest after VRAM's; the other %zu "
                                     "read from the model's files (%.1f s)\n",
                             ram_held, ram_slots.size() * (double) maxb / 1e9, warm.size() - (size_t) ram_held,
                             (now_ms() - t0) / 1000.0);
            return true;
        }
        size_t nw = 0;   // (DS4_IQ_WARM=split: none here - warm_split, after the upload)
        for (double b = 0; !iq_warm_split && nw < warm.size() && b + (double) L[(size_t) warm[nw].layer].blob <= (double) warm_bytes; ++nw)
            b += (double) L[(size_t) warm[nw].layer].blob;
        std::atomic<size_t> wi{0};
        std::vector<std::thread> wt;
        for (int t = 0; t < 12 && nw > 0; ++t)
            wt.emplace_back([&] {
                std::vector<uint8_t> buf(maxb);
                for (size_t i; (i = wi.fetch_add(1)) < nw;) read_mx(L[(size_t) warm[i].layer], warm[i].e, buf.data());
            });
        for (auto& t : wt) t.join();
        if (o.verbose && nw > 0)
            std::fprintf(stderr, "ds4: %zu CPU-lane experts read ahead (%.1f GB) in %.1f s\n", nw,
                         nw * (double) L[(size_t) lo].blob / 1e9, (now_ms() - t0) / 1000.0);
        return true;
    }

    // a GGUF's CPU-lane experts copied into RAM (copy_mode), decided once VRAM's are in - on a unified-memory APU the GPU
    // has taken its share of the RAM by then, so the copies only get what it leaves.  `hot`: the CPU lane's experts, the
    // most-routed first.  --mmap-experts off: all that fit beside kRamMargin (a warning if not all do); on and auto:
    // none - read in place: the page cache follows a conversation's experts where a static set (the profile's
    // hottest) does not (the RAM tier's lesson), and copies that all fit were no faster; DS4_COPY_GB=<n>: the
    // hottest n GB (the static hybrid, for tests).  Each copy in NH parts on the NUMA nodes (DS4_COPY_NUMA=1: one), read from the files in file
    // order by a few threads, anonymous memory on huge pages (Windows: plain), not pinned (only the CPU reads it; what
    // goes to a GPU passes the pinned staging) - a layer that cannot get its memory keeps its experts in place
    bool copy_experts(const std::vector<std::pair<int, int>>& hot, std::string& err) {
        if (o.mmap_experts > 0 || hot.empty()) return true;
        const double t0 = now_ms();
        auto spare_of = [&](const Layer& l) { return o.expert_cache && l.spare_vram >= 0 ? 1 : 0; };
        double all = 0;
        {
            std::vector<char> seen(L.size(), 0);
            for (const auto& h : hot) {
                const Layer& l = L[(size_t) h.first];
                all += (double) l.blob * (1 + (seen[(size_t) l.il]++ == 0 ? spare_of(l) : 0));
            }
        }
        const uint64_t avail = ram_avail();
        // (Windows: the copies are committed memory, within what its commit still has beside the other programs' margin)
        const double can = std::min(avail == ~uint64_t{0} ? 1e300 : std::max(0.0, (double) avail - (double) kRamMargin),
                                    commit_room());
        // (auto: none - copies that all fit were measured no faster than the page cache: the CPU lane's rows are
        // compute-bound, ~46 GB/s from either on 2 Xeon sockets, NUMA parts included; a part copied cost decode speed
        // on 64 GB and on 32 GB of RAM)
        double budget = o.mmap_experts == 0 ? std::min(all, can) : 0.0;
        if (const char* v = std::getenv("DS4_COPY_GB")) budget = std::atof(v) * 1e9;
        if (o.mmap_experts == 0 && all > can)
            std::fprintf(stderr, "ds4: --mmap-experts off: the CPU lane's experts (%.1f GB) do not all fit the RAM this "
                                 "process can get (%.1f GB beside %.0f GB for the rest) - the most-routed copied, the "
                                 "others read in place\n", all / 1e9, can / 1e9, kRamMargin / 1e9);
        if (budget <= 0) {
            if (o.verbose)
                std::fprintf(stderr, "ds4: the CPU lane's experts (%.1f GB) stay in place: the RAM this process can get "
                                     "(%.1f GB beside %.0f GB for the rest) does not hold them all\n",
                             all / 1e9, can / 1e9, kRamMargin / 1e9);
            return true;
        }
        // NUMA parts: whole rows of each matrix, so any row count the nodes divide
        int nh = (int) std::min<size_t>(2, std::max<size_t>(1, cores.size()));
        if (const char* v = std::getenv("DS4_COPY_NUMA"))   // (0 / 1: one part; 2: two, even on one node - a test)
            nh = std::atoi(v) >= 2 ? 2 : 1;
        if (g.ff % nh != 0 || g.dim % nh != 0) nh = 1;
        NH = nh;
        cp_lay.assign(L.size(), {0, 0, 0, 0});
        for (const Layer& l : L) {
            if (elsewhere(l)) continue;
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                cp_lay[(size_t) l.il][(size_t) r] = off;
                off += align256((size_t) (mx_n(r) / NH) * mx_row_b(l, r));
            }
            cp_lay[(size_t) l.il][3] = off;
        }
        // the hottest that fit the budget (a layer's first one brings its spare)
        std::vector<int> n_copy(L.size(), 0);
        double used = 0;
        size_t take = 0;
        for (; take < hot.size(); ++take) {
            const Layer& l = L[(size_t) hot[take].first];
            const double add = (double) l.blob * (1 + (n_copy[(size_t) l.il] == 0 ? spare_of(l) : 0));
            if (used + add > budget) break;
            used += add;
            ++n_copy[(size_t) l.il];
        }
        // each layer's slots: one block per node of (copies + spare) parts
        std::vector<int> first(L.size(), -1);
        copy_spare.assign(L.size(), -1);
        for (Layer& l : L) {
            l.ram.assign((size_t) l.n_exp, -1);
            const int ns = n_copy[(size_t) l.il] + (n_copy[(size_t) l.il] > 0 ? spare_of(l) : 0);
            if (ns == 0) continue;
            const size_t part = cp_lay[(size_t) l.il][3], bytes = (size_t) ns * part;
            std::array<uint8_t*, 2> base = {nullptr, nullptr};
            bool ok = true;
            for (int X = 0; X < NH && ok; ++X) {
                base[(size_t) X] = (uint8_t*) node_map(bytes, X, NH);
                ok = base[(size_t) X] != nullptr;
                if (ok) {
                    ram_chunks.push_back(base[(size_t) X]);
                    ram_chunk_bytes.push_back(bytes);
                }
            }
            if (!ok) {
                std::fprintf(stderr, "ds4: no RAM for layer %d's %d copied experts (%.1f GB): they stay in place\n", l.il,
                             ns, (double) bytes * NH / 1e9);
                n_copy[(size_t) l.il] = 0;
                continue;
            }
            first[(size_t) l.il] = (int) ram_part.size();
            for (int i = 0; i < ns; ++i)
                ram_part.push_back({base[0] + (size_t) i * part, NH > 1 ? base[1] + (size_t) i * part : nullptr});
            if (spare_of(l)) copy_spare[(size_t) l.il] = first[(size_t) l.il] + ns - 1;
        }
        struct Job {
            int layer, e, slot;
        };
        std::vector<Job> jobs;
        {
            std::vector<int> next = first;
            for (size_t i = 0; i < take; ++i) {
                const int il = hot[i].first;
                if (n_copy[(size_t) il] == 0) continue;
                jobs.push_back({il, hot[i].second, next[(size_t) il]++});
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {   // (file order: the disk reads ahead)
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> ji{0};
        std::atomic<bool> bad{false};
        std::atomic<int64_t> done_b{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 12 && !jobs.empty(); ++t)
            th.emplace_back([&] {
                for (size_t i; (i = ji.fetch_add(1)) < jobs.size() && !bad.load();) {
                    const Job& j = jobs[i];
                    const Layer& l = L[(size_t) j.layer];
                    for (int r = 0; r < 3; ++r) {
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X)
                            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) j.e * l.mx_stride[r] + (uint64_t) X * nb, nb,
                                            ram_part[(size_t) j.slot][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r]))
                                bad.store(true);
                    }
                    const int64_t db = done_b.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && i % 2048 == 0)
                        std::fprintf(stderr, "ds4: copying experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
            });
        for (auto& t : th) t.join();
        if (bad.load()) {
            err = "copying the CPU lane's experts from the model's files failed";
            return false;
        }
        for (const Job& j : jobs) L[(size_t) j.layer].ram[(size_t) j.e] = j.slot;
        copy_mode = true;
        ram_held = (int) jobs.size();
        double cb = 0;
        for (const Job& j : jobs) cb += (double) L[(size_t) j.layer].blob;
        std::fprintf(stderr, "ds4: %zu of the CPU lane's %zu experts copied into RAM (%.1f GB, %d NUMA part%s) in %.1f s%s\n",
                     jobs.size(), hot.size(), cb / 1e9, NH, NH > 1 ? "s" : "", (now_ms() - t0) / 1000.0,
                     jobs.size() < hot.size() ? "; the others read in place" : "");
        return true;
    }

    bool load_experts(std::string& err) {
        if (!L.empty() && L[(size_t) lo].lay.fmt == gg::kFmtMXFP4) return load_experts_mx(err);
        // every expert's region, in file order, by a few threads; VRAM ones through a pinned staging blob.  In place
        // (file_experts) the CPU lane's experts are not copied: the most-routed of them (the profile's share of their
        // layer) are read once, up to warm_bytes, so the page cache starts with them, and the VRAM ones' pages are
        // dropped from it again
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        for (const Layer& l : L)
            for (int e = 0; e < l.n_exp; ++e)
                if (!file_experts || l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        uint64_t warm = 0;
        if (file_experts) {
            std::vector<std::pair<double, Job>> cand;
            for (const Layer& l : L) {
                const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) cand.push_back({sum > 0 ? p[e] / sum : 0.0, Job{l.il, e}});
            }
            std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (const auto& c : cand) {
                const uint64_t b = L[(size_t) c.second.layer].region_bytes[(size_t) c.second.e];
                if (warm + b > warm_bytes) break;
                warm += b;
                jobs.push_back(c.second);
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.region_file[(size_t) a.e] != lb.region_file[(size_t) b.e])
                return la.region_file[(size_t) a.e] < lb.region_file[(size_t) b.e];
            return la.region_off[(size_t) a.e] < lb.region_off[(size_t) b.e];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        const int nth = 12;
        size_t maxr = 0, maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (uint64_t r : l.region_bytes) maxr = std::max<size_t>(maxr, r);
        }
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&] {
                std::vector<uint8_t> region(maxr);
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                int cur_dev = -1;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    if (!pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], l.region_bytes[(size_t) e],
                                    region.data())) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "reading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    if (l.slot[(size_t) e] >= 0) {
                        repack(l, region.data(), stage);
                        bool up = true;
                        for (int m = 0; m < l.nm && up; ++m) {   // (a group: each member its part)
                            const Layer& x = member(l, m);
                            const Dev& d = devs[(size_t) x.dev];
                            if (cur_dev != d.id) {
                                cudaSetDevice(d.id);
                                cur_dev = d.id;
                            }
                            if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                            up = cudaMemcpy(x.vram_base + (size_t) l.slot[(size_t) e] * unit(x), l.nm > 1 ? stage2 : stage,
                                            unit(x), cudaMemcpyHostToDevice) == cudaSuccess;
                        }
                        if (!up) {
                            std::lock_guard<std::mutex> lk(emu);
                            ferr = "uploading layer " + std::to_string(l.il) + " expert " + std::to_string(e);
                            failed.store(true);
                            break;
                        }
                        if (file_experts)
                            os::advise(pk.fds[(size_t) l.region_file[(size_t) e]], l.region_off[(size_t) e], l.region_bytes[(size_t) e], false);
                    } else if (!file_experts) {
                        for (int X = 0; X < NH; ++X) repack_part(l, region.data(), X, l.hh[(size_t) e][(size_t) X]);
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.region_bytes[(size_t) e]) +
                                       (int64_t) l.region_bytes[(size_t) e];
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(),
                                     db / 1e9, db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        if (file_experts)
            std::fprintf(stderr, "ds4: experts loaded in %.1f s (%.1f GB of the CPU lane's read ahead)\n",
                         (now_ms() - t0) / 1000.0, warm / 1e9);
        return true;
    }

    std::string describe() const {
        std::ostringstream s;
        for (size_t di = 0; di < devs.size(); ++di) {
            const Dev& d = devs[di];
            int a = -1, b = -1, ev = 0, et = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di) {
                        if (a < 0) a = l.il;
                        b = l.il;
                        ev += l.n_vram;
                        et += l.n_exp;
                    }
            size_t fr = 0, tot = 0;
            cudaSetDevice(d.id);
            cudaMemGetInfo(&fr, &tot);
            const size_t gsz = groups.empty() ? 1 : groups[(size_t) d.grp].mem.size();
            char tp[64] = "";
            if (gsz > 1) std::snprintf(tp, sizeof tp, " (tensor parallel %zu, part %d)", gsz, d.mi);
            char line[384];
            std::snprintf(line, sizeof line,
                          "ds4: CUDA%d layers [%d, %d]%s%s: weights %.2f GB, experts %.2f GB (%d of %d in VRAM), caches "
                          "%.2f GB, %.0f MiB free\n",
                          d.id, a, b, tp, (int) di != head_dev ? "" : o.stage_worker ? " + the link's rows" : " + head",
                          d.weights / 1e9, d.experts / 1e9, ev, et,
                          d.caches / 1e9, fr / 1048576.0);
            s << line;
        }
        int nh = 0;
        size_t hb_ = 0;
        for (const Layer& l : L) {
            nh += l.n_exp - l.n_vram;
            hb_ += (size_t) (l.n_exp - l.n_vram) * l.blob;
        }
        char line[200];
        if (copy_mode)
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB): %d copied into RAM (%d NUMA "
                                             "parts), the rest in place in the model's files; %d threads\n",
                          nh, hb_ / 1e9, ram_held, NH, cpu_threads());
        else if (file_experts)
            std::snprintf(line, sizeof line,
                          "ds4: %d routed experts on the CPU lane (%.1f GB, in place in the model's files), %d threads\n",
                          nh, hb_ / 1e9, cpu_threads());
        else
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB of RAM, %d NUMA parts), %d threads\n", nh,
                          hb_ / 1e9, NH, cpu_threads());
        s << line;
        return s.str();
    }

    double store_gb() const {
        static double cached = -2.0;   // (decided once: the layout of the blobs depends on it)
        if (cached > -1.5) return cached;
        double gb = o.ssd_cache_gb;
        if (const char* v = std::getenv("DS4_STORE_GB")) gb = std::atof(v);
        if (!pk.gguf) gb = 0.0;
        else if (gb < 0.0) {   // auto: the RAM free now less 12 GB for the system and the owner's programs, 48 GB at most
            const uint64_t avail = ram_avail();
            const double room = avail == ~uint64_t{0} ? 0.0 : (double) avail / 1e9 - 12.0;
            gb = room >= 4.0 ? std::min(48.0, room) : 0.0;
        }
        return cached = gb;
    }
    void init_store() {
        const double gb = store_gb();
        if (gb <= 0.0 || L.empty() || L[(size_t) lo].lay.fmt != gg::kFmtMXFP4 || !file_experts) return;
        std::vector<ExpertStore::LayerGeo> geo((size_t) g.nl);   // (the trunk's layers: the drafter's bigger experts stay in place)
        for (const Layer& l : L) {
            if (l.il >= g.nl || elsewhere(l)) continue;   // (another process's: no experts here)
            ExpertStore::LayerGeo& gl = geo[(size_t) l.il];
            gl.n_exp = l.n_exp;
            for (int r = 0; r < 3; ++r) {
                gl.file[r] = l.mx_file[r];
                gl.off[r] = l.mx_off[r];
                gl.stride[r] = l.mx_stride[r];
                gl.bytes[r] = mx_part_bytes(l, r);
                gl.tr[r] = l.lay.trellis[r];
            }
            gl.blob = l.blob;
        }
        ExpertStore::Cfg cfg;
        cfg.budget_bytes = (size_t) (gb * 1e9);
        cfg.threads = std::getenv("DS4_STORE_THREADS") ? std::atoi(std::getenv("DS4_STORE_THREADS")) : 8;
        cfg.floor_bytes = (size_t) (std::getenv("DS4_STORE_FLOOR_GB") ? std::atof(std::getenv("DS4_STORE_FLOOR_GB")) : 8.0) << 30;
        cfg.policy = (ExpertStore::Policy) (std::getenv("DS4_STORE_POLICY") ? std::atoi(std::getenv("DS4_STORE_POLICY")) : o.ssd_policy);
        cfg.ram_avail = [] { return ram_avail(); };
        cfg.reserve = (size_t) (std::getenv("DS4_STORE_RESERVE") ? std::atoi(std::getenv("DS4_STORE_RESERVE")) : 192);
        cfg.vis_percent = std::getenv("DS4_STORE_ZC_FRAC") ? std::atoi(std::getenv("DS4_STORE_ZC_FRAC")) : 50;
        // (the GPU computing the store's experts over PCIe is opt-in, DS4_STORE_ZC=1: measured slower than the CPU lane alone on a
        // 5070 / PCIe 5.0 - the staging copy sits in the layer's critical path - and its answers depend on the timing of which lane
        // took which expert, the CPU lane's are the mmap path's byte for byte)
        const bool want_zc = std::getenv("DS4_STORE_ZC") && std::atoi(std::getenv("DS4_STORE_ZC")) != 0 && devs.size() == 1 &&
                             L[(size_t) lo].nm == 1;
        if (want_zc) {   // the arena's chunks registered for the GPU to read in place (zero-copy over PCIe)
            const int dev0 = devs[0].id;
            cfg.map_chunk = [dev0](void* h, size_t n) -> void* {
                cudaSetDevice(dev0);
                if (cudaHostRegister(h, n, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
                    (void) cudaGetLastError();
                    return nullptr;
                }
                void* d = nullptr;
                if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
                    (void) cudaGetLastError();
                    cudaHostUnregister(h);
                    return nullptr;
                }
                if (d != h) std::fprintf(stderr, "ds4: RAM store: chunk host %p -> device %p (not the identity)\n", h, d);
                return d;
            };
            cfg.unmap_chunk = [dev0](void* h) {
                cudaSetDevice(dev0);
                cudaHostUnregister(h);
            };
        }
        bool direct = true;
        for (os::fd_t f : pk.dfds) direct = direct && f != os::kBadFd;
        auto st = std::make_unique<ExpertStore>();
        if (!st->init(cfg, geo, pk.dfds, pk.pfds)) {
            std::fprintf(stderr, "ds4: --ssd-cache: the store could not start\n");
            return;
        }
        store = std::move(st);
        // (the router lookahead's guesses are not read ahead with the store: measured 4.5 against 4.7 tok/s, 62% of the guesses are right and
        // the SSD is the narrow part; DS4_AHEAD=1 turns them on)
        if (std::getenv("DS4_AHEAD") == nullptr) ahead_mode = 0;
        for (Layer& l : L)
            if (l.il < g.nl && want_zc) l.zc.assign((size_t) l.n_exp, 0);
        std::fprintf(stderr, "ds4: RAM store: up to %zu slots of %.1f MB (%.1f GB), policy %d, %d reader threads, %s reads\n",
                     store->slots_max(), store->slot_bytes() / 1e6, store->slots_max() * (double) store->slot_bytes() / 1e9,
                     (int) cfg.policy, cfg.threads, direct ? "unbuffered" : "BUFFERED (no direct handle)");
        if (want_zc) {
            if (!store->gpu_mapped()) {
                std::fprintf(stderr, "ds4: RAM store: the arena could not be mapped for the GPU: zero-copy off\n");
                for (Layer& l : L) l.zc.clear();
            } else {
                zc_stride = (store->max_blob() + 4095) & ~(size_t) 4095;
                cudaSetDevice(devs[0].id);
                if (cudaMalloc((void**) &zc_stage_buf, zc_stride * kPicks) != cudaSuccess) {
                    (void) cudaGetLastError();
                    zc_stage_buf = nullptr;
                    for (Layer& l : L) l.zc.clear();
                } else {
                    for (Layer& l : L)
                        if (!l.zc.empty() && cudaMalloc((void**) &l.zc_tab, (size_t) l.n_exp * 8) == cudaSuccess) cudaMemset(l.zc_tab, 0, (size_t) l.n_exp * 8);
                        else if (!l.zc.empty()) {
                            (void) cudaGetLastError();
                            l.zc.clear();
                        }
                }
            }
            std::fprintf(stderr, "ds4: RAM store: the GPU takes its slots over PCIe (%s), %zu slots kept evictable\n",
                         zc_stage_buf != nullptr ? "copied into a staging area at the PCIe rate first" : "off", cfg.reserve);
        }
    }
    // the profile's hottest experts (VRAM's left out) read into the store before the first token (DS4_STORE_PRELOAD=0: not)
    void preload_store() {
        if (!store) return;
        if (std::getenv("DS4_STORE_PRELOAD") && std::atoi(std::getenv("DS4_STORE_PRELOAD")) == 0) return;
        std::vector<std::pair<double, std::pair<int, int>>> cand;
        double tot = 0;
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) tot += p[e];
        }
        if (tot <= 0) return;   // (no profile yet: the first tokens fill it)
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            const uint8_t* srv = rx_row(l, 0);   // (the expert server's, sent at rx_setup: the decode's lane does not read them)
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0 && (srv == nullptr || srv[e] == 0) && p[e] > 0)
                    cand.push_back({(double) p[e] / (double) l.blob, {l.il, e}});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::pair<int, int>> list;
        list.reserve(cand.size());
        for (const auto& c : cand) list.push_back(c.second);
        const double t0 = now_ms();
        const size_t leave = (size_t) (std::getenv("DS4_STORE_RESERVE") ? std::atoi(std::getenv("DS4_STORE_RESERVE")) : 192) + 64;
        const size_t n = store->preload(list, leave);
        std::fprintf(stderr, "ds4: RAM store: %zu of the profile's hottest experts read in (%.1f GB) in %.1f s\n", n,
                     (double) n * (double) L[(size_t) lo].blob / 1e9, (now_ms() - t0) / 1000.0);
    }
    // at a window boundary: the store's ready slots become the device's (zero-copy) experts, taken back when it needs the room;
    // each touched layer's whole table is written again (VRAM slots, the store's, 0), in stream order before the next window
    void store_sync() {
        if (!store || !store->gpu_mapped()) return;
        std::vector<ExpertStore::Vis> ev;
        store->sync(ev, [&](int layer, int e) { return L[(size_t) layer].slot[(size_t) e] < 0; });
        if (ev.empty()) return;
        std::vector<char> dirty(L.size(), 0);
        for (const auto& v : ev) {
            Layer& l = L[(size_t) v.layer];
            if (l.zc.empty()) continue;
            l.zc[(size_t) v.e] = v.dev != 0 ? 1 : 0;
            dirty[(size_t) v.layer] = 1;
        }
        for (Layer& l : L) {
            if (!dirty[(size_t) l.il]) continue;
            if (l.zc_tab == nullptr) continue;
            std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.zc[(size_t) e]) {
                    const int s = l.slot[(size_t) e] < 0 ? store->find(l.il, e) : -1;
                    tab[(size_t) e] = s >= 0 ? store->dev_blob(s) : 0ull;
                    if (s < 0) l.zc[(size_t) e] = 0;
                }
            cudaSetDevice(devs[0].id);
            cudaMemcpyAsync(l.zc_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice, devs[0].st);
        }
    }

    int cpu_threads() const {
        int n = 0;
        for (const auto& pl : pools) n += pl->size() - 1;
        return n + 1;
    }
    // the CPU lane's workers: a pool per NUMA node (NH of them) on its cores; in place with one part, one pool over
    // every node's cores (the pages lie where the cache put them)
    void make_pools() {
        std::vector<std::vector<int>> cs = cores;
        if ((int) cs.size() < NH && !cs.empty()) {   // (DS4_COPY_NUMA=2 on one node, a test: its cores split in NH)
            std::vector<int> all = cs[0];
            cs.assign((size_t) NH, {});
            for (size_t i = 0; i < all.size(); ++i) cs[i * (size_t) NH / all.size()].push_back(all[i]);
        }
        static const int lane_free = [] {
            const char* v = std::getenv("STRATA_DS4_LANE_FREE");
            return v != nullptr ? std::max(0, std::atoi(v)) : 2;
        }();
        size_t lane_cpus = 0;
        for (const auto& v : cs) lane_cpus += v.size();
        const bool lane_takes_all = lane_cpus >= (size_t) std::max(1u, std::thread::hardware_concurrency());
        for (int X = 0; X < (int) cs.size() && X < std::max(NH, 1); ++X) {
            std::vector<int> c = cs[(size_t) X];
            if (file_experts && NH == 1)
                for (size_t Y = 1; Y < cs.size(); ++Y) c.insert(c.end(), cs[Y].begin(), cs[Y].end());
            if (ep_cpu >= 0) c = rx::off_link(c, ep_sibs);   // (the link core: the service thread helps node 0's pool from it)
            else if (X == 0 && !c.empty()) c.erase(c.begin());   // the host thread's core (it helps node 0's pool)
            // cores of the first pool left free beside the host thread's, for the thread that launches the decode's kernels
            // (and waits on the device) and whatever else the PC runs: with a worker spinning on every other logical CPU
            // that thread waits for a core (Project Maya #359, a 20-core Windows 11 PC: decode 28.3 -> 55.5 tok/s).  Only
            // when the lane would take every logical CPU and --cpu-threads is not given.
            // STRATA_DS4_LANE_FREE: the cores left free (2 by default, 0: as before)
            if (X == 0 && o.cpu_threads <= 0 && lane_free > 0 && (int) c.size() > lane_free + 1 && lane_takes_all)
                c.resize(c.size() - (size_t) lane_free);
            int nw = (int) c.size();
            if (o.cpu_threads > 0) nw = std::max(X == 0 ? 0 : 1, std::min(nw, o.cpu_threads / NH - (X == 0 ? 1 : 0)));
            pools.push_back(std::make_unique<Pool>(nw, c, 20000));   // spin through a token's layers (20 ms)
        }
        (void) gg::mx_tier();   // (measured here, not on the first token)
        if (o.verbose) {
            int nthr = 0;
            for (auto& p : pools) nthr += p->size();
            std::fprintf(stderr, "ds4: CPU lane: %d threads, MXFP4 rows %s (%s)\n", nthr, gg::mx_tier_name(gg::mx_tier()),
                         gg::mx_tier_report());
            if (lane_prof)
                std::fprintf(stderr, "ds4: DS4_LANE_PROF=1: the i-quant CPU lane's phases timed (%d threads)\n", cpu_threads());
            if (iq_pf == 0) std::fprintf(stderr, "ds4: DS4_IQ_PF=0: the i-quant CPU lane's WILLNEED hints off\n");
            else if (iq_pf == 1) std::fprintf(stderr, "ds4: DS4_IQ_PF=1: the i-quant CPU lane's WILLNEED hints on every range\n");
            else if (iq_pf > 1)
                std::fprintf(stderr, "ds4: DS4_IQ_PF=%d: the i-quant CPU lane's WILLNEED hints only on an expert the layer's "
                                     "last %d calls did not take\n", iq_pf, iq_pf);
            if (lane_tr != nullptr)
                std::fprintf(stderr, "ds4: DS4_LANE_TRACE=%s: the i-quant CPU lane's calls written there\n", std::getenv("DS4_LANE_TRACE"));
        }
    }

    void reset() {
        psettle.next();   // (another prompt from here)
        pos = 0;
        toks.clear();
        ctoks.clear();
        slot_held = false;
        for (Check& c0 : ck) c0.pos = -1;
        for (Layer& l0 : L)
            for (int m = 0; m < l0.nm && !elsewhere(l0); ++m) {
                Layer& l = member(l0, m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemsetAsync(l.win, 0, (size_t) g.win * win_rb(), d.st);
                if (l.c_kv) cudaMemsetAsync(l.c_kv, 0, (size_t) l.ratio * g.hd * 4, d.st);
                if (l.c_score) cudaMemsetAsync(l.c_score, 0, (size_t) l.ratio * g.hd * 4, d.st);
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        stage_lost = false;   // (a remote stage's main process: the worker starts again with it)
        if (stage_reset && !stage_reset(stage_err)) stage_lost = true;
    }

    // the state at pos into checkpoint `slot` (the window rings and the compressors' partial groups: the rest of the
    // state is indexed by position, and what lies past a restored position is written again)
    void checkpoint(int slot) {
        if (!ckpt_ok()) return;   // (a remote stage: the worker's layers would need theirs too - none, restore() goes nowhere)
        Check& c = ck[slot & (kChecks - 1)];
        c.pos = pos;
        c.win.resize(L.size());
        c.comp.resize(L.size() * 2);
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            if (elsewhere(l)) continue;
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            c.win[i].resize((size_t) g.win * win_rb());
            cudaMemcpyAsync(c.win[i].data(), l.win, c.win[i].size(), cudaMemcpyDeviceToHost, d.st);
            if (l.c_kv) {
                c.comp[2 * i].resize((size_t) l.ratio * g.hd);
                c.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                cudaMemcpyAsync(c.comp[2 * i].data(), l.c_kv, c.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, d.st);
                cudaMemcpyAsync(c.comp[2 * i + 1].data(), l.c_score, c.comp[2 * i + 1].size() * 4,
                                cudaMemcpyDeviceToHost, d.st);
            }
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        // (a remote stage's main process: the worker takes its layers' checkpoint of this role, at this position; one it
        // did not keep is none - and a worker out of step fails the next window, as a failed hop does)
        if (o.remote_stage && !stage_lost) {
            bool stored = false;
            std::string e;
            if (!stage_ck.save(stage::kCkRole0 + (slot & (kChecks - 1)), pos, stored, e) || !stored) {
                c.pos = -1;
                stage_lost = true;
                stage_err = e.empty() ? "the worker did not keep its checkpoint" : e;
            }
        } else if (o.remote_stage)
            c.pos = -1;
    }

    // the checkpoint at or before position limit (the state's tokens agree with a new prompt up to there), or -1
    int best_check(int64_t limit) const {
        int b = -1;
        for (int i = 0; i < kChecks; ++i)
            if (ck[i].pos >= 0 && ck[i].pos <= limit && ck[i].pos <= pos && (b < 0 || ck[i].pos > ck[b].pos)) b = i;
        return b;
    }
    int64_t restorable(int64_t limit) const {
        const int b = best_check(limit);
        return b < 0 ? -1 : ck[b].pos;
    }

    // checkpoint 1 (the position before the last prompt's end) becomes the anchor too
    void pin() {
        if (!ckpt_ok() || ck[1].pos <= 0) return;
        ck[2] = ck[1];
        if (!o.remote_stage) return;
        bool stored = false;   // (the worker does as well, at its own position)
        std::string e;
        if (stage_lost || !stage_ck.save(stage::kCkPin, pos, stored, e) || !stored) {
            ck[2].pos = -1;
            if (!stage_lost) {
                stage_lost = true;
                stage_err = e.empty() ? "the worker did not pin its checkpoint" : e;
            }
        }
    }
    // the checkpoints the state can go back to (what a slot of it keeps), latest first
    std::vector<int64_t> check_positions() const {
        std::vector<int64_t> v;
        for (const Check& c : ck)
            if (c.pos > 0 && c.pos <= pos && c.win.size() == L.size() &&
                std::find(v.begin(), v.end(), c.pos) == v.end())
                v.push_back(c.pos);
        std::sort(v.rbegin(), v.rend());
        return v;
    }
    // the checkpoints a slot keeps (each with its role: 0, 1 or the anchor)
    std::vector<int> slot_checks() const {
        std::vector<int> v;
        for (int i = 0; i < kChecks; ++i)
            if (ck[i].pos > 0 && ck[i].pos <= pos && ck[i].win.size() == L.size()) v.push_back(i);
        return v;
    }
    size_t check_bytes() const {
        size_t n = 16;
        for (const Layer& l : L)
            if (!elsewhere(l)) n += (size_t) g.win * win_rb() + (l.c_kv ? 2 * (size_t) l.ratio * g.hd * 4 : 0);
        return n;
    }

    bool restore(int64_t limit) {
        psettle.next();   // (another prompt from here)
        const int b = ckpt_ok() ? best_check(limit) : -1;   // (a remote stage keeps none, unless it was asked to)
        if (b < 0) return false;
        const Check& c = ck[b];
        if (o.remote_stage) {   // the worker goes back to the same position first; it does not hold it: nothing changes
            if (stage_lost) return false;
            bool missing = false;
            std::string e;
            if (!stage_ck.restore(stage::kCkAt, c.pos, missing, e)) {
                if (!missing) {   // (its session is lost: both sides start again)
                    std::fprintf(stderr, "ds4: remote stage: going back to token %lld failed (%s)\n", (long long) c.pos, e.c_str());
                    reset();
                }
                return false;
            }
        }
        for (size_t i = 0; i < L.size(); ++i)
            for (int m = 0; m < L[i].nm && !elsewhere(L[i]); ++m) {   // (a group's members hold the same state: the lead's copy)
                Layer& l = member(L[i], m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemcpyAsync(l.win, c.win[i].data(), c.win[i].size(), cudaMemcpyHostToDevice, d.st);
                if (l.c_kv) {
                    cudaMemcpyAsync(l.c_kv, c.comp[2 * i].data(), c.comp[2 * i].size() * 4, cudaMemcpyHostToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, c.comp[2 * i + 1].data(), c.comp[2 * i + 1].size() * 4,
                                    cudaMemcpyHostToDevice, d.st);
                }
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        pos = c.pos;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        slot_held = false;
        for (Check& o2 : ck)   // a later checkpoint saw tokens the state no longer has
            if (o2.pos > pos) o2.pos = -1;
        return true;
    }

    // ---- the kept conversations (a slot): the state out of the model and back, through a file or a host buffer.
    // A prompt that continues the state is reused in place (checkpoint / restore, the compressed KV left where it is);
    // a conversation that leaves the model for another one is a slot - the whole state: the position, its tokens,
    // every window ring (the DSpark blocks' included), the compressors' partial groups and each KV source's compressed
    // KV and index keys up to it (rows 0 .. pos / ratio, the groups its position completed; one the prompt has not
    // completed yet lives in the compressor's partial group).  A tensor parallel group's members hold the same state,
    // so one member's ring is saved and written back into all of them.  Text only: a conversation that holds pictures
    // is not a slot (the Engram history's rows are the tokens' again, no -1 among them).
    static constexpr uint64_t kSlotMagic = 0x31544f4c53345344ull;   // "DS4SLOT1" (little endian)

    int64_t src_rows(const Src& s) const { return pos / std::max(1, s.ratio); }

    // the slot's bytes: the header, the tokens, every layer's ring and partial group, every source's rows, the end
    size_t slot_bytes() const {
        size_t n = 8 + 8 + slot_identity().size() + 8 + (size_t) pos * 4;
        for (const Layer& l : L)
            if (!elsewhere(l)) n += 8 + (size_t) g.win * win_rb() + 8 + (l.c_kv ? 2 * (size_t) l.ratio * g.hd * 4 : 0);
        for (const Src& s : srcs) n += 8 + (size_t) src_rows(s) * (ckv_rb() + ik_rb());
        n += 8 + slot_checks().size() * check_bytes();   // (the checkpoints: what the slot can be taken back at)
        return n + 8;
    }

    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err, int64_t remote_id = 0) {
        if (!slot_ok()) {
            err = "this process holds only layers [" + std::to_string(lo) + ", " + std::to_string(hi) + ")";
            return 0;
        }
        if (linked() && !o.stage_worker && remote_id == 0) {   // (a state of only this side would be the wrong half)
            err = "a remote stage's conversation is set aside under a slot id";
            return 0;
        }
        if (slot_held) {
            err = "it is already set aside";
            return 0;
        }
        if (pos <= 0) {
            err = "there is no conversation";
            return 0;
        }
        if ((int64_t) toks.size() != pos) {
            err = "the state's tokens do not match its position";
            return 0;
        }
        if (!imgs.empty()) {
            err = "it holds pictures";
            return 0;
        }
        if (remote_id != 0) {   // the worker's layers' part first, under the main process's id, where it is now
            if (!o.remote_stage || stage_lost || !stage_ck.save) {
                err = stage_lost ? "remote stage: " + stage_err : std::string("remote stage: no link");
                return 0;
            }
            bool stored = false;
            if (!stage_ck.save(remote_id, pos, stored, err)) {   // (its session is lost: both sides start again)
                std::fprintf(stderr, "ds4: remote stage: setting the worker's part aside failed (%s)\n", err.c_str());
                reset();
                return 0;
            }
            if (!stored) {
                err = "the worker did not keep its part";
                return 0;
            }
        }
        const std::string id = slot_identity();
        out.assign(slot_bytes(), 0);
        uint8_t* p = out.data();
        size_t off = 0;
        const auto put = [&](const void* s, size_t b) {
            std::memcpy(p + off, s, b);
            off += b;
        };
        const uint64_t il = id.size(), n = (uint64_t) pos;
        put(&kSlotMagic, 8);
        put(&il, 8);
        put(id.data(), (size_t) il);
        put(&n, 8);
        put(toks.data(), (size_t) n * 4);
        for (Dev& d : devs) {   // (the devices' work ends before their state is read)
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        for (size_t i = 0; i < L.size(); ++i) {
            const Layer& l = L[i];
            if (elsewhere(l)) continue;   // (another process's layer: its part is that process's slot)
            const Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            const uint64_t wb = (uint64_t) g.win * win_rb();
            put(&wb, 8);
            if (cudaMemcpy(p + off, l.win, (size_t) wb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = l.name + "window ring: " + cudaGetErrorString(cudaGetLastError());
                return 0;
            }
            off += (size_t) wb;
            const uint64_t cn = l.c_kv ? (uint64_t) l.ratio * (uint64_t) g.hd : 0;
            put(&cn, 8);
            if (cn > 0) {
                const size_t cb = (size_t) cn * 4;
                if (cudaMemcpy(p + off, l.c_kv, cb, cudaMemcpyDeviceToHost) != cudaSuccess ||
                    cudaMemcpy(p + off + cb, l.c_score, cb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    err = l.name + "compressor's group: " + cudaGetErrorString(cudaGetLastError());
                    return 0;
                }
                off += 2 * cb;
            }
        }
        for (const Src& s : srcs) {
            const uint64_t rows = (uint64_t) src_rows(s);
            put(&rows, 8);
            if (rows == 0) continue;
            const int di = L[(size_t) s.layer].dev;
            const uint8_t* ckv = s.ckv[(size_t) di];
            const uint8_t* ik = s.ik[(size_t) di];   // (a pack whose sources are not index sources: none, zeros)
            if (ckv == nullptr) {
                err = "source layer " + std::to_string(s.layer) + "'s rows are not on its device";
                return 0;
            }
            const size_t cb = (size_t) rows * ckv_rb(), ib = (size_t) rows * ik_rb();
            cudaSetDevice(devs[(size_t) di].id);
            if (cudaMemcpy(p + off, ckv, cb, cudaMemcpyDeviceToHost) != cudaSuccess ||
                (ik != nullptr && cudaMemcpy(p + off + cb, ik, ib, cudaMemcpyDeviceToHost) != cudaSuccess)) {
                err = "source layer " + std::to_string(s.layer) + "'s compressed KV: " + cudaGetErrorString(cudaGetLastError());
                return 0;
            }
            off += cb + ib;
        }
        {
            const std::vector<int> cs = slot_checks();
            const uint64_t nc = cs.size();
            put(&nc, 8);
            for (int k : cs) {
                const Check& c = ck[k];
                const uint64_t role = (uint64_t) k, cp = (uint64_t) c.pos;
                put(&role, 8);
                put(&cp, 8);
                for (size_t i = 0; i < L.size(); ++i) {
                    if (elsewhere(L[i])) continue;
                    put(c.win[i].data(), (size_t) g.win * win_rb());
                    if (L[i].c_kv) {
                        put(c.comp[2 * i].data(), (size_t) L[i].ratio * g.hd * 4);
                        put(c.comp[2 * i + 1].data(), (size_t) L[i].ratio * g.hd * 4);
                    }
                }
            }
        }
        if (off + 8 != out.size()) {
            err = "the state's size does not add up";
            return 0;
        }
        put(&kSlotMagic, 8);
        slot_held = true;
        return (uint64_t) off;
    }

    bool slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err, int64_t remote_id = 0) {
        if (!slot_ok()) {
            err = "this process holds only layers [" + std::to_string(lo) + ", " + std::to_string(hi) + ")";
            return false;
        }
        if (linked() && !o.stage_worker && remote_id == 0) {
            err = "a remote stage's conversation is taken back under a slot id";
            return false;
        }
        size_t off = 0;
        bool wrote = false, ok = false;
        // a load that stops after some of another conversation's state went into the layers leaves a state no
        // checkpoint can repair: it is reset (the prompt is read from token 0), never served from the mix
        struct FailClosed {
            Impl& p;
            const bool& wrote;
            const bool& ok;
            ~FailClosed() {
                if (!ok && wrote) p.reset();
            }
        } guard{*this, wrote, ok};
        const auto room = [&](size_t b) { return b <= bytes && off <= bytes - b; };
        uint64_t magic = 0, il = 0, n = 0;
        if (!room(16)) {
            err = "the slot is cut short";
            return false;
        }
        std::memcpy(&magic, data + off, 8);
        std::memcpy(&il, data + off + 8, 8);
        off += 16;
        if (magic != kSlotMagic || il == 0 || il > (1u << 20)) {
            err = "not a saved conversation";
            return false;
        }
        if (!room((size_t) il + 8)) {
            err = "the slot is cut short";
            return false;
        }
        {
            std::string id((size_t) il, '\0');
            std::memcpy(id.data(), data + off, (size_t) il);
            off += (size_t) il;
            std::memcpy(&n, data + off, 8);
            off += 8;
            if (id != slot_identity()) {
                err = "another model's, or another engine version's, conversation";
                return false;
            }
        }
        if (n != (uint64_t) n_pos || n_pos <= 0 || n_pos > o.max_context || !room((size_t) n * 4)) {
            err = "the conversation's position is not this prompt's";
            return false;
        }
        std::vector<int32_t> toks_in((size_t) n);
        std::memcpy(toks_in.data(), data + off, (size_t) n * 4);
        off += (size_t) n * 4;
        if (remote_id != 0) {   // the worker puts its layers' part back first: it holds the state at n_pos, or nothing changed
            if (!o.remote_stage || stage_lost || !stage_ck.restore) {
                err = stage_lost ? "remote stage: " + stage_err : std::string("remote stage: no link");
                return false;
            }
            bool missing = false;
            if (!stage_ck.restore(remote_id, n_pos, missing, err)) {
                if (!missing) {   // (its session is lost: both sides start again)
                    std::fprintf(stderr, "ds4: remote stage: taking the worker's part back failed (%s)\n", err.c_str());
                    reset();
                } else
                    err = "the worker does not hold its part of it";
                return false;
            }
            wrote = true;   // (the conversation on the worker is the slot's now: a failure below resets both sides)
        }
        for (Dev& d : devs) {   // (their work ends before the state is written over)
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            if (elsewhere(l)) continue;
            const Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            uint64_t wb = 0, cn = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&wb, data + off, 8);
            off += 8;
            if (wb != (uint64_t) g.win * win_rb() || !room((size_t) wb + 8)) {
                err = l.name + "window ring of another size";
                return false;
            }
            wrote = true;
            for (int m = 0; m < l.nm; ++m) {
                Layer& lm = member(l, m);
                if (cudaMemcpy(lm.win, data + off, (size_t) wb, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = l.name + "window ring: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
            off += (size_t) wb;
            std::memcpy(&cn, data + off, 8);
            off += 8;
            if (cn != (l.c_kv ? (uint64_t) l.ratio * (uint64_t) g.hd : 0)) {
                err = l.name + "compressor's group of another size";
                return false;
            }
            if (cn > 0) {
                const size_t cb = (size_t) cn * 4;
                if (!room(2 * cb)) {
                    err = "the slot is cut short";
                    return false;
                }
                for (int m = 0; m < l.nm; ++m) {
                    Layer& lm = member(l, m);
                    if (cudaMemcpy(lm.c_kv, data + off, cb, cudaMemcpyHostToDevice) != cudaSuccess ||
                        cudaMemcpy(lm.c_score, data + off + cb, cb, cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = l.name + "compressor's group: " + cudaGetErrorString(cudaGetLastError());
                        return false;
                    }
                }
                off += 2 * cb;
            }
        }
        for (const Src& s : srcs) {
            uint64_t rows = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&rows, data + off, 8);
            off += 8;
            if (rows != (uint64_t) (n_pos / std::max(1, s.ratio))) {
                err = "source layer " + std::to_string(s.layer) + "'s rows are not this conversation's";
                return false;
            }
            if (rows == 0) continue;
            const size_t cb = (size_t) rows * ckv_rb(), ib = (size_t) rows * ik_rb();
            if (!room(cb + ib)) {
                err = "the slot is cut short";
                return false;
            }
            for (size_t di = 0; di < devs.size(); ++di) {
                if (s.ckv[di] == nullptr && s.ik[di] == nullptr) continue;
                cudaSetDevice(devs[di].id);
                if (s.ckv[di] != nullptr && cudaMemcpy(s.ckv[di], data + off, cb, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "source layer " + std::to_string(s.layer) + "'s compressed KV: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                if (s.ik[di] != nullptr &&
                    cudaMemcpy(s.ik[di], data + off + cb, ib, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "source layer " + std::to_string(s.layer) + "'s index keys: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
            off += cb + ib;
        }
        Check got[kChecks];   // the slot's checkpoints: the positions it can be taken back at
        {
            uint64_t nc = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&nc, data + off, 8);
            off += 8;
            if (nc > (uint64_t) kChecks) {
                err = "the slot's checkpoints are damaged";
                return false;
            }
            for (uint64_t k = 0; k < nc; ++k) {
                uint64_t role = 0, cp = 0;
                if (!room(16)) {
                    err = "the slot is cut short";
                    return false;
                }
                std::memcpy(&role, data + off, 8);
                std::memcpy(&cp, data + off + 8, 8);
                off += 16;
                if (role >= (uint64_t) kChecks || got[role].pos >= 0 || cp == 0 || cp > (uint64_t) n_pos) {
                    err = "the slot's checkpoints are damaged";
                    return false;
                }
                Check& c = got[role];
                c.pos = (int64_t) cp;
                c.win.resize(L.size());
                c.comp.resize(L.size() * 2);
                for (size_t i = 0; i < L.size(); ++i) {
                    if (elsewhere(L[i])) continue;
                    const size_t wb = (size_t) g.win * win_rb(), cb = L[i].c_kv ? (size_t) L[i].ratio * g.hd * 4 : 0;
                    if (!room(wb + 2 * cb)) {
                        err = "the slot is cut short";
                        return false;
                    }
                    c.win[i].assign(data + off, data + off + wb);
                    off += wb;
                    if (cb > 0) {
                        c.comp[2 * i].resize(cb / 4);
                        c.comp[2 * i + 1].resize(cb / 4);
                        std::memcpy(c.comp[2 * i].data(), data + off, cb);
                        std::memcpy(c.comp[2 * i + 1].data(), data + off + cb, cb);
                        off += 2 * cb;
                    }
                }
            }
        }
        uint64_t end = 0;
        if (!room(8)) {
            err = "the slot is cut short";
            return false;
        }
        std::memcpy(&end, data + off, 8);
        off += 8;
        if (end != kSlotMagic) {
            err = "the slot does not end where it should";
            return false;
        }
        pos = n_pos;
        toks.swap(toks_in);
        ctoks.resize((size_t) pos);
        for (size_t i = 0; i < (size_t) pos; ++i)
            ctoks[i] = toks[i] < pk.eg_vocab ? pk.eg_map[(size_t) toks[i]] : 0;
        for (int k = 0; k < kChecks; ++k) ck[k] = std::move(got[k]);   // (the slot's own; the old ones held the state
                                                               // this one replaced)
        slot_held = false;
        stage_lost = false;   // (a remote stage's main process: both ends hold the slot's state)
        ok = true;
        return true;
    }

    std::string slot_identity() const {
        const auto size_of = [](const std::string& p) {
            std::error_code ec;
            const auto s = std::filesystem::file_size(p, ec);
            return ec ? 0ull : (unsigned long long) s;
        };
        std::ostringstream id;
        id << "strata-ds4-session-3 engine=" << STRATA_VERSION << " mtp=" << g.n_mtp << " dim=" << g.dim << " hd=" << g.hd
           << " win=" << g.win << " ratios=";
        for (int r : g.ratios) id << r << ",";
        id << " kv=";
        for (int v : g.kv_src) id << v << ",";
        id << " ix=";
        for (int v : g.ix_src) id << v << ",";
        if (ranged) id << " layers=" << lo << "-" << hi;   // (a remote stage's two ends: each its own part)
        id << " ds4.txt:" << size_of(pk.dir + "/ds4.txt") << " engram.bin:" << size_of(pk.dir + "/engram.bin")
           << " tensors.txt:" << size_of(pk.dir + "/tensors.txt");
        for (size_t i = 0; i < pk.files.size(); ++i)
            id << " " << pk.files[i] << ":" << os::file_size(pk.fds[i]);
        return id.str();
    }

    size_t bounce_bytes() const {
        return (size_t) kWin * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }
    size_t topk_stride() const { return (size_t) std::max(g.ix_topk, 32); }
    size_t cand_stride() const { return (size_t) std::max(g.cand_topk, 32); }
    size_t bm_stride() const { return (size_t) (o.max_context + 64) / std::max(1, g.cand_blk) + 64; }
    static constexpr int kSlots = kWin * kPicks;   // the CPU lane's (token, expert) rows a window can need

    // ---- the vision tower (the checkpoint's vision.* / aligner.*) on the first GPU, its weights streamed from the
    // host; a prompt's pictures as their spans' rows (set_images)
    VisionTower vt;
    VisionGeo vg;
    int img_tok = -1;
    std::string vision_err;              // why the tower did not load (pictures off), or empty
    std::vector<uint16_t> img_mark[3];   // image_start, image_newline, image_end: BF16 [dim]
    std::vector<Image> imgs;             // the pictures of the prompt being fed, by start
    // a BF16 (or, rounded to nearest even, F32) tensor of n values into dst
    bool read_bf16(const Tensor& t, size_t n, void* dst) {
        if (t.dtype == "BF16") return t.bytes == n * 2 && pk.read(t, dst);
        if (t.dtype != "F32" || t.bytes != n * 4) return false;
        std::vector<float> f(n);
        if (!pk.read(t, f.data())) return false;
        uint16_t* o = (uint16_t*) dst;
        for (size_t i = 0; i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, &f[i], 4);
            o[i] = (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
        return true;
    }
    bool load_vision(std::string& err) {
        vg.layers = (int) pk.num("vision.num_hidden_layers", 0);
        const int tok = (int) pk.num("image_token_id", -1);
        if (vg.layers <= 0 || tok < 0 || pk.get("vision.patch_embed.proj.weight") == nullptr) return true;   // none
        vg.dim = (int) pk.num("vision.hidden_size", 1024), vg.heads = (int) pk.num("vision.num_attention_heads", 16);
        vg.inter = (int) pk.num("vision.intermediate_size", 2816), vg.patch = (int) pk.num("vision.patch_size", 14);
        vg.merge = (int) pk.num("vision.downsample_ratio", 3), vg.theta = (float) pk.num("vision.rope_theta", 1e4);
        vg.max_tokens = (int) pk.num("vision.max_image_tokens", 1024), vg.out = g.dim;
        const char* mark[3] = {"image_start", "image_newline", "image_end"};
        for (int i = 0; i < 3; ++i) {
            const Tensor* t = pk.get(mark[i]);
            img_mark[i].resize((size_t) g.dim);
            if (t == nullptr || !read_bf16(*t, (size_t) g.dim, img_mark[i].data())) {
                err = std::string(mark[i]) + ": not a BF16 (or F32) row of the model's width";
                return false;
            }
        }
        const double t0 = now_ms();
        Dev& d = devs[0];
        // the tower's tensors as BF16: the checkpoint's own, or a GGUF's mmproj (F32 norms / biases / marker rows, which
        // came from BF16, converted back exactly; mlp.w1 as the two files' gate and up, the halves of its rows)
        auto read = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) {
            const Tensor* t = pk.get(n);
            if (t == nullptr && n.size() > 13 && n.compare(n.size() - 13, 13, "mlp.w1.weight") == 0) {
                const std::string pre = n.substr(0, n.size() - 13);
                const Tensor* tg = pk.get(pre + "mlp.gate.weight");
                const Tensor* tu = pk.get(pre + "mlp.up.weight");
                if (tg == nullptr || tu == nullptr || tg->shape.size() != 2 || tg->shape != tu->shape) return false;
                shape = {tg->shape[0] * 2, tg->shape[1]};
                const size_t half = (size_t) tg->shape[0] * (size_t) tg->shape[1];
                return dst == nullptr || (read_bf16(*tg, half, dst) && read_bf16(*tu, half, (uint16_t*) dst + half));
            }
            if (t == nullptr) return false;
            shape = t->shape;
            size_t n_el = 1;
            for (int64_t x : t->shape) n_el *= (size_t) x;
            return dst == nullptr || read_bf16(*t, n_el, dst);
        };
        if (!vt.load(vg, d.id, read, err)) return false;
        img_tok = tok;
        CK(cudaSetDevice(d.id));
        CK(cudaMemGetInfo(&d.free0, &d.total));   // (the split sees what the tower left)
        if (o.verbose)
            std::fprintf(stderr, "ds4: vision tower (%d blocks) on CUDA%d: %.0f MiB, up to %d patches, %.1f s\n",
                         vg.layers, d.id, (double) vt.device_bytes() / (1 << 20), vt.max_patches(),
                         (now_ms() - t0) / 1000.0);
        return true;
    }
    bool encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err) {
        if (img_tok < 0) {
            err = "this model has no vision tower (or it is off)";
            return false;
        }
        std::vector<uint16_t> rows;
        int nh = 0, nw = 0;
        if (!vt.encode(rgb, w, h, rows, nh, nw, err)) return false;
        const size_t D = (size_t) g.dim;
        span.resize(((size_t) nh * (nw + 1) + 2) * D);
        uint16_t* o = span.data();
        auto put = [&](const uint16_t* r) {
            std::memcpy(o, r, D * 2);
            o += D;
        };
        put(img_mark[0].data());
        for (int i = 0; i < nh; ++i) {
            for (int j = 0; j < nw; ++j) put(rows.data() + ((size_t) i * nw + j) * D);
            put(img_mark[1].data());
        }
        put(img_mark[2].data());
        return true;
    }
    // the row of a picture's span at position p (null: p is in none)
    const uint16_t* image_row(int64_t p) const {
        if (imgs.empty()) return nullptr;
        auto it = std::upper_bound(imgs.begin(), imgs.end(), p, [](int64_t v, const Image& im) { return v < im.start; });
        if (it == imgs.begin()) return nullptr;
        const Image& im = *(it - 1);
        const int64_t r = p - im.start;
        if (!im.rows || r >= (int64_t) (im.rows->size() / (size_t) g.dim)) return nullptr;
        return im.rows->data() + (size_t) r * g.dim;
    }
    // token tk's embedding row at position p: inside a picture's span, the picture's
    const uint16_t* emb_row(int64_t p, int tk) const {
        const uint16_t* r = tk == img_tok ? image_row(p) : nullptr;
        return r != nullptr ? r : embed_pinned + (size_t) tk * g.dim;
    }
    bool in_image(int64_t p) const { return img_tok >= 0 && toks[(size_t) p] == img_tok && image_row(p) != nullptr; }
    // the Engram's compressed id of token tk at p (-1: a picture's, which no n-gram spans)
    int32_t ctok(int64_t p, int tk) const {
        if (tk == img_tok && image_row(p) != nullptr) return -1;
        return tk < pk.eg_vocab ? pk.eg_map[(size_t) tk] : 0;
    }
    // a chunk's image rows (route()'s bits); false: none
    bool vl_bits(int64_t p0, int T, VlBits& b) const {
        if (imgs.empty()) return false;
        std::memset(&b, 0, sizeof b);
        bool any = false;
        for (int t = 0; t < T && t < kVlRows; ++t)
            if (in_image(p0 + t)) {
                b.w[t >> 5] |= 1u << (t & 31);
                any = true;
            }
        return any;
    }

    // ---- Engram: a window's rows of one table (Engram layer li), tokens p0 .. p0 + T - 1 into dst ([token] of [cols]
    // [hd] codes then the scales; nullptr: h_eg).  eg_issue serves the rows already fetched from the row cache (and a
    // row the window needs twice from one read) and hands the rest to the reader's threads; eg_finish waits for them,
    // copies the repeats and keeps the new rows in the cache.  The rows are the file's bytes either way.
    void eg_issue(int64_t p0, int T, int li, uint8_t* dst, EgBatch& b) {
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, hd = pk.eg_hd;
        const size_t per = (size_t) cols * (hd + hd / 32);
        b.clear();
        b.reads.reserve((size_t) T * cols);
        const std::string P = "layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.";
        const Tensor* tw = pk.get(P + "weight");
        const Tensor* ts = pk.get(P + "scale");
        const int fmt = eg_fmt[(size_t) li];
        const bool mx8 = fmt == 0;   // (a GGUF: the scales between the codes, one per 32)
        const bool q6k = fmt == 1;   // (a GGUF: one Q6_K superblock a 256 values, its scales inside)
        const bool iq4 = fmt == 2;   // (a GGUF: 18-byte blocks a 32 values, d and 16 nibble bytes, no scales)
        std::unordered_map<uint64_t, int> first;   // a row this batch reads already -> its read
        int64_t hits = 0;
        {
            std::lock_guard<std::mutex> lk(eg_cache_mu);
            for (int t = 0; t < T; ++t) {
                const int64_t p = p0 + t;
                uint8_t* rows = dst != nullptr ? dst + (size_t) t * per : h_eg + ((size_t) li * eg_cap + t) * per;
                uint8_t* scales = rows + (size_t) cols * hd;
                if (ctoks[(size_t) p] < 0) {   // a picture's token: no rows (zeros: its gate adds nothing)
                    std::memset(rows, 0, per);
                    continue;
                }
                int64_t prod[8];
                bool blocked = false;
                for (int s = 0; s < pk.eg_ngram; ++s) {   // (the look-back stops at the start and at a picture)
                    blocked = blocked || p < s || ctoks[(size_t) (p - s)] < 0;
                    const int64_t tk = blocked ? pk.eg_pad : ctoks[(size_t) (p - s)];
                    prod[s] = tk * pk.eg_mult[(size_t) li * pk.eg_ngram + s];
                }
                int64_t rolling = prod[0];
                for (int n = 1; n < pk.eg_ngram; ++n) {
                    rolling ^= prod[n];
                    for (int h = 0; h < pk.eg_heads; ++h) {
                        const int c = (n - 1) * pk.eg_heads + h;
                        const int64_t id = rolling % pk.eg_primes[(size_t) li * cols + c] + pk.eg_off[(size_t) li * cols + c];
                        uint8_t* rd = rows + (size_t) c * hd;
                        uint8_t* sd = scales + (size_t) c * (hd / 32);
                        const uint64_t key = (uint64_t) id * 8 + (uint64_t) li;
                        if (const uint8_t* hit = eg_cache.find(key)) {
                            std::memcpy(rd, hit, (size_t) hd);
                            std::memcpy(sd, hit + hd, (size_t) hd / 32);
                            ++hits;
                            continue;
                        }
                        const auto fi = first.find(key);
                        if (fi != first.end()) {
                            b.dups.push_back({fi->second, {rd, sd}});
                            continue;
                        }
                        first.emplace(key, (int) b.reads.size());
                        EgRead r;
                        r.file = tw->file;
                        r.dst = rd;
                        r.scales = sd;
                        r.key = key;
                        if (mx8) {
                            r.mx8 = true;
                            r.n = (uint32_t) (hd / 32 * 33);
                            r.off = tw->off + (uint64_t) id * r.n;
                        } else if (q6k || iq4) {   // (into the row's code bytes; its scale bytes unused)
                            r.n = (uint32_t) (q6k ? hd / 256 * 210 : hd / 32 * 18);
                            r.off = tw->off + (uint64_t) id * r.n;
                        } else {
                            r.n = (uint32_t) hd;
                            r.off = tw->off + (uint64_t) id * hd;
                            r.file2 = ts->file;
                            r.n2 = (uint32_t) (hd / 32);
                            r.off2 = ts->off + (uint64_t) id * (hd / 32);
                        }
                        b.reads.push_back(r);
                    }
                }
            }
        }
        eg_rows_n += hits + (int64_t) b.reads.size() + (int64_t) b.dups.size();
        eg_hits_n += hits + (int64_t) b.dups.size();
        eg_reads_n += (int64_t) b.reads.size();
        eg_io.submit(b);
    }
    // b's rows all in place (any thread; a group's members may ask at once: one finishes, the others wait for it)
    void eg_finish(EgBatch& b) {
        if (b.post.load(std::memory_order_acquire) == 2) return;
        const int64_t t0 = now_ns();
        eg_io.wait(b);
        int want = 0;
        if (b.post.compare_exchange_strong(want, 1, std::memory_order_acq_rel)) {
            const int hd = pk.eg_hd;
            for (const auto& d : b.dups) {
                const EgRead& r = b.reads[(size_t) d.first];
                std::memcpy(d.second.first, r.dst, (size_t) hd);
                std::memcpy(d.second.second, r.scales, (size_t) hd / 32);
            }
            {
                std::lock_guard<std::mutex> lk(eg_cache_mu);
                for (size_t i = 0; i < b.reads.size(); ++i) {
                    const EgRead& r = b.reads[i];
                    if (!b.ok[i]) {   // (a failed read: zeros, its gate adds nothing; not kept)
                        std::memset(r.dst, 0, (size_t) hd);
                        std::memset(r.scales, 0, (size_t) hd / 32);
                        continue;
                    }
                    if (uint8_t* c = eg_cache.put(r.key)) {
                        std::memcpy(c, r.dst, (size_t) hd);
                        std::memcpy(c + hd, r.scales, (size_t) hd / 32);
                    }
                }
            }
            eg_io_ns += b.t1.load() - b.t0;
            b.live = false;
            b.post.store(2, std::memory_order_release);
        } else {
            while (b.post.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        }
        eg_wait_ns += now_ns() - t0;
    }
    void eg_stats() {
        st.eg_rows = eg_rows_n.load();
        st.eg_hits = eg_hits_n.load();
        st.eg_reads = eg_reads_n.load();
        st.eg_io_ms = (double) eg_io_ns.load() / 1e6;
        std::lock_guard<std::mutex> lk(eg_cache_mu);
        st.eg_cached = (int64_t) eg_cache.size();
        st.eg_cache_cap = (int64_t) eg_cache.capacity();
    }
    // the chunked prompt path: rows of Engram layer `only` into dst, now (a helper thread, a chunk ahead)
    void engram_rows(int64_t p0, int T, int only, uint8_t* dst) {
        EgBatch b;
        eg_issue(p0, T, only, dst, b);
        eg_finish(b);
    }

    // ---- the CPU lane: a window's routed experts that live on the host, as entries (an expert and up to 8 of its
    // tokens, in rows slot0 ..) computed together: each node its column part of every matrix, from memory on that node,
    // in four rounds (the gate / up split-K pieces, each preparing its two input blocks of every token; each band's
    // sums / codebook / Hadamard / swiglu and that block of the down input, which both nodes then read whole; the down
    // pieces; the down bands).  yd[slot] gets each row's expert output.
    static constexpr int kEntMax = kWin > kc::kExl3CpuMaxN ? kWin : kc::kExl3CpuMaxN;
    struct Ent {
        int e = 0, n = 0, slot0 = 0;
        int tok[kEntMax] = {};
    };
    // tokens an entry holds: the i-quant GGUF experts take a whole window's tokens of an expert in one call (the CPU
    // lane's exact multi-token rows, 7+ tokens an expert: decoded once for all), the others (MXFP4, EXL3) up to 8
    // (cpu_experts' second slice: the entries from `from` on over n_ff values [ff0, ff0 + ffn) instead; from < 0: none)
    // (constructors, not member initializers: gcc takes no `= {}` default argument of a nested class with them
    // before the enclosing class ends)
    struct FfSplit {
        int from, ff0, ffn;
        FfSplit() : from(-1), ff0(0), ffn(0) {}
        FfSplit(int f, int a, int n) : from(f), ff0(a), ffn(n) {}
    };
    static int ent_cap(const Layer& l) {
        return l.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(l.lay) ? kEntMax : kc::kExl3CpuMaxN;
    }
    void run_nodes(int n, const std::function<void(int)>* f) {
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    }
    // (DS4_LANE_PROF) run_nodes timed: its wall time, its tail (the caller's wait once it ran out of jobs: the
    // barrier's wait for the slowest worker's last job) and its jobs' time in all (busy), each added to its own
    void run_nodes_prof(int n, const std::function<void(int)>* f, double& wall, double& tail, double& busy) {
        std::atomic<int64_t> ns{0};
        std::function<void(int)> g[4];
        for (int X = 0; X < NH; ++X)
            g[X] = [&, X](int j) {
                const auto a = std::chrono::steady_clock::now();
                f[X](j);
                ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count(),
                             std::memory_order_relaxed);
            };
        const double t0 = now_ms();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, g[X]);
        pools[0]->help();
        const double th = now_ms();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
        const double t1 = now_ms();
        wall += t1 - t0;
        tail += t1 - th;
        busy += (double) ns.load(std::memory_order_relaxed) / 1e6;
    }

    // a GGUF's MXFP4 experts on the CPU lane, read from the file's mapping or their copy in RAM (mx_part_ptr): each
    // node its rows of every matrix - the gate / up rows [X n_ff / NH, ..) of each entry (the swiglu into its h row),
    // then (h whole) the down rows [X dim / NH, ..) into yd.  A row's sum is the same whatever the split: the same bits
    // in a copy as in place.  The device's arithmetic (gg::mx_rows: int8 activations rounded as the device rounds
    // them, exact integer sums a block; every CPU tier the same bits): each token's input quantized once, whatever
    // number of entries take it, and each job its band of h straight into the down's int8 input.
    // (ord: the order the entries are computed in - their slots, so the sums, stay the same; null: as they are)
    void cpu_experts_mx(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord, int ff0,
                        int ffn, FfSplit sb = FfSplit()) {
        if (!gg::mx_experts(l.lay)) return cpu_experts_iq(l, x, ents, ldx, ord, ff0, ffn, sb);
        // (MXFP4 here: whole - a slice is the i-quants' alone, and only a layer that takes it is ever asked for one:
        // rx_split_l, ep_slices_ok)
        constexpr int RB = 64;   // rows a job (whole blocks of h: RB % 32 == 0)
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
        const size_t gr = gg::mx_row_bytes(E), dr = gg::mx_row_bytes(FF);
        const size_t xa = gg::mx_act_bytes(E), ha = gg::mx_act_bytes(FF);
        const bool band_q = FFh % 32 == 0;   // (each job's rows whole blocks of h)
        const float lim = g.lim;
        static_assert(kc::kExl3CpuMaxN <= gg::kMxCpuMaxN, "a CPU-lane entry's tokens in one call");
        {   // the experts read from the files (not in the RAM tier) that are not resident: all their pages asked for at
            // once, so the SSD reads them with large requests side by side while the first entries compute
            // (os::prefetch_cold: the resident ones are left alone - the hint walks every page of a range, which cost
            // more than computing a warm expert, 0.34 ms a 6.3 MB matrix on Windows)
            std::vector<std::pair<const void*, size_t>> want;
            for (const Ent& en : ents)
                if (!store && (l.ram.empty() || l.ram[(size_t) en.e] < 0))
                    for (int r = 0; r < 3; ++r) want.push_back({mx_mat(l, en.e, r), mx_part_bytes(l, r)});
            os::prefetch_cold(want);
        }
        // the int8 inputs in c_xh's bytes: token t's (t < kWin) at t xa, then the slots' h at (kWin + slot) ha; a token
        // index past the window (none on the decode or chunk paths) gets a row of its own after them
        uint8_t* xq = (uint8_t*) c_xh.data();
        uint8_t* hq = xq + (size_t) kWin * xa;
        uint8_t* xq_extra = hq + (size_t) kSlots * ha;
        auto x_act = [&](const Ent& en, int i) -> const void* {
            const int t = en.tok[i];
            return t < kWin ? xq + (size_t) t * xa : xq_extra + (size_t) (en.slot0 + i) * xa;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::mx_quant(x + (size_t) t * ldx, E, (void*) x_act(en, i));
                }
        }
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
                const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float yg[gg::kMxCpuMaxN][RB], yu[gg::kMxCpuMaxN][RB];
                float *gp[gg::kMxCpuMaxN], *up[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = x_act(en, i);
                    gp[i] = yg[i];
                    up[i] = yu[i];
                }
                // (part X's rows, from where they are: a copy's part on node X, or the file's)
                gg::mx_rows(mx_part_ptr(l, en.e, 0, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, gp);
                gg::mx_rows(mx_part_ptr(l, en.e, 1, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, up);
                for (int i = 0; i < en.n; ++i) {
                    float* h = band_q ? yg[i] : c_xhd.data() + (size_t) (en.slot0 + i) * FF + r0;
                    for (int r = r0; r < r1; ++r) {
                        float gv = yg[i][r - r0], uv = yu[i][r - r0];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        h[r - r0] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    if (band_q) gg::mx_quant_blocks(h, r0 / 32, r1 / 32, hq + (size_t) (en.slot0 + i) * ha, FF);
                }
            };
        run_nodes(ne * GB, f);
        if (!band_q)
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i)
                    gg::mx_quant(c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF, hq + (size_t) (en.slot0 + i) * ha);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
                const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float* yp[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = hq + (size_t) (en.slot0 + i) * ha;
                    yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + r0;
                }
                gg::mx_rows(mx_part_ptr(l, en.e, 2, X), dr, FF, r0 - X * Eh, r1 - X * Eh, ap, en.n, yp);
            };
        run_nodes(ne * DB, f);
    }
    // (DS4_IQ_PF > 1) the layer's call count, this call's; and whether expert e was not taken in the layer's last iq_pf
    // calls (never, or longer ago), marked taken now - a layer past the counts' size: always
    uint32_t pf_tick(const Layer& l) {
        if (pf_calls.empty()) {
            for (const Layer& x : L) pf_ne = std::max(pf_ne, (size_t) std::max(0, x.n_exp));
            pf_calls.assign(L.size(), 0);
            pf_seen.assign(L.size() * pf_ne, 0);
        }
        return l.il >= 0 && (size_t) l.il < pf_calls.size() ? ++pf_calls[(size_t) l.il] : 0;
    }
    bool pf_stale(const Layer& l, int e, uint32_t tick) {
        if (tick == 0 || e < 0 || (size_t) e >= pf_ne) return true;
        uint32_t& s = pf_seen[(size_t) l.il * pf_ne + (size_t) e];
        const bool stale = s == 0 || tick - s > (uint32_t) iq_pf;
        s = tick;
        return stale;
    }
    // (DS4_LANE_TRACE) the call's record
    void lane_trace(const Layer& l, const std::vector<Ent>& ents, int ff0, int ffn) {
        const uint8_t hd[4] = {(uint8_t) l.il, (uint8_t) (ff0 / 256), (uint8_t) ((ff0 + ffn) / 256), (uint8_t) std::min<size_t>(255, ents.size())};
        std::fwrite(hd, 1, 4, lane_tr);
        for (size_t i = 0; i < (size_t) hd[3]; ++i) {
            const int16_t e = (int16_t) ents[i].e;
            const uint8_t n = (uint8_t) std::min(255, ents[i].n);
            std::fwrite(&e, 2, 1, lane_tr);
            std::fwrite(&n, 1, 1, lane_tr);
        }
        if (++lane_tr_n % 4096 == 0) std::fflush(lane_tr);
    }
    // ... the low-bit ones (ds4_iq's types): as above, through ggml-cpu's dots (gg::iq_cpu_*) - each token's input
    // quantized once to the weight type's activation (Q8_K) in its c_xh row's bytes, then (h whole) its h the same.
    // Over the n_ff slice [ff0, ff0 + ffn) (cpu_experts' caller's: the whole n_ff, or a split pick's share - H',
    // ds4_rx_split.hpp): the gate / up rows of the slice, and the down over the slice's values in place (gg::iq_cpu_slice) -
    // each entry's yd then its part of the expert's output.  sb: the entries from sb.from on over sb's slice instead, in
    // the same rounds (the hybrid's cut W picks, [0, cut): one pass with the lane's own [0, k) - ord orders the others)
    void cpu_experts_iq(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord, int ff0,
                        int ffn, FfSplit sb = FfSplit()) {
        constexpr int RB = 64;   // rows a job
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int ne = (int) ents.size(), na = sb.from >= 0 && sb.from < ne ? sb.from : ne;   // (entries [na, ne): sb's)
        const int ns = na < ne ? 2 : 1;
        const int rf0[2] = {ff0, sb.ff0}, rfn[2] = {ffn, sb.ffn};
        int glo[2][4], ghi[2][4], GB[2] = {0, 0};   // node X's gate / up rows: its part's [X FFh, ..) within each slice
        bool sliced[2] = {false, false};
        for (int s = 0; s < ns; ++s) {
            int gmax = 0;
            for (int X = 0; X < NH; ++X) {
                glo[s][X] = std::max(X * FFh, rf0[s]);
                ghi[s][X] = std::min(X * FFh + FFh, rf0[s] + rfn[s]);
                gmax = std::max(gmax, ghi[s][X] - glo[s][X]);
            }
            GB[s] = (gmax + RB - 1) / RB;
            sliced[s] = rf0[s] != 0 || rfn[s] != FF;
        }
        const int DB = (Eh + RB - 1) / RB;
        auto set_of = [&](int q) { return q >= na ? 1 : 0; };
        auto ent_at = [&](int i) { return i < na && ord != nullptr ? ord[i] : i; };   // (the i-th entry computed)
        gg::IqCpu fm, fs[2];
        size_t d_off[2] = {0, 0};
        std::string ferr;
        bool ok = gg::iq_cpu_fmt(gg::expert_type(l.lay, 0), gg::expert_type(l.lay, 2), E, FF, fm, ferr) &&
                  fm.act_bytes <= (size_t) E * 4 && fm.h_bytes <= (size_t) E * 4;
        for (int s = 0; s < ns && ok; ++s) ok = !sliced[s] || gg::iq_cpu_slice(fm, rf0[s], rfn[s], fs[s], d_off[s], ferr);
        if (!ok) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true)) std::fprintf(stderr, "ds4: layer %d's CPU-lane experts: %s\n", l.il, ferr.c_str());
            for (const Ent& en : ents) std::memset(c_yd.data() + (size_t) en.slot0 * E, 0, (size_t) en.n * E * 4);
            return;
        }
        const float lim = g.lim;
        if (lane_tr != nullptr) {   // (a record a slice)
            if (ns == 1) {
                lane_trace(l, ents, ff0, ffn);
            } else {
                lane_trace(l, std::vector<Ent>(ents.begin(), ents.begin() + na), ff0, ffn);
                lane_trace(l, std::vector<Ent>(ents.begin() + na, ents.end()), sb.ff0, sb.ffn);
            }
        }
        const gg::IqCpu* fd[2] = {sliced[0] ? &fs[0] : &fm, sliced[1] ? &fs[1] : &fm};
        // (DS4_IQ_DOWN_PARTS) the entries' down rows from the parts, contiguous - an expert read from the files only (a
        // RAM-tier copy or the store keeps the pack's layout): dpart(e, s) its slice's row 0, fp the rows' stride - each
        // slice's own (the parts serve a slice inside theirs: the cut W picks' [0, cut) too while cut <= b; past b the pack's)
        const DownParts* dp[2] = {nullptr, nullptr};
        gg::IqCpu fp[2] = {*fd[0], *fd[1]};
        for (int s = 0; s < ns; ++s) {
            dp[s] = store || dp_pack ? nullptr : down_parts(l, rf0[s], rfn[s]);
            if (dp[s] != nullptr) fp[s].d_stride = dp[s]->rb;
        }
        auto dpart = [&](int e, int s) -> const uint8_t* {
            return dp[s] != nullptr && (l.ram.empty() || l.ram[(size_t) e] < 0) ? dpart_at(*dp[s], e, rf0[s]) : nullptr;
        };
        const double tp0 = lane_prof ? now_ms() : 0.0;   // (DS4_LANE_PROF: the prefetch and the x quant from here)
        if (iq_pf != 0) {   // (os::prefetch_cold, as above: the resident ones are left alone - on Windows the hint walks every page of a
            // range, ~0.13 ms an IQ2_XXS expert, on the lane's critical path before the first entry computes.
            // off by default (the mmap path, owner PC: not faster in every pair); DS4_IQ_PF_COLD=1 turns it on)
            const uint32_t tick = iq_pf > 1 ? pf_tick(l) : 0;
            std::vector<std::pair<const void*, size_t>> want;
            for (int q = 0; q < ne; ++q) {
                const Ent& en = ents[(size_t) q];
                const int s = set_of(q);
                if (!store && (l.ram.empty() || l.ram[(size_t) en.e] < 0) && (iq_pf <= 1 || pf_stale(l, en.e, tick)))
                    for (int r = 0; r < 3; ++r) {   // (a slice's gate / up rows; the down's rows hold every slice)
                        if (r == 2 && dpart(en.e, s) != nullptr) {   // (the parts' rows of the slice)
                            want.push_back({dpart(en.e, s), (size_t) (E - 1) * dp[s]->rb + fd[s]->d_row});
                            continue;
                        }
                        want.push_back({mx_mat(l, en.e, r) + (r < 2 ? (size_t) rf0[s] * mx_row_b(l, r) : 0),
                                        r < 2 ? (size_t) rfn[s] * mx_row_b(l, r) : mx_part_bytes(l, r)});
                    }
            }
            static const bool pf_all = std::getenv("DS4_IQ_PF_COLD") == nullptr || std::atoi(std::getenv("DS4_IQ_PF_COLD")) == 0;
            if (pf_all || iq_pf > 0) os::prefetch(want);
            else os::prefetch_cold(want);
        }
        // the activations in c_xh's bytes, E 4 apart: token t's (t < kWin) at row t - quantized once, whatever number of
        // entries take it - then a row a slot (kWin + slot): a token past the window's input (none on the decode or
        // chunk paths), then (x no longer needed) the slot's h
        uint8_t* aq = (uint8_t*) c_xh.data();
        auto x_act = [&](const Ent& en, int i) -> uint8_t* {
            const int t = en.tok[i];
            return aq + (size_t) (t < kWin ? t : kWin + en.slot0 + i) * E * 4;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::iq_cpu_quant(fm.gu_act, x + (size_t) t * ldx, E, x_act(en, i));
                }
        }
        if (lane_prof) {
            st.lp_prep_ms += now_ms() - tp0;
            ++st.lp_calls;
            st.lp_threads = cpu_threads();
        }
        // the jobs: the i-th entry computed's (ent_at) gate / up band b of part X, its h quantized, its down band b of
        // part X - the first na entries' GB[0] gate / up bands each, then the others' GB[1]
        const int ja = na * GB[0], jg = ja + (ne - na) * GB[1];
        auto gu_job = [&](int X, int j) {
            const int s = j < ja ? 0 : 1, jj = s == 0 ? j : j - ja;
            const Ent& en = ents[(size_t) ent_at(s == 0 ? jj / GB[0] : na + jj / GB[1])];
            const int r0 = glo[s][X] + (jj % GB[s]) * RB, r1 = std::min(ghi[s][X], r0 + RB);
            if (r0 >= r1) return;   // (a node with fewer of the slice's rows)
            const void* ap[kEntMax];
            float* hp[kEntMax];
            for (int i = 0; i < en.n; ++i) {
                ap[i] = x_act(en, i);
                hp[i] = c_xhd.data() + (size_t) (en.slot0 + i) * FF + (size_t) X * FFh;   // (part X's rows of h)
            }
            // (part X's rows, from where they are: a copy's part on node X, or the file's - whole rows, so any
            // block size of the type)
            gg::iq_cpu_gate_up(fm, mx_part_ptr(l, en.e, 0, X), mx_part_ptr(l, en.e, 1, X), ap, en.n, hp,
                               r0 - X * FFh, r1 - X * FFh, lim);
        };
        auto h_quant = [&](int q) {
            const Ent& en = ents[(size_t) q];
            const int s = set_of(q);
            for (int i = 0; i < en.n; ++i)
                gg::iq_cpu_quant(fm.d_act, c_xhd.data() + (size_t) (en.slot0 + i) * FF + rf0[s], rfn[s],
                                 aq + (size_t) (kWin + en.slot0 + i) * E * 4);
        };
        auto dn_job = [&](int X, int i, int b) {
            const int q = ent_at(i), s = set_of(q);
            const Ent& en = ents[(size_t) q];
            const int r0 = X * Eh + b * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
            const void* hp[kEntMax];
            float* yp[kEntMax];
            for (int k = 0; k < en.n; ++k) {
                hp[k] = aq + (size_t) (kWin + en.slot0 + k) * E * 4;
                yp[k] = c_yd.data() + (size_t) (en.slot0 + k) * E + (size_t) X * Eh;   // (part X's rows of it)
            }
            if (const uint8_t* p = dpart(en.e, s))   // (the parts' rows: part X's from row X Eh)
                gg::iq_cpu_down(fp[s], p + (size_t) (X * Eh) * dp[s]->rb, hp, en.n, yp, r0 - X * Eh, r1 - X * Eh);
            else
                gg::iq_cpu_down(*fd[s], mx_part_ptr(l, en.e, 2, X) + d_off[s], hp, en.n, yp, r0 - X * Eh, r1 - X * Eh);
        };
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { gu_job(X, j); };
        if (lane_prof) {   // (DS4_LANE_PROF: the same calls, timed)
            run_nodes_prof(jg, f, st.lp_gu_ms, st.lp_gu_tail_ms, st.lp_gu_busy_ms);
            const double th = now_ms();
            for (int q = 0; q < ne; ++q) h_quant(q);
            st.lp_hq_ms += now_ms() - th;
            for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { dn_job(X, j / DB, j % DB); };
            run_nodes_prof(ne * DB, f, st.lp_dn_ms, st.lp_dn_tail_ms, st.lp_dn_busy_ms);
            return;
        }
        run_nodes(jg, f);
        for (int q = 0; q < ne; ++q) h_quant(q);
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { dn_job(X, j / DB, j % DB); };
        run_nodes(ne * DB, f);
    }

    // the window's CPU-lane experts made ready (and pinned) in the store for the length of the call
    struct StoreHold {
        Impl* m = nullptr;
        int layer = 0;
        std::vector<int> es;
        std::vector<char> ok;
        StoreHold(Impl* m_, const Layer& l, const std::vector<Ent>& ents) {
            if (!m_->store) return;
            m = m_;
            layer = l.il;
            for (const Ent& en : ents)
                if ((l.ram.empty() || l.ram[(size_t) en.e] < 0) && std::find(es.begin(), es.end(), en.e) == es.end()) es.push_back(en.e);
            m->store->ensure(layer, es.data(), (int) es.size(), ok);
        }
        ~StoreHold() {
            if (m) m->store->release(layer, es.data(), (int) es.size(), ok);
        }
    };
    // ents over the n_ff range [ff0, ff0 + ffn) - each caller says which (ds4_rx_split.hpp): the whole n_ff, but [0, k)
    // for the picks a request just sent tagged slice (serve_lane) and [k, nb) for those of a reply late or lost (lane_add);
    // sb: the entries from sb.from on over its range (the i-quant lane in the same pass, else a call of their own)
    void cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord, int ff0,
                     int ffn, FfSplit sb = FfSplit()) {
        if (ldx == 0) ldx = (size_t) g.dim;
        if (sb.from >= (int) ents.size()) sb = FfSplit{};
        if (sb.from >= 0 && !(l.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(l.lay))) {   // (one slice a call there)
            const std::vector<Ent> a(ents.begin(), ents.begin() + sb.from), b(ents.begin() + sb.from, ents.end());
            if (!a.empty()) cpu_experts(l, x, a, ldx, ord, ff0, ffn);
            return cpu_experts(l, x, b, ldx, nullptr, sb.ff0, sb.ffn);
        }
        if (l.lay.fmt == gg::kFmtMXFP4) {
            StoreHold hold(this, l, ents);
            return cpu_experts_mx(l, x, ents, ldx, ord, ff0, ffn, sb);
        }
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
        const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
        const int ne = (int) ents.size();
        const float lim = g.lim;
        auto part = [&](int e, int X, int r) {
            const uint8_t* b = l.hh[(size_t) e][(size_t) X];
            kc::Exl3Part p;
            p.trellis = (const uint32_t*) (b + l.hl.trellis[r]);
            p.suh = (const uint16_t*) (b + l.hl.suh[r]);
            p.svh = (const uint16_t*) (b + l.hl.svh[r]);
            p.k = l.hl.k[r];
            p.n = l.hl.n[r];
            p.K = l.lay.K[r];
            return p;
        };
        auto xh = [&](int X, int sl, int r) { return c_xh.data() + ((size_t) (X * kSlots + sl) * 2 + r) * E; };
        auto xsb = [&](int X, int sl, int r) { return c_xsb.data() + ((size_t) (X * kSlots + sl) * 2 + r) * EB; };
        auto acc = [&](int X, int sl, int r, int q) {
            return c_acc.data() + (((size_t) (X * kSlots + sl) * 2 + r) * GQ + q) * FFh;
        };
        auto xhd = [&](int sl) { return c_xhd.data() + (size_t) sl * FF; };
        auto xsd = [&](int sl) { return c_xsd.data() + (size_t) sl * FB; };
        auto accd = [&](int X, int sl, int q) { return c_accd.data() + ((size_t) (X * kSlots + sl) * DQ + q) * Eh; };
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / (2 * GQ))];
                const int r = (j / GQ) % 2, q = j % GQ;
                const kc::Exl3Part p = part(en.e, X, r);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    const float* xt = x + (size_t) en.tok[i] * ldx;
                    for (int b = 2 * q; b < 2 * q + 2; ++b)
                        xsb(X, sl, r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, xt + b * 128, xh(X, sl, r) + b * 128);
                    xp[i] = xh(X, sl, r);
                    ap[i] = acc(X, sl, r, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 16, (q + 1) * 16, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 16, (q + 1) * 16, ap);
            };
        run_nodes(ne * 2 * GQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / GB)];
                const int b = j % GB;
                const kc::Exl3Part pg = part(en.e, X, 0), pu = part(en.e, X, 1), pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float y[2][128], a[128], hb[128];
                    for (int r = 0; r < 2; ++r) {
                        for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                        for (int q = 0; q < GQ; ++q) {
                            const float* aq = acc(X, sl, r, q) + b * 128;
                            for (int c = 0; c < 128; ++c) a[c] += aq[c];
                        }
                        double xs = 0.0;
                        for (int k = 0; k < EB; ++k) xs += xsb(X, sl, r)[k];
                        kc::exl3_cpu_band_out(r == 0 ? pg : pu, a, (float) xs, b * 128, y[r]);
                    }
                    for (int c = 0; c < 128; ++c) {
                        float gv = y[0][c], uv = y[1][c];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        hb[c] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    const int gb = X * GB + b;
                    xsd(sl)[gb] = kc::exl3_cpu_prepare_block(pd.suh + gb * 128, hb, xhd(sl) + gb * 128);
                }
            };
        run_nodes(ne * GB, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DQ)];
                const int q = j % DQ;
                const kc::Exl3Part p = part(en.e, X, 2);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    xp[i] = xhd(en.slot0 + i);
                    ap[i] = accd(X, en.slot0 + i, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 8, (q + 1) * 8, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 8, (q + 1) * 8, ap);
            };
        run_nodes(ne * DQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DB)];
                const int b = j % DB;
                const kc::Exl3Part pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float a[128];
                    for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                    for (int q = 0; q < DQ; ++q) {
                        const float* aq = accd(X, sl, q) + b * 128;
                        for (int c = 0; c < 128; ++c) a[c] += aq[c];
                    }
                    double xs = 0.0;
                    for (int k = 0; k < FB; ++k) xs += xsd(sl)[k];
                    kc::exl3_cpu_band_out(pd, a, (float) xs, b * 128, c_yd.data() + (size_t) sl * E + (size_t) X * Eh + b * 128);
                }
            };
        run_nodes(ne * DB, f);
    }

    // ---- the expert cache.  A swap moves a CPU-lane expert (`in`) into the layer's spare VRAM slot and the VRAM expert it
    // replaces (`out`) into the spare host parts - both copies of weights that do not change, made while the model runs
    // (the copy stream, the host's repacking) - and cache_tick switches the tables between windows, so the device's
    // table and the host's view of a layer always agree within a window.  In place (file_experts) `out` needs no copy:
    // the CPU lane reads it from the file again.
    void ram_loop() {   // the RAM tier's moves in: an expert read from the file into a free slot
        for (;;) {
            RamSwap rs;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                ram_cv.wait(lk, [&] { return quit.load() || !ram_queue.empty(); });
                if (quit.load()) return;
                rs = ram_queue.front();
                ram_queue.pop_front();
            }
            const Layer& l = L[(size_t) rs.layer];
            if (read_mx_file(l, rs.in, ram_slots[(size_t) rs.slot])) {
                for (int r = 0; r < 3; ++r)   // (one copy of it in RAM)
                    os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) rs.in * l.mx_stride[r],
                               mx_part_bytes(l, r), false);
            } else {
                rs.in = -1;   // (failed: the slot goes back)
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_ready.push_back(rs);
        }
    }

    void cache_loop(SwapBufs& B) {
        for (;;) {
            Swap sw;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                sw_cv.wait(lk, [&] { return quit.load() || !sw_queue.empty(); });
                if (quit.load()) return;
                sw = sw_queue.front();
                sw_queue.pop_front();
            }
            Layer& l = L[(size_t) sw.layer];
            if (l.lay.fmt == gg::kFmtMXFP4) {   // in: from the file; out: nothing to keep (the file has it)
                swap_in_mx(l, sw, B);
                std::lock_guard<std::mutex> lk(sw_mu);
                sw_ready.push_back(sw);
                continue;
            }
            parts_to_blob(l, l.hh[(size_t) sw.in], B.a);
            if (l.nm == 1) {
                const Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaStream_t cs = B.st[(size_t) l.dev];
                cudaMemcpyAsync(l.vram_base + (size_t) l.spare_vram * l.blob, B.a, l.blob, cudaMemcpyHostToDevice, cs);
                if (!file_experts)
                    cudaMemcpyAsync(B.b, l.vram_base + (size_t) l.slot[(size_t) sw.out] * l.blob, l.blob,
                                    cudaMemcpyDeviceToHost, cs);
                cudaStreamSynchronize(cs);
            } else {   // a group: each member's part in, each member's part out
                for (int m = 0; m < l.nm; ++m) {
                    const Layer& x = member(l, m);
                    cudaSetDevice(devs[(size_t) x.dev].id);
                    cudaStream_t cs = B.st[(size_t) x.dev];
                    blob_to_half(l, B.a, m, B.h);
                    cudaMemcpyAsync(x.vram_base + (size_t) l.spare_vram * x.hblob, B.h, x.hblob, cudaMemcpyHostToDevice, cs);
                    cudaStreamSynchronize(cs);
                    if (file_experts) continue;
                    cudaMemcpyAsync(B.h, x.vram_base + (size_t) l.slot[(size_t) sw.out] * x.hblob, x.hblob,
                                    cudaMemcpyDeviceToHost, cs);
                    cudaStreamSynchronize(cs);
                    half_to_blob(l, B.h, m, B.b);
                }
            }
            if (!file_experts)
                for (int X = 0; X < NH; ++X) blob_to_part(l, B.b, X, l.spare_host[(size_t) X]);
            if (check_swaps) {   // DS4_CHECK_SWAPS: both moves against the model file, byte for byte
                std::vector<uint8_t> reg, blob(l.blob), part(l.hl.bytes);
                bool ok = true;
                for (int pass = 0; pass < (file_experts ? 1 : 2); ++pass) {   // (in place: no host copy of `out`)
                    const int e = pass == 0 ? sw.in : sw.out;
                    reg.resize(l.region_bytes[(size_t) e]);
                    pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], reg.size(), reg.data());
                    if (pass == 0) {
                        repack(l, reg.data(), blob.data());
                        std::vector<uint8_t> dev(l.blob);
                        if (l.nm == 1)
                            cudaMemcpy(dev.data(), l.vram_base + (size_t) l.spare_vram * l.blob, l.blob, cudaMemcpyDeviceToHost);
                        else
                            for (int m = 0; m < l.nm; ++m) {   // (the members' parts, put together again)
                                std::vector<uint8_t> h(member(l, m).hblob);
                                cudaSetDevice(devs[(size_t) member(l, m).dev].id);
                                cudaMemcpy(h.data(), member(l, m).vram_base + (size_t) l.spare_vram * member(l, m).hblob, member(l, m).hblob,
                                           cudaMemcpyDeviceToHost);
                                half_to_blob(l, h.data(), m, dev.data());
                            }
                        for (int r = 0; r < 3; ++r)
                            ok = ok && std::memcmp(dev.data() + l.lay.trellis[r], blob.data() + l.lay.trellis[r], l.piece_bytes[r][0]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.suh[r], blob.data() + l.lay.suh[r], l.piece_bytes[r][1]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.svh[r], blob.data() + l.lay.svh[r], l.piece_bytes[r][2]) == 0;
                    } else {
                        for (int X = 0; X < NH; ++X) {
                            repack_part(l, reg.data(), X, part.data());
                            for (int r = 0; r < 3; ++r) {
                                const size_t tb = (size_t) (l.hl.k[r] / 16) * (l.hl.n[r] / 16) * 32 * l.lay.K[r];
                                ok = ok && std::memcmp(part.data() + l.hl.trellis[r], l.spare_host[(size_t) X] + l.hl.trellis[r], tb) == 0 &&
                                     std::memcmp(part.data() + l.hl.suh[r], l.spare_host[(size_t) X] + l.hl.suh[r], (size_t) l.hl.k[r] * 2) == 0 &&
                                     std::memcmp(part.data() + l.hl.svh[r], l.spare_host[(size_t) X] + l.hl.svh[r], (size_t) l.hl.n[r] * 2) == 0;
                            }
                        }
                    }
                }
                std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out: %s\n", l.il, sw.in, sw.out,
                             ok ? "both match the model file" : "MISMATCH");
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            sw_ready.push_back(sw);
        }
    }

    // a GGUF's swap: expert sw.in from the file (copy_mode: its copy) into the spare VRAM slot (each member its part);
    // the expert it replaces goes back to the CPU lane, which reads it from the file's mapping - or (copy_mode, sw.ram)
    // from the layer's spare RAM slot, which it is copied into from VRAM here
    void swap_in_mx(Layer& l, Swap& sw, SwapBufs& B) {
        uint8_t *sw_a = B.a, *sw_b = B.b, *sw_h = B.h;   // (this thread's)
        const std::vector<cudaStream_t>& copy_st = B.st;
        bool ok = copy_mode ? read_mx(l, sw.in, sw_a) : read_mx_file(l, sw.in, sw_a);   // (a copy holds its slot till
                                                                                          // this swap's switch)
        for (int m = 0; m < l.nm && ok; ++m) {
            const Layer& x = member(l, m);
            cudaSetDevice(devs[(size_t) x.dev].id);
            cudaStream_t cs = copy_st[(size_t) x.dev];
            if (l.nm > 1) blob_to_half(l, sw_a, m, sw_h);
            ok = cudaMemcpyAsync(x.vram_base + (size_t) l.spare_vram * unit(l), l.nm > 1 ? sw_h : sw_a, unit(l),
                                 cudaMemcpyHostToDevice, cs) == cudaSuccess &&
                 cudaStreamSynchronize(cs) == cudaSuccess;
        }
        if (sw.ram >= 0) {   // `out` from its VRAM slot (each member its part) into the spare RAM slot
            bool got = ok;
            for (int m = 0; m < l.nm && got; ++m) {
                const Layer& x = member(l, m);
                cudaSetDevice(devs[(size_t) x.dev].id);
                cudaStream_t cs = copy_st[(size_t) x.dev];
                got = cudaMemcpyAsync(l.nm > 1 ? sw_h : sw_b, x.vram_base + (size_t) l.slot[(size_t) sw.out] * unit(l),
                                      unit(l), cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
                      cudaStreamSynchronize(cs) == cudaSuccess;
                if (got && l.nm > 1) half_to_blob(l, sw_h, m, sw_b);
            }
            if (got) blob_to_ram(l, sw_b, sw.ram);
            else sw.ram = -1;   // (back to the file's pages then)
        }
        if (check_swaps) {   // DS4_CHECK_SWAPS: the slot (and `out`'s RAM copy) against the model file, byte for byte
            std::vector<uint8_t> dev(l.blob), h(unit(l)), file(l.blob);
            for (int m = 0; m < l.nm && ok; ++m) {
                const Layer& x = member(l, m);
                cudaSetDevice(devs[(size_t) x.dev].id);
                cudaMemcpy(l.nm > 1 ? h.data() : dev.data(), x.vram_base + (size_t) l.spare_vram * unit(l), unit(l),
                           cudaMemcpyDeviceToHost);
                if (l.nm > 1) half_to_blob(l, h.data(), m, dev.data());
            }
            ok = ok && read_mx_file(l, sw.in, file.data());
            for (int r = 0; r < 3 && ok; ++r)
                ok = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            bool ok_out = true;
            if (sw.ram >= 0) {
                ok_out = read_mx_file(l, sw.out, file.data());
                ram_to_blob(l, sw.ram, dev.data());
                for (int r = 0; r < 3 && ok_out; ++r)
                    ok_out = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            }
            std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out%s: %s\n", l.il, sw.in, sw.out,
                         sw.ram >= 0 ? " (into RAM)" : "", ok && ok_out ? "matches the model file" : "MISMATCH");
        } else if (!ok) {
            std::fprintf(stderr, "ds4 cache: layer %d expert %d: the move into VRAM failed\n", l.il, sw.in);
        }
    }

    // after a chunked prompt: the expert cache catches up with what the prompt picked at once - a window prompt gets
    // its swaps one tick at a time over its hundreds of windows, a chunked one has a single tick - so the decode that
    // follows finds the hot experts in VRAM.  Rounds of the ordinary swap rule (the hottest CPU-lane pick of each layer
    // in, its coldest VRAM expert out) until nothing is worth moving or DS4_SETTLE_MS (6000) is spent
    void cache_settle() {
        psettle.settled();
        ++st.prompt_settles;
        if (o.verbose && rx_on)   // (the CT's check: settles == prompts)
            std::fprintf(stderr, "ds4: expert cache: settle %lld (one a prompt with the expert server)\n",
                         (long long) st.prompt_settles);
        if (!o.expert_cache || cache_ths.empty()) return;
        static const double budget = env_f("DS4_SETTLE_MS", 6000.0f);
        if (budget <= 0.0) return;
        const double t0 = now_ms();
        const int64_t s0 = st.swaps;
        while (now_ms() - t0 < budget) {
            for (Layer& l : L) {
                if (l.mtp || l.spare_vram < 0 || l.swap_pending) continue;
                l.cpu_hits.clear();
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && l.heat[(size_t) e] >= 2.0f) l.cpu_hits.push_back(e);
            }
            cache_tick_swaps(0);
            if (sw_inflight == 0) break;   // (nothing queued: every layer's best CPU-lane expert is not hotter than its coldest VRAM one)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (store && env_f("DS4_SETTLE_STORE", 1.0f) != 0.0f) {   // the RAM store too: a window prompt fills it through the CPU
            // lane's picks, a chunked one streams past it - the hottest CPU-lane experts of the prompt into its free slots
            std::vector<std::pair<float, std::pair<int, int>>> hot;
            for (const Layer& l : L) {
                if (l.mtp) continue;
                const uint8_t* srv = rx_row(l, 0);   // (the expert server's: the decode's lane does not read them)
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && (srv == nullptr || srv[e] == 0) && l.heat[(size_t) e] >= 4.0f)
                        hot.push_back({l.heat[(size_t) e], {l.il, e}});
            }
            std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<std::pair<int, int>> list;
            for (const auto& h : hot) {
                if (list.size() >= store->slots_max()) break;
                list.push_back(h.second);
            }
            const double ts = now_ms();
            const size_t n = store->preload(list, 0);
            if (o.verbose) std::fprintf(stderr, "ds4: RAM store: %zu hot experts read after the prompt (%.0f ms)\n", n, now_ms() - ts);
        }
        if (o.verbose && st.swaps > s0)
            std::fprintf(stderr, "ds4: expert cache: %lld swaps after the prompt (%.0f ms)\n", (long long) (st.swaps - s0),
                         now_ms() - t0);
    }
    void cache_tick(int T, int serve_T = -1) {
        cache_tick_swaps(T, serve_T);
        store_sync();
        store_watch();
    }
    // the store gives RAM back when the PC runs short (the owner opened something): below 4 GB free, a 0.7 GB chunk at a time
    // until 5 GB are free again (checked every 16 windows); it grows again by itself once 8 GB beside its chunk are free
    void store_watch() {
        if (!store) return;
        static int tick = 0;
        if (++tick % 16 != 0) return;
        const uint64_t avail = ram_avail();
        if (avail == ~uint64_t{0} || avail >= (4ull << 30)) return;
        const size_t chunks = 1 + (size_t) (((5ull << 30) - avail) / (700ull << 20));
        const size_t n = store->shrink_chunks(chunks);
        if (n > 0) std::fprintf(stderr, "ds4: RAM store: %.1f GB free: gave back %zu chunks (%zu slots left)\n", avail / 1e9, n, store->slots_allocated());
    }
    // (T: the tokens the hits came from; serve_T: the window whose rows rx_row asks about - T, or 0 after a chunked
    // prompt: the decode's)
    void cache_tick_swaps(int T, int serve_T = -1) {
        if (serve_T < 0) serve_T = T;
        if (!o.expert_cache || cache_ths.empty()) return;
        std::vector<Swap> ready;
        std::vector<RamSwap> rready;
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            ready.swap(sw_ready);
            rready.swap(ram_ready);
        }
        for (const RamSwap& rs : rready) {   // (nothing reads the tier between windows)
            --ram_inflight;
            Layer& l = L[(size_t) rs.layer];
            if (rs.in < 0 || l.ram[(size_t) rs.in] >= 0 || l.slot[(size_t) rs.in] >= 0) {
                ram_free.push_back(rs.slot);
                continue;
            }
            l.ram[(size_t) rs.in] = rs.slot;
            if (rs.out >= 0 && l.ram[(size_t) rs.out] >= 0) {
                ram_free.push_back(l.ram[(size_t) rs.out]);
                l.ram[(size_t) rs.out] = -1;
            }
            ++st.ram_moves;
        }
        for (const Swap& sw : ready) {
            Layer& l = L[(size_t) sw.layer];
            const int old_slot = l.slot[(size_t) sw.out];
            const std::array<uint8_t*, 4> old_parts = l.hh[(size_t) sw.in];
            if (copy_mode) {   // (`out` takes the RAM place `in` leaves: through the spare, which `in`'s slot becomes)
                const int s_in = l.ram[(size_t) sw.in];
                l.ram[(size_t) sw.in] = -1;
                if (sw.ram >= 0) {
                    l.ram[(size_t) sw.out] = sw.ram;
                    copy_spare[(size_t) l.il] = s_in;
                } else if (s_in >= 0) {   // (the move into RAM failed: `out` back to the file, one copy fewer)
                    --ram_held;
                    if (copy_spare[(size_t) l.il] < 0) copy_spare[(size_t) l.il] = s_in;
                }
            } else if (!l.ram.empty() && l.ram[(size_t) sw.in] >= 0) {   // (in VRAM now: its RAM slot for the next hottest)
                ram_free.push_back(l.ram[(size_t) sw.in]);
                l.ram[(size_t) sw.in] = -1;
            }
            l.slot[(size_t) sw.in] = l.spare_vram;
            if (!l.zc.empty() && l.zc[(size_t) sw.in]) {   // (VRAM's now: not the store's to compute any more)
                l.zc[(size_t) sw.in] = 0;
                if (l.zc_tab != nullptr) {   // (the device's table too: its entry pointed into the slot the store now reuses - when `in` left VRAM later,
                                             // zc_stage copied whatever lay there, another expert's blob, into the plan)
                    cudaSetDevice(devs[0].id);
                    cudaMemsetAsync(l.zc_tab + sw.in, 0, 8, devs[0].st);
                }
                store->unmark(l.il, sw.in);
            }
            if (l.nm == 1) set_vm(l, sw.in, l.spare_vram);
            l.hh[(size_t) sw.in] = {nullptr, nullptr, nullptr, nullptr};
            l.host[(size_t) sw.in] = nullptr;
            l.slot[(size_t) sw.out] = -1;
            if (l.lay.fmt == gg::kFmtMXFP4) {   // (the file has it)
                l.host[(size_t) sw.out] = (uint8_t*) mx_mat(l, sw.out, 0);
            } else if (file_experts) {   // (back to its pages in the file)
                l.hh[(size_t) sw.out] = {file_part(l, sw.out), nullptr, nullptr, nullptr};
                l.host[(size_t) sw.out] = l.hh[(size_t) sw.out][0];
            } else {
                l.hh[(size_t) sw.out] = l.spare_host;
                l.host[(size_t) sw.out] = l.spare_host[0];
                l.spare_host = old_parts;
            }
            l.spare_vram = old_slot;
            if (!rx_vout.empty()) rx_vout[(size_t) sw.layer] = -1;   // (out has left VRAM: the remote tier's dup rule again)
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const Dev& dm = devs[(size_t) x.dev];
                l.tab_pin[2 * m] = (unsigned long long) (x.vram_base + (size_t) l.slot[(size_t) sw.in] * unit(x));
                l.tab_pin[2 * m + 1] = 0ull;
                cudaSetDevice(dm.id);   // (stream order: before the next window)
                cudaMemcpyAsync(x.vram_tab + sw.in, &l.tab_pin[2 * m], 8, cudaMemcpyHostToDevice, dm.st);
                cudaMemcpyAsync(x.vram_tab + sw.out, &l.tab_pin[2 * m + 1], 8, cudaMemcpyHostToDevice, dm.st);
            }
            l.swap_pending = false;
            --sw_inflight;
            ++st.swaps;
        }
        if (rx_fill) rx_fill_in();
        decay_tokens += T;
        if (decay_tokens >= 256) {   // older use counts less: x0.9 every 256 tokens
            decay_tokens = 0;
            for (Layer& l : L)
                for (float& h : l.heat) h *= c_decay;
        }
        bool queued = false, rqueued = false;
        for (Layer& l : L) {
            if (l.cpu_hits.empty()) continue;
            std::vector<int> hits;
            hits.swap(l.cpu_hits);
            // (the expert server's experts not in after a window it serves, nor at the settle or after a chunked prompt
            // (serve_T 0, for the decode): there they cost the lane nothing, so the slot would be lost - serve_lane
            // lists none of them, the settle and a chunked prompt's do.  After a window it does not serve (DS4_SYNC, T
            // past async_max_t: rx_row null) the lane computed them, so they may; a zero-copy one is this GPU's work,
            // so it may)
            int best = rx::cache_in(hits, l.slot.data(), l.heat.data(), rx_row(l, serve_T), l.zc.empty() ? nullptr : l.zc.data()),
                vin = -1;
            if (best >= 0 && l.spare_vram >= 0 && !l.swap_pending && sw_inflight < c_inflight) {   // VRAM: the hottest in
                int cold = -1;
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0 && (cold < 0 || l.heat[(size_t) e] < l.heat[(size_t) cold])) cold = e;
                if (cold >= 0 && l.heat[(size_t) best] > l.heat[(size_t) cold] * c_margin + c_bonus) {
                    l.swap_pending = true;
                    ++sw_inflight;
                    vin = best;
                    Swap sw{l.il, best, cold};
                    if (copy_mode && l.ram[(size_t) best] >= 0) sw.ram = copy_spare[(size_t) l.il];   // (-1: none)
                    if (!rx_vout.empty()) rx_vout[(size_t) l.il] = cold;
                    std::lock_guard<std::mutex> lk(sw_mu);
                    sw_queue.push_back(sw);
                    queued = true;
                }
            }
            if (rx_fill) rx_tier_add(l, hits, vin);
            if (l.ram.empty() || ram_free.empty() || ram_inflight >= kMaxSwaps) continue;
            // RAM: the hottest read from the file, in place of the layer's coldest held (or a free slot) - not one the
            // expert server holds (the lane does not compute it where the server serves: rx_row, as the VRAM tier)
            const int rb = rx::ram_in(hits, l.slot.data(), l.ram.data(), l.heat.data(), rx_row(l, serve_T), vin);
            if (rb < 0) continue;
            int rc = -1;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.ram[(size_t) e] >= 0 && (rc < 0 || l.heat[(size_t) e] < l.heat[(size_t) rc])) rc = e;
            const bool room = ram_free.size() > 1;   // (slots VRAM freed: filled at once)
            if (!room && !(rc >= 0 && l.heat[(size_t) rb] > l.heat[(size_t) rc] * 1.25f + 1.0f)) continue;
            RamSwap rs;
            rs.layer = l.il;
            rs.in = rb;
            rs.out = room ? -1 : rc;
            rs.slot = ram_free.back();
            ram_free.pop_back();
            ++ram_inflight;
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_queue.push_back(rs);
            rqueued = true;
        }
        if (rx_fill) rx_fill_out();
        if (queued) sw_cv.notify_all();
        if (rqueued) ram_cv.notify_one();
    }

    // lookahead's thread: the ranges the CPU lane asked for, read in (os::prefetch) off the lane, which computes
    void prefetch_loop() {
        std::vector<std::pair<const void*, size_t>> b;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [&] { return quit.load(std::memory_order_relaxed) || !pf_q.empty(); });
                if (quit.load(std::memory_order_relaxed)) return;
                b.swap(pf_q);
            }
            os::prefetch_cold(b);   // (the resident ones would only cost the page walk)
            b.clear();
        }
    }
    // an expert the CPU lane reads from the files (neither in VRAM nor in the RAM tier)
    static bool file_read(const Layer& l, int e) {
        return l.slot[(size_t) e] < 0 && (l.ram.empty() || l.ram[(size_t) e] < 0);
    }
    // the layer whose picks layer l0 guesses a layers ahead (1, 2): the next ones of its stack on its device, MXFP4
    // experts read in place, some of them from the files now (none: nothing to read early - null)
    const Layer* ahead_layer(const Layer& l0, int a) const {
        const int j = l0.il + a, end = l0.mtp ? g.nl + g.n_mtp : g.nl;
        if (j >= end || j >= (int) L.size()) return nullptr;
        const Layer& n = L[(size_t) j];
        if (n.dev != l0.dev || n.mtp != l0.mtp || n.lay.fmt != gg::kFmtMXFP4 ||
            (n.gate_w == nullptr && n.gate_bf == nullptr) || n.n_exp > 1024 || (int) n.slot.size() < n.n_exp)
            return nullptr;
        for (int e = 0; e < n.n_exp; ++e)
            if (file_read(n, e)) return &n;
        return nullptr;
    }
    // lookahead on the CPU lane: request r (layer li, seq) carries the device's guesses for the next layers - the
    // guessed experts read from the files asked for now (each once a window), the guesses kept for the hit counts
    // (count: not a small prompt's window - the experts asked for early are counted for the decode's tokens only)
    std::vector<std::pair<const void*, size_t>> pf_want;
    void ahead_ask(int li, const Req& r, unsigned seq, int T, bool count) {
        pf_want.clear();
        for (int a = 1; a <= 2; ++a) {
            const int m = std::min(r.na[a - 1], a == 1 ? kAhead1 : kPicks), j = li + a;
            if (m <= 0 || j >= (int) L.size()) continue;
            const Layer& n = L[(size_t) j];
            uint8_t* fl = ahead_fl.data() + (size_t) j * 1024;
            uint8_t* rk = ahead_rank.data() + (size_t) j * 1024;
            if (ahead_for[(size_t) j] != seq + (unsigned) a) {
                std::memset(fl, 0, 1024);
                std::memset(rk, 0, 1024);
                ahead_for[(size_t) j] = seq + (unsigned) a;
            }
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < m; ++i) {
                    const int e = r.ahead[t][(a - 1) * kAhead1 + i];
                    if (e < 0 || e >= n.n_exp || e >= (int) n.slot.size()) continue;
                    fl[e] |= a == 2 ? 4 : (i < n.k_exp ? 3 : 2);
                    if (a == 1 && (rk[e] == 0 || rk[e] > i + 1)) rk[e] = (uint8_t) (i + 1);
                    if ((fl[e] & 8) || !file_read(n, e)) continue;
                    fl[e] |= 8;
#if defined(__linux__)
                    if (count && !store && !rx_has.empty() && rx_has[(size_t) j * g.n_exp + e]) {   // (the server's: diagnostics)
                        ++st.rx_ahead;
                        bool res = true;
                        for (int q = 0; q < 3 && res; ++q) res = os::resident_sampled(mx_mat(n, e, q), mx_part_bytes(n, q), 4);
                        st.rx_ahead_cold += res ? 0 : 1;
                    }
#endif
                    if (store) store->prefetch(j, e);
                    else
                        for (int q = 0; q < 3; ++q) pf_want.push_back({mx_mat(n, e, q), mx_part_bytes(n, q)});
                    st.ahead_pf += count ? 1 : 0;
                }
        }
        if (pf_want.empty()) return;
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_q.insert(pf_q.end(), pf_want.begin(), pf_want.end());
        }
        pf_cv.notify_one();
    }

    // ---- expert parallelism H' (DS4_EP_K=k, 1 <= k < nb; ep-u6-plan.md): which layers split (rx_split_l) - every trunk
    // layer, each one the server's Geo serves, on the i-quant CPU lane - given a server whose parts take k (its hello's lo
    // <= k).  Any of it not so: a line why, and k stays nb (U4 as it is: the server's GPU experts alone).  The layers'
    // ep_ff0 / ep_ffn then say [0, 256 k), as ep_slices_ok checks them (tagged: the lane slices only a pick it sent tagged)
    void rx_split(const rx::Main& link, const std::vector<rx::GeoLayer>& gl) {
        const char* kv = std::getenv("DS4_EP_K");
        if (kv == nullptr || *kv == '\0') return;
        const int k = std::atoi(kv);
        if (k >= rx_nb) return;   // (nb: off)
        std::string why;
        if (k < 1) why = std::string("DS4_EP_K=") + kv + " is not 1 to " + std::to_string(rx_nb);
        else if (g.ff % 256 != 0) why = "n_ff " + std::to_string(g.ff) + " is not whole 256-value blocks";
        else if (link.lo() <= 0) why = "the expert server has no parts (--expert-server-parts)";
        else if (link.lo() > k)
            why = "the expert server's parts hold blocks [" + std::to_string(link.lo()) + ", " + std::to_string(rx_nb) +
                  "), not [" + std::to_string(k) + ", " + std::to_string(rx_nb) + ")";
        std::vector<uint8_t> sl(L.size(), 0);
        for (size_t li = 0; why.empty() && li < gl.size() && li < L.size(); ++li) {
            const Layer& l = L[li];
            if (gl[li].blob == 0) why = "layer " + std::to_string(li) + "'s experts are not the expert server's (none in the Geo)";
            else if (!(l.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(l.lay)))
                why = "layer " + std::to_string(li) + "'s CPU lane computes its experts whole (only the GGUF i-quants slice)";
            sl[li] = 1;
        }
        if (why.empty()) {
            for (size_t li = 0; li < sl.size(); ++li)
                if (sl[li]) {
                    L[li].ep_ff0 = 0;
                    L[li].ep_ffn = 256 * k;
                }
            rx_tagged = true;
            if (!ep_slices_ok(why)) {
                for (Layer& l : L) {
                    l.ep_ff0 = 0;
                    l.ep_ffn = g.ff;
                }
                rx_tagged = false;
            }
        }
        if (!why.empty()) {
            std::fprintf(stderr, "ds4: expert parallelism H' (DS4_EP_K=%s): %s - not split (the server's GPU experts alone)\n",
                         kv, why.c_str());
            return;
        }
        rx_k = k;
        rx_split_l = sl;
        const char* vv = std::getenv("DS4_EP_SPLIT_VERIFY");
        rx_verify = vv != nullptr && *vv != '\0' ? std::max(0, std::min(2, std::atoi(vv))) : 0;
        std::fprintf(stderr, "ds4: expert parallelism H': k %d - an EP window's CPU-lane picks that go to the expert server "
                             "split at n_ff block %d of %d: [0, %d) here, [%d, %d) on its CPU lane (%d workers; its parts "
                             "from block %d)%s\n",
                     k, k, rx_nb, k, k, rx_nb, link.lane_threads(), link.lo(),
                     rx_verify == 2 ? "; DS4_EP_SPLIT_VERIFY=2: the server's slice sums wrong on purpose (the check's check)"
                     : rx_verify    ? "; DS4_EP_SPLIT_VERIFY=1: each reply's slice sums checked here"
                                    : "");
        // the hybrid (U5c): the server's W - a split layer's CPU-lane pick of W goes tagged pool, whole on its CPU lane,
        // and this lane skips it.  Not with DS4_EP_SPLIT_VERIFY (its check is of [k, nb) alone); DS4_EP_POOL=0: off
        const char* pv = std::getenv("DS4_EP_POOL");
        if (link.pool().empty() || (pv != nullptr && *pv == '0') || rx_verify > 0) {
            if (!link.pool().empty())
                std::fprintf(stderr, "ds4: the hybrid: the server's W of %zu experts not used (%s)\n", link.pool().size(),
                             rx_verify > 0 ? "DS4_EP_SPLIT_VERIFY" : "DS4_EP_POOL=0");
            return;
        }
        rx_pool.assign(L.size() * (size_t) g.n_exp, 0);
        int64_t n = 0;
        for (uint32_t id : link.pool()) {
            const size_t li = id >> 16, e = id & 0xFFFFu;
            if (li >= L.size() || !sl[li] || e >= (size_t) g.n_exp || L[li].mtp || L[li].n_exp != g.n_exp) continue;
            rx_pool[li * (size_t) g.n_exp + e] = 1;
            ++n;
        }
        st.rx_pool_w = n;
        const char* bv = std::getenv("DS4_EP_POOL_BAL");
        rx_bal = bv == nullptr || *bv != '0';
        const char* cv = std::getenv("DS4_EP_POOL_CUT");
        rx_cut = rx_bal && (cv == nullptr || *cv != '0');
        const char* chv = std::getenv("DS4_EP_POOL_CUT_CHECK");
        rx_cut_chk = rx_cut && chv != nullptr && *chv == '1';
        if (const char* mv = std::getenv("DS4_EP_POOL_MARGIN"); mv != nullptr && *mv != '\0')
            rx_bal_margin = std::max(0.0, std::atof(mv));
        char bal[256];
        if (rx_bal)
            std::snprintf(bal, sizeof bal, "; a decode request's W picks there only while the reply should come in before "
                                           "this lane's part (DS4_EP_POOL_BAL, margin %.2f ms)%s", rx_bal_margin,
                          rx_cut ? ", by the block: a pick's [cut, k) there, [0, cut) here (DS4_EP_POOL_CUT)"
                                 : ", whole picks (DS4_EP_POOL_CUT=0)");
        else
            std::snprintf(bal, sizeof bal, "; all of them (DS4_EP_POOL_BAL=0)");
        std::fprintf(stderr, "ds4: the hybrid (the server's DS4_RX_POOL): its W of %lld experts - a CPU-lane pick of W "
                             "goes whole to its CPU lane ([0, %d) from its pool, [%d, %d) from its parts), not computed "
                             "here (a late reply: here whole)%s\n", (long long) n, k, k, rx_nb, bal);
    }

    // ---- remote experts (--remote-experts, ds4_rx.hpp): an expert server's GPU holds some of the CPU lane's experts
    // the load: wait for the server, send it the geometry and the hottest CPU-lane experts by the profile that fit its
    // room (none came, or the link failed: the lane computes all of them, as without the option)
    void rx_setup() {
        if (o.remote_experts.empty()) return;
        rx_nb = std::max(1, g.ff / 256);
        rx_k = rx_nb;   // (H' off until rx_split says otherwise)
        if (const char* v = std::getenv("DS4_RX_TIMEOUT_MS"); v != nullptr && *v != '\0') {
            rx_timeout_ms = std::max(0.05, std::atof(v));
            rx_wait_fixed = true;
        }
        if (const char* v = std::getenv("DS4_RX_BACKOFF"); v != nullptr && *v != '\0') rx_back.hold = std::max(1, std::atoi(v));
        const char* wv = std::getenv("DS4_RX_WAIT_S");
        const int wait_s = wv != nullptr && *wv != '\0' ? std::max(1, std::atoi(wv)) : 120;
        const char* mv_ = std::getenv("DS4_RX_MAX");   // (tests: at most this many experts)
        const int64_t max_n = mv_ != nullptr && *mv_ != '\0' ? std::atoll(mv_) : INT64_MAX;
        const char* tv = std::getenv("STRATA_STAGE_TOKEN");
        const std::string token = tv != nullptr ? tv : "";
        if (token.empty() && o.remote_experts.rfind("127.0.0.1:", 0) != 0) {   // (no token: loopback only)
            std::fprintf(stderr, "ds4: remote experts: no STRATA_STAGE_TOKEN, refusing to listen on %s (only 127.0.0.1:PORT "
                                 "without one) - the CPU lane computes all its experts\n", o.remote_experts.c_str());
            return;
        }
        std::fprintf(stderr, "ds4: remote experts: waiting up to %d s for an expert server on %s\n", wait_s,
                     o.remote_experts.c_str());
        auto link = std::make_unique<rx::Main>();
        std::string e;
        if (!link->accept(o.remote_experts, token, wait_s, e)) {
            std::fprintf(stderr, "ds4: remote experts: %s - the CPU lane computes all its experts\n", e.c_str());
            return;
        }
        const int NL = std::min(g.nl, (int) L.size());
        rx::GeoMsg gm;
        gm.n_embd = g.dim;
        gm.n_ff = g.ff;
        gm.n_layers = NL;
        gm.limit = g.lim;
        std::vector<rx::GeoLayer> gl((size_t) NL);
        for (int li = 0; li < NL; ++li) {   // (the layers here whose experts the device kernels take: the others none)
            const Layer& l = L[(size_t) li];
            if (l.mtp || elsewhere(l) || l.cpu_only || l.lay.fmt != gg::kFmtMXFP4 || l.n_exp != g.n_exp ||
                (int) l.slot.size() < l.n_exp)
                continue;
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                gl[(size_t) li].K[r] = gg::expert_type(l.lay, r);
                gl[(size_t) li].off[r] = off;
                off += align256(mx_part_bytes(l, r));
            }
            gl[(size_t) li].blob = off;
        }
        gm.pack_hash = stage::pack_hash(o.pack);   // (a server's parts must be cut from this pack: refused at the Geo if not)
        // the k rx_split tries (DS4_EP_K; else nb: no split, so a server's parts that do not fit are only its warning)
        const char* kv = std::getenv("DS4_EP_K");
        const int kw = kv != nullptr && *kv != '\0' ? std::atoi(kv) : rx_nb;
        gm.k = kw >= 1 && kw < rx_nb ? kw : rx_nb;
        struct Cand {
            uint32_t n;
            int li, e;
        };
        std::vector<Cand> cand;   // the CPU lane's experts (not in VRAM here), the most routed first
        double all_n = 0;
        for (int li = 0; li < NL; ++li) {
            if (gl[(size_t) li].blob == 0) continue;
            const Layer& l = L[(size_t) li];
            for (int x = 0; x < l.n_exp; ++x)
                if (l.slot[(size_t) x] < 0) {
                    cand.push_back({prof[(size_t) li * g.n_exp + x], li, x});
                    all_n += (double) cand.back().n;
                }
        }
        std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.n > b.n; });
        // DS4_RX_HELD_LOAD=FILE (a set DS4_RX_HELD_SAVE wrote): its experts first, in its order, the room left by the
        // profile - the server starts from the set an earlier run's fills reached, not the profile's
        const char* hl = std::getenv("DS4_RX_HELD_LOAD");
        std::vector<int> hrank;   // [NL * n_exp] the file's order, -1 not in it
        std::string hnote;
        const int hn = hl != nullptr && *hl != '\0' ? rx_held_read(hl, gm.pack_hash, NL, hrank, hnote) : -1;
        if (hn > 0)
            std::stable_sort(cand.begin(), cand.end(), [&](const Cand& a, const Cand& b) {
                const int ra = hrank[(size_t) a.li * g.n_exp + a.e], rb = hrank[(size_t) b.li * g.n_exp + b.e];
                if ((ra >= 0) != (rb >= 0)) return ra >= 0;
                return ra >= 0 && ra < rb;   // (neither in it: the profile's order, kept)
            });
        const uint64_t room = link->room();
        uint64_t used = 0;
        double sent_n = 0;
        std::vector<Cand> pick;
        for (const Cand& c : cand) {
            if ((int64_t) pick.size() >= max_n) break;
            const uint64_t b = gl[(size_t) c.li].blob;
            if (used + b > room) continue;
            pick.push_back(c);
            used += b;
            sent_n += (double) c.n;
        }
        std::fprintf(stderr, "ds4: remote experts: server %s (%s, %.2f GB of VRAM for experts): sending %zu of the CPU "
                             "lane's %zu experts (%.2f GB, %.1f%% of their routings in the profile)\n",
                     link->peer().c_str(), link->name().c_str(), (double) room / 1e9, pick.size(), cand.size(),
                     (double) used / 1e9, all_n > 0 ? 100.0 * sent_n / all_n : 0.0);
        if (hn > 0) {
            size_t lane = 0, sent = 0;
            for (const Cand& c : cand) lane += hrank[(size_t) c.li * g.n_exp + c.e] >= 0;
            for (const Cand& c : pick) sent += hrank[(size_t) c.li * g.n_exp + c.e] >= 0;
            std::fprintf(stderr, "ds4: DS4_RX_HELD_LOAD=%s: %d experts in it, %zu of them the CPU lane's here, %zu of them "
                                 "sent first; the other %zu by the profile\n", hl, hn, lane, sent, pick.size() - sent);
        } else if (hn == 0 || (hn < 0 && hl != nullptr && *hl != '\0')) {
            std::fprintf(stderr, "ds4: DS4_RX_HELD_LOAD=%s not used (%s) - the profile's set\n", hl, hnote.c_str());
        }
        // (lead 2026-10-25: what the upload does to the page cache) the CPU lane's other experts not in it, before and
        // after (4 pages a matrix sampled), the bytes the upload read from the disk, the memory around it
#if defined(__linux__)
        std::vector<uint8_t> picked((size_t) NL * g.n_exp, 0);
        for (const Cand& c : pick) picked[(size_t) c.li * g.n_exp + c.e] = 1;
        auto cold_others = [&]() {
            long long n = 0;
            for (const Cand& c : cand) {
                if (picked[(size_t) c.li * g.n_exp + c.e]) continue;
                const Layer& l = L[(size_t) c.li];
                bool res = true;
                for (int r = 0; r < 3 && res; ++r) res = os::resident_sampled(mx_mat(l, c.e, r), mx_part_bytes(l, r), 4);
                n += res ? 0 : 1;
            }
            return n;
        };
        const long long cold0 = cold_others();
        const std::string mem0 = rx_mem_note();
#endif
        uint64_t up_cold = 0;
        const double t0 = now_ms();
        if (!link->send_geo(gm, gl, e)) {
            std::fprintf(stderr, "ds4: remote experts: %s - the CPU lane computes all its experts\n", e.c_str());
            return;
        }
        std::vector<uint8_t> blob;
        for (size_t i = 0; i < pick.size(); ++i) {
            const Layer& l = L[(size_t) pick[i].li];
            const rx::GeoLayer& G = gl[(size_t) pick[i].li];
            blob.assign((size_t) G.blob, 0);
            for (int r = 0; r < 3; ++r) {   // (each matrix whole: its NH parts in order, from wherever they are)
                const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                for (int X = 0; X < NH; ++X) {
                    up_cold += os::cold_bytes(mx_part_ptr(l, pick[i].e, r, X), nb);
                    std::memcpy(blob.data() + G.off[r] + (size_t) X * nb, mx_part_ptr(l, pick[i].e, r, X), nb);
                }
            }
            if (!link->send_blob(pick[i].li, pick[i].e, blob.data(), blob.size(), e)) {
                std::fprintf(stderr, "ds4: remote experts: %s - the CPU lane computes all its experts\n", e.c_str());
                return;
            }
        }
        std::vector<uint8_t> ok;
        std::vector<uint32_t> on_gpu;   // (this GPU's experts: never the CPU lane's, none for the server's W - U5c)
        for (int li = 0; li < NL; ++li) {
            const Layer& l = L[(size_t) li];
            if (gl[(size_t) li].blob == 0) continue;
            for (int x = 0; x < l.n_exp; ++x)
                if (l.slot[(size_t) x] >= 0) on_gpu.push_back((uint32_t) li << 16 | (uint32_t) x);
        }
        if (!link->end_blobs((int) pick.size(), ok, 120000.0, e, on_gpu)) {
            std::fprintf(stderr, "ds4: remote experts: %s - the CPU lane computes all its experts\n", e.c_str());
            return;
        }
        rx_has.assign((size_t) NL * g.n_exp, 0);
        int64_t held = 0;
        uint64_t held_b = 0;
        for (size_t i = 0; i < pick.size(); ++i)
            if (ok[i]) {
                rx_has[(size_t) pick[i].li * g.n_exp + pick[i].e] = 1;
                ++held;
                held_b += gl[(size_t) pick[i].li].blob;
            }
        st.rx_held = held;
        st.rx_upload_bytes = (int64_t) held_b;
        st.rx_upload_ms = now_ms() - t0;
        rx_split(*link, gl);
#if defined(__linux__)
        std::fprintf(stderr, "ds4: remote experts: the upload read %.2f of its %.2f GB from the disk (not in the page "
                             "cache before it); the CPU lane's other %zu experts not in the page cache (4 pages a matrix "
                             "sampled): %lld before, %lld after; before %s; after %s\n",
                     (double) up_cold / 1e9, (double) used / 1e9, cand.size() - pick.size(), cold0, cold_others(),
                     mem0.c_str(), rx_mem_note().c_str());
#else
        (void) up_cold;
#endif
        if (rx_wait_fixed)
            std::fprintf(stderr, "ds4: remote experts: %lld experts (%.2f GB) in the server's VRAM, uploaded in %.1f s (%.0f "
                                 "MB/s); a reply later than %.2f ms after the lane's own picks (DS4_RX_TIMEOUT_MS): its "
                                 "picks computed here\n",
                         (long long) held, (double) held_b / 1e9, st.rx_upload_ms / 1000.0,
                         st.rx_upload_ms > 0 ? (double) held_b / 1e3 / st.rx_upload_ms : 0.0, rx_timeout_ms);
        else
            std::fprintf(stderr, "ds4: remote experts: %lld experts (%.2f GB) in the server's VRAM, uploaded in %.1f s (%.0f "
                                 "MB/s); a reply not in once the lane's own picks are done and as long again as the lane "
                                 "would take for its picks (%.1f ms at least; %.2f ms until the lane's time a pick is "
                                 "known and the link has given a kernel arrival time - always without them, Windows): "
                                 "its picks computed here\n",
                         (long long) held, (double) held_b / 1e9, st.rx_upload_ms / 1000.0,
                         st.rx_upload_ms > 0 ? (double) held_b / 1e3 / st.rx_upload_ms : 0.0, rx::Gate::kMinWaitMs,
                         rx_timeout_ms);
        {   // the gate's start: 16 real 1-row requests (one pick of the hottest expert held, x of 0.01: the frame and
            // the server's GPU work of a decode window; the first also warms a cold server, whose first reply took 178
            // ms in a slot), their median round trip (send to the kernel's arrival on Linux) seeds the estimate
            size_t hi = 0;
            while (hi < pick.size() && !ok[hi]) ++hi;
            std::vector<double> pr;
            std::vector<float> wx((size_t) g.dim, 0.01f);
            const float* wxs[1] = {wx.data()};
            rx::ReqRow wr;
            wr.n = 1;
            wr.w[0] = 1.0f;
            for (uint32_t i = 0; hi < pick.size() && i < 16; ++i) {
                wr.e[0] = (int16_t) pick[hi].e;
                const uint16_t* yw = nullptr;
                int su = 0;
                const double s = now_ms();
                if (!link->send_req(0xFFFF0000u + i, pick[hi].li, 1, &wr, wxs, e)) break;
                const int wg = link->wait_resp(0xFFFF0000u + i, 1, s + 1000.0, &yw, su, e);
                if (wg < 0) break;
                if (wg == 0) continue;
                const double r = now_ms(), a = link->arrival_ms();
                pr.push_back(a >= s && a <= r ? a - s : r - s);
            }
            if (pr.size() >= 8) {
                std::sort(pr.begin(), pr.end());
                rx_seed_ms = pr[pr.size() / 2];
                rx_gate.seed(1, rx_seed_ms);
                std::fprintf(stderr, "ds4: remote experts: %zu warm-up requests of 1 row: round trip median %.3f ms (%.3f "
                                     "- %.3f): the gate's estimate at 1 row starts there\n",
                             pr.size(), rx_seed_ms, pr.front(), pr.back());
            } else {
                std::fprintf(stderr, "ds4: remote experts: %zu of 16 warm-up requests answered%s%s: the gate learns from "
                                     "the first requests\n", pr.size(), e.empty() ? "" : " - ", e.c_str());
            }
        }
        st.rx_bytes_out = (int64_t) link->bytes_out();   // (the per-token bytes count from here)
        st.rx_bytes_in = (int64_t) link->bytes_in();
        rx_link = std::move(link);
        rx_on = held > 0 || rx_k < rx_nb;   // (H': the slice picks need no blob held)
        if (const char* fv = std::getenv("DS4_RX_FILL"); fv != nullptr && *fv == '1' && held > 0) rx_fill_open(token, gl);
        rx_rtt.reserve(1 << 20);
        rx_wait.reserve(1 << 20);
        rx_arr.reserve(1 << 20);
    }
    // the held set's file (DS4_RX_HELD_SAVE / _LOAD): "strata-dsv41 held set 1", "pack <hash> layers <NL> experts
    // <n_exp>", then a "<layer> <expert>" line an expert, the hottest first.  rank[li * n_exp + e]: its line, or -1;
    // the experts read (0: none, -1: the file not used, `note` says why)
    int rx_held_read(const char* path, uint64_t pack, int NL, std::vector<int>& rank, std::string& note) const {
        std::FILE* f = std::fopen(path, "r");
        if (f == nullptr) {
            note = "cannot open it";
            return -1;
        }
        char ln[256];
        unsigned long long ph = 0;
        int nl = 0, ne = 0;
        const bool head = std::fgets(ln, sizeof ln, f) != nullptr && std::strncmp(ln, "strata-dsv41 held set 1", 23) == 0 &&
                          std::fgets(ln, sizeof ln, f) != nullptr &&
                          std::sscanf(ln, "pack %llx layers %d experts %d", &ph, &nl, &ne) == 3;
        if (!head || (uint64_t) ph != pack || nl != NL || ne != g.n_exp) {
            std::fclose(f);
            note = !head ? "not a held set" : "another pack's or another geometry's";
            return -1;
        }
        rank.assign((size_t) NL * g.n_exp, -1);
        int n = 0, li = 0, e = 0;
        while (std::fgets(ln, sizeof ln, f) != nullptr)
            if (std::sscanf(ln, "%d %d", &li, &e) == 2 && li >= 0 && li < NL && e >= 0 && e < g.n_exp &&
                rank[(size_t) li * g.n_exp + e] < 0)
                rank[(size_t) li * g.n_exp + e] = n++;
        std::fclose(f);
        if (n == 0) note = "no experts in it";
        return n;
    }
    // (DS4_RX_HELD_SAVE) the experts the server holds now, or whose fill is under way, into `path` (a temporary
    // beside it, then renamed), the hottest here first
    bool rx_held_save(const std::string& path, std::string& note) const {
        if (rx_has.empty() || !rx_link) {
            note = "no expert server";
            return false;
        }
        const int NL = (int) (rx_has.size() / (size_t) g.n_exp);
        std::vector<std::pair<float, std::pair<int, int>>> h;
        for (int li = 0; li < NL; ++li)
            for (int e = 0; e < g.n_exp; ++e)
                if (rx_has[(size_t) li * g.n_exp + e] != 0)   // (kHeld, or kFilling: chosen, its fill under way)
                    h.push_back({li < (int) L.size() && e < (int) L[(size_t) li].heat.size() ? L[(size_t) li].heat[(size_t) e] : 0.0f,
                                 {li, e}});
        std::stable_sort(h.begin(), h.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        const std::string tmp = path + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            note = "cannot write " + tmp;
            return false;
        }
        std::fprintf(f, "strata-dsv41 held set 1\npack %llx layers %d experts %d\n",
                     (unsigned long long) stage::pack_hash(o.pack), NL, g.n_exp);
        for (const auto& x : h) std::fprintf(f, "%d %d\n", x.second.first, x.second.second);
        const bool ok = std::fclose(f) == 0;
        std::remove(path.c_str());   // (Windows' rename does not replace)
        if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
            note = "cannot write " + path;
            return false;
        }
        note = std::to_string(h.size()) + " experts";
        return true;
    }
    // (DS4_RX_FILL=1) the fills' socket after the upload: DS4_RX_FILL_PACE GB/s (0.6), DS4_RX_FILL_R fills a token (6),
    // DS4_RX_FILL_Q pending (2), DS4_RX_FILL_PORT (the rx port + 1).  A server that takes none: the set stays the load's
    void rx_fill_open(const std::string& token, const std::vector<rx::GeoLayer>& gl) {
        const double pace = (double) env_f("DS4_RX_FILL_PACE", 0.6f) * 1e9;
        const int R = std::max(1, (int) env_f("DS4_RX_FILL_R", 6.0f)), Q = std::max(1, (int) env_f("DS4_RX_FILL_Q", 2.0f));
        const int port = (int) env_f("DS4_RX_FILL_PORT", 0.0f);
        rx_fgeo.assign(gl.size(), rx::FillGeo());
        rx_fill_max = 0.0;
        for (size_t li = 0; li < gl.size(); ++li) {
            if (gl[li].blob == 0) continue;
            const Layer& l = L[li];
            const uint64_t rb[3] = {mx_row_b(l, 0), mx_row_b(l, 1), mx_row_b(l, 2)};
            rx_fgeo[li] = rx::FillGeo(gl[li], g.dim, g.ff, rb);
            rx_fill_max = std::max(rx_fill_max, (double) gl[li].blob);   // (the largest fill: a whole blob)
        }
        auto fl = std::make_unique<rx::FillLink>();
        // DS4_RX_FILL_ZC: a split fill's gate and up from the pack - 1 the mapping, 2 sendfile from its file (0: gathered)
        fl->zc = std::clamp((int) env_f("DS4_RX_FILL_ZC", 0.0f), 0, 2);
        std::string e;
        const int cpu = rx::fill_cpu();
        if (!fl->open(*rx_link, o.remote_experts, port, token, pace, cpu, 3000, e)) {
            std::fprintf(stderr, "ds4: remote experts: no fills (%s) - the server keeps the load's experts\n", e.c_str());
            return;
        }
        rx_book = std::make_unique<rx::FillBook>(R, Q, 0.0, 0.0, rx_fill_max);
        rx_vout.assign(L.size(), -1);
        rx_thits.assign(L.size(), {});
        std::fprintf(stderr, "ds4: remote experts: fills on (port %d, %.2f GB/s, %d a token, %d pending, %s; the fill "
                             "thread on CPU %d)\n", fl->port(), pace / 1e9, R, Q,
                     rx_k < rx_nb ? ("the main's [0, " + std::to_string(rx_k) + ") of a split layer").c_str()
                                  : "whole blobs", cpu);
        if (fl->zc > 0)
            std::fprintf(stderr, "ds4: DS4_RX_FILL_ZC=%d: a split fill's gate and up %s, its downs gathered\n", fl->zc,
                         fl->zc == 2 ? "through sendfile from the pack's file" : "sent straight from the pack's mapping");
        rx_fill = std::move(fl);
    }
    // (a token boundary, cache_tick_swaps) the fills the server loaded since the last one: theirs from now; the link
    // gone: those under way back to the lane, no more fills
    void rx_fill_in() {
        for (const rx::FillLink::Done& d : rx_fill->take()) {
            rx_book->done();
            if (d.layer >= 0 && (size_t) (d.layer + 1) * g.n_exp <= rx_has.size() && d.in >= 0 && d.in < g.n_exp)
                rx::fill_landed(rx_has.data() + (size_t) d.layer * g.n_exp, d.in, d.status == 0);
            ++(d.status == 0 ? st.rx_fill_ok : st.rx_fill_fail);
        }
        st.rx_fill_bytes = (int64_t) rx_fill->bytes.load();
        st.rx_fill_ms = (double) rx_fill->send_us.load() / 1000.0;
        st.rx_fill_direct = (int64_t) rx_fill->direct.load();
        st.rx_fill_filed = (int64_t) rx_fill->filed.load();
        st.rx_fill_zc = rx_fill->zc;
        if (rx_fill->zc == 2 && !rx_zc_noted) {   // (sendfile refused once: said once)
            const std::string n = rx_fill->zc_note();
            if (!n.empty()) {
                std::fprintf(stderr, "ds4: DS4_RX_FILL_ZC=2: %s - the gate and up from the mapping from now on\n", n.c_str());
                rx_zc_noted = true;
            }
        }
        if (rx_fill->up() && rx_on) return;
        const std::string why = rx_fill->up() ? "the link given up" : rx_fill->why();
        rx_fill->close();
        for (uint8_t& h : rx_has)
            if (h == rx::kFilling) h = 0;
        std::fprintf(stderr, "ds4: remote experts: fills off (%s)\n", why.c_str());
        rx_fill.reset();
    }
    // (cache_tick_swaps, a layer's hits) the layer as the remote tier sees it this tick
    void rx_tier_add(const Layer& l, const std::vector<int>& hits, int vin) {
        if (l.mtp || l.n_exp != g.n_exp || (size_t) l.il >= rx_fgeo.size() || rx_fgeo[(size_t) l.il].blob == 0 ||
            (size_t) (l.il + 1) * g.n_exp > rx_has.size())
            return;
        rx_thits[(size_t) l.il] = hits;
        rx::TierLayer t;
        t.layer = l.il;
        t.hits = &rx_thits[(size_t) l.il];
        t.slot = l.slot.data();
        t.ram = l.ram.empty() ? nullptr : l.ram.data();
        t.zc = l.zc.empty() ? nullptr : l.zc.data();
        t.heat = l.heat.data();
        t.has = rx_has.data() + (size_t) l.il * g.n_exp;
        t.n_exp = l.n_exp;
        t.vin = vin;
        t.vout = rx_vout[(size_t) l.il];
        const bool spl = rx_k < rx_nb && (size_t) l.il < rx_split_l.size() && rx_split_l[(size_t) l.il];
        t.bytes = (double) rx_fgeo[(size_t) l.il].ship_bytes(spl ? rx_k : 0);
        if (t.bytes > rx_fill_max) {   // (the book's burst is sized by the largest fill)
            std::fprintf(stderr, "ds4: remote experts: layer %d's fill (%.0f bytes) past the largest (%.0f)\n", l.il,
                         t.bytes, rx_fill_max);
            std::abort();
        }
        rx_tier.push_back(t);
    }
    // (cache_tick_swaps, after the layers) this tick's fills: chosen (out off the server's list, in kFilling), queued
    void rx_fill_out() {
        rx_book->tick(now_ms(), rx_back.left > 0);
        for (const rx::Fill& f : rx::remote_choose(rx_tier, *rx_book, c_margin, c_bonus)) {
            const Layer& l = L[(size_t) f.layer];
            rx::FillLink::Job j;
            j.seq = ++rx_fill_seq;
            j.layer = f.layer;
            j.in = f.in;
            j.out = f.out;
            const bool spl = rx_k < rx_nb && (size_t) l.il < rx_split_l.size() && rx_split_l[(size_t) l.il];
            j.k = spl ? rx_k : 0;
            j.geo = rx_fgeo[(size_t) f.layer];
            for (int r = 0; r < 3; ++r) {   // (the file's: `in` is in neither VRAM nor RAM, and the mapping outlives the link)
                j.foff[r] = l.mx_off[r] + (uint64_t) f.in * l.mx_stride[r];
                j.src[r] = pk.at(l.mx_file[r], j.foff[r]);
                const size_t fi = (size_t) l.mx_file[r];
                j.fd[r] = fi < pk.fds.size() && pk.fds[fi] != os::kBadFd ? (std::intptr_t) pk.fds[fi] : -1;
            }
            rx_fill->queue(j);
            ++st.rx_fills;
        }
        rx_tier.clear();
    }
    // layer l's experts the expert server holds, for the hits of a window of T rows (0: the settle, the tick after a
    // chunked prompt or a window prompt's ticks - the decode's) ([n_exp]; null:
    // none, the link given up - the lane's again -, or a decode window the server does not serve: DS4_SYNC, T past
    // async_max_t - rx::lane_serves - whose CPU lane computed them itself)
    const uint8_t* rx_row(const Layer& l, int T) const {
        return rx_on && rx::lane_serves(T, par, sync_decode, async_max_t) && !l.mtp && l.n_exp == g.n_exp &&
                       (size_t) (l.il + 1) * g.n_exp <= rx_has.size()
                   ? rx_has.data() + (size_t) l.il * g.n_exp
                   : nullptr;
    }
    int64_t rx_in_vram() const {   // (the server's experts in this GPU's VRAM too)
        int64_t n = 0;
        for (const Layer& l : L)
            if (!l.mtp && l.n_exp == g.n_exp && (size_t) (l.il + 1) * g.n_exp <= rx_has.size())
                for (int e = 0; e < l.n_exp; ++e) n += l.slot[(size_t) e] >= 0 && rx_has[(size_t) l.il * g.n_exp + e];
        return n;
    }
    // the link given up: the lane computes every pick from now on (said once)
    void rx_lost(const std::string& why) {
        if (!rx_on.exchange(false)) return;
        if (rx_link) rx_link->close();
        std::fprintf(stderr, "ds4: remote experts: %s - the CPU lane computes all its experts from now on\n", why.c_str());
    }
    // (the tokens with a remote pick: the server's whole ones, then its slice ones)
    static int rx_nrows(const std::vector<std::pair<int, int>>& rpk, const std::vector<std::pair<int, int>>& spk = {}) {
        unsigned m = 0;
        for (const auto& q : rpk) m |= 1u << (q.second / kPicks);
        for (const auto& q : spk) m |= 1u << (q.second / kPicks);
        return (int) std::bitset<32>(m).count();
    }
    // the picks a request's server computes, as picks of the lane: the whole ones and the slice ones' [k, nb) share
    // (rx::server_range) - the gate's and the deadline's count
    size_t rx_npicks(size_t whole, size_t slice) const {
        return whole + rx::slice_share(slice, rx_k, rx_nb);
    }
    // request seq's picks the server holds ((expert, t * kPicks + i), in token order) and its slice picks (spk: tagged,
    // from block rx_k) as rows of r's tokens
    // (wpk: the request's W picks - tagged slice and pool, [cut, nb) on the server's CPU lane: whole at cut 0)
    bool rx_send(unsigned seq, int li, int T, const Req& r, const std::vector<std::pair<int, int>>& rpk,
                 const std::vector<std::pair<int, int>>& spk, const std::vector<std::pair<int, int>>& wpk = {},
                 int cut = 0) {
        rx_rows.clear();
        const float* xs[kWin];
        int row_of[kWin];
        for (int t = 0; t < kWin; ++t) row_of[t] = -1;
        for (int pass = 0; pass < 3; ++pass)
            for (const auto& q : pass == 0 ? rpk : pass == 1 ? spk : wpk) {
                const int t = q.second / kPicks, i = q.second % kPicks;
                if (t >= T) continue;
                if (row_of[t] < 0) {
                    row_of[t] = (int) rx_rows.size();
                    rx_rows.emplace_back();
                    rx_rows.back().t = (uint8_t) t;
                    xs[row_of[t]] = r.x[t];
                }
                rx::ReqRow& w = rx_rows[(size_t) row_of[t]];
                if (pass >= 1) w.slice |= (uint16_t) (1u << w.n);
                if (pass == 2) w.slice |= (uint16_t) (1u << (rx::kPoolBit + w.n));
                w.e[w.n] = (int16_t) q.first;
                w.w[w.n] = r.w[t][i];
                ++w.n;
            }
        const int32_t c = spk.empty() && wpk.empty()
                              ? 0
                              : rx_k | (rx_verify > 0 ? rx::kReqSets : 0) | (rx_verify == 2 ? rx::kReqWrong : 0) |
                                    (wpk.empty() ? 0 : cut << rx::kReqCutShift);
        std::string e;
        if (!rx_link->send_req(seq, li, (int) rx_rows.size(), rx_rows.data(), xs, e, c)) {
            rx_lost("a request did not go out (" + e + ")");
            return false;
        }
        ++st.rx_reqs;
        ++st.rx_req_t[T];
        return true;
    }
    // picks (expert, t * kPicks + i) of request r over the n_ff blocks br (rx::fallback_range: the server's share of
    // them) computed on the lane and added to rp's parts
    void lane_add(const Layer& l, const Req& r, Resp& rp, int T, std::vector<std::pair<int, int>> pk2, rx::BlockRange br) {
        lane_sum(l, &r.x[0][0], r, T, std::move(pk2), br, &rp.part[0][0], rp.cpu);
    }
    // ... into out (kWin rows, kMaxDim apart; has[t]: row t holds a sum, else it is set by the first pick added) for x
    // (kWin rows, kMaxDim apart: r's, or the verify's in bf16)
    void lane_sum(const Layer& l, const float* x, const Req& r, int T, std::vector<std::pair<int, int>> pk2,
                  rx::BlockRange br, float* out, int* has) {
        if (br.empty() || pk2.empty()) return;
        std::sort(pk2.begin(), pk2.end());
        std::vector<Ent> ents;
        float wv[kWin * kPicks];
        int tok_of[kWin * kPicks];
        int nslots = 0;
        for (size_t a = 0; a < pk2.size();) {
            Ent en;
            en.e = pk2[a].first;
            en.slot0 = nslots;
            while (a < pk2.size() && pk2[a].first == en.e && en.n < ent_cap(l)) {
                const int t = pk2[a].second / kPicks, i = pk2[a].second % kPicks;
                en.tok[en.n++] = t;
                tok_of[nslots] = t;
                wv[nslots] = r.w[t][i];
                ++nslots;
                ++a;
            }
            ents.push_back(en);
        }
        int ff0, ffn;
        rx::ff_range(br, rx_nb, g.ff, ff0, ffn);   // (the last block: to n_ff)
        cpu_experts(l, x, ents, (size_t) kMaxDim, nullptr, ff0, ffn);
        const int D = g.dim;
        for (int t = 0; t < T; ++t)
            for (int sl = 0; sl < nslots; ++sl) {
                if (tok_of[sl] != t) continue;
                float* o = out + (size_t) t * kMaxDim;
                if (!has[t]) {
                    for (int c = 0; c < D; ++c) o[c] = 0.0f;
                    has[t] = 1;
                }
                const float* y = c_yd.data() + (size_t) sl * D;
                for (int c = 0; c < D; ++c) o[c] += wv[sl] * y[c];
            }
        st.cpu_experts += nslots;
    }
    // a round trip sample into the gate (the first 8 at 1 row kept for the stats: how it started)
    void rx_seen(int rows, double ms, bool waited, double sent) {
        rx_gate.rtt_seen(rows, ms, waited, sent);
        if (rows != 1 || rx_first_n >= 8) return;
        ++rx_first_n;
        std::lock_guard<std::mutex> lk(rx_mu);
        rx_first.push_back({(float) (waited ? ms : -ms), (float) rx_gate.rtt[1]});   // (< 0: an upper bound)
    }
    // the server's picks (expert, slot) of layer l computed here after all: counted, with the experts not all in the
    // page cache now (Linux; their pages read from the disk by the lane next)
    void rx_here_note(const Layer& l, const std::vector<std::pair<int, int>>& rpk) {
        st.rx_here += (int64_t) rpk.size();
#if defined(__linux__)
        int last = -1;
        std::vector<std::pair<int, int>> s(rpk);
        std::sort(s.begin(), s.end());
        for (const auto& q : s) {
            if (q.first == last) continue;
            last = q.first;
            bool res = true;
            for (int r = 0; r < 3 && res; ++r) res = os::resident_sampled(mx_mat(l, q.first, r), mx_part_bytes(l, r), 4);
            st.rx_here_cold += res ? 0 : 1;
        }
#else
        (void) l;
#endif
    }
    // a sent request's lateness for the link (past the fixed wait's deadline, DS4_RX_TIMEOUT_MS: not the lane's wait)
    void rx_back_seen(bool late) {
        if (late) ++st.rx_late_link;
        const int bk = rx_back.seen(late);
        if (bk == rx::Backoff::Silent)
            rx_lost("8 of the last 64 replies past the link's bound again, and no reply at all since the link was last "
                    "idle");
        else if (bk == rx::Backoff::Idle && ++st.rx_backoffs == 1)
            std::fprintf(stderr, "ds4: remote experts: 8 of the last 64 replies past the link's bound (%.2f ms after the "
                                 "lane's part) - the CPU lane computes all its experts for the next %d requests, then "
                                 "the link is tried again (said once; the count is in the stats)\n",
                         rx_timeout_ms, rx_back.hold);
    }
    // the reply to request seq (sent at t0) into rp's parts; late or no link: those picks computed here.  The deadline:
    // after the lane's own picks, as long as the lane would take for these (rx::Gate::wait_ms), but not before 1.5 x the
    // round trip expected at these rows; the link's lateness (rx::Backoff) against the fixed wait's (rx::Gate::deadlines)
    // (spk: the request's slice picks - the reply holds their [k, nb), and so does the fallback: sp's late ranges)
    void rx_finish(const Layer& l, const Req& r, Resp& rp, int T, unsigned seq, double t0,
                   const std::vector<std::pair<int, int>>& rpk, const std::vector<std::pair<int, int>>& spk,
                   const rx::SplitPlan& sp, const std::vector<std::pair<int, int>>& wpk, double mb = 0.0, int cut = 0) {
        const double tw = rx::now_ms();
        const int rows = (int) rx_rows.size();
        // (the lane's picks' worth the server computes: a W pick whole)
        const size_t n_srv = rx_npicks(rpk.size() + wpk.size(), spk.size());
        const auto dl = rx_gate.deadlines(T, n_srv, rows, t0, tw, rx_timeout_ms, rx_wait_fixed, rx_kts);
        const double deadline = dl.first, bound = dl.second;   // (the lane's; the link's, rx::Backoff's)
        const uint16_t* y = nullptr;
        int srv_us = 0, lane_us = 0;
        const int sets = !spk.empty() && rx_verify > 0 ? 2 : 1;
        std::string e;
        const int got = rx_on ? rx_link->wait_resp(seq, rows, deadline, &y, srv_us, e, sets, &lane_us) : -1;
        const double t1 = rx::now_ms();
        const double ta = got == 1 ? rx_link->arrival_ms() : -1.0;   // (the kernel's time the reply came in: Linux)
        const bool stale = got == 0 && rx_link->last_stale();   // (a fill took a slot it named: no reply will come)
        // earlier late requests' replies, skipped here: their round trips.  A stamp is a read's last skb: one equal to
        // the next reply's is that reply's; one read with only the next reply's head still over-states by that head
        const auto late = rx_link->take_late();
        if (!late.empty()) rx_back.reply();
        for (size_t i = 0; i < late.size(); ++i) {
            const double la = late[i].second;
            double ts;
            int rw, ly;
            if (la >= 0) rx_kts = true;
            bool open = false;
            double lb = -1;
            if (!rx_late_book.take(late[i].first, ts, rw, &ly, &open, &lb)) continue;
            ++st.rx_late_came;
            const bool shared = la == ta || (i + 1 < late.size() && late[i + 1].second == la);
            if (open) rx_back_seen(rx::LateBook::past(la, ts, t1, lb, shared));   // (the link late: past its bound)
            if (la < ts || la > t1) continue;   // (no kernel time, or the realtime clock stepped)
            if (shared) {
                ++st.rx_late_shared;
                continue;
            }
            rx_seen(rw, la - ts, true, ts);
            std::lock_guard<std::mutex> lk(rx_mu);
            ++st.rx_late_timed;
            if (rx_late_arr.size() < (size_t(1) << 20)) rx_late_arr.push_back({(float) (la - ts), ly});
        }
        if (ta >= 0) rx_kts = true;
        if (ta >= t0 && ta <= t1) rx_seen(rows, ta - t0, true, t0);   // (outside: the realtime clock stepped)
        else if (got == 1 || (got == 0 && !stale && !rx_kts))   // (late, with the kernel's times: its reply, if it
            rx_seen(rows, t1 - t0, t1 - tw > 0.05, t0);   // comes, says) (read within 50 us of the lane's part: there already)
        for (int n = rx_late_book.expire(t1); n > 0; --n) rx_back_seen(true);   // (past the link's bound, not in)
        if (got == 0 && !stale)   // (late for the lane's wait; for the link only past the fixed wait's deadline)
            for (int n = rx_late_book.add_late(seq, t0, rows, l.il, deadline, bound); n > 0; --n) rx_back_seen(true);
        st.rx_wait_ms += t1 - tw;
        if (T == 1 && !small_of[seq % kRing]) {   // (by the W blocks of [0, k) this request moved there: where the lane's
                                                  //  wait goes; a decode request's - not a small prompt's)
            const int64_t wb = (int64_t) wpk.size() * (rx_k - cut);
            const int bk = wb == 0 ? 0 : wb <= 3 ? 1 : wb <= 6 ? 2 : wb <= 12 ? 3 : 4;
            ++st.rx_wb_n[bk];
            st.rx_wb_part[bk] += tw - t0;
            st.rx_wb_wait[bk] += t1 - tw;
            if (got == 1 && ta >= t0 && ta <= t1) {
                ++st.rx_wb_na[bk];
                st.rx_wb_arr[bk] += ta - t0;
            }
        }
        st.rx_bytes_out = (int64_t) rx_link->bytes_out();
        st.rx_bytes_in = (int64_t) rx_link->bytes_in();
        st.rx_stale = (int64_t) rx_link->stales;
        {
            std::lock_guard<std::mutex> lk(rx_mu);
            if (rx_rtt.size() < (size_t(1) << 22)) {   // (4M requests: the percentiles are of the first ones after)
                rx_rtt.push_back(got == 1 ? (float) (t1 - t0) : -1.0f);
                rx_wait.push_back((float) (t1 - tw));
                rx_arr.push_back(ta >= t0 && ta <= t1 ? (float) (ta - t0) : -1.0f);
            }
        }
        if (got == 1) {
            const int D = g.dim;
            if (sets == 2) rx_split_check(l, r, T, spk, y + (size_t) rows * D);   // (before y is added: the same bits)
            for (int s = 0; s < sets; ++s)   // (DS4_EP_SPLIT_VERIFY: the whole picks' sums, then the slice picks')
                for (size_t j = 0; j < rx_rows.size(); ++j) {
                    const int t = rx_rows[j].t;
                    float* o = rp.part[t];
                    const uint16_t* yj = y + ((size_t) s * rows + j) * (size_t) D;
                    if (rp.cpu[t]) {
                        for (int c = 0; c < D; ++c) o[c] += rx::bf2f(yj[c]);
                    } else {
                        for (int c = 0; c < D; ++c) o[c] = rx::bf2f(yj[c]);
                        rp.cpu[t] = 1;
                    }
                }
            st.rx_picks += (int64_t) rpk.size();
            st.rx_split_picks += (int64_t) spk.size();
            st.rx_pool_picks += (int64_t) wpk.size();
            st.rx_pool_blocks += (int64_t) wpk.size() * (rx_nb - cut);
            if (!spk.empty() || !wpk.empty()) {
                ++st.rx_split_reqs;
                st.rx_srv_lane_ms += lane_us / 1000.0;
                // the balance's samples (T=1, the kernel's arrival known): the server lane's ms a byte, the rest of the
                // reply's time, this lane's ms a byte from the send to its part done
                const double sb = (double) spk.size() * (double) l.blob * (rx_nb - rx_k) / rx_nb +
                                  (double) wpk.size() * (double) l.blob * (rx_nb - cut) / rx_nb;
                if (rx_bal && T == 1 && ta >= t0 && ta <= t1 && sb > 0 && lane_us > 0) {
                    rx_bal_rs = rx::ewma(rx_bal_rs, lane_us / 1000.0 / sb, 0.05);
                    rx_bal_fix = rx::ewma(rx_bal_fix, (ta - t0) - rx_bal_rs * sb, 0.05);
                    ++rx_bal_ns;
                }
                if (rx_bal && T == 1 && mb > 0) {
                    rx_bal_rm = rx::ewma(rx_bal_rm, (tw - t0) / mb, 0.05);
                    ++rx_bal_nm;
                }
            }
            st.rx_rtt_ms += t1 - t0;
            st.rx_srv_ms += srv_us / 1000.0;
            rx_back.reply();   // (a long wait, past the bound: in time for the lane, late for the link)
            rx_back_seen(rx::LateBook::in_time_past(deadline, bound, ta, t0, t1));
            return;
        }
        if (got == 0) {
            ++st.rx_late_t[T];
            if (l.il >= 0 && l.il < 64) ++st.rx_late_l[l.il];
            if (++st.rx_late == 1)
                std::fprintf(stderr, "ds4: remote experts: a reply past its deadline (%.2f ms after the lane's part, %.2f ms "
                                     "after its send; layer %d): its %zu picks computed here (said once; the count is in "
                                     "the stats)\n", deadline - tw, deadline - t0, l.il, rpk.size());
        } else {
            rx_lost(e.empty() ? "the link failed" : e);
        }
        rx_here_note(l, rpk);
        const double tf = rx::now_ms();
        // exactly the server's share (the late reply is dropped when it comes): its experts whole, the slice picks' [k, nb)
        lane_add(l, r, rp, T, rpk, sp.late_whole);
        const int64_t ce = st.cpu_experts;   // (a slice pick is in cpu_experts once: serve_lane counted its [0, k))
        lane_add(l, r, rp, T, spk, sp.late_slice);
        st.cpu_experts = ce;
        lane_add(l, r, rp, T, wpk, rx::BlockRange{cut, rx_nb});   // (W's: [cut, nb) here, [0, cut) computed before)
        st.rx_fallback += (int64_t) rpk.size();
        st.rx_split_fallback += (int64_t) spk.size();
        st.rx_pool_fallback += (int64_t) wpk.size();
        st.rx_fallback_ms += rx::now_ms() - tf;
    }
    // DS4_EP_SPLIT_VERIFY: a reply's slice sums (ys: the request's rows, bf16) against this lane's [k, nb) of the same
    // picks for x in bf16, as the server has it (so both quantize the same activations: the bf16 sums' rounding is the
    // difference, ~4e-3 of a row); the largest |d| / |ref| by layer since the load
    void rx_split_check(const Layer& l, const Req& r, int T, const std::vector<std::pair<int, int>>& spk, const uint16_t* ys) {
        const int D = g.dim;
        rx_vx.resize((size_t) kWin * kMaxDim);
        rx_vy.resize((size_t) kWin * kMaxDim);
        int has[kWin] = {};
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < D; ++c) rx_vx[(size_t) t * kMaxDim + c] = rx::bf2f(rx::f2bf(r.x[t][c]));
        const int64_t ce = st.cpu_experts;   // (the check's own work: not the lane's)
        lane_sum(l, rx_vx.data(), r, T, spk, rx::server_range(rx::PickLane::kSplit, rx_k, rx_nb), rx_vy.data(), has);
        st.cpu_experts = ce;
        for (size_t j = 0; j < rx_rows.size(); ++j) {
            const int t = rx_rows[j].t;
            if (rx_rows[j].slice == 0 || !has[t]) continue;
            double dd = 0, rr = 0;
            for (int c = 0; c < D; ++c) {
                const double ref = rx_vy[(size_t) t * kMaxDim + c], d = (double) rx::bf2f(ys[j * (size_t) D + c]) - ref;
                dd += d * d;
                rr += ref * ref;
            }
            const float rel = (float) (rr > 0 ? std::sqrt(dd / rr) : std::sqrt(dd) > 0 ? 1.0 : 0.0);
            ++st.rx_split_n;
            if (l.il >= 0 && l.il < 64 && !(rel <= st.rx_split_err[l.il])) st.rx_split_err[l.il] = rel;
        }
        st.rx_split_wrong = rx_verify == 2;
    }
    std::string rx_lines(const Stats& s0, double nt) const {
        if (st.rx_held == 0 && rx_k >= rx_nb) return "";   // (H' with no blob held: its slice picks still)
        std::vector<float> rt, wt, at, lt;
        float lt_max = -1.0f;   // (the slowest late reply's time, its layer)
        int lt_layer = -1;
        {
            std::lock_guard<std::mutex> lk(rx_mu);
            for (size_t i = (size_t) std::max<int64_t>(0, s0.rx_reqs); i < rx_rtt.size(); ++i) {
                if (rx_rtt[i] >= 0) rt.push_back(rx_rtt[i]);
                if (rx_arr[i] >= 0) at.push_back(rx_arr[i]);
                wt.push_back(rx_wait[i]);
            }
            for (size_t i = (size_t) std::max<int64_t>(0, s0.rx_late_timed); i < rx_late_arr.size(); ++i) {
                lt.push_back(rx_late_arr[i].first);
                if (rx_late_arr[i].first > lt_max) {
                    lt_max = rx_late_arr[i].first;
                    lt_layer = rx_late_arr[i].second;
                }
            }
        }
        auto pct = [](std::vector<float>& v, std::string& out) {
            if (v.empty()) {
                out = "-";
                return;
            }
            std::sort(v.begin(), v.end());
            double sum = 0;
            for (float x : v) sum += x;
            out.clear();
            add_f(out, "mean %.3f p50 %.3f p99 %.3f max %.3f", sum / (double) v.size(), v[v.size() / 2],
                  v[std::min(v.size() - 1, (size_t) ((double) v.size() * 0.99))], v.back());
        };
        std::string rts, wts, ats, lts;
        pct(rt, rts);
        pct(wt, wts);
        pct(at, ats);
        pct(lt, lts);
        const double req = (double) (st.rx_reqs - s0.rx_reqs), served = (double) (st.rx_picks - s0.rx_picks);
        const double local = (double) (st.cpu_experts - s0.cpu_experts);
        std::string out;
        add_f(out,
              "remote experts: %.1f%% of the CPU lane's picks served by the expert server (%.2f a token of %.2f), "
              "%.2f a token computed here after all (%lld late replies, %lld past the link's bound; the link "
              "idle %lld times after 8 of 64 past it, %lld requests kept here then%s); %.2f requests a token\n"
              "remote experts per request: round trip %s ms (the server's own %.3f ms; send to the reply's arrival "
              "in the kernel %s ms), the lane's wait after its part %s ms\n"
              "remote experts per token: the lane's own part %.2f ms + its wait %.2f ms + the late ones %.2f ms; the "
              "link %.1f KB out, %.1f KB in; the upload %lld experts, %.2f GB in %.1f s, %lld of them in this "
              "GPU's VRAM too now (the cache's: none but zero-copy ones, after the link was given up, or after "
              "decode windows the server does not serve: DS4_SYNC, T past DS4_ASYNC_T)\n",
              local + served > 0 ? 100.0 * served / (local + served) : 0.0, served / nt, (local + served) / nt,
              (double) (st.rx_fallback - s0.rx_fallback) / nt, (long long) (st.rx_late - s0.rx_late),
              (long long) (st.rx_late_link - s0.rx_late_link), (long long) (st.rx_backoffs - s0.rx_backoffs),
              (long long) (st.rx_idle - s0.rx_idle),
              rx_on ? "" : ", the link given up", req / nt, rts.c_str(),
              (st.rx_srv_ms - s0.rx_srv_ms) / std::max<double>(1.0, (double) rt.size()), ats.c_str(), wts.c_str(),
              (st.cpu_ms - s0.cpu_ms) / nt, (st.rx_wait_ms - s0.rx_wait_ms) / nt,
              (st.rx_fallback_ms - s0.rx_fallback_ms) / nt,
              (double) (st.rx_bytes_out - s0.rx_bytes_out) / 1e3 / nt, (double) (st.rx_bytes_in - s0.rx_bytes_in) / 1e3 / nt,
              (long long) st.rx_held, (double) st.rx_upload_bytes / 1e9, st.rx_upload_ms / 1000.0,
              (long long) rx_in_vram());
        add_f(out,
              "remote experts' late replies that came after all: %lld (%lld read with a later reply, %lld with no "
              "kernel time: no time of their own); send to their arrival in the kernel %s ms",
              (long long) (st.rx_late_came - s0.rx_late_came), (long long) (st.rx_late_shared - s0.rx_late_shared),
              (long long) (st.rx_late_came - s0.rx_late_came - (st.rx_late_shared - s0.rx_late_shared) -
                           (st.rx_late_timed - s0.rx_late_timed)),
              lts.c_str());
        if (lt_layer >= 0) {
            add_f(out, " (the slowest %.3f ms at layer %d)", lt_max, lt_layer);
        }
        out += "; late by layer:";
        {
            std::vector<std::pair<long long, int>> bl;   // (the 6 layers with the most)
            for (int li = 0; li < 64; ++li)
                if (st.rx_late_l[li] - s0.rx_late_l[li] > 0) bl.push_back({st.rx_late_l[li] - s0.rx_late_l[li], li});
            std::sort(bl.begin(), bl.end(), [](const auto& x, const auto& y) {
                return x.first != y.first ? x.first > y.first : x.second < y.second;
            });
            for (size_t i = 0; i < bl.size() && i < 6; ++i) {
                add_f(out, " %d x%lld", bl[i].second, bl[i].first);
            }
            if (bl.empty()) out += " -";
        }
        out += "\n";
        {
            std::vector<std::pair<float, float>> fs;
            {
                std::lock_guard<std::mutex> lk(rx_mu);
                fs = rx_first;
            }
            if (rx_seed_ms >= 0)
                add_f(out, "remote experts' gate at 1 row: from the load's warm-up %.3f ms; ", rx_seed_ms);
            else
                add_f(out, "remote experts' gate at 1 row: no warm-up start; ");
            out += "the first samples (ms, the estimate after; - : an upper bound):";
            for (const auto& f : fs) {
                add_f(out, " %s%.3f/%.3f", f.first < 0 ? "-" : "", std::fabs(f.first), f.second);
            }
            const long long hn = st.rx_here - s0.rx_here, hc = st.rx_here_cold - s0.rx_here_cold;
            const double eb = st.rx_held > 0 ? (double) st.rx_upload_bytes / (double) st.rx_held : 0.0;
#if defined(__linux__)
            add_f(out, "; the server's picks the lane computed (kept by the gate or idle, late, lost) "
                       "%lld (%.2f a token), their experts not in the page cache then (4 pages a matrix "
                       "sampled) %lld (%.1f MB a token at %.1f MB an expert)\n",
                  hn, (double) hn / nt, hc, (double) hc * eb / 1e6 / nt, eb / 1e6);
#else
            (void) hc, (void) eb;
            add_f(out, "; the server's picks the lane computed (kept by the gate or idle, late, lost) "
                       "%lld (%.2f a token)\n", hn, (double) hn / nt);
#endif
        }
        out += "remote experts by T (requests / late / kept here by the send gate):";
        for (int t = 1; t <= kWin; ++t) {
            const long long q = st.rx_req_t[t] - s0.rx_req_t[t], la = st.rx_late_t[t] - s0.rx_late_t[t],
                            sk = st.rx_skip_t[t] - s0.rx_skip_t[t];
            if (q == 0 && sk == 0) continue;
            add_f(out, " T=%d %lld / %lld / %lld (EWMA: the round trip at %d rows %.3f ms, the lane %.3f "
                  "ms a pick);", t, q, la, sk, t, rx_gate.rtt[t], rx_gate.pick[t]);
        }
        out += "\n";
        if (rx_book) {   // (DS4_RX_FILL=1 and the fills' socket came up)
            int64_t held = 0;
            for (uint8_t h : rx_has) held += h == rx::kHeld;
            add_f(out, "remote experts' fills: %lld queued (%.2f a token), %lld loaded, %lld refused, %.1f MB a token "
                       "sent (%.2f s in the paced sends), %lld requests Stale; the server holds %lld experts now%s\n",
                  (long long) (st.rx_fills - s0.rx_fills), (double) (st.rx_fills - s0.rx_fills) / nt,
                  (long long) (st.rx_fill_ok - s0.rx_fill_ok), (long long) (st.rx_fill_fail - s0.rx_fill_fail),
                  (double) (st.rx_fill_bytes - s0.rx_fill_bytes) / 1e6 / nt, (st.rx_fill_ms - s0.rx_fill_ms) / 1000.0,
                  (long long) (st.rx_stale - s0.rx_stale), (long long) held, rx_fill ? "" : " (the fills' link gone)");
            if (st.rx_fill_zc > 0) {
                const double all = (double) (st.rx_fill_bytes - s0.rx_fill_bytes);
                const double di = (double) (st.rx_fill_direct - s0.rx_fill_direct), fi = (double) (st.rx_fill_filed - s0.rx_fill_filed);
                add_f(out, "remote experts' fills (DS4_RX_FILL_ZC=%d): %.1f MB a token through sendfile, %.1f straight from "
                           "the mapping, %.1f gathered (%.0f%% not copied by the gather)\n",
                      st.rx_fill_zc, fi / 1e6 / nt, di / 1e6 / nt, std::max(0.0, all - di - fi) / 1e6 / nt,
                      all > 0 ? 100.0 * (di + fi) / all : 0.0);
            }
        }
        out += ep_line(s0, nt);
        return out;
    }
    // the link core and keep-awake (ds4_rx.hpp): where the link thread runs, the Wakes, the lane's naps
    std::string ep_line(const Stats& s0, double nt) const {
        std::string sibs;
        for (int c : ep_sibs) sibs += " " + std::to_string(c);
        std::string b;
        add_f(b,
              "remote experts' link: the service thread on CPU %d%s (the CPU lane's pools off CPUs%s); keep-awake "
              "%s: %lld Wakes sent (%.3f a token), %lld requests found the lane napping\n",
              ep_cpu, ep_cpu >= 0 ? "" : ep_freed ? " (none: no server held any expert, the core went back to the CPU lane)"
                                                 : " (DS4_EP_CPU off: as without the link)", sibs.empty() ? " -" : sibs.c_str(),
              ep_wake ? "on" : "off (DS4_EP_WAKE=0)", (long long) (st.rx_wakes - s0.rx_wakes),
              (double) (st.rx_wakes - s0.rx_wakes) / std::max(1.0, nt), (long long) (st.lane_nap_reqs - s0.lane_nap_reqs));
        if (rx_k < rx_nb) {   // H' (DS4_EP_K)
            const double nr = (double) std::max<int64_t>(1, st.rx_split_reqs - s0.rx_split_reqs);
            add_f(b,
                  "expert parallelism H': k %d of %d n_ff blocks - %lld slice picks served (%.1f a token; this lane [0, "
                  "%d), the server's CPU lane [%d, %d) of them: %.1f us a request of %lld), %lld computed here after all "
                  "([%d, %d): late replies, a lost link)%s\n",
                  rx_k, rx_nb, (long long) (st.rx_split_picks - s0.rx_split_picks),
                  (double) (st.rx_split_picks - s0.rx_split_picks) / std::max(1.0, nt), rx_k, rx_k, rx_nb,
                  (st.rx_srv_lane_ms - s0.rx_srv_lane_ms) * 1000.0 / nr, (long long) (st.rx_split_reqs - s0.rx_split_reqs),
                  (long long) (st.rx_split_fallback - s0.rx_split_fallback), rx_k, rx_nb,
                  rx_verify == 2 ? "; DS4_EP_SPLIT_VERIFY=2" : rx_verify ? "; DS4_EP_SPLIT_VERIFY=1" : "");
            if (!rx_pool.empty()) {
                char bal[256] = "";
                if (rx_bal)   // (the EWMAs as rates: GB/s = 1 / (ms a byte) / 1e6)
                    std::snprintf(bal, sizeof bal,
                                  "; the balance kept %lld W picks here (%.1f a token; at the end this lane %.1f GB/s from "
                                  "the send, the server's lane %.1f GB/s, the rest of a reply %.3f ms)",
                                  (long long) (st.rx_pool_kept - s0.rx_pool_kept),
                                  (double) (st.rx_pool_kept - s0.rx_pool_kept) / std::max(1.0, nt),
                                  rx_bal_rm > 0 ? 1e-6 / rx_bal_rm : 0.0, rx_bal_rs > 0 ? 1e-6 / rx_bal_rs : 0.0, rx_bal_fix);
                add_f(b,
                      "the hybrid: the server's W of %lld experts - %lld W picks whole on its CPU lane (%.1f a token; not "
                      "computed here), %lld computed here whole after all (late replies, a lost link)%s\n",
                      (long long) st.rx_pool_w, (long long) (st.rx_pool_picks - s0.rx_pool_picks),
                      (double) (st.rx_pool_picks - s0.rx_pool_picks) / std::max(1.0, nt),
                      (long long) (st.rx_pool_fallback - s0.rx_pool_fallback), bal);
                const int64_t wp = st.rx_pool_picks - s0.rx_pool_picks;
                if (rx_cut && wp > 0)
                    add_f(b, "the hybrid's cut (DS4_EP_POOL_CUT): the server computed %.2f of a W pick's %d n_ff blocks "
                             "(%.1f blocks a token; [0, k) is %d of them)\n",
                          (double) (st.rx_pool_blocks - s0.rx_pool_blocks) / (double) wp, rx_nb,
                          (double) (st.rx_pool_blocks - s0.rx_pool_blocks) / std::max(1.0, nt), rx_k);
                if (rx_cut_chk)
                    add_f(b, "the hybrid's cut check (DS4_EP_POOL_CUT_CHECK): %lld requests' one pass against two calls, "
                             "%lld not the same bits (the second W call's down from the pack, the pass's from the down "
                             "parts where they serve [0, cut))\n",
                          (long long) (st.rx_cut_chk - s0.rx_cut_chk), (long long) (st.rx_cut_bad - s0.rx_cut_bad));
                if (rx_cut && st.rx_cut_sets > s0.rx_cut_sets)
                    add_f(b, "the hybrid's cut down: %lld of %lld W sets computed here read [0, cut)'s down from the down "
                             "parts (DS4_IQ_DOWN_PARTS), the rest the pack's rows\n",
                          (long long) (st.rx_cut_dp - s0.rx_cut_dp), (long long) (st.rx_cut_sets - s0.rx_cut_sets));
                std::string wb;
                static const char* const kWb[5] = {"0", "1-3", "4-6", "7-12", "13+"};
                for (int q = 0; q < 5; ++q) {
                    const int64_t n = st.rx_wb_n[q] - s0.rx_wb_n[q], na = st.rx_wb_na[q] - s0.rx_wb_na[q];
                    if (n <= 0) continue;
                    char one[200];
                    std::snprintf(one, sizeof one, "%s%s - %.1f a token: send to the lane's part %.3f, to the reply's "
                                                   "arrival %.3f, the wait after the part %.3f ms",
                                  wb.empty() ? "" : "; ", kWb[q], (double) n / std::max(1.0, nt),
                                  (st.rx_wb_part[q] - s0.rx_wb_part[q]) / (double) n,
                                  na > 0 ? (st.rx_wb_arr[q] - s0.rx_wb_arr[q]) / (double) na : -1.0,
                                  (st.rx_wb_wait[q] - s0.rx_wb_wait[q]) / (double) n);
                    wb += one;
                }
                if (!wb.empty())
                    add_f(b, "the hybrid's decode requests by the W blocks of [0, k) they moved to the server: %s\n",
                          wb.c_str());
            }
            std::string np;   // (every split arm, W or not: the lane's part at the same pick count)
            for (int q = 0; q < 10; ++q) {
                const int64_t n = st.rx_np_n[q] - s0.rx_np_n[q];
                if (n <= 0) continue;
                char one[160];
                std::snprintf(one, sizeof one, "%s%d%s - %.1f a token, the lane's part %.3f ms, %.2f W blocks moved",
                              np.empty() ? "" : "; ", q, q == 9 ? "+" : "", (double) n / std::max(1.0, nt),
                              (st.rx_np_ms[q] - s0.rx_np_ms[q]) / (double) n, (st.rx_np_wb[q] - s0.rx_np_wb[q]) / (double) n);
                np += one;
            }
            if (!np.empty())
                add_f(b, "the decode requests sent by their CPU-lane picks (W's among them): %s\n", np.c_str());
        }
        return b;
    }

    // the decode path's host side: each published routing in turn - its CPU lane experts (if any) into the answer's
    // part, then the answer's seq.  Spins while requests come (a token's 40 layers), naps after 2 ms without one.
    void serve_lane() {
        unsigned next = 1;
        auto last = std::chrono::steady_clock::now();
        bool napped = false;
        std::vector<Ent> ents;
        double lp_ans = 0.0;   // (DS4_LANE_PROF: the last answer's time and its layer)
        int lp_li = -1;
        while (!quit.load(std::memory_order_relaxed)) {
            Req& r = reqs[next % kRing];
            if (*(volatile unsigned*) &r.seq != next) {
                cpu_relax();
                // (--remote-experts) a window began: a Wake to the server if the link has gone quiet; no naps while
                // the server still spins (rx::lane_naps)
                if (ep_wake && rx_wake.load(std::memory_order_relaxed) && rx_wake.exchange(false) && rx_on) {
                    std::string e;
                    if (rx_link->wake(e)) ++st.rx_wakes;
                }
                if (rx::lane_naps(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last).count(),
                                  rx_on && ep_wake)) {
                    napped = true;
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                continue;
            }
            if (napped) ++st.lane_nap_reqs;   // (a request that found the lane napping)
            napped = false;
            std::atomic_thread_fence(std::memory_order_acquire);
            const int li = layer_of[next % kRing].load(std::memory_order_acquire);
            const double lp_req = lane_prof ? now_ms() : 0.0;
            if (lane_prof && lp_ans > 0.0) (li == lp_li + 1 ? st.lp_wait_ms : st.lp_gap_ms) += lp_req - lp_ans;
            const bool ep = ep_of[next % kRing];   // (written before the request's launch: ep_of)
            Resp& rp = resps[next % kRing];
            const int T = std::max(1, std::min(kWin, r.T));
            for (int t = 0; t < T; ++t) rp.cpu[t] = 0;
            if (li >= 0 && li < (int) L.size()) {
                Layer& l = L[(size_t) li];
                uint32_t* pr = prof.data() + (size_t) li * g.n_exp;
                ents.clear();
                if (store && li > 0 && li < g.nl) store->cancel_stale(li - 1);   // (guesses for layers the token has passed: dropped unread)
                // lookahead: the guesses made for this layer (when they are for this request), counted on decode
                // (a small prompt's window of one is not a decode window: its guesses are asked for and kept all the
                // same - ahead_ask, ahead_fl - but not counted)
                const bool dec = !small_of[next % kRing];
                const uint8_t* fl = T == 1 && dec && !ahead_for.empty() && ahead_for[(size_t) li] == next
                                        ? ahead_fl.data() + (size_t) li * 1024 : nullptr;
                // the window's CPU-lane picks by expert (up to 8 tokens an entry), their weights by entry row
                float wv[kWin * kPicks];
                int tok_of[kWin * kPicks];
                int nslots = 0;
                std::vector<std::pair<int, int>> picks;   // (expert, t * kPicks + i), CPU-lane only
                // (--remote-experts) the CPU-lane picks the expert server holds: sent before the lane computes the rest
                std::vector<std::pair<int, int>> rpicks;
                const bool rxl = rx_on && !l.mtp && li < g.nl && l.n_exp == g.n_exp &&
                                 li < (int) (rx_has.size() / (size_t) g.n_exp);
                // (H', DS4_EP_K) an EP window's CPU-lane picks of a layer that splits go tagged slice with the request
                // (the server's CPU lane [k, nb) of them): ds4_rx_split.hpp's kSplit when it is sent, kMainWhole when not
                const bool spl = rxl && ep && rx_k < rx_nb && (size_t) li < rx_split_l.size() && rx_split_l[(size_t) li];
                if (trace_f != nullptr && !l.mtp) {   // DS4_TRACE=file: one record a route request: u8 layer, u8 T, T x k int16 expert ids
                    // DS4_TRACE_AHEAD=1: T has bit 7 set and the guesses follow: u8 na1, u8 na2, T x (na1 + na2) int16 (the
                    // next layer's likely picks by rank, then the one after's), as the lookahead made them for THIS layer's
                    // predecessors (the request carries the guesses for the layers after it)
                    static const bool with_ahead = std::getenv("DS4_TRACE_AHEAD") != nullptr;
                    uint8_t hd[2] = {(uint8_t) li, (uint8_t) (T | (with_ahead ? 0x80 : 0))};
                    std::fwrite(hd, 1, 2, trace_f);
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < l.k_exp; ++i) {
                            const int16_t v = (int16_t) r.ids[t][i];
                            std::fwrite(&v, 2, 1, trace_f);
                        }
                    if (with_ahead) {
                        const uint8_t na[2] = {(uint8_t) std::max(0, std::min(r.na[0], kAhead1)), (uint8_t) std::max(0, std::min(r.na[1], kPicks))};
                        std::fwrite(na, 1, 2, trace_f);
                        for (int t = 0; t < T; ++t) {
                            for (int i = 0; i < na[0]; ++i) {
                                const int16_t v = (int16_t) r.ahead[t][i];
                                std::fwrite(&v, 2, 1, trace_f);
                            }
                            for (int i = 0; i < na[1]; ++i) {
                                const int16_t v = (int16_t) r.ahead[t][kAhead1 + i];
                                std::fwrite(&v, 2, 1, trace_f);
                            }
                        }
                    }
                }
                const bool dsk = disk_skip_w > 0.0f && !l.mtp && li >= disk_skip_from && li < g.nl;
                for (int t = 0; t < T; ++t) {
                    float wsum = 0.0f;   // (DISK_SKIP_W: the token's route weights)
                    if (dsk)
                        for (int i = 0; i < l.k_exp; ++i)
                            if (r.ids[t][i] >= 0 && r.ids[t][i] < l.n_exp) wsum += r.w[t][i];
                    for (int i = 0; i < l.k_exp; ++i) {
                        const int e = r.ids[t][i];
                        if (e < 0 || e >= l.n_exp) continue;
                        ++pr[e];
                        l.heat[(size_t) e] += 1.0f;
                        if (fl != nullptr) {
                            ++st.ahead_picks;
                            st.ahead_hit_k += fl[e] & 1;
                            st.ahead_hit += (fl[e] & 6) != 0;
                            st.ahead_hit2 += (fl[e] >> 2) & 1;
                        }
                        if (l.slot[(size_t) e] < 0) {
                            const bool zc = T == 1 && !l.zc.empty() && l.zc[(size_t) e];
                            if (!zc && rxl && rx_has[(size_t) li * g.n_exp + e] == rx::kHeld) {   // (the server's,
                                rpicks.push_back({e, t * kPicks + i});   // not one it is filling: not a swap candidate;
                                continue;                                // one the device computes from the store stays
                            }                                            // the device's)
                            l.cpu_hits.push_back(e);
                            if (zc) {   // the device computed it from the store's slot
                                store->touch(li, e);
                                ++st.store_zc;
                                continue;
                            }
                            if (dsk && r.w[t][i] < disk_skip_w * wsum && !in_ram(l, e)) {
                                ++st.skipped;
                                continue;
                            }
                            if (!l.ram.empty() && l.ram[(size_t) e] >= 0) ++st.ram_experts;
                            else if (fl != nullptr) {
                                ++st.ahead_cold;
                                st.ahead_cold_hit += (fl[e] >> 3) & 1;
                            }
                            picks.push_back({e, t * kPicks + i});
                        }
                    }
                }
                if (fl != nullptr) {   // the guesses by rank: how many were picked
                    const uint8_t* rk = ahead_rank.data() + (size_t) li * 1024;
                    for (int e = 0; e < l.n_exp && e < 1024; ++e) {
                        if (rk[e] == 0 || rk[e] > 16) continue;
                        bool hit = false;
                        for (int i = 0; i < l.k_exp; ++i) hit = hit || r.ids[0][i] == e;
                        ++st.ahead_rank_n[rk[e] - 1];
                        st.ahead_rank_hit[rk[e] - 1] += hit;
                    }
                }
                if (ahead_mode > 0 && !ahead_for.empty()) ahead_ask(li, r, next, T, dec);
                double rx_t0 = 0.0;
                // a request goes out (the gate and the link allowing) for the server's experts, or for slice picks
                bool want = !rpicks.empty() || (spl && !picks.empty());
                if (want && rx_back.idle()) {   // (8 late of 64 a while ago: the link idle for now)
                    rx_here_note(l, rpicks);
                    picks.insert(picks.end(), rpicks.begin(), rpicks.end());
                    ++st.rx_idle;
                    rpicks.clear();
                    want = false;
                }
                // (the lane's picks whole in the gate's local cost, split or not: sent, the lane's [0, k) of them hides
                // the round trip - U4's rule; slice_share is the deadline's)
                if (want && !rx_gate.send(T, rx_nrows(rpicks, spl ? picks : std::vector<std::pair<int, int>>()), rpicks.size(),
                                          picks.size())) {
                    rx_here_note(l, rpicks);
                    picks.insert(picks.end(), rpicks.begin(), rpicks.end());   // (the round trip would cost more)
                    ++st.rx_skip_t[T];
                    rpicks.clear();
                    want = false;
                }
                // the slice picks: every CPU-lane pick of the request (not one on this GPU: picks holds none) - which go
                // tagged, the lane's range for them and the fallback's: rx::split_plan (ds4_rx_split.hpp)
                std::vector<std::pair<int, int>> spicks, wpicks;
                int wcut = 0;   // (W picks' first block on the server: [0, wcut) of them here - DS4_EP_POOL_CUT)
                rx::SplitPlan sp = rx::split_plan(spl, want, true, rx_k, rx_nb);
                const int nlane = (int) picks.size();   // (the request's CPU-lane picks, W's among them: the lane's stats)
                if (want) {
                    if (sp.tagged && (size_t) (li + 1) * g.n_exp <= rx_pool.size()) {   // (the hybrid, U5c: W's picks
                        const uint8_t* pw = rx_pool.data() + (size_t) li * g.n_exp;   // whole on the server, not here)
                        int nc = 0;
                        for (const auto& q : picks) nc += pw[q.first] != 0;
                        int keep = nc;   // (the balance: W candidates sent, the rest kept here - T=1 only)
                        if (rx_bal && T == 1 && nc > 0) {
                            const double wk = (double) l.blob * rx_k / rx_nb, wt = (double) l.blob - wk;
                            keep = 0;   // (until both are seeded: none)
                            if (rx_bal_nm >= 16 && rx_bal_ns >= 16) {
                                if (rx_cut) {
                                    const rx::PoolCut pc = rx::pool_cut(nc, (int) picks.size(), rx_k, wk, wt, rx_bal_rm,
                                                                        rx_bal_rs, rx_bal_fix, rx_bal_margin);
                                    keep = pc.n;
                                    wcut = pc.cut;
                                } else {
                                    keep = rx::pool_keep(nc, (int) picks.size(), wk, wt, rx_bal_rm, rx_bal_rs, rx_bal_fix,
                                                         rx_bal_margin);
                                }
                            }
                            st.rx_pool_kept += nc - keep;
                        }
                        size_t n = 0;
                        for (const auto& q : picks) {
                            if (pw[q.first] && keep > 0) {
                                wpicks.push_back(q);
                                --keep;
                            } else {
                                picks[n++] = q;
                            }
                        }
                        picks.resize(n);
                    }
                    if (sp.tagged) spicks = picks;
                    rx_t0 = rx::now_ms();
                    if (!rx_send(next, li, T, r, rpicks, spicks, wpicks, wcut)) {   // (no link: the lane computes them as it did before)
                        rx_here_note(l, rpicks);
                        picks.insert(picks.end(), rpicks.begin(), rpicks.end());
                        picks.insert(picks.end(), wpicks.begin(), wpicks.end());
                        st.rx_fallback += (int64_t) rpicks.size();
                        rpicks.clear();
                        spicks.clear();
                        wpicks.clear();
                        wcut = 0;
                        sp = rx::split_plan(spl, want, false, rx_k, rx_nb);
                    }
                }
                // the lane's n_ff range for its picks: [0, k) of the slice picks just sent, else whole
                const rx::BlockRange mr = sp.here;
                std::sort(picks.begin(), picks.end());
                for (size_t a = 0; a < picks.size();) {
                    Ent en;
                    en.e = picks[a].first;
                    en.slot0 = nslots;
                    en.n = 0;
                    while (a < picks.size() && picks[a].first == en.e && en.n < ent_cap(l)) {
                        const int t = picks[a].second / kPicks, i = picks[a].second % kPicks;
                        en.tok[en.n++] = t;
                        tok_of[nslots] = t;
                        wv[nslots] = r.w[t][i];
                        ++nslots;
                        ++a;
                    }
                    ents.push_back(en);
                }
                // the W picks cut (DS4_EP_POOL_CUT): their [0, wcut) here, in slots and entries after the lane's own
                // (cpu_experts' second slice: the same pass)
                const int nmain = nslots, na = (int) ents.size();
                if (wcut > 0) {
                    std::vector<std::pair<int, int>> wp(wpicks);
                    std::sort(wp.begin(), wp.end());
                    for (size_t a = 0; a < wp.size();) {
                        Ent en;
                        en.e = wp[a].first;
                        en.slot0 = nslots;
                        en.n = 0;
                        while (a < wp.size() && wp[a].first == en.e && en.n < ent_cap(l)) {
                            const int t = wp[a].second / kPicks, i = wp[a].second % kPicks;
                            en.tok[en.n++] = t;
                            tok_of[nslots] = t;
                            wv[nslots] = r.w[t][i];
                            ++nslots;
                            ++a;
                        }
                        ents.push_back(en);
                    }
                }
                // lookahead: the entries asked for early (or in the RAM tier) computed first, so the ones asked for
                // only now get the others' time to come in from the SSD (their slots, so the sums, stay the same)
                int ord[kWin * kPicks];
                const int* po = nullptr;
                if (ahead_order && !ahead_for.empty() && ahead_for[(size_t) li] == next && na > 1 && na <= kWin * kPicks) {
                    const uint8_t* fa = ahead_fl.data() + (size_t) li * 1024;   // (the lane's own entries)
                    int n = 0;
                    for (int pass = 0; pass < 2; ++pass)
                        for (size_t q = 0; q < (size_t) na; ++q) {
                            const int e = ents[q].e;
                            const bool early = !file_read(l, e) || (fa[e] & 8) != 0;
                            if (early == (pass == 0)) ord[n++] = (int) q;
                        }
                    po = ord;
                }
                double bal_mb = 0.0;   // (this lane's bytes of the request: the balance's sample)
                if (!ents.empty()) {
                    const double tc = now_ms();
                    if (lane_prof) st.lp_plan_ms += tc - lp_req;
                    int ff0, ffn, wf0 = 0, wfn = 0;
                    rx::ff_range(mr, rx_nb, g.ff, ff0, ffn);
                    FfSplit wsb;
                    if (na < (int) ents.size()) {
                        rx::ff_range(rx::BlockRange{0, wcut}, rx_nb, g.ff, wf0, wfn);
                        wsb = FfSplit{na, wf0, wfn};
                        ++st.rx_cut_sets;
                        if (!store && down_parts(l, wf0, wfn) != nullptr) ++st.rx_cut_dp;
                    }
                    cpu_experts(l, &r.x[0][0], ents, (size_t) kMaxDim, po, ff0, ffn, wsb);
                    if (rx_cut_chk && wsb.from >= 0) {   // (DS4_EP_POOL_CUT_CHECK: the pass against two calls, bit for bit)
                        const size_t D = (size_t) g.dim;
                        const std::vector<float> y1(c_yd.begin(), c_yd.begin() + (ptrdiff_t) ((size_t) nslots * D));
                        const std::vector<Ent> a(ents.begin(), ents.begin() + na), w(ents.begin() + na, ents.end());
                        if (!a.empty()) cpu_experts(l, &r.x[0][0], a, (size_t) kMaxDim, po, ff0, ffn);
                        dp_pack = true;   // (the W part's down from the pack: the pass's from the parts checked against it)
                        cpu_experts(l, &r.x[0][0], w, (size_t) kMaxDim, nullptr, wf0, wfn);
                        dp_pack = false;
                        ++st.rx_cut_chk;
                        if (std::memcmp(y1.data(), c_yd.data(), (size_t) nslots * D * sizeof(float)) != 0) ++st.rx_cut_bad;
                    }
                    const double ta = lane_prof ? now_ms() : 0.0;
                    const int D = g.dim;
                    for (int t = 0; t < T; ++t) {
                        bool any = false;
                        for (int sl = 0; sl < nslots; ++sl) any = any || tok_of[sl] == t;
                        if (!any) continue;
                        float* o = rp.part[t];
                        for (int e = 0; e < D; ++e) o[e] = 0.0f;
                        for (int sl = 0; sl < nslots; ++sl) {
                            if (tok_of[sl] != t) continue;
                            const float* y = c_yd.data() + (size_t) sl * D;
                            for (int e = 0; e < D; ++e) o[e] += wv[sl] * y[e];
                        }
                        rp.cpu[t] = 1;
                    }
                    if (lane_prof) st.lp_add_ms += now_ms() - ta;
                    const double tl = now_ms() - tc;
                    st.cpu_ms += tl;
                    if (T == 1 && dec && sp.sent) {   // (by the request's CPU-lane picks: the lane's part at the same count;
                                                      //  a decode request's - not a small prompt's)
                        const int q = std::min(nlane, 9);
                        ++st.rx_np_n[q];
                        st.rx_np_ms[q] += tl;
                        st.rx_np_wb[q] += (double) wpicks.size() * (rx_k - wcut);
                    }
                    st.cpu_experts += nmain;   // (a W pick cut: the reply's, as one whole there)
                    // (an expert's bytes once: past ent_cap's tokens it takes more than one entry, side by side)
                    size_t nexp = 0, nwexp = 0;
                    for (size_t q = 0; q < (size_t) na; ++q) nexp += q == 0 || ents[q].e != ents[q - 1].e;
                    for (size_t q = (size_t) na; q < ents.size(); ++q) nwexp += q == (size_t) na || ents[q].e != ents[q - 1].e;
                    bal_mb = ((double) nexp * ffn + (double) nwexp * wfn) * (double) l.blob / (double) g.ff;
                    st.lane_bytes += bal_mb;
                    // (a pick's time whole: the slices computed - [0, k) of nb blocks, a cut W pick's [0, cut))
                    const double whole = ((double) nmain * (double) (mr.b1 - mr.b0) + (double) (nslots - nmain) * wcut) /
                                         (double) std::max(1, rx_nb);
                    if (rx_on && whole > 0) rx_gate.pick_seen(T, tl / whole);
                }
                if (sp.sent) {
                    const double tr = lane_prof ? now_ms() : 0.0;
                    rx_finish(l, r, rp, T, next, rx_t0, rpicks, spicks, sp, wpicks, bal_mb, wcut);
                    if (lane_prof) st.lp_rx_ms += now_ms() - tr;
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile unsigned*) &rp.seq = next;
            if (lane_prof) {
                lp_ans = now_ms();
                lp_li = li;
            }
            ++next;
            last = std::chrono::steady_clock::now();
        }
    }

    // a GGUF's matrices (BF16 / MXFP4) to its own kernels, EXL3 ones to x3's
    static bool mv_any(const x3::MvJob* j, int n, cudaStream_t s) {
        bool g = false;
        for (int i = 0; i < n; ++i) g = g || j[i].m->fmt != 0;
        return g ? gg::mv(j, n, s) : x3::mv(j, n, s);
    }
    bool mv(Dev& d, std::initializer_list<x3::MvJob> jobs, std::string& err) {
        x3::MvJob j[x3::kMaxMvJobs];
        int n = 0;
        for (const x3::MvJob& x : jobs) j[n++] = x;
        if (!mv_any(j, n, d.st)) {
            err = "exl3 mv failed";
            return false;
        }
        return true;
    }
    bool mv_n(Dev& d, x3::MvJob* j, int n, std::string& err) {
        for (int a = 0; a < n; a += x3::kMaxMvJobs)
            if (!mv_any(j + a, std::min(x3::kMaxMvJobs, n - a), d.st)) {
                err = "exl3 mv failed";
                return false;
            }
        return true;
    }
    // y[c] = x[c] W for the T activations x + c * x_cs (floats), into y + c * y_cs
    static x3::MvJob job(const x3::Mat& m, const float* x, float* y, int T = 1, size_t x_cs = 0, size_t y_cs = 0,
                         float alpha = 1.0f) {
        x3::MvJob j;
        j.m = &m;
        j.xf = x;
        j.y = y;
        j.alpha = alpha;
        j.ncol = T;
        j.xf_cs = x_cs;
        j.y_cs = y_cs;
        return j;
    }

    // ---- the DSpark drafter (DeepSeek's DSparkBlock / forward_spec).  Its blocks' rings hold the trunk's main KV:
    // seed() turns a window's taps (the streams' mean entering layers 37..39) into it, through main_proj / main_norm
    // and each block's wkv / kv_norm / rope / FP8, for the positions the trunk kept.  draft() runs the block [y, noise x
    // (block - 1)] at positions pos .. through the blocks (each query attending to the ring and to the whole block),
    // the trunk's head, the markov bias token by token (greedy) and the confidence head.
    bool seed_drafter(int64_t p0, int n, int row0, std::string& err) {
        if (g.n_mtp == 0 || n <= 0) return true;
        if (n > kWin * kPicks) {
            err = "seeding the drafter with " + std::to_string(n) + " rows";
            return false;
        }
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        const int nt = (int) g.d_targets.size(), D = g.dim, HD = g.hd;
        if (dump_d != nullptr) {   // DS4_DSPARK_DUMP: 'S', the first position, the rows, then the taps
            for (Dev& e : devs) {
                cudaSetDevice(e.id);
                CK(cudaStreamSynchronize(e.st));
            }
            cudaSetDevice(d.id);
            const int64_t hdr[4] = {'S', p0, n, (int64_t) nt * D};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(h_taps + (size_t) row0 * nt * D, 4, (size_t) n * nt * D, dump_d);
            std::fflush(dump_d);
        }
        for (int a = 0; a < n; a += kWin) {   // (a chunk's last 128 rows: 16 a GEMV)
            const int na = std::min(kWin, n - a);
            CK(cudaMemcpyAsync(d.xe, h_taps + (size_t) (row0 + a) * nt * D, (size_t) na * nt * D * 4,
                               cudaMemcpyHostToDevice, d.st));
            if (!mv(d, {job(d_main_proj, d.xe, d.ed + (size_t) a * D, na, (size_t) nt * D, D)}, err)) return false;
        }
        CK(cudaEventRecord(ev_seed, d.st));
        Finish fm;
        fm.w = d_main_norm;
        fm.eps = g.eps;
        finish(d.ed, n, D, D, fm, d.st);
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (l.dev != d_dev) {
                err = "the DSpark blocks are split across devices";
                return false;
            }
            for (int a = 0; a < n; a += kWin) {
                const int na = std::min(kWin, n - a);
                if (!mv(d, {job(l.wkv, d.ed + (size_t) a * D, d.kvr, na, D, HD)}, err)) return false;
                Finish fk;
                fk.w = l.kv_norm;
                fk.eps = g.eps;
                fk.rd = g.rd;
                fk.pos = (int) (p0 + a);
                fk.freqs = d.freqs_win;
                fk.quant = 1;
                fk.dst = d.ext + (size_t) g.win * HD;
                fk.dst_stride = HD;
                finish(d.kvr, na, HD, HD, fk, d.st);
                ext_to_ring(d.ext, l.win, p0 + a, g.win, na, HD, d.st);
            }
        }
        return true;
    }

    bool draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature, float top_p,
               int top_k, float min_p, uint64_t seed, uint64_t counter) {
        const double t0 = now_ms();
        drafts_q = temperature > 0.0f && sample_drafts;
        dq_temp = temperature;
        dq_top_p = top_p;
        dq_top_k = top_k;
        dq_min_p = min_p;
        dq_seed = seed;
        dq_counter = counter;
        const bool ok = draft_(y, out, conf, err);
        if (!ok) drafts_q = false;
        if (max_drafts > 0 && (int) out.size() > max_drafts) {
            out.resize((size_t) max_drafts);
            conf.resize((size_t) max_drafts);
        }
        st.draft_ms += now_ms() - t0;
        ++st.drafts_rounds;
        return ok;
    }
    bool draft_(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
        out.clear();
        conf.clear();
        const int T = g.d_block, D = g.dim, H = g.hc, V = g.vocab, R = g.d_rank;
        const int64_t p0 = pos;
        if (g.n_mtp == 0 || T < 1 || T > kWin || p0 + T > o.max_context) return true;
        int tk[kWin];
        tk[0] = y;
        for (int t = 1; t < T; ++t) tk[t] = g.d_noise;
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        for (int t = 0; t < T; ++t) std::memcpy(h_emb + (size_t) t * D, embed_pinned + (size_t) tk[t] * D, (size_t) D * 2);
        CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
        embed(d.emb_rows, T, d.h, D, H, d.st);
        init_pre(d.pre, T, d.st);
        WinPlan dp(false);   // (window attention alone: the mailbox seqs; the drafter's layers: the main's alone, whole)
        dp.async = !sync_decode && T <= async_max_t;
        for (int i = 0; i < g.n_mtp && dp.async; ++i) {
            const int il = g.nl + i;
            dp.seq[il] = ++seq_issued;
            ep_of[dp.seq[il] % kRing] = dp.ep;
            small_of[dp.seq[il] % kRing] = small_feed;
            layer_of[dp.seq[il] % kRing].store(il, std::memory_order_release);
        }
        const size_t hs = (size_t) T * H * D, rs = (size_t) g.win * g.hd;
        if (dump_d != nullptr && h_dump == nullptr &&
            (!halloc(h_dump, (size_t) g.n_mtp * kWin * H * D, err) || !halloc(h_dump_ring, (size_t) g.n_mtp * rs, err)))
            return false;
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (dump_d != nullptr) {   // (stream order: the ring the block reads - as BF16 rows - the stream it leaves)
                ring_to_ext(l.win, d.ext, g.win, g.win, g.hd, d.st);
                CK(cudaMemcpyAsync(h_dump_ring + (size_t) i * rs, d.ext, rs * 2, cudaMemcpyDeviceToHost, d.st));
            }
            if (!layer(l, 0, p0, T, dp, err)) return false;
            if (dump_d != nullptr) {
                cudaSetDevice(d.id);
                CK(cudaMemcpyAsync(h_dump + (size_t) i * hs, d.h, hs * 4, cudaMemcpyDeviceToHost, d.st));
            }
        }
        cudaSetDevice(d.id);
        hc_pre_norm(d.h, d.pre, nullptr, D, H, T, g.eps, d.x, d.st);   // (the confidence head reads it before the norm)
        Dev& hd = devs[(size_t) head_dev];
        // (through the host to the head's device, and to the head group's other members - the drafter's device can be
        // the head group's lead itself, a group of every GPU)
        if (d_dev != head_dev || groups[(size_t) hd.grp].mem.size() > 1) {
            CK(cudaMemcpyAsync(d.bounce, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, d.st));
            CK(cudaEventRecord(d.ev_out, d.st));
        }
        if (d_dev != head_dev) {
            cudaSetDevice(hd.id);
            CK(cudaStreamWaitEvent(hd.st, d.ev_out, 0));
            CK(cudaMemcpyAsync(hd.x, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, hd.st));
        }
        // the drafter's final norm and the trunk's head, on every member of the head group (the others' shares first)
        const std::vector<int>& hg = groups[(size_t) hd.grp].mem;
        const bool mk = mk_split && !drafts_q;   // (sampled drafts: the lead's whole rows, gathered)
        for (int m = (int) hg.size() - 1; m >= 0; --m) {
            Dev& e = devs[(size_t) hg[(size_t) m]];
            cudaSetDevice(e.id);
            if (m > 0) {
                CK(cudaStreamWaitEvent(e.st, d.ev_out, 0));
                CK(cudaMemcpyAsync(e.ao, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, e.st));
            } else
                CK(cudaMemcpyAsync(e.ao, e.x, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, e.st));
            Finish fn;
            fn.w = e.d_norm;
            fn.eps = g.eps;
            finish(e.ao, T, D, D, fn, e.st);
            if (!head_share(e, m, e.ao, T, err, !mk)) return false;
        }
        h_idx[0] = y;
        if (mk) {
            // the markov steps over the group: each member its rows of the bias and its share's argmax, every member
            // then the global one (the sampler's tie rule) and its E row - the same drafts as the lead's whole rows,
            // each step ~1/nm of their 66 MB and no gather of the shares
            const int nm = (int) hg.size();
            const unsigned seq = ++mk_seq;
            for (int m = 0; m < nm; ++m) {
                Dev& e = devs[(size_t) hg[(size_t) m]];
                cudaSetDevice(e.id);
                CK(cudaMemcpyAsync(e.mk_ints, h_idx, 4, cudaMemcpyHostToDevice, e.st));
                CK(cudaMemcpyAsync(e.mk_erow, h_mk_embed + (size_t) y * R, (size_t) R * 2, cudaMemcpyHostToDevice, e.st));
            }
            for (int i = 0; i < T; ++i)
                for (int m = 0; m < nm; ++m) {
                    Dev& e = devs[(size_t) hg[(size_t) m]];
                    cudaSetDevice(e.id);
                    float* row = m == 0 ? hd.logits + (size_t) i * V : e.logits_part + (size_t) i * e.mk_vn;
                    markov_share(row, e.mk_share, e.mk_erow, e.mk_vn, R, e.st);
                    argmax_share(row, e.mk_vn, e.mk_v0, mk_slots + ((size_t) i * nm + m) * 4, seq, e.st);
                    markov_pick(mk_slots + (size_t) i * nm * 4, nm, seq, e.mk_ints + i + 1, h_mk_embed, e.mk_erow, R,
                                e.st);
                }
            cudaSetDevice(hd.id);
        } else {
            CK(cudaMemcpyAsync(d_ints, h_idx, 4, cudaMemcpyHostToDevice, hd.st));
            strata::dsv41k::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            if (drafts_q) {   // the request's own chain: draft i drawn as the model's token i would be (verify's counter + i)
                sp.greedy = false;
                sp.temperature = dq_temp * (q_temp_scale > 0.0f ? q_temp_scale : 1.0f);
                sp.top_p = dq_top_p;
                sp.top_k = dq_top_k;
                sp.min_p = dq_min_p;
                sp.seed = dq_seed;
            }
            for (int i = 0; i < T; ++i) {   // the markov bias of the token before, then that position's draft
                markov_add(hd.logits + (size_t) i * V, d_mk_embed, d_mk_head, d_ints + i, V, R, hd.st);
                if (drafts_q) {
                    sp.counter = dq_counter + (uint64_t) i;
                    strata::dsv41k::sample_draft(hd.logits + (size_t) i * V, 1, V, sp, d_specq + i, d_ints + i + 1, hd.st);
                } else
                    strata::dsv41k::sample_tokens(hd.logits + (size_t) i * V, 1, V, nullptr, 0, sp, d_ints + i + 1, hd.st);
            }
        }
        confidence(hd.x, d_mk_embed, d_ints, d_conf, T, D, R, d_confv, hd.st);
        int tok[kWin + 1];
        float cf[kWin];
        CK(cudaMemcpyAsync(h_idx, d_ints, (size_t) (T + 1) * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaMemcpyAsync(h_w, d_confv, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaStreamSynchronize(hd.st));
        for (int i = 0; i <= T; ++i) tok[i] = h_idx[i];
        for (int i = 0; i < T; ++i) cf[i] = h_w[i];
        if (dump_d != nullptr) {   // 'D', the block's position, y, T, then the drafts, their confidences, the hidden
            std::vector<float> xs((size_t) T * D);   // rows before the final norm; then each block's ring and stream
            CK(cudaMemcpy(xs.data(), hd.x, xs.size() * 4, cudaMemcpyDeviceToHost));
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
            const int64_t hdr[4] = {'D', p0, y, T};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(tok + 1, 4, (size_t) T, dump_d);
            std::fwrite(cf, 4, (size_t) T, dump_d);
            std::fwrite(xs.data(), 4, xs.size(), dump_d);
            for (int i = 0; i < g.n_mtp; ++i) {
                std::fwrite(h_dump_ring + (size_t) i * rs, 2, rs, dump_d);
                std::fwrite(h_dump + (size_t) i * hs, 4, hs, dump_d);
            }
            std::fflush(dump_d);
            cudaSetDevice(hd.id);
        }
        for (int i = 0; i < T; ++i) {
            out.push_back(tok[i + 1]);
            conf.push_back(cf[i]);
        }
        return true;
    }

    int sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                   const float* bias) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::dsv41k::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        sp.logit_bias = bias;
        strata::dsv41k::sample_tokens(d.logits + (size_t) row * g.vocab, 1, g.vocab, nullptr, 0, sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    // [y, drafts] at positions pos ..: the trunk's verdict, its next token, and the state back to what it kept.
    // Greedy: the drafts the trunk's argmax agrees with.  Sampled: the speculative-sampling rule over the request's own
    // chain (top_k, top_p, min_p, temperature): draft d stands with probability min(1, p(d) / q(d)), q the distribution
    // it was drawn from (the drafter's argmax: a point at d, so p(d)), a rejected one is replaced by a draw from the
    // residual max(0, p - q) - so each kept token is distributed as the trunk's own draw.
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err) {
        const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
        const int T = n + 1, V = g.vocab;
        const int64_t p0 = pos;
        int tk[kWin];
        tk[0] = y;
        for (int i = 0; i < n; ++i) tk[i + 1] = drafts[(size_t) i];
        spec_win = true;
        const double tv = now_ms();
        const bool ok = forward(tk, T, T, true, err, 0);
        st.verify_ms += now_ms() - tv;
        spec_win = false;
        if (!ok) return false;
        Dev& hd = devs[(size_t) head_dev];
        cudaSetDevice(hd.id);
        int m = 0;
        if (temperature <= 0.0f) {
            strata::dsv41k::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            strata::dsv41k::sample_tokens(hd.logits, T, V, nullptr, 0, sp, d_ints + 16, hd.st);   // (the T rows at once)
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = h_idx[m];
        } else {
            // row i keeps draft i with probability min(1, p(d) / q(d)), else draws from the residual max(0, p - q)
            // (never d itself): each row a token distributed as the model's own draw at that position, p through the
            // request's chain - the top_k / top_p / min_p cut included (the drafter's argmax can lie outside it: p(d) 0,
            // never kept); past the first rejection the rows do not count
            const bool own = drafts_q && dq_temp == temperature && dq_top_p == top_p && dq_top_k == top_k &&
                             dq_min_p == min_p && dq_seed == seed && dq_counter == counter;
            if (!own) {   // the drafter's argmax: q a point at each draft
                for (int i = 0; i < n; ++i) {
                    h_specq[(size_t) i].n = 1;
                    h_specq[(size_t) i].ids[0] = drafts[(size_t) i];
                    h_specq[(size_t) i].p[0] = 1.0;
                }
                CK(cudaMemcpyAsync(d_specq, h_specq.data(), (size_t) n * sizeof(strata::dsv41k::SpecList),
                                   cudaMemcpyHostToDevice, hd.st));
            }
            for (int i = 0; i < n; ++i) h_idx[i] = drafts[(size_t) i];
            CK(cudaMemcpyAsync(d_ints + 32, h_idx, (size_t) n * 4, cudaMemcpyHostToDevice, hd.st));
            strata::dsv41k::SamplerParams sp;
            sp.greedy = false;
            sp.temperature = temperature;
            sp.top_p = top_p;
            sp.top_k = top_k;
            sp.min_p = min_p;
            sp.seed = seed;
            sp.counter = counter;
            strata::dsv41k::sample_verify(hd.logits, n, V, sp, d_specq, d_ints + 32, d_ints + 16, hd.st);
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) n * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = m < n ? h_idx[m] : sample_row(n, temperature, top_p, top_k, min_p, seed, counter + (uint64_t) n, nullptr);
        }
        drafts_q = false;
        rollback(p0, T, m + 1);
        if (o.remote_stage && !stage_keep(m + 1, err)) return false;   // (the worker's layers: the same rows)
        if (!seed_drafter(p0, m + 1, 0, err)) return false;
        accepted = m;
        st.drafts_offered += n;
        st.drafts_accepted += m;
        for (int i = 0; i < n && i <= m && i < 8; ++i) {
            ++st.draft_reached[i];
            if (i < m) ++st.draft_kept[i];
        }
        return true;
    }

    // a verified window keeps its first `keep` positions: the ring rows and the compressors' state of the rest undone
    // (the compressed KV and the index keys past them are written again when those positions come)
    void rollback(int64_t p0, int T, int keep) {
        if (keep >= T) return;
        // the ring rows: one launch a device for all its layers (a launch a layer and member was ~300 on 8-way tensor
        // parallel, a millisecond of the host's before the next draft)
        std::vector<RingBatch> rb(devs.size());
        for (int il = lo; il < hi; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                RingBatch& b = rb[(size_t) l.dev];
                if (b.n == kMaxRingBatch) {   // (more layers on a device than a batch takes: this one alone)
                    cudaSetDevice(devs[(size_t) l.dev].id);
                    ring_restore(l.win, l.ring_bak, p0, g.win, keep, T, g.hd, devs[(size_t) l.dev].st);
                    continue;
                }
                b.ring[b.n] = l.win;
                b.bak[b.n++] = l.ring_bak;
            }
        for (size_t di = 0; di < devs.size(); ++di)
            if (rb[di].n > 0) {
                cudaSetDevice(devs[di].id);
                ring_restore_batch(rb[di], p0, g.win, keep, T, g.hd, devs[di].st);
            }
        for (int il = lo; il < hi; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                Dev& d = devs[(size_t) l.dev];
                if (l.c_snap != nullptr) {
                    cudaSetDevice(d.id);
                    const size_t ns = (size_t) l.ratio * g.hd * 4;
                    cudaMemcpyAsync(l.c_kv, l.c_snap, ns, cudaMemcpyDeviceToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, l.c_snap + (size_t) l.ratio * g.hd, ns, cudaMemcpyDeviceToDevice, d.st);
                    compress_win(l.c_rows, l.c_rows + (size_t) kWin * g.hd, l.c_kv, l.c_score, l.ratio, (int) p0, keep,
                                 g.hd, l.c_norm, g.eps, d.latent, d.st);
                }
            }
        pos = p0 + keep;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        slot_held = false;
    }

    // ---- a window: T tokens at positions pos .. pos + T - 1 through every layer; the logits of its last `logits` rows
    bool step(int token, bool want_logits, std::string& err) { return forward(&token, 1, want_logits ? 1 : 0, true, err); }


    // layer L0 for a chunk of T tokens at p0 as group member mi: layer()'s arithmetic over the chunk's rows (the
    // projections as GEMMs, the attention and the indexer in slabs, the routed experts as sorted rows: the VRAM ones in
    // place, the CPU lane's streamed in batches).  kv_only (a long prompt's first pass, Run::stop): only what the layers
    // after it read of these rows - its compressed KV and index keys - and nothing of its own attention or experts
    bool layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                     bool kv_only = false) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        Chunk& c = *chunks[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        prof_mark(d, 0);
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ab = groups[(size_t) L0.grp].arb;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        // Engram, in slabs of 256 tokens
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            const int slot = wp.eg_slot >= 0 ? wp.eg_slot : wp.k % 2;   // (read ahead by the run: two in flight)
            uint8_t* host = c.eg_host[slot] + (size_t) l.eg * c.C * per;
            CK(cudaMemcpyAsync(c.eg_rows, host, (size_t) T * per, cudaMemcpyHostToDevice, s));
            CK(cudaEventRecord(c.eg_ev[slot], s));
            for (int s0 = 0; s0 < T; s0 += 256) {
                const int ns = std::min(256, T - s0);
                engram_dequant_tok(c.eg_rows + (size_t) s0 * per, per, cols, ns, ehd, c.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
                x3::rows_f16(c.eg_x, cols * ehd, nullptr, ns, cols * ehd, c.eg_x16, s);
                if (!gemm(d, c, l.eg_wkv, c.eg_x16, cols * ehd, ns, c.eg_kv, (H + 1) * D, 0.0f, err)) return false;
                engram_gate(c.h + (size_t) s0 * H * D, c.eg_kv, l.eg_w, H, D, ns, g.eps, s);
            }
        }
        prof_mark(d, 4);
        // attention
        if (!hc_mix_chunk(d, c, c.h, l.hca_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, c.a_pre, c.a_post, c.a_comb, s);
        hc_pre_norm(c.h, c.pre, l.attn_norm, D, H, T, g.eps, c.x, s);
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        prof_mark(d, 5);
        if (!kv_only && (!gemm(d, c, l.wq_a, c.x16, D, T, c.qr, g.q_lora, 0.0f, err) ||
                         !gemm(d, c, l.wkv, c.x16, D, T, c.kvr, HD, 0.0f, err)))
            return false;
        if (l.kv_src && !gemm(d, c, l.c_wkv, c.x16, D, T, c.ckr, HD, 0.0f, err)) return false;
        if (l.kv_src && l.ratio > 1 && !gemm(d, c, l.c_wgate, c.x16, D, T, c.cgr, HD, 0.0f, err)) return false;
        prof_mark(d, 6);
        const bool snap = !kv_only && wp.snap_t >= 0 && mi == 0 && L0.snap_win != nullptr;
        if (!kv_only) {
            Finish fq;
            fq.w = l.q_norm;
            fq.eps = g.eps;
            finish(c.qr, T, g.q_lora, g.q_lora, fq, s);
            x3::rows_f16(c.qr, g.q_lora, nullptr, T, g.q_lora, c.qr16, s);
            // the window rows: the ring's (positions p0 - win ..) then the chunk's
            ring_to_ext(l.win, c.ext, p0, g.win, HD, s);
            Finish fk;
            fk.w = l.kv_norm;
            fk.eps = g.eps;
            fk.rd = g.rd;
            fk.pos = (int) p0;
            fk.freqs = freqs;
            fk.quant = 1;
            fk.dst = c.ext + (size_t) g.win * HD;
            fk.dst_stride = HD;
            finish(c.kvr, T, HD, HD, fk, s);
            // feed_mark's checkpoint: the ring as it was before row snap_t - the old ring, then the chunk's rows before it
            if (snap) {
                CK(cudaMemcpyAsync(L0.snap_win, l.win, (size_t) g.win * win_rb(), cudaMemcpyDeviceToDevice, s));
                if (wp.snap_t > 0) ext_to_ring(c.ext, L0.snap_win, p0, g.win, wp.snap_t, HD, s, nullptr);
            }
            ext_to_ring(c.ext, l.win, p0, g.win, T, HD, s, nullptr);
        }
        const uint8_t* ckv = nullptr;
        const long long ss = (long long) o.max_context + 64;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            const int64_t g_first = p0 / R, g_end = (p0 + T) / R;
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            const float* latent = c.latent;
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(c.ckr, T, HD, HD, fc, s);
                    latent = c.ckr;
                } else {
                    if (snap && L0.snap_kv != nullptr) {   // (each slot: the latest row before snap_t that holds it)
                        const int64_t ps = p0 + wp.snap_t;
                        for (int sl = 0; sl < R; ++sl) {
                            const int64_t q = ps - 1 - (((ps - 1 - sl) % R) + R) % R;
                            const float* kv = q >= p0 ? c.ckr + (size_t) (q - p0) * HD : l.c_kv + (size_t) sl * HD;
                            const float* sc = q >= p0 ? c.cgr + (size_t) (q - p0) * HD : l.c_score + (size_t) sl * HD;
                            CK(cudaMemcpyAsync(L0.snap_kv + (size_t) sl * HD, kv, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                            CK(cudaMemcpyAsync(L0.snap_sc + (size_t) sl * HD, sc, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                        }
                    }
                    compress_win(c.ckr, c.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, c.latent, s);
                }
            }
            const int clen_max = (int) ((p0 + T) / R);
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    x3::rows_f16(latent, HD, nullptr, ng, HD, c.lat16, s);
                    if (!gemm(d, c, l.i_wk, c.lat16, HD, ng, c.ikr, g.ix_hd, 0.0f, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(c.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                }
                if (clen_max > 0 && !kv_only) {
                    if (!gemm_f16w(c, l.i_wproj, g.ix_heads, D, c.x16, D, T,
                                   1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), c.iw, 64, err))
                        return false;
                    for (int a0 = 0; a0 < T; a0 += c.AI) {
                        const int na = std::min(c.AI, T - a0);
                        const int64_t pa = p0 + a0;
                        if (!gemm(d, c, l.i_wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.iq,
                                  g.ix_heads * g.ix_hd, 0.0f, err))
                            return false;
                        Finish fi;
                        fi.rd = g.rd;
                        fi.pos = (int) pa;
                        fi.vpp = g.ix_heads;
                        fi.freqs = freqs;
                        fi.quant = 3;
                        finish(c.iq, na * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                        Rows lim;
                        lim.gen = 1;
                        lim.p0 = pa;
                        lim.R = R;
                        lim.kcap = g.ix_topk;
                        lim.ccap = g.cand_topk;
                        lim.blk = std::max(1, g.cand_blk);
                        const int nlast = (int) ((pa + na) / R);   // the slab's widest query
                        int* tk = c.topk + (size_t) a0 * topk_stride();
                        int* cd = c.cand + (size_t) a0 * cand_stride();
                        const float* iw = c.iw + (size_t) a0 * 64;
                        if (nlast <= 0) continue;
                        if (l.uses_cand && (nlast + g.cand_blk - 1) / g.cand_blk > g.cand_topk) {
                            Rows cl = lim;
                            cl.gen = 2;
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, g.cand_topk * g.cand_blk,
                                        cd, (int) cand_stride(), g.cand_blk, cl, c.scores, ss, s);
                            Rows tr = lim;
                            tr.gen = 3;
                            topk(c.scores, ss, na, tr, cd, (int) cand_stride(), g.cand_blk, tk, (int) topk_stride(), s);
                        } else {
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, nlast, nullptr, 0, 1,
                                        lim, c.scores, ss, s);
                            if (l.cand_src) {
                                Rows cr = lim;
                                cr.kcap = g.cand_topk;
                                cand_blocks(c.scores, ss, na, cr, g.cand_blk, c.bm, (long long) bm_stride(), cd,
                                            (int) cand_stride(), s);
                            }
                            topk(c.scores, ss, na, lim, nullptr, 0, 1, tk, (int) topk_stride(), s);
                        }
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(const_cast<float*>(latent), ng, HD, HD, fc, s);
                if (mi == 0 && sr.remote) {   // the new rows for the other groups' readers (slot k % 2, once
                    const int slot = wp.k % 2;  // they took the chunk before last's)
                    if (wp.k >= 2)
                        for (int r : sr.readers) wait_flags(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, 1, 32, wp.win - 2, s);
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.stage(slot);
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.stage(slot) + x_cap() * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.flag(slot), wp.win, d.xcnt, s);
                }
            }
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (kv_only) return true;
        prof_mark(d, 7);
        // the attention in slabs: q, the sparse attention, wo_a, wo_b (this member's share) into acc
        if (NHL == 0) CK(cudaMemsetAsync(c.acc, 0, (size_t) T * D * sizeof(*c.acc), s));   // (no heads)
        // DS4_DQ_ONCE (more than one slab): wq_b, wo_a's groups and wo_b reconstructed once for the chunk, in the pieces
        // gemm() cuts, and every slab's GEMMs over them (the same operands and calls: the same bits)
        const bool once = NHL > 0 && T > c.A && c.w16_once != nullptr && once_elems(l) <= c.w16_once_elems;
        const gemm16::Places wp16 = once ? gemm16::places(c.w16_once, l.wq_b, l.wo_a, OG, l.wo_b) : gemm16::Places{};
        if (once) {
            dq_whole(d, c, l.wq_b, wp16.qb);
            for (int gi = 0; gi < OG; ++gi) dq_whole(d, c, l.wo_a[gi], wp16.oa[gi]);
            dq_whole(d, c, l.wo_b, wp16.ob);
        }
        for (int a0 = 0; a0 < T && NHL > 0; a0 += c.A) {
            const int na = std::min(c.A, T - a0);
            if (!gemm(d, c, l.wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.q, HQ, 0.0f, err, wp16.qb)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) (p0 + a0);
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(c.q, na * NHL, HD, HD, fr, s);
            prof_mark(d, 8);
            AttnRows ar;
            ar.gen = 1;
            ar.p0 = p0;
            ar.t0 = a0;
            ar.win = g.win;
            ar.R = l.ratio;
            ar.kcap = g.ix_topk;
            sparse_attn(c.q, c.ext, ckv, c.topk + (size_t) a0 * topk_stride(), (int) topk_stride(), ar, l.sink, NHL, HD,
                        na, 1.0f / std::sqrt((float) HD), c.o, s, kKvFp4, 0, &d.attn_scr);   // (a short last slab)
            prof_mark(d, 9);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) (p0 + a0);
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(c.o, na * NHL, HD, HD, fo, s);
            x3::rows_f16(c.o, HQ, nullptr, na, HQ, c.o16, s);
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                if (!gemm(d, c, l.wo_a[gi], c.o16 + (size_t) gi * per, HQ, na, c.oa + (size_t) gi * g.o_lora,
                          OG * g.o_lora, 0.0f, err, wp16.oa[gi]))
                    return false;
            x3::rows_f16(c.oa, OG * g.o_lora, nullptr, na, OG * g.o_lora, c.oa16, s);
            if (!gemm(d, c, l.wo_b, c.oa16, OG * g.o_lora, na, c.acc + (size_t) a0 * D, D, 0.0f, err, wp16.ob)) return false;
        }
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        hc_post_inplace(c.acc, c.h, c.a_post, c.a_comb, D, H, T, s);
        prof_mark(d, 1);
        // the MoE
        if (!hc_mix_chunk(d, c, c.h, l.hcf_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, c.f_pre, c.f_post, c.f_comb, s);
        hc_pre_norm(c.h, c.a_pre, l.ffn_norm, D, H, T, g.eps, c.x, s);
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        gate_gemv(l, c.x, T, c.glog, s);   // (F32 x: the token path's bits)
        for (int r0 = 0; r0 < T; r0 += kVlRows) {   // (route()'s image bits cover kVlRows rows a call)
            const int nr = std::min(kVlRows, T - r0);
            VlBits vb;
            const bool vl = l.gate_bias_vl != nullptr && vl_bits(p0 + r0, nr, vb);
            route(c.glog + (size_t) r0 * 1024, l.gate_bias, l.n_exp, l.k_exp, nr, g.route_scale,
                  c.ridx + (size_t) r0 * kPicks, c.rw + (size_t) r0 * kPicks, s, l.gate_bias_vl, vl ? &vb : nullptr);
        }
        const int n_vs = L0.n_slots + (int) L0.host_list.size();
        if (bc_once != 0 && c.flow && c.vs_k == wp.k && c.vs_il == L0.il && c.vs_mi == mi)
            CK(cudaStreamWaitEvent(s, c.vs_ev, 0));   // (DS4_BC_ONCE: sent on cs when flow_spec decided them)
        else
            CK(cudaMemcpyAsync(l.vs_dev, l.vs_pin, (size_t) l.n_exp * 4, cudaMemcpyHostToDevice, s));   // (this member's)
        moe_sort(c.ridx, T, l.k_exp, l.vs_dev, n_vs, c.bounds, c.row_tok, c.pos, s);
        CK(cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (n_vs + 1) * 4, cudaMemcpyDeviceToHost, s));
        prof_mark(d, 10);
        // the shared expert into acc while the host waits for the bounds
        if (!gemm(d, c, l.s1, c.x16, D, T, c.sg, FF, 0.0f, err) || !gemm(d, c, l.s3, c.x16, D, T, c.su, FF, 0.0f, err))
            return false;
        swiglu(c.sg, c.su, T * FF, g.lim, c.sg, s);
        x3::rows_f16(c.sg, FF, nullptr, T, FF, c.sh16, s);
        if (!gemm(d, c, l.s2, c.sh16, FF, T, c.acc, D, 0.0f, err)) return false;
        prof_mark(d, 2);
        // the expert stream's copies keep going while the device runs this layer's attention, routing and shared expert:
        // the next items (this layer's likely experts, queued when the layer before routed) go out as they are ready and
        // ring slots free, not only once the bounds are back (Project Maya #360; DS4_FLOW_PUMP_WAIT=0: wait as before)
        static const bool pump_wait = std::getenv("DS4_FLOW_PUMP_WAIT") == nullptr ||
                                      std::atoi(std::getenv("DS4_FLOW_PUMP_WAIT")) != 0;
        if (c.flow && pump_wait)
            while (cudaStreamQuery(s) == cudaErrorNotReady) {
                if (!flow_pump(c, err)) return false;
                std::this_thread::yield();
            }
        CK(cudaStreamSynchronize(s));
        std::vector<int> b(c.bounds_pin, c.bounds_pin + n_vs + 1);
        if (dump_r != nullptr && mi == 0) {
            const size_t nk = (size_t) T * l.k_exp, np = (size_t) T * kPicks;
            std::vector<int> ri(np), rt(nk);
            std::vector<float> gl((size_t) T * l.n_exp), gb((size_t) l.n_exp);
            CK(cudaMemcpy(ri.data(), c.ridx, np * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rt.data(), c.row_tok, nk * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy2D(gl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                            cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gb.data(), l.gate_bias, gb.size() * 4, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(dump_mu);
            const int64_t hdr[8] = {l.il, p0, T, l.k_exp, l.n_exp, n_vs, L0.n_slots, kPicks};
            std::fwrite(hdr, 8, 8, dump_r);
            std::fwrite(ri.data(), 4, np, dump_r);
            std::fwrite(L0.vs_pin, 4, (size_t) l.n_exp, dump_r);
            std::fwrite(b.data(), 4, b.size(), dump_r);
            std::fwrite(rt.data(), 4, nk, dump_r);
            std::fwrite(gl.data(), 4, gl.size(), dump_r);
            std::fwrite(gb.data(), 4, gb.size(), dump_r);
            std::fflush(dump_r);
        }
        if (mi == 0 && !l.mtp && (pred_f != nullptr || xin_f != nullptr)) {
            std::lock_guard<std::mutex> lk(dump_mu);
            if (pred_f != nullptr) {
                float* dp = nullptr;
                CK(cudaMalloc(&dp, (size_t) T * 256 * 4));
                pred_proj_k<<<T, 256, 0, s>>>(c.x, (int) D, dp);
                std::vector<float> hp((size_t) T * 256), hl((size_t) T * l.n_exp);
                CK(cudaMemcpyAsync(hp.data(), dp, hp.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpy2DAsync(hl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                                     cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                CK(cudaFree(dp));
                std::vector<uint16_t> o(hp.size() + hl.size());
                for (size_t i = 0; i < hl.size(); ++i) o[i] = strata::kernels::f16_from_f32(hl[i]);
                for (size_t i = 0; i < hp.size(); ++i) o[hl.size() + i] = strata::kernels::f16_from_f32(hp[i]);
                const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (T & 255), (uint8_t) (T >> 8)};
                std::fwrite(hd, 1, 3, pred_f);
                std::fwrite(o.data(), 2, o.size(), pred_f);
            }
            if (xin_f != nullptr && xin_got[(size_t) l.il] < xin_n) {   // up to 32 evenly spaced tokens of the chunk
                const int tk = std::min(std::min(T, 32), xin_n - xin_got[(size_t) l.il]);
                const int stride = std::max(1, T / tk);
                std::vector<float> hx((size_t) T * D);
                CK(cudaMemcpyAsync(hx.data(), c.x, hx.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (tk & 255), (uint8_t) (tk >> 8)};
                std::fwrite(hd, 1, 3, xin_f);
                for (int i = 0; i < tk; ++i) std::fwrite(hx.data() + (size_t) std::min(T - 1, i * stride) * D, 4, (size_t) D, xin_f);
                xin_got[(size_t) l.il] += tk;
            }
        }
        if (mi == 0 && trace_f != nullptr && !l.mtp) {   // DS4_TRACE: a chunk's picks, u8 (layer | 0x80), u16 T, T x k int16 ids
            std::vector<int> ri((size_t) T * kPicks);
            CK(cudaMemcpy(ri.data(), c.ridx, ri.size() * 4, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(dump_mu);
            const uint8_t hd[3] = {(uint8_t) (l.il | 0x80), (uint8_t) (T & 255), (uint8_t) (T >> 8)};
            std::fwrite(hd, 1, 3, trace_f);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < l.k_exp; ++i) {
                    const int16_t v = (int16_t) ri[(size_t) t * kPicks + i];
                    std::fwrite(&v, 2, 1, trace_f);
                }
        }
        if (mi == 0) {   // the profile (the CPU lane's experts: by the plan's order)
            uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) {
                const int v = L0.vs_pin[e];
                const int cnt = b[(size_t) v + 1] - b[(size_t) v];
                pr[e] += (uint32_t) cnt;
                if (cnt > 0) {   // (the expert cache learns from a chunked prompt as from windows: heat, and the CPU lane's hits)
                    l.heat[(size_t) e] += (float) cnt;
                    if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                }
            }
            st.experts += (int64_t) T * l.k_exp;
        }
        if (c.flow) {
            if (!flow_layer(d, c, L0, mi, wp.k, T, b, err)) return false;
            if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
            hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
            CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            prof_mark(d, 3);
            return true;
        }
        // the streamed experts as their batches arrive (each batch's slots freed for the copies at once), the VRAM ones
        // in shares between them (the device works while the copies run)
        const int di = l.dev, P = Chunk::kParts;
        int nb = 0;
        for (size_t q = S.kp; q < S.b.size() && S.b[q].il == L0.il && S.b[q].k == wp.k; ++q) ++nb;
        int v_done = 0;
        for (int bi = 0; bi < nb; ++bi) {
            const Batch& bt = S.b[S.kp];
            const int half = (int) (S.kp % P);
            CK(cudaStreamWaitEvent(s, c.filled[half], 0));
            const int v0 = L0.n_slots + bt.j0;
            if (!run_set(d, c, l, c.ring + (size_t) half * (c.ring_n / P) * c.ring_stride, v0, c.ring_stride, c.bounds,
                         b.data(), v0, v0 + bt.n, T, err))
                return false;
            CK(cudaEventRecord(c.freed[half], s));
            ++S.kp;
            if (!copy_ahead(d, c, S, di, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (bi + 1) / nb);
            if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
        CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        prof_mark(d, 3);
        return true;
    }

    // the CPU lane's experts of every layer, in index order, and each expert's virtual slot (before a chunk)
    void plan_streams() {
        if (bc_once != 0) {   // (DS4_BC_ONCE: a run that stopped before its sort leaves flow_spec's copy maybe on cs)
            int cur = 0;
            cudaGetDevice(&cur);
            for (auto& cp : chunks)
                if (cp != nullptr && cp->vs_ev != nullptr) {
                    cudaSetDevice(devs[(size_t) cp->dev].id);
                    cudaEventSynchronize(cp->vs_ev);
                    cp->vs_k = cp->vs_il = cp->vs_mi = -1;
                }
            cudaSetDevice(cur);
        }
        for (Layer& l : L) {
            if (l.mtp || l.vs_pin == nullptr) continue;
            l.host_list.clear();
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0) l.host_list.push_back(e);
            for (int m = 0; m < l.nm; ++m) {   // (a group's members: each its own copy)
                Layer& x = member(l, m);
                int j = 0;
                for (int e = 0; e < l.n_exp; ++e) x.vs_pin[e] = l.slot[(size_t) e] >= 0 ? l.slot[(size_t) e] : l.n_slots + j++;
            }
        }
    }

    // a chunk's hand-off between groups: h | pre | topk | cand of a chunk of T rows
    Xfer chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {c.h, c.pre, c.topk, c.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    // a run of chunks (start positions p0[k], sizes T[k], seqs seq[k]) on device di: its layers chunk after chunk, the
    // groups pipelined (a group takes chunk k + 1 while the next takes chunk k: the hand-offs and the sources' rows
    // two deep, each slot reused once its readers acknowledged it), the streamed experts' copies running ahead across
    // layers and chunks
    struct Run {
        std::vector<int64_t> p0;
        std::vector<int> T;
        std::vector<unsigned> seq;
        int64_t snap_pos = -1;   // feed_mark: the position whose state before it the layers keep (-1: none)
        // a long prompt's skip (forward_chunks): the cut at row cut_r of chunk cut_k - the chunks before it go through
        // the layers before `yoco`, that layer as layer_chunk's kv_only and none after it; chunk cut_k's rows before
        // cut_r the same, its rows from cut_r on and the chunks after it through every layer (yoco -1: no skip)
        int yoco = -1, cut_k = 0, cut_r = 0;
        bool runs(int k, int il) const { return yoco < 0 || k >= cut_k || il <= yoco; }
        bool kv_only(int k, int il) const { return yoco >= 0 && k < cut_k && il == yoco; }
        int r0(int k, int il) const { return yoco >= 0 && k == cut_k && il >= yoco ? cut_r : 0; }   // the first row
    };
    // layer L0 over chunk k per a long prompt's skip (Run): its kv_only, or its rows from Run::r0 on - the chunk's
    // buffers seen from that row (the yoco layer: its kv_only over the rows before it first)
    bool run_layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                   const Run& run, int k, int il) {
        if (run.kv_only(k, il)) return layer_chunk(L0, mi, p0, T, wp, S, err, true);
        const int r0 = run.r0(k, il);
        if (r0 <= 0) return layer_chunk(L0, mi, p0, T, wp, S, err);
        if (il == run.yoco) {
            WinPlan w0 = wp;
            w0.snap_t = -1;
            if (!layer_chunk(L0, mi, p0, r0, w0, S, err, true)) return false;
        }
        Chunk& c = *chunks[(size_t) member(L0, mi).dev];
        float* const h0 = c.h;
        float* const pre0 = c.pre;
        c.h += (size_t) r0 * g.hc * g.dim;
        c.pre += (size_t) r0 * 8;
        WinPlan w1 = wp;
        w1.snap_t = wp.snap_t >= r0 ? wp.snap_t - r0 : -1;
        const bool ok = layer_chunk(L0, mi, p0 + r0, T - r0, w1, S, err);
        c.h = h0;
        c.pre = pre0;
        return ok;
    }

    // ---- a remote stage's prompt chunks (ds4_stage.hpp: the rows to the worker block by block, its reply row by row)
    // (the worker) the chunk's rows from the link (stage_in) into c's h | pre | topk | cand, on stream s
    bool link_chunk_in(Chunk& c, int T, cudaStream_t s, std::string& err) {
        if (stage_in == nullptr) {
            err = "remote stage: a prompt chunk without its rows";
            return false;
        }
        const Xfer x = chunk_xfer((uint8_t*) const_cast<float*>(stage_in), c, T, false);
        for (int i = 0; i < x.n; ++i) CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, s));
        return true;
    }
    // the rows chunk k's reply leaves out: none with every row's logits, all but the head's of the run's last chunk,
    // every one of the chunks before it
    int link_skip(const Run& run, int k) const {
        const int T = run.T[(size_t) k];
        return logits_all ? 0 : k + 1 == (int) run.T.size() ? T - n_logits : T;
    }
    // where chunk k's row 0 sits in the main process's buffer of replies (rx): the run's rows with every row's logits,
    // else one chunk's - only the last chunk's reply has rows then, and every other's writes none
    size_t link_rx_row(const Run& run, int k) const {
        return logits_all ? (size_t) (run.p0[(size_t) k] - run.p0[0]) : 0;
    }
    // chunk k through this process's last layer (device d, its chunk state c).  The worker: its rows from stage_skip on
    // into stage_out, row by row (streams, then pre).  The main process: its rows into slot k % 2 - the send two chunks
    // back is out (the one before this one waited for it) - and to the worker on the link's thread, once the chunk
    // before is out; the device's next chunk runs meanwhile
    bool link_chunk_out(Dev& d, Chunk& c, const Run& run, int k, std::string& err) {
        const int T = run.T[(size_t) k];
        const size_t W = (size_t) full_floats(), HD = (size_t) g.hc * g.dim;
        cudaSetDevice(d.id);
        if (o.stage_worker) {
            const int sk = std::max(0, std::min(stage_skip, T)), nr = T - sk;
            if (nr <= 0) return true;
            if (stage_out == nullptr) {
                err = "remote stage: a prompt chunk with nowhere for its rows";
                return false;
            }
            CK(cudaMemcpy2DAsync(stage_out + (size_t) sk * W, W * 4, c.h + (size_t) sk * HD, HD * 4, HD * 4, (size_t) nr,
                                 cudaMemcpyDeviceToHost, d.st));
            CK(cudaMemcpy2DAsync(stage_out + (size_t) sk * W + HD, W * 4, c.pre + (size_t) sk * 8, 32, 32, (size_t) nr,
                                 cudaMemcpyDeviceToHost, d.st));
            return true;
        }
        const int64_t p0 = run.p0[(size_t) k];
        float* const tx = link_cx[k % 2];
        if (!stage_chunks.send || tx == nullptr) {
            err = "remote stage: no link for prompt chunks";
            return false;
        }
        const Xfer x = chunk_xfer((uint8_t*) tx, c, T, true);
        for (int i = 0; i < x.n; ++i) CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, d.st));
        CK(cudaStreamSynchronize(d.st));   // (pageable: done as each copy returns - and so is the chunk)
        if (!stage_chunks.send(toks.data() + p0, T, p0, k + 1 == (int) run.T.size(), tx, link_skip(run, k), err))
            return false;
        {
            std::lock_guard<std::mutex> lk(link_mu);
            ++link_sent;
        }
        link_cv.notify_all();
        return true;
    }
    // (the main process) the head over the worker's rows of the run - rx, row by row: with every row's logits the
    // run's rows from its first, else the last chunk's from its first (link_rx_row) - as the head group's device runs
    // it after the last layer (chunk by chunk, 16 rows at a time, with every row's logits; else the last chunk's last
    // n_logits rows)
    bool link_head(const Run& run, const float* rx, std::string& err) {
        Dev& d = devs[(size_t) head_dev];
        Chunk& c = *chunks[(size_t) head_dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc;
        const size_t W = (size_t) full_floats(), HD = (size_t) H * D;
        auto rows = [&](int64_t r, int nr) -> bool {   // the run's rows [r, r + nr) into the chunk state's first nr
            CK(cudaMemcpy2DAsync(c.h, HD * 4, rx + (size_t) r * W, W * 4, HD * 4, (size_t) nr, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpy2DAsync(c.pre, 32, rx + (size_t) r * W + HD, W * 4, 32, (size_t) nr, cudaMemcpyHostToDevice, s));
            return true;
        };
        const int nk = (int) run.T.size();
        if (logits_all) {
            for (int k = 0; k < nk; ++k) {
                const int64_t r = run.p0[(size_t) k] - run.p0[0];
                const int T = run.T[(size_t) k];
                for (int r0 = 0; r0 < T; r0 += kWin) {
                    const int nr = std::min(kWin, T - r0);
                    if (!rows(r + r0, nr)) return false;
                    hc_pre_norm(c.h, c.pre, d.norm_w, D, H, nr, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, nr, err)) return false;
                    CK(cudaMemcpyAsync(chunk_logits.data() + (size_t) (r + r0) * g.vocab, d.logits, (size_t) nr * g.vocab * 4,
                                       cudaMemcpyDeviceToHost, s));
                    CK(cudaStreamSynchronize(s));
                }
            }
        } else if (n_logits > 0) {
            if (!rows(run.T.back() - n_logits, n_logits)) return false;
            hc_pre_norm(c.h, c.pre, d.norm_w, D, H, n_logits, g.eps, d.x, s);
            if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
        }
        return true;
    }
    // (the main process) whether the worker holds this conversation: after a failed hop or chunk, or a worker's failed
    // reset, only a conversation from position 0 goes on - its reset tried again first
    bool stage_live(std::string& err) {
        if (stage_lost && pos == 0 && stage_reset && stage_reset(stage_err)) stage_lost = false;
        if (!stage_lost) return true;
        err = stage_err + " - the conversation starts again from its first token";   // (the link's: "remote stage ...")
        return false;
    }

    // ---- DS4_LM_STATE_OVERLAP: layer-major's state moves (h | pre | topk | cand) off the compute stream
    // a run's unit (layer, chunk) in run_dev_layers' order: its state loaded before it (every layer after the device's
    // first), saved after it (not a kv_only one's, not the last layer's), with its topk / cand (tc) only when a unit
    // since the chunk's last save could have written them (an index layer's, not kv_only: layer_chunk's indexer) or it
    // was never saved; a state saved and then loaded by the very next unit (the same chunk: the layers after a long
    // prompt's yoco layer) stays on the device, neither move made (stay) - the same bits; set: the state set it runs on
    struct LsUnit {
        int k = 0, T = 0, set = 0;
        bool load = false, save = false, tc = false, stay = false;
    };
    void ls_plan(const Chunk& c, const Run& run, std::vector<LsUnit>& lsu) const {
        const int nk = (int) run.T.size();
        std::vector<char> ix;   // (a unit's: an index layer's, not kv_only)
        for (size_t li = 0; li < c.fl_layers.size(); ++li) {
            const int il = c.fl_layers[li];
            for (int k = 0; k < nk; ++k) {
                if (!run.runs(k, il)) continue;
                LsUnit x;
                x.k = k;
                x.T = run.T[(size_t) k];
                x.load = li > 0;
                x.save = !run.kv_only(k, il) && li + 1 < c.fl_layers.size();
                x.stay = x.load && !lsu.empty() && lsu.back().save && lsu.back().k == k;
                if (x.stay) x.load = lsu.back().save = false;
                lsu.push_back(x);
                ix.push_back(L[(size_t) il].ix_src && !run.kv_only(k, il));
            }
        }
        std::vector<char> saved((size_t) nk, 0), dirty((size_t) nk, 0);
        int set = 1;
        for (size_t u = 0; u < lsu.size(); ++u) {
            LsUnit& x = lsu[u];
            if (ix[u]) dirty[(size_t) x.k] = 1;
            if (x.save) {
                x.tc = dirty[(size_t) x.k] || !saved[(size_t) x.k];
                saved[(size_t) x.k] = 1;
                dirty[(size_t) x.k] = 0;
            }
            if (!x.stay) set ^= 1;   // (ping-pong: each unit on the other set, a stay on its own)
            x.set = c.ls_mode == 1 ? set : 0;
        }
        if (c.ls_mode == 1 && !lsu.empty() && lsu.back().set == 1)   // (the run ends on the chunk's own buffers)
            for (LsUnit& x : lsu) x.set ^= 1;
    }
    static void ls_set(Chunk& c, int set) {
        c.h = c.ls_h[set];
        c.pre = c.ls_pre[set];
        c.topk = c.ls_tk[set];
        c.cand = c.ls_cd[set];
    }
    // chunk_xfer of state set `set`
    Xfer ls_xfer(Chunk& c, int set, int k, int T, bool put) const {
        float *const h = c.h, *const pre = c.pre;
        int *const tk = c.topk, *const cd = c.cand;
        ls_set(c, set);
        const Xfer x = chunk_xfer(c.lm_state + (size_t) k * chunk_hand_bytes(), c, T, put);
        c.h = h;
        c.pre = pre;
        c.topk = tk;
        c.cand = cd;
        return x;
    }
    // piece i of a move (x: chunk_xfer's, T rows): i < kLsPieces h's row slice i (the device's rows, C / kLsPieces a
    // slice: a save's piece i and the next load's touch the same bytes), then pre, topk, cand; false: an empty one
    bool ls_piece(const Chunk& c, const Xfer& x, int T, int i, const void*& src, void*& dst, size_t& n) const {
        if (i >= Chunk::kLsPieces) {
            const int r = i - Chunk::kLsPieces + 1;
            src = x.src[r];
            dst = x.dst[r];
            n = x.bytes[r];
            return n > 0;
        }
        const size_t rb = (size_t) g.hc * g.dim * 4, per = ((size_t) c.C + Chunk::kLsPieces - 1) / Chunk::kLsPieces;
        const size_t r0 = (size_t) i * per, r1 = std::min((size_t) T, r0 + per);
        if (r0 >= r1) return false;
        src = (const uint8_t*) x.src[0] + r0 * rb;
        dst = (uint8_t*) x.dst[0] + r0 * rb;
        n = (r1 - r0) * rb;
        return true;
    }
    // unit x's save (put: its topk / cand only with tc) or load on stream st, piece by piece: each piece after its
    // save's event where after[i], each piece's event recorded where rec
    bool ls_move(Chunk& c, const LsUnit& x, bool put, cudaStream_t st, const bool* after, bool* rec, std::string& err) {
        const Xfer xf = ls_xfer(c, x.set, x.k, x.T, put);
        for (int i = 0; i < Chunk::kLsPieces + 3; ++i) {
            if (put && !x.tc && i > Chunk::kLsPieces) continue;   // (topk, cand: unchanged since its last save)
            const void* src = nullptr;
            void* dst = nullptr;
            size_t n = 0;
            if (!ls_piece(c, xf, x.T, i, src, dst, n)) continue;
            if (after != nullptr && after[i]) CK(cudaStreamWaitEvent(st, c.ls_pc[i], 0));
            CK(cudaMemcpyAsync(dst, src, n, put ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, st));
            if (rec != nullptr) {
                CK(cudaEventRecord(c.ls_pc[i], st));
                rec[i] = true;
            }
        }
        return true;
    }
    // unit u (on chunk k) about to start on s: the unit before it done there (ls_comp of its set) and its save issued
    // on ls_out; this unit's state in - mode 2: loaded on ls_in now, a piece as soon as the save has it out (s waits
    // for all of it); mode 1: loaded into its set while the unit before computed (s waits for it), and the next unit's
    // load issued now, into the other set once the unit that used it last is done and saved
    bool ls_enter(Chunk& c, cudaStream_t s, const std::vector<LsUnit>& lsu, size_t u, int k, std::string& err) {
        if (u >= lsu.size() || lsu[u].k != k) {
            err = "DS4_LM_STATE_OVERLAP: the run's units out of step with its plan";
            return false;
        }
        const LsUnit& x = lsu[u];
        bool rec[Chunk::kLsPieces + 3] = {};
        if (u > 0) {
            const LsUnit& p = lsu[u - 1];
            CK(cudaEventRecord(c.ls_comp[p.set], s));
            if (c.ls_mode == 2 && x.load) {   // (the earlier saves: before ls_saved takes this one)
                CK(cudaStreamWaitEvent(c.ls_in, c.ls_comp[p.set], 0));
                CK(cudaStreamWaitEvent(c.ls_in, c.ls_saved, 0));
            }
            if (p.save) {
                CK(cudaStreamWaitEvent(c.ls_out, c.ls_comp[p.set], 0));
                if (!ls_move(c, p, true, c.ls_out, nullptr, c.ls_mode == 2 ? rec : nullptr, err)) return false;
                CK(cudaEventRecord(c.ls_save[p.set], c.ls_out));
                CK(cudaEventRecord(c.ls_saved, c.ls_out));
            }
        }
        if (c.ls_mode == 2) {
            if (x.load) {
                if (!ls_move(c, x, false, c.ls_in, rec, nullptr, err)) return false;
                CK(cudaStreamWaitEvent(c.ls_in, c.ls_saved, 0));   // (all of the save out before s writes the set)
                CK(cudaEventRecord(c.ls_load[0], c.ls_in));
                CK(cudaStreamWaitEvent(s, c.ls_load[0], 0));
            } else
                CK(cudaStreamWaitEvent(s, c.ls_save[0], 0));
            return true;
        }
        ls_set(c, x.set);
        CK(cudaStreamWaitEvent(s, x.load ? c.ls_load[x.set] : c.ls_save[x.set], 0));
        if (u + 1 < lsu.size() && lsu[u + 1].load) {
            const LsUnit& q = lsu[u + 1];
            if (q.set == x.set) {
                err = "DS4_LM_STATE_OVERLAP: a load planned into the set its unit before runs on";
                return false;
            }
            CK(cudaStreamWaitEvent(c.ls_in, c.ls_comp[q.set], 0));
            CK(cudaStreamWaitEvent(c.ls_in, c.ls_saved, 0));
            if (!ls_move(c, q, false, c.ls_in, nullptr, nullptr, err)) return false;
            CK(cudaEventRecord(c.ls_load[q.set], c.ls_in));
        }
        return true;
    }
    // DS4_LM_STATE_OVERLAP=1 or 2 (a layer-major run's start; unset or 0: off): the states' moves on ls_out / ls_in -
    // 1: ping-pong when VRAM has room for a second state set (the chunk's own buffers' sizes) and 256 MiB more, the
    // set allocated for the run (the experts' placement as it was), else as 2; 2: one set, piece by piece.  States in
    // pageable memory, a remote stage's runs: off.  Its line printed when it differs from the last one
    void ls_begin(Chunk& c) {
        static const char* env = std::getenv("DS4_LM_STATE_OVERLAP");
        static const int want = env != nullptr ? std::atoi(env) : 0;
        static std::string said;
        c.ls_mode = 0;
        if (want != 1 && want != 2) {   // (set to anything else: said, once - a mistyped arm is not silently off)
            static bool told = false;
            if (env != nullptr && !told)
                std::fprintf(stderr, "ds4: DS4_LM_STATE_OVERLAP=%s: off - 1 (ping-pong where VRAM allows) or 2 (one "
                                     "set) turns it on\n", env);
            told = true;
            return;
        }
        Dev& d = devs[0];
        cudaSetDevice(d.id);
        char b[320];
        std::string why;
        if (!c.lm_pinned) why = "off - the states' host memory is not pinned";
        else if (linked()) why = "off - a remote stage's run";
        else if (c.ls_out == nullptr && !ls_streams(c)) why = "off - its copy streams or events were not created";
        if (why.empty()) {
            const size_t C = (size_t) c.C;
            c.ls_h[0] = c.h;
            c.ls_pre[0] = c.pre;
            c.ls_tk[0] = c.topk;
            c.ls_cd[0] = c.cand;
            c.ls_bytes[0] = align256(C * g.hc * g.dim * 4);
            c.ls_bytes[1] = align256(C * 8 * 4);
            c.ls_bytes[2] = align256(C * topk_stride() * 4);
            c.ls_bytes[3] = align256(C * cand_stride() * 4);
            const size_t need = c.ls_bytes[0] + c.ls_bytes[1] + c.ls_bytes[2] + c.ls_bytes[3];
            int engines = 0;
            cudaDeviceGetAttribute(&engines, cudaDevAttrAsyncEngineCount, d.id);
            c.ls_mode = 2;
            std::snprintf(b, sizeof b, "DS4_LM_STATE_OVERLAP=2");
            if (want == 1) {
                size_t fr = 0, tot = 0;
                cudaMemGetInfo(&fr, &tot);
                if (fr < need + ((size_t) 256 << 20))
                    std::snprintf(b, sizeof b, "%zu MiB free, a second set needs %zu MiB and 256 more", fr >> 20, need >> 20);
                else if (!ls_alloc(c, d))
                    std::snprintf(b, sizeof b, "a second set's cudaMalloc failed");
                else
                    c.ls_mode = 1;
            }
            if (c.ls_mode == 1)
                std::snprintf(b, sizeof b, "ping-pong: a second state set of %zu MiB in VRAM for the run", need >> 20);
            else {
                const std::string r = b;
                std::snprintf(b, sizeof b, "one set: piece by piece, both directions at once (%s)", r.c_str());
            }
            why = "the chunks' states moved on two copy streams (" + std::to_string(engines) + " copy engines) - " + b;
        }
        why = "ds4: DS4_LM_STATE_OVERLAP=" + std::to_string(want) + ": " + why;
        if (why != said) {
            std::fprintf(stderr, "%s\n", why.c_str());
            said = why;
        }
    }
    bool ls_streams(Chunk& c) {
        cudaEvent_t* ev[] = {&c.ls_comp[0], &c.ls_comp[1], &c.ls_save[0], &c.ls_save[1], &c.ls_load[0], &c.ls_load[1],
                             &c.ls_saved};
        c.ls_out = c.ls_in = nullptr;
        for (cudaEvent_t* e : ev) *e = nullptr;
        for (cudaEvent_t& e : c.ls_pc) e = nullptr;
        bool ok = cudaStreamCreateWithFlags(&c.ls_out, cudaStreamNonBlocking) == cudaSuccess &&
                  cudaStreamCreateWithFlags(&c.ls_in, cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* e : ev) ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
        for (cudaEvent_t& e : c.ls_pc) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        if (!ok) {   // (what was made undone: the next run tries again)
            cudaGetLastError();
            for (cudaEvent_t* e : ev)
                if (*e != nullptr) cudaEventDestroy(*e), *e = nullptr;
            for (cudaEvent_t& e : c.ls_pc)
                if (e != nullptr) cudaEventDestroy(e), e = nullptr;
            if (c.ls_in != nullptr) cudaStreamDestroy(c.ls_in);
            if (c.ls_out != nullptr) cudaStreamDestroy(c.ls_out);
            c.ls_out = c.ls_in = nullptr;
        }
        return ok;
    }
    // mode 1's second set: the chunk's own buffers copied in whole (what no unit has written yet the same bits as
    // there), on d.st before the run's first unit
    bool ls_alloc(Chunk& c, Dev& d) {
        void* p[4] = {nullptr, nullptr, nullptr, nullptr};
        const void* own[4] = {c.h, c.pre, c.topk, c.cand};
        bool ok = true;
        for (int i = 0; i < 4 && ok; ++i) ok = cudaMalloc(&p[i], c.ls_bytes[i]) == cudaSuccess;
        for (int i = 0; i < 4 && ok; ++i)
            ok = cudaMemcpyAsync(p[i], own[i], c.ls_bytes[i], cudaMemcpyDeviceToDevice, d.st) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            cudaStreamSynchronize(d.st);
            for (void* q : p)
                if (q != nullptr) cudaFree(q);
            return false;
        }
        c.ls_h[1] = (float*) p[0];
        c.ls_pre[1] = (float*) p[1];
        c.ls_tk[1] = (int*) p[2];
        c.ls_cd[1] = (int*) p[3];
        return true;
    }
    // (lm_end, d.st done) the copy streams done, the chunk's own buffers back, mode 1's second set freed
    void ls_end(Chunk& c) {
        cudaStreamSynchronize(c.ls_out);
        cudaStreamSynchronize(c.ls_in);
        ls_set(c, 0);
        if (c.ls_mode == 1) {
            cudaFree(c.ls_h[1]);
            cudaFree(c.ls_pre[1]);
            cudaFree(c.ls_tk[1]);
            cudaFree(c.ls_cd[1]);
            c.ls_h[1] = c.ls_pre[1] = nullptr;
            c.ls_tk[1] = c.ls_cd[1] = nullptr;
        }
        c.ls_mode = 0;
    }

    // layer-major (c.lm: a run of several chunks, device di every trunk layer, the flow set up by run_dev_chunks):
    // layer after layer, each through all the chunks in order - a chunk's state to host memory after each of its
    // layers and back before the next - so a layer's experts are read from the files once for the whole run (the
    // layer cache keeps what a chunk read for the chunks after it), where chunk after chunk reads them once a chunk
    bool run_dev_layers(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        Streamer S;   // (none: the flow streams)
        const size_t hb = chunk_hand_bytes();
        // DS4_LM_STATE_OVERLAP (c.ls_mode): the run's units planned (ls_plan), each one's moves made by ls_enter
        std::vector<LsUnit> lsu;
        size_t ls_u = 0;
        if (c.ls_mode != 0) {
            ls_plan(c, run, lsu);
            if (o.verbose || prof_chunk) {
                const double rb = (double) g.hc * g.dim * 4 + 32, tb = (double) (topk_stride() + cand_stride()) * 4;
                int nl = 0, ns = 0, nt = 0, nst = 0;
                double mv = 0, mv0 = 0;
                for (size_t u = 0; u < lsu.size(); ++u) {
                    const LsUnit& x = lsu[u];
                    nl += x.load, ns += x.save, nt += x.tc, nst += x.stay;
                    mv += x.T * ((x.load ? rb + tb : 0) + (x.save ? rb + (x.tc ? tb : 0) : 0));
                    const bool s0 = x.save || (u + 1 < lsu.size() && lsu[u + 1].stay);
                    mv0 += x.T * ((x.load || x.stay ? rb + tb : 0) + (s0 ? rb + tb : 0));
                }
                std::fprintf(stderr,
                             "ds4 lm state: mode %d, %zu units: %d loads, %d saves (%d with topk / cand), %d states kept on "
                             "the device for the next unit; %.1f GB moved (off: %.1f GB)\n",
                             c.ls_mode, lsu.size(), nl, ns, nt, nst, mv / 1e9, mv0 / 1e9);
            }
        }
        struct LsBack {   // (every way out: the chunk's own buffers back)
            Chunk& c;
            ~LsBack() {
                if (c.ls_mode != 0) Impl::ls_set(c, 0);
            }
        } ls_back{c};
        // the Engram rows: (layer, chunk) after (layer, chunk) in the run's order, each read on a helper thread while
        // the one before runs (slot u % 2, once unit u - 2's copy of it is done)
        std::vector<std::pair<int, int>> egu;   // (Engram layer, chunk)
        for (int il : c.fl_layers)
            if (L[(size_t) il].eg >= 0)
                for (int k = 0; k < nk; ++k)
                    if (run.runs(k, il)) egu.push_back({L[(size_t) il].eg, k});
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egu, eg_per](size_t u) {
            if (u >= 2) cudaEventSynchronize(c.eg_ev[u % 2]);
            const int li = egu[u].first, k = egu[u].second;
            engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[u % 2] + (size_t) li * c.C * eg_per);
        };
        size_t eg_u = 0;
        std::future<void> eg_next;
        if (!egu.empty()) eg_next = std::async(std::launch::async, read_eg, (size_t) 0);
        for (size_t li = 0; li < c.fl_layers.size(); ++li) {
            const int il = c.fl_layers[li];
            Layer& L0 = L[(size_t) il];
            const bool last_l = li + 1 == c.fl_layers.size();
            for (int k = 0; k < nk; ++k) {
                if (!run.runs(k, il)) continue;   // (a skipped early chunk: nothing after the yoco layer)
                if (halt_now()) return halt_end(di, err);   // (a Stop: one GPU, this one)
                const int64_t p0 = run.p0[(size_t) k];
                const int T = run.T[(size_t) k], slot = k % 2;
                const bool last_k = k + 1 == nk;
                WinPlan wp(false);   // (a prompt's chunk: whole)
                wp.par = true;
                wp.async = true;
                wp.win = run.seq[(size_t) k];
                wp.k = k;
                wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
                if (L0.eg >= 0) {   // this unit's rows ready, the next one's on their way
                    const double te = now_ms();
                    eg_next.get();
                    st.engram_ms += now_ms() - te;
                    wp.eg_slot = (int) (eg_u % 2);
                    if (++eg_u < egu.size()) eg_next = std::async(std::launch::async, read_eg, eg_u);
                }
                if (c.ls_mode != 0 && !ls_enter(c, s, lsu, ls_u++, k, err)) return false;
                if (li == 0 && o.stage_worker) {   // (a remote stage's worker: the main process's rows)
                    if (!link_chunk_in(c, T, s, err)) return false;
                } else if (li == 0) {   // the chunk's input: its embedding rows
                    if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                    for (int t = 0; t < T; ++t)
                        std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                    (size_t) D * 2);
                    CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    CK(cudaEventRecord(c.emb_ev[slot], s));
                    embed(c.emb, T, c.h, D, H, s);
                    init_pre(c.pre, T, s);
                } else if (c.ls_mode == 0) {   // or its state after the layer before (DS4_LM_STATE_OVERLAP: ls_enter's)
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, false);
                    for (int i = 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, s));
                }
                if (last_k && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                const bool kvo = run.kv_only(k, il);
                if (!run_layer(L0, 0, p0, T, wp, S, err, run, k, il)) return false;
                if (dump_l != nullptr) dump_layer(d, c.h, il, p0, T);
                if (kvo) continue;   // (no layer after it reads this chunk's state)
                if (!last_l && c.ls_mode != 0) continue;   // (DS4_LM_STATE_OVERLAP: the next unit's ls_enter saves it)
                if (!last_l) {   // the chunk's state kept for the next layer
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, true);
                    for (int i = 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, s));
                    continue;
                }
                // through every layer: the chunk's end, and the logits
                if ((size_t) k < chunk_done.size()) {
                    CK(cudaEventRecord(chunk_done[(size_t) k], s));
                    chunk_rec[(size_t) k].store(1, std::memory_order_release);
                }
                if (linked()) {   // (a remote stage: the chunk's rows over the link - the head after the worker's)
                    if (!link_chunk_out(d, c, run, k, err)) return false;
                } else if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
            lm_layers.store((int) li + 1, std::memory_order_release);
        }
        return true;
    }
    bool run_dev_chunks(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        Streamer S;
        const int B = std::max(1, c.ring_n / Chunk::kParts);
        if (c.flow) {   // (the flow decides its streams as the layers route; nothing planned here)
            CK(cudaStreamSynchronize(s));
            CK(cudaStreamSynchronize(c.cs));
            c.items.clear();
            c.base = c.n_read = c.n_dma = c.n_comp = 0;
            c.n_file = 0;
            std::fill(c.stage_item.begin(), c.stage_item.end(), -1);
            c.fl_layers.clear();
            for (int il = 0; il < g.nl; ++il)   // (a group's layers: on each member)
                for (int m = 0; m < L[(size_t) il].nm; ++m)
                    if (member(L[(size_t) il], m).dev == di) c.fl_layers.push_back(il);
            c.fl_T = run.T;
            c.fl_yoco = run.yoco;
            c.fl_cut_k = run.cut_k;
            c.fl_cut_r = run.cut_r;
            c.spec_k = c.spec_il = -1;
            c.wait_ms = 0;
            c.it_file = c.it_ram = c.it_spec = c.it_spec_unused = c.it_lc = 0;
            c.lc_il = -1;
            c.lc_used = 0;
            c.lc_first = c.lc_chk = 0;
            c.io_d0 = c.io->direct_bytes();
            c.io_b0 = c.io->buffered_bytes();
            c.io_busy0 = c.io->busy_ms();
            c.io_s0 = c.io->split_stats();
            if (!c.fl_layers.empty()) {
                flow_spec(c, 0, c.fl_layers[0], run.T[0]);
                if (!flow_pump(c, err)) return false;
            }
            if (c.lm) return run_dev_layers(di, run, err);
        }
        for (int k = 0; k < nk && !c.flow; ++k)
            for (int il = 0; il < g.nl; ++il) {
                if (!run.runs(k, il) || run.kv_only(k, il)) continue;   // (a long prompt's skip: no experts there)
                const Layer& L0 = L[(size_t) il];
                bool here = false;
                for (int m = 0; m < L0.nm; ++m) here = here || member(L0, m).dev == di;
                if (!here) continue;
                for (int j0 = 0; j0 < (int) L0.host_list.size(); j0 += B)
                    S.b.push_back({k, il, j0, std::min(B, (int) L0.host_list.size() - j0)});
            }
        if (!S.b.empty() && c.ring == nullptr) {
            err = "no slots to stream experts into on CUDA" + std::to_string(d.id);
            return false;
        }
        std::atomic<bool> ra_stop{false};
        std::thread ra;
        if (file_experts && !S.b.empty()) ra = std::thread([this, &S, di, &ra_stop] { read_ahead(S, di, ra_stop); });
        struct Join {   // (every way out of the run: the read-ahead ends before S does)
            std::atomic<bool>& stop;
            std::thread& th;
            ~Join() {
                stop.store(true);
                if (th.joinable()) th.join();
            }
        } ra_join{ra_stop, ra};
        if (!copy_ahead(d, c, S, di, err)) return false;
        // this device's Engram layers: chunk k's rows read on a helper thread while chunk k - 1 runs (slot k % 2, once
        // chunk k - 2's copy of it is done)
        std::vector<int> egs;
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di && !l0.mtp && l0.eg >= 0) egs.push_back(l0.eg);
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egs, eg_per](int k) {
            if (k >= 2) cudaEventSynchronize(c.eg_ev[k % 2]);
            for (int li : egs)
                engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[k % 2] + (size_t) li * c.C * eg_per);
        };
        std::future<void> eg_next;
        if (!egs.empty()) eg_next = std::async(std::launch::async, read_eg, 0);
        for (int k = 0; k < nk; ++k) {
            if (!chunk_go(k)) return halt_end(di, err);   // (a Stop: the chunks every device reaches are done)
            const int64_t p0 = run.p0[(size_t) k];
            const int T = run.T[(size_t) k], slot = k % 2;
            const bool last_k = k + 1 == nk;
            if (!egs.empty()) {   // chunk k's rows ready, chunk k + 1's on their way
                const double te = now_ms();
                eg_next.get();
                if (d.mi == 0) st.engram_ms += now_ms() - te;
                if (k + 1 < nk) eg_next = std::async(std::launch::async, read_eg, k + 1);
            }
            WinPlan wp(false);   // (a prompt's chunk: whole)
            wp.par = true;
            wp.async = true;
            wp.win = run.seq[(size_t) k];
            wp.k = k;
            wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
            bool started = false;
            std::vector<char> got(srcs.size(), 0);
            for (int il = 0; il < g.nl && run.runs(k, il); ++il) {
                if (halt_now()) return halt_end(di, err);   // (one GPU: after the layer it is on)
                Layer& L0 = L[(size_t) il];
                int mi = -1;
                for (int m = 0; m < L0.nm; ++m)
                    if (member(L0, m).dev == di) mi = m;
                if (mi < 0) continue;
                if (!started) {   // the chunk's input: its embedding rows, or the previous group's hand-off
                    started = true;
                    if (il == lo && o.stage_worker) {   // (a remote stage's worker: the main process's rows)
                        if (!link_chunk_in(c, T, s, err)) return false;
                    } else if (il == lo) {
                        if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                        for (int t = 0; t < T; ++t)
                            std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                        (size_t) D * 2);
                        CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                        CK(cudaEventRecord(c.emb_ev[slot], s));
                        embed(c.emb, T, c.h, D, H, s);
                        init_pre(c.pre, T, s);
                    } else {
                        Chunk& a = *chunks[(size_t) L[(size_t) il - 1].dev];
                        xfer_get(chunk_xfer(a.hand[slot], c, T, false), a.hand_flag[slot], wp.win, s);
                        set_flag(a.hand_ack + ((size_t) slot * 8 + mi) * 32, wp.win, s);
                    }
                }
                for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                    if (sidx < 0 || got[(size_t) sidx]) continue;
                    got[(size_t) sidx] = 1;
                    Src& sr = srcs[(size_t) sidx];
                    if (L[(size_t) sr.layer].grp == L0.grp) continue;
                    const int64_t g_first = p0 / sr.ratio;
                    const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                    Xfer x;
                    if (ng > 0 && sr.ckv[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot);
                        x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * ckv_rb();
                        x.bytes[x.n++] = (size_t) ng * ckv_rb();
                    }
                    if (ng > 0 && sr.ik[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot) + x_cap() * ckv_rb();
                        x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_rb();
                        x.bytes[x.n++] = (size_t) ng * ik_rb();
                    }
                    if (x.n > 0) xfer_get(x, sr.flag(slot), wp.win, s);
                    set_flag(sr.x_ack + ((size_t) slot * devs.size() + di) * 32, wp.win, s);   // (taken, or none)
                }
                if (last_k && mi == 0 && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                if (!run_layer(L0, mi, p0, T, wp, S, err, run, k, il)) return false;
                if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, il, p0, T);
                const bool last = il + 1 == hi || L[(size_t) il + 1].grp != L0.grp;
                if (last && mi == 0 && il + 1 < hi) {   // the hand-off, once the next group took the chunk before last
                    const int nnext = L[(size_t) il + 1].nm;
                    if (k >= 2) wait_flags(c.hand_ack + (size_t) slot * 8 * 32, nnext, 32, wp.win - 2, s);
                    xfer_put(chunk_xfer(c.hand[slot], c, T, true), c.hand_flag[slot], wp.win, d.xcnt, s);
                }
            }
            if (di == head_dev && (size_t) k < chunk_done.size()) {
                CK(cudaEventRecord(chunk_done[(size_t) k], s));
                chunk_rec[(size_t) k].store(1, std::memory_order_release);
            }
            if (linked()) {   // (a remote stage: the chunk's rows over the link from the last layer's GPU - load
                              // checked it is alone there - and the head after the worker's)
                if (di == head_dev && !link_chunk_out(d, c, run, k, err)) return false;
            } else if (d.grp == devs[(size_t) head_dev].grp) {   // the head group: each member its share of the logits
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        if (di != head_dev) continue;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
        }
        return true;
    }

    // ---- a prompt's path: the flow or windows of 16
    // this PC's speeds for pick_path (every device on the flow, the experts in place): the DMA from pinned staging to a
    // GPU; the readers copying RAM into staging; the disk read by them (experts not in RAM); the CPU lane on experts in
    // RAM and on ones it reads from the disk first - each timed once, a few experts' worth (DS4_PATH_PICK=0: always the
    // flow, as before)
    void calibrate_paths() {
        if (chunk <= 0 || chunk_min != kFlowChunkMin || std::getenv("DS4_CHUNK_MIN") != nullptr) return;
        if (const char* v = std::getenv("DS4_PATH_PICK"))
            if (std::atoi(v) == 0) return;
        int di = -1;
        for (size_t i = 0; i < chunks.size() && di < 0; ++i)
            if (chunks[i] != nullptr && chunks[i]->flow && chunks[i]->S >= 2) di = (int) i;
        if (di < 0) return;
        Chunk& c = *chunks[(size_t) di];
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const double t0 = now_ms();
        // the DMA: staging -> the ring (once to warm up, once timed)
        const size_t nb = std::min({(size_t) 256 << 20, (size_t) c.S * c.stage_stride, (size_t) c.ring_n * c.ring_stride});
        double ms = 0;
        for (int rep = 0; rep < 2; ++rep) {
            const double a = now_ms();
            if (cudaMemcpyAsync(c.ring, c.stage, nb, cudaMemcpyHostToDevice, c.cs) != cudaSuccess ||
                cudaStreamSynchronize(c.cs) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            ms = now_ms() - a;
        }
        pm.dma = (double) nb / std::max(1e-3, ms / 1000.0);
        // the readers: RAM -> staging (a buffer of its own, touched first), in as many jobs as they are
        auto run_jobs = [&](std::vector<IoQueue::Job>& jobs, std::atomic<int>& left) {
            const double a = now_ms();
            left.store((int) jobs.size());
            c.io->push(jobs.data(), (int) jobs.size());
            while (left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            return now_ms() - a;
        };
        {
            const size_t part = std::min(nb, (size_t) 64 << 20) / 8 / 4096 * 4096;
            std::vector<uint8_t> src(part * 8, 1);
            std::vector<IoQueue::Job> jobs(8);
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < 8; ++i) {
                jobs[(size_t) i].mem = src.data() + (size_t) i * part;
                jobs[(size_t) i].dst = c.stage + (size_t) i * part;
                jobs[(size_t) i].len = part;
                jobs[(size_t) i].left = &left;
                jobs[(size_t) i].bad = &bad;
            }
            run_jobs(jobs, left);
            const double t1 = run_jobs(jobs, left), t2 = run_jobs(jobs, left);
            pm.copy = (double) (part * 8) / std::max(1e-3, std::min(t1, t2) / 1000.0);
        }
        // experts not in RAM (none of their sampled pages resident), of the trunk layer with the most CPU-lane ones:
        // up to min(8, staging slots) read by the readers as the flow reads them, then 8 others through the CPU lane
        // three times (from the disk, then in RAM: the faster of two)
        int best = -1, most = 0;
        for (int il = 0; il < g.nl; ++il) {
            int n = 0;
            for (int e = 0; e < L[(size_t) il].n_exp; ++e) n += L[(size_t) il].slot[(size_t) e] < 0;
            if (n > most) most = n, best = il;
        }
        if (best < 0) return;
        const Layer& l = L[(size_t) best];
        std::vector<int> cold;
        const int n_read = std::min(8, c.S), n_cpu = 8;
        for (int e = 0; e < l.n_exp && (int) cold.size() < n_read + n_cpu; ++e) {
            if (l.slot[(size_t) e] >= 0 || (!l.ram.empty() && l.ram[(size_t) e] >= 0)) continue;
            bool res = false;
            for (int r = 0; r < 3; ++r) res = res || os::resident_sampled(mx_mat(l, e, r), mx_part_bytes(l, r), 4);
            if (!res) cold.push_back(e);
        }
        if ((int) cold.size() >= n_read + n_cpu) {
            std::vector<IoQueue::Job> jobs;
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < n_read; ++i) {
                uint8_t* sb = c.stage + (size_t) i * c.stage_stride;
                for (int r = 0; r < 3; ++r) {
                    IoQueue::Job j;
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) cold[(size_t) i] * l.mx_stride[r];
                    j.len = mx_part_bytes(l, r);
                    j.dst = sb;
                    j.left = &left;
                    j.bad = &bad;
                    j.map = pk.at(l.mx_file[r], j.off);
                    jobs.push_back(j);
                    sb += flow_region(r);
                }
            }
            const double rd = run_jobs(jobs, left);
            if (!bad.load()) pm.disk = (double) n_read * (double) l.blob / std::max(1e-3, rd / 1000.0);
        }
        {   // the CPU lane: entries of one token (as most of a window's are), from the disk (if any were cold), then RAM
            const size_t first = (int) cold.size() >= n_read + n_cpu ? (size_t) n_read : 0;
            std::vector<Ent> ents;
            for (int i = 0; i < n_cpu; ++i) {
                Ent en;
                en.e = first + (size_t) i < cold.size() ? cold[first + (size_t) i] : -1;
                if (en.e < 0)   // (no cold ones: the first CPU-lane experts, wherever they are)
                    for (int e = 0, k = 0; e < l.n_exp && en.e < 0; ++e)
                        if (l.slot[(size_t) e] < 0 && k++ == i) en.e = e;
                if (en.e < 0) break;
                en.n = 1;
                en.slot0 = i;
                en.tok[0] = 0;
                ents.push_back(en);
            }
            if (ents.empty()) return;
            std::vector<float> x((size_t) g.dim);
            for (size_t i = 0; i < x.size(); ++i) x[i] = 0.02f * std::sin(0.37f * (float) i);
            const double bytes = (double) ents.size() * (double) l.blob;
            double a = now_ms();
            cpu_experts(l, x.data(), ents, (size_t) g.dim, nullptr, 0, g.ff);   // (whole: the main's speed)
            const double t_first = now_ms() - a;
            double t_warm = 1e30;
            for (int rep = 0; rep < 2; ++rep) {
                a = now_ms();
                cpu_experts(l, x.data(), ents, (size_t) g.dim, nullptr, 0, g.ff);
                t_warm = std::min(t_warm, now_ms() - a);
            }
            pm.cpu = bytes / std::max(1e-3, t_warm / 1000.0);
            if (first > 0 || (int) cold.size() >= n_cpu) pm.cpu_cold = bytes / std::max(1e-3, t_first / 1000.0);
        }
        pm.on = pm.dma > 0 && pm.copy > 0 && pm.cpu > 0;
        if (o.verbose) {
            char dk[32] = "not measured", cc[32] = "not measured";
            if (pm.disk > 0) std::snprintf(dk, sizeof dk, "%.2f GB/s", pm.disk / 1e9);
            if (pm.cpu_cold > 0) std::snprintf(cc, sizeof cc, "%.2f GB/s", pm.cpu_cold / 1e9);
            std::fprintf(stderr,
                         "ds4: prompts through the flow or windows by this PC's speeds: DMA %.1f GB/s, RAM to staging "
                         "%.1f GB/s, the disk %s; the CPU lane %.1f GB/s in RAM, %s from the disk (%.0f ms)\n",
                         pm.dma / 1e9, pm.copy / 1e9, dk, pm.cpu / 1e9, cc, now_ms() - t0);
        }
    }
    // layer l's CPU-lane experts (VRAM holds the rest) and the chance a token picks a given one of them: the profile's
    // share of the layer's picks they took, spread evenly over them, times pm.m (what the flow's runs found)
    double pm_rate(const Layer& l, int& n_cold) const {
        const uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        double sum = 0, cold = 0;
        n_cold = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            sum += pr[e];
            if (l.slot[(size_t) e] < 0) cold += pr[e], ++n_cold;
        }
        if (n_cold == 0) return 0.0;
        const double share = sum > 0 && cold > 0 ? cold / sum : (double) n_cold / l.n_exp;
        return pm.m * (double) l.k_exp * share / n_cold;
    }
    // the CPU-lane experts n tokens pick in the trunk's layers, expected (each layer's n_cold (1 - e^(-rate n)))
    double pm_picks(int n) const {
        double d = 0;
        for (int il = 0; il < g.nl; ++il) {
            int nc = 0;
            const double a = pm_rate(L[(size_t) il], nc);
            d += nc * (1.0 - std::exp(-a * n));
        }
        return d;
    }
    // a run of one chunk of n tokens on the flow picked `picked` CPU-lane experts: pm.m to what makes the expectation
    // that (bisection over its log; a running mean of the log)
    void pm_learn(int n, double picked) {
        if (!pm.on || n < 2 || picked < 1) return;
        const double m0 = pm.m;
        double lo = std::log(1e-3), hi = std::log(1e3);
        for (int it = 0; it < 40; ++it) {
            pm.m = std::exp(0.5 * (lo + hi));
            (pm_picks(n) < picked ? lo : hi) = 0.5 * (lo + hi);
        }
        const double obs = 0.5 * (lo + hi);
        pm.m = std::exp(pm.n_m == 0 ? obs : 0.7 * std::log(m0) + 0.3 * obs);
        ++pm.n_m;
    }
    // a prompt of n tokens that the flow could take (n >= chunk_min): true the flow, false windows of 16 - each path's
    // time predicted from the bytes of CPU-lane experts it would move (the flow: each picked one once, n_cold (1 -
    // e^(-rate n)) a layer, pm_rate) or compute (windows: each window's own picks), those in RAM (each layer's share:
    // sampled pages of up to 32 of its CPU-lane experts, mincore / QueryWorkingSetEx; the RAM tier's) and those not at
    // this PC's speeds - the flow's times what it really took against its predictions so far for prompts of that size
    // (pm_k), windows' once some ran what a window took times the windows; the flow unless windows are predicted
    // faster (by half, with less than kPickRes in RAM); said with --verbose
    bool pick_path(int n) {
        pm.last = -1;
        if (!pm.on || n >= kPickMax) return true;
        double f_res = 0, f_cold = 0, w = 0;
        const int nw = n / kWin, rem = n % kWin;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            int nc = 0;
            const double a = pm_rate(l, nc);
            if (nc == 0) continue;
            int sampled = 0, res = 0, seen = 0;
            const int step = std::max(1, nc / 32);
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0 || seen++ % step != 0) continue;
                bool r = !l.ram.empty() && l.ram[(size_t) e] >= 0;
                if (!r) {
                    r = true;
                    for (int p = 0; p < 3 && r; ++p) r = os::resident_sampled(mx_mat(l, e, p), mx_part_bytes(l, p), 2);
                }
                ++sampled;
                res += r;
            }
            const double share = sampled > 0 ? (double) res / sampled : 0.0, b = (double) l.blob;
            auto u = [&](int t) { return nc * (1.0 - std::exp(-a * t)); };
            f_res += share * u(n) * b;
            f_cold += (1.0 - share) * u(n) * b;
            w += ((double) nw * u(kWin) + (rem > 0 ? u(rem) : 0.0)) * b;
        }
        const double r_fr = std::min(pm.copy, pm.dma), r_fc = pm.disk > 0 ? std::min(pm.disk, pm.dma) : r_fr;
        const double r_wc = pm.cpu_cold > 0 ? pm.cpu_cold : 1.0 / (1.0 / r_fc + 1.0 / pm.cpu);
        const double t_flow = f_res / r_fr + f_cold / r_fc;
        const double t_win = std::max(0.0, w - f_cold) / pm.cpu + f_cold / r_wc;
        const int cls = pm_cls(n), n_win = (n + kWin - 1) / kWin;
        // (windows: once some ran, what a window took on average times the windows the prompt needs - each pays the
        // fixed costs of every layer on every GPU again, which a pass of the flow pays once)
        const double pf = pm_k(cls) * t_flow, pw = pm.win_n > 0 ? pm.win_s * n_win : t_win;
        // (the flow unless windows are predicted faster - in half its time at least, unless kPickRes of the CPU-lane
        // experts' bytes are in RAM: the flow wins on every PC whose experts come from an SSD, where a wrong pick would
        // cost the most; windows win where they are all in RAM)
        const double res_share = f_res / std::max(1.0, f_res + f_cold);
        const bool flow = !(pw < pf && (res_share >= kPickRes || 2.0 * pw <= pf));
        pm.last = flow ? 1 : 0;
        pm.last_cls = cls;
        pm.last_nw = n_win;
        pm.last_pred = flow ? t_flow : t_win;
        if (o.verbose)
            std::fprintf(stderr,
                         "ds4: a prompt of %d tokens through %s: predicted %.2f s, %s %.2f s (%.0f%% of its CPU-lane "
                         "experts' %.1f GB in RAM)\n",
                         n, flow ? "the flow" : "windows", flow ? pf : pw, flow ? "windows" : "the flow", flow ? pw : pf,
                         100.0 * res_share, (f_res + f_cold) / 1e9);
        return flow;
    }
    // a prompt's size class for pm.k: 2-31, 32-127, 128-511, 512 tokens and more
    static int pm_cls(int n) { return n < 32 ? 0 : n < 128 ? 1 : n < 512 ? 2 : 3; }
    // the flow's time factor for size class c: its own, else its nearest class's, else 1 (never the windows': theirs
    // is mostly the fixed costs a window pays again and again)
    double pm_k(int c) const {
        for (int dc = 0; dc < 4; ++dc)
            for (int cc : {c - dc, c + dc})
                if (cc >= 0 && cc < 4 && pm.kn[0][cc] > 0) return pm.k[0][cc];
        return 1.0;
    }
    // n tokens through windows of 16, the last one's logits_rows rows of logits
    bool feed_windows(const int* tokens, int n, int logits_rows, std::string& err) {
        logits_all = false;
        {
            struct Off {   // (win_prompt false again on every way out)
                bool& f;
                ~Off() { f = false; }
            } off{win_prompt};
            win_prompt = true;   // (its ticks as the chunked prompt's: for the decode that follows)
            for (int a = 0; a < n; a += kWin) {
                const int T = std::min(kWin, n - a);
                if (!forward(tokens + a, T, a + T == n ? logits_rows : 0, false, err)) return false;   // (a prompt's: whole)
            }
        }
        // with the expert server: the settle, as after a chunked prompt - the cache's swaps for the decode that follows -,
        // once a prompt (rx::PromptSettle: the feed that asks for logits, unless the prompt's chunked part settled
        // already).  Before U5a nothing settled after a window prompt, and its ticks (serve_T = T: past async_max_t
        // the lane's) put the server's experts into this GPU's VRAM (slot 20261025T1243Z's ep2 arm: no settle line, 9
        // server experts in VRAM); win_prompt keeps them the server's now, as the decode's (rx::lane_serves(0, ..):
        // unless DS4_SYNC).  Without the server as before
        if (psettle.after_feed(rx_on, logits_rows) && (!o.stage_worker || stage_last)) cache_settle();
        return true;
    }
    // DS4_SMALL_PROMPT: read once, at the load - N (0 off; past kPickMax held to it: from there on pick_path always
    // takes the flow, and a long run of one-row windows would be a long prompt without the chunked path's progress),
    // and DS4_SMALL_WIN's rows held to 1..min(kWin, DS4_ASYNC_T) (past DS4_ASYNC_T a window is synchronous, which is what
    // the small path is not).  A text that is not a number, or out of range, is off (or 1) with a line saying why
    void small_setup() {
        std::string note;
        small_n = small_prompt_n(std::getenv("DS4_SMALL_PROMPT"), kPickMax, note);
        if (!note.empty()) std::fprintf(stderr, "ds4: %s\n", note.c_str());
        if (small_n < 2) {
            small_n = 0;
            return;
        }
        if (o.remote_stage || o.stage_worker) {   // (the link's prompts keep their chunks and windows of 16)
            std::fprintf(stderr, "ds4: DS4_SMALL_PROMPT=%d: off for this run - a remote stage\n", small_n);
            small_n = 0;
            return;
        }
        const int asked = small_win_ask(std::getenv("DS4_SMALL_WIN"), note);
        if (!note.empty()) std::fprintf(stderr, "ds4: %s\n", note.c_str());
        small_t = small_win(asked, kWin, async_max_t);
        if (small_t != asked)
            std::fprintf(stderr, "ds4: DS4_SMALL_WIN=%d: clamped to %d (1 to the smaller of %d and DS4_ASYNC_T %d)\n",
                         asked, small_t, kWin, async_max_t);
        std::fprintf(stderr, "ds4: DS4_SMALL_PROMPT=%d: prompts of 2-%d tokens in windows of %d through the decode's path%s\n",
                     small_n, small_n, small_t, small_t == 1 && ep_defer && !ep_verify
                                                    ? ", with the DS4_EP_DEFER deferral (the decode's approximation)" : "");
    }
    // DS4_SMALL_PROMPT: n tokens through windows of small_t rows as the decode runs them (ep: the n_ff slices, the
    // expert server's async windows, at 1 row DS4_EP_DEFER's deferral), the last one's logits_rows rows of logits; the
    // rest as feed_windows (win_prompt the whole prompt, the settle at its end).  A Stop (set_cancel, asked between
    // windows as forward_chunks asks it while its run goes on) ends it there as a chunked prompt's does: false, halted,
    // the state at the window it reached (the caller restores a checkpoint or resets).  Not before the first window: the
    // serve loop asks before each feed, and feed_mark's last token comes after its checkpoint(slot) - a Stop there
    // would leave the caller no checkpoint at the prompt's start (restore fails, the conversation is reset), where a
    // chunked prompt never stops after its snapshot
    bool feed_small(const int* tokens, int n, int logits_rows, std::string& err) {
        logits_all = false;
        {
            struct Off {   // (win_prompt, small_feed false again on every way out)
                bool& w;
                bool& s;
                ~Off() { w = false; s = false; }
            } off{win_prompt, small_feed};
            win_prompt = true;   // (its ticks the decode's; forward leaves psettle to the end here)
            small_feed = true;   // (a prompt's windows: not the decode's statistics)
            for (int a = 0; a < n; a += small_t) {
                if (a > 0 && cancel && cancel()) {
                    halted = true;
                    err = "stopped while reading the prompt";
                    return false;
                }
                const int T = std::min(small_t, n - a);
                if (!forward(tokens + a, T, a + T == n ? logits_rows : 0, true, err)) return false;
            }
        }
        if (psettle.after_feed(rx_on, logits_rows) && (!o.stage_worker || stage_last)) cache_settle();   // (feed_windows')
        return true;
    }
    // a small prompt of n tokens took ms: its own running mean, ms a token (pick_path's pm learns nothing from it)
    void small_done(int n, double ms) {
        const double a = ms / std::max(1, n);
        small_ms = small_k == 0 ? a : 0.7 * small_ms + 0.3 * a;
        ++small_k;
        if (o.verbose)   // (ms a token: the running mean)
            std::fprintf(stderr, "ds4: a prompt of %d tokens in windows of %d (DS4_SMALL_PROMPT=%d): %.0f ms (%.1f ms a token)\n",
                         n, small_t, small_n, ms, small_ms);
    }
    // the picked path took ms: the flow's predictions for the prompt's size class scaled by what it really takes (taken
    // / predicted), windows' time a window - running means
    void path_done(double ms) {
        if (pm.last < 0 || pm.last_pred <= 0) return;
        if (pm.last == 0) {
            const double w = ms / 1000.0 / std::max(1, pm.last_nw);
            pm.win_s = pm.win_n == 0 ? w : 0.7 * pm.win_s + 0.3 * w;
            ++pm.win_n;
        } else {
            const double ratio = std::min(8.0, std::max(0.25, ms / 1000.0 / pm.last_pred));
            double& k = pm.k[0][pm.last_cls];
            int64_t& n = pm.kn[0][pm.last_cls];
            k = n == 0 ? ratio : 0.7 * k + 0.3 * ratio;
            ++n;
        }
        pm.last = -1;
    }
    // a run layer-major? - one device holding every layer on the flow, several chunks, and what the chunks after the
    // first would read from the files again more than the chunks' states moved to host memory and back at every layer
    // (DS4_LAYER_MAJOR=0: never); then its host memory: the chunks' states (pinned, else pageable) and the layer cache,
    // the largest layer's experts read from the files if RAM allows (ram_avail() less kRamMargin; DS4_LAYER_CACHE_MB
    // sets it) - at least one expert, or not layer-major
    // DS4_LM_PRE=1, at the load (before the store's arena and the warm read): the layer-major prompt's host memory once
    // for the process - the layer cache for a layer whose every expert but the RAM tier's is the CPU lane's (lm_begin's
    // need over any placement the expert cache moves to; DS4_LAYER_CACHE_MB caps it), the chunks' states for the longest
    // prompt the context allows (ceil(max_context / chunk) chunks: lm_begin's state_b at its largest), pinned where the
    // driver gives it - each faulted in now (os::populate_write; pinned memory is), so no prompt allocates or faults
    // either: the 6.7 GB lm_begin took at every such prompt's start, where the cgroup may be at memory.high and every
    // fault reclaims (reanalysis-r09 section 3.3).  Not where no run can be layer-major, nor where it does not fit the
    // RAM this process can get beside kRamMargin; then lm_begin allocates at each run, as before.  The warm read's
    // budget is held to what is left
    void lm_prealloc() {
        if (!lm_pre) return;
        static const char* env = std::getenv("DS4_LAYER_MAJOR");
        const int nc = chunk > 0 ? (int) ((o.max_context + chunk - 1) / chunk) : 0;
        const char* why = devs.size() != 1                                            ? "more than one GPU"
                          : chunks.empty() || chunks[0] == nullptr || !chunks[0]->flow ? "the prompt is not on the flow"
                          : nc < 2                                                     ? "a prompt the context allows takes one chunk"
                          : !stage::layer_major_allowed(linked(), env)                 ? "no layer-major prompt here"
                                                                                       : nullptr;
        if (why != nullptr) {
            std::fprintf(stderr, "ds4: DS4_LM_PRE=1: nothing allocated at the load (%s)\n", why);
            return;
        }
        Chunk& c = *chunks[0];
        size_t need = 0;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            size_t ne = 0;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.ram.empty() || l.ram[(size_t) e] < 0) ++ne;
            need = std::max(need, ne * ((l.blob + 63) & ~(size_t) 63));
        }
        size_t lc_b = need;
        if (const char* mb = std::getenv("DS4_LAYER_CACHE_MB")) lc_b = std::min(need, (size_t) std::atoll(mb) << 20);
        const size_t st_b = (size_t) nc * chunk_hand_bytes();
        const uint64_t avail = ram_avail();
        if (lc_b == 0 || (avail != ~uint64_t{0} && (double) lc_b + (double) st_b + (double) kRamMargin > (double) avail)) {
            std::fprintf(stderr, "ds4: DS4_LM_PRE=1: nothing allocated at the load (the layer cache %.2f GB and the chunks' "
                                 "states %.2f GB do not fit the %.1f GB of RAM this process can get beside %.0f GB)\n",
                         (double) lc_b / 1e9, (double) st_b / 1e9, avail == ~uint64_t{0} ? 0.0 : (double) avail / 1e9,
                         (double) kRamMargin / 1e9);
            return;
        }
        const double t = now_ms();
        void* st = nullptr;
        const bool pinned = cudaHostAlloc(&st, st_b, cudaHostAllocPortable) == cudaSuccess;
        const char* st_how = "pinned";
        if (!pinned) {
            cudaGetLastError();
            st = os::anon_alloc(st_b, -1, 1, lc_thp);
            if (st != nullptr) st_how = os::populate_write(st, st_b);
        }
        uint8_t* lc = st != nullptr ? (uint8_t*) os::anon_alloc(lc_b, -1, 1, lc_thp) : nullptr;
        if (lc == nullptr) {
            if (st != nullptr) {
                if (pinned) cudaFreeHost(st);
                else os::anon_free(st, st_b);
            }
            std::fprintf(stderr, "ds4: DS4_LM_PRE=1: nothing allocated at the load (%.2f GB not had)\n",
                         (double) (lc_b + st_b) / 1e9);
            return;
        }
        const char* lc_how = os::populate_write(lc, lc_b);
        c.lc_kept = lc;
        c.lc_cap = lc_b;
        c.st_kept = (uint8_t*) st;
        c.st_cap = st_b;
        c.st_kept_pinned = pinned;
        const uint64_t av = ram_avail();
        if (av != ~uint64_t{0}) warm_bytes = av > kRamMargin ? std::min<uint64_t>(warm_bytes, av - kRamMargin) : 0;
        std::fprintf(stderr, "ds4: DS4_LM_PRE=1: the layer-major prompt's layer cache %.2f GB (%s, %s) and chunk states "
                             "%.2f GB (%d chunks of %d rows, %s) allocated at the load and faulted in, in %.1f s; kept for "
                             "the process\n",
                     (double) lc_b / 1e9, lc_thp ? "huge pages advised" : "4 KiB pages", lc_how, (double) st_b / 1e9, nc,
                     chunk, pinned ? "pinned" : st_how, (now_ms() - t) / 1000.0);
    }
    bool lm_begin(const Run& run) {
        static const char* env = std::getenv("DS4_LAYER_MAJOR");
        // (a remote stage: chunk after chunk, which is what overlaps with the other side - layer-major only with
        // DS4_LAYER_MAJOR=1)
        if (devs.size() != 1 || run.T.size() < 2 || !stage::layer_major_allowed(linked(), env)) return false;
        Chunk& c = *chunks[0];
        if (!c.flow) return false;
        size_t need = 0, n_exp = 0, blob_max = 0;
        double file_b = 0;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            size_t ne = 0;
            for (int e : l.host_list)
                if (l.ram.empty() || l.ram[(size_t) e] < 0) ++ne;
            const size_t stride = (l.blob + 63) & ~(size_t) 63;
            need = std::max(need, ne * stride);
            n_exp = std::max(n_exp, (size_t) l.n_exp);
            blob_max = std::max(blob_max, stride);
            file_b += (double) ne * (double) l.blob;
        }
        int64_t rows = 0;
        for (int T : run.T) rows += T;
        const size_t hb = chunk_hand_bytes(), state_b = run.T.size() * hb;
        const double moved = 2.0 * (double) rows * ((double) hb / (double) chunk) * (double) (hi - lo);   // (its layers)
        if (need == 0 || (double) (run.T.size() - 1) * file_b <= moved) return false;
        // (DS4_LM_PRE: the load's layer cache, at most lc_cap - the RAM is the process's already, no check; its
        // chunks' states where they hold the run's, else this run's own, as without it)
        const bool kept = c.lc_kept != nullptr;
        size_t lc_bytes = need;
        if (kept) lc_bytes = std::min(need, c.lc_cap);
        else if (const char* mb = std::getenv("DS4_LAYER_CACHE_MB")) lc_bytes = std::min(need, (size_t) std::atoll(mb) << 20);
        else {
            const uint64_t avail = ram_avail();
            if (avail != ~uint64_t{0})
                lc_bytes = std::min<uint64_t>(need, avail > kRamMargin + state_b ? avail - kRamMargin - state_b : 0);
        }
        if (lc_bytes < blob_max) return false;
        void* st_p = nullptr;
        if (kept && c.st_cap >= state_b) {
            st_p = c.st_kept;
            c.lm_pinned = c.st_kept_pinned;
        } else {
            c.lm_pinned = cudaHostAlloc(&st_p, state_b, cudaHostAllocPortable) == cudaSuccess;
            if (!c.lm_pinned) {
                cudaGetLastError();
                st_p = os::anon_alloc(state_b, -1, 1, lc_thp);
                if (st_p == nullptr) return false;
            }
        }
        c.lc = kept ? c.lc_kept : (uint8_t*) os::anon_alloc(lc_bytes, -1, 1, lc_thp);
        if (c.lc == nullptr) {
            if (c.lm_pinned) cudaFreeHost(st_p);
            else os::anon_free(st_p, state_b);
            return false;
        }
        c.lm_state = (uint8_t*) st_p;
        c.lm_state_bytes = state_b;
        c.lc_bytes = lc_bytes;
        c.lc_of.assign(n_exp, -1);
        c.lc_writer.assign(n_exp, -1);
        ls_begin(c);
        c.lm = true;
        lm_layers.store(0);
        return true;
    }
    void lm_end() {
        for (size_t di = 0; di < chunks.size(); ++di) {
            Chunk& c = *chunks[di];
            if (!c.lm) continue;
            cudaSetDevice(devs[di].id);
            cudaStreamSynchronize(devs[di].st);   // (nothing of the run's still reads or writes them: the device's
            cudaStreamSynchronize(c.cs);          // copies done, every read the run issued done - a failed run's too)
            if (c.ls_mode != 0) ls_end(c);
            for (size_t i = c.base; i < c.n_read; ++i)
                while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            if (c.lm_state != c.st_kept) {   // (DS4_LM_PRE: the load's ones stay, Chunk::lc_kept)
                if (c.lm_pinned) cudaFreeHost(c.lm_state);
                else os::anon_free(c.lm_state, c.lm_state_bytes);
            }
            if (c.lc != c.lc_kept) os::anon_free(c.lc, c.lc_bytes);
            c.lm_state = c.lc = nullptr;
            c.lm_state_bytes = c.lc_bytes = 0;
            c.lm = c.lm_pinned = false;
        }
    }

    // the first layer of a long prompt's second half (forward_chunks' two passes), or -1: the first with compress ratio 1
    // (V4.1: 20), every one after it ratio 1 too and owning no compressed KV or index keys of its own (they read its),
    // every trunk layer on one GPU.  The settings' MAYA_DS4_PROMPT_SKIP (STRATA_DS4_PROMPT_SKIP; DS4_YOCO for tests):
    // 1 (the default) the skip, 2 DeepSeek's own serving's (the last 128 rows only: faster, the answers shift a
    // little), 0 off
    static int yoco_mode() {
        static const int m = [] {
            const char* v = std::getenv("DS4_YOCO");
            if (v == nullptr) v = std::getenv("STRATA_DS4_PROMPT_SKIP");
            return v != nullptr ? std::atoi(v) : 1;
        }();
        return m;
    }
    int yoco_layer() const {   // (a layer range: never - the skip's layers and the ones before it on two processes)
        if (yoco_mode() == 0 || chunk <= 0 || ranged || g.nl < 2 || (int) g.ratios.size() < g.nl) return -1;
        int yl = -1;
        for (int il = 0; il < g.nl && yl < 0; ++il)
            if (g.ratios[(size_t) il] == 1) yl = il;
        if (yl <= 0 || !L[(size_t) yl].kv_src) return -1;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.nm != 1 || l.dev != L[0].dev) return -1;
            if (il > yl && (g.ratios[(size_t) il] != 1 || l.kv_src)) return -1;
        }
        return yl;
    }

    // n tokens at positions pos .. in chunks of equal size (at most `chunk`), pipelined over the groups; the logits of
    // the last min(logits_rows, 16) rows (logits_rows > 16: a run of one chunk, every row's into chunk_logits)
    bool forward_chunks(const int* tk, int n, int logits_rows, std::string& err, int snap_slot = -1) {
        if (n < 1) return true;
        if (pos + n > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < n; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        if (o.remote_stage && !stage_live(err)) return false;
        // (a remote stage: no state before the run's last row - the worker's layers would need theirs too; one that keeps
        // checkpoints takes it between the run and its last row, Model::feed_mark)
        if (linked()) snap_slot = -1;
        // a long prompt's skip: the layers from yoco_layer() on (V4.1: 20-39) read the earlier positions only through
        // that layer's compressed KV and index keys, and each looks back one window (win - 1 positions) more than the
        // layer before - so they need their own work only on the last keep = (nl - yoco) (win - 1) + win rows for the
        // run's last rows and the rings the decode reads to be what the full work gives (up to the experts' summing
        // order).  The chunks stay as they are: the ones before the cut go through the layers before yoco and its
        // kv_only, the cut's chunk through those over all its rows and through the rest from the cut on (Run::r0, a
        // view of its buffers), the chunks after it through every layer - each layer's experts read as without the
        // skip.  One GPU (DS4_YOCO=1; 0: off); not with every row's logits (tests)
        const int yl = logits_rows > kWin ? -1 : yoco_layer();
        // (DS4_YOCO_KEEP=<rows>: fewer - DeepSeek's own serving replays the last 128 tokens through those layers
        //  (DeepInfer's "SWA bounded replay"), which shifts the answers a little; at least one window)
        static const int keep_env = std::getenv("DS4_YOCO_KEEP") ? std::atoi(std::getenv("DS4_YOCO_KEEP")) : 0;
        const int keep = yl <= 0 ? 0
                         : keep_env > 0 ? std::max(keep_env, g.win)
                         : yoco_mode() == 2 ? g.win : (g.nl - yl) * (g.win - 1) + g.win;
        const int a = yl > 0 && n - keep >= kYocoMin ? n - keep : 0;
        const int nc = (n + chunk - 1) / chunk, per = (n + nc - 1) / nc;
        Run run;
        for (int x = 0; x < n; x += per) {
            run.p0.push_back(pos + x);
            run.T.push_back(std::min(per, n - x));
            run.seq.push_back(++win_seq);
        }
        if (a > 0) {
            run.yoco = yl;
            run.cut_k = a / per;
            run.cut_r = a - run.cut_k * per;
        }
        logits_all = logits_rows > kWin;
        if (logits_all) chunk_logits.resize((size_t) n * g.vocab);
        for (int t = 0; t < n; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        plan_streams();
        if (snap_slot >= 0) {   // feed_mark: the state before the run's last row, kept by its layers on the way
            run.snap_pos = pos + n - 1;
            for (int il = 0; il < g.nl; ++il) {
                Layer& l = L[(size_t) il];
                Dev& d = devs[(size_t) l.dev];
                if (l.snap_win == nullptr && !dalloc(d, l.snap_win, (size_t) g.win * win_rb(), err)) return false;
                if (l.c_kv != nullptr && l.snap_kv == nullptr &&
                    (!dalloc(d, l.snap_kv, (size_t) l.ratio * g.hd, err) || !dalloc(d, l.snap_sc, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
        }
        const int T_last = run.T.back();
        n_logits = logits_all ? 0 : std::min(logits_rows, T_last);
        const double tl = now_ms();
        {   // each chunk's end on the head device, for the progress reports
            cudaSetDevice(devs[(size_t) head_dev].id);
            while (chunk_done.size() < run.T.size()) {
                cudaEvent_t e = nullptr;
                CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                chunk_done.push_back(e);
            }
            chunk_rec.reset(new std::atomic<int>[chunk_done.size()]);
            for (size_t i = 0; i < chunk_done.size(); ++i) chunk_rec[i].store(0);
        }
        prof_chunk = prof_on;
        const bool lm = lm_begin(run);
        struct LmEnd {   // (every way out: its host memory freed once the device is done with it)
            Impl* p;
            bool on;
            ~LmEnd() {
                if (on) p->lm_end();
            }
        } lm_end_{this, lm};
        // (a remote stage's main process) the worker's replies on a thread of their own, each once its chunk is out,
        // into rx, row by row (link_rx_row: the run's rows with every row's logits, else the longest chunk's).  A reply
        // that fails ends the devices' run (failed, as a Stop does); every way out of a run that did not take them all
        // shuts the link - the worker's state is then unknown (stage_lost)
        struct LinkRx {
            Impl* p;
            float* rx = nullptr;
            size_t bytes = 0;
            std::thread th;
            std::atomic<int> got{0};
            std::atomic<bool> failed{false};   // (set once err is: the thread then returns)
            bool ok = true, open = false;
            std::string err, why;
            ~LinkRx() {
                {
                    std::lock_guard<std::mutex> lk(p->link_mu);
                    p->link_stop = true;
                }
                p->link_cv.notify_all();
                if (open) {
                    if (p->stage_chunks.abort) p->stage_chunks.abort();
                    p->stage_lost = true;
                    p->stage_err =
                        why.empty() ? "remote stage: a run of prompt chunks ended before the worker's replies were in" : why;
                }
                if (th.joinable()) th.join();
                if (rx != nullptr) os::anon_free(rx, bytes);
            }
        } lrx{this};
        if (o.remote_stage) {
            if (!stage_chunks.send || !stage_chunks.recv || !stage_chunks.wait || link_cx[0] == nullptr) {
                err = "remote stage: no link for prompt chunks";
                return false;
            }
            int rows = 0;
            for (const int T : run.T) rows = std::max(rows, T);
            lrx.bytes = (size_t) (logits_all ? n : rows) * (size_t) full_floats() * 4;
            lrx.rx = (float*) os::anon_alloc(lrx.bytes, -1, 1, lc_thp);
            if (lrx.rx == nullptr) {
                err = "remote stage: " + std::to_string(lrx.bytes >> 20) + " MiB of memory for the worker's replies";
                lrx.bytes = 0;
                return false;
            }
            {
                std::lock_guard<std::mutex> lk(link_mu);
                link_sent = 0;
                link_stop = false;
            }
            lrx.open = true;
            lrx.th = std::thread([this, &run, &lrx] {
                const size_t W = (size_t) full_floats();
                for (size_t k = 0; k < run.T.size(); ++k) {
                    {
                        std::unique_lock<std::mutex> lk(link_mu);
                        link_cv.wait(lk, [&] { return link_sent > (int) k || link_stop; });
                        if (link_sent <= (int) k) {
                            lrx.ok = false;
                            lrx.err = "remote stage: the run ended before its chunk " + std::to_string(k) + " was sent";
                            lrx.failed.store(true, std::memory_order_release);
                            return;
                        }
                    }
                    if (!stage_chunks.recv(lrx.rx + link_rx_row(run, (int) k) * W, run.T[k], link_skip(run, (int) k),
                                           lrx.err)) {
                        lrx.ok = false;
                        lrx.failed.store(true, std::memory_order_release);
                        return;
                    }
                    lrx.got.store((int) k + 1, std::memory_order_release);
                }
            });
        }
        std::vector<std::string> errs(devs.size());
        halted = false;
        halt.store(false);
        {
            std::lock_guard<std::mutex> lk(halt_mu);
            halt_k = INT_MAX;
            halt_seen = -1;
        }
        bool halting = false;
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, &run, &errs] {
                std::string e;
                if (!run_dev_chunks((int) di, run, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        size_t reported = 0;
        for (;;) {
            bool all = true;
            for (auto& w : workers) all = all && w->idle();
            if (!all && !halting && ((cancel && cancel()) || lrx.failed.load(std::memory_order_acquire))) {
                // a Stop, or a worker's reply that failed: the devices end the run early
                halting = true;
                std::lock_guard<std::mutex> lk(halt_mu);
                halt_k = halt_seen + 1;
                halt.store(true, std::memory_order_release);
            }
            if (progress && o.remote_stage) {   // (a remote stage: the chunks whose rows came back from the worker)
                const int r = lrx.got.load(std::memory_order_acquire);
                if (r > 0 && r < (int) run.T.size()) progress(run.p0[(size_t) r - 1] + run.T[(size_t) r - 1] - run.p0[0]);
            } else if (progress && lm) {   // (layer-major: the share of the layers every chunk went through)
                const int nd = lm_layers.load(std::memory_order_acquire);
                if (nd > 0 && nd < g.nl) progress((int64_t) n * nd / g.nl);
            } else if (progress) {   // chunks through every layer (the last layer's device past their event)
                while (reported < run.T.size() && chunk_rec[reported].load(std::memory_order_acquire) &&
                       cudaEventQuery(chunk_done[reported]) == cudaSuccess)
                    ++reported;
                if (reported > 0 && reported < run.T.size())
                    progress(run.p0[reported - 1] + run.T[reported - 1] - run.p0[0]);
            }
            if (all) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (o.remote_stage) {   // every chunk sent (the last one out) and every reply in, or the link is shut (lrx)
            bool fine = true;
            for (const std::string& e : errs) fine = fine && e.empty();
            std::string e;
            if (fine && !stage_chunks.wait(e)) fine = false;
            if (fine || lrx.failed.load(std::memory_order_acquire)) {   // (a failed reply: its thread is returning)
                lrx.th.join();
                if (!lrx.ok) {
                    fine = false;
                    e = lrx.err;
                }
            }
            if (fine) lrx.open = false;
            else {   // (the link is shut on the way out: ~LinkRx) a device's own failure first, then a Stop's early end
                lrx.why = e;
                bool dev_err = false, stop = false;
                for (const std::string& x : errs) {
                    dev_err = dev_err || (!x.empty() && x != kHalted);
                    stop = stop || x == kHalted;
                }
                if (!dev_err && (!stop || lrx.failed.load(std::memory_order_acquire))) {
                    err = e.empty() ? "remote stage: a run of prompt chunks failed" : e;
                    return false;
                }
            }
        }
        for (const std::string& e : errs)   // (a device's failure first: a Stop's early end is not one)
            if (!e.empty() && e != kHalted) {
                err = e;
                return false;
            }
        for (const std::string& e : errs)
            if (e == kHalted) {
                halted = true;
                err = "stopped while reading the prompt";
                return false;
            }
        if (o.remote_stage && !link_head(run, lrx.rx, err)) return false;
        st.layer_ms += now_ms() - tl;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        if (!head_wait(d.st, err)) return false;
        if (prof_chunk) {   // DS4_PROF: the run's GPU time by stage, every device's summed
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
            cudaSetDevice(d.id);
            prof_collect();
            std::fprintf(stderr,
                         "ds4 prompt GPU time, %d tokens: attention %.0f ms, routing + shared expert %.0f ms, experts %.0f ms "
                         "(with the waits for streamed ones), between layers %.0f ms\n",
                         n, cprof_ms[1], cprof_ms[2], cprof_ms[3], cprof_ms[0]);
            {   // the finer table (ms, % of the sum)
                static const char* nm_[16] = {"between layers", "attn rest (q_b, wo_a/b, hc_post)", "shared expert", "experts (incl waits)",
                                              "engram", "attn mHC (mix/split/norm)", "attn projections", "compressor + indexer",
                                              "q_b + rope", "sparse attention", "ffn mHC + router + sort", "", "", "", "", ""};
                double sum = 0;
                for (double v : cprof_ms) sum += v;
                for (int i = 0; i < 11; ++i)
                    std::fprintf(stderr, "ds4 prof  %-34s %9.1f ms %5.1f%%\n", nm_[i], cprof_ms[i], sum > 0 ? 100.0 * cprof_ms[i] / sum : 0.0);
                struct Agg {
                    double ms = 0, flops = 0;
                    long calls = 0, rows = 0, experts = 0;
                };
                std::map<std::pair<int, int>, Agg> agg;   // (type, role 0 gate+up, 1 down)
                {
                    std::lock_guard<std::mutex> lk(prof_mu);
                    for (const EkMark& m : ek_marks) {
                        float a = 0, b = 0;
                        cudaSetDevice(m.dev);
                        cudaEventElapsedTime(&a, m.e0, m.e1);
                        cudaEventElapsedTime(&b, m.e1, m.e2);
                        Agg& x = agg[{m.tg, 0}];
                        x.ms += a;
                        x.flops += 2.0 * 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                        x.calls++, x.rows += m.nrows, x.experts += m.nexp;
                        Agg& y = agg[{m.td, 1}];
                        y.ms += b;
                        y.flops += 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                        y.calls++, y.rows += m.nrows, y.experts += m.nexp;
                        ek_pool[m.dev].push_back(m.e0);
                        ek_pool[m.dev].push_back(m.e1);
                        ek_pool[m.dev].push_back(m.e2);
                    }
                    ek_marks.clear();
                }
                double tot = 0;
                for (auto& kv : agg) tot += kv.second.ms;
                for (auto& kv : agg)
                    std::fprintf(stderr,
                                 "ds4 prof  expert kernels type %-8s %-7s %9.1f ms  calls %5ld  experts %6ld (%.1f us each, %.1f a call)  rows/expert %6.1f  ~%5.1f TFLOPS\n",
                                 gg::iq::name(kv.first.first), kv.first.second ? "down" : "gate+up", kv.second.ms, kv.second.calls,
                                 kv.second.experts, kv.second.experts > 0 ? 1e3 * kv.second.ms / (double) kv.second.experts : 0.0,
                                 kv.second.calls > 0 ? (double) kv.second.experts / (double) kv.second.calls : 0.0,
                                 kv.second.experts > 0 ? (double) kv.second.rows / (double) kv.second.experts : 0.0,
                                 kv.second.ms > 0 ? kv.second.flops / kv.second.ms / 1e9 : 0.0);
                std::fprintf(stderr, "ds4 prof  expert kernels total %.1f ms\n", tot);
                // gemm(): W's FP16 reconstruction and cuBLAS's GEMM apart, by the matrix (no "%" at the end: not stages)
                struct GAgg {
                    double dq = 0, mm = 0, flops = 0;
                    long calls = 0, pieces = 0;
                };
                std::map<std::pair<int, int>, GAgg> ga;   // (k, n)
                {
                    std::lock_guard<std::mutex> lk(prof_mu);
                    for (const GmMark& m : gm_marks) {
                        float a = 0, b = 0;
                        cudaSetDevice(m.dev);
                        cudaEventElapsedTime(&a, m.e0, m.e1);
                        cudaEventElapsedTime(&b, m.e1, m.e2);
                        GAgg& x = ga[{m.k, m.n}];
                        x.dq += a;
                        x.mm += b;
                        x.flops += 2.0 * m.rows * (double) m.k * (double) m.nc;
                        x.calls += m.first, x.pieces++;
                        ek_pool[m.dev].push_back(m.e0);
                        ek_pool[m.dev].push_back(m.e1);
                        ek_pool[m.dev].push_back(m.e2);
                    }
                    gm_marks.clear();
                }
                double gdq = 0, gmm = 0;
                for (auto& kv : ga) {
                    gdq += kv.second.dq, gmm += kv.second.mm;
                    std::fprintf(stderr,
                                 "ds4 prof  gemm k %5d n %6d: calls %6ld  pieces %6ld  dequant %9.1f ms  gemm %9.1f ms (~%5.1f TFLOPS)\n",
                                 kv.first.first, kv.first.second, kv.second.calls, kv.second.pieces, kv.second.dq, kv.second.mm,
                                 kv.second.mm > 0 ? kv.second.flops / kv.second.mm / 1e9 : 0.0);
                }
                std::fprintf(stderr, "ds4 prof  gemm total: dequant %.1f ms, gemm %.1f ms\n", gdq, gmm);
            }
            for (double& v : cprof_ms) v = 0;
            prof_chunk = false;
        }
        if (o.verbose || std::getenv("DS4_FLOW_STATS") != nullptr) {
            const double ms = now_ms() - tl;
            for (size_t di = 0; di < chunks.size(); ++di) {
                const Chunk& c = *chunks[di];
                if (!c.flow) continue;
                const double db = (double) (c.io->direct_bytes() - c.io_d0) / 1e9;
                const double bb = (double) (c.io->buffered_bytes() - c.io_b0) / 1e9;
                const double fb = db + bb, rb = (double) c.it_ram * (double) L[0].blob / 1e9;   // (a member: its parts)
                char lcs[160] = "", sps[256] = "", sts[128] = "";
                if (c.S > c.ring_n)   // (DS4_STAGE_N)
                    std::snprintf(sts, sizeof sts, "; DS4_STAGE_N: %d staging slots (%.2f GB), %d in the ring", c.S,
                                  (double) c.S * (double) c.stage_stride / 1e9, c.ring_n);
                if (c.io->split()) {
                    const IoQueue::Split s = c.io->split_stats();
                    int w = std::snprintf(sps, sizeof sps, "; DS4_IO_SPLIT: %llu reads split, %.1f GB of them from the "
                                          "page cache, %.1f GB from the disk unbuffered",
                                          (unsigned long long) (s.jobs - c.io_s0.jobs), (double) (s.lead - c.io_s0.lead) / 1e9,
                                          (double) (s.direct - c.io_s0.direct) / 1e9);
                    if (s.checked > c.io_s0.checked && w > 0 && (size_t) w < sizeof sps)
                        std::snprintf(sps + w, sizeof sps - (size_t) w, " (DS4_IO_SPLIT_CHECK: %llu read again, %llu "
                                      "not the same bytes)", (unsigned long long) (s.checked - c.io_s0.checked),
                                      (unsigned long long) (s.bad - c.io_s0.bad));
                }
                if (c.lm)
                    std::snprintf(lcs, sizeof lcs, ", %llu again from the layer cache (layer-major, %d chunks, %.1f GB "
                                  "of RAM)", (unsigned long long) c.it_lc, (int) run.T.size(), (double) c.lc_bytes / 1e9);
                std::fprintf(stderr,
                             "ds4 prompt stream, CUDA%d: %d tokens in %.0f ms; %llu experts streamed: %llu from the files "
                             "(%.1f GB, %.2f GB/s: %.1f GB from the disk unbuffered, %.1f GB through the page cache), %llu "
                             "from RAM (%.1f GB)%s; %llu likely ones queued early, %llu of them unpicked; the compute waited "
                             "%.0f ms for reads%s%s; the readers busy %.0f ms in all\n",
                             devs[di].id, n, ms, (unsigned long long) (c.it_file + c.it_ram + c.it_lc),
                             (unsigned long long) c.it_file, fb, fb / std::max(1e-3, ms / 1000.0), db, bb,
                             (unsigned long long) c.it_ram, rb, lcs, (unsigned long long) c.it_spec,
                             (unsigned long long) c.it_spec_unused, c.wait_ms, sps, sts, c.io->busy_ms() - c.io_busy0);
            }
        }
        if (pm.on && run.T.size() == 1) {   // (the picks' spread for pick_path: the CPU-lane experts this run picked)
            bool tp = false;
            for (const Layer& l : L) tp = tp || l.nm > 1;
            double picked = 0;
            for (const auto& cp : chunks)
                if (cp != nullptr && cp->flow) picked += (double) (cp->it_file + cp->it_ram) - (double) cp->it_spec_unused;
            if (!tp) pm_learn(n, picked);
        }
        if (logits_all) n_logits = (T_last - 1) % kWin + 1;   // (sample() takes the last row of d.logits)
        pos += n;
        st.tokens += n;
        cache_tick(n, 0);   // (the server's experts as at the settle: the chunks streamed them to the GPU as any - the
                            // decode that follows is what the server serves)
        if (lm_end_.on) {   // (the layer cache's RAM back before the store and the swaps want it)
            lm_end();
            lm_end_.on = false;
        }
        if (!o.stage_worker || stage_last) cache_settle();   // (a worker: after its main process's whole run)
        const int nr = std::min(T_last, tap_cap);
        if (!seed_drafter(pos - nr, nr, 0, err)) return false;
        if (snap_slot >= 0) {   // checkpoint `snap_slot` from what the layers kept (the drafter's blocks: as they are -
            // the next feed seeds them again from its own rows, past this position)
            Check& ckp = ck[snap_slot & 1];
            ckp.pos = run.snap_pos;
            ckp.win.resize(L.size());
            ckp.comp.resize(L.size() * 2);
            for (size_t i = 0; i < L.size(); ++i) {
                Layer& l = L[i];
                Dev& dv = devs[(size_t) l.dev];
                cudaSetDevice(dv.id);
                const bool kept = !l.mtp && l.snap_win != nullptr;
                ckp.win[i].resize((size_t) g.win * win_rb());
                CK(cudaMemcpyAsync(ckp.win[i].data(), kept ? l.snap_win : l.win, ckp.win[i].size(),
                                   cudaMemcpyDeviceToHost, dv.st));
                if (l.c_kv) {
                    ckp.comp[2 * i].resize((size_t) l.ratio * g.hd);
                    ckp.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                    CK(cudaMemcpyAsync(ckp.comp[2 * i].data(), kept && l.snap_kv ? l.snap_kv : l.c_kv,
                                       ckp.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                    CK(cudaMemcpyAsync(ckp.comp[2 * i + 1].data(), kept && l.snap_sc ? l.snap_sc : l.c_score,
                                       ckp.comp[2 * i + 1].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                }
            }
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
        }
        return true;
    }

    // DS4_EP_DEFER: on only where v1's deferral holds - the whole trunk here on one GPU (no group, no range, no threads
    // mode), the async decode path through moe_gate_up / moe_down, no DSpark drafter (its verify windows would wait in
    // place; loaded, it turns the deferral off for the run)
    bool defer_setup(std::string& err) {
        auto on = [](const char* n) {
            const char* v = std::getenv(n);
            return v != nullptr && *v != '\0' && std::atoi(v) != 0;
        };
        ep_verify = on("DS4_EP_DEFER_VERIFY");
        v_wrong = ep_verify && std::atoi(std::getenv("DS4_EP_DEFER_VERIFY")) == 2;
        if (!on("DS4_EP_DEFER") && !ep_verify) return true;
        const char* nv = v_wrong ? "DS4_EP_DEFER_VERIFY=2" : ep_verify ? "DS4_EP_DEFER_VERIFY=1" : "DS4_EP_DEFER=1";
        if (const char* f = std::getenv("DS4_EP_DEFER_FROM"); f != nullptr && *f != '\0')
            defer_from = std::max(0, std::min(g.nl - 1, std::atoi(f)));
        const char* why = nullptr;
        if (g.n_mtp > 0) why = "the DSpark drafter is loaded (--dspark none runs without it)";
        else if (par) why = "threads mode";
        else if (ranged || o.remote_stage || o.stage_worker) why = "a layer range or a remote stage";
        else if (sync_decode || async_max_t < 1) why = "DS4_SYNC or DS4_ASYNC_T=0 (decode not on the async path)";
        else if (!moe_token_path) why = "DS4_MOE_TOKEN=0";
        for (int il = lo; il < hi && why == nullptr; ++il)
            if (L[(size_t) il].nm != 1 || L[(size_t) il].dev != L[(size_t) lo].dev) why = "the layers on more than one GPU";
        if (why != nullptr) {
            std::fprintf(stderr, "ds4: %s: off for this run - %s\n", nv, why);
            ep_verify = v_wrong = false;
            return true;
        }
        Dev& d = devs[(size_t) L[(size_t) lo].dev];
        if (!dalloc(d, d.late_post, (size_t) kWin * 8, err)) return false;
        if (ep_verify) {
            const size_t W = kWin, D = (size_t) g.dim, H = (size_t) g.hc;
            if (!dalloc(d, d.v_moe, W * D, err) || !dalloc(d, d.v_h, W * H * D, err) || !dalloc(d, d.v_h2, W * H * D, err) ||
                !dalloc(d, v_err, (size_t) kMaxL, err))
                return false;
        }
        ep_defer = true;
        if (ep_verify)
            std::fprintf(stderr, "ds4: DS4_EP_DEFER_VERIFY=%d: decode stays exact (each layer waits for its part); the "
                                 "deferred streams of layers %d-%d are built beside it from the same attention and "
                                 "compared (a wiring check, slower)%s\n", v_wrong ? 2 : 1, defer_from, g.nl - 1,
                         v_wrong ? " - with a wrong comb on purpose in the layers' adds (not the head's): the comparison "
                                   "must show it there" : "");
        else
            std::fprintf(stderr, "ds4: DS4_EP_DEFER=1: in decode each layer's CPU-lane part (with the remote experts') of "
                                 "layers %d-%d is added a layer late, after the next layer's attention - an approximation "
                                 "(the last layer's add, before the head, is exact)\n", defer_from, g.nl - 1);
        return true;
    }
    // DS4_EP_DEFER: the part still out (late_slot) into d.h once the service thread answered it - comb: the attention
    // hc_post's that ran since (null: none, the head); DS4_PROF counts the wait as the decode's wait for the CPU lane
    // DS4_EP_DEFER_VERIFY: the same add into the deferred streams (v_h after the next attention's hc_post from the same
    // ao / a_post / comb, or v_h itself before the head), then compared with d.h, the exact ones
    void late_flush(Dev& d, const float* comb, int T) {
        if (late_slot < 0) return;
        Resp& rp = resps[late_slot];
        wait_seq(&rp.seq, late_seq, d.st);
        prof_mark(d, 4);
        float* h = d.h;
        if (ep_verify) {
            h = d.v_h;
            if (comb != nullptr) {   // (the exact side's attention hc_post, on the deferred streams; the add's comb is the
                                     //  caller's, the thing checked)
                hc_post(d.ao, d.v_h, d.a_post, d.a_comb, g.dim, g.hc, T, d.v_h2, d.st);
                h = d.v_h2;
            }
        }
        hc_late_add(h, &rp.part[0][0], (size_t) kMaxDim, rp.cpu, d.late_post, comb, g.dim, g.hc, T, d.st);
        if (ep_verify) {
            max_rel_diff(d.h, h, (size_t) T * g.hc * g.dim, v_err + late_il, d.st);
            if (!small_feed) ++st.ep_verify_n;   // (the decode's checks: a small prompt's are made, but not counted; the
                                                 //  largest differences, ep_verify[], stay since the load's)
        }
        late_slot = -1;
    }

    // the window's plan, in window order (the indexer's and the candidate source's sizes carry from window to window)
    // (ep: an EP window - decode, a draft's verification - whichever lane async picks for it)
    void plan_window(int64_t p0, int T, bool ep, WinPlan& wp) {
        wp.par = par;
        wp.async = par || (!sync_decode && T <= async_max_t);
        wp.ep = ep;
        wp.defer = ep_defer && ep && T == 1 && wp.async && !par;
        wp.vl = 0;
        for (int t = 0; t < T && !imgs.empty(); ++t)
            if (in_image(p0 + t)) wp.vl |= 1u << t;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.ratio > 0) {
                const int R = l.ratio;
                int n[kWin], k[kWin], clen_max = 0;
                for (int t = 0; t < T; ++t) {
                    n[t] = (int) ((p0 + t + 1) / R);
                    k[t] = std::min(g.ix_topk, n[t]);
                    clen_max = std::max(clen_max, n[t]);
                }
                if (l.ix_src) {
                    for (int t = 0; t < T; ++t) wp.cand[il][t] = cur_cand_n[t];
                    if (clen_max > 0 && l.cand_src)
                        for (int t = 0; t < T; ++t) cur_cand_n[t] = std::min(g.cand_topk, (n[t] + g.cand_blk - 1) / g.cand_blk);
                    for (int t = 0; t < T; ++t) cur_topk_n[t] = k[t];
                }
                for (int t = 0; t < T; ++t) wp.topk[il][t] = cur_topk_n[t];
            }
            if (wp.async && !elsewhere(l)) {   // (the service thread answers the seqs in order: only the layers run here)
                wp.seq[il] = ++seq_issued;
                ep_of[wp.seq[il] % kRing] = wp.ep;
                small_of[wp.seq[il] % kRing] = small_feed;
                layer_of[wp.seq[il] % kRing].store(il, std::memory_order_release);
            }
        }
    }

    int head_nm() const { return (int) groups[(size_t) devs[(size_t) head_dev].grp].mem.size(); }
    // the head's columns on member mi of its group of nm: an EXL3 head's split128 of the vocabulary (mat's), a GGUF's
    // BF16 / Q8_0 head's even shares (mat_bf16's) - head_share's, its gather's and the markov steps' rows
    void head_cols(int nm, int mi, int& v0, int& vn) const {
        const Tensor* tw = pk.get("head.weight");
        if (tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0")) {
            vn = g.vocab / nm;
            v0 = mi * vn;
        } else
            split128(g.vocab, nm, mi, v0, vn);
    }
    // the head over `rows` (<= 16) normed rows xn on member mi of the head group (device d): its share of each row's
    // logits (columns [v0, v0 + vn), head_cols); the lead gathers the others' into d.logits (rows V apart) - two
    // slots by seq (Vmax apart, the widest share's), a member reusing one once the lead acknowledged it.  gather
    // false: each member keeps its share (the drafter's markov steps over the group, mk_split)
    bool head_share(Dev& d, int mi, const float* xn, int rows, std::string& err, bool gather = true) {
        const int V = g.vocab, D = g.dim, nm = head_nm();
        int v0 = 0, vn = 0, Vmax = 0, z = 0;
        head_cols(nm, mi, v0, vn);
        head_cols(nm, 0, z, Vmax);
        Group& G = groups[(size_t) d.grp];
        float* out = mi == 0 ? d.logits : d.logits_part;
        const size_t ld = mi == 0 ? (size_t) V : (size_t) vn;
        cudaSetDevice(d.id);
        // (an EXL3 head: as many activations at a time as its split-K partials fit the scratch - 8 for a shared head, 4
        // for a whole one, each pass reading the whole share; a GGUF's: 4 at a time, as before)
        const int per = d.head.fmt != 0 ? 4 : std::max(1, x3::mv_cols(d.head));
        for (int c = 0; c < rows; c += per)
            if (!mv(d, {job(d.head, xn + (size_t) c * D, out + (size_t) c * ld, std::min(per, rows - c), D, ld)}, err))
                return false;
        if (nm == 1 || !gather) return true;
        const unsigned seq = ++d.head_seq;
        const int slot = (int) (seq % 2);
        auto stage = [&](int m) { return G.lg_stage + ((size_t) slot * (nm - 1) + (m - 1)) * kWin * Vmax; };
        auto flag = [&](int m) { return G.lg_flag + ((size_t) slot * (nm - 1) + (m - 1)) * 32; };
        if (mi > 0) {
            if (seq > 2) wait_flags(G.lg_ack + (size_t) slot * 32, 1, 32, seq - 2, d.st);
            Xfer x;
            x.n = 1;
            x.src[0] = d.logits_part;
            x.dst[0] = stage(mi);
            x.bytes[0] = (size_t) rows * vn * 4;
            xfer_put(x, flag(mi), seq, d.xcnt, d.st);
            return true;
        }
        for (int m = 1; m < nm; ++m) {
            int m0 = 0, mn = 0;
            head_cols(nm, m, m0, mn);
            Xfer x;
            x.n = 1;
            x.src[0] = stage(m);
            x.dst[0] = d.logits_part;
            x.bytes[0] = (size_t) rows * mn * 4;
            xfer_get(x, flag(m), seq, d.st);
            CK(cudaMemcpy2DAsync(d.logits + (size_t) m0, (size_t) V * 4, d.logits_part, (size_t) mn * 4,
                                 (size_t) mn * 4, (size_t) rows, cudaMemcpyDeviceToDevice, d.st));
        }
        set_flag(G.lg_ack + (size_t) slot * 32, seq, d.st);
        return true;
    }

    // the head's rows (the last n_logits of the window) on head group member di (its h / pre: the group's)
    bool head_rows_dev(int di, int T, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const int D = g.dim, H = g.hc;
        if (n_logits <= 0) return true;
        const int t0 = T - n_logits;
        hc_pre_norm(d.h + (size_t) t0 * H * D, d.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, d.st);
        return head_share(d, d.mi, d.x, n_logits, err);
    }

    // the DSpark drafter's taps: the streams' mean entering target layer il, on device d (after the last seeding read
    // h_taps).  A remote stage's worker: into its reply (ds4_stage.hpp's reply_tap_at), for its main process's drafter
    bool tap(Dev& d, int il, int T, std::string& err) {
        const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
        if ((g.n_mtp == 0 && !o.stage_worker) || it == g.d_targets.end()) return true;
        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
        const int D = g.dim;
        cudaSetDevice(d.id);
        hc_mean(d.h, T, g.hc, D, d.tap, D, d.st);
        if (o.stage_worker) {
            if (!stage::reply_fits(D, (int) nt, (int64_t) g.hc * D)) return true;   // (load said so: no DSpark there)
            CK(cudaMemcpy2DAsync(link_tx + stage::reply_tap_at(T, D, (int) nt, 0, (int) j), nt * D * 4, d.tap,
                                 (size_t) D * 4, (size_t) D * 4, (size_t) T, cudaMemcpyDeviceToHost, d.st));
            return true;
        }
        CK(cudaStreamWaitEvent(d.st, ev_seed, 0));
        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4, (size_t) T,
                             cudaMemcpyDeviceToHost, d.st));
        return true;
    }

    // (ep: an EP window - decode, a draft's verification - its layers' n_ff slices; false: the prompt's, whole)
    bool forward(const int* tk, int T, int logits_rows, bool ep, std::string& err, int seed_rows = -1) {
        if (T < 1 || T > kWin) {
            err = "a window of " + std::to_string(T) + " tokens";
            return false;
        }
        if (pos + T > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < T; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        if (rx_link) rx_wake.store(true, std::memory_order_relaxed);   // (serve_lane: a Wake if the link went quiet)
        if (stage_lost && pos == 0 && stage_reset && stage_reset(stage_err)) stage_lost = false;   // (the worker back)
        if (stage_lost) {   // (the worker's layers hold another state, or none: only a conversation from 0 goes on)
            err = stage_err + " - the conversation starts again from its first token";   // (the link's: "remote stage ...")
            return false;
        }
        const int64_t p0 = pos;
        slot_held = false;   // (the state grows now: what slot_save took is no longer the conversation)
        for (int t = 0; t < T; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        // the Engram rows asked for now (the first Engram layer's first), collected right before their layer: what is
        // left of their reads by then is the window's Engram time
        const double te = now_ms();
        const int64_t eg_w0 = eg_wait_ns.load();
        for (int li = 0; li < pk.eg_layers; ++li) {
            if (ranged && (li >= (int) eg_win.size() || !local(g.eg_layers[(size_t) li]))) continue;   // (another process's)
            EgBatch& b = *eg_win[(size_t) li];
            if (b.live) eg_finish(b);   // (a window that failed midway: its reads end before h_eg is reused)
            eg_issue(p0, T, li, nullptr, b);
        }
        st.engram_ms += now_ms() - te;
        const int D = g.dim;
        for (int t = 0; t < T && !o.stage_worker; ++t)   // (h_emb is written again by the next window, after this one's last sync)
            std::memcpy(h_emb + (size_t) t * D, emb_row(p0 + t, tk[t]), (size_t) D * 2);
        late_slot = -1;   // (DS4_EP_DEFER: a window that failed midway leaves nothing for this one)
        plan_window(p0, T, ep, wplan);
        prof_window = prof_on && T == 1 && !small_feed;
        n_logits = std::min(logits_rows, T);
        const double tl = now_ms();
        if (par) {
            if (!forward_par(p0, T, err)) return false;
        } else if (!forward_serial(p0, T, err)) return false;
        if (late_slot >= 0) {   // (forward_serial adds the last one before the head: a part never crosses a window)
            std::fprintf(stderr, "ds4: DS4_EP_DEFER: a window ended with layer parts still out\n");
            std::abort();
        }
        st.layer_ms += now_ms() - tl;
        for (auto& b : eg_win)
            if (b->live) eg_finish(*b);
        st.engram_ms += (double) (eg_wait_ns.load() - eg_w0) / 1e6;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        if (!head_wait(d.st, err)) return false;
        if (ep_verify) {   // (the window's checks are in: its stream is done)
            CK(cudaMemcpy(st.ep_verify, v_err, sizeof(float) * (size_t) std::min(g.nl, 64), cudaMemcpyDeviceToHost));
            st.ep_verify_wrong = v_wrong;
        }
        prof_collect();
        pos += T;
        st.tokens += T;
        cache_tick(T, win_prompt ? 0 : -1);   // (a window prompt's: the server's experts as the decode's, rx_row(.., 0))
        psettle.window_end(ep, win_prompt);   // (a decode window: the next prompt settles again; a small prompt's
                                              //  (feed_small) is the prompt's, which settles once at its end)
        if (seed_rows != 0 && !seed_drafter(p0, seed_rows < 0 ? T : seed_rows, 0, err)) return false;
        return true;
    }

    // one thread: the layers device after device, the hand-offs through pinned memory and events
    bool forward_serial(int64_t p0, int T, std::string& err) {
        const int D = g.dim, H = g.hc;
        int cur = L[(size_t) lo].dev;
        {
            Dev& d = devs[(size_t) cur];
            cudaSetDevice(d.id);
            if (o.stage_worker) {   // (a remote stage's worker: the main process's rows)
                if (!link_rows(link_rx, d, T, false, 4, err)) return false;
            } else {
                CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
                embed(d.emb_rows, T, d.h, D, H, d.st);
                init_pre(d.pre, T, d.st);
            }
        }
        std::vector<float> dump;
        if (dump_h != nullptr) dump.resize((size_t) T * g.nl * H * D);
        for (int il = lo; il < hi; ++il) {
            Layer& l = L[(size_t) il];
            if (l.dev != cur) {
                Dev& a = devs[(size_t) cur];
                Dev& b = devs[(size_t) l.dev];
                cudaSetDevice(a.id);
                const size_t nh = (size_t) T * H * D * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                             nc = (size_t) T * cand_stride() * 4;
                uint8_t* bb = a.bounce;
                CK(cudaMemcpyAsync(bb, a.h, nh, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh, a.pre, np, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np, a.topk, nk, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np + nk, a.cand, nc, cudaMemcpyDeviceToHost, a.st));
                CK(cudaEventRecord(a.ev_out, a.st));
                cudaSetDevice(b.id);
                CK(cudaStreamWaitEvent(b.st, a.ev_out, 0));
                CK(cudaMemcpyAsync(b.h, bb, nh, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.pre, bb + nh, np, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.topk, bb + nh + np, nk, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.cand, bb + nh + np + nk, nc, cudaMemcpyHostToDevice, b.st));
                cur = l.dev;
            }
            if (!tap(devs[(size_t) l.dev], il, T, err)) return false;
            if (!layer(l, 0, p0, T, wplan, err)) return false;
            if (dump_h != nullptr) {   // DS4_DUMP_H: every layer's residual stream, for tools/ds4/compare.py --layers
                Dev& dd = devs[(size_t) l.dev];
                std::vector<float> hv((size_t) T * H * D);
                cudaSetDevice(dd.id);
                CK(cudaMemcpyAsync(hv.data(), dd.h, hv.size() * 4, cudaMemcpyDeviceToHost, dd.st));
                CK(cudaStreamSynchronize(dd.st));
                for (int t = 0; t < T; ++t)
                    std::memcpy(dump.data() + ((size_t) t * g.nl + l.il) * H * D, hv.data() + (size_t) t * H * D,
                                (size_t) H * D * 4);
            }
        }
        if (dump_h != nullptr) std::fwrite(dump.data(), 4, dump.size(), dump_h);
        late_flush(devs[(size_t) cur], nullptr, T);   // (DS4_EP_DEFER: the last layer's part, before any of the exits)
        if (cur != head_dev) {
            err = "the head is not on the last layer's GPU";
            return false;
        }
        if (o.stage_worker) return stage_reply(devs[(size_t) head_dev], T, err);   // (back to the main process)
        if (o.remote_stage) return remote_hop(head_dev, p0, T, err);
        return head_rows_dev(head_dev, T, err);
    }

    // a remote stage's rows (host, Model::StageGeo's layout) <-> device d's h | pre | topk | cand - the first `blocks`
    // of them - on d's stream
    bool link_rows(float* host, Dev& d, int T, bool to_host, int blocks, std::string& err) {
        const size_t nb[4] = {(size_t) T * g.hc * g.dim * 4, (size_t) T * 8 * 4, (size_t) T * topk_stride() * 4,
                              (size_t) T * cand_stride() * 4};
        void* dv[4] = {d.h, d.pre, d.topk, d.cand};
        uint8_t* hb = (uint8_t*) host;
        cudaSetDevice(d.id);
        for (int i = 0; i < blocks; hb += nb[i], ++i)
            CK(cudaMemcpyAsync(to_host ? (void*) hb : dv[i], to_host ? (const void*) dv[i] : (const void*) hb, nb[i],
                               to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, d.st));
        return true;
    }
    // a remote stage's worker after its last layer (device d): the window's rows through the final norm, then (tap()
    // wrote them) the drafter's taps - the reply's layout, ds4_stage.hpp's reply_tap_at
    bool stage_reply(Dev& d, int T, std::string& err) {
        cudaSetDevice(d.id);
        hc_pre_norm(d.h, d.pre, d.norm_w, g.dim, g.hc, T, g.eps, d.x, d.st);
        CK(cudaMemcpyAsync(link_tx, d.x, (size_t) T * g.dim * 4, cudaMemcpyDeviceToHost, d.st));
        return true;
    }
    // the main process's hop after its last layer (device di): the window's rows to the worker, which runs the rest of
    // the trunk; its rows through the final norm back for the head, the drafter's taps of its layers into h_taps.  A
    // window kept whole is committed here; a draft's check (spec_win) by verify(), once it knows what stands.  A
    // failure leaves the worker's state unknown: stage_lost
    bool remote_hop(int di, int64_t p0, int T, std::string& err) {
        Dev& d = devs[(size_t) di];
        if (!link_rows(link_tx, d, T, true, 4, err)) return false;
        CK(cudaStreamSynchronize(d.st));
        const bool bf = bf16_rows();
        if (bf) {   // (the compact rows: this window's out, and the reply's streams and pre back)
            const double t0 = now_ms();
            stage::rows_pack(row_geo(), T, link_tx, wire_tx);
            conv_out_ms += now_ms() - t0;
            ++conv_out_n;
        }
        if (!stage_hop || !stage_hop(T, toks.data() + p0, p0, bf ? wire_tx : link_tx, bf ? wire_rx : link_rx, err)) {
            if (!stage_hop) err = "remote stage: no link";
            stage_lost = true;
            stage_err = err;
            return false;
        }
        if (bf) {   // (the reply's streams block: the normed rows and the taps)
            const double t0 = now_ms();
            stage::rows_unpack(row_geo(), T, wire_rx, link_rx, false);
            conv_in_ms += now_ms() - t0;
            ++conv_in_n;
        }
        if (!spec_win && !stage_keep(T, err)) return false;
        if (g.n_mtp > 0) {   // (after the last seeding's read of h_taps)
            CK(cudaEventSynchronize(ev_seed));
            stage::taps_from_reply(link_rx, T, g.dim, g.d_targets, hi, h_taps);
        }
        if (n_logits <= 0) return true;
        const int t0 = T - n_logits;
        cudaSetDevice(d.id);
        CK(cudaMemcpyAsync(d.x, link_rx + (size_t) t0 * g.dim, (size_t) n_logits * g.dim * 4, cudaMemcpyHostToDevice,
                           d.st));
        return head_share(d, d.mi, d.x, n_logits, err);
    }
    // the worker keeps the first n rows of the window it ran last (a failure: stage_lost)
    bool stage_keep(int n, std::string& err) {
        if (stage_commit && stage_commit(n, err)) return true;
        if (!stage_commit) err = "remote stage: no link";
        stage_lost = true;
        stage_err = err;
        return false;
    }

    // threads: each device's thread enqueues its layers (a group's members each their share) at once, the
    // devices waiting on each other through host-mapped flags (the window's seq)
    bool forward_par(int64_t p0, int T, std::string& err) {
        wplan.win = ++win_seq;
        std::vector<std::string> errs(devs.size());
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, p0, T, &errs] {
                std::string e;
                if (!run_dev((int) di, p0, T, wplan, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        for (auto& w : workers) w->wait();
        for (const std::string& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
        return true;
    }

    // the hand-off between groups: h | pre | topk | cand of device d <-> the host-mapped buffer
    Xfer hand_xfer(uint8_t* host, Dev& d, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {d.h, d.pre, d.topk, d.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    bool run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc;
        bool started = false;
        std::vector<char> got(srcs.size(), 0);
        for (int il = lo; il < hi; ++il) {
            Layer& L0 = L[(size_t) il];
            int mi = -1;
            for (int m = 0; m < L0.nm; ++m)
                if (member(L0, m).dev == di) mi = m;
            if (mi < 0) continue;
            if (!started) {   // the window's input: the embedding rows, or the previous group's hand-off
                started = true;
                if (il == lo && o.stage_worker) {   // (a remote stage's worker: the main process's rows)
                    if (!link_rows(link_rx, d, T, false, 4, err)) return false;
                } else if (il == lo) {
                    CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    embed(d.emb_rows, T, d.h, D, H, s);
                    init_pre(d.pre, T, s);
                } else {
                    const Dev& a = devs[(size_t) L[(size_t) il - 1].dev];
                    xfer_get(hand_xfer(a.hand, d, T, false), a.hand_flag, wp.win, s);
                }
            }
            // the compressed KV / index keys another group's source layer wrote this window
            for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                if (sidx < 0 || got[(size_t) sidx]) continue;
                got[(size_t) sidx] = 1;
                Src& sr = srcs[(size_t) sidx];
                if (L[(size_t) sr.layer].grp == L0.grp) continue;
                const int64_t g_first = p0 / sr.ratio;
                const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                if (ng == 0) continue;
                Xfer x;
                if (sr.ckv[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage;
                    x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * ckv_rb();
                    x.bytes[x.n++] = (size_t) ng * ckv_rb();
                }
                if (sr.ik[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage + x_cap() * ckv_rb();
                    x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_rb();
                    x.bytes[x.n++] = (size_t) ng * ik_rb();
                }
                if (x.n > 0) xfer_get(x, sr.x_flag, wp.win, s);
            }
            if (mi == 0 && !tap(d, il, T, err)) return false;
            if (!layer(L0, mi, p0, T, wp, err)) return false;
            if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, il, p0, T);
            const bool last = il + 1 == hi || L[(size_t) il + 1].grp != L0.grp;
            if (last && mi == 0 && il + 1 < hi) xfer_put(hand_xfer(d.hand, d, T, true), d.hand_flag, wp.win, d.xcnt, s);
        }
        // (a remote stage: the head's group is the head device alone - load checked it)
        if (o.stage_worker) return di != head_dev || stage_reply(d, T, err);   // (back to the main process)
        if (o.remote_stage) return di != head_dev || remote_hop(di, p0, T, err);
        if (d.grp == devs[(size_t) head_dev].grp) return head_rows_dev(di, T, err);
        return true;
    }

    // a source's new cache rows (src on device `from`, bytes of them) to the same place of its other devices' copies,
    // through pinned memory: the copy out on the source's stream, each copy in on its device's stream after it
    bool replicate(std::vector<uint8_t*>& copies, uint8_t* stage, cudaEvent_t ev, const uint8_t* src, size_t at,
                   size_t bytes, int from, std::string& err) {
        bool any = false;
        for (size_t od = 0; od < devs.size(); ++od) any = any || (copies[od] != nullptr && (int) od != from);
        if (!any) return true;
        Dev& d = devs[(size_t) from];
        CK(cudaMemcpyAsync(stage, src, bytes, cudaMemcpyDeviceToHost, d.st));
        CK(cudaEventRecord(ev, d.st));
        for (size_t od = 0; od < devs.size(); ++od) {
            if (copies[od] == nullptr || (int) od == from) continue;
            Dev& o2 = devs[od];
            cudaSetDevice(o2.id);
            CK(cudaStreamWaitEvent(o2.st, ev, 0));
            CK(cudaMemcpyAsync(copies[od] + at, stage, bytes, cudaMemcpyHostToDevice, o2.st));
        }
        cudaSetDevice(d.id);
        return true;
    }

    // layer L0 for the window, as group member mi (its share of the heads, of the output projection and of the FFNs;
    // the partial sums all-reduced across the group)
    bool layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ag = groups[(size_t) L0.grp].ar;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        // U9 hc: a sublayer's hc_mix / hc_split on st2, beside its other kernels (the pre-norm reads the previous
        // sublayer's pre, so only hc_post needs them: it waits for ev_join)
        const bool hc2 = nm == 1 && d.st2 != nullptr;   // (made when kHc is on)
        cudaStream_t sx = hc2 ? d.st2 : s;
        prof_mark(d, 0);
        // Engram
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            eg_finish(*eg_win[(size_t) l.eg]);   // (asked for when the window began)
            uint8_t* src = h_eg + (size_t) l.eg * eg_cap * per;
            for (int t = 0; t < T; ++t)   // [token][rows | scales] -> [all rows][all scales]
                CK(cudaMemcpyAsync(d.eg_rows + (size_t) t * cols * ehd, src + (size_t) t * per, (size_t) cols * ehd,
                                   cudaMemcpyHostToDevice, s));
            uint8_t* sc = d.eg_rows + (size_t) T * cols * ehd;
            for (int t = 0; t < T; ++t)
                CK(cudaMemcpyAsync(sc + (size_t) t * cols * (ehd / 32), src + (size_t) t * per + (size_t) cols * ehd,
                                   (size_t) cols * (ehd / 32), cudaMemcpyHostToDevice, s));
            engram_dequant(d.eg_rows, sc, T * cols, ehd, d.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
            if (!mv(d, {job(l.eg_wkv, d.eg_x, d.eg_kv, T, (size_t) cols * ehd, (size_t) (H + 1) * D)}, err)) return false;
            const bool vg = ep_verify && late_slot >= 0;   // (DS4_EP_DEFER_VERIFY: the same add onto the deferred streams)
            const size_t nh = (size_t) T * H * D;
            if (vg) CK(cudaMemcpyAsync(d.v_h2, d.h, nh * 4, cudaMemcpyDeviceToDevice, s));
            engram_gate(d.h, d.eg_kv, l.eg_w, H, D, T, g.eps, s);
            if (vg) add_diff(d.v_h, d.h, d.v_h2, nh, s);
        }
        // attention
        if (hc2) {
            CK(cudaEventRecord(d.ev_fork, s));
            CK(cudaStreamWaitEvent(d.st2, d.ev_fork, 0));
        }
        hc_mix(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, sx);
        hc_split(d.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, d.a_pre, d.a_post, d.a_comb, sx);
        if (hc2) CK(cudaEventRecord(d.ev_join, d.st2));
        hc_pre_norm(d.h, d.pre, l.attn_norm, D, H, T, g.eps, d.x, s);
        {
            x3::MvJob j[4];
            int nj = 0;
            j[nj++] = job(l.wq_a, d.x, d.qr, T, D, g.q_lora);
            j[nj++] = job(l.wkv, d.x, d.kvr, T, D, HD);
            if (l.kv_src) j[nj++] = job(l.c_wkv, d.x, d.ckr, T, D, HD);
            if (l.kv_src && l.ratio > 1) j[nj++] = job(l.c_wgate, d.x, d.cgr, T, D, HD);
            if (!mv_n(d, j, nj, err)) return false;
        }
        Finish fq;
        fq.w = l.q_norm;
        fq.eps = g.eps;
        finish(d.qr, T, g.q_lora, g.q_lora, fq, s);
        if (NHL > 0) {
            if (!mv(d, {job(l.wq_b, d.qr, d.q, T, g.q_lora, HQ)}, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) p0;
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(d.q, T * NHL, HD, HD, fr, s);
        }
        // the window rows: the ring's (positions p0 - win ..) then this window's, each query its own 128.  A DSpark
        // block's ring holds the trunk's main KV (positions <= p0 - 1, seeded); its block of drafts attends to all of
        // that ring and to the whole block, and leaves the ring as it was
        ring_to_ext(l.win, d.ext, l.mtp ? (int64_t) g.win : p0, g.win, HD, s);   // (a block's ring: its rows as they lie)
        Finish fk;
        fk.w = l.kv_norm;
        fk.eps = g.eps;
        fk.rd = g.rd;
        fk.pos = (int) p0;
        fk.freqs = freqs;
        fk.quant = 1;
        fk.dst = d.ext + (size_t) g.win * HD;
        fk.dst_stride = HD;
        finish(d.kvr, T, HD, HD, fk, s);
        if (!l.mtp) ext_to_ring(d.ext, l.win, p0, g.win, T, HD, s, spec_win ? l.ring_bak : nullptr);
        AttnRows ar;
        for (int t = 0; t < T; ++t) {
            if (l.mtp) {
                ar.win0[t] = 0;
                ar.nwin[t] = (int) std::min<int64_t>(g.win, p0);
                ar.nsel[t] = T;
                continue;
            }
            const int first = (int) std::max<int64_t>(g.win - p0, t + 1);   // ext row of position max(0, p - win + 1)
            ar.win0[t] = first;
            ar.nwin[t] = g.win + t - first + 1;
            ar.nsel[t] = 0;
        }
        const void* ckv = nullptr;
        int ckv_fmt = kKvFp4;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            // the groups this window completes: g_first .. g_first + ng - 1
            const int64_t g_first = p0 / R;              // the first group whose last position is >= p0
            const int64_t g_end = (p0 + T) / R;          // groups completed by p0 + T - 1
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(d.ckr, T, HD, HD, fc, s);
                    CK(cudaMemcpyAsync(d.latent, d.ckr, (size_t) T * HD * 4, cudaMemcpyDeviceToDevice, s));
                } else {
                    if (spec_win && l.c_snap != nullptr) {   // to replay the accepted rows if drafts are rejected
                        const size_t ns = (size_t) R * HD * 4, nr = (size_t) T * HD * 4;
                        CK(cudaMemcpyAsync(l.c_snap, l.c_kv, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_snap + (size_t) R * HD, l.c_score, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows, d.ckr, nr, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows + (size_t) kWin * HD, d.cgr, nr, cudaMemcpyDeviceToDevice, s));
                    }
                    compress_win(d.ckr, d.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, d.latent, s);
                }
            }
            Rows lim;
            int clen_max = 0;
            for (int t = 0; t < T; ++t) {
                lim.n[t] = (int) ((p0 + t + 1) / R);
                lim.k[t] = std::min(g.ix_topk, lim.n[t]);
                clen_max = std::max(clen_max, lim.n[t]);
            }
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    if (!mv(d, {job(l.i_wk, d.latent, d.ikr, ng, HD, g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(d.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                    if (!wp.par && !replicate(ks.ik, ks.h_ik, ks.ev_ik, fi.pdst, (size_t) g_first * ik_rb(),
                                              (size_t) ng * ik_rb(), l.dev, err))
                        return false;
                }
                // U9 ixfast: every query keeps all its compressed rows (n <= index_topk, no candidate restriction):
                // topk's answer is then 0 .. n - 1 whatever the scores, so the query rows, the weights and the scores
                // are skipped (not on the candidate source layer: its scores pick the candidate blocks)
                bool keep_all = clen_max > 0 && u9::on(u9::kIxfast) && !l.cand_src;
                for (int t = 0; keep_all && t < T; ++t)
                    keep_all = lim.k[t] >= lim.n[t] &&
                               !(l.uses_cand && (lim.n[t] + g.cand_blk - 1) / g.cand_blk > g.cand_topk);
                if (keep_all) {
                    topk(d.scores, (long long) o.max_context + 64, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s);
                } else if (clen_max > 0) {
                    if (!mv(d, {job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, (size_t) g.ix_heads * g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.rd = g.rd;
                    fi.pos = (int) p0;
                    fi.vpp = g.ix_heads;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                    // (iw rows are 64 apart)
                    gemv_f16_rows(l.i_wproj, d.x, (size_t) D, g.ix_heads, D, T,
                                  1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), d.iw, 64, s);
                    const long long ss = (long long) o.max_context + 64;
                    bool restrict_ = false;
                    for (int t = 0; t < T; ++t)
                        restrict_ = restrict_ || (lim.n[t] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                    if (l.uses_cand && restrict_) {
                        Rows cl;   // the candidate positions of each query: its blocks x blk
                        int nmax = 0;
                        for (int t = 0; t < T; ++t) {
                            cl.n[t] = lim.n[t];
                            cl.k[t] = wp.cand[l.il][t];
                            nmax = std::max(nmax, wp.cand[l.il][t] * g.cand_blk);
                        }
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, nmax, d.cand,
                                    (int) cand_stride(), g.cand_blk, cl, d.scores, ss, s);
                        Rows tr;
                        for (int t = 0; t < T; ++t) {
                            tr.n[t] = wp.cand[l.il][t] * g.cand_blk;
                            tr.k[t] = lim.k[t];
                        }
                        topk(d.scores, ss, T, tr, d.cand, (int) cand_stride(), g.cand_blk, d.topk, (int) topk_stride(), s);
                    } else {
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, clen_max, nullptr, 0, 1,
                                    lim, d.scores, ss, s);
                        if (l.cand_src) {
                            Rows cr;
                            for (int t = 0; t < T; ++t) {
                                cr.n[t] = lim.n[t];
                                cr.k[t] = g.cand_topk;
                            }
                            cand_blocks(d.scores, ss, T, cr, g.cand_blk, d.bm, (long long) bm_stride(), d.cand,
                                        (int) cand_stride(), s);
                        }
                        topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s);
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(d.latent, ng, HD, HD, fc, s);
                if (!wp.par && !replicate(sr.ckv, sr.h_ckv, sr.ev_ckv, fc.pdst, (size_t) g_first * ckv_rb(),
                                          (size_t) ng * ckv_rb(), l.dev, err))
                    return false;
                if (wp.par && mi == 0 && sr.remote) {   // the new rows for the other groups' readers
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.x_stage;
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.x_stage + x_cap() * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.x_flag, wp.win, d.xcnt, s);
                }
            }
            for (int t = 0; t < T; ++t) ar.nsel[t] = lim.n[t] > 0 ? wp.topk[l.il][t] : 0;
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (l.mtp) {   // the block's rows (BF16), after the ring's in ext, as every query's selection
            ckv = d.ext + (size_t) g.win * HD;
            ckv_fmt = kKvBf16;
            iota_rows(d.topk, T, T, (int) topk_stride(), s);
        }
        if (NHL > 0) {
            sparse_attn(d.q, d.ext, ckv, d.topk, (int) topk_stride(), ar, l.sink, NHL, HD, T,
                        1.0f / std::sqrt((float) HD), d.o, s, ckv_fmt, 0, &d.attn_scr);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) p0;
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(d.o, T * NHL, HD, HD, fo, s);
            x3::MvJob j[16];
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                j[gi] = job(l.wo_a[gi], d.o + (size_t) gi * per, d.oa + (size_t) gi * g.o_lora, T, HQ, (size_t) OG * g.o_lora);
            if (!mv_n(d, j, OG, err)) return false;
            if (!mv(d, {job(l.wo_b, d.oa, d.ao, T, (size_t) OG * g.o_lora, D)}, err)) return false;
        } else CK(cudaMemsetAsync(d.ao, 0, (size_t) T * D * sizeof(*d.ao), s));   // (no heads: nothing to add)
        if (nm > 1) allreduce(d.ao, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
        if (hc2) CK(cudaStreamWaitEvent(s, d.ev_join, 0));
        hc_post(d.ao, d.h, d.a_post, d.a_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, 1000 + l.il, p0, T);   // (DS4_DUMP_L: after the attention)
        prof_mark(d, 1);
        late_flush(d, v_wrong ? d.f_comb : d.a_comb, T);   // (DS4_EP_DEFER: the layer before's part, through this
                                                           //  attention's comb)
        // the MoE
        if (hc2) {
            CK(cudaEventRecord(d.ev_fork, s));
            CK(cudaStreamWaitEvent(d.st2, d.ev_fork, 0));
        }
        hc_mix(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, sx);
        hc_split(d.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, d.f_pre, d.f_post, d.f_comb, sx);
        if (hc2) CK(cudaEventRecord(d.ev_join, d.st2));
        hc_pre_norm(d.h, d.a_pre, l.ffn_norm, D, H, T, g.eps, d.x, s);
        if (wp.async) {
            // the routing goes to the service thread through the mailbox (a group's lead publishes it, the members
            // make the same plan) while the device runs the shared expert and the VRAM experts (from the plan the
            // router made), and waits for the answer only before the experts' sum
            const unsigned sq = wp.seq[l.il];
            const int slot = (int) (sq % kRing);
            const bool lead = mi == 0;
            // DS4_EP_DEFER: the mailbox part neither waited for nor added here (skip) - the next layer's late_flush adds
            // it; DS4_EP_DEFER_VERIFY: waited for and added here, and the deferred path built beside it (Dev::v_*)
            const bool late = wp.defer && lead && !l.mtp && T == 1 && moe_token_path && l.il >= defer_from;
            const bool skip = late && !ep_verify;
            int na[2] = {0, 0};   // lookahead: the next layers' guesses (their gate on this FFN input), published below
            if (u9::on(u9::kRoute)) {   // U9 route: the gates in one GEMV, the lookahead and the routing in one launch
                const Layer* nx[2] = {nullptr, nullptr};
                int nn = 0;
                for (int a = 1; lead && a <= std::min(2, ahead_mode); ++a) {
                    nx[a - 1] = ahead_layer(L0, a);
                    if (nx[a - 1] == nullptr) break;
                    na[a - 1] = a == 1 ? std::max(1, std::min(kAhead1, nx[0]->k_exp + ahead_extra))
                                       : std::min(kPicks, nx[1]->k_exp);
                    nn = a;
                }
                bool bf = l.gate_bf != nullptr;
                for (int a = 0; a < nn; ++a) bf = bf && nx[a]->gate_bf != nullptr;
                if (bf) {
                    GemvBf16 gb = {};
                    gb.w[0] = l.gate_bf;
                    gb.y[0] = d.glog;
                    gb.n[0] = l.n_exp;
                    for (int a = 0; a < nn; ++a) {
                        gb.w[1 + a] = nx[a]->gate_bf;
                        gb.y[1 + a] = d.glog_a + (size_t) a * kWin * 1024;
                        gb.n[1 + a] = nx[a]->n_exp;
                    }
                    gemv_bf16(gb, 1 + nn, d.x, D, T, 1024, s);
                } else {
                    gate_gemv(l, d.x, T, d.glog, s);
                    for (int a = 0; a < nn; ++a) gate_gemv(*nx[a], d.x, T, d.glog_a + (size_t) a * kWin * 1024, s);
                }
                RouteAhead ra = {};
                int nra = 0;
                for (int a = 0; a < nn; ++a)
                    if (na[a] > 0 && T > 0 && nx[a]->n_exp <= 1024 && a * kAhead1 + na[a] <= kAhead) {   // (route_ahead's)
                        ra.logits[nra] = d.glog_a + (size_t) a * kWin * 1024;
                        ra.bias[nra] = nx[a]->gate_bias;
                        ra.n[nra] = nx[a]->n_exp;
                        ra.m[nra] = na[a];
                        ra.off[nra] = a * kAhead1;
                        ++nra;
                    }
                route_fused(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, l.vram_tab, d.x, D, d.ridx, d.rw,
                            d.plan_ptr, lead ? reqs + slot : nullptr, sq, s, l.gate_bias_vl, l.mtp ? 0u : wp.vl, na[0],
                            na[1], ra, nra);
            } else {
                gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
                for (int a = 1; lead && a <= std::min(2, ahead_mode); ++a) {
                    const Layer* nx = ahead_layer(L0, a);
                    if (nx == nullptr) break;
                    const int m = a == 1 ? std::max(1, std::min(kAhead1, nx->k_exp + ahead_extra)) : std::min(kPicks, nx->k_exp);
                    float* gl = d.glog_a + (size_t) (a - 1) * kWin * 1024;
                    gemv_f32(nx->gate_w, d.x, nx->n_exp, D, T, gl, 1024, s);   // (normed with the next layer's FFN
                                                                             // weights instead: the same hit rate)
                    route_ahead(gl, nx->gate_bias, nx->n_exp, m, T, reqs + slot, (a - 1) * kAhead1, s);
                    na[a - 1] = m;
                }
                route_publish(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, l.vram_tab, d.x, D, d.ridx, d.rw,
                              d.plan_ptr, lead ? reqs + slot : nullptr, sq, s, l.gate_bias_vl, l.mtp ? 0u : wp.vl, na[0],
                              na[1]);
            }
            if (!mv(d, {job(l.s1, d.x, d.sg, T, D, FF), job(l.s3, d.x, d.su, T, D, FF)}, err)) return false;
            swiglu(d.sg, d.su, T * FF, g.lim, d.sh, s);
            if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, FF, D)}, err)) return false;
            prof_mark(d, 2);
            const x3::ExpertLayout& lay = nm > 1 ? l.hlay : l.lay;
            if (T == 1 && moe_token_path) {
                x3::MoeArgs ma;
                ma.plan_ptr = d.plan_ptr;
                ma.plan_w = d.rw;
                ma.cpu_part = lead && !skip ? resps[slot].part[0] : nullptr;
                ma.cpu_flag = lead && !skip ? &resps[slot].cpu[0] : nullptr;
                ma.lay = lay;
                ma.k = l.k_exp;
                ma.n_embd = D;
                ma.n_ff = FF;
                ma.limit = g.lim;
                const bool mx = lay.fmt == gg::kFmtMXFP4;
                if (zc_stage_buf != nullptr && l.zc_tab != nullptr)   // (the store's experts copied to VRAM at the PCIe rate first)
                    zc_stage(d.plan_ptr, d.ridx, l.zc_tab, l.k_exp, zc_stage_buf, zc_stride, l.blob, s);
                if (mx) gg::moe_gate_up(ma, d.x, d.xg, s);
                else x3::moe_gate_up(ma, d.x, d.xg, s);
                // U9 tail: the device experts' down before the wait (it reads the plan, not the answer; DS4_PROF's
                // gate/up stretch then holds it)
                const bool early = mx && u9::on(u9::kTail);
                const bool down_ok = early && gg::moe_down_experts(ma, d.xg, s);   // (false: printed once, counted)
                prof_mark(d, 3);
                if (lead && !skip) wait_seq(&resps[slot].seq, sq, s);
                prof_mark(d, 4);
                if (early) {
                    if (down_ok) gg::moe_down_combine(ma, d.sh_out, d.moe_out, s);
                } else if (mx) gg::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
                else x3::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
                if (late && ep_verify) {   // (the deferred path's FFN output: the same call without the part - a GGUF's
                    x3::MoeArgs mv = ma;   //  combine of the partials the down above left, an EXL3's down again)
                    mv.cpu_part = nullptr;
                    mv.cpu_flag = nullptr;
                    if (mx) {
                        if (!early || down_ok) gg::moe_down_combine(mv, d.sh_out, d.v_moe, s);
                    } else x3::moe_down(mv, d.xg, d.sh_out, d.v_moe, s);
                }
            } else {   // each distinct expert read once for the window's tokens
                gf::MoeWin mw;
                mw.plan_ptr = d.plan_ptr;
                mw.T = T;
                mw.k = l.k_exp;
                mw.grp_ptr = d.grp_ptr;
                mw.grp_start = d.grp_start;
                mw.grp_ent = d.grp_ent;
                mw.n_groups = d.n_groups;
                gf::moe_win_group(mw, s);
                x3::WinArgs wa;
                wa.grp_ptr = d.grp_ptr;
                wa.grp_start = d.grp_start;
                wa.grp_ent = d.grp_ent;
                wa.n_groups = d.n_groups;
                wa.plan_ptr = d.plan_ptr;
                wa.plan_w = d.rw;
                wa.cpu_part = lead ? &resps[slot].part[0][0] : nullptr;
                wa.cpu_stride = kMaxDim;
                wa.cpu_flag = lead ? resps[slot].cpu : nullptr;
                wa.wait_ptr = lead ? &resps[slot].seq : nullptr;
                wa.wait_val = sq;
                wa.x = d.x;
                wa.x_stride = (size_t) D;
                wa.h = d.xg;
                wa.part = d.ed;
                wa.sh_out = d.sh_out;
                wa.out = d.moe_out;
                wa.out_stride = (size_t) D;
                wa.lay = lay;
                wa.T = T;
                wa.k = l.k_exp;
                wa.n_embd = D;
                wa.n_ff = FF;
                wa.limit = g.lim;
                if (!(lay.fmt == gg::kFmtMXFP4 ? gg::win_experts(wa, s) : x3::win_experts(wa, s))) {
                    err = "exl3 win_experts failed";
                    return false;
                }
            }
            if (lead) st.experts += (int64_t) T * l.k_exp;
            if (nm > 1) allreduce(d.moe_out, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
            if (hc2) CK(cudaStreamWaitEvent(s, d.ev_join, 0));
            if (late && ep_verify) hc_post(d.v_moe, d.h, d.f_post, d.f_comb, D, H, T, d.v_h, s);   // (the deferred streams)
            hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
            std::swap(d.h, d.h2);
            CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            if (late) {   // (f_post's own copy: the next layer's FFN hc_split writes f_post only after its late_flush
                          //  today; the copy keeps it so if that order changes)
                CK(cudaMemcpyAsync(d.late_post, d.f_post, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
                late_slot = slot;
                late_seq = sq;
                late_il = l.il;
                if (!ep_verify && !small_feed) ++st.ep_late;   // (the decode's layers: a small prompt's are not)
            }
            prof_mark(d, 5);
            return true;
        }
        if (nm > 1) {
            err = "a tensor-parallel layer on the host-routed path";
            return false;
        }
        // (glog rows are 1024 apart: route reads them so)
        gate_gemv(l, d.x, T, d.glog, s);
        VlBits vb = {};
        vb.w[0] = l.mtp ? 0u : wp.vl;
        route(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, d.ridx, d.rw, s, l.gate_bias_vl,
              vb.w[0] != 0 ? &vb : nullptr);
        CK(cudaMemcpyAsync(h_idx, d.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_w, d.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_x, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaEventRecord(d.ev, s));
        // the shared expert while the host looks at the picks
        if (!mv(d, {job(l.s1, d.x, d.sg, T, D, g.ff), job(l.s3, d.x, d.su, T, D, g.ff)}, err)) return false;
        swiglu(d.sg, d.su, T * g.ff, g.lim, d.sh, s);
        if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, g.ff, D)}, err)) return false;
        const double tw = now_ms();
        CK(cudaEventSynchronize(d.ev));
        st.gpu_wait_ms += now_ms() - tw;
        // the picks by expert: the VRAM ones' rows (gathered inputs, one GEMV per matrix for all their tokens), the
        // CPU lane's entries
        Combine cm;
        Gather ga;
        uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        int vex[kWin * kPicks], vrow0[kWin * kPicks], vn[kWin * kPicks];
        int nvex = 0, nrows = 0;
        std::vector<Ent> ents;
        std::vector<std::pair<int, float>> cpu_rows;   // slot -> (token, weight)
        std::vector<std::vector<std::pair<int, int>>> per((size_t) l.n_exp);   // expert -> (token, pick)
        for (int t = 0; t < T; ++t) {
            cm.n[t] = 0;
            cm.cpu[t] = 0;
            for (int i = 0; i < l.k_exp; ++i) {
                const int e = h_idx[t * kPicks + i];
                if (e < 0 || e >= l.n_exp) {
                    err = "router picked expert " + std::to_string(e);
                    return false;
                }
                ++pr[e];
                l.heat[(size_t) e] += 1.0f;
                if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                per[(size_t) e].push_back({t, i});
            }
        }
        int nslots = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            const auto& pe = per[(size_t) e];
            if (pe.empty()) continue;
            if (l.slot[(size_t) e] >= 0) {
                vex[nvex] = e;
                vrow0[nvex] = nrows;
                vn[nvex] = (int) pe.size();
                for (const auto& tp : pe) {
                    ga.src[nrows] = tp.first;
                    const int t = tp.first;
                    cm.row[t][cm.n[t]] = nrows;
                    cm.w[t][cm.n[t]] = h_w[t * kPicks + tp.second];
                    ++cm.n[t];
                    ++nrows;
                }
                ++nvex;
            } else {
                for (size_t a = 0; a < pe.size(); a += (size_t) ent_cap(l)) {
                    Ent en;
                    en.e = e;
                    en.slot0 = nslots;
                    en.n = (int) std::min<size_t>((size_t) ent_cap(l), pe.size() - a);
                    for (int i = 0; i < en.n; ++i) {
                        en.tok[i] = pe[a + (size_t) i].first;
                        cpu_rows.push_back({pe[a + (size_t) i].first, h_w[pe[a + (size_t) i].first * kPicks + pe[a + (size_t) i].second]});
                        cm.cpu[en.tok[i]] = 1;
                    }
                    nslots += en.n;
                    ents.push_back(en);
                }
            }
        }
        st.experts += (int64_t) T * l.k_exp;
        st.cpu_experts += nslots;
        if (nrows > 0) {
            gather_rows(d.x, ga, nrows, D, d.xe, s);
            x3::MvJob j[2 * kWin * kPicks];
            int nj = 0;
            for (int i = 0; i < nvex; ++i) {
                const auto& vm = l.vm[(size_t) vex[i]];
                j[nj++] = job(vm[0], d.xe + (size_t) vrow0[i] * D, d.xg + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
                j[nj++] = job(vm[1], d.xe + (size_t) vrow0[i] * D, d.xu + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
            }
            if (!mv_n(d, j, nj, err)) return false;
            swiglu(d.xg, d.xu, nrows * g.ff, g.lim, d.xhh, s);
            nj = 0;
            for (int i = 0; i < nvex; ++i)
                j[nj++] = job(l.vm[(size_t) vex[i]][2], d.xhh + (size_t) vrow0[i] * g.ff, d.ed + (size_t) vrow0[i] * D,
                              vn[i], g.ff, D);
            if (!mv_n(d, j, nj, err)) return false;
        }
        if (!ents.empty()) {
            const double tc = now_ms();
            int ff0, ffn;
            rx::ff_range(rx::host_range(rx_k < rx_nb, rx_k, rx_nb), rx_nb, g.ff, ff0, ffn);   // (whole: no request)
            cpu_experts(l, h_x, ents, 0, nullptr, ff0, ffn);
            std::fill(h_part, h_part + (size_t) T * D, 0.0f);
            for (int sl = 0; sl < nslots; ++sl) {
                float* o = h_part + (size_t) cpu_rows[(size_t) sl].first * D;
                const float wv = cpu_rows[(size_t) sl].second;
                const float* y = c_yd.data() + (size_t) sl * D;
                for (int e = 0; e < D; ++e) o[e] += wv * y[e];
            }
            st.cpu_ms += now_ms() - tc;
            CK(cudaMemcpyAsync(d.cpu_part, h_part, (size_t) T * D * 4, cudaMemcpyHostToDevice, s));
        }
        combine(d.ed, ents.empty() ? nullptr : d.cpu_part, d.sh_out, cm, T, D, d.moe_out, s);
        if (hc2) CK(cudaStreamWaitEvent(s, d.ev_join, 0));
        hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        // (h_part / h_x are written again only after the next layer's route has synchronized, which follows these)
        return true;
    }

    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::dsv41k::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        strata::dsv41k::sample_tokens(d.logits + (size_t) std::max(0, n_logits - 1) * g.vocab, 1, g.vocab, nullptr, 0,
                                       sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    bool logits(std::vector<float>& out) {
        if (logits_all) {
            out = chunk_logits;
            return true;
        }
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        out.resize((size_t) std::max(1, n_logits) * g.vocab);
        return cudaMemcpy(out.data(), d.logits, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void save_profile() const {
        if (o.profile.empty()) return;
        // (a layer range: its layers' rows into the profile as it is now - the other layers' are another process's,
        // which may share the file)
        std::vector<uint32_t> mine;
        if (ranged) {
            mine.assign((size_t) g.nl * g.n_exp, 0u);
            if (std::FILE* r = std::fopen(o.profile.c_str(), "rb")) {
                std::vector<uint32_t> b(1 << 16);
                mine.clear();
                for (size_t n; (n = std::fread(b.data(), 4, b.size(), r)) > 0;) mine.insert(mine.end(), b.begin(), b.begin() + (long) n);
                std::fclose(r);
                if (mine.size() < (size_t) g.nl * g.n_exp) mine.resize((size_t) g.nl * g.n_exp, 0u);
            }
            for (int il = lo; il < hi; ++il)
                std::copy(prof.begin() + (long) ((size_t) il * g.n_exp), prof.begin() + (long) ((size_t) (il + 1) * g.n_exp),
                          mine.begin() + (long) ((size_t) il * g.n_exp));
        }
        const std::vector<uint32_t>& out = ranged ? mine : prof;
        const std::string tmp = o.profile + ".tmp" + (ranged ? "." + std::to_string(lo) : std::string());
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) return;
        const bool ok = std::fwrite(out.data(), 4, out.size(), f) == out.size();
        std::fclose(f);
        // (the old profile replaced: std::rename does not replace an existing file on Windows, so there the profile
        // never moved on and only .tmp files were left)
        if (!ok || !os::replace_file(tmp, o.profile)) std::remove(tmp.c_str());
    }

    std::vector<std::string> info_lines() const {
        std::vector<std::string> v;
        int nh = 0;
        for (const Layer& l : L) nh += l.n_exp - l.n_vram;
        const int n_here = ranged ? (hi - lo) * g.n_exp : g.nl * g.n_exp;   // (a layer range: its layers' experts)
        char b[512];
        // kv=fp4: DeepSeek's own cache layout (the compressed KV and the index keys FP4, the window FP8)
        std::snprintf(b, sizeof b, "INFO context=%lld kv=fp4 kv_resident=%lld expert_slots=%d engine_kind=ds4 "
                                   "cpu_experts=%d ram_experts=%d engine=ds4-0.1",
                      (long long) o.max_context, (long long) o.max_context, n_here - nh, nh, ram_held);
        v.push_back(b);
        if (commit_kept > 0) {   // (Windows, an integrated GPU: the commit kept experts off the GPU - the server says so)
            std::snprintf(b, sizeof b,
                          "NOTE experts: %.0f GB stay off the GPU - Windows' page file is too small to hold them there; "
                          "a page file of %.0f GB keeps them on the GPU (System > About > Advanced system settings > "
                          "Performance > Advanced > Virtual memory), then restart Maya", commit_kept / 1e9, pagefile_hint);
            v.push_back(b);
            std::snprintf(b, sizeof b, "INFO gpu_commit_capped_gb=%.0f pagefile_hint_gb=%.0f", std::ceil(commit_kept / 1e9),
                          pagefile_hint);
            v.push_back(b);
        }
        if (pk.eg_layers > 0) {   // the n-gram memory's (Engram's) table rows a token reads, for the Monitor
            std::snprintf(b, sizeof b, "INFO engram_layers=%d engram_lookups=%d", pk.eg_layers,
                          pk.eg_layers * (pk.eg_ngram - 1) * pk.eg_heads);
            v.push_back(b);
        }
        if (img_tok >= 0) {   // the server's image resize (DeepSeek's image_processor.py)
            std::snprintf(b, sizeof b,
                          "INFO vision=ds4 vision_patch=%d vision_merge=%d vision_min_pixels=%lld vision_max_tokens=%d "
                          "vision_max_wh_ratio=%lld image_token=%d vision_gpu=%d vision_mib=%lld",
                          vg.patch, vg.merge, (long long) pk.num("vision.min_pixels", 0), vg.max_tokens,
                          (long long) pk.num("vision.max_wh_ratio", 0), img_tok, vt.device(),
                          (long long) (vt.device_bytes() >> 20));
            v.push_back(b);
        } else if (!vision_err.empty()) {   // (the server offers no pictures and says why)
            std::string why = vision_err;
            for (char& ch : why)
                if (ch == '\n' || ch == '\r') ch = ' ';
            v.push_back("NOTE images: OFF - the model's vision tower did not load (" + why + ")");
            v.push_back("INFO vision=off");
        }
        return v;
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;
bool Model::load(const Options& o, std::string& err) { return p_->load(o, err); }
bool Model::step(int token, bool logits, std::string& err) { return p_->step(token, logits, err); }
bool Model::feed(const int* tokens, int n, int logits_rows, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    p.halted = false;
    if (small_prompt(n, logits_rows, p.small_n, p.o.remote_stage || p.o.stage_worker)) {   // (DS4_SMALL_PROMPT: before
        const double t0 = now_ms();   // pick_path and the windows of 16 below chunk_min; pm.last stays -1)
        const bool ok = p.feed_small(tokens, n, logits_rows, err);
        if (ok) p.small_done(n, now_ms() - t0);
        if (logits_rows > 0) p.psettle.next();   // (as below)
        return ok;
    }
    if (p.chunk > 0 && n >= p.chunk_min) {   // a long run - or one the flow could take, if it is the faster path here
        const double t0 = now_ms();   // (every row's logits (tests): the chunked path, which keeps them all; a remote
                                      // stage's main process: chunks, where windows would cross the link 16 rows a time)
        const bool ok = logits_rows > kWin || p.o.remote_stage || p.pick_path(n)
                            ? p.forward_chunks(tokens, n, logits_rows, err)
                            : p.feed_windows(tokens, n, logits_rows, err);
        if (ok) p.path_done(now_ms() - t0);
        if (logits_rows > 0) p.psettle.next();   // (a feed that asks for logits ends a prompt: the next is another's)
        return ok;
    }
    const bool ok = p.feed_windows(tokens, n, logits_rows, err);
    if (logits_rows > 0) p.psettle.next();
    return ok;
}
bool Model::feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    p.halted = false;
    const double t0 = now_ms();
    if (small_prompt(n, logits_rows, p.small_n, p.o.remote_stage || p.o.stage_worker)) {   // (DS4_SMALL_PROMPT, as feed's:
        // the run but its last token in small windows, the checkpoint, then the last token as a window of 1)
        const bool ok = p.feed_small(tokens, n - 1, 0, err) &&
                        (p.checkpoint(slot), p.feed_small(tokens + n - 1, 1, logits_rows, err));
        if (ok) p.small_done(n, now_ms() - t0);
        if (logits_rows > 0) p.psettle.next();
        return ok;
    }
    if (n >= 2 && p.chunk > 0 && n >= p.chunk_min && p.o.remote_stage && p.ckpt_ok()) {
        // (a remote stage that keeps checkpoints: the worker takes its part of one at the position it is at, so the run
        // stops short of its last row - its chunks, the checkpoint on both sides, then the last row as a window)
        const bool ok = (n - 1 >= p.chunk_min ? p.forward_chunks(tokens, n - 1, 0, err)
                                              : p.feed_windows(tokens, n - 1, 0, err)) &&
                        (p.checkpoint(slot), p.feed_windows(tokens + n - 1, 1, logits_rows, err));
        if (ok) p.path_done(now_ms() - t0);
        if (logits_rows > 0) p.psettle.next();   // (as feed's)
        return ok;
    }
    if (n >= 2 && p.chunk > 0 && n >= p.chunk_min && (p.o.remote_stage || p.pick_path(n))) {   // (the last token rides along; its layers
        const bool ok = p.forward_chunks(tokens, n, logits_rows, err, slot);   // keep the state before it)
        if (ok) p.path_done(now_ms() - t0);
        if (logits_rows > 0) p.psettle.next();
        return ok;
    }
    if (n > 1 && !p.feed_windows(tokens, n - 1, 0, err)) return false;
    p.checkpoint(slot);
    const bool ok = n < 1 || p.feed_windows(tokens + n - 1, 1, logits_rows, err);
    if (ok) p.path_done(now_ms() - t0);   // (windows, when pick_path chose them)
    if (logits_rows > 0) p.psettle.next();
    return ok;
}
int Model::feed_unit() const { return p_->chunk > 0 ? std::max(p_->chunk, (int) p_->o.max_context) : kWin; }
void Model::set_progress(std::function<void(int64_t)> f) { p_->progress = std::move(f); }
void Model::set_cancel(std::function<bool()> f) { p_->cancel = std::move(f); }
bool Model::cancelled() const { return p_->halted; }
int Model::sample(float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter) {
    return p_->sample(t, tp, tk, mp, seed, counter);
}
bool Model::logits(std::vector<float>& out) { return p_->logits(out); }
int Model::image_token() const { return p_->img_tok; }
int Model::dim() const { return p_->g.dim; }
bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err) {
    return p_->encode_image(rgb, width, height, span, err);
}
void Model::set_images(std::vector<Image> images) {
    std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) { return a.start < b.start; });
    p_->imgs = std::move(images);
}
int Model::draft_block() const { return p_->g.n_mtp > 0 ? p_->g.d_block : 0; }
bool Model::draft(int y, std::vector<int>& d, std::vector<float>& c, std::string& err, float t, float tp, int tk,
                  float mp, uint64_t seed, uint64_t counter) {
    return p_->draft(y, d, c, err, t, tp, tk, mp, seed, counter);
}
bool Model::verify(int y, const std::vector<int>& d, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter,
                   int& accepted, int& next, std::string& err) {
    return p_->verify(y, d, t, tp, tk, mp, seed, counter, accepted, next, err);
}
int64_t Model::pos() const { return p_->pos; }
int64_t Model::max_context() const { return p_->o.max_context; }
int Model::vocab() const { return p_->g.vocab; }
const std::vector<int32_t>& Model::tokens() const { return p_->toks; }
void Model::reset() { p_->reset(); }
void Model::checkpoint(int slot) { p_->checkpoint(slot); }
bool Model::restore(int64_t limit) { return p_->restore(limit); }
void Model::pin() { p_->pin(); }
std::vector<int64_t> Model::check_positions() const { return p_->check_positions(); }
int64_t Model::restorable(int64_t limit) const { return p_->restorable(limit); }
uint64_t Model::slot_save(std::vector<uint8_t>& out, std::string& err, int64_t remote_id) {
    return p_->slot_save(out, err, remote_id);
}
bool Model::slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err, int64_t remote_id) {
    return p_->slot_load(data, bytes, n_pos, err, remote_id);
}
std::string Model::slot_identity() const { return p_->slot_identity(); }
bool Model::ranged() const { return p_->ranged; }
bool Model::stage_ckpt_on() const { return p_->linked() && p_->ckpt_ok(); }
void Model::set_stage_ckpt(StageCkpt c) { p_->stage_ck = std::move(c); }
Model::StageGeo Model::stage_geo() const {
    const Impl& p = *p_;
    StageGeo s;
    s.dim = p.g.dim;
    s.hc = p.g.hc;
    s.n_layers = p.g.nl;
    s.n_expert = p.g.n_exp;
    s.layer_begin = p.lo;
    s.layer_end = p.hi;
    s.max_t = kWin;
    s.handoff_floats = p.handoff_floats();
    s.full_floats = p.full_floats();
    s.topk = (int64_t) p.topk_stride();
    s.cand = (int64_t) p.cand_stride();
    s.rows_bf16 = p.bf16_rows();
    // (the link's buffers: made at load for a remote stage's chunks only)
    s.chunk = p.linked() ? (p.chunk_own >= 0 ? p.chunk_own : p.chunk) : 0;
    return s;
}
void Model::set_stage_link(StageRun run, std::function<bool(int n_keep, std::string& err)> commit,
                           std::function<bool(std::string& err)> reset) {
    p_->stage_hop = std::move(run);
    p_->stage_commit = std::move(commit);
    p_->stage_reset = std::move(reset);
}
float* Model::stage_rows_in() { return p_->bf16_rows() ? p_->wire_rx : p_->link_rx; }
float* Model::stage_rows_out() { return p_->bf16_rows() ? p_->wire_tx : p_->link_tx; }
Model::StageConv Model::stage_conv() const {
    StageConv c;
    c.in_ms = p_->conv_in_ms;
    c.out_ms = p_->conv_out_ms;
    c.in_n = p_->conv_in_n;
    c.out_n = p_->conv_out_n;
    return c;
}
bool Model::stage_run(const int32_t* tokens, int T, int64_t p0, std::string& err) {
    Impl& p = *p_;
    if (!p.o.stage_worker) {
        err = "not a remote stage's worker";
        return false;
    }
    if (p0 != p.pos) {   // (the main process resets the worker when it starts a conversation again)
        err = "out of step: this worker is at position " + std::to_string(p.pos) + ", the window starts at " +
              std::to_string(p0);
        return false;
    }
    if (T < 1 || T > kWin) {   // (before the rows are read: the link's bound, here too)
        err = "a window of " + std::to_string(T) + " tokens";
        return false;
    }
    std::vector<int> tk(tokens, tokens + T);
    const bool bf = p.bf16_rows();
    if (bf) {   // (the main process's compact rows into the layout the first layer's copy reads)
        const double t0 = now_ms();
        stage::rows_unpack(p.row_geo(), T, p.wire_rx, p.link_rx);
        p.conv_in_ms += now_ms() - t0;
        ++p.conv_in_n;
    }
    p.stage_t = 0;
    p.spec_win = T > 1;   // (a window of 2 or more rows may be a draft's check: its layers keep their backups)
    // (the link carries no window kind: no slices here, ep_slices_ok; its cache ticks keep serve_T = T - a stage worker is
    // not on the expert server's path)
    const bool ok = p.forward(tk.data(), T, 0, false, err);
    p.spec_win = false;
    if (!ok) return false;
    if (T > 1) p.stage_t = T;
    if (bf) {   // (forward ended with its last stream synchronized: the reply - normed rows, taps - is in link_tx)
        const double t0 = now_ms();
        stage::rows_pack(p.row_geo(), T, p.link_tx, p.wire_tx);
        p.conv_out_ms += now_ms() - t0;
        ++p.conv_out_n;
    }
    return true;
}
bool Model::stage_rollback(int n_keep, std::string& err) {
    Impl& p = *p_;
    const int T = p.stage_t;
    p.stage_t = 0;
    if (!p.o.stage_worker || T < 2 || n_keep < 1 || n_keep >= T) {
        err = "remote stage: " + std::to_string(n_keep) + " rows of the last window kept, which ran " + std::to_string(T) +
              " with backups (a window of 2 or more rows keeps them)";
        return false;
    }
    p.rollback(p.pos - T, T, n_keep);   // (as verify() keeps a draft's check in one process)
    return true;
}
void Model::set_stage_chunks(StageChunks c) { p_->stage_chunks = std::move(c); }
void Model::set_stage_chunk(int n) {
    Impl& p = *p_;
    if (!p.o.remote_stage) return;
    if (p.chunk_own < 0) p.chunk_own = p.chunk;   // (the first link: what the load left - the buffers' rows)
    p.chunk = (int) stage::chunk_sent(p.chunk_own, n);   // (each link again: its worker's, never more than the load's)
}
float* Model::stage_chunk_rows(int i) { return i >= 0 && i < 4 ? p_->link_pf[i] : nullptr; }
bool Model::stage_prefill(const int64_t* tokens, int T, int64_t p0, bool last, int skip, const float* in, float* out,
                          std::string& err) {
    Impl& p = *p_;
    if (!p.o.stage_worker || p.chunk <= 0) {
        err = p.o.stage_worker ? "this worker reads windows only (no chunked prompt path)" : "not a remote stage's worker";
        return false;
    }
    if (T < 1 || T > p.chunk) {
        err = "a prompt chunk of " + std::to_string(T) + " tokens: this worker takes 1 to " + std::to_string(p.chunk);
        return false;
    }
    if (p0 != p.pos) {   // (the main process resets the worker when it starts a conversation again)
        err = "out of step: this worker is at position " + std::to_string(p.pos) + ", the chunk starts at " +
              std::to_string(p0);
        return false;
    }
    const bool bf = p.bf16_rows();
    if (bf && (p.link_st[0] == nullptr || p.link_st[1] == nullptr)) {
        err = "remote stage: a bf16 prompt chunk with no fp32 rows to run it on";
        return false;
    }
    std::vector<int> tk(tokens, tokens + T);
    if (bf) stage::rows_unpack(p.row_geo(), T, in, p.link_st[0]);   // (the compact rows into the layout the copies read)
    p.stage_t = 0;   // (a chunk is kept whole: no backups, and no Commit undoes a window before it)
    p.stage_in = bf ? p.link_st[0] : in;
    p.stage_out = bf ? p.link_st[1] : out;
    p.stage_skip = skip;
    p.stage_last = last;
    p.halted = false;
    const bool ok = p.forward_chunks(tk.data(), T, 0, err);   // (one chunk: the run's, chunk after chunk on this side)
    p.stage_in = nullptr;
    p.stage_out = nullptr;
    p.stage_last = true;
    // (forward_chunks returned with the reply's rows copied in: they go out compact, from skip on)
    if (ok && bf) stage::reply_rows_pack(p.row_geo(), T, skip, p.link_st[1], out);
    return ok;
}
void Model::save_profile() const { p_->save_profile(); }
const Stats& Model::stats() const {
    Stats& st = p_->st;
    if (p_->store) {   // (the store keeps its own counters: copied in when asked)
        const ExpertStore& x = *p_->store;
        st.store_hits = x.stats().hits.load();
        st.store_misses = x.stats().misses.load();
        st.store_prefetched = x.stats().prefetched.load();
        st.store_pf_used = x.stats().prefetch_used.load();
        st.store_pf_wasted = x.stats().prefetch_wasted.load();
        st.store_evictions = x.stats().evictions.load();
        st.store_bytes = x.stats().bytes_read.load();
        st.store_fails = x.stats().fails.load();
        st.store_wait_ms = x.stats().wait_ns.load() / 1e6;
        st.store_busy_ms = x.busy_ms();
        st.store_zc_visible = (int64_t) x.visible_count();
        st.store_slots = (int64_t) x.slots_allocated();
        st.store_slots_max = (int64_t) x.slots_max();
    }
    return st;
}
std::string Model::describe() const { return p_->describe(); }
std::string Model::rx_lines(const Stats& s0, double nt) const { return p_->rx_lines(s0, nt); }
bool Model::rx_held_save(const std::string& path, std::string& note) const { return p_->rx_held_save(path, note); }
std::vector<std::string> Model::info_lines() const { return p_->info_lines(); }

}  // namespace strata::ds4
