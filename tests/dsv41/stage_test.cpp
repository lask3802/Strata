// The DeepSeek engine's remote stage without a GPU (src/dsv41/ds4_stage.hpp): the pack fingerprint both hellos carry
// (the same model on two PCs whose model_dir differs; any other line of ds4.txt or tensors.txt changes it), every
// main-process hello a worker refuses and the worker's hello a main process refuses, the reply's layout (the normed
// rows, then the drafter's taps: which ones the main process takes), what a Commit keeps; the compact bf16 rows
// (STRATA_REMOTE_ROWS=bf16): the picks exact, the streams rounded as the link's bf16 rounds, the hello refusing a
// mismatch - over a real link on the loopback; and, over the loopback through src/net/stage_link.cpp, a main
// process's draft checks committed in part on a worker that runs ds4_main.cpp's Run and Commit (worker_run,
// worker_commit) over a stand-in model keeping a history, undone past what a Commit keeps (each reply must be what the
// kept tokens alone give); bf16 rows and prompt chunks never together, a chunk never undone by a Commit; and prompt
// chunks over a loopback link - the worker's buffers as ds4_main.cpp lends them, the main process's ChunkLink sending
// ahead of the replies, the rows block by block there and row by row back, the replies' skip, the last chunk's flag, a
// chunk too large for the worker (the worker's handler stands in for Model::stage_prefill: it checks and turns the
// rows as the layout says, no layers).
//   ds4_stage_test            exit 0: all passed
#include "ds4_stage.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace st = strata::ds4::stage;
namespace net = strata::net;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);               \
            ++fails;                                                       \
        }                                                                  \
    } while (0)

static void put(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
}

// ---- the compact rows
static uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static float float_of(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// fp32 to bf16 and back, round to nearest even (what the link's conversion does), finite values only
static float bf16_round(float f) {
    uint32_t u = bits_of(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return float_of(u & 0xffff0000u);
}

// T rows in the fp32 block layout: the streams and pre (finite floats of every size), then the picks and the
// candidates as int32 words that include every bit pattern a float conversion would change
static std::vector<float> make_rows(const st::RowGeo& g, int T, std::mt19937& rng) {
    std::vector<float> r((size_t) T * (size_t) g.full());
    const size_t nf = (size_t) T * (size_t) (g.h + g.pre), ni = (size_t) T * (size_t) (g.topk + g.cand);
    std::normal_distribution<float> nd(0.f, 3.f);
    for (size_t i = 0; i < nf; ++i) r[i] = i % 7 == 0 ? nd(rng) * 1e-6f : i % 11 == 0 ? nd(rng) * 1e4f : nd(rng);
    const uint32_t special[] = {0u, 1u, 2u, 16777217u, 123457u, 0x7fffffffu, 0x80000000u, 0xffffffffu,
                                0x7fc00001u, 0x7f800001u, 0x7f800000u, 0xff800000u, 511u, 2047u, 65537u, 0x3f800001u};
    for (size_t i = 0; i < ni; ++i)
        r[nf + i] = float_of(i % 3 == 0 ? special[(i / 3) % (sizeof special / sizeof *special)] : (uint32_t) rng());
    return r;
}

// a worker for the loopback checks: its "layers" add 1 to every stream value; the picks go through untouched
struct Loop {
    st::RowGeo g;
    std::vector<float> in, out;
    std::atomic<bool> stop{false};
    std::atomic<int> served{-1};
    net::StageStats stats;
    std::thread t;
    std::string at;
    bool worker_bf16 = false;
    net::StageHello hello_of(bool bf16) const {
        net::StageHello h;
        h.n_embd = 64;
        h.hc = 4;
        h.n_layers = 43;
        h.n_expert = 384;
        h.layer_begin = 20;
        h.max_t = 16;
        h.kv_type = -1;
        h.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem | st::kRowsNormed |
                        (bf16 ? st::kRowsBf16Cap : 0u);
        h.max_context = 8192;
        h.handoff_floats = bf16 ? g.compact() : g.full();
        h.pack_hash = 0x1234;
        return h;
    }
    bool start(bool bf16) {
        worker_bf16 = bf16;
        const size_t w = (size_t) (bf16 ? g.compact() : g.full());
        in.assign(16 * w, 0.f);
        out.assign(16 * w, 0.f);
        net::StageHandlers hs;
        const net::StageHello me = hello_of(bf16);
        hs.hello = [me](const net::StageHello& theirs, net::StageHello& mine, std::string& e) {
            mine = me;
            if (theirs.protocol != net::kStageProtocol) return true;
            return st::worker_accepts(theirs, me, 8192, e);
        };
        hs.run = [this](int T, const int32_t*, int64_t, std::string&) {
            const size_t nf = (size_t) T * (size_t) (g.h + g.pre);
            std::vector<float> full((size_t) T * (size_t) g.full());
            if (worker_bf16) st::rows_unpack(g, T, in.data(), full.data());
            else std::memcpy(full.data(), in.data(), full.size() * 4);
            for (size_t i = 0; i < nf; ++i) full[i] += 1.f;
            if (worker_bf16) st::rows_pack(g, T, full.data(), out.data());
            else std::memcpy(out.data(), full.data(), full.size() * 4);
            return true;
        };
        hs.commit = [](int, std::string&) { return true; };
        hs.prefill = [](const int64_t*, int64_t, int64_t, int64_t, int64_t, const float*, float*, std::string& e) {
            e = "no chunks";
            return false;
        };
        hs.reset = [](const std::vector<int64_t>&, std::string&) { return true; };
        hs.disconnected = [] {};
        net::StageBuffers bufs;
        bufs.run_in = in.data();
        bufs.run_out = out.data();
        bufs.run_floats = 16 * w;
        bufs.handoff_floats = (int64_t) w;
        bufs.pf_row_floats = 256;
        std::mt19937 rng(std::random_device{}());
        for (int attempt = 0; attempt < 8 && at.empty(); ++attempt) {
            const int port = 20000 + (int) (rng() % 40000);
            served = -1;
            t = std::thread([this, port, hs, bufs] { served = net::serve_stage("127.0.0.1", port, "", hs, bufs, stats, &stop); });
            const std::string where = "127.0.0.1:" + std::to_string(port);
            for (int i = 0; i < 100 && served.load() < 0; ++i) {   // (until it listens: its own refusal of a protocol 4 hello)
                net::StageHello bad = hello_of(bf16), peer;
                bad.protocol = 4;
                std::string e;
                net::StageClient c;
                if (c.connect(where, bad, peer, e) || e.find("the worker refused:") != std::string::npos) {
                    at = where;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (at.empty()) end();
        }
        return !at.empty();
    }
    void end() {
        if (!t.joinable()) return;
        stop = true;
        t.join();
        stop = false;
    }
    ~Loop() { end(); }
};

// one window over the link, from a main process of one row format to the worker (of the same or the other): true when
// the link took it (`err`: why not); `back` the rows the main process reads, the picks included, in the fp32 layout
static bool link_window(Loop& w, bool main_bf16, int T, const std::vector<float>& rows, std::vector<float>& back,
                        std::string& err) {
    net::StageClient c;
    net::StageHello peer;
    if (!c.connect(w.at, w.hello_of(main_bf16), peer, err, 10)) return false;
    const st::RowGeo& g = w.g;
    const size_t wd = (size_t) (main_bf16 ? g.compact() : g.full());
    std::vector<float> send(wd * (size_t) T), recv(wd * (size_t) T);
    if (main_bf16) st::rows_pack(g, T, rows.data(), send.data());
    else send = rows;
    std::vector<int32_t> tok((size_t) T, 1);
    if (!c.run(T, tok.data(), 0, send.data(), send.size(), recv.data(), recv.size(), err)) return false;
    back.assign((size_t) T * (size_t) g.full(), 0.f);
    if (main_bf16) st::rows_unpack(g, T, recv.data(), back.data());
    else back = recv;
    return c.rows() == net::StageRows::F32;   // (the link's own format stays fp32 either way)
}

// the reply's layout: the real geometry fits, a target the main process holds keeps its own slot
static void replies() {
    const int64_t ds41 = 4 * 5120;   // (a row's streams, hc x dim: ds4.txt's)
    CHECK(st::reply_fits(5120, 3, ds41));
    CHECK(!st::reply_fits(5120, 4, ds41));
    CHECK(!st::reply_fits(5120, 3, ds41 - 1));
    CHECK(st::reply_fits(5120, 0, 5120) && !st::reply_fits(5120, 1, 2 * 5120 - 1));
    const int T = 5, dim = 4, nt = 3;
    const std::vector<int> targets = {37, 38, 39};
    CHECK(st::reply_tap_at(T, dim, nt, 0, 0) == (size_t) T * dim);   // (after the T normed rows)
    CHECK(st::reply_tap_at(T, dim, nt, 1, 0) == st::reply_tap_at(T, dim, nt, 0, 2) + dim);   // (row after row)
    std::vector<float> reply((size_t) T * 64, -1.0f);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < nt; ++j)
            for (int i = 0; i < dim; ++i) reply[st::reply_tap_at(T, dim, nt, t, j) + (size_t) i] = (float) (t * 100 + j * 10 + i);
    for (int end : {20, 38, 40}) {   // the main process's layers end there: the targets from it on are the worker's
        std::vector<float> taps((size_t) T * nt * dim, -7.0f);
        st::taps_from_reply(reply.data(), T, dim, targets, end, taps.data());
        bool ok = true;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < nt; ++j)
                for (int i = 0; i < dim; ++i) {
                    const float v = taps[((size_t) t * nt + j) * dim + (size_t) i];
                    ok = ok && v == (targets[(size_t) j] >= end ? (float) (t * 100 + j * 10 + i) : -7.0f);
                }
        CHECK(ok);
    }
}

// what a Commit keeps of the worker's last window
static void commits() {
    std::string e;
    CHECK(st::commit_kind(1, 1, e) == st::Keep::All);
    CHECK(st::commit_kind(16, 16, e) == st::Keep::All);
    CHECK(st::commit_kind(1, 6, e) == st::Keep::Part);
    CHECK(st::commit_kind(5, 6, e) == st::Keep::Part);
    CHECK(st::commit_kind(7, 6, e) == st::Keep::Bad && e.find("of a window of 6") != std::string::npos);
    CHECK(st::commit_kind(0, 6, e) == st::Keep::Bad);
    CHECK(st::commit_kind(-1, 6, e) == st::Keep::Bad);
    CHECK(st::commit_kind(1, 0, e) == st::Keep::Bad && e.find("no window") != std::string::npos);   // (twice)
}

// A main process and a worker over the loopback: the worker's stand-in model keeps the tokens it ran (for its layers'
// state), every window of 2 or more rows undoable as Model::stage_rollback undoes one; its reply's normed rows and
// taps are a function of everything it holds up to each row.  The main process runs windows kept whole (a prompt, a
// plain step) and draft checks kept in part, committing as ds4_model.cu does; every reply must be what the tokens it
// kept give, as one process's would.
namespace {
constexpr int kDim = 8, kHc = 4, kNt = 3;
constexpr int64_t kHand = (int64_t) kHc * kDim + 8 + 32 + 32;   // (topk_stride / cand_stride 32 at least)

float sum_of(const std::vector<int32_t>& h, size_t n) {   // (a row's value: the history up to it)
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += (double) h[i] * (double) (i + 1);
    return (float) std::fmod(s, 65521.0);
}

// Model's stage_run / stage_rollback over a history, writing the reply into out
struct Held {
    std::vector<int32_t> held;
    std::vector<float>* out = nullptr;
    int stage_t = 0;
    bool stage_run(const int32_t* tokens, int T, int64_t p0, std::string& e) {
        stage_t = 0;
        if (p0 != (int64_t) held.size()) {
            e = "out of step: at " + std::to_string(held.size()) + ", the window at " + std::to_string(p0);
            return false;
        }
        for (int t = 0; t < T; ++t) held.push_back(tokens[t]);
        for (int t = 0; t < T; ++t) {   // the reply: normed rows, then each target's tap
            float* o = out->data();
            o[(size_t) t * kDim] = sum_of(held, (size_t) (p0 + t + 1));
            for (int j = 0; j < kNt; ++j) o[st::reply_tap_at(T, kDim, kNt, t, j)] = (float) (j + 1) * o[(size_t) t * kDim];
        }
        if (T > 1) stage_t = T;
        return true;
    }
    bool stage_rollback(int n_keep, std::string& e) {
        const int T = stage_t;
        stage_t = 0;
        if (T < 2 || n_keep < 1 || n_keep >= T) {
            e = "no window to keep " + std::to_string(n_keep) + " rows of";
            return false;
        }
        held.resize(held.size() - (size_t) (T - n_keep));
        return true;
    }
};

net::StageHello hello_of(int layer_begin) {
    net::StageHello h;
    h.n_embd = kDim;
    h.hc = kHc;
    h.n_layers = 40;
    h.n_expert = 384;
    h.layer_begin = layer_begin;
    h.max_t = 16;
    h.kv_type = -1;
    h.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem | st::kRowsNormed;
    h.max_context = 4096;
    h.handoff_floats = kHand;
    h.pack_hash = 0x5eed;
    return h;
}
}  // namespace

static void link_drafts() {
    std::vector<float> in((size_t) 16 * kHand), out((size_t) 16 * kHand);
    Held model;   // the worker's: the tokens its layers hold
    model.out = &out;
    int last_t = 0;
    net::StageHandlers hs;
    const net::StageHello me = hello_of(20);
    hs.hello = [&](const net::StageHello& theirs, net::StageHello& mine, std::string& e) {
        mine = me;
        return st::worker_accepts(theirs, me, 4096, e);
    };
    hs.run = [&](int T, const int32_t* tokens, int64_t p0, std::string& e) {
        return st::worker_run(model, tokens, T, p0, last_t, e);
    };
    hs.commit = [&](int n_keep, std::string& e) { return st::worker_commit(model, n_keep, last_t, e); };
    hs.prefill = [](const int64_t*, int64_t, int64_t, int64_t, int64_t, const float*, float*, std::string& e) {
        e = "no chunks";
        return false;
    };
    hs.reset = [&](const std::vector<int64_t>&, std::string&) {
        model.held.clear();
        last_t = 0;
        return true;
    };
    hs.disconnected = [] {};   // (ds4_main.cpp's resets: the history is read after the main process went away)
    net::StageBuffers bufs;
    bufs.run_in = in.data();
    bufs.run_out = out.data();
    bufs.run_floats = in.size();
    bufs.handoff_floats = kHand;
    bufs.pf_row_floats = (int64_t) kHc * kDim;
    net::StageStats wst;
    std::atomic<bool> stop{false};
    std::atomic<int> served{-1};
    std::thread th;
    std::unique_ptr<net::StageClient> cp;
    net::StageHello peer;
    std::string e;
    bool up = false;
    std::mt19937 rng(std::random_device{}());
    for (int attempt = 0; attempt < 8 && !up; ++attempt) {   // a free loopback port
        const int port = 20000 + (int) (rng() % 40000);
        served = -1;
        th = std::thread([&, port] { served = net::serve_stage("127.0.0.1", port, "", hs, bufs, wst, &stop); });
        for (int i = 0; i < 100 && served.load() < 0 && !up; ++i) {   // (until it listens)
            cp.reset(new net::StageClient);
            up = cp->connect("127.0.0.1:" + std::to_string(port), hello_of(20), peer, e);
            if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!up) {
            stop = true;
            th.join();
            stop = false;
        }
    }
    CHECK(up);
    if (!up) return;
    net::StageClient& c = *cp;
    CHECK(st::main_accepts(peer, hello_of(20), e));
    CHECK(c.reset({}, e));
    // the main process: its kept tokens, then each window - (tokens, rows it keeps)
    std::vector<int32_t> kept;
    std::vector<float> tx((size_t) 16 * kHand, 0.0f), rx((size_t) 16 * kHand);
    struct W {
        std::vector<int32_t> tk;
        int keep;
    };
    const std::vector<W> ws = {
        {{5, 9, 2, 7, 1, 3, 8, 8, 4, 6, 0, 2, 9, 9, 1, 5}, 16},   // a prompt window, whole
        {{11, 12, 13, 14, 15, 16}, 3},                         // a draft's check: y and 2 drafts stand
        {{21}, 1},                                             // a plain step
        {{31, 32, 33, 34, 35, 36}, 6},                         // every draft stands
        {{41, 42, 43, 44}, 1},                                 // none stands
        {{51, 52, 53, 54, 55, 56}, 5},
        {{61, 62}, 1},
    };
    bool replies_ok = true;
    for (const W& w : ws) {
        const int T = (int) w.tk.size();
        const int64_t p0 = (int64_t) kept.size();
        const size_t n = (size_t) T * (size_t) kHand;
        if (!c.run(T, w.tk.data(), p0, tx.data(), n, rx.data(), n, e)) {
            std::printf("FAIL window at %lld: %s\n", (long long) p0, e.c_str());
            ++fails;
            break;
        }
        std::vector<int32_t> h = kept;   // what one process holds through this window's rows
        h.insert(h.end(), w.tk.begin(), w.tk.end());
        std::vector<float> taps((size_t) T * kNt * kDim, 0.0f);
        st::taps_from_reply(rx.data(), T, kDim, {37, 38, 39}, 20, taps.data());
        for (int t = 0; t < T; ++t) {
            const float want = sum_of(h, (size_t) (p0 + t + 1));
            replies_ok = replies_ok && rx[(size_t) t * kDim] == want;
            for (int j = 0; j < kNt; ++j) replies_ok = replies_ok && taps[((size_t) t * kNt + j) * kDim] == (float) (j + 1) * want;
        }
        CHECK(c.commit(w.keep, e));
        kept.insert(kept.end(), w.tk.begin(), w.tk.begin() + w.keep);
    }
    CHECK(replies_ok);
    // a commit of more rows than the window had: the worker's error comes back on the next window
    {
        const int32_t tk[3] = {71, 72, 73};
        const size_t n = 3 * (size_t) kHand;
        CHECK(c.run(3, tk, (int64_t) kept.size(), tx.data(), n, rx.data(), n, e));
        CHECK(c.commit(4, e));
        CHECK(!c.run(1, tk, (int64_t) kept.size() + 3, tx.data(), (size_t) kHand, rx.data(), (size_t) kHand, e) &&
              e.find("of a window of 3") != std::string::npos);
    }
    c.close();
    stop = true;
    th.join();
    CHECK(model.held.size() >= kept.size() && std::equal(kept.begin(), kept.end(), model.held.begin()));
}

static bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// ---- prompt chunks over a loopback link: a geometry of hc 2 x dim 8 streams, topk_stride 4, cand_stride 4 (its own
// names: the draft checks' above are another geometry)
namespace chunked {
constexpr int kDim = 8, kHc = 2, kTopk = 4, kCand = 4, kHD = kHc * kDim;
constexpr int64_t kW = kHD + 8 + kTopk + kCand;   // a hand-off row's floats

static net::StageHello hello_of(int64_t chunk) {
    net::StageHello h;
    h.n_embd = kDim;
    h.hc = kHc;
    h.n_layers = 4;
    h.n_expert = 2;
    h.layer_begin = 2;
    h.max_t = 16;
    h.kv_type = -1;
    h.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem | st::kRowsNormed;
    h.max_context = 4096;
    h.chunk = chunk;
    h.handoff_floats = kW;
    h.pack_hash = 0x5eed;
    return h;
}

// row r of a chunk at p0, as the main process's last layer leaves it (block by block): its streams, pre, picks
static float h_of(int64_t p, int j) { return (float) p + 0.001f * (float) j; }
static void chunk_rows(float* rows, int64_t p0, int T) {
    float* h = rows;
    float* pre = h + (size_t) T * kHD;
    int32_t* topk = (int32_t*) (pre + (size_t) T * 8);
    int32_t* cand = topk + (size_t) T * kTopk;
    for (int r = 0; r < T; ++r) {
        for (int j = 0; j < kHD; ++j) h[(size_t) r * kHD + j] = h_of(p0 + r, j);
        for (int j = 0; j < 8; ++j) pre[(size_t) r * 8 + j] = -(float) (p0 + r) - (float) j;
        for (int j = 0; j < kTopk; ++j) topk[(size_t) r * kTopk + j] = (int32_t) (p0 + r) * 16 + j;
        for (int j = 0; j < kCand; ++j) cand[(size_t) r * kCand + j] = -(int32_t) (p0 + r) - 1000 * j;
    }
}

// the stand-in worker: Model::stage_prefill's checks and layout - the rows in block by block (the picks bit for bit),
// its "last layer" h + 1 and pre x 2, out row by row from skip on
struct FakeWorker {
    std::mutex mu;
    int64_t pos = 0;
    int64_t chunk = 0;
    bool rows_ok = true;
    std::vector<int> flags;   // each chunk's kChunkLast
    std::vector<std::vector<float>> run_in_out{std::vector<float>(16 * kW), std::vector<float>(16 * kW)};
    std::vector<std::vector<float>> pf;
    net::StageHello mine;
    net::StageHandlers hs;
    net::StageBuffers bufs;
    explicit FakeWorker(int64_t c) : chunk(c), mine(hello_of(c)) {
        pf.assign(4, std::vector<float>((size_t) std::max<int64_t>(c, 1) * kW, 0.0f));
        float* const p[4] = {pf[0].data(), pf[1].data(), pf[2].data(), pf[3].data()};
        bufs = st::worker_buffers(run_in_out[0].data(), run_in_out[1].data(), 16, p, chunk, kW, kW);
        hs.hello = [this](const net::StageHello& theirs, net::StageHello& m, std::string& e) {
            m = mine;
            if (theirs.protocol != net::kStageProtocol) return true;
            return st::worker_accepts(theirs, mine, mine.max_context, e);
        };
        hs.run = [](int, const int32_t*, int64_t, std::string& e) {
            e = "no windows here";
            return false;
        };
        hs.commit = [](int, std::string&) { return true; };
        hs.reset = [this](const std::vector<int64_t>&, std::string&) {
            std::lock_guard<std::mutex> lk(mu);
            pos = 0;
            return true;
        };
        hs.prefill = [this](const int64_t* tokens, int64_t T, int64_t p0, int64_t fl, int64_t skip, const float* in,
                            float* out, std::string& e) {
            std::lock_guard<std::mutex> lk(mu);
            if (p0 != pos) {
                e = "out of step";
                return false;
            }
            const float* h = in;
            const float* pre = h + (size_t) T * kHD;
            const int32_t* topk = (const int32_t*) (pre + (size_t) T * 8);
            const int32_t* cand = topk + (size_t) T * kTopk;
            for (int64_t r = 0; r < T; ++r) {
                rows_ok = rows_ok && tokens[r] == 100 + p0 + r && h[(size_t) r * kHD + 3] == h_of(p0 + r, 3) &&
                          pre[(size_t) r * 8 + 1] == -(float) (p0 + r) - 1.0f &&
                          topk[(size_t) r * kTopk + 2] == (int32_t) (p0 + r) * 16 + 2 &&
                          cand[(size_t) r * kCand + 3] == -(int32_t) (p0 + r) - 3000;
                if (r < skip) continue;
                for (int j = 0; j < kHD; ++j) out[(size_t) r * kW + j] = h[(size_t) r * kHD + j] + 1.0f;
                for (int j = 0; j < 8; ++j) out[(size_t) r * kW + kHD + j] = pre[(size_t) r * 8 + j] * 2.0f;
            }
            flags.push_back((fl & st::kChunkLast) != 0 ? 1 : 0);
            pos = p0 + T;
            return true;
        };
    }
};

// a worker on a free loopback port until stopped
struct Served {
    std::atomic<bool> stop{false};
    std::atomic<int> rc{-1};
    net::StageStats stats;
    std::thread t;
    std::string at;
    bool start(FakeWorker& w) {
        std::mt19937 rng(std::random_device{}());
        for (int attempt = 0; attempt < 8 && at.empty(); ++attempt) {
            const int port = 20000 + (int) (rng() % 40000);
            rc = -1;
            t = std::thread([&, port] { rc = net::serve_stage("127.0.0.1", port, "", w.hs, w.bufs, stats, &stop); });
            const std::string where = "127.0.0.1:" + std::to_string(port);
            for (int i = 0; i < 100 && rc.load() < 0; ++i) {   // (until it listens: its refusal of a protocol 4 hello)
                net::StageHello bad = hello_of(0), peer;
                bad.protocol = 4;
                std::string e;
                net::StageClient c;
                if (c.connect(where, bad, peer, e) || has(e, "the worker refused:")) {
                    at = where;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (at.empty()) end();
        }
        return !at.empty();
    }
    void end() {
        if (!t.joinable()) return;
        stop = true;
        t.join();
        stop = false;
    }
};

static void chunks_over_a_link() {
    // the hellos: the main process's chunks against the worker's
    {
        FakeWorker w(64);
        Served s;
        CHECK(s.start(w));
        if (s.at.empty()) return;
        std::string e;
        net::StageHello peer;
        {
            net::StageClient c;   // (larger chunks than the worker takes: refused there, said with both sizes)
            CHECK(!c.connect(s.at, hello_of(128), peer, e) && has(e, "the worker refused:") && has(e, "64 at most") &&
                  has(e, "128"));
        }
        {
            net::StageClient c;   // (windows only: any worker)
            CHECK(c.connect(s.at, hello_of(0), peer, e) && peer.chunk == 64);
        }
        net::StageClient c;
        net::StageHello mine = hello_of(64);
        CHECK(c.connect(s.at, mine, peer, e) && peer.chunk == 64 && st::main_accepts(peer, mine, e));
        CHECK(c.reset({}, e));
        st::ChunkLink link(c, kW);
        std::vector<float> a((size_t) 64 * kW), b((size_t) 64 * kW), rx((size_t) 100 * kW, 0.0f);
        std::vector<int32_t> tok(100);
        for (int i = 0; i < 100; ++i) tok[(size_t) i] = 100 + i;
        // three chunks of a run (40, 40, 20 rows), the second sent before the first's reply is read; only the last
        // chunk's last 3 rows come back
        chunk_rows(a.data(), 0, 40);
        CHECK(link.send(tok.data(), 40, 0, 0, a.data(), 40, e));
        chunk_rows(b.data(), 40, 40);
        CHECK(link.send(tok.data() + 40, 40, 40, 0, b.data(), 40, e));
        CHECK(link.recv(rx.data(), 40, 40, e));
        chunk_rows(a.data(), 80, 20);   // (a's send is out: the second send waited for it)
        CHECK(link.send(tok.data() + 80, 20, 80, st::kChunkLast, a.data(), 17, e));
        CHECK(link.recv(rx.data() + (size_t) 40 * kW, 40, 40, e));
        CHECK(link.recv(rx.data() + (size_t) 80 * kW, 20, 17, e));
        CHECK(link.wait(e));
        bool back = true, untouched = true;
        for (int r = 0; r < 100; ++r)
            for (int j = 0; j < kHD + 8; ++j) {
                const float v = rx[(size_t) r * kW + j];
                if (r < 97) untouched = untouched && v == 0.0f;   // (the skipped rows: nothing written)
                else if (j < kHD) back = back && v == h_of(r, j) + 1.0f;
                else back = back && v == (-(float) r - (float) (j - kHD)) * 2.0f;
            }
        CHECK(back);
        CHECK(untouched);
        {
            std::lock_guard<std::mutex> lk(w.mu);
            CHECK(w.rows_ok);
            CHECK(w.pos == 100);
            CHECK((w.flags == std::vector<int>{0, 0, 1}));
        }
        // a chunk larger than the worker takes: its error, and the link still in step (a reset, then a chunk of the
        // most it takes with every row back - every row's logits)
        std::vector<float> big((size_t) 65 * kW, 0.0f);
        std::vector<int32_t> tok65(65, 100);
        e.clear();
        const bool sent = link.send(tok65.data(), 65, 100, st::kChunkLast, big.data(), 65, e) && link.wait(e);
        CHECK(sent);
        CHECK(!link.recv(big.data(), 65, 65, e) && has(e, "bad prompt chunk size"));
        CHECK(c.reset({}, e));
        chunk_rows(a.data(), 0, 64);
        for (int i = 0; i < 64; ++i) tok[(size_t) i] = 100 + i;
        std::fill(rx.begin(), rx.end(), 0.0f);
        CHECK(link.send(tok.data(), 64, 0, st::kChunkLast, a.data(), 0, e) && link.recv(rx.data(), 64, 0, e));
        CHECK(rx[(size_t) 0 * kW + 5] == h_of(0, 5) + 1.0f && rx[(size_t) 63 * kW + kHD] == -63.0f * 2.0f);
        c.close();
        s.end();
    }
    // a worker that reads windows only: a main process's chunks refused there; the main process's own check of a
    // worker's smaller chunks
    {
        FakeWorker w(0);
        CHECK(w.bufs.pf_floats == 0 && w.bufs.pf_row_floats == kW && w.bufs.run_floats == (size_t) 16 * kW);
        Served s;
        CHECK(s.start(w));
        if (s.at.empty()) return;
        std::string e;
        net::StageHello peer;
        net::StageClient c;
        CHECK(!c.connect(s.at, hello_of(64), peer, e) && has(e, "the worker refused:") && has(e, "windows of 16 only"));
        s.end();
        net::StageHello worker = hello_of(32), mine = hello_of(64);
        CHECK(!st::main_accepts(worker, mine, e) && has(e, "refused") && has(e, "32 at most"));
        mine.chunk = 32;
        CHECK(st::main_accepts(worker, mine, e));
        mine.chunk = 0;
        CHECK(st::main_accepts(hello_of(0), mine, e));
    }
}
}  // namespace chunked

int main() {
    const fs::path root = fs::temp_directory_path() / ("ds4_stage_test_" + std::to_string((long long) std::time(nullptr)));
    const fs::path a = root / "a", b = root / "b";
    fs::create_directories(a);
    fs::create_directories(b);
    const std::string tensors = "file 0 model-00001.safetensors\nt layers.0.attn.wq_a.weight BF16 0 0 1024 32 16\n";
    const std::string geo = "hidden_size 4096\nnum_hidden_layers 43\nhc_mult 4\n";

    // the fingerprint: model_dir left out, wherever the line sits
    put(a / "tensors.txt", tensors);
    put(a / "ds4.txt", "model_dir /srv/models/ds4\n" + geo);
    put(b / "tensors.txt", tensors);
    put(b / "ds4.txt", geo + "model_dir D:\\models\\ds4\n");
    const uint64_t ha = st::pack_hash(a.string()), hb = st::pack_hash(b.string());
    CHECK(ha != 0);
    CHECK(ha == hb);
    put(b / "ds4.txt", geo);   // (no model_dir at all: the same model)
    CHECK(st::pack_hash(b.string()) == ha);
    put(b / "ds4.txt", "model_dir x\nhidden_size 4096\nnum_hidden_layers 42\nhc_mult 4\n");   // another geometry
    CHECK(st::pack_hash(b.string()) != ha);
    put(b / "ds4.txt", geo + "model_dirx y\n");   // (only the model_dir key itself is left out)
    CHECK(st::pack_hash(b.string()) != ha);
    put(b / "ds4.txt", geo);
    put(b / "tensors.txt", tensors + "t head.weight BF16 0 1024 64 4 8\n");   // another tensor index
    CHECK(st::pack_hash(b.string()) != ha);
    fs::remove(b / "tensors.txt");   // a missing index: not the same pack
    CHECK(st::pack_hash(b.string()) != ha);
    CHECK(st::pack_hash((root / "none").string()) == 0);

    // the worker's checks of a main process's hello
    net::StageHello mine, theirs;
    mine.max_t = theirs.max_t = 16;
    theirs.max_context = 8192;
    theirs.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem | st::kRowsNormed;
    std::string e;
    CHECK(st::worker_accepts(theirs, mine, 8192, e));
    {
        net::StageHello t = theirs;   // (an earlier build's: it would read the normed rows as the streams)
        t.row_formats &= ~st::kRowsNormed;
        CHECK(!st::worker_accepts(t, mine, 8192, e) && e.find("earlier build") != std::string::npos);
        CHECK(!st::main_accepts(t, theirs, e) && e.find("earlier build") != std::string::npos);
        CHECK(st::main_accepts(theirs, theirs, e));
    }
    {
        net::StageHello t = theirs;   // (a worker with no chunked path: windows only)
        t.chunk = 2048;
        CHECK(!st::worker_accepts(t, mine, 8192, e) && e.find("chunks") != std::string::npos &&
              e.find("windows of 16 only") != std::string::npos);
        net::StageHello w = mine;   // (one that takes chunks: up to its own)
        w.chunk = 2048;
        CHECK(st::worker_accepts(t, w, 8192, e));
        t.chunk = 1024;
        CHECK(st::worker_accepts(t, w, 8192, e));
        t.chunk = 4096;
        CHECK(!st::worker_accepts(t, w, 8192, e) && e.find("takes 2048 at most") != std::string::npos &&
              e.find("--prefill-chunk 2048") != std::string::npos);
        t.chunk = -1;
        CHECK(!st::worker_accepts(t, w, 8192, e));
    }
    {
        net::StageHello t = theirs;
        t.vision = 1;
        CHECK(!st::worker_accepts(t, mine, 8192, e));
    }
    {
        net::StageHello t = theirs;
        t.row_formats |= 1u << (uint32_t) net::StageRows::BF16;
        CHECK(!st::worker_accepts(t, mine, 8192, e) && e.find("fp32") != std::string::npos);
    }
    {
        net::StageHello t = theirs;
        t.ckpt_max = 3;
        CHECK(!st::worker_accepts(t, mine, 8192, e));
    }
    {
        net::StageHello t = theirs;
        t.max_context = 16384;
        CHECK(!st::worker_accepts(t, mine, 8192, e));
        t.max_context = 4096;   // (a smaller context: served)
        CHECK(st::worker_accepts(t, mine, 8192, e));
        t.max_t = 17;
        CHECK(!st::worker_accepts(t, mine, 8192, e));
    }

    // the compact rows' capability bit: both sides or neither
    {
        net::StageHello w = mine, t = theirs;
        w.row_formats |= st::kRowsBf16Cap;
        // a bf16 worker, an fp32 main process
        CHECK(!st::worker_accepts(t, w, 8192, e) && e.find("STRATA_REMOTE_ROWS=bf16") != std::string::npos);
        t.row_formats |= st::kRowsBf16Cap;
        CHECK(st::worker_accepts(t, w, 8192, e));
        // an fp32 worker, a bf16 main process
        CHECK(!st::worker_accepts(t, mine, 8192, e) && e.find("STRATA_REMOTE_ROWS=bf16") != std::string::npos);
        // (the link's own bf16 stays refused: it would round the picks)
        t.row_formats |= 1u << (uint32_t) net::StageRows::BF16;
        CHECK(!st::worker_accepts(t, w, 8192, e) && e.find("fp32") != std::string::npos);
    }

    // bf16 rows and prompt chunks: never both - the link's Prefill converts a chunk's whole rows (the picks with them),
    // so a bf16 main process sends chunk 0 (windows), and a worker refuses a bf16 hello with chunks even when it takes
    // chunks itself
    {
        net::StageHello w = mine, t = theirs;
        w.row_formats |= st::kRowsBf16Cap;
        w.chunk = 2048;
        t.row_formats |= st::kRowsBf16Cap;
        CHECK(st::worker_accepts(t, w, 8192, e));   // (chunk 0: windows)
        t.chunk = 1024;
        CHECK(!st::worker_accepts(t, w, 8192, e) && e.find("bf16") != std::string::npos &&
              e.find("fp32 only") != std::string::npos);
        t.row_formats &= ~st::kRowsBf16Cap;   // (fp32 rows: the same chunks taken)
        w.row_formats &= ~st::kRowsBf16Cap;
        CHECK(st::worker_accepts(t, w, 8192, e));
    }

    // the worker's buffers: a window's rows at the link's width (the compact one with bf16 rows), a chunk's at the
    // fp32 width - the chunks cross as fp32 only
    {
        st::RowGeo real;
        real.h = 4 * 5120;
        real.topk = 512;
        real.cand = 2048;
        std::vector<float> b0(4), b1(4), b2(4), b3(4);
        float* const pf[4] = {b0.data(), b1.data(), b2.data(), b3.data()};
        const net::StageBuffers b = st::worker_buffers(nullptr, nullptr, 16, pf, 1024, real.compact(), real.full());
        CHECK(b.handoff_floats == real.compact() && b.run_floats == (size_t) 16 * (size_t) real.compact());
        CHECK(b.pf_row_floats == real.full() && b.pf_floats == (size_t) 1024 * (size_t) real.full());
        const net::StageBuffers none = st::worker_buffers(nullptr, nullptr, 16, pf, 0, real.full(), real.full());
        CHECK(none.pf_floats == 0 && none.pf_in[0] == nullptr);
    }

    // a prompt chunk is never a draft's check: kept whole, no Commit undoes it or the window before it, whatever its
    // rows (more than a window's 16 here) - Prefill (worker_prefill) clears what a Commit would keep
    {
        struct Stand {
            int runs = 0, rollbacks = 0, prefills = 0;
            bool stage_run(const int32_t*, int, int64_t, std::string&) { return ++runs > 0; }
            bool stage_rollback(int, std::string&) { return ++rollbacks > 0; }
            bool stage_prefill(const int64_t*, int, int64_t, bool, int, const float*, float*, std::string&) {
                return ++prefills > 0;
            }
        } m;
        std::vector<int32_t> tw(4, 1);
        std::vector<int64_t> tc(64, 1);
        int last_t = 0;
        CHECK(st::worker_run(m, tw.data(), 4, 0, last_t, e) && last_t == 4);
        CHECK(st::worker_prefill(m, tc.data(), 64, 4, true, 64, nullptr, nullptr, last_t, e) && last_t == 0);
        CHECK(!st::worker_commit(m, 2, last_t, e) && m.rollbacks == 0);    // (the window before it: not undone now)
        CHECK(st::worker_run(m, tw.data(), 4, 68, last_t, e) && st::worker_commit(m, 4, last_t, e));
        CHECK(st::worker_prefill(m, tc.data(), 64, 72, true, 64, nullptr, nullptr, last_t, e));
        CHECK(!st::worker_commit(m, 64, last_t, e) && !st::worker_commit(m, 32, last_t, e) && m.rollbacks == 0);
        CHECK(m.runs == 2 && m.prefills == 2);
    }

    // the compact rows: the streams as the link's bf16 rounds them; pre and the picks exact
    {
        std::mt19937 rng(7);
        st::RowGeo g;
        g.h = 4 * 64;
        g.topk = 40;
        g.cand = 70;
        CHECK(g.compact_ok() && g.full() == 256 + 8 + 110 && g.compact() == 128 + 8 + 110);
        st::RowGeo odd = g;
        odd.h = 255;
        CHECK(!odd.compact_ok());
        st::RowGeo real;   // the real geometry: hidden 5120, hc 4, index_topk 512, candidate_topk_blocks 2048
        real.h = 4 * 5120;
        real.topk = 512;
        real.cand = 2048;
        CHECK(real.full() * 4 == 92192 && real.compact() * 4 == 51232);
        for (const int T : {1, 3, 16}) {
            const std::vector<float> rows = make_rows(g, T, rng);
            const size_t nh = (size_t) T * (size_t) g.h, np = (size_t) T * (size_t) g.pre, nf = nh + np,
                         ni = (size_t) T * (size_t) (g.topk + g.cand);
            std::vector<float> wire((size_t) T * (size_t) g.compact(), 0.f), back(rows.size(), 0.f);
            st::rows_pack(g, T, rows.data(), wire.data());
            st::rows_unpack(g, T, wire.data(), back.data());
            bool floats_ok = true, close = true, changed = false;
            for (size_t i = 0; i < nh; ++i) {
                floats_ok = floats_ok && bits_of(back[i]) == bits_of(bf16_round(rows[i]));
                close = close && std::fabs(back[i] - rows[i]) <= std::fabs(rows[i]) * 0.00391f;   // (2^-8)
                changed = changed || bits_of(back[i]) != bits_of(rows[i]);
            }
            CHECK(floats_ok);
            CHECK(close);
            CHECK(changed);   // (the streams do lose bits: the check above is not trivially true)
            CHECK(std::memcmp(back.data() + nh, rows.data() + nh, np * 4) == 0);        // pre, bit for bit
            CHECK(std::memcmp(back.data() + nf, rows.data() + nf, ni * 4) == 0);        // the picks, bit for bit
            CHECK(std::memcmp(wire.data() + nh / 2, rows.data() + nh, (np + ni) * 4) == 0);   // ... on the wire too
            // a reply's unpack (no ints) brings pre and leaves the picks' place alone
            const std::vector<float> marker(rows.size(), float_of(0xdeadbeefu));
            std::vector<float> keep = marker;
            st::rows_unpack(g, T, wire.data(), keep.data(), false);
            CHECK(std::memcmp(keep.data() + nh, rows.data() + nh, np * 4) == 0);
            CHECK(std::memcmp(keep.data() + nf, marker.data() + nf, ni * 4) == 0);
            // pre and the picks through the link's own bf16 would not survive: why the engine packs the rows itself
            std::vector<unsigned char> lossy((np + ni) * 2);
            std::vector<float> lossy_back(np + ni);
            net::stage_rows_encode(net::StageRows::BF16, rows.data() + nh, 1, np + ni, lossy.data());
            net::stage_rows_decode(net::StageRows::BF16, lossy.data(), 1, np + ni, lossy_back.data());
            CHECK(std::memcmp(lossy_back.data(), rows.data() + nh, np * 4) != 0);
            CHECK(std::memcmp(lossy_back.data() + np, rows.data() + nf, ni * 4) != 0);
        }
    }

    // over a real link (loopback): bf16 to bf16, fp32 to fp32, and the two mismatches refused by the hello
    {
        std::mt19937 rng(11);
        Loop w;
        w.g.h = 4 * 64;
        w.g.topk = 40;
        w.g.cand = 70;
        const int T = 16;
        const std::vector<float> rows = make_rows(w.g, T, rng);
        const size_t nh = (size_t) T * (size_t) w.g.h, nf = nh + (size_t) T * (size_t) w.g.pre,
                     ni = (size_t) T * (size_t) (w.g.topk + w.g.cand);
        for (const bool bf : {true, false}) {
            CHECK(w.start(bf));
            std::vector<float> back;
            std::string err;
            CHECK(link_window(w, bf, T, rows, back, err));
            if (!err.empty()) std::printf("  %s link: %s\n", bf ? "bf16" : "fp32", err.c_str());
            bool floats_ok = back.size() == rows.size();
            for (size_t i = 0; floats_ok && i < nf; ++i)   // (the worker's layers added 1; only the streams are rounded)
                floats_ok = bf && i < nh ? bits_of(back[i]) == bits_of(bf16_round(bf16_round(rows[i]) + 1.f))
                                         : bits_of(back[i]) == bits_of(rows[i] + 1.f);
            CHECK(floats_ok);
            CHECK(back.size() == rows.size() && std::memcmp(back.data() + nf, rows.data() + nf, ni * 4) == 0);
            // the other format against this worker: refused at the hello, with the way out
            std::vector<float> none;
            err.clear();
            CHECK(!link_window(w, !bf, T, rows, none, err) && err.find("STRATA_REMOTE_ROWS=bf16") != std::string::npos);
            w.end();
            w.at.clear();
        }
    }

    std::error_code ec;
    fs::remove_all(root, ec);
    replies();
    commits();
    link_drafts();
    chunked::chunks_over_a_link();
    if (fails == 0) std::printf("ds4_stage_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
