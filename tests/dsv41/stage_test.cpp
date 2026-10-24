// The DeepSeek engine's remote stage without a GPU (src/dsv41/ds4_stage.hpp): the pack fingerprint both hellos carry
// (the same model on two PCs whose model_dir differs, or whose files' lines end in CRLF on one; any other line of
// ds4.txt or tensors.txt changes it), every
// main-process hello a worker refuses and the worker's hello a main process refuses, the reply's layout (the normed
// rows, then the drafter's taps: which ones the main process takes), what a Commit keeps; the compact bf16 rows
// (STRATA_REMOTE_ROWS=bf16): the picks exact, the streams rounded as the link's bf16 rounds, the hello refusing a
// mismatch - over a real link on the loopback; and, over the loopback through src/net/stage_link.cpp, a main
// process's draft checks committed in part on a worker that runs ds4_main.cpp's Run and Commit (worker_run,
// worker_commit) over a stand-in model keeping a history, undone past what a Commit keeps (each reply must be what the
// kept tokens alone give); bf16 rows with prompt chunks taken, a chunk never undone by a Commit; and prompt chunks over
// a loopback link - the worker's buffers as ds4_main.cpp lends them, the main process's ChunkLink sending ahead of the
// replies, the rows block by block there and row by row back, the replies' skip, the last chunk's flag, a chunk too
// large for the worker (the worker's handler stands in for Model::stage_prefill: it checks and turns the rows as the
// layout says, no layers) - with fp32 rows and with the compact ones, packed at the link's width on both sides
// (ChunkLink; the stand-in unpacks and packs as Model::stage_prefill does): the picks and pre exact, the streams
// rounded as bf16 rounds them each way.
// And the checkpoints and slots the main process has its worker keep (STRATA_DS4_SLOT_KEEP): the ids CkptSave /
// CkptRestore carry (ck_save_op / ck_restore_op), the worker's store of its parts of slots (ds4_partstore.hpp: RAM cap,
// folder budget, keep[], a folder left by an earlier run), and the worker's handlers (worker_ck_save / worker_ck_restore,
// the engine's own) over the real link on loopback with a stand-in for the model: the worker's checkpoints follow the
// main process's by role and position, a slot is set aside and taken back whole, and what the worker does not hold
// (never saved, pruned, dropped for its bounds, gone with its connection, kept by no folder) comes back as "missing"
// with nothing changed - the main process then reads the conversation from token 0.
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
        h.row_formats = st::hello_rows(bf16);
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
    h.row_formats = st::hello_rows(false);
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
static st::RowGeo geo() {
    st::RowGeo g;
    g.h = kHD;
    g.topk = kTopk;
    g.cand = kCand;
    return g;
}
static const int64_t kC = geo().compact();   // ... the compact row's words

static net::StageHello hello_of(int64_t chunk, bool bf16 = false) {
    net::StageHello h;
    h.n_embd = kDim;
    h.hc = kHc;
    h.n_layers = 4;
    h.n_expert = 2;
    h.layer_begin = 2;
    h.max_t = 16;
    h.kv_type = -1;
    h.row_formats = st::hello_rows(bf16);
    h.max_context = 4096;
    h.chunk = chunk;
    h.handoff_floats = bf16 ? kC : kW;
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
// its "last layer" h + 1 and pre x 2, out row by row from skip on; with the compact rows (bf16) the link's buffers are
// compact() words a row, and the chunk runs on fp32 rows unpacked from them, its reply packed back (as stage_prefill)
struct FakeWorker {
    std::mutex mu;
    int64_t pos = 0;
    int64_t chunk = 0;
    bool bf16 = false;
    bool rows_ok = true;
    std::vector<int> flags;   // each chunk's kChunkLast
    std::vector<std::vector<float>> run_in_out{std::vector<float>(16 * kW), std::vector<float>(16 * kW)};
    std::vector<std::vector<float>> pf;
    std::vector<float> st_in, st_out;   // (bf16) the fp32 rows a chunk runs on
    net::StageHello mine;
    net::StageHandlers hs;
    net::StageBuffers bufs;
    explicit FakeWorker(int64_t c, bool compact = false) : chunk(c), bf16(compact), mine(hello_of(c, compact)) {
        const int64_t w = compact ? kC : kW;
        pf.assign(4, std::vector<float>((size_t) std::max<int64_t>(c, 1) * w, 0.0f));
        float* const p[4] = {pf[0].data(), pf[1].data(), pf[2].data(), pf[3].data()};
        bufs = st::worker_buffers(run_in_out[0].data(), run_in_out[1].data(), 16, p, chunk, w);
        st_in.assign((size_t) std::max<int64_t>(c, 1) * kW, 0.0f);
        st_out.assign(st_in.size(), 0.0f);
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
            float* const wire_out = out;
            if (bf16) {   // (the compact rows in: fp32 for the "layers", the reply packed below)
                st::rows_unpack(geo(), (int) T, in, st_in.data());
                in = st_in.data();
                out = st_out.data();
            }
            const float* h = in;
            const float* pre = h + (size_t) T * kHD;
            const int32_t* topk = (const int32_t*) (pre + (size_t) T * 8);
            const int32_t* cand = topk + (size_t) T * kTopk;
            for (int64_t r = 0; r < T; ++r) {
                const float h3 = bf16 ? bf16_round(h_of(p0 + r, 3)) : h_of(p0 + r, 3);   // (the streams rounded)
                rows_ok = rows_ok && tokens[r] == 100 + p0 + r && h[(size_t) r * kHD + 3] == h3 &&
                          pre[(size_t) r * 8 + 1] == -(float) (p0 + r) - 1.0f &&
                          topk[(size_t) r * kTopk + 2] == (int32_t) (p0 + r) * 16 + 2 &&
                          cand[(size_t) r * kCand + 3] == -(int32_t) (p0 + r) - 3000;
                if (r < skip) continue;
                for (int j = 0; j < kHD; ++j) out[(size_t) r * kW + j] = h[(size_t) r * kHD + j] + 1.0f;
                for (int j = 0; j < 8; ++j) out[(size_t) r * kW + kHD + j] = pre[(size_t) r * 8 + j] * 2.0f;
            }
            if (bf16) st::reply_rows_pack(geo(), (int) T, (int) skip, out, wire_out);
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
            net::StageClient c;   // (larger chunks than the worker takes: served - the main process sends the worker's)
            const net::StageHello big = hello_of(128);
            CHECK(c.connect(s.at, big, peer, e) && peer.chunk == 64 && st::main_accepts(peer, big, e) &&
                  st::chunk_sent(big.chunk, peer.chunk) == 64);
        }
        {
            net::StageClient c;   // (windows only: any worker)
            CHECK(c.connect(s.at, hello_of(0), peer, e) && peer.chunk == 64);
        }
        net::StageClient c;   // (a main process of 128-row chunks: what it sends below is the worker's 64 at most)
        net::StageHello mine = hello_of(128);
        CHECK(c.connect(s.at, mine, peer, e) && peer.chunk == 64 && st::main_accepts(peer, mine, e) &&
              st::chunk_sent(mine.chunk, peer.chunk) == 64);
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
        net::StageHello worker = hello_of(32), mine = hello_of(64);   // (smaller chunks there: this side sends those)
        CHECK(st::main_accepts(worker, mine, e) && st::chunk_sent(mine.chunk, worker.chunk) == 32);
        mine.chunk = 32;
        CHECK(st::main_accepts(worker, mine, e) && st::chunk_sent(mine.chunk, worker.chunk) == 32);
        mine.chunk = 16;
        CHECK(st::chunk_sent(mine.chunk, worker.chunk) == 16);
        CHECK(!st::main_accepts(hello_of(0), hello_of(64), e) && has(e, "refused") && has(e, "windows of 16 only"));
        mine.chunk = 0;
        CHECK(st::main_accepts(hello_of(0), mine, e) && st::chunk_sent(mine.chunk, 0) == 0);
        CHECK(st::chunk_sent(64, 0) == 0);
    }
}

// the same run of chunks with the compact rows: the link's buffers and messages compact() words a row, the main
// process's ChunkLink packing each chunk (two slots in turn) and unpacking the replies, the worker unpacking and packing
static void chunks_bf16_over_a_link() {
    FakeWorker w(64, true);
    CHECK(w.bufs.pf_row_floats == kC && w.bufs.pf_floats == (size_t) 64 * (size_t) kC && kC < kW);
    Served s;
    CHECK(s.start(w));
    if (s.at.empty()) return;
    std::string e;
    net::StageHello peer;
    {
        net::StageClient c;   // (fp32 rows against a bf16 worker: refused at the hello, chunks or not)
        CHECK(!c.connect(s.at, hello_of(64), peer, e) && has(e, "STRATA_REMOTE_ROWS"));
    }
    net::StageClient c;
    e.clear();
    const net::StageHello mine = hello_of(128, true);   // (bf16 rows and chunks together: taken)
    CHECK(c.connect(s.at, mine, peer, e) && peer.chunk == 64 && st::main_accepts(peer, mine, e) &&
          st::chunk_sent(mine.chunk, peer.chunk) == 64);
    if (!e.empty()) std::printf("  bf16 chunks: %s\n", e.c_str());
    CHECK(c.reset({}, e));
    st::ChunkLink link(c, geo(), true);
    std::vector<float> a((size_t) 64 * kW), b((size_t) 64 * kW), rx((size_t) 100 * kW, 0.0f);
    std::vector<int32_t> tok(100);
    for (int i = 0; i < 100; ++i) tok[(size_t) i] = 100 + i;
    // three chunks (40, 40, 20), the second out before the first's reply is read, the third written into the first's
    // buffer once the second's send returned; the last chunk's last 3 rows back
    chunk_rows(a.data(), 0, 40);
    CHECK(link.send(tok.data(), 40, 0, 0, a.data(), 40, e));
    chunk_rows(b.data(), 40, 40);
    CHECK(link.send(tok.data() + 40, 40, 40, 0, b.data(), 40, e));
    CHECK(link.recv(rx.data(), 40, 40, e));
    chunk_rows(a.data(), 80, 20);
    CHECK(link.send(tok.data() + 80, 20, 80, st::kChunkLast, a.data(), 17, e));
    CHECK(link.recv(rx.data() + (size_t) 40 * kW, 40, 40, e));
    CHECK(link.recv(rx.data() + (size_t) 80 * kW, 20, 17, e));
    CHECK(link.wait(e));
    bool back = true, untouched = true, rounded = false;
    for (int r = 0; r < 100; ++r)
        for (int j = 0; j < kHD + 8; ++j) {
            const float v = rx[(size_t) r * kW + j];
            if (r < 97) untouched = untouched && v == 0.0f;
            else if (j < kHD) {   // (rounded on the way there, the worker's +1, rounded on the way back)
                back = back && bits_of(v) == bits_of(bf16_round(bf16_round(h_of(r, j)) + 1.0f));
                rounded = rounded || v != h_of(r, j) + 1.0f;
            } else
                back = back && v == (-(float) r - (float) (j - kHD)) * 2.0f;   // (pre: exact)
        }
    CHECK(back);
    CHECK(untouched);
    CHECK(rounded);   // (the streams do lose bits: the check above is not the fp32 one)
    {
        std::lock_guard<std::mutex> lk(w.mu);
        CHECK(w.rows_ok);   // (the picks and pre exact, the streams bf16-rounded, every row of every chunk)
        CHECK(w.pos == 100);
        CHECK((w.flags == std::vector<int>{0, 0, 1}));
    }
    // every row back (every row's logits), a chunk of the most the worker takes
    CHECK(c.reset({}, e));
    chunk_rows(a.data(), 0, 64);
    std::fill(rx.begin(), rx.end(), 0.0f);
    CHECK(link.send(tok.data(), 64, 0, st::kChunkLast, a.data(), 0, e) && link.recv(rx.data(), 64, 0, e) && link.wait(e));
    CHECK(rx[(size_t) 0 * kW + 5] == bf16_round(bf16_round(h_of(0, 5)) + 1.0f) && rx[(size_t) 63 * kW + kHD] == -63.0f * 2.0f);
    c.close();
    s.end();
}
}  // namespace chunked

namespace slots {

// ---- checkpoints and slots across the link ----------------------------------------------------------------------------

// A stand-in for the Model on the worker's side of the handlers (ds4_stage.hpp's worker_ck_save / worker_ck_restore): the
// same checkpoints by role at positions, the same slot semantics (a conversation set aside once, until something changes
// it; loaded back at its position with the checkpoints it had), over tokens instead of layers.
struct FakeModel {
    int64_t p = 0;
    std::vector<int32_t> toks;
    int64_t ck[4] = {-1, -1, -1, -1};
    bool held = false;
    int64_t pos() const { return p; }
    void feed(const int32_t* t, int n) {
        toks.insert(toks.end(), t, t + n);
        p += n;
        held = false;
    }
    void reset() {
        p = 0;
        toks.clear();
        for (int64_t& c : ck) c = -1;
        held = false;
    }
    void checkpoint(int role) { ck[role & 3] = p; }
    void pin() {
        if (ck[1] > 0) ck[2] = ck[1];
    }
    int best(int64_t limit) const {
        int b = -1;
        for (int i = 0; i < 4; ++i)
            if (ck[i] >= 0 && ck[i] <= limit && ck[i] <= p && (b < 0 || ck[i] > ck[b])) b = i;
        return b;
    }
    int64_t restorable(int64_t limit) const {
        const int b = best(limit);
        return b < 0 ? -1 : ck[b];
    }
    bool restore(int64_t limit) {
        const int b = best(limit);
        if (b < 0) return false;
        p = ck[b];
        toks.resize((size_t) p);
        for (int64_t& c : ck)
            if (c > p) c = -1;
        held = false;
        return true;
    }
    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err) {
        if (held) {
            err = "it is already set aside";
            return 0;
        }
        if (p <= 0) {
            err = "there is no conversation";
            return 0;
        }
        out.clear();
        auto put = [&out](const void* src, size_t n) {
            const uint8_t* b = static_cast<const uint8_t*>(src);
            out.insert(out.end(), b, b + n);
        };
        put(&p, 8);
        put(ck, sizeof ck);
        put(toks.data(), toks.size() * 4);
        held = true;
        return out.size();
    }
    bool slot_load(const uint8_t* d, size_t n, int64_t n_pos, std::string& err) {
        int64_t q = 0;
        if (n < 8 + sizeof ck) {
            err = "the slot is cut short";
            return false;
        }
        std::memcpy(&q, d, 8);
        if (q != n_pos || n != 8 + sizeof ck + (size_t) q * 4) {
            err = "the conversation's position is not this prompt's";
            return false;
        }
        p = q;
        std::memcpy(ck, d + 8, sizeof ck);
        toks.resize((size_t) q);
        std::memcpy(toks.data(), d + 8 + sizeof ck, (size_t) q * 4);
        held = false;
        return true;
    }
    // a prompt chunk (Model::stage_prefill): read at the position the session is at, or refused
    bool stage_prefill(const int64_t* t, int T, int64_t p0, bool, int, const float*, float*, std::string& err) {
        if (p0 != p) {
            err = "a prompt chunk at " + std::to_string(p0) + ", this worker is at " + std::to_string(p);
            return false;
        }
        for (int i = 0; i < T; ++i) toks.push_back((int32_t) t[i]);
        p += T;
        held = false;
        return true;
    }
};

static net::StageHello link_hello(bool worker) {
    net::StageHello h;
    h.n_embd = 4;
    h.hc = 1;
    h.n_layers = 4;
    h.n_expert = 2;
    h.layer_begin = 2;
    h.max_t = 4;
    h.kv_type = -1;
    h.row_formats = st::hello_rows(false);   // (the reply's layout: the worker refuses a main without it)
    h.max_context = 64;
    h.handoff_floats = 4;
    h.pack_hash = 0x44533456;
    h.ckpt_max = worker ? 0 : st::kMainSlotsMax + 1;
    return h;
}

// a worker (serve_stage on loopback) whose handlers are the engine's own (worker_ck_save / worker_restore, and with
// `chunk` worker_prefill) over a FakeModel, and the main process's side of the link: windows of tokens through Run +
// Commit, prompt chunks of up to `chunk` tokens (0: windows only), and the checkpoint messages
struct LinkTest {
    FakeModel model;           // the worker's, behind mu
    std::mutex mu;
    st::PartStore parts;
    net::StageHandlers h;
    net::StageBuffers buf;
    std::vector<float> rin, rout;
    int chunk = 0, last_t = 0;   // (last_t: the worker's handler's, as ds4_main.cpp keeps it)
    std::vector<float> pf_mem;
    net::StageStats stats;
    std::atomic<bool> stop{false};
    std::atomic<int> served{-1};
    std::thread t;
    std::string where, err;
    net::StageClient c;
    net::StageHello mine = link_hello(false), peer;

    LinkTest(const std::string& dir, size_t ram, uint64_t disk, int chunk_rows = 0)
        : parts(dir, ram, disk), rin(16), rout(16), chunk(chunk_rows) {
        mine.chunk = chunk;
        h.hello = [this](const net::StageHello& theirs, net::StageHello& m, std::string& e) {
            m = link_hello(true);
            m.chunk = chunk;
            return st::worker_accepts(theirs, m, 64, e);
        };
        h.run = [this](int T, const int32_t* tokens, int64_t, std::string&) {
            std::lock_guard<std::mutex> lk(mu);
            model.feed(tokens, T);
            return true;
        };
        h.commit = [](int, std::string&) { return true; };
        h.prefill = [this](const int64_t* tokens, int64_t T, int64_t p0, int64_t flags, int64_t skip, const float* in,
                           float* out, std::string& e) {
            std::lock_guard<std::mutex> lk(mu);
            return st::worker_prefill(model, tokens, (int) T, p0, (flags & st::kChunkLast) != 0, (int) skip, in, out,
                                      last_t, e);
        };
        h.reset = [this](const std::vector<int64_t>& keep, std::string&) {
            std::lock_guard<std::mutex> lk(mu);
            st::worker_reset(model, parts, keep);
            return true;
        };
        h.disconnected = [this] {
            std::lock_guard<std::mutex> lk(mu);
            st::worker_disconnected(model, parts);
        };
        // (ds4_main.cpp's handlers, each under mu here: the test reads the stand-in from its own thread)
        st::worker_ckpt_handlers(model, parts, last_t, h, nullptr);
        const auto save = h.ckpt_save;
        const auto restore = h.ckpt_restore;
        h.ckpt_save = [this, save](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e) -> int {
            std::lock_guard<std::mutex> lk(mu);
            return save(id, L, keep, e);
        };
        h.ckpt_restore = [this, restore](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e) {
            std::lock_guard<std::mutex> lk(mu);
            return restore(id, L, keep, e);
        };
        pf_mem.resize((size_t) 4 * (size_t) std::max(chunk, 1) * 4);
        float* const pf[4] = {pf_mem.data(), pf_mem.data() + pf_mem.size() / 4, pf_mem.data() + pf_mem.size() / 2,
                              pf_mem.data() + pf_mem.size() / 4 * 3};
        buf = st::worker_buffers(rin.data(), rout.data(), 4, pf, chunk, 4);   // (chunk 0: windows only)
    }
    ~LinkTest() { shutdown(); }
    bool start() {
        for (int port = 23100; port < 23140 && where.empty(); ++port) {
            served = -1;
            t = std::thread([&, port] { served = net::serve_stage("127.0.0.1", port, "", h, buf, stats, &stop); });
            const std::string at = "127.0.0.1:" + std::to_string(port);
            for (int i = 0; i < 100 && served.load() < 0 && where.empty(); ++i) {   // (until it listens)
                net::StageHello bad = mine, pr;
                bad.protocol = 4;
                std::string e;
                net::StageClient probe;
                if (probe.connect(at, bad, pr, e) || e.find("the worker refused:") != std::string::npos) where = at;
                else std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (where.empty()) {
                stop = true;
                t.join();
                stop = false;
            }
        }
        return !where.empty() && connect();
    }
    bool connect() {
        err.clear();
        c.close();
        return c.connect(where, mine, peer, err, 10);
    }
    void shutdown() {
        c.close();
        if (t.joinable()) {
            stop = true;
            t.join();
        }
    }
    // a window of n tokens (first, first + 1, ...) at p0, committed whole
    bool window(int64_t p0, int n, int first) {
        int32_t tk[4];
        for (int i = 0; i < n; ++i) tk[i] = first + i;
        std::vector<float> in((size_t) n * 4, 1.0f), out((size_t) n * 4);
        return c.run(n, tk, p0, in.data(), in.size(), out.data(), out.size(), err) && c.commit(n, err);
    }
    // n tokens (first, first + 1, ...) in windows of four from p0
    bool feed(int64_t p0, int n, int first = 0) {
        for (int done = 0; done < n; done += 4)
            if (!window(p0 + done, std::min(4, n - done), first + done)) return false;
        return true;
    }
    FakeModel snap() {
        std::lock_guard<std::mutex> lk(mu);
        return model;
    }
    bool has(int64_t id) {
        std::lock_guard<std::mutex> lk(mu);
        return parts.has(id);
    }
    bool save(int64_t id, int64_t L, const std::vector<int64_t>& keep) {
        bool stored = false;
        return c.ckpt_save(id, L, keep, stored, err) && stored;
    }
    // restore: 1 done, 0 missing (nothing changed), -1 an error
    int restore(int64_t id, int64_t L, const std::vector<int64_t>& keep) {
        bool missing = false;
        if (c.ckpt_restore(id, L, keep, missing, err)) return 1;
        return missing ? 0 : -1;
    }
};

static void test_ids() {
    int role = -1;
    CHECK(st::ck_save_op(st::kCkRole0, &role) == st::CkOp::Role && role == 0);
    CHECK(st::ck_save_op(st::kCkRole0 + 3, &role) == st::CkOp::Role && role == 3);
    CHECK(st::ck_save_op(st::kCkRole0 + 4) == st::CkOp::Pin && st::ck_save_op(st::kCkPin) == st::CkOp::Pin);
    CHECK(st::ck_save_op(0) == st::CkOp::Bad && st::ck_save_op(-5) == st::CkOp::Bad);
    CHECK(st::ck_save_op(st::kCkAt) == st::CkOp::Bad && st::ck_save_op(st::kSlotId0 - 1) == st::CkOp::Bad);
    CHECK(st::ck_save_op(st::kSlotId0) == st::CkOp::Slot && st::ck_save_op((int64_t) 1 << 40) == st::CkOp::Slot);
    CHECK(st::ck_restore_op(st::kCkAt) == st::CkOp::At);
    CHECK(st::ck_restore_op(st::kCkRole0) == st::CkOp::Bad && st::ck_restore_op(st::kCkPin) == st::CkOp::Bad);
    CHECK(st::ck_restore_op(st::kSlotId0) == st::CkOp::Slot && st::ck_restore_op(st::kSlotId0 - 1) == st::CkOp::Bad);
    // a slot id never collides with a checkpoint's, and the main process's slots plus the one being saved fit keep[]
    CHECK(st::kSlotId0 > st::kCkAt && st::kMainSlotsMax + 1 <= net::kStageCkptMax);
}

static std::vector<uint8_t> bytes_of(size_t n, uint8_t v) { return std::vector<uint8_t>(n, v); }

static size_t files_in(const fs::path& d) {
    std::error_code ec;
    size_t n = 0;
    for (const auto& de : fs::directory_iterator(d, ec)) {
        (void) de;
        ++n;
    }
    return n;
}

static void test_partstore(const fs::path& root) {
    const fs::path d = root / "parts";
    std::string why;
    std::vector<uint8_t> out;
    {   // RAM only: put / get, a part at another position, replaced, pruned
        st::PartStore s("", 2, 0);
        CHECK(s.put(20, 100, bytes_of(1000, 1), why));
        CHECK(s.get(20, 100, out, why) == st::PartStore::Get::Ok && out == bytes_of(1000, 1));
        CHECK(s.get(20, 101, out, why) == st::PartStore::Get::Failed && why.find("saved at 100") != std::string::npos);
        CHECK(s.get(21, 100, out, why) == st::PartStore::Get::Missing);
        CHECK(s.put(20, 100, bytes_of(500, 2), why) && s.size() == 1);   // (replaced)
        CHECK(s.get(20, 100, out, why) == st::PartStore::Get::Ok && out == bytes_of(500, 2));
        CHECK(!s.put(22, 5, {}, why));   // (an empty part is none)
        CHECK(s.put(21, 7, bytes_of(10, 3), why) && s.put(22, 7, bytes_of(10, 4), why));   // past the RAM cap, no folder:
        CHECK(s.size() == 2 && !s.has(20) && s.has(21) && s.has(22));                        // the least recently used goes
        s.prune({22});
        CHECK(s.size() == 1 && s.has(22));
        s.clear();
        CHECK(s.size() == 0);
    }
    {   // no RAM, no folder: nothing is kept (the main process reads a conversation that left the model again)
        st::PartStore s(d.string(), 0, 0);
        CHECK(!s.put(20, 5, bytes_of(10, 1), why) && why.find("keeps no slots") != std::string::npos && s.size() == 0);
    }
    {   // the RAM cap spills the least recently used part to the folder; the folder's budget drops the oldest there
        st::PartStore s(d.string(), 1, 2500);
        CHECK(s.put(20, 1, bytes_of(1000, 1), why) && s.ram_parts() == 1 && s.disk_parts() == 0);
        CHECK(s.put(21, 2, bytes_of(1000, 2), why) && s.ram_parts() == 1 && s.disk_parts() == 1 && s.disk_bytes() == 1000);
        CHECK(fs::exists(s.file_of(20)) && s.file_of(21).empty());
        CHECK(s.get(20, 1, out, why) == st::PartStore::Get::Ok && out == bytes_of(1000, 1));   // (read back from the file)
        CHECK(s.put(22, 3, bytes_of(1000, 3), why));   // 20 was used last: 21 (in RAM) goes to the folder, now 2 there
        CHECK(s.ram_parts() == 1 && s.disk_parts() == 2 && s.disk_bytes() == 2000);
        CHECK(s.put(23, 4, bytes_of(1000, 4), why));   // 3 in the folder would be 3000 > 2500: the least recently used goes
        CHECK(s.disk_bytes() <= 2500 && s.size() == 3 && s.has(23));
        CHECK(files_in(d) == s.disk_parts());   // (a dropped part's file is gone)
        s.prune({23});
        CHECK(s.size() == 1 && files_in(d) == 0);
    }
    {   // a part bigger than the budget stays (the one just put is never the one dropped), as the main process's folder
        st::PartStore s(d.string(), 0, 100);
        CHECK(s.put(20, 1, bytes_of(1000, 1), why) && s.disk_parts() == 1);
        CHECK(s.get(20, 1, out, why) == st::PartStore::Get::Ok && out.size() == 1000);
        CHECK(s.put(21, 2, bytes_of(1000, 2), why) && s.has(21) && !s.has(20));   // (the older one made room)
    }
    {   // a file that was cut short: failed, and the part is dropped
        st::PartStore s(d.string(), 0, 1 << 20);
        CHECK(s.put(20, 1, bytes_of(1000, 1), why));
        put(s.file_of(20), "short");
        CHECK(s.get(20, 1, out, why) == st::PartStore::Get::Failed && !s.has(20));
    }
    {   // a folder left by an earlier run is emptied when the store is made, and when it ends
        put(d / "part-99.bin", "old");
        put(d / "part-98.bin.tmp", "old");
        put(d / "slot-1.bin", "a single process's slot: not ours");
        std::string f20;
        {
            st::PartStore s(d.string(), 0, 1 << 20);
            CHECK(!fs::exists(d / "part-99.bin") && !fs::exists(d / "part-98.bin.tmp") && fs::exists(d / "slot-1.bin"));
            CHECK(s.put(20, 1, bytes_of(100, 1), why) && fs::exists(s.file_of(20)));
            f20 = s.file_of(20);
        }
        CHECK(!f20.empty() && !fs::exists(f20));
        fs::remove(d / "slot-1.bin");
    }
    {   // two stores on one folder (two workers on one pack) have files of their own: same id and length, other bytes
        st::PartStore a(d.string(), 0, 1 << 20);
        st::PartStore b(d.string(), 0, 1 << 20);
        CHECK(a.put(16, 7, bytes_of(300, 1), why) && b.put(16, 7, bytes_of(300, 2), why));
        CHECK(a.file_of(16) != b.file_of(16));
        CHECK(a.get(16, 7, out, why) == st::PartStore::Get::Ok && out == bytes_of(300, 1));
        CHECK(b.get(16, 7, out, why) == st::PartStore::Get::Ok && out == bytes_of(300, 2));
    }
}

static void test_link(const fs::path& root) {
    const fs::path d = root / "linkparts";
    LinkTest t(d.string(), 2, 1 << 20);
    CHECK(t.start());
    if (t.where.empty()) return;
    const int64_t S0 = st::kSlotId0;
    // the hello: a main process that has the worker keep slots is served (earlier builds refused any ckpt_max)
    CHECK(t.peer.layer_begin == 2);
    {
        net::StageHello greedy = t.mine;
        greedy.ckpt_max = net::kStageCkptMax + 1;
        net::StageClient other;
        net::StageHello pr;
        std::string e;
        t.c.close();   // (the worker serves one main process at a time)
        CHECK(!other.connect(t.where, greedy, pr, e, 10) && e.find("at most 64") != std::string::npos);
        CHECK(t.connect());
    }

    // 1. checkpoints by role, the pin, and going back: the worker's follow the main process's, position for position
    CHECK(t.c.reset({}, t.err));
    CHECK(t.feed(0, 8));                                 // pos 8
    CHECK(t.save(st::kCkRole0 + 1, 8, {}));              // (role 1: the position before the prompt's last token)
    CHECK(t.feed(8, 4, 8));                              // pos 12
    CHECK(t.save(st::kCkRole0 + 0, 12, {}));
    CHECK(t.save(st::kCkPin, 12, {}));                   // (pin: checkpoint 1 also becomes the anchor, at 8)
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 12 && w.ck[0] == 12 && w.ck[1] == 8 && w.ck[2] == 8 && w.ck[3] == -1);
    }
    CHECK(!t.save(st::kCkRole0 + 2, 7, {}) && t.err.find("session is at 12") != std::string::npos);   // (the link checks L)
    CHECK(t.connect() && t.c.reset({}, t.err) && t.feed(0, 12) && t.save(st::kCkRole0 + 1, 12, {}) && t.save(st::kCkRole0 + 0, 12, {}));
    CHECK(t.feed(12, 4, 12));                            // pos 16
    CHECK(t.restore(st::kCkAt, 12, {}) == 1);            // back to 12: the session continues from there
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 12 && w.toks.size() == 12);
    }
    CHECK(t.feed(12, 4, 100));                           // (the link is at 12 too)
    CHECK(t.restore(st::kCkAt, 10, {}) == 0);            // no checkpoint at 10: missing, nothing changes
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 16 && w.toks.size() == 16 && w.ck[0] == 12);
    }
    CHECK(t.feed(16, 4, 200));
    CHECK(t.restore(st::kCkAt, 20, {}) == 0);            // a position past the checkpoints: missing too
    CHECK(t.feed(20, 4, 300));                           // the session is where it was: the next window runs

    // 2. slots: a conversation set aside under the main process's id, another one run, the first taken back whole
    CHECK(t.c.reset({}, t.err) && t.feed(0, 8, 1000));
    CHECK(t.save(st::kCkRole0 + 0, 8, {}));
    CHECK(t.save(S0, 8, {S0}));
    CHECK(t.has(S0));
    CHECK(t.c.reset({S0}, t.err) && t.has(S0));          // (a reset keeps what the main process holds)
    CHECK(t.feed(0, 6, 2000));
    CHECK(t.save(S0 + 1, 6, {S0, S0 + 1}));
    CHECK(t.restore(S0, 8, {S0, S0 + 1}) == 1);
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 8 && w.toks.size() == 8 && w.toks[0] == 1000 && w.toks[7] == 1007 && w.ck[0] == 8);   // (and its checkpoint)
    }
    CHECK(t.feed(8, 4, 1008));                           // the link took the restored position
    CHECK(t.restore(S0 + 1, 6, {S0, S0 + 1}) == 1);
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 6 && w.toks[0] == 2000);
    }

    // 3. the worker does not hold it (the main process falls back to token 0): never saved, pruned, at another position
    CHECK(t.restore(S0 + 9, 8, {S0}) == 0);
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 6 && w.toks[0] == 2000);            // (nothing changed on a miss)
    }
    CHECK(t.feed(6, 4, 2006));
    CHECK(t.restore(S0, 7, {S0, S0 + 1}) == -1 && t.err.find("saved at 8") != std::string::npos);
    CHECK(t.c.connected());                              // (a state error, not a lost link: the main process resets)
    CHECK(t.c.reset({S0}, t.err));                       // slot +1 is not in what the main process holds
    CHECK(!t.has(S0 + 1) && t.has(S0));
    CHECK(t.restore(S0 + 1, 6, {S0}) == 0);
    CHECK(t.restore(st::kCkRole0, 8, {S0}) == -1);       // (an id that is not a restore's)
    CHECK(t.c.reset({S0}, t.err));
    CHECK(!t.save(st::kCkAt, 0, {}));                    // (nor a save's)
    CHECK(t.connect());                                  // (an error from the worker leaves the link up; a fresh one for the rest)

    // 4. a conversation set aside twice without a change: the second is not kept (the main process has it already)
    CHECK(t.c.reset({}, t.err) && t.feed(0, 8, 2500) && t.save(S0, 8, {S0}));
    CHECK(t.c.reset({S0}, t.err) && t.has(S0) && t.feed(0, 8, 3000) && t.save(S0 + 2, 8, {S0 + 2}));
    CHECK(!t.has(S0) && t.has(S0 + 2));                  // (a save names what the main process holds: the rest goes)
    {
        bool stored = true;
        CHECK(t.c.ckpt_save(S0 + 3, 8, {S0 + 2, S0 + 3}, stored, t.err) && !stored);
        CHECK(!t.has(S0 + 3) && t.has(S0 + 2));
    }

    // 5. the worker's bounds: two in RAM, the rest in the folder (a megabyte at most); the main process's keep[] decides
    // what lives; a worker that restarts - or a main process that is taken over - holds none
    CHECK(t.c.reset({}, t.err));
    const std::vector<int64_t> all = {S0 + 10, S0 + 11, S0 + 12};
    for (size_t i = 0; i < all.size(); ++i) {
        const std::vector<int64_t> keep(all.begin(), all.begin() + (long) i + 1);
        CHECK(t.feed(0, 4, (int) (4000 + 10 * i)) && t.save(all[i], 4, keep) && t.c.reset(keep, t.err));
    }
    {
        std::lock_guard<std::mutex> lk(t.mu);
        CHECK(t.parts.ram_parts() == 2 && t.parts.disk_parts() == 1);
    }
    CHECK(t.restore(all[0], 4, all) == 1);               // (the oldest, read from the folder)
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 4 && w.toks[0] == 4000);
    }
    CHECK(t.connect());                                  // a new connection: the worker's parts went with the old one
    {
        std::lock_guard<std::mutex> lk(t.mu);
        CHECK(t.parts.size() == 0);
    }
    CHECK(t.c.reset({}, t.err) && t.restore(all[0], 4, all) == 0 && t.restore(all[2], 4, all) == 0);
    CHECK(t.feed(0, 4, 5000));                           // (and it goes on from token 0)

    // 6. a worker that keeps no slots (CAP and GB 0): a CkptSave of a slot is "not kept", the checkpoints still work
    t.shutdown();
    LinkTest u((root / "none").string(), 0, 0);
    CHECK(u.start());
    if (u.where.empty()) return;
    CHECK(u.c.reset({}, u.err) && u.feed(0, 8));
    {
        bool stored = true;
        CHECK(u.c.ckpt_save(S0, 8, {S0}, stored, u.err) && !stored && u.parts.size() == 0);
    }
    CHECK(u.save(st::kCkRole0, 8, {}) && u.restore(st::kCkAt, 8, {}) == 1);
    CHECK(u.restore(S0, 8, {S0}) == 0);                  // (so the main process finds it missing, and reads on)
}

// a prompt read on in chunks after the worker went back (a checkpoint, or a slot's part, taken back - then the rest of the
// prompt from there): a chunk is read where the worker's session is, so the restore has to put it at the chunk's p0
static void test_restore_chunks(const fs::path& root) {
    LinkTest t((root / "chunkparts").string(), 2, 1 << 20, 4);
    CHECK(t.start());
    if (t.where.empty()) return;
    CHECK(t.peer.chunk == 4);
    st::ChunkLink link(t.c, 4);
    std::vector<float> rows(16, 1.0f), back(16, 0.0f);
    const auto chunk_at = [&](int64_t p0, int first) {   // (4 tokens first .. first + 3, every row back)
        int32_t tk[4];
        for (int i = 0; i < 4; ++i) tk[i] = first + i;
        return link.send(tk, 4, p0, st::kChunkLast, rows.data(), 0, t.err) && link.recv(back.data(), 4, 0, t.err) &&
               link.wait(t.err);
    };
    CHECK(t.c.reset({}, t.err));
    CHECK(t.feed(0, 8));                         // pos 8
    CHECK(t.save(st::kCkRole0 + 1, 8, {}));
    CHECK(chunk_at(8, 8));                       // pos 12, a chunk at p0 8 > 0
    CHECK(t.save(st::kCkRole0 + 0, 12, {}));
    CHECK(t.feed(12, 4, 12));                    // pos 16
    {
        std::lock_guard<std::mutex> lk(t.mu);
        t.last_t = 3;   // (as after a draft's check of 3 rows whose Commit has not come)
    }
    CHECK(t.restore(st::kCkAt, 8, {}) == 1);     // back to 8, then the prompt's rest in a chunk from there
    {
        std::lock_guard<std::mutex> lk(t.mu);
        CHECK(t.last_t == 0);                    // (the handler ds4_main.cpp has: the window forgotten)
    }
    CHECK(chunk_at(8, 50));
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 12 && w.toks.size() == 12 && w.toks[8] == 50 && w.toks[11] == 53 && w.toks[7] == 7);
    }
    // a slot's part taken back, then a chunk at its end
    CHECK(t.save(st::kSlotId0, 12, {}));
    CHECK(t.c.reset({st::kSlotId0}, t.err));     // (another conversation: from token 0, the slot kept)
    CHECK(t.feed(0, 4, 90));
    CHECK(t.restore(st::kSlotId0, 12, {st::kSlotId0}) == 1);
    CHECK(chunk_at(12, 60));
    {
        const FakeModel w = t.snap();
        CHECK(w.p == 16 && w.toks[8] == 50 && w.toks[12] == 60 && w.toks[0] == 0);
    }
    // a chunk anywhere else than the worker's position: refused there (the main process's restore is what makes it right)
    CHECK(!chunk_at(8, 70) && t.err.find("this worker is at 16") != std::string::npos);
}

// the worker's restore handler forgets the last window's rows: no Commit after a restore undoes any
static void test_restore_last_t(const fs::path& root) {
    FakeModel m;
    st::PartStore parts((root / "lastparts").string(), 2, 1 << 20);
    std::vector<int32_t> tk(8);
    for (int i = 0; i < 8; ++i) tk[(size_t) i] = i;
    m.feed(tk.data(), 8);
    m.checkpoint(1);
    int last_t = 3;   // (a draft's check of 3 rows run, its Commit not yet in)
    std::string e, note;
    CHECK(st::worker_restore(m, parts, st::kCkAt, 8, {}, last_t, e, note) && last_t == 0 && m.p == 8);
    CHECK(st::commit_kind(2, last_t, e) == st::Keep::Bad);
    last_t = 3;   // (and when the restore fails: the session is not the window's either)
    CHECK(!st::worker_restore(m, parts, st::kCkAt, 5, {}, last_t, e, note) && last_t == 0);
}

}  // namespace slots

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
    {   // a pack written on Windows (CRLF lines) and on Linux: the same fingerprint, the model_dir line left out still
        auto crlf = [](const std::string& s) {
            std::string r;
            for (const char ch : s) r += ch == '\n' ? std::string("\r\n") : std::string(1, ch);
            return r;
        };
        put(b / "tensors.txt", crlf(tensors));
        put(b / "ds4.txt", crlf("model_dir D:\\models\\ds4\n" + geo));
        CHECK(st::pack_hash(b.string()) == ha);
        put(b / "ds4.txt", crlf(geo) + "\r");   // (a '\r' after the last line, with no newline)
        CHECK(st::pack_hash(b.string()) == ha);
        put(b / "ds4.txt", crlf("hidden_size 4096\nnum_hidden_layers 42\nhc_mult 4\n"));   // another geometry
        CHECK(st::pack_hash(b.string()) != ha);
    }

    // the worker's checks of a main process's hello
    net::StageHello mine, theirs;
    mine.max_t = theirs.max_t = 16;
    theirs.max_context = 8192;
    theirs.row_formats = st::hello_rows(false);
    std::string e;
    CHECK(st::worker_accepts(theirs, mine, 8192, e));
    // (stage_hello's row_formats: fp32 for the link, the normed reply always, the compact rows' bit with bf16 only)
    CHECK(st::hello_rows(false) == ((1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem | st::kRowsNormed));
    CHECK(st::hello_rows(true) == (st::hello_rows(false) | st::kRowsBf16Cap));
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
        net::StageHello w = mine;   // (one that takes chunks: any main process's, which sends up to the worker's)
        w.chunk = 2048;
        CHECK(st::worker_accepts(t, w, 8192, e));
        t.chunk = 1024;
        CHECK(st::worker_accepts(t, w, 8192, e));
        t.chunk = 4096;
        CHECK(st::worker_accepts(t, w, 8192, e) && st::chunk_sent(t.chunk, w.chunk) == 2048);
        // (the chunked path's policy on a remote stage: chunks of kLinkChunk unless given, chunk after chunk unless
        // DS4_LAYER_MAJOR=1; one process as before - its own gate, DS4_LAYER_MAJOR=0 never)
        CHECK(st::kLinkChunk == 2048);
        CHECK(!st::layer_major_allowed(true, nullptr) && !st::layer_major_allowed(true, "0") &&
              st::layer_major_allowed(true, "1") && !st::layer_major_allowed(true, "2"));
        CHECK(st::layer_major_allowed(false, nullptr) && !st::layer_major_allowed(false, "0") &&
              st::layer_major_allowed(false, "1") && st::layer_major_allowed(false, "2"));
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
        t.ckpt_max = 3;   // (a main process that has this worker keep its checkpoints and slots: served)
        CHECK(st::worker_accepts(t, mine, 8192, e));
        t.ckpt_max = st::kMainSlotsMax + 1;
        CHECK(st::worker_accepts(t, mine, 8192, e));
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
        CHECK(!st::worker_accepts(t, w, 8192, e) && e.find("STRATA_REMOTE_ROWS=fp32 there") != std::string::npos);
        t.row_formats |= st::kRowsBf16Cap;
        CHECK(st::worker_accepts(t, w, 8192, e));
        // an fp32 worker, a bf16 main process
        CHECK(!st::worker_accepts(t, mine, 8192, e) && e.find("STRATA_REMOTE_ROWS=fp32 here") != std::string::npos);
        // (the link's own bf16 stays refused: it would round the picks)
        t.row_formats |= 1u << (uint32_t) net::StageRows::BF16;
        CHECK(!st::worker_accepts(t, w, 8192, e) && e.find("fp32") != std::string::npos);
    }

    // bf16 rows and prompt chunks together: taken (the engine packs a chunk's rows as a window's), as fp32 rows' are
    {
        net::StageHello w = mine, t = theirs;
        w.row_formats |= st::kRowsBf16Cap;
        w.chunk = 2048;
        t.row_formats |= st::kRowsBf16Cap;
        CHECK(st::worker_accepts(t, w, 8192, e));   // (chunk 0: windows)
        t.chunk = 1024;
        CHECK(st::worker_accepts(t, w, 8192, e));
        w.row_formats |= st::kRowsNormed;   // (this block's worker hello: the reply's layout too)
        CHECK(st::main_accepts(w, t, e) && st::chunk_sent(t.chunk, w.chunk) == 1024);
        t.row_formats &= ~st::kRowsBf16Cap;   // (fp32 rows: the same chunks taken)
        w.row_formats &= ~st::kRowsBf16Cap;
        CHECK(st::worker_accepts(t, w, 8192, e));
    }

    // the worker's buffers: a window's rows and a chunk's at the link's width (the compact one with bf16 rows)
    {
        st::RowGeo real;
        real.h = 4 * 5120;
        real.topk = 512;
        real.cand = 2048;
        std::vector<float> b0(4), b1(4), b2(4), b3(4);
        float* const pf[4] = {b0.data(), b1.data(), b2.data(), b3.data()};
        const net::StageBuffers b = st::worker_buffers(nullptr, nullptr, 16, pf, 1024, real.compact());
        CHECK(b.handoff_floats == real.compact() && b.run_floats == (size_t) 16 * (size_t) real.compact());
        CHECK(b.pf_row_floats == real.compact() && b.pf_floats == (size_t) 1024 * (size_t) real.compact());
        const net::StageBuffers f = st::worker_buffers(nullptr, nullptr, 16, pf, 1024, real.full());
        CHECK(f.pf_row_floats == real.full() && f.pf_floats == (size_t) 1024 * (size_t) real.full());
        const net::StageBuffers none = st::worker_buffers(nullptr, nullptr, 16, pf, 0, real.full());
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

    // a chunk's reply, row by row: rows [skip, T) compact (streams bf16, pre exact) and back; the rows before skip and
    // each row's rest not written either way
    {
        std::mt19937 rng(9);
        st::RowGeo g;
        g.h = 4 * 64;
        g.topk = 40;
        g.cand = 70;
        const size_t W = (size_t) g.full(), C = (size_t) g.compact(), H = (size_t) g.h;
        for (const int T : {1, 5, 16})
            for (const int skip : {0, 1, T - 1, T}) {
                std::vector<float> full((size_t) T * W);   // (finite floats: bf16_round's)
                std::normal_distribution<float> nd(0.f, 3.f);
                for (float& x : full) x = nd(rng);
                const float mark = float_of(0xdeadbeefu);
                std::vector<float> wire((size_t) T * C, mark), back((size_t) T * W, mark);
                st::reply_rows_pack(g, T, skip, full.data(), wire.data());
                st::reply_rows_unpack(g, T, skip, wire.data(), back.data());
                bool ok = true, rounded = skip >= T;
                for (int r = 0; r < T; ++r)
                    for (size_t j = 0; j < W; ++j) {
                        const float v = back[(size_t) r * W + j], f = full[(size_t) r * W + j];
                        if (r < skip || j >= H + (size_t) g.pre) ok = ok && bits_of(v) == bits_of(mark);
                        else if (j < H) {
                            ok = ok && bits_of(v) == bits_of(bf16_round(f));
                            rounded = rounded || bits_of(v) != bits_of(f);
                        } else
                            ok = ok && bits_of(v) == bits_of(f);
                    }
                for (int r = 0; r < T; ++r)   // (on the wire: the row's rest, and the rows before skip, untouched)
                    for (size_t j = (r < skip ? 0 : H / 2 + (size_t) g.pre); j < C; ++j)
                        ok = ok && bits_of(wire[(size_t) r * C + j]) == bits_of(mark);
                CHECK(ok);
                CHECK(rounded);
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
            CHECK(!link_window(w, !bf, T, rows, none, err) && err.find("STRATA_REMOTE_ROWS") != std::string::npos);
            w.end();
            w.at.clear();
        }
    }

    slots::test_ids();
    slots::test_partstore(root);
    slots::test_link(root);
    slots::test_restore_chunks(root);
    slots::test_restore_last_t(root);

    std::error_code ec;
    fs::remove_all(root, ec);
    replies();
    commits();
    link_drafts();
    chunked::chunks_over_a_link();
    chunked::chunks_bf16_over_a_link();
    if (fails == 0) std::printf("ds4_stage_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
