// The DeepSeek engine's remote stage without a GPU (src/dsv41/ds4_stage.hpp): the pack fingerprint both hellos carry
// (the same model on two PCs whose model_dir differs; any other line of ds4.txt or tensors.txt changes it), every
// main-process hello a worker refuses and the worker's hello a main process refuses, the reply's layout (the normed
// rows, then the drafter's taps: which ones the main process takes), what a Commit keeps - and, over the loopback
// through src/net/stage_link.cpp, a main process's draft checks committed in part on a worker that runs ds4_main.cpp's
// Run and Commit (worker_run, worker_commit) over a stand-in model keeping a history, undone past what a Commit keeps
// (each reply must be what the kept tokens alone give).
//   ds4_stage_test            exit 0: all passed
#include "ds4_stage.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
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

// the reply's layout: the real geometry fits, a target the main process holds keeps its own slot
static void replies() {
    const int64_t ds41 = 4 * 5120 + 8 + 512 + 2048;   // (hc x dim + pre + topk_stride + cand_stride: ds4.txt's)
    CHECK(st::reply_fits(5120, 3, ds41));
    CHECK(!st::reply_fits(5120, 4, ds41));
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
    CHECK(st::main_accepts(peer, e));
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
        CHECK(!st::main_accepts(t, e) && e.find("earlier build") != std::string::npos);
        CHECK(st::main_accepts(theirs, e));
    }
    {
        net::StageHello t = theirs;
        t.chunk = 2048;
        CHECK(!st::worker_accepts(t, mine, 8192, e) && e.find("chunks") != std::string::npos);
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

    std::error_code ec;
    fs::remove_all(root, ec);
    replies();
    commits();
    link_drafts();
    if (fails == 0) std::printf("ds4_stage_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
