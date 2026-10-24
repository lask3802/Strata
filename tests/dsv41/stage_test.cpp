// The DeepSeek engine's remote stage without a GPU (src/dsv41/ds4_stage.hpp): the pack fingerprint both hellos carry
// (the same model on two PCs whose model_dir differs; any other line of ds4.txt or tensors.txt changes it), every
// main-process hello a worker refuses and the worker's hello its main process refuses (the prompt chunks' sizes), and
// prompt chunks over a loopback link (src/net/stage_link.cpp) - the worker's buffers as ds4_main.cpp lends them, the
// main process's ChunkLink sending ahead of the replies, the rows block by block there and row by row back, the
// replies' skip, the last chunk's flag, a chunk too large for the worker.  The worker's handler stands in for the
// model's (Model::stage_prefill): it checks and turns the rows as the layout says, no layers.
//   ds4_stage_test            exit 0: all passed
#include "ds4_stage.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
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

static bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// ---- prompt chunks over a loopback link: a geometry of hc 2 x dim 8 streams, topk_stride 4, cand_stride 4
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
    h.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem;
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
        bufs = st::worker_buffers(run_in_out[0].data(), run_in_out[1].data(), 16, p, chunk, kW);
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
    theirs.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem;
    std::string e;
    CHECK(st::worker_accepts(theirs, mine, 8192, e));
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

    chunks_over_a_link();

    std::error_code ec;
    fs::remove_all(root, ec);
    if (fails == 0) std::printf("ds4_stage_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
