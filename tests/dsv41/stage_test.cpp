// The DeepSeek engine's remote stage without a GPU (src/dsv41/ds4_stage.hpp): the pack fingerprint both hellos carry
// (the same model on two PCs whose model_dir differs; any other line of ds4.txt or tensors.txt changes it), and every
// main-process hello a worker refuses; the compact bf16 rows (STRATA_REMOTE_ROWS=bf16): the picks exact, the streams
// rounded as the link's bf16 rounds, the hello refusing a mismatch - and both over a real link on the loopback.
//   ds4_stage_test            exit 0: all passed
#include "ds4_stage.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
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
        h.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem | (bf16 ? st::kRowsBf16Cap : 0u);
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
    if (fails == 0) std::printf("ds4_stage_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
