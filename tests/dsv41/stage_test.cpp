// The DeepSeek engine's remote stage without a GPU (src/dsv41/ds4_stage.hpp): the pack fingerprint both hellos carry
// (the same model on two PCs whose model_dir differs; any other line of ds4.txt or tensors.txt changes it), and every
// main-process hello a worker refuses; and the checkpoints and slots the main process has its worker keep (STRATA_DS4_SLOT_KEEP):
// the ids CkptSave / CkptRestore carry (ck_save_op / ck_restore_op), the worker's store of its parts of slots
// (ds4_partstore.hpp: RAM cap, folder budget, keep[], a folder left by an earlier run), and the worker's handlers
// (worker_ck_save / worker_ck_restore, the engine's own) over the real link on loopback with a stand-in for the model: the
// worker's checkpoints follow the main process's by role and position, a slot is set aside and taken back whole, and what
// the worker does not hold (never saved, pruned, dropped for its bounds, gone with its connection, kept by no folder) comes
// back as "missing" with nothing changed - the main process then reads the conversation from token 0.
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
    h.row_formats = (1u << (uint32_t) net::StageRows::F32) | net::kStageCapKvElem;
    h.max_context = 64;
    h.handoff_floats = 4;
    h.pack_hash = 0x44533456;
    h.ckpt_max = worker ? 0 : st::kMainSlotsMax + 1;
    return h;
}

// a worker (serve_stage on loopback) whose handlers are the engine's own (worker_ck_save / worker_ck_restore) over a
// FakeModel, and the main process's side of the link: windows of tokens through Run + Commit, and the checkpoint messages
struct LinkTest {
    FakeModel model;           // the worker's, behind mu
    std::mutex mu;
    st::PartStore parts;
    net::StageHandlers h;
    net::StageBuffers buf;
    std::vector<float> rin, rout;
    net::StageStats stats;
    std::atomic<bool> stop{false};
    std::atomic<int> served{-1};
    std::thread t;
    std::string where, err;
    net::StageClient c;
    net::StageHello mine = link_hello(false), peer;

    LinkTest(const std::string& dir, size_t ram, uint64_t disk) : parts(dir, ram, disk), rin(16), rout(16) {
        h.hello = [this](const net::StageHello& theirs, net::StageHello& m, std::string& e) {
            m = link_hello(true);
            return st::worker_accepts(theirs, m, 64, e);
        };
        h.run = [this](int T, const int32_t* tokens, int64_t, std::string&) {
            std::lock_guard<std::mutex> lk(mu);
            model.feed(tokens, T);
            return true;
        };
        h.commit = [](int, std::string&) { return true; };
        h.prefill = [](const int64_t*, int64_t, int64_t, int64_t, int64_t, const float*, float*, std::string& e) {
            e = "a prompt chunk: this worker reads windows only";
            return false;
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
        h.ckpt_save = [this](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e) -> int {
            std::lock_guard<std::mutex> lk(mu);
            std::string note;
            return st::worker_ck_save(model, parts, id, L, keep, e, note);
        };
        h.ckpt_restore = [this](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e) {
            std::lock_guard<std::mutex> lk(mu);
            std::string note;
            return st::worker_ck_restore(model, parts, id, L, keep, e, note);
        };
        buf.run_in = rin.data();
        buf.run_out = rout.data();
        buf.run_floats = rin.size();
        buf.handoff_floats = 4;
        buf.pf_row_floats = 4;
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

    test_ids();
    test_partstore(root);
    test_link(root);

    std::error_code ec;
    fs::remove_all(root, ec);
    if (fails == 0) std::printf("ds4_stage_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
