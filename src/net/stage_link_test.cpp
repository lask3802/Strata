// src/net/stage_link_test.cpp - the remote stage's link over loopback, no GPU: a worker (serve_stage) with handlers
// that copy rows, keep a StageCkptStore and record the image positions they get, and a StageClient driving it.  What
// it checks (docs/remote-stage/CHECKPOINTS.md; docs/remote-stage/ENGINEERING.md, "Images"):
//
//   0. the worker's table from what a Positions message carries (stage_positions_fill): the identity, the cells
//      sent, the cells past them;
//   1. the hello: an older side (protocol 4) is refused for its protocol, a main process that would have the worker
//      keep more than kStageCkptMax checkpoints is refused, a main process with --vision refuses a worker without it,
//      a matching one is served;
//   2. Reset / CkptSave / CkptRestore carry their id, position and keep[] whole (int64 ids beyond 32 bits);
//   3. the worker's position: a checkpoint is saved only where the session is (0 after a reset, a chunk's end, a
//      window's first position plus its accepted count, L after a restore); a failed Commit comes back on the
//      CkptSave, and a CkptRestore drops it (the session is replaced);
//   4. CkptSaveOk a=0 (not stored) leaves the store as it was; a restore of an id the store does not hold is the
//      missing-id error (the main process's fallback), any other restore error is not;
//   5. a keep[] longer than kStageCkptMax is a malformed message; a disconnect clears the store;
//   6. Positions carries the table whole (cells x (t, h, w), values past 16 bits) and the identity as no table; a
//      failed Positions comes back on the next reply - also when that is a Reset or a CkptRestore, which drop a failed
//      Commit's error but not the next request's positions - and the link stays usable; a later table replaces a
//      failed one; a handler's success is its return value; a malformed table is the next reply's error, not a reply
//      of its own; a Positions does not move the session's position;
//   7. a worker without a positions handler (started without --vision) answers a Positions with an error naming
//      --vision.
// The handlers run on the worker's serving thread: what they share with the test is behind a mutex or atomic.
#include "strata/net/stage_ckpt_store.hpp"
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::net;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr int64_t kHbf = 4, kD = 4, kChunk = 8, kMaxT = 4;
StageHello hello_of(int32_t vision = 1) {
    StageHello h;
    h.n_embd = 4;
    h.hc = 1;
    h.n_layers = 4;
    h.n_expert = 2;
    h.layer_begin = 2;
    h.max_t = (int32_t) kMaxT;
    h.max_context = 64;
    h.chunk = kChunk;
    h.handoff_floats = kHbf;
    h.pack_hash = 0x5354524d;
    h.vision = vision;
    return h;
}
bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// the worker's side the test looks at
struct Shared {
    std::mutex mu;
    StageCkptStore store;
    std::vector<int64_t> last_keep;
    int64_t last_id = -1, last_L = -1;
    int pos_calls = 0;   // the positions handler's
    bool pos_identity = false;
    std::vector<int32_t> pos_table;
    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
    bool has(int64_t id) {
        std::lock_guard<std::mutex> lk(mu);
        return store.find(id) != nullptr;
    }
    bool last(int64_t id, int64_t L, const std::vector<int64_t>& keep) {
        std::lock_guard<std::mutex> lk(mu);
        return last_id == id && last_L == L && last_keep == keep;
    }
    bool last_keep_is(const std::vector<int64_t>& keep) {
        std::lock_guard<std::mutex> lk(mu);
        return last_keep == keep;
    }
    bool positions_were(int n, bool identity, const std::vector<int32_t>& table) {
        std::lock_guard<std::mutex> lk(mu);
        return pos_calls == n && pos_identity == identity && pos_table == table;
    }
};

// one worker on a free loopback port, until stop()
struct Worker {
    std::atomic<bool> stop_flag{false};
    std::atomic<int> served{-1};
    StageStats st;
    std::thread t;
    std::string where;
    bool start(const StageHandlers& h, const StageBuffers& buf) {
        std::mt19937 rng(std::random_device{}());
        for (int attempt = 0; attempt < 8 && where.empty(); ++attempt) {
            const int port = 20000 + (int) (rng() % 40000);
            served = -1;
            t = std::thread([&, port] { served = serve_stage("127.0.0.1", port, "", h, buf, st, &stop_flag); });
            const std::string at = "127.0.0.1:" + std::to_string(port);
            // until it listens: the worker's own refusal of a protocol 4 hello (not the system's "connection refused")
            for (int i = 0; i < 100 && served.load() < 0; ++i) {
                StageHello bad = hello_of();
                bad.protocol = 4;
                StageHello peer;
                std::string e;
                StageClient c;
                if (c.connect(at, bad, peer, e) || contains(e, "the worker refused:")) { where = at; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (where.empty()) stop();
        }
        return !where.empty();
    }
    void stop() {
        if (!t.joinable()) return;
        stop_flag = true;
        t.join();
        stop_flag = false;
    }
};
}  // namespace

int main() {
    std::printf("stage_link_test\n");
    // 0. the worker's table from a Positions message (no link)
    {
        std::vector<int32_t> out(10 * 3, -1);
        stage_positions_fill(out.data(), 10, nullptr, 0);
        bool ident = true;
        for (int c = 0; c < 10; ++c) ident = ident && out[c * 3] == c && out[c * 3 + 1] == c && out[c * 3 + 2] == c;
        check(ident, "no table: every cell its own index");
        // as the main process builds it: text 0-1, a 2x2 picture at t = 2 (cells 2-5), the text after it from 4 (the
        // picture's extent, 2, not its 4 cells); 8 cells sent of 10
        const int32_t sent[8 * 3] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5};
        stage_positions_fill(out.data(), 10, sent, 8);
        check(std::equal(sent, sent + 24, out.begin()), "the cells sent are taken as they are");
        check(out[8 * 3] == 6 && out[8 * 3 + 1] == 6 && out[8 * 3 + 2] == 6 && out[9 * 3] == 7 && out[9 * 3 + 2] == 7,
              "the cells past them continue from the last one, as the main process's do");
        const int32_t high[3] = {INT32_MAX - 1, 0, 0};
        stage_positions_fill(out.data(), 3, high, 1);
        check(out[3] == INT32_MAX, "the continuation is computed in 64 bits");
    }
    // ---- the worker
    std::vector<float> run_in((size_t) (kMaxT * kHbf)), run_out(run_in.size());
    std::vector<float> pf_in0((size_t) (kChunk * kD)), pf_in1(pf_in0.size()), pf_out0(pf_in0.size()),
        pf_out1(pf_in0.size());
    StageBuffers buf;
    buf.run_in = run_in.data();
    buf.run_out = run_out.data();
    buf.run_floats = run_in.size();
    buf.pf_in[0] = pf_in0.data();
    buf.pf_in[1] = pf_in1.data();
    buf.pf_out[0] = pf_out0.data();
    buf.pf_out[1] = pf_out1.data();
    buf.pf_floats = pf_in0.size();
    buf.handoff_floats = kHbf;
    buf.pf_row_floats = kD;
    Shared w;
    std::atomic<int32_t> worker_vision{1};
    std::atomic<bool> refuse_save{false}, fail_commit{false}, fail_positions{false}, stale_text{false};
    std::atomic<int> disconnects{0};
    StageHandlers h;
    h.hello = [&](const StageHello&, StageHello& mine, std::string&) {
        mine = hello_of(worker_vision.load());
        return true;
    };
    h.run = [&](int T, const int32_t*, int64_t, std::string&) {
        std::memcpy(run_out.data(), run_in.data(), (size_t) (T * kHbf) * 4);
        return true;
    };
    h.commit = [&](int, std::string& e) {
        if (!fail_commit.load()) return true;
        e = "a commit failed (test)";
        return false;
    };
    h.prefill = [&](const int64_t*, int64_t T, int64_t, int64_t, int64_t, const float* in, float* out, std::string&) {
        std::memcpy(out, in, (size_t) (T * kD) * 4);
        return true;
    };
    h.reset = [&](const std::vector<int64_t>& keep, std::string&) {
        std::lock_guard<std::mutex> lk(w.mu);
        w.last_keep = keep;
        w.store.prune(keep);
        return true;
    };
    h.positions = [&](const int32_t* table, int64_t cells, std::string& e) {
        std::lock_guard<std::mutex> lk(w.mu);
        ++w.pos_calls;
        w.pos_identity = table == nullptr;
        w.pos_table.clear();
        if (table != nullptr) w.pos_table.assign(table, table + cells * 3);
        if (stale_text.load()) e = "a message left from a retry (test)";   // and it succeeds
        // fail_positions, or a table starting at t = -7 (no reply tells when the worker has read a Positions, so a
        // flag flipped right after one is a race; the table's own content is not)
        if (!fail_positions.load() && (table == nullptr || table[0] != -7)) return true;
        e = "the positions failed (test)";
        return false;
    };
    h.ckpt_save = [&](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string&) -> int {
        std::lock_guard<std::mutex> lk(w.mu);
        w.last_id = id;
        w.last_L = L;
        w.last_keep = keep;
        if (refuse_save.load() || !w.store.admits(id, keep)) return 0;
        StageCkptStore::Part p;
        p.L = L;
        w.store.put(id, std::move(p), keep);
        return 1;
    };
    h.ckpt_restore = [&](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e) {
        std::lock_guard<std::mutex> lk(w.mu);
        w.last_keep = keep;
        if (id == 99) {   // a relay's forwarded missing id: its own prefix first
            e = std::string("relay: the next worker: remote stage (worker): ") + kStageCkptMissing + "99";
            return false;
        }
        const StageCkptStore::Part* p = w.store.find(id);
        if (p == nullptr) {
            e = kStageCkptMissing + std::to_string(id);
            return false;
        }
        if (p->L != L) {
            e = "checkpoint saved at another position";
            return false;
        }
        w.store.prune(keep);
        return true;
    };
    h.disconnected = [&] {
        {
            std::lock_guard<std::mutex> lk(w.mu);
            w.store.clear();
        }
        ++disconnects;
    };
    Worker wk;
    if (!wk.start(h, buf)) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    StageClient c;
    std::string err;
    StageHello peer;
    auto greet_at = [&](const std::string& at, StageHello mine) {
        err.clear();
        c.close();
        return c.connect(at, mine, peer, err);
    };
    auto greet = [&](StageHello mine) { return greet_at(wk.where, mine); };

    // 1. the hello
    StageHello old_side = hello_of();
    old_side.protocol = 4;
    check(!greet(old_side) && contains(err, "the worker refused:") && contains(err, "protocol 4/6"),
          "a protocol 4 main process is refused for its protocol");
    StageHello greedy = hello_of();
    greedy.ckpt_max = kStageCkptMax + 1;
    check(!greet(greedy) && contains(err, "at most 64"),
          "a main process asking for more than 64 checkpoints is refused");
    worker_vision = 0;
    check(!greet(hello_of(1)) && contains(err, "--vision"), "a main with --vision refuses a worker without it");
    check(greet(hello_of(0)), "... one without --vision is served by it");
    worker_vision = 1;
    StageHello mine = hello_of();
    mine.ckpt_max = kStageCkptMax;
    check(greet(mine) && peer.vision == 1, "a matching main process is served");

    // 2-3. a reset, a prompt chunk, a checkpoint at its end, a window and its commit, another checkpoint
    bool stored = false, missing = false;
    check(c.reset({}, err) && w.last_keep_is({}), "a reset that keeps nothing");
    std::vector<int64_t> tok((size_t) kChunk);
    std::vector<float> rows((size_t) (kChunk * kD)), back(rows.size(), 0.0f);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (float) i;
    for (size_t i = 0; i < tok.size(); ++i) tok[i] = (int64_t) i;
    auto chunk = [&] {
        std::fill(back.begin(), back.end(), 0.0f);
        return c.prefill_send(tok.data(), kChunk, 0, 0, rows.data(), (size_t) kD, 0, err) &&
               c.prefill_recv(back.data(), (size_t) kD, kChunk, 0, err) && back == rows;
    };
    check(chunk(), "a prompt chunk's rows come back");
    check(c.reset({}, err) && !c.ckpt_save(7, 3, {7}, stored, err) && contains(err, "session is at 0") &&
              w.size() == 0,
          "after a reset the session is at 0, not at the chunk's end: a checkpoint elsewhere is a state error");
    check(chunk(), "the prompt chunk again");
    const int64_t big = ((int64_t) 1 << 40) + 7;
    check(!c.ckpt_save(big, 5, {big}, stored, err) && contains(err, "session is at 8") && c.connected(),
          "a checkpoint where the session is not is a state error (the link stays)");
    check(c.ckpt_save(big, kChunk, {big}, stored, err) && stored && w.last(big, kChunk, {big}),
          "CkptSave carries id, position and keep whole; stored at a chunk's end");
    std::vector<int32_t> wtok(3, 1);
    std::vector<float> win((size_t) (3 * kHbf), 2.0f), wout(win.size(), 0.0f);
    check(c.run(3, wtok.data(), kChunk, win.data(), win.size(), wout.data(), wout.size(), err) && wout == win &&
              c.commit(2, err),
          "a window and its commit (two of three accepted)");
    check(c.ckpt_save(2, kChunk + 2, {big, 2}, stored, err) && stored && w.size() == 2,
          "a checkpoint at the window's position plus its accepted count");
    refuse_save = true;
    check(c.ckpt_save(3, kChunk + 2, {big, 2, 3}, stored, err) && !stored && w.size() == 2 && !w.has(3),
          "CkptSaveOk a=0: not stored, the store as it was");
    refuse_save = false;

    // 4. restores
    check(c.ckpt_restore(big, kChunk, {big}, missing, err) && !missing && w.size() == 1 && w.has(big),
          "a restore puts the part back and prunes to its keep");
    check(c.ckpt_save(5, kChunk, {big, 5}, stored, err) && stored && w.size() == 2,
          "after a restore the session is at its L: a checkpoint there is stored");
    check(!c.ckpt_restore(2, kChunk + 2, {big, 2}, missing, err) && missing && c.connected(),
          "a restore of an id the store does not hold is the missing-id error");
    check(!c.ckpt_restore(big, kChunk + 1, {big}, missing, err) && !missing,
          "any other restore error is not the missing-id one");
    check(!c.ckpt_restore(99, kChunk, {big}, missing, err) && !missing && contains(err, kStageCkptMissing),
          "a relay's forwarded missing id is not the missing-id error (the state class)");
    check(c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err),
          "a window after the restore");
    fail_commit = true;
    check(c.commit(1, err), "a commit that fails on the worker (one way: no reply)");
    check(!c.ckpt_save(4, kChunk + 1, {big, 5, 4}, stored, err) && contains(err, "a commit failed (test)"),
          "the failed commit comes back on the next CkptSave");
    check(c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err) &&
              c.commit(1, err),
          "another window whose commit fails");
    // (the restore waits for its reply, so the worker has run that commit, failing, before fail_commit goes back)
    check(c.ckpt_restore(big, kChunk, {big, 5}, missing, err), "a restore while the failed commit's error is pending");
    fail_commit = false;
    check(c.ckpt_save(6, kChunk, {big, 5, 6}, stored, err) && stored && w.size() == 3,
          "a restore drops the failed commit: the next CkptSave is stored");

    // 5. a malformed keep, a reset that keeps one, a disconnect
    std::vector<int64_t> too_many((size_t) kStageCkptMax + 1, 1);
    check(!c.reset(too_many, err) && contains(err, "bad checkpoint list"), "a keep[] of more than 64 ids is refused");
    check(c.reset({big}, err) && w.last_keep_is({big}) && w.size() == 1, "a reset keeps the parts it names");
    const int before = disconnects.load();
    c.close();
    for (int i = 0; i < 200 && disconnects.load() == before; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(disconnects.load() > before && w.size() == 0, "a disconnect clears the store");
    check(greet(mine) && !c.ckpt_restore(big, kChunk, {big}, missing, err) && missing,
          "the next main process finds none of the last one's parts");

    // 6. image positions: the table whole, the identity as none, a failure outlives a reset and a restore
    const int64_t cells = 40;
    std::vector<int32_t> table((size_t) cells * 3);
    for (int64_t i = 0; i < cells; ++i) {   // a grid's (t, h, w), positions past 16 bits
        table[(size_t) i * 3] = 70000;
        table[(size_t) i * 3 + 1] = 70000 + (int32_t) (i / 8);
        table[(size_t) i * 3 + 2] = 70000 + (int32_t) (i % 8);
    }
    err.clear();
    check(c.positions(table.data(), cells, err) && c.reset({}, err) && w.positions_were(1, false, table),
          "a table arrives whole, before the reset");
    check(c.positions(nullptr, 0, err) && c.reset({}, err) && w.positions_were(2, true, {}),
          "the identity arrives as no table");
    const int32_t one[3] = {5, 5, 5};
    fail_positions = true;
    err.clear();
    bool sent = c.positions(one, 1, err);
    check(sent && !c.reset({}, err) && contains(err, "the positions failed (test)"),
          "a failed Positions comes back on the reset");
    fail_positions = false;
    err.clear();
    check(c.reset({}, err), "... once: the link is usable after it");
    fail_commit = true;
    check(c.commit(1, err) && c.positions(one, 1, err) && c.reset({}, err) && w.positions_were(4, false, {5, 5, 5}),
          "a failed Commit is dropped by the reset (the positions after it applied)");
    const int32_t failing[3] = {-7, -7, -7};
    err.clear();
    sent = c.commit(1, err) && c.positions(failing, 1, err);
    check(sent && !c.reset({}, err) && contains(err, "the positions failed (test)") && !contains(err, "a commit failed"),
          "a failed Commit, then a failed Positions: the reset reports the positions only");
    fail_commit = false;
    err.clear();
    sent = c.positions(failing, 1, err);
    check(sent && c.positions(nullptr, 0, err) && c.reset({}, err) && w.positions_were(7, true, {}),
          "a later table replaces a failed one (a refused request: no reset)");
    stale_text = true;
    check(c.positions(one, 1, err) && c.reset({}, err), "a handler that succeeds with text left in its error: a success");
    stale_text = false;
    std::vector<int32_t> huge((size_t) (((int64_t) 1 << 22) + 1) * 3, 1);
    err.clear();
    sent = c.positions(huge.data(), (int64_t) (huge.size() / 3), err);
    check(sent && !c.reset({}, err) && contains(err, "bad position table size") && w.positions_were(8, false, {5, 5, 5}),
          "a malformed table (over 4M cells): the next reply's error, no handler call");
    check(c.reset({}, err), "... and only that reply's: the link is in step after it");
    check(chunk() && c.positions(one, 1, err) && c.ckpt_save(1, kChunk, {1}, stored, err) && stored,
          "a Positions does not move the session: a checkpoint at the chunk's end after one");
    fail_commit = true;
    err.clear();
    sent = c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err) &&
           c.commit(1, err) && c.positions(failing, 1, err);
    check(sent && !c.ckpt_restore(1, kChunk, {1}, missing, err) && !missing &&
              contains(err, "the positions failed (test)") && !contains(err, "a commit failed") && w.has(1),
          "a failed Commit, then a failed Positions: a restore reports the positions only");
    fail_commit = false;
    fail_positions = true;
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.ckpt_restore(1, kChunk, {1}, missing, err) && !missing &&
              contains(err, "the positions failed (test)") && w.has(1),
          "a failed Positions comes back on a restore (not the missing-id error)");
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.ckpt_save(2, kChunk, {1, 2}, stored, err) && contains(err, "the positions failed (test)") &&
              !w.has(2),
          "... and on a CkptSave");
    fail_positions = false;
    check(c.ckpt_restore(1, kChunk, {1}, missing, err), "a restore after them");
    fail_positions = true;
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err) &&
              contains(err, "the positions failed (test)"),
          "... and on the next window");
    fail_positions = false;
    check(c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err),
          "... and the next window runs");
    c.close();
    wk.stop();

    // 7. a worker started without --vision: no positions handler
    worker_vision = 0;
    StageHandlers plain = h;
    plain.positions = nullptr;
    Worker wk2;
    if (!wk2.start(plain, buf)) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet_at(wk2.where, hello_of(0)), "a main without --vision is served by a worker without it");
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.reset({}, err) && contains(err, "--vision"), "a Positions to it is an error naming --vision");
    c.close();
    wk2.stop();

    if (g_fail == 0) std::printf("stage_link_test: all passed\n");
    else std::printf("stage_link_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
