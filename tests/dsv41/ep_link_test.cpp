// tests/dsv41/ep_link_test.cpp - the EP link (src/net/ep_link.{hpp,cpp}): its rules between two processes on
// 127.0.0.1, and its token-shaped probe between two PCs.  No GPU, no model.
//
//   ds4_ep_link_test                  every case below; this process is the main, a child of itself (--child CASE PORT)
//                                     the worker, on 127.0.0.1
//   ds4_ep_link_test --serve [BIND:]PORT [--for S]
//                                     the probe's far end (the worker's PC): a worker's hello, then it answers Pings and
//                                     takes the probe's settings (spin or poll, SO_BUSY_POLL, busy threads, pinning) for
//                                     each arm; exits after S seconds (600)
//   ds4_ep_link_test --probe HOST[:PORT] [--load N] [--pin CPU] [--busy-poll US] [--rounds R] [--n N]
//                                     the main's end: EP-sized round trips (10,320-byte frames each way) at gaps of
//                                     0.5 / 2 / 30 ms and token-shaped (40 at 0.5 ms, then 2 ms, R rounds), each with
//                                     both ends spinning or both waiting in poll (blocking), with SO_BUSY_POLL off and at
//                                     US on the far end; p50 / p90 / p99 / max.  --load N: N busy threads on each side
//                                     (the CPU lanes); --pin CPU: the link's thread on CPU (both sides), the busy
//                                     threads off it and its SMT siblings.  Arrival times come from the kernel's
//                                     rx stamps (SO_TIMESTAMPNS, Linux), never from a read that waits behind other
//                                     work: wire (send -> the reply in the kernel) apart from each end's wake-up
//
// The cases: the hello accepted, and refused for the pack, the slice table (a range, a type), the token, the protocol,
// the geometry, a bulk socket's session; a request's reply (y = 2x, exact in bf16); the seq rules and two outstanding;
// a reply out of order; a timeout -> the caller's local path, then the late reply dropped; three misses in a row ->
// down; a peer killed mid-frame; a bad magic; the spin budget's expiry and a Wake; fills and epochs on the bulk socket
// (a request with an epoch not acked refused).
// The probe is run between two PCs by scripts/dsv41/ep-u1-probe.sh (strata-winbuild-lab); on one PC do not --pin both
// ends to the same CPU (they would take turns on it).
#include "ep_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::net::ep;

namespace {

constexpr int kD = 5120, kL = 40, kE = 384, kK = 6, kFF = 2304, kB = 256, kBlocks = kFF / kB;
constexpr char kToken[] = "ep-link-test-token";

EpHello hello_of(uint32_t role) {
    EpHello h;
    h.role = role;
    h.pack_hash = 0x1234abcd5678ef01ull;
    h.parts_hash = 0x42;
    h.n_embd = kD;
    h.n_layers = kL;
    h.n_experts = kE;
    h.k_picks = kK;
    h.n_ff = kFF;
    h.block = kB;
    h.max_t = kMaxT;
    h.rows_bf16 = 1;
    h.gpu_slots = role == 1 ? 28 : 0;
    h.slot_bytes = role == 1 ? 8648704 : 0;
    std::snprintf(h.build, sizeof h.build, "ep-link-test");
    const char* tk = std::getenv("STRATA_STAGE_TOKEN");
    std::snprintf(h.token, sizeof h.token, "%s", tk != nullptr && *tk ? tk : kToken);
    return h;
}
// IQ2_XXS / IQ2_XS / IQ1_M-like types and their whole rows' bytes; the worker stores [lo, hi)
std::vector<EpSlice> table_of(int lo, int hi) {
    std::vector<EpSlice> t((size_t) kL);
    for (int l = 0; l < kL; ++l) {
        t[(size_t) l].lo = (uint8_t) lo;
        t[(size_t) l].hi = (uint8_t) hi;
        const uint8_t ty[3] = {16, 16, (uint8_t) (l % 2 ? 17 : 29)};
        for (int m = 0; m < 3; ++m) {
            t[(size_t) l].type[m] = ty[m];
            t[(size_t) l].row_bytes[m] = m < 2 ? 1320 : (ty[m] == 17 ? 666 : 504);
        }
    }
    return t;
}
std::vector<int> k_of(int k) { return std::vector<int>((size_t) kL, k); }

int fails = 0;
void check(bool ok, const std::string& what) {
    std::printf("%-74s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++fails;
}
bool has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

void sleep_ms(double ms) { std::this_thread::sleep_for(std::chrono::microseconds((int64_t) (ms * 1000.0))); }

// ---------------------------------------------------------------------------------------------------- a child process
std::string self_path() {
#if defined(_WIN32)
    char b[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, b, MAX_PATH);
    return b;
#else
    char b[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    return n > 0 ? std::string(b, (size_t) n) : std::string();
#endif
}

struct Child {
#if defined(_WIN32)
    HANDLE h = nullptr;
#else
    pid_t pid = -1;
#endif
    bool start(const std::string& mode, int port) {
        const std::string exe = self_path(), p = std::to_string(port);
#if defined(_WIN32)
        std::string cmd = "\"" + exe + "\" --child " + mode + " " + p;
        STARTUPINFOA si{};
        si.cb = sizeof si;
        PROCESS_INFORMATION pi{};
        if (!CreateProcessA(exe.c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return false;
        CloseHandle(pi.hThread);
        h = pi.hProcess;
        return true;
#else
        pid = fork();
        if (pid == 0) {
            execl(exe.c_str(), exe.c_str(), "--child", mode.c_str(), p.c_str(), (char*) nullptr);
            _exit(127);
        }
        return pid > 0;
#endif
    }
    /// its exit code (-1: still running after ms, or no child)
    int wait(int ms) {
#if defined(_WIN32)
        if (h == nullptr || WaitForSingleObject(h, (DWORD) ms) != WAIT_OBJECT_0) return -1;
        DWORD code = 0;
        GetExitCodeProcess(h, &code);
        CloseHandle(h);
        h = nullptr;
        return (int) code;
#else
        for (int i = 0; pid > 0 && i <= ms; ++i) {
            int st = 0;
            if (waitpid(pid, &st, WNOHANG) == pid) {
                pid = -1;
                return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
            }
            sleep_ms(1);
        }
        return -1;
#endif
    }
    void kill() {
#if defined(_WIN32)
        if (h != nullptr) TerminateProcess(h, 9);
#else
        if (pid > 0) ::kill(pid, SIGKILL);
#endif
    }
};

// ---------------------------------------------------------------------------------------------------- the worker child
// y = 2x for every row (exact in bf16)
std::vector<uint16_t> doubled(const uint16_t* x, int T) {
    std::vector<uint16_t> y((size_t) T * kD);
    for (size_t i = 0; i < y.size(); ++i) y[i] = f2bf(2.0f * bf2f(x[i]));
    return y;
}

int child(const std::string& mode, int port) {
    const std::string at = "127.0.0.1:" + std::to_string(port);
    std::string err;
    WorkerLink w;
    EpHello me = hello_of(1);
    std::vector<EpSlice> table = table_of(6, kBlocks);
    if (mode == "bad-pack") me.pack_hash ^= 1;
    if (mode == "bad-token") std::snprintf(me.token, sizeof me.token, "not-the-token");
    if (mode == "bad-proto") me.proto = kProto + 1;
    if (mode == "bad-geo") me.n_ff = 2048;
    if (mode == "bad-slice") table = table_of(7, kBlocks);
    if (mode == "bad-type") table[3].type[2] = 23;
    if (!w.req.dial(at, 5000, err)) return 10;
    Peer m;
    const bool ok = hello_worker(w.req, me, table, m, err);
    if (has(mode, "bad-")) {   // (refused: the main's reason here, the token never in it)
        const bool why = !ok && has(err, "refused") && !has(err, me.token) && !has(err, kToken);
        std::printf("  child %s: %s\n", mode.c_str(), err.c_str());
        return why ? 0 : 11;
    }
    if (!ok) {
        std::printf("  child %s: hello failed: %s\n", mode.c_str(), err.c_str());
        return 12;
    }
    w.start(me);
    if (mode == "bulk-session") {   // (a bulk socket that names another session: refused)
        Conn b;
        if (!b.dial(at, 5000, err)) return 13;
        return !bulk_worker(b, me, m.hello.session ^ 7, err) && has(err, "refused") ? 0 : 14;
    }
    if (mode == "swap" || mode == "badmagic" || mode == "half" || mode == "close") {   // (a worker that breaks the rules: the raw socket)
        EpHdr h;
        std::vector<uint8_t> p;
        if (w.req.recv(h, p, now_ms() + 5000, err) != 1 || h.type != Req) return 15;
        if (mode == "close") {   // an orderly close (FIN, not a reset) with a request out, and no bye
            w.req.close();
            sleep_ms(500);
            return 0;
        }
        const int T = h.T;
        std::vector<uint16_t> y = doubled((const uint16_t*) (p.data() + (size_t) T * kK * sizeof(EpPick)), T);
        uint16_t head[2] = {(uint16_t) ((1u << T) - 1), 0};
        if (mode == "badmagic") {   // a reply's header with its magic wrong
            EpHdr bad;
            bad.type = Resp;
            bad.seq = h.seq;
            bad.magic = 0x01020304u;
            if (!w.req.send_bytes(&bad, sizeof bad, err)) return 16;
            sleep_ms(500);
            return 0;
        }
        if (mode == "half") {   // half of a reply's frame, then nothing until killed
            EpHdr r;
            r.type = Resp;
            r.seq = h.seq;
            r.bytes = (uint32_t) (4 + y.size() * 2);
            std::vector<uint8_t> raw(sizeof r + 4 + y.size());   // (the header, the head, half the rows' bytes)
            std::memcpy(raw.data(), &r, sizeof r);
            std::memcpy(raw.data() + sizeof r, head, 4);
            std::memcpy(raw.data() + sizeof r + 4, y.data(), y.size());
            if (!w.req.send_bytes(raw.data(), raw.size(), err)) return 17;
            sleep_ms(30000);
            return 0;
        }
        // swap: the second request's reply before the first's
        EpHdr h2;
        std::vector<uint8_t> p2;
        if (w.req.recv(h2, p2, now_ms() + 5000, err) != 1 || h2.type != Req) return 18;
        std::vector<uint16_t> y2 = doubled((const uint16_t*) (p2.data() + (size_t) h2.T * kK * sizeof(EpPick)), h2.T);
        EpHdr r;
        r.type = Resp;
        r.seq = h2.seq;
        uint16_t head2[2] = {(uint16_t) ((1u << h2.T) - 1), 0};
        (void) w.req.send(r, head2, 4, err, y2.data(), y2.size() * 2);
        sleep_ms(500);
        return 0;
    }
    if (mode == "spin") {   // the spin budget: past it the worker waits in poll; a Wake gets it spinning again
        w.req.spin_ms = 30.0;
        uint64_t polls_at_wake = 0;
        w.on_wake = [&] { polls_at_wake = w.req.stat.polls; };
        w.on_ping = [&](const EpHdr&, std::vector<uint8_t>& p) {   // (the Pong says: polls by the Wake, polls now)
            const uint64_t v[2] = {polls_at_wake, w.req.stat.polls};
            p.assign((const uint8_t*) v, (const uint8_t*) v + sizeof v);
        };
        WorkerLink::Frame f;
        const int got = w.next(f, now_ms() + 10000, err);   // (the Wake, the Ping, then the Bye)
        return got == -1 && has(err, "bye") ? 0 : 19;
    }
    std::thread bulk_th;
    if (mode == "fill") {   // the bulk socket on a thread of its own, as the worker will run it
        if (!w.bulk.dial(at, 5000, err) || !bulk_worker(w.bulk, me, m.hello.session, err)) return 20;
        bulk_th = std::thread([&w] {
            std::string e;
            const WorkerLink::Loader load = [](int, int e_in, int, uint32_t, const uint8_t* blob, size_t n) -> uint16_t {
                if (e_in == 99) return 1;   // (this one does not load)
                return n == 0 || (n == 4096 && blob[0] == 0xab) ? 0 : 2;
            };
            while (w.serve_fill(now_ms() + 10000, load, e) == 1) {
            }
        });
    }
    // serve: y = 2x; "slow-first": the first request's reply 150 ms late; "silent": none
    int served = 0;
    for (;;) {
        WorkerLink::Frame f;
        const int got = w.next(f, now_ms() + 10000, err);
        if (got != 1) break;
        if (mode == "silent") {
            // (no reply: the request stays in hand, so take the next ones raw)
            for (;;) {
                EpHdr h;
                std::vector<uint8_t> p;
                if (w.req.recv(h, p, now_ms() + 10000, err) != 1) break;
            }
            break;
        }
        if (mode == "slow-first" && served == 0) sleep_ms(150);
        const std::vector<uint16_t> y = doubled(f.x, f.T);
        if (!w.send_resp(f.h.seq, (uint16_t) ((1u << f.T) - 1), 0, y.data(), 7, err)) break;
        ++served;
    }
    if (bulk_th.joinable()) bulk_th.join();   // (it ends when the main closes the bulk socket)
    return 0;
}

// ---------------------------------------------------------------------------------------------------- the main side
// a listener on a free port and the child dialing it, its hello taken: true with the link ready
struct Pair {
    Listener ls;
    Child ch;
    MainLink m;
    EpHello me = hello_of(0);
    Peer w;
    std::string err;
    bool hello_ok = false;
    bool open(const std::string& mode, const std::vector<int>& k = k_of(6)) {
        if (!ls.open("127.0.0.1:0", err) || !ch.start(mode, ls.port())) return false;
        if (!ls.accept(m.req, 10000, err)) return false;
        hello_ok = hello_main(m.req, me, table_of(0, kBlocks), k, w, err);
        if (hello_ok) m.start(me);
        return true;
    }
    int end(int ms = 10000) {
        const int rc = ch.wait(ms);
        if (rc < 0) ch.kill();
        return rc < 0 ? ch.wait(2000) : rc;
    }
};

// a request of T rows: x[t][i] = (t + 1) * small values, picks tagged to the worker's slice lane
struct ReqData {
    std::vector<EpPick> picks;
    std::vector<uint16_t> x;
    explicit ReqData(int T, int salt = 0) : picks((size_t) T * kK), x((size_t) T * kD) {
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < kK; ++i) {
                picks[(size_t) t * kK + i].e = (int16_t) ((t * 7 + i * 31 + salt) % kE);
                picks[(size_t) t * kK + i].lane = LaneSlice;
                picks[(size_t) t * kK + i].w = 0.1f * (float) (i + 1);
            }
        for (size_t j = 0; j < x.size(); ++j) x[j] = f2bf(0.001f * (float) ((j * 37 + (size_t) salt) % 2001) - 1.0f);
    }
    bool is_doubled(const Reply& r, int T) const {
        if (r.rowmask != (uint16_t) ((1u << T) - 1) || r.status != 0) return false;
        for (size_t j = 0; j < x.size(); ++j)
            if (r.rows[j] != f2bf(2.0f * bf2f(x[j]))) return false;
        return true;
    }
};

void case_pure() {
    const EpHello m = hello_of(0), w = hello_of(1);
    const auto mt = table_of(0, kBlocks), wt = table_of(6, kBlocks);
    check(main_check(m, mt, k_of(6), w, wt).empty(), "pure: the worker's hello and slice table accepted at k 6");
    check(main_check(m, mt, k_of(7), w, wt).empty(), "pure: ... and at k 7 (less asked of the worker)");
    check(has(main_check(m, mt, k_of(5), w, wt), "stores blocks [6, 9)"), "pure: refused at k 5 (the worker lacks block 5)");
    std::vector<int> k = k_of(6);
    k[5] = kBlocks;   // (a layer the worker is not asked for)
    auto wt2 = wt;
    wt2[5].lo = wt2[5].hi = 0;
    check(main_check(m, mt, k, w, wt2).empty(), "pure: a layer the worker stores none of, asked for none");
    check(!slices_cover(wt2, k_of(6), kBlocks).empty(), "pure: ... refused once that layer is asked for");
    auto wt3 = wt;
    wt3[9].hi = 8;
    check(has(main_check(m, mt, k_of(6), w, wt3), "stores blocks"), "pure: a range that stops short of n_ff refused");
    auto wt4 = wt;
    wt4[2].row_bytes[1] += 2;
    check(has(main_check(m, mt, k_of(6), w, wt4), "rows"), "pure: a row size that differs refused");
    EpHello w2 = w;
    w2.role = 0;
    check(has(main_check(m, mt, k_of(6), w2, wt), "both sides"), "pure: two mains refused");
    w2 = w;
    w2.max_t = 8;
    check(has(worker_check(w, m) + main_check(m, mt, k_of(6), w2, wt), "max T"), "pure: max T differs refused");
    w2 = w;
    w2.token[0] ^= 1;
    const std::string why = worker_check(w2, m);
    check(has(why, "token") && !has(why, m.token), "pure: the worker refuses the main's token (never printing it)");
    check(f2bf(1.0f) == 0x3f80 && f2bf(1.00390625f) == 0x3f80 && f2bf(1.01171875f) == 0x3f82 && bf2f(0x4000) == 2.0f,
          "pure: bf16 rounds to nearest even");
}

void case_refusals() {
    struct C {
        const char* mode;
        const char* why;
    };
    const C cs[] = {{"bad-pack", "pack"},       {"bad-token", "token"}, {"bad-proto", "protocol"},
                    {"bad-geo", "n_ff"},        {"bad-slice", "stores blocks"}, {"bad-type", "type"}};
    for (const C& c : cs) {
        Pair p;
        const bool opened = p.open(c.mode);
        const int rc = p.end();
        check(opened && !p.hello_ok && has(p.err, c.why) && rc == 0,
              std::string("hello refused: ") + c.mode + " (both sides say why)");
    }
}

void case_accept_and_rules() {
    Pair p;
    const bool opened = p.open("good");
    check(opened && p.hello_ok && p.w.hello.parts_hash == 0x42 && p.w.hello.gpu_slots == 28 && p.w.table.size() == (size_t) kL,
          "hello accepted: the worker's parts hash, GPU slots and slice table read");
    std::string err;
    Reply r;
    ReqData a(1, 1), b(3, 2), c(16, 3);
    check(p.m.send_req(10, 0, 1, 0, 0, a.picks.data(), a.x.data(), err) == 1 &&
              p.m.wait_resp(10, now_ms() + 2000, r, err) == 1 && r.seq == 10 && a.is_doubled(r, 1) && r.lane_us == 7,
          "a request of 1 row and its reply (y = 2x, bf16 exact)");
    check(p.m.send_req(11, 39, 16, 0, kReqDeferred, c.picks.data(), c.x.data(), err) == 1 &&
              p.m.wait_resp(11, now_ms() + 2000, r, err) == 1 && c.is_doubled(r, 16),
          "a request of 16 rows (the widest window)");
    check(p.m.send_req(11, 1, 1, 0, 0, a.picks.data(), a.x.data(), err) == 0 && has(err, "not after"),
          "a seq not after the last one: not sent (computed here)");
    check(p.m.send_req(20, 1, 1, 0, 0, a.picks.data(), a.x.data(), err) == 1 &&
              p.m.send_req(21, 2, 3, 0, 0, b.picks.data(), b.x.data(), err) == 1 &&
              p.m.send_req(22, 3, 1, 0, 0, a.picks.data(), a.x.data(), err) == 0 && has(err, "two") && p.m.outstanding() == 2,
          "two requests out: a third is not sent");
    check(p.m.wait_resp(21, now_ms() + 2000, r, err) == -1 && has(err, "oldest") && p.m.up(),
          "waiting out of order: refused, the link stays up");
    check(p.m.wait_resp(20, now_ms() + 2000, r, err) == 1 && a.is_doubled(r, 1) &&
              p.m.wait_resp(21, now_ms() + 2000, r, err) == 1 && b.is_doubled(r, 3),
          "both replies in order");
    check(p.m.send_req(23, 0, 1, 1, 0, a.picks.data(), a.x.data(), err) == 0 && has(err, "epoch"),
          "an epoch the worker has not loaded: not sent");
    p.m.bye();
    check(p.end() == 0, "the worker ends at the main's bye");
}

void case_order() {
    Pair p;
    p.open("swap");
    std::string err;
    Reply r;
    ReqData a(1, 1), b(1, 2);   // (the same T: only the seq tells the two replies apart)
    p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
    p.m.send_req(2, 0, 1, 0, 0, b.picks.data(), b.x.data(), err);
    const int got = p.m.wait_resp(1, now_ms() + 3000, r, err);
    check(got == -1 && has(err, "seq 2") && has(err, "oldest") && !p.m.up() && p.m.st.downs == 1,
          "a reply out of order: the link is down");
    p.end();
}

void case_closed() {
    Pair p;
    p.open("close");
    std::string err;
    Reply r;
    ReqData a(1, 1);
    p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
    const double t0 = now_ms();
    const int got = p.m.wait_resp(1, t0 + 3000, r, err);
    check(got == -1 && has(err, "closed") && !has(err, "mid-frame") && !p.m.up() && now_ms() - t0 < 2000,
          "the worker closes its socket: the link is down at once, not at the deadline (" + err + ")");
    p.end();
}

void case_timeouts() {
    {
        Pair p;
        p.open("slow-first");
        std::string err;
        Reply r;
        ReqData a(1, 1), b(1, 2);
        p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        const int t1 = p.m.wait_resp(1, now_ms() + 20, r, err);
        check(t1 == 0 && p.m.up() && p.m.st.timeouts == 1, "no reply by the deadline: 0, the caller computes it here");
        p.m.send_req(2, 0, 1, 0, 0, b.picks.data(), b.x.data(), err);
        const int t2 = p.m.wait_resp(2, now_ms() + 2000, r, err);
        check(t2 == 1 && r.seq == 2 && b.is_doubled(r, 1) && p.m.st.late_dropped == 1,
              "the late reply dropped by seq, the next one taken");
        p.m.bye();
        p.end();
    }
    {   // a worker that never answers: two timeouts, then no room for a third request (both out are late) - that is
        // the third miss in a row, and the link is down
        Pair p;
        p.open("silent");
        std::string err;
        Reply r;
        ReqData a(1, 1);
        const int s1 = p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        const int w1 = p.m.wait_resp(1, now_ms() + 10, r, err);
        const int s2 = p.m.send_req(2, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        const int w2 = p.m.wait_resp(2, now_ms() + 10, r, err);
        const bool up2 = p.m.up();
        const int s3 = p.m.send_req(3, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        check(s1 == 1 && w1 == 0 && s2 == 1 && w2 == 0 && up2 && s3 == 0 && !p.m.up() && has(err, "in a row") && p.m.st.downs == 1,
              "three misses in a row (two timeouts, then no room): the link is down");
        const int s4 = p.m.send_req(4, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        check(s4 == 0 && has(err, "down"), "... and every request after it computed here");
        p.end();
    }
}

void case_killed() {
    Pair p;
    p.open("half");
    std::string err;
    Reply r;
    ReqData a(2, 1);
    p.m.send_req(1, 0, 2, 0, 0, a.picks.data(), a.x.data(), err);
    const int w1 = p.m.wait_resp(1, now_ms() + 300, r, err);   // (half a frame in: not a reply)
    const bool partial = p.m.req.buffered() > 0;
    p.ch.kill();
    (void) p.end(3000);
    const int s2 = p.m.send_req(2, 0, 2, 0, 0, a.picks.data(), a.x.data(), err);
    const int w2 = s2 == 1 ? p.m.wait_resp(2, now_ms() + 3000, r, err) : s2;
    check(w1 == 0 && partial && w2 == -1 && has(err, "mid-frame") && !p.m.up(),
          "the worker killed mid-frame: the half frame is no reply, then the link is down (" + err + ")");
}

void case_badmagic() {
    Pair p;
    p.open("badmagic");
    std::string err;
    Reply r;
    ReqData a(1, 1);
    p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
    const int got = p.m.wait_resp(1, now_ms() + 3000, r, err);
    check(got == -1 && has(err, "magic") && !p.m.up(), "a frame with a bad magic: the link is down");
    p.end();
}

void case_spin_wake() {
    Pair p;
    p.open("spin");
    std::string err;
    sleep_ms(120);   // (past the worker's 30 ms budget: it waits in poll)
    const bool woke = p.m.wake(err);
    sleep_ms(2);     // (inside the budget the Wake restarted)
    std::vector<uint8_t> v(16, 0);
    uint32_t held = 0;
    const int got = p.m.ping(5, 0, now_ms() + 2000, held, err, &v);
    uint64_t s[2] = {0, 0};
    if (v.size() == sizeof s) std::memcpy(s, v.data(), sizeof s);
    check(woke && got == 1 && s[0] > 0, "past the spin budget the worker waited in poll (" + std::to_string(s[0]) + " waits)");
    check(got == 1 && s[1] == s[0], "after a Wake it spins: no poll between the Wake and the next frame");
    p.m.bye();
    check(p.end() == 0, "the worker ends at the bye");
}

void case_fill() {
    Pair p;
    p.open("fill");
    std::string err;
    const bool bulk = p.ls.accept(p.m.bulk, 5000, err) && bulk_main(p.m.bulk, p.me, err);
    check(bulk, "the bulk socket accepted for the session");
    std::vector<uint8_t> blob(4096, 0xab);
    int layer = -1, e = -1;
    uint32_t ep = 0;
    uint16_t status = 9;
    const bool f1 = p.m.send_fill(3, 17, 5, 1, blob.data(), blob.size(), err) && p.m.send_fill(4, 18, -1, 1, nullptr, 0, err);
    const bool ready_early = p.m.epoch_ready() == 0;
    const int a = p.m.poll_fill_ok(now_ms() + 2000, layer, e, ep, status, err);
    const bool first = a == 1 && layer == 3 && e == 17 && ep == 1 && status == 0 && p.m.epoch_ready() == 0;
    const int b = p.m.poll_fill_ok(now_ms() + 2000, layer, e, ep, status, err);
    check(f1 && ready_early && first && b == 1 && p.m.epoch_ready() == 1, "two fills for epoch 1: the epoch is ready after both FillOks");
    Reply r;
    ReqData x(1, 1);
    check(p.m.send_req(1, 0, 1, 1, 0, x.picks.data(), x.x.data(), err) == 1 && p.m.wait_resp(1, now_ms() + 2000, r, err) == 1 &&
              x.is_doubled(r, 1),
          "a request tagged with epoch 1 served");
    check(!p.m.send_fill(3, 20, 6, 3, nullptr, 0, err) && has(err, "epoch"), "a fill that skips an epoch: refused here");
    const bool f2 = p.m.send_fill(5, 99, 1, 2, nullptr, 0, err);
    const int c = p.m.poll_fill_ok(now_ms() + 2000, layer, e, ep, status, err);
    check(f2 && c == 1 && status == 1 && p.m.epoch_ready() == 1 && p.m.st.fill_fails == 1,
          "a fill the worker could not load: epoch 2 not ready");
    check(p.m.send_req(2, 0, 1, 2, 0, x.picks.data(), x.x.data(), err) == 0 && has(err, "epoch"),
          "a request tagged with epoch 2: not sent");
    // the same request sent past MainLink's check: the worker refuses it
    EpHdr h;
    h.type = Req;
    h.T = 1;
    h.seq = 3;
    h.epoch = 2;
    std::string e2;
    (void) p.m.req.send(h, x.picks.data(), x.picks.size() * sizeof(EpPick), e2, x.x.data(), x.x.size() * 2);
    EpHdr g;
    std::vector<uint8_t> pl;
    const int got = p.m.req.recv(g, pl, now_ms() + 2000, e2);
    check(got == 1 && g.type == Error && has(std::string(pl.begin(), pl.end()), "epoch 2"),
          "a request with an epoch the worker has not acked: refused by the worker");
    p.m.bye();
    p.end();
}

void case_bulk_session() {
    Pair p;
    p.open("bulk-session");
    std::string err;
    const bool acc = p.ls.accept(p.m.bulk, 5000, err);
    const bool ok = acc && bulk_main(p.m.bulk, p.me, err);
    check(acc && !ok && has(err, "session"), "a bulk socket naming another session: refused");
    check(p.end() == 0, "... and the worker hears why");
}

// ---------------------------------------------------------------------------------------------------- the probe
struct ProbeCfg {   // the probe's settings for the far end, in a Ping of aux kCfg (its Pong: the same, results filled)
    int32_t spin = 1, busy_us = 0, load = 0, pin = -1;
    int32_t busy_ok = 0, cpus = 0, stamps_ok = 0;
    char note[100] = {};
};
constexpr uint32_t kCfg = 0x100;

std::vector<int> siblings(int cpu) {
    std::vector<int> v{cpu};
#if !defined(_WIN32)
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    std::string s;
    if (f >> s) {   // "3,19" or "3-4"
        v.clear();
        for (size_t a = 0; a < s.size();) {
            size_t b = s.find(',', a);
            if (b == std::string::npos) b = s.size();
            const std::string r = s.substr(a, b - a);
            const size_t d = r.find('-');
            const int lo = std::atoi(r.c_str()), hi = d == std::string::npos ? lo : std::atoi(r.c_str() + d + 1);
            for (int c = lo; c <= hi; ++c) v.push_back(c);
            a = b + 1;
        }
    }
#endif
    return v;
}
void pin_to(int cpu) {   // (-1: any CPU again)
#if defined(_WIN32)
    DWORD_PTR proc = 0, sys = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
    SetThreadAffinityMask(GetCurrentThread(), cpu >= 0 && cpu < 64 ? (DWORD_PTR) 1 << cpu : proc);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    const int n = (int) std::max(1u, std::thread::hardware_concurrency());
    for (int c = 0; c < n; ++c)
        if (cpu < 0 || c == cpu) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof set, &set);
#endif
}

// N busy threads streaming through memory of their own (a CPU lane's load), off the reserved CPUs when pin >= 0
class Load {
public:
    ~Load() { stop(); }
    void start(int n, int pin) {
        stop();
        quit_ = false;
        const int cpus = (int) std::max(1u, std::thread::hardware_concurrency());
        std::vector<int> free_cpus;
        const std::vector<int> res = pin >= 0 ? siblings(pin) : std::vector<int>{};
        for (int c = 0; c < cpus; ++c)
            if (std::find(res.begin(), res.end(), c) == res.end()) free_cpus.push_back(c);
        for (int i = 0; i < n; ++i)
            th_.emplace_back([this, i, pin, free_cpus] {
                if (pin >= 0 && !free_cpus.empty()) pin_to(free_cpus[(size_t) i % free_cpus.size()]);
                std::vector<float> buf((size_t) 4 << 20, 1.0f);   // (16 MB: past the caches)
                float s = 0.0f;
                while (!quit_.load(std::memory_order_relaxed))
                    for (size_t j = 0; j < buf.size(); j += 16) s += buf[j];
                sink_ += s;
            });
    }
    void stop() {
        quit_ = true;
        for (auto& t : th_) t.join();
        th_.clear();
    }

private:
    std::vector<std::thread> th_;
    std::atomic<bool> quit_{false};
    volatile float sink_ = 0.0f;
};

int serve(const std::string& at, double for_s) {
    Listener ls;
    std::string err;
    if (!ls.open(at, err)) {
        std::fprintf(stderr, "ds4_ep_link_test --serve: %s\n", err.c_str());
        return 2;
    }
    std::printf("ds4_ep_link_test --serve: listening on port %d for %.0f s\n", ls.port(), for_s);
    std::fflush(stdout);
    const double end = now_ms() + for_s * 1000.0;
    Load load;
    while (now_ms() < end) {
        WorkerLink w;
        if (!ls.accept(w.req, (int) std::min(60000.0, end - now_ms()), err)) continue;
        const EpHello me = hello_of(1);
        Peer m;
        if (!hello_worker(w.req, me, table_of(6, kBlocks), m, err)) {
            std::printf("  hello from %s: %s\n", w.req.peer().c_str(), err.c_str());
            continue;
        }
        w.start(me);
        std::string se;
        const bool stamps = w.req.rx_stamps(se);
        w.on_ping = [&](const EpHdr& h, std::vector<uint8_t>& p) {
            if ((h.aux & kCfg) == 0) {   // (a probe's ping: the first 8 bytes back say how long it sat in the kernel here)
                const int64_t rx = w.req.last_rx_ns();
                const int64_t wait = stamps && rx > 0 ? wall_ns() - rx : -1;
                if (p.size() >= sizeof wait) std::memcpy(p.data(), &wait, sizeof wait);
                return;
            }
            if (p.size() != sizeof(ProbeCfg)) return;
            ProbeCfg c;
            std::memcpy(&c, p.data(), sizeof c);
            w.req.spin_ms = c.spin ? kSpinMs : 0.0;
            std::string e;
            c.busy_ok = w.req.busy_poll(c.busy_us, e) ? 1 : 0;
            c.stamps_ok = stamps ? 1 : 0;
            std::snprintf(c.note, sizeof c.note, "%s", c.busy_ok ? "" : e.c_str());
            load.start(c.load, c.pin);
            pin_to(c.pin);
            c.cpus = (int32_t) std::thread::hardware_concurrency();
            std::memcpy(p.data(), &c, sizeof c);
        };
        WorkerLink::Frame f;
        const int got = w.next(f, end, err);
        load.stop();
        std::printf("  probe from %s ended (%s), %llu pings\n", w.req.peer().c_str(), got < 0 ? err.c_str() : "time",
                    (unsigned long long) w.st.pings);
        std::fflush(stdout);
    }
    return 0;
}

// rtt: send -> this thread has the Pong (steady clock).  With the kernel's rx stamps (SO_TIMESTAMPNS, both ends):
// wire = send -> the Pong's arrival in the kernel here, so it holds no wait of this thread (the far end's wake and
// held are in it: the network's own share is wire - there - held); here / there = arrival ->
// the link thread has the frame (the Pong here, the Ping on the far end): the thread's own wake-up, what the busy
// threads can stretch.  held: the far end's Ping in hand -> its Pong sent.
struct Arm {
    std::vector<double> rtt, held, wire, here, there;
};
void report(const char* name, Arm& a) {
    auto pct = [](std::vector<double> v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t) (q * (double) (v.size() - 1) + 0.5))];
    };
    auto mx = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    std::printf("  %-36s n %4zu  rtt p50 %.3f  p90 %.3f  p99 %.3f  max %.3f ms   far end held p50 %.1f us\n", name, a.rtt.size(),
                pct(a.rtt, 0.5), pct(a.rtt, 0.9), pct(a.rtt, 0.99), mx(a.rtt), pct(a.held, 0.5));
    if (!a.wire.empty())
        std::printf("  %-36s wire p50 %.3f  p99 %.3f  max %.3f ms   wake here p50 %.1f p99 %.1f max %.1f us   there p50 %.1f p99 %.1f max %.1f us\n",
                    "", pct(a.wire, 0.5), pct(a.wire, 0.99), mx(a.wire), pct(a.here, 0.5), pct(a.here, 0.99), mx(a.here),
                    pct(a.there, 0.5), pct(a.there, 0.99), mx(a.there));
    std::fflush(stdout);
}

int probe(const std::string& host, int load_n, int pin, int busy_us, int rounds, int n_arm) {
    const std::string at = host.find(':') == std::string::npos ? host + ":" + std::to_string(kDefaultPort) : host;
    MainLink m;
    std::string err;
    if (!m.req.dial(at, 5000, err)) {
        std::fprintf(stderr, "ds4_ep_link_test --probe: %s\n", err.c_str());
        return 2;
    }
    EpHello me = hello_of(0);
    Peer w;
    if (!hello_main(m.req, me, table_of(0, kBlocks), k_of(6), w, err)) {
        std::fprintf(stderr, "ds4_ep_link_test --probe: %s\n", err.c_str());
        return 2;
    }
    m.start(me);
    std::string se;
    const bool stamps = m.req.rx_stamps(se);
    pin_to(pin);
    Load load;
    load.start(load_n, pin);
    constexpr size_t kPay = 10320 - sizeof(EpHdr);   // (an EP request's frame at T = 1: 10,320 bytes each way)
    uint32_t seq = 0;
    std::printf("ds4_ep_link_test --probe %s: frames of %zu bytes each way; busy threads %d here and there; link thread on CPU %d\n",
                at.c_str(), kPay + sizeof(EpHdr), load_n, pin);
    std::printf("  (the reference: ping -A of 10,240 bytes CT 580 -> 10.10.10.2, 0.060 / 0.067 / 0.101 ms; adaptive-rx.txt)\n");
    std::printf("  kernel rx stamps here: %s\n", stamps ? "on (wire / wake columns)" : se.c_str());
    std::vector<uint8_t> buf(kPay);
    int rc = 0;
    for (const int bp : {0, busy_us}) {
        if (bp == 0 && busy_us == 0 && rc == 1) break;
        for (const int spin : {1, 0}) {
            ProbeCfg c;
            c.spin = spin;
            c.busy_us = bp;
            c.load = load_n;
            c.pin = pin;
            std::vector<uint8_t> v(sizeof c);
            std::memcpy(v.data(), &c, sizeof c);
            uint32_t held = 0;
            if (m.ping(++seq, 0, now_ms() + 5000, held, err, &v, kCfg) != 1) {
                std::fprintf(stderr, "  the far end did not take the settings: %s\n", err.c_str());
                return 2;
            }
            std::memcpy(&c, v.data(), sizeof c);
            m.req.spin_ms = spin ? kSpinMs : 0.0;
            std::printf("arm: %s, SO_BUSY_POLL %d us on the far end (%s), far end %d CPUs, rx stamps there %s\n",
                        spin ? "both ends spin (keep-awake)" : "both ends wait in poll (blocking recv)", bp,
                        bp == 0 ? "off" : c.busy_ok ? "set" : c.note, c.cpus, c.stamps_ok ? "on" : "off");
            sleep_ms(200);   // (the far end's busy threads up)
            auto one = [&](Arm& a) {
                uint32_t hu = 0;
                std::fill(buf.begin(), buf.end(), (uint8_t) 0);
                buf.resize(kPay);
                const int64_t w0 = wall_ns();
                const double t0 = now_ms();
                const int got = m.ping(++seq, kPay, t0 + 1000.0, hu, err, &buf);
                const double t1 = now_ms();
                const int64_t w1 = wall_ns();
                if (got == 1) {
                    a.rtt.push_back(t1 - t0);
                    a.held.push_back((double) hu / 1000.0);
                    const int64_t rx = m.req.last_rx_ns();
                    int64_t there = -1;
                    if (buf.size() >= sizeof there) std::memcpy(&there, buf.data(), sizeof there);
                    if (stamps && rx > 0 && c.stamps_ok && there >= 0) {
                        a.wire.push_back((double) (rx - w0) / 1.0e6);
                        a.here.push_back((double) (w1 - rx) / 1.0e3);
                        a.there.push_back((double) there / 1.0e3);
                    }
                }
                return got >= 0;
            };
            auto gap = [&](double ms) {
                if (spin) {
                    const double until = now_ms() + ms;
                    while (now_ms() < until) relax();
                } else
                    sleep_ms(ms);
            };
            Arm warm;
            for (int i = 0; i < 20; ++i) one(warm);
            for (const double g : {0.5, 2.0, 30.0}) {
                Arm a;
                const int n = g >= 30.0 ? std::max(20, n_arm / 4) : n_arm;
                for (int i = 0; i < n; ++i) {
                    gap(g);
                    if (!one(a)) return 2;
                }
                char name[64];
                std::snprintf(name, sizeof name, "gap %.1f ms", g);
                report(name, a);
            }
            Arm tok;
            for (int r = 0; r < rounds; ++r) {
                gap(2.0);
                for (int i = 0; i < 40; ++i) {
                    if (i) gap(0.5);
                    if (!one(tok)) return 2;
                }
            }
            char name[64];
            std::snprintf(name, sizeof name, "token-shaped (40 x 0.5 ms, 2 ms) x %d", rounds);
            report(name, tok);
        }
        if (busy_us == 0) break;
    }
    load.stop();
    m.bye();
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--child") return child(argv[2], std::atoi(argv[3]));
    if (argc >= 2 && (std::string(argv[1]) == "--serve" || std::string(argv[1]) == "--probe")) {
        std::string serve_at, probe_at;
        int load = 0, pin = -1, busy = 0, rounds = 3, n = 400;
        double for_s = 600;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const bool v = i + 1 < argc;
            if (a == "--serve" && v) serve_at = argv[++i];
            else if (a == "--probe" && v) probe_at = argv[++i];
            else if (a == "--load" && v) load = std::atoi(argv[++i]);
            else if (a == "--pin" && v) pin = std::atoi(argv[++i]);
            else if (a == "--busy-poll" && v) busy = std::atoi(argv[++i]);
            else if (a == "--rounds" && v) rounds = std::atoi(argv[++i]);
            else if (a == "--n" && v) n = std::atoi(argv[++i]);
            else if (a == "--for" && v) for_s = std::atof(argv[++i]);
            else {
                std::fprintf(stderr, "usage: ds4_ep_link_test [--serve [BIND:]PORT [--for S] | --probe HOST[:PORT] [--load N] [--pin CPU] "
                                     "[--busy-poll US] [--rounds R] [--n N]]\n");
                return 2;
            }
        }
        return !serve_at.empty() ? serve(serve_at, for_s) : probe(probe_at, load, pin, busy, rounds, n);
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: ds4_ep_link_test (the cases) | --serve ... | --probe ...\n");
        return 2;
    }
    case_pure();
    case_refusals();
    case_accept_and_rules();
    case_order();
    case_closed();
    case_timeouts();
    case_killed();
    case_badmagic();
    case_spin_wake();
    case_fill();
    case_bulk_session();
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails != 0;
}
