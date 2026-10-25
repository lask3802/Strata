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
//                                     work: wire (send -> the reply in the kernel) apart from each end's wake-up; each
//                                     row counts its pings that timed out and those that came unstamped
//
// The cases: the hello accepted, and refused (both ways, the reason said to the peer) for the pack, the slice table (a
// range, a type), the token, the protocol, the geometry, a short or oversized hello; no token: a WARNING on both
// sides; a new session every hello; one worker per listener, the bulk socket matched by its session; a request's reply
// (y = 2x, exact in bf16); the seq and epoch rules and two outstanding; a reply out of order; a timeout -> the caller's
// local path, then the late reply dropped; no room is not a miss; three drops in a row, or the oldest request over a
// second old -> down (a request dropped at its deadline, not one whose reply is in); a peer killed mid-frame, closed,
// reset; a bad magic; the spin budget's expiry and a Wake; the read buffer kept small; fills (an expert-sized one) in
// batches on the fill thread, the epochs they make ready, one fill a layer a batch, a fill waiting for the request that
// frees its slots, the bulk socket's failure taking the link down, a send to a peer that does not read cut short by the
// link's end, a request-path failure in the middle of a fill and the reconnect after it; protocols 1 and 3 and an
// oversize hello refused with their protocol; two threads in one listener's accept; the kernel's rx stamps; pin_to's
// restore.
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
#include <functional>
#include <mutex>
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
// the token in s (one too short to tell from other text is not looked for)
bool leaks(const std::string& s, const char* token) { return std::strlen(token) >= 8 && has(s, token); }
bool wait_until(const std::function<bool()>& f, double ms) {
    const double end = now_ms() + ms;
    while (!f() && now_ms() < end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return f();
}
constexpr size_t kBig = 8648704;   // an expert's blob (the worker's slot_bytes)

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

// a raw hello (no handshake behind it): the main's answer, an Error, is read back; 0 when it says `why`
int raw_hello(Conn& c, const std::vector<uint8_t>& payload, const char* why) {
    std::string err;
    EpHdr h;
    h.type = Hello;
    if (!c.send(h, payload.data(), payload.size(), err)) return 40;
    EpHdr g;
    std::vector<uint8_t> p;
    if (c.recv(g, p, now_ms() + 5000, err) != 1 || g.type != Error) return 41;
    const std::string said(p.begin(), p.end());
    std::printf("  child: the main said: %s\n", said.c_str());
    return has(said, why) ? 0 : 42;
}

// a worker whose first connection fails under a stuck fill (the bulk socket is never read, then a reply with a bad
// magic) and which dials again, as the engine's reconnect does: a new hello, a new bulk socket, fills and requests served
int child_reconnect(const std::string& at, const EpHello& me, const std::vector<EpSlice>& table, WorkerLink& w, Peer& m) {
    std::string err;
    if (!w.bulk.dial(at, 5000, err) || !bulk_worker(w.bulk, me, m.hello.session, err)) return 51;
    EpHdr h;
    std::vector<uint8_t> p;
    if (w.req.recv(h, p, now_ms() + 5000, err) != 1 || h.type != Req) return 52;
    EpHdr bad;
    bad.type = Resp;
    bad.seq = h.seq;
    bad.magic = 0x01020304u;
    (void) w.req.send_bytes(&bad, sizeof bad, err);
    sleep_ms(300);
    w.req.close();
    w.bulk.close();
    if (!w.req.dial(at, 5000, err)) return 53;
    Peer m2;
    if (!hello_worker(w.req, me, table, m2, err)) return 54;
    w.start(me);
    if (!w.bulk.dial(at, 5000, err) || !bulk_worker(w.bulk, me, m2.hello.session, err)) return 55;
    std::thread th([&w] {
        std::string e;
        const WorkerLink::Loader load = [](int, int, int, uint32_t, const uint8_t*, size_t n) -> uint16_t { return n == 0 ? 0 : 2; };
        while (w.serve_fill(now_ms() + 10000, load, e) == 1) {
        }
    });
    for (;;) {
        WorkerLink::Frame f;
        if (w.next(f, now_ms() + 10000, err) != 1) break;
        const std::vector<uint16_t> y = doubled(f.x, f.T);
        if (!w.send_resp(f.h.seq, (uint16_t) ((1u << f.T) - 1), 0, y.data(), 7, err)) break;
    }
    th.join();
    return 0;
}

int child(const std::string& mode, int port) {
    const std::string at = "127.0.0.1:" + std::to_string(port);
    std::string err;
    WorkerLink w;
    Conn quiet;   // (bulk-silent: a connection that says nothing)
    EpHello me = hello_of(1);
    std::vector<EpSlice> table = table_of(6, kBlocks);
    if (mode == "bad-pack") me.pack_hash ^= 1;
    if (mode == "bad-token") std::snprintf(me.token, sizeof me.token, "not-the-token");
    if (mode == "bad-proto") me.proto = kProto + 1;
    if (mode == "bad-geo") me.n_ff = 2048;
    if (mode == "bad-slice") table = table_of(7, kBlocks);
    if (mode == "bad-type") table[3].type[2] = 23;
    if (mode == "no-token") std::memset(me.token, 0, sizeof me.token);
    if (mode == "extra-token") me.token[0] ^= 1;
    if (!w.req.dial(at, 5000, err)) return 10;
    if (mode == "raw-short") {   // a hello of the right magic and protocol, cut short
        std::vector<uint8_t> p(100, 0);
        std::memcpy(p.data(), &me, 8);
        return raw_hello(w.req, p, "short");
    }
    if (mode == "raw-proto3") {   // another protocol's hello, its table an odd size: refused for the protocol
        EpHello h2 = me;
        h2.proto = 3;
        std::vector<uint8_t> p(sizeof h2 + 7, 0);
        std::memcpy(p.data(), &h2, sizeof h2);
        return raw_hello(w.req, p, "protocol 3");
    }
    if (mode == "raw-proto1") {   // the hello of the build before protocol 2: the same size, a table of 40 rows
        EpHello h1 = me;
        h1.proto = 1;
        const std::vector<EpSlice> t1 = table_of(6, kBlocks);
        std::vector<uint8_t> p(sizeof h1 + t1.size() * sizeof(EpSlice), 0);
        std::memcpy(p.data(), &h1, sizeof h1);
        std::memcpy(p.data() + sizeof h1, t1.data(), t1.size() * sizeof(EpSlice));
        return raw_hello(w.req, p, "protocol 1");
    }
    if (mode == "big-proto3") {   // a hello larger than a hello, of another protocol: told the protocol, not only "over"
        EpHdr h;
        h.type = Hello;
        h.bytes = 3000;
        const uint32_t pm[2] = {kMagic, 3};
        std::vector<uint8_t> raw(sizeof h + sizeof pm);
        std::memcpy(raw.data(), &h, sizeof h);
        std::memcpy(raw.data() + sizeof h, pm, sizeof pm);
        if (!w.req.send_bytes(raw.data(), raw.size(), err)) return 43;
        EpHdr g;
        std::vector<uint8_t> p;
        if (w.req.recv(g, p, now_ms() + 5000, err) != 1 || g.type != Error) return 41;
        const std::string said(p.begin(), p.end());
        std::printf("  child: the main said: %s\n", said.c_str());
        return has(said, "protocol 3") ? 0 : 42;
    }
    if (mode == "big-hello") {   // a header that claims a megabyte of hello, and nothing after it
        EpHdr h;
        h.type = Hello;
        h.bytes = 1u << 20;
        if (!w.req.send_bytes(&h, sizeof h, err)) return 43;
        sleep_ms(3000);
        return 0;
    }
    Peer m;
    const bool ok = hello_worker(w.req, me, table, m, err);
    if (has(mode, "bad-") || mode == "refused-here" || mode == "extra" || mode == "extra-token") {   // (refused: the reason here, the token never in it)
        const char* want = mode == "refused-here" ? "pack" : mode == "extra" ? "already connected" : mode == "extra-token" ? "token" : "refused";
        // (a second worker without the token is not told a worker is here)
        const bool why = !ok && has(err, "refused") && has(err, want) && !(mode == "extra-token" && has(err, "already")) &&
                         !leaks(err, me.token) && !leaks(err, kToken);
        std::printf("  child %s: %s\n", mode.c_str(), err.c_str());
        return why ? 0 : 11;
    }
    if (!ok) {
        std::printf("  child %s: hello failed: %s\n", mode.c_str(), err.c_str());
        return 12;
    }
    if (mode == "no-token" && token_warnings() != 1) return 13;
    w.start(me);
    if (mode == "bulk-check") {   // the main's bulk answers (raw): another session, then another token - both refused
        Conn b1, b2;
        std::string e1, e2;
        const bool r1 = b1.dial(at, 5000, e1) && !bulk_worker(b1, me, m.hello.session, e1) && has(e1, "session");
        const bool r2 = b2.dial(at, 5000, e2) && !bulk_worker(b2, me, m.hello.session, e2) && has(e2, "token") &&
                        !leaks(e2, me.token);
        std::printf("  child bulk-check: %s / %s\n", e1.c_str(), e2.c_str());
        return r1 && r2 ? 0 : 14;
    }
    if (mode == "swap" || mode == "badmagic" || mode == "half" || mode == "close" || mode == "reset") {   // (a worker that breaks the rules: the raw socket)
        EpHdr h;
        std::vector<uint8_t> p;
        if (w.req.recv(h, p, now_ms() + 5000, err) != 1 || h.type != Req) return 15;
        if (mode == "close" || mode == "reset") {   // an orderly close (FIN) / a reset (RST) with a request out, no bye
            if (mode == "close") w.req.close();
            else w.req.reset();
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
    if (mode == "bulk-session") {   // before the bulk socket: one naming another session, then a second worker's hello
        Conn b1, b2;
        std::string e1, e2;
        Peer m2;
        const bool r1 = b1.dial(at, 5000, e1) && !bulk_worker(b1, me, m.hello.session ^ 7, e1) && has(e1, "unknown session");
        const bool r2 = b2.dial(at, 5000, e2) && !hello_worker(b2, me, table, m2, e2) && has(e2, "already connected");
        Conn b3;
        std::string e3;
        EpHello wrong = me;
        wrong.token[0] ^= 1;   // (no token: not told a worker is here)
        const bool r3 = b3.dial(at, 5000, e3) && !hello_worker(b3, wrong, table, m2, e3) && has(e3, "token") && !has(e3, "already");
        std::printf("  child bulk-session: %s / %s / %s\n", e1.c_str(), e2.c_str(), e3.c_str());
        if (!r1 || !r2 || !r3) return 20;
    }
    if (mode == "reconnect") return child_reconnect(at, me, table, w, m);
    std::thread bulk_th;
    std::atomic<int> gate_bad{0};   // gate: a fill loaded before the request that frees its slots was read
    std::string bulk_end;           // why the bulk thread ended
    if (mode == "fill" || mode == "bulk-session" || mode == "fill-drop" || mode == "bulk-silent" || mode == "gate" || mode == "gate-stuck" ||
        mode == "fill-stall") {   // the bulk socket on a thread of its own
        if (mode == "bulk-silent") {   // a connection that says nothing, before the worker's own bulk socket
            if (!quiet.dial(at, 5000, err)) return 23;
            sleep_ms(100);
        }
        if (!w.bulk.dial(at, 5000, err) || !bulk_worker(w.bulk, me, m.hello.session, err)) return 21;
        if (w.bulk.spin_ms != 0.0) return 22;   // (keep-awake is the request socket's)
        if (mode == "fill-drop") {   // the bulk socket closed under the main's first fill
            EpHdr h;
            std::vector<uint8_t> p;
            (void) w.bulk.recv(h, p, now_ms() + 5000, err);
            w.bulk.close();
        } else if (mode != "fill-stall") {   // (fill-stall: the socket is held and never read)
            bulk_th = std::thread([&] {
                std::string e;
                const WorkerLink::Loader load = [&](int, int e_in, int, uint32_t epoch, const uint8_t* blob, size_t n) -> uint16_t {
                    if (e_in == 99) return 1;   // (this one does not load)
                    if (mode == "gate" && epoch == 2 && w.epoch_used.load() < 1) {   // (the old behaviour: loaded at once)
                        gate_bad = 1;
                        return 3;
                    }
                    if (n == kBig) {            // (an expert's blob, loaded slowly: its batch stays out a while)
                        sleep_ms(200);
                        return blob[0] == 0xcd && blob[n - 1] == 0xcd ? 0 : 2;
                    }
                    return n == 0 || (n == 4096 && blob[0] == 0xab) ? 0 : 2;
                };
                while (w.serve_fill(now_ms() + 10000, load, e) == 1) {
                }
                bulk_end = e;
            });
        }
    }
    if (mode == "gate") sleep_ms(400);         // (the request thread starts late: the fill for epoch 2 is in before it reads Req 1)
    if (mode == "gate-stuck") sleep_ms(2500);  // ... or far too late
    // serve: y = 2x; "slow-first": the first request's reply 150 ms late; "slow-all": every reply 60 ms late;
    // "late": the first request's 1.1 s late; "silent": none
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
        if ((mode == "slow-first" && served == 0) || mode == "slow-all") sleep_ms(mode == "slow-all" ? 60 : 150);
        if (mode == "late" && served == 0) sleep_ms(1100);
        const std::vector<uint16_t> y = doubled(f.x, f.T);
        if (!w.send_resp(f.h.seq, (uint16_t) ((1u << f.T) - 1), 0, y.data(), 7, err)) break;
        ++served;
    }
    if (bulk_th.joinable()) bulk_th.join();   // (it ends when the main closes the bulk socket)
    if (mode == "gate") return gate_bad ? 31 : 0;
    if (mode == "gate-stuck") {   // the bulk thread said why it gave up
        std::printf("  child gate-stuck: %s\n", bulk_end.c_str());
        return has(bulk_end, "request tagged 1") ? 0 : 32;
    }
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

void pin_to(int cpu);

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
    check(has(why, "token") && !leaks(why, m.token), "pure: the worker refuses the main's token (never printing it)");
    check(f2bf(1.0f) == 0x3f80 && f2bf(1.00390625f) == 0x3f80 && f2bf(1.01171875f) == 0x3f82 && bf2f(0x4000) == 2.0f,
          "pure: bf16 rounds to nearest even");
    EpHello none = m;
    std::memset(none.token, 0, sizeof none.token);
    check(no_token_warning(m).empty() && has(no_token_warning(none), "WARNING") && has(no_token_warning(none), "STRATA_STAGE_TOKEN"),
          "pure: no token: the hello's WARNING (none with one)");
    std::string h, pt;
    const bool v6 = split_host_port("[::1]:7870", h, pt) && h == "::1" && pt == "7870";
    const bool v4 = split_host_port("10.10.10.2:7870", h, pt) && h == "10.10.10.2" && pt == "7870";
    const bool bare = split_host_port("7870", h, pt) && h.empty() && pt == "7870";
    check(v6 && v4 && bare && !split_host_port("::1:7870", h, pt) && !split_host_port("[::1]7870", h, pt),
          "pure: [V6]:PORT, HOST:PORT and PORT parse; a bare IPv6 address does not");
    // the worker's batches: one epoch, i of n in order, acked only when every fill of it loaded
    FillBatch b;
    const bool two = b.take(1, 0, 2, 0, 3).empty() && !b.done(true) && b.take(1, 1, 2, 0, 4).empty() && b.done(true);
    check(two, "pure: a batch of two: the epoch acked at the second fill, not the first");
    FillBatch c;
    const bool order = c.take(1, 0, 3, 0, 3).empty() && !c.done(true) && has(c.take(1, 2, 3, 0, 5), "order") &&
                       has(c.take(2, 0, 1, 0, 3), "only 1") && has(c.take(1, 1, 2, 0, 4), "order");
    check(order, "pure: a fill out of its batch's order, or for an epoch not the next, refused");
    FillBatch d;
    const bool bad = d.take(1, 0, 2, 0, 3).empty() && !d.done(false) && d.take(1, 1, 2, 0, 4).empty() && !d.done(true) &&
                     d.take(1, 0, 1, 0, 3).empty() && d.done(true);
    check(bad, "pure: a batch with a fill not loaded is not acked; its retry (from 0) is");
    FillBatch e;
    const bool dup = e.take(1, 0, 3, 0, 3).empty() && !e.done(true) && e.take(1, 1, 3, 0, 4).empty() && !e.done(true) &&
                     has(e.take(1, 2, 3, 0, 3), "second fill for layer 3") && e.take(1, 0, 3, 0, 3).empty();
    check(dup, "pure: a second fill for one layer in a batch refused (one spare slot a layer); a new batch may use it again");
}

void case_refusals() {
    struct C {
        const char* mode;
        const char* why;
    };
    const C cs[] = {{"bad-pack", "pack"},       {"bad-token", "token"}, {"bad-proto", "protocol"},
                    {"bad-geo", "n_ff"},        {"bad-slice", "stores blocks"}, {"bad-type", "type"},
                    {"raw-short", "short"},     {"raw-proto3", "protocol 3"},
                    {"raw-proto1", "protocol 1"}, {"big-proto3", "protocol 3"}};
    for (const C& c : cs) {
        Pair p;
        const bool opened = p.open(c.mode);
        const int rc = p.end();
        check(opened && !p.hello_ok && has(p.err, c.why) && rc == 0,
              std::string("hello refused: ") + c.mode + " (both sides say why)");
    }
    {   // a header that claims more than a hello: refused at once, its payload never waited for
        Pair p;
        const double t0 = now_ms();
        const bool opened = p.open("big-hello");
        const double ms = now_ms() - t0;
        p.end();
        check(opened && !p.hello_ok && has(p.err, "over") && ms < 2000, "hello refused: a hello frame of 1 MB, at once (" + p.err + ")");
    }
    {   // the worker refuses the main (a raw main: its HelloOk names another pack) and says why on the wire
        Listener ls;
        Child ch;
        Conn c;
        std::string err;
        EpHdr h;
        std::vector<uint8_t> pl;
        const bool up = ls.open("127.0.0.1:0", err) && ch.start("refused-here", ls.port()) && ls.accept(c, 10000, err) &&
                        c.recv(h, pl, now_ms() + 5000, err) == 1 && h.type == Hello;
        EpHello bad = hello_of(0);
        bad.pack_hash ^= 1;
        bad.session = 5;
        const std::vector<EpSlice> t = table_of(0, kBlocks);
        EpHdr ok;
        ok.type = HelloOk;
        const bool sent = up && c.send(ok, &bad, sizeof bad, err, t.data(), t.size() * sizeof(EpSlice));
        const bool told = sent && c.recv(h, pl, now_ms() + 5000, err) == 1 && h.type == Error &&
                          has(std::string(pl.begin(), pl.end()), "pack");
        const int rc = ch.wait(10000);
        check(told && rc == 0, "hello refused by the worker: the main reads the reason (\"" + std::string(pl.begin(), pl.end()) + "\")");
    }
    {   // no token on either side: the link comes up, each side printing the WARNING once
        Pair p;
        std::memset(p.me.token, 0, sizeof p.me.token);
        const int before = token_warnings();
        p.open("no-token");
        check(p.hello_ok && token_warnings() == before + 1, "no token: the hello goes on, with a WARNING here");
        p.m.bye();
        check(p.end() == 0, "... and one on the worker");
    }
}

std::atomic<int> g_woke{0};

// (Listener::after_wait: both threads have seen the connection before either takes it; on Linux a poll that wakes late
// finds the queue empty and sleeps on, so without this the race is rare)
void both_woke() {
    g_woke.fetch_add(1);
    const double until = now_ms() + 2000.0;
    while (g_woke.load() < 2 && now_ms() < until) sleep_ms(1);
}

void case_sessions_and_listeners() {
    {   // the same hello reused: each hello_main names a new session
        Pair p;
        p.open("good");
        const uint64_t s1 = p.me.session;
        Child c2;
        MainLink m2;
        Peer w2;
        std::string e2;
        const bool ok2 = c2.start("good", p.ls.port()) && p.ls.accept(m2.req, 10000, e2) &&
                         hello_main(m2.req, p.me, table_of(0, kBlocks), k_of(6), w2, e2);
        check(p.hello_ok && ok2 && s1 != 0 && p.me.session != s1, "each hello_main names a new session (the same hello reused)");
        m2.start(p.me);
        m2.bye();
        p.m.bye();
        const int rc2 = c2.wait(10000);
        check(p.end() == 0 && rc2 == 0, "... both workers end at their bye");
    }
    {   // one worker per listener: a second one is told so
        Pair p;
        p.open("good");
        Child c2;
        const bool started = c2.start("extra", p.ls.port());
        const bool refused = started && MainLink::refuse_extra(p.ls, p.me, 10000);
        const int rc2 = c2.wait(10000);
        check(refused && rc2 == 0 && p.m.up(), "one worker per listener: a second worker hears \"a worker is already connected\"");
        p.m.bye();
        p.end();
    }
    {   // a second worker without the token is not told a worker is here
        Pair p;
        p.open("good");
        Child c2;
        const bool started = c2.start("extra-token", p.ls.port());
        const bool refused = started && MainLink::refuse_extra(p.ls, p.me, 10000);
        const int rc2 = c2.wait(10000);
        check(refused && rc2 == 0 && p.m.up(), "a second worker without the token hears \"wrong token\", not \"a worker is already connected\"");
        p.m.bye();
        p.end();
    }
    {   // two threads wake on one connection: the loser waits again to its own deadline; it is not stuck in accept
        Listener ls;
        std::string e;
        const bool up = ls.open("127.0.0.1:0", e);
        g_woke = 0;
        ls.after_wait = &both_woke;
        struct R {
            bool ok = false;
            double ms = 0.0;
            std::string err;
            std::atomic<bool> done{false};
        } r[2];
        Conn got[2];
        auto run = [&](int i) {
            const double t0 = now_ms();
            r[i].ok = ls.accept(got[i], 1500, r[i].err);
            r[i].ms = now_ms() - t0;
            r[i].done = true;
        };
        std::thread a(run, 0), b(run, 1);
        sleep_ms(300);   // (both are waiting)
        Conn client, release;
        const bool dialed = up && client.dial("127.0.0.1:" + std::to_string(ls.port()), 5000, e);
        const bool both = wait_until([&] { return r[0].done.load() && r[1].done.load(); }, 4000);
        if (!both) (void) release.dial("127.0.0.1:" + std::to_string(ls.port()), 5000, e);   // (frees a thread stuck in a blocking accept)
        a.join();
        b.join();
        const int loser = r[0].ok ? 1 : 0;
        check(dialed && both && r[0].ok != r[1].ok && !r[loser].ok && has(r[loser].err, "no connection") && r[loser].ms >= 1200.0,
              "two threads in accept, one connection: the loser waits out its own deadline (" + std::to_string((int) r[loser].ms) + " ms, " +
                  r[loser].err + ")");
    }
    {   // a port another listener holds
        Listener a, b;
        std::string ea, eb;
        const bool first = a.open("127.0.0.1:0", ea);
        const bool second = first && b.open("127.0.0.1:" + std::to_string(a.port()), eb);
        check(first && !second, "a second listener on a port in use: refused (" + eb + ")");
    }
    {   // IPv6: a bracketed bind, a dial, the peer's name
        Listener ls;
        std::string e;
        if (!ls.open("[::1]:0", e)) {
            std::printf("  IPv6 on ::1: skipped here (%s)\n", e.c_str());
        } else {
            Conn c, s;
            const bool ok = c.dial("[::1]:" + std::to_string(ls.port()), 5000, e) && ls.accept(s, 5000, e);
            check(ok && s.peer() == "::1", "IPv6: [::1]:PORT listened on, dialed, the peer named (" + s.peer() + ")");
        }
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
    for (const char* mode : {"close", "reset"}) {
        Pair p;
        p.open(mode);
        std::string err;
        Reply r;
        ReqData a(1, 1);
        p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        const double t0 = now_ms();
        const int got = p.m.wait_resp(1, t0 + 3000, r, err);
        const bool cls = std::string(mode) == "close" ? has(err, "closed") : !has(err, "closed") && has(err, "recv");
        check(got == -1 && cls && !has(err, "mid-frame") && !p.m.up() && now_ms() - t0 < 2000,
              std::string("the worker ") + (mode[0] == 'r' ? "resets" : "closes") + " its socket: the link is down at once, not at the deadline (" + err + ")");
        p.end();
    }
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
    {   // every reply 60 ms late, each deadline 10 ms: the late replies free the slots, the drops count up to three
        Pair p;
        p.open("slow-all");
        std::string err;
        Reply r;
        ReqData a(1, 1);
        int s[3] = {0, 0, 0}, w[3] = {0, 0, 0};
        bool up = true;
        for (int i = 0; i < 3; ++i) {
            s[i] = p.m.send_req((uint32_t) i + 1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
            w[i] = p.m.wait_resp((uint32_t) i + 1, now_ms() + 10, r, err);
            if (i < 2) up = up && p.m.up();
            if (i < 2) sleep_ms(100);
        }
        check(s[0] == 1 && s[1] == 1 && s[2] == 1 && w[0] == 0 && w[1] == 0 && up && p.m.st.late_dropped >= 1 && w[2] == -1 &&
                  has(err, "3 requests in a row") && !p.m.up() && p.m.st.downs == 1,
              "three requests in a row dropped at their deadline: the link is down (" + err + ")");
        p.end();
    }
    {   // a worker that never answers: two drops, then no room - not a miss, the link stays up; then the oldest request
        // out passes a second: the link is down
        Pair p;
        p.open("silent");
        std::string err;
        Reply r;
        ReqData a(1, 1);
        const int s1 = p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        const int w1 = p.m.wait_resp(1, now_ms() + 10, r, err);
        const int s2 = p.m.send_req(2, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        const int w2 = p.m.wait_resp(2, now_ms() + 10, r, err);
        const int s3 = p.m.send_req(3, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        check(s1 == 1 && w1 == 0 && s2 == 1 && w2 == 0 && s3 == 0 && has(err, "two") && p.m.up() && p.m.st.downs == 0,
              "two drops, then no room for a third: not a miss, the link stays up");
        sleep_ms(1100);
        const int s4 = p.m.send_req(4, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        check(s4 == 0 && has(err, "1000 ms") && !p.m.up() && p.m.st.downs == 1,
              "the oldest request out over a second old: the link is down (" + err + ")");
        const int s5 = p.m.send_req(5, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
        check(s5 == 0 && has(err, "down"), "... and every request after it computed here");
        p.end();
    }
}

// the 1 s rule judges a request dropped at its deadline, not one still awaited whose reply is already in the socket
void case_late_reply() {
    Pair p;
    p.open("late");
    std::string err;
    Reply r;
    ReqData a(1, 1), b(1, 2);
    const int s1 = p.m.send_req(1, 0, 1, 0, 0, a.picks.data(), a.x.data(), err);
    sleep_ms(1300);   // (the reply came at 1.1 s and sits unread: the request is over a second old)
    const int s2 = p.m.send_req(2, 0, 1, 0, 0, b.picks.data(), b.x.data(), err);
    const int w1 = p.m.wait_resp(1, now_ms() + 2000, r, err);
    const bool d1 = w1 == 1 && a.is_doubled(r, 1);
    const int w2 = p.m.wait_resp(2, now_ms() + 2000, r, err);
    check(s1 == 1 && s2 == 1 && d1 && w2 == 1 && b.is_doubled(r, 1) && p.m.up() && p.m.st.downs == 0,
          "a reply that came and is unread while its request is over a second old: not stale (" + err + ")");
    p.m.bye();
    p.end();
    {   // the same for a request dropped at its deadline: the reply arrives after the 1 s mark and sits unread; the next send takes it first
        Pair q;
        q.open("late");
        std::string e;
        Reply r2;
        ReqData c(1, 1), d(1, 2);
        const double t0 = now_ms();
        const int s3 = q.m.send_req(1, 0, 1, 0, 0, c.picks.data(), c.x.data(), e);
        const int w3 = s3 == 1 ? q.m.wait_resp(1, now_ms() + 50.0, r2, e) : 9;   // (dropped at its deadline)
        sleep_ms(t0 + 1300.0 - now_ms());   // (the reply came at 1.1 s: past the 1 s since the drop, unread)
        const int s4 = q.m.send_req(2, 0, 1, 0, 0, d.picks.data(), d.x.data(), e);
        const int w4 = s4 == 1 ? q.m.wait_resp(2, now_ms() + 2000, r2, e) : 9;
        check(s3 == 1 && w3 == 0 && s4 == 1 && w4 == 1 && d.is_doubled(r2, 1) && q.m.up() && q.m.st.downs == 0,
              "a dropped request's reply that came after the 1 s mark and is unread: taken first, the link stays up (" + e + ")");
        q.m.bye();
        q.end();
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

// the read buffer: frames read one after another, no read ending on a frame's end, keep it small (its consumed bytes
// are dropped past a megabyte)
void case_read_buffer() {
    Listener ls;
    Conn a, b;
    std::string err;
    const bool up = ls.open("127.0.0.1:0", err) && a.dial("127.0.0.1:" + std::to_string(ls.port()), 5000, err) &&
                    ls.accept(b, 5000, err);
    constexpr int kFrames = 800;
    constexpr size_t kPay = 10289;   // (an odd frame size: the reads' ends do not fall on the frames')
    std::vector<uint8_t> all;
    for (int i = 0; i <= kFrames; ++i) {
        EpHdr h;
        h.type = Ping;
        h.seq = (uint32_t) i;
        h.bytes = (uint32_t) kPay;
        const size_t at = all.size();
        all.resize(at + sizeof h + (i < kFrames ? kPay : kPay / 2));   // (the last one half a frame)
        std::memcpy(all.data() + at, &h, sizeof h);
    }
    std::thread tx([&] {
        std::string e;
        (void) a.send_bytes(all.data(), all.size(), e);
    });
    size_t peak = 0;
    int got = 0;
    for (; up && got < kFrames; ++got) {
        EpHdr h;
        std::vector<uint8_t> p;
        if (b.recv(h, p, now_ms() + 5000, err) != 1 || h.seq != (uint32_t) got) break;
        peak = std::max(peak, b.held());
    }
    tx.join();
    check(up && got == kFrames && peak < ((size_t) 3 << 19),
          "the read buffer: " + std::to_string(got) + " frames read, at most " + std::to_string(peak) + " bytes held (under 1.5 MB)");
}

// the kernel's rx stamps (the probe's arrival times): each reply's is set, later than the last one's, and between its
// request's send and the moment wait_resp returned it; a read without one leaves none (Linux; elsewhere skipped with a
// note)
void case_rx_stamps() {
    Pair p;
    p.open("good");
    std::string err;
    if (!p.m.req.rx_stamps(err)) {
        std::printf("  the kernel's rx stamps: skipped here (%s)\n", err.c_str());
        p.m.bye();
        p.end();
        return;
    }
    Reply r;
    ReqData a(1, 1);
    // (the kernel turns stamping on a little after the first socket asks - a deferred static key - so frames just
    // after rx_stamps can come unstamped: wait for the first stamp, then check)
    uint32_t s = 0;
    int warm = 0;
    bool served = true;
    for (; warm < 200; ++warm) {
        ++s;
        served = p.m.send_req(s, 0, 1, 0, 0, a.picks.data(), a.x.data(), err) == 1 && p.m.wait_resp(s, now_ms() + 2000, r, err) == 1;
        if (!served || p.m.req.last_rx_ns() != 0) break;
        sleep_ms(1);
    }
    std::printf("  the kernel's rx stamps: the first after %d unstamped replies\n", warm);
    bool set = served, rising = true, within = true;
    int64_t last = p.m.req.last_rx_ns();
    std::string why;
    const uint32_t first = s + 1;
    for (s = first; s < first + 5; ++s) {
        const int64_t w0 = wall_ns();
        const bool ok = p.m.send_req(s, 0, 1, 0, 0, a.picks.data(), a.x.data(), err) == 1 &&
                        p.m.wait_resp(s, now_ms() + 2000, r, err) == 1;
        const int64_t w1 = wall_ns(), rx = p.m.req.last_rx_ns();
        set = set && ok && rx > 0;
        rising = rising && rx > last;
        within = within && rx >= w0 && rx <= w1;
        if (why.empty() && !(ok && rx > 0 && rx > last && rx >= w0 && rx <= w1))
            why = " (reply " + std::to_string(s) + ": stamp - send " + std::to_string(rx - w0) + " ns, read - stamp " +
                  std::to_string(w1 - rx) + " ns, after the last by " + std::to_string(rx - last) + " ns)";
        last = rx;
    }
    check(set && rising && within, "the kernel's rx stamps: each reply's set, rising, between its send and its read" + why);
    p.m.req.drop_stamps = 1000;   // (the reads from here on act as if the kernel gave no stamp)
    const bool ok = p.m.send_req(s, 0, 1, 0, 0, a.picks.data(), a.x.data(), err) == 1 && p.m.wait_resp(s, now_ms() + 2000, r, err) == 1;
    check(ok && last > 0 && p.m.req.last_rx_ns() == 0, "the kernel's rx stamps: a reply read without one after a stamped one has none");
    p.m.bye();
    p.end();
}

void case_fill() {
    Pair p;
    p.open("fill");
    std::string err;
    std::mutex mu;
    std::vector<std::pair<int, uint16_t>> oks;   // (e_in, status) as the FillOks come
    uint32_t ready_at_first = 99;
    p.m.on_fill_ok = [&](int, int e_in, uint32_t, uint16_t status) {
        std::lock_guard<std::mutex> g(mu);
        if (oks.empty()) ready_at_first = p.m.epoch_ready();
        oks.push_back({e_in, status});
    };
    auto n_oks = [&] {
        std::lock_guard<std::mutex> g(mu);
        return oks.size();
    };
    const bool bulk = p.m.accept_bulk(p.ls, 5000, err);
    check(bulk && p.m.bulk.spin_ms == 0.0, "the bulk socket of this session accepted, the fill thread up; no spin on it");
    std::vector<uint8_t> blob(4096, 0xab);
    const bool f1 = p.m.queue_fill(3, 17, 5, 1, 0, 2, blob.data(), blob.size(), err) && p.m.queue_fill(4, 18, -1, 1, 1, 2, nullptr, 0, err);
    const bool ok1 = wait_until([&] { return p.m.fills_pending() == 0; }, 3000);
    check(f1 && ok1 && n_oks() == 2 && ready_at_first == 0 && p.m.epoch_ready() == 1,
          "a batch of two fills for epoch 1: ready after both FillOks, not after the first");
    check(!p.m.queue_fill(3, 20, 6, 2, 0, 1, nullptr, 0, err) && has(err, "before a request tagged 1"),
          "a fill for epoch 2 before a request tagged 1 was sent: refused here");
    check(!p.m.queue_fill(3, 20, 6, 2, 0, 1, nullptr, 8, err) && has(err, "null"), "a fill with a null blob and bytes: refused here");
    check(!p.m.queue_fill(3, 20, 6, 2, 0, 1, blob.data(), kBulkFrame, err) && has(err, "bulk frame"),
          "a fill with a blob over a bulk frame: refused here");
    Reply r;
    ReqData x(1, 1);
    check(p.m.send_req(1, 0, 1, 1, 0, x.picks.data(), x.x.data(), err) == 1 && p.m.wait_resp(1, now_ms() + 2000, r, err) == 1 &&
              x.is_doubled(r, 1),
          "a request tagged with epoch 1 served");
    check(p.m.send_req(2, 0, 1, 0, 0, x.picks.data(), x.x.data(), err) == 0 && has(err, "before the last"),
          "a request tagged with an epoch before the last one sent: not sent");
    check(!p.m.queue_fill(3, 20, 6, 3, 0, 1, nullptr, 0, err) && has(err, "epoch 3"), "a fill that skips an epoch: refused here");
    check(!p.m.queue_fill(40, 20, 6, 2, 0, 1, nullptr, 0, err) && has(err, "layer 40"), "a fill for layer 40 of 40: refused here");
    check(!p.m.queue_fill(5, 20, 6, 2, 1, 2, nullptr, 0, err) && has(err, "order"), "a batch's fill 1 before its fill 0: refused here");
    // epoch 2: an expert's blob (over the request socket's 4 MB frames), which the worker loads slowly
    std::vector<uint8_t> big(kBig, 0xcd);
    const bool b0 = p.m.queue_fill(5, 21, 7, 2, 0, 2, big.data(), big.size(), err);
    const bool early = !p.m.queue_fill(5, 22, 8, 2, 0, 1, nullptr, 0, err) && has(err, "still to come");
    check(!p.m.queue_fill(5, 24, -1, 2, 1, 2, nullptr, 0, err) && has(err, "second fill for layer 5"),
          "a second fill for a layer in one batch: refused here (one spare slot a layer)");
    const bool b1 = p.m.queue_fill(6, 22, -1, 2, 1, 2, nullptr, 0, err);
    const bool out = !p.m.queue_fill(5, 22, 8, 2, 0, 1, nullptr, 0, err) && has(err, "fills are out");
    const bool ok2 = wait_until([&] { return p.m.fills_pending() == 0; }, 5000);
    bool loaded = false;
    {
        std::lock_guard<std::mutex> g(mu);
        loaded = oks.size() == 4 && oks[2].first == 21 && oks[2].second == 0 && oks[3].second == 0;
    }
    check(b0 && b1 && ok2 && loaded && p.m.epoch_ready() == 2 && p.m.up(),
          "a fill of " + std::to_string(kBig) + " bytes (an expert) loaded on the worker: epoch 2 ready");
    check(early && out, "a new batch while one's fills are still to come, or out: refused here");
    check(p.m.send_req(3, 0, 1, 2, 0, x.picks.data(), x.x.data(), err) == 1 && p.m.wait_resp(3, now_ms() + 2000, r, err) == 1,
          "a request tagged with epoch 2 served");
    // epoch 3: fill 0 loads, fill 1 does not
    const bool f3 = p.m.queue_fill(7, 23, -1, 3, 0, 2, nullptr, 0, err) && p.m.queue_fill(8, 99, 1, 3, 1, 2, nullptr, 0, err);
    const bool ok3 = wait_until([&] { return p.m.fills_pending() == 0; }, 3000);
    check(f3 && ok3 && p.m.epoch_ready() == 2 && p.m.st.fill_fails == 1, "a batch with a fill the worker could not load: epoch 3 not ready");
    check(p.m.send_req(4, 0, 1, 3, 0, x.picks.data(), x.x.data(), err) == 0 && has(err, "epoch"),
          "a request tagged with epoch 3: not sent");
    // the same request sent past MainLink's check: the worker refuses it (it acks an epoch per batch too)
    EpHdr h;
    h.type = Req;
    h.T = 1;
    h.seq = 100;
    h.epoch = 3;
    std::string e2;
    (void) p.m.req.send(h, x.picks.data(), x.picks.size() * sizeof(EpPick), e2, x.x.data(), x.x.size() * 2);
    EpHdr g;
    std::vector<uint8_t> pl;
    const int got = p.m.req.recv(g, pl, now_ms() + 2000, e2);
    check(got == 1 && g.type == Error && has(std::string(pl.begin(), pl.end()), "epoch 3"),
          "a request with an epoch whose batch did not load whole: refused by the worker");
    p.m.bye();
    check(p.end() == 0, "the worker's bulk thread ends at the bye");
}

// a Fill on a raw bulk socket (a main that does not go through MainLink's own rules), and the worker's FillOk: 0 loaded;
// -1 none by the deadline or the link failed (what says which); -2 an Error (what is its text); -3 another frame
bool raw_fill(Conn& b, int layer, uint32_t epoch, int e_in, int i, int n, std::string& err) {
    EpHdr h;
    h.type = Fill;
    h.layer = (uint8_t) layer;
    h.epoch = epoch;
    h.aux = (uint32_t) e_in;
    EpFill f;
    f.e_in = e_in;
    f.i = (uint16_t) i;
    f.n = (uint16_t) n;
    return b.send(h, &f, sizeof f, err);
}
int raw_fill_ok(Conn& b, double ms, std::string& what) {
    EpHdr h;
    std::vector<uint8_t> pl;
    std::string err;
    const int got = b.recv(h, pl, now_ms() + ms, err);
    if (got != 1) {
        what = got == 0 ? "none by the deadline" : err;
        return -1;
    }
    if (h.type == Error) {
        what = std::string(pl.begin(), pl.end());
        return -2;
    }
    if (h.type != FillOk || pl.size() != 2) {
        what = "a frame of type " + std::to_string(h.type);
        return -3;
    }
    uint16_t st = 0;
    std::memcpy(&st, pl.data(), 2);
    return st;
}

// the worker's own rule for a batch, on the wire: a second fill for one layer
void case_fill_layers() {
    Pair p;
    p.open("fill");
    Conn b;
    std::string err, what;
    const bool up = p.hello_ok && p.ls.accept(b, 5000, err) && bulk_main(b, p.me, err);
    const bool a = up && raw_fill(b, 3, 1, 17, 0, 2, err) && raw_fill_ok(b, 3000, what) == 0;
    const bool second = a && raw_fill(b, 3, 1, 18, 1, 2, err);
    const int st = second ? raw_fill_ok(b, 3000, what) : -9;
    check(a && st == -2 && has(what, "second fill for layer 3"), "the worker refuses a second fill for one layer in a batch (" + what + ")");
    p.m.bye();
    check(p.end() == 0, "... and ends at the bye");
}

// section 3.7 on the worker: a fill for epoch n + 1 waits until a request tagged n was read (the loader may overwrite the
// slots of epoch n - 1 only after one was), and gives up after a second with an Error that takes the link down
void case_fill_gate() {
    {   // the request thread reads Req 1 about 400 ms after the fill for epoch 2 is in: the fill is loaded after it
        Pair p;
        p.open("gate");
        std::string err;
        Reply r;
        ReqData x(1, 1);
        const bool bulk = p.m.accept_bulk(p.ls, 5000, err);
        const bool f1 = bulk && p.m.queue_fill(3, 17, -1, 1, 0, 1, nullptr, 0, err) && wait_until([&] { return p.m.epoch_ready() == 1; }, 3000);
        const bool s1 = f1 && p.m.send_req(1, 0, 1, 1, 0, x.picks.data(), x.x.data(), err) == 1;
        const double t0 = now_ms();
        const bool f2 = s1 && p.m.queue_fill(4, 18, -1, 2, 0, 1, nullptr, 0, err);
        const bool ready2 = f2 && wait_until([&] { return p.m.epoch_ready() == 2; }, 5000);
        const double ms = now_ms() - t0;
        const bool resp = s1 && p.m.wait_resp(1, now_ms() + 3000, r, err) == 1 && x.is_doubled(r, 1);
        p.m.bye();
        const int rc = p.end();
        check(ready2 && ms >= 150.0 && resp && rc == 0 && p.m.up() == false,
              "a fill for epoch 2 is loaded only after the request tagged 1 was read (" + std::to_string((int) ms) + " ms, child " +
                  std::to_string(rc) + ")");
    }
    {   // the request thread reads nothing for 2.5 s: the fill gives up after a second, an Error, the link down
        Pair p;
        p.open("gate-stuck");
        std::string err;
        ReqData x(1, 1);
        const bool bulk = p.m.accept_bulk(p.ls, 5000, err);
        const bool f1 = bulk && p.m.queue_fill(3, 17, -1, 1, 0, 1, nullptr, 0, err) && wait_until([&] { return p.m.epoch_ready() == 1; }, 3000);
        const bool s1 = f1 && p.m.send_req(1, 0, 1, 1, 0, x.picks.data(), x.x.data(), err) == 1;
        const double t0 = now_ms();
        const bool f2 = s1 && p.m.queue_fill(4, 18, -1, 2, 0, 1, nullptr, 0, err);
        const bool down = f2 && wait_until([&] { return !p.m.up(); }, 4000);
        const double ms = now_ms() - t0;
        const int s2 = p.m.send_req(2, 0, 1, 1, 0, x.picks.data(), x.x.data(), err);
        check(down && ms >= 800.0 && ms < 3000.0 && s2 == 0 && has(err, "request tagged 1") && p.m.epoch_ready() == 1,
              "a fill whose request never comes is refused after a second, the link down (" + std::to_string((int) ms) + " ms, " + err + ")");
        p.m.bye();
        check(p.end() == 0, "... the worker's bulk thread said why and ended");
    }
}

// ending a link does not wait for a send to a peer that is not reading (a full buffer was waited on for 30 s)
void case_fill_stop() {
    {   // 40 MB cannot sit in the buffers
        Pair p;
        p.open("fill-stall");
        std::string err;
        std::vector<uint8_t> huge((size_t) 40 << 20, 0x5a);
        const bool bulk = p.m.accept_bulk(p.ls, 5000, err);
        const bool q = bulk && p.m.queue_fill(3, 17, 5, 1, 0, 1, huge.data(), huge.size(), err);
        sleep_ms(300);   // (the fill thread is stuck in the send)
        const double t0 = now_ms();
        p.m.bye();
        const double ms = now_ms() - t0;
        check(q && ms < 2000.0, "bye() with a fill stuck in a send to a peer that does not read: " + std::to_string((int) ms) + " ms");
        check(p.end() == 0, "... the worker ends at the bye");
    }
    {   // a paced socket sleeps between its 64 KB pieces (20 KB/s: one piece takes 3 s)
        Pair p;
        p.open("fill");
        std::string err;
        std::vector<uint8_t> big(kBig, 0xcd);
        const bool bulk = p.m.accept_bulk(p.ls, 5000, err);
        p.m.bulk.pace_bps = 20000.0;   // (before the first fill: the fill thread has not touched the socket yet)
        const bool q = bulk && p.m.queue_fill(3, 17, 5, 1, 0, 1, big.data(), big.size(), err);
        sleep_ms(300);
        const double t0 = now_ms();
        p.m.bye();
        const double ms = now_ms() - t0;
        check(q && ms < 1000.0, "bye() with a fill held back by the pacing: " + std::to_string((int) ms) + " ms");
        p.end();
    }
}

// decision 3 of the review: the request path takes the link down in the middle of a fill; start() ends the fill thread
// (the bulk socket is its own to close) and a new accept_bulk works
void case_reconnect() {
    Pair p;
    p.open("reconnect");
    std::string err;
    std::vector<uint8_t> huge((size_t) 40 << 20, 0x5a);
    const bool bulk = p.m.accept_bulk(p.ls, 5000, err);
    const bool q = bulk && p.m.queue_fill(3, 17, 5, 1, 0, 1, huge.data(), huge.size(), err);
    sleep_ms(200);   // (the fill thread is stuck in the send: the worker never reads its bulk socket)
    Reply r;
    ReqData x(1, 1);
    const bool sent = p.m.send_req(1, 0, 1, 0, 0, x.picks.data(), x.x.data(), err) == 1;
    const int w1 = sent ? p.m.wait_resp(1, now_ms() + 3000, r, err) : 9;
    const bool downed = w1 == -1 && has(err, "magic") && !p.m.up();
    const bool bulk_left = p.m.bulk.ok();   // (the bulk socket is the fill thread's: it closes it, in about 20 ms, not down_)
    const bool pending = p.m.fills_pending() == 1;   // (the abandoned fill gets no FillOk: pending until start())
    Peer w2;
    const bool again = downed && p.ls.accept(p.m.req, 10000, err) && hello_main(p.m.req, p.me, table_of(0, kBlocks), k_of(6), w2, err);
    const double t0 = now_ms();
    p.m.start(p.me);
    const double ms = now_ms() - t0;
    const bool clean = p.m.up() && p.m.fills_pending() == 0 && p.m.epoch_ready() == 0;
    const bool bulk2 = again && p.m.accept_bulk(p.ls, 5000, err);
    const bool f = bulk2 && p.m.queue_fill(3, 17, 5, 1, 0, 1, nullptr, 0, err) && wait_until([&] { return p.m.epoch_ready() == 1; }, 3000);
    const bool served = f && p.m.send_req(2, 0, 1, 1, 0, x.picks.data(), x.x.data(), err) == 1 &&
                        p.m.wait_resp(2, now_ms() + 3000, r, err) == 1 && x.is_doubled(r, 1);
    check(q && downed && bulk_left && pending && again && ms < 2000.0 && clean,
          "a request-path failure in the middle of a fill: pending until start(), which ends the fill thread (" +
              std::to_string((int) ms) + " ms, " + err + ")");
    check(bulk2 && f && served, "... then accept_bulk, a fill and a request on the new connection work");
    p.m.bye();
    check(p.end() == 0, "... and the worker ends at the bye");
}

void case_fill_drop() {
    Pair p;
    p.open("fill-drop");
    std::string err;
    const bool bulk = p.m.accept_bulk(p.ls, 5000, err);
    const bool q = bulk && p.m.queue_fill(3, 17, 5, 1, 0, 1, nullptr, 0, err);
    const bool down = wait_until([&] { return !p.m.up(); }, 3000);
    Reply r;
    ReqData x(1, 1);
    const int s = p.m.send_req(1, 0, 1, 0, 0, x.picks.data(), x.x.data(), err);
    check(q && down && s == 0 && has(err, "bulk socket") && p.m.st.downs == 1,
          "the bulk socket closed under a fill: the link is down, the request computed here (" + err + ")");
    check(p.end() == 0, "... and the worker sees its request socket closed");
    {   // the decode thread has not acted yet (up() is false): ping and accept_bulk look at the bulk socket too
        Pair pg2;
        pg2.open("fill-drop");
        const bool bulk2 = pg2.m.accept_bulk(pg2.ls, 5000, err);
        const bool q2 = bulk2 && pg2.m.queue_fill(3, 17, 5, 1, 0, 1, nullptr, 0, err);
        const bool down2 = wait_until([&] { return !pg2.m.up(); }, 3000);
        uint32_t held = 0;
        const int pg = pg2.m.ping(5, 64, now_ms() + 500, held, err);
        check(q2 && down2 && pg == -1 && has(err, "bulk socket"), "ping after the bulk socket failed: the link is down (" + err + ")");
        pg2.end();
        Pair pg3;
        pg3.open("fill-drop");
        const bool bulk3 = pg3.m.accept_bulk(pg3.ls, 5000, err);
        const bool f3 = bulk3 && pg3.m.queue_fill(3, 17, 5, 1, 0, 1, nullptr, 0, err);
        const bool d3 = wait_until([&] { return !pg3.m.up(); }, 3000);
        const bool again = pg3.m.accept_bulk(pg3.ls, 300, err);
        check(f3 && d3 && !again && has(err, "bulk socket:"), "accept_bulk after the bulk socket failed: the link is down (" + err + ")");
        pg3.end();
    }
}

void case_bulk_accept() {
    {   // the bulk sockets that are not this session's: refused (each told why), the wait goes on
        Pair p;
        p.open("bulk-session");
        std::string err;
        const bool ok = p.m.accept_bulk(p.ls, 5000, err);
        const bool fill = ok && p.m.queue_fill(3, 17, 5, 1, 0, 1, nullptr, 0, err) && wait_until([&] { return p.m.epoch_ready() == 1; }, 3000);
        check(ok && fill, "the bulk socket: another session's and a second worker's hello refused, then this session's taken (" + err + ")");
        p.m.bye();
        check(p.end() == 0, "... and the worker heard \"unknown session\" and \"a worker is already connected\"");
    }
    {   // a connection that says nothing is closed after kBulkHelloMs, the worker behind it taken
        Pair p;
        p.open("bulk-silent");
        std::string err;
        const double t0 = now_ms();
        const bool ok = p.m.accept_bulk(p.ls, 5000, err);
        const double ms = now_ms() - t0;
        check(ok && ms >= 150.0 && ms < 2500.0, "a silent connection is closed after 200 ms, the worker's bulk socket behind it taken (" +
                                                   std::to_string((int) ms) + " ms, " + err + ")");
        p.m.bye();
        p.end();
    }
    {   // the worker checks the main's bulk answer: another session, another token (a raw main)
        Pair p;
        p.open("bulk-check");
        bool told = p.hello_ok;
        std::string said;
        for (int i = 0; i < 2 && told; ++i) {
            Conn b;
            std::string err;
            EpHdr h;
            std::vector<uint8_t> pl;
            told = p.ls.accept(b, 5000, err) && b.recv(h, pl, now_ms() + 5000, err) == 1 && h.type == Hello;
            EpHello bm = p.me;
            bm.channel = 1;
            if (i == 0) bm.session ^= 2;
            else bm.token[0] ^= 1;
            EpHdr ok;
            ok.type = HelloOk;
            told = told && b.send(ok, &bm, sizeof bm, err) && b.recv(h, pl, now_ms() + 5000, err) == 1 && h.type == Error;
            said += std::string(pl.begin(), pl.end()) + (i == 0 ? " / " : "");
        }
        const int rc = p.end();
        check(told && rc == 0 && has(said, "session") && has(said, "token"),
              "the worker refuses a bulk answer naming another session or token, and says why (" + said + ")");
    }
}

// pin_to(-1): the thread's affinity before its first pin_to, not every CPU
void case_pin_restore() {
    bool ok = false, skipped = false;
    std::thread([&] {
#if defined(_WIN32)
        DWORD_PTR proc = 0, sys = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
        if ((proc & 7) != 7) {
            skipped = true;
            return;
        }
        const HANDLE t = GetCurrentThread();
        SetThreadAffinityMask(t, 3);
        pin_to(1);
        const DWORD_PTR pinned = SetThreadAffinityMask(t, 2);   // (it returns the mask it replaces: the same set again)
        pin_to(-1);
        const DWORD_PTR after = SetThreadAffinityMask(t, 3);
        ok = pinned == 2 && after == 3;
#else
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(0, &s);
        CPU_SET(1, &s);
        if (std::thread::hardware_concurrency() < 3 || sched_setaffinity(0, sizeof s, &s) != 0) {
            skipped = true;
            return;
        }
        pin_to(1);
        cpu_set_t g;
        sched_getaffinity(0, sizeof g, &g);
        const bool one = CPU_COUNT(&g) == 1 && CPU_ISSET(1, &g);
        pin_to(-1);
        sched_getaffinity(0, sizeof g, &g);
        ok = one && CPU_COUNT(&g) == 2 && CPU_ISSET(0, &g) && CPU_ISSET(1, &g);
#endif
    }).join();
    if (skipped) std::printf("  pin_to's restore: skipped here (fewer than 3 CPUs)\n");
    else check(ok, "pin_to(-1): the thread's mask before its first pin again ({0, 1}, not every CPU)");
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
void pin_to(int cpu) {   // the calling thread on CPU (-1: the mask it had before its first pin_to again)
#if defined(_WIN32)
    thread_local DWORD_PTR before = 0;
    if (cpu >= 0 && cpu < 64) {
        const DWORD_PTR was = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
        if (before == 0) before = was;
    } else if (cpu < 0 && before != 0) {
        SetThreadAffinityMask(GetCurrentThread(), before);
        before = 0;
    }
#else
    thread_local bool saved = false;
    thread_local cpu_set_t before;
    if (cpu >= 0) {
        if (!saved) saved = sched_getaffinity(0, sizeof before, &before) == 0;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof set, &set);
    } else if (saved) {
        sched_setaffinity(0, sizeof before, &before);
        saved = false;
    }
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
// threads can stretch.  held: the far end's Ping in hand -> its Pong sent.  timeouts: pings with no Pong in a second
// (in no column); unstamped: Pongs without both ends' stamps, with stamps on (in rtt, not in wire / here / there).
struct Arm {
    std::vector<double> rtt, held, wire, here, there;
    int timeouts = 0, unstamped = 0;
    bool stamped = false;   // (both ends' stamps on: the wire columns are due for every Pong)
};
void report(const char* name, Arm& a) {
    auto pct = [](std::vector<double> v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t) (q * (double) (v.size() - 1) + 0.5))];
    };
    auto mx = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    char unst[32] = "-";
    if (a.stamped) std::snprintf(unst, sizeof unst, "%d", a.unstamped);
    std::printf("  %-36s n %4zu  rtt p50 %.3f  p90 %.3f  p99 %.3f  max %.3f ms   far end held p50 %.1f us   timeouts %d  unstamped %s\n",
                name, a.rtt.size(), pct(a.rtt, 0.5), pct(a.rtt, 0.9), pct(a.rtt, 0.99), mx(a.rtt), pct(a.held, 0.5), a.timeouts, unst);
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
    for (const int bp : {0, busy_us}) {
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
                a.stamped = stamps && c.stamps_ok;
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
                    if (a.stamped && rx > 0 && there >= 0) {
                        a.wire.push_back((double) (rx - w0) / 1.0e6);
                        a.here.push_back((double) (w1 - rx) / 1.0e3);
                        a.there.push_back((double) there / 1.0e3);
                    } else if (a.stamped)
                        ++a.unstamped;
                }
                if (got == 0) ++a.timeouts;
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
    return 0;
}

// a case that runs this long is hung, or its waits ran to their deadlines (a peer's close not noticed): the run says which case
// and fails, instead of the whole run going on, or hanging (the cases take well under 20 s each)
constexpr double kCaseMs = 30000.0;
std::atomic<const char*> g_case{nullptr};
std::atomic<double> g_case_t0{0.0};

void run_case(const char* name, void (*fn)()) {
    g_case_t0 = now_ms();
    g_case = name;
    fn();
    g_case = nullptr;
}

void watchdog() {
    for (;;) {
        sleep_ms(200);
        const char* c = g_case.load();
        if (c != nullptr && now_ms() - g_case_t0.load() > kCaseMs) {
            std::printf("%s: still running after %d s (hung, or waiting out its deadlines) FAIL\nFAILED\n", c, (int) (kCaseMs / 1000.0));
            std::fflush(stdout);
            std::_Exit(1);
        }
    }
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
    std::thread(watchdog).detach();
    run_case("case_pure", case_pure);
    run_case("case_refusals", case_refusals);
    run_case("case_sessions_and_listeners", case_sessions_and_listeners);
    run_case("case_accept_and_rules", case_accept_and_rules);
    run_case("case_order", case_order);
    run_case("case_closed", case_closed);
    run_case("case_timeouts", case_timeouts);
    run_case("case_late_reply", case_late_reply);
    run_case("case_killed", case_killed);
    run_case("case_badmagic", case_badmagic);
    run_case("case_spin_wake", case_spin_wake);
    run_case("case_read_buffer", case_read_buffer);
    run_case("case_fill", case_fill);
    run_case("case_fill_layers", case_fill_layers);
    run_case("case_fill_gate", case_fill_gate);
    run_case("case_fill_stop", case_fill_stop);
    run_case("case_reconnect", case_reconnect);
    run_case("case_fill_drop", case_fill_drop);
    run_case("case_bulk_accept", case_bulk_accept);
    run_case("case_pin_restore", case_pin_restore);
    run_case("case_rx_stamps", case_rx_stamps);
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails != 0;
}
