// Strata. The remote experts (src/dsv41/ds4_rx.hpp) between two processes on 127.0.0.1: this program is the main
// process's end (rx::Main, the engine's own code), a child of it the expert server - the CPU one (rx::serve_cpu, what
// strata-dsv41 --expert-server-cpu runs: the same serve_loop / serve_next as the GPU server, the CPU lane's arithmetic).
// Random legal i-quant blobs; each window's picks split between the server and this end, as serve_lane splits them.
//   - the server's reply is the CPU lane's sum of its picks for x in bf16, rounded to bf16, bit for bit; the split sum
//     is the all-here reference within the server's roundings (bf16 x moves some int8 activations: a few % of its part);
//   - a reply later than its deadline: 0, the picks computed here (the reference exactly), the next request gets its
//     own reply and the late one is in take_late;
//   - a killed server: the wait ends at once with -1 (no hang), the picks computed here;
//   - --expert-server-once: the server exits 0 when the main leaves;
//   - keep-awake: a Wake (only after kWakeQuietMs without a send) keeps the server's receive spinning (its waits in
//     poll do not move), and without one they do; the server takes them;
//   - the link core: the server's serve loop on DS4_EP_CPU; os::pin_self(-1) restores the mask; os::cpu_siblings;
//     rx::off_link takes the core's CPUs out; rx::link_cpu's env; serve_lane's nap rule (rx::lane_naps);
//   - Linux: TCP_QUICKACK after each read.
// No GPU, no model.
//   ds4_rx_loop_test [PORT (27893; PORT + 1 too)]
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_os.hpp"
#include "../../src/dsv41/ds4_rx.hpp"

#include "ggml.h"   // ggml_type_size / ggml_blck_size: the blocks' sizes

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace gg = strata::ds4::gguf;
namespace os = strata::ds4::os;
namespace rx = strata::ds4::rx;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++fails;
}
static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v == nullptr ? "" : v);
#else
    if (v == nullptr) unsetenv(k);
    else setenv(k, v, 1);
#endif
}
static const char* kToken = "rx-loop-test-0123456789";
static const int E = 5120, FF = 2304;
static const float kLim = 10.0f;

// ---------------------------------------------------------------------------------------------------- a child process
static std::string self_path() {
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
    bool start(int port) {
        const std::string exe = self_path(), p = std::to_string(port);
#if defined(_WIN32)
        std::string cmd = "\"" + exe + "\" --child " + p;
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
            execl(exe.c_str(), exe.c_str(), "--child", p.c_str(), (char*) nullptr);
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

// ---------------------------------------------------------------------------------------------------- the experts
static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    const uint32_t m = u & 0x7fffffu;
    uint32_t v = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
// random legal blocks of `type`, n bytes of whole blocks (iq_expert_test's generator)
static void fill(std::mt19937_64& rng, int type, uint8_t* p, size_t n) {
    const size_t bb = ggml_type_size((ggml_type) type);
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) rng();
    for (size_t b = 0; b < n / bb; ++b) {
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* q = p + b * bb;
        if (type == 29) {   // block_iq1_m: its F16 scale is the top nibbles of its four u16 scales
            q[49] = (uint8_t) ((q[49] & 0x0f) | ((d & 0x000f) << 4));
            q[51] = (uint8_t) ((q[51] & 0x0f) | (d & 0x00f0));
            q[53] = (uint8_t) ((q[53] & 0x0f) | ((d & 0x0f00) >> 4));
            q[55] = (uint8_t) ((q[55] & 0x0f) | ((d & 0xf000) >> 8));
        } else {
            std::memcpy(q, &d, 2);
        }
    }
}
struct Layer {
    rx::GeoLayer geo;
    gg::IqCpu fm;
    std::vector<std::vector<uint8_t>> blobs;
};
static bool make_layers(std::vector<Layer>& L, const std::vector<std::pair<int, int>>& types, int n_exp) {
    std::mt19937_64 rng(20261025);
    for (const auto& t : types) {
        Layer l;
        std::string err;
        if (!gg::iq_cpu_fmt(t.first, t.second, E, FF, l.fm, err)) {
            std::printf("layer types %d / %d: %s\n", t.first, t.second, err.c_str());
            return false;
        }
        const size_t b[3] = {(size_t) FF * l.fm.gu_row, (size_t) FF * l.fm.gu_row, (size_t) E * l.fm.d_row};
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            l.geo.K[r] = r < 2 ? t.first : t.second;
            l.geo.off[r] = off;
            off += (b[r] + 255) & ~(size_t) 255;
        }
        l.geo.blob = off;
        for (int e = 0; e < n_exp; ++e) {
            std::vector<uint8_t> blob(off, 0);
            for (int r = 0; r < 3; ++r) fill(rng, l.geo.K[r], blob.data() + l.geo.off[r], b[r]);
            l.blobs.push_back(std::move(blob));
        }
        L.push_back(std::move(l));
    }
    return true;
}
// sum_i w_i expert_i(x) over picks [i0, i1) of a row, the CPU lane's arithmetic, in pick order (the CPU server's)
static void lane_sum(const Layer& l, const rx::ReqRow& rw, int i0, int i1, const float* x, std::vector<float>& y) {
    y.assign((size_t) E, 0.0f);
    std::vector<uint8_t> act(l.fm.act_bytes), hact(l.fm.h_bytes);
    std::vector<float> h((size_t) FF), yo((size_t) E);
    gg::iq_cpu_quant(l.fm.gu_act, x, E, act.data());
    for (int i = i0; i < i1; ++i) {
        const uint8_t* b = l.blobs[(size_t) rw.e[i]].data();
        const void* ap[1] = {act.data()};
        float* hp[1] = {h.data()};
        gg::iq_cpu_gate_up(l.fm, b + l.geo.off[0], b + l.geo.off[1], ap, 1, hp, 0, FF, kLim);
        gg::iq_cpu_quant(l.fm.d_act, h.data(), FF, hact.data());
        const void* hq[1] = {hact.data()};
        float* yp[1] = {yo.data()};
        gg::iq_cpu_down(l.fm, b + l.geo.off[2], hq, 1, yp, 0, E);
        for (int c = 0; c < E; ++c) y[(size_t) c] += rw.w[i] * yo[(size_t) c];
    }
}
static double norm(const std::vector<float>& a) {
    double s = 0;
    for (float v : a) s += (double) v * v;
    return std::sqrt(s);
}

// a main process up to the blobs: false (and why) when it did not get there
static bool connect(rx::Main& m, int port, const std::vector<Layer>& L, std::string& err) {
    if (!m.accept("127.0.0.1:" + std::to_string(port), kToken, 30, err)) return false;
    rx::GeoMsg gm;
    gm.n_embd = E;
    gm.n_ff = FF;
    gm.n_layers = (int) L.size();
    gm.limit = kLim;
    std::vector<rx::GeoLayer> gl;
    for (const Layer& l : L) gl.push_back(l.geo);
    if (!m.send_geo(gm, gl, err)) return false;
    int nb = 0;
    for (size_t li = 0; li < L.size(); ++li)
        for (size_t e = 0; e < L[li].blobs.size(); ++e, ++nb)
            if (!m.send_blob((int) li, (int) e, L[li].blobs[e].data(), L[li].blobs[e].size(), err)) return false;
    std::vector<uint8_t> ok;
    if (!m.end_blobs(nb, ok, 60000.0, err)) return false;
    for (uint8_t v : ok)
        if (v != 1) {
            err = "a blob not held";
            return false;
        }
    return true;
}

// ---------------------------------------------------------------------------------------------------- the cases
static void case_rules() {
    check(!rx::lane_naps(1.5, false) && rx::lane_naps(2.5, false), "lane_naps: no link: a nap after 2 ms idle");
    check(!rx::lane_naps(150.0, true) && rx::lane_naps(rx::kSpinMs + 1.0, true),
          "lane_naps: the link up: no nap before kSpinMs idle (150 ms: none), one after");
    const std::vector<int> c = rx::off_link({0, 1, 2, 3, 4, 18}, {2, 18});
    check(c == std::vector<int>({0, 1, 3, 4}), "off_link: the link core's CPUs (2, 18) out, the rest in order");
    const int hw = (int) std::thread::hardware_concurrency();
    set_env("DS4_EP_CPU", nullptr);
    const int d = rx::link_cpu();
    set_env("DS4_EP_CPU", "off");
    const int off = rx::link_cpu();
    set_env("DS4_EP_CPU", "1");
    const int one = rx::link_cpu();
    set_env("DS4_EP_CPU", "100000");
    const int big = rx::link_cpu();
    set_env("DS4_EP_CPU", nullptr);
    check(d == (hw > 2 ? 2 : -1) && off == -1 && one == (hw > 1 ? 1 : -1) && big == -1,
          "link_cpu: 2 by default, off: -1, 1: 1, a CPU this PC has not: -1");
    // the siblings: each CPU among its own, the relation symmetric; Linux: as many as the kernel lists
    bool own = true, sym = true;
    for (int a = 0; a < std::min(hw, 64); ++a) {
        const std::vector<int> s = os::cpu_siblings(a);
        own = own && std::find(s.begin(), s.end(), a) != s.end();
        for (int b : s) {
            const std::vector<int> t = os::cpu_siblings(b);
            sym = sym && std::find(t.begin(), t.end(), a) != t.end();
        }
    }
    check(own && sym, "cpu_siblings: every CPU among its core's, the relation symmetric");
    const std::vector<int> s2 = os::cpu_siblings(hw > 2 ? 2 : 0);
    std::string sl;
    for (int x : s2) sl += " " + std::to_string(x);
    std::printf("  CPU %d's core:%s\n", hw > 2 ? 2 : 0, sl.c_str());
#if !defined(_WIN32)
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(hw > 2 ? 2 : 0) + "/topology/thread_siblings_list");
    std::string raw;
    if (f >> raw && raw.find('-') == std::string::npos)
        check((size_t) std::count(raw.begin(), raw.end(), ',') + 1 == s2.size(),
              "cpu_siblings: the kernel's list (" + raw + ")");
#endif
}

// os::pin_self(-1): the thread's mask before its first pin again, not every CPU (U1's rule)
static void case_pin() {
    bool ok = false, skipped = false;
    int pinned = -2, after = -2;
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
        os::pin_self(1);
        pinned = os::pinned_cpu();
        os::pin_self(0);   // (a second pin: what -1 restores is still the mask from before the first)
        os::pin_self(-1);
        after = os::pinned_cpu();
        const DWORD_PTR was = SetThreadAffinityMask(t, 3);
        ok = was == 3;
#else
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(0, &s);
        CPU_SET(1, &s);
        if (std::thread::hardware_concurrency() < 3 || sched_setaffinity(0, sizeof s, &s) != 0) {
            skipped = true;
            return;
        }
        os::pin_self(1);
        pinned = os::pinned_cpu();
        os::pin_self(0);   // (a second pin: what -1 restores is still the mask from before the first)
        os::pin_self(-1);
        after = os::pinned_cpu();
        cpu_set_t g;
        sched_getaffinity(0, sizeof g, &g);
        ok = CPU_COUNT(&g) == 2 && CPU_ISSET(0, &g) && CPU_ISSET(1, &g);
#endif
    }).join();
    if (skipped) {
        std::printf("  pin_self's restore: skipped here (fewer than 3 CPUs)\n");
        return;
    }
    check(pinned == 1, "pin_self(1): the thread on CPU 1 alone (pinned_cpu)");
    check(ok && after == -1, "pin_self(-1) after pins to 1 and 0: the mask before the first pin again ({0, 1}, not every CPU)");
}

// TCP_QUICKACK after each read (Linux): Conn counts the reads it set it after
static void case_quickack(int port) {
    rx::Conn a, b;
    std::string ea, eb;
    bool la = false;
    std::thread t([&] { la = a.listen_accept("127.0.0.1:" + std::to_string(port), 10, ea); });
    sleep_ms(200);
    const bool lb = b.dial("127.0.0.1:" + std::to_string(port), 5, eb);
    t.join();
    if (!la || !lb) {
        check(false, "a loopback link for the QUICKACK case (" + ea + eb + ")");
        return;
    }
    std::vector<uint8_t> p(100, 1), q;
    rx::Hdr h;
    std::string e;
    for (uint32_t i = 1; i <= 3; ++i) {
        b.send(rx::Ping, i, 0, 0, 0, p.data(), p.size(), e);
        (void) a.recv(h, q, rx::now_ms() + 2000.0, e);
    }
#if defined(__linux__)
    check(a.quickacks >= 3, "Linux: TCP_QUICKACK after each read (" + std::to_string(a.quickacks) + " of 3+ reads)");
#else
    std::printf("  TCP_QUICKACK: not on this OS (%llu)\n", (unsigned long long) a.quickacks);
#endif
}

int child_main(int port) {
    const char* tv = std::getenv("STRATA_STAGE_TOKEN");
    return rx::serve_cpu("127.0.0.1:" + std::to_string(port), tv != nullptr ? tv : "", true);
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--child") return child_main(std::atoi(argv[2]));
    const int port = argc > 1 ? std::atoi(argv[1]) : 27893;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    case_rules();
    case_pin();
    case_quickack(port + 2);

    const int hw = (int) std::thread::hardware_concurrency();
    const int lc = hw > 2 ? 2 : hw > 1 ? 1 : -1;   // (the children's link core)
    set_env("STRATA_STAGE_TOKEN", kToken);
    set_env("DS4_EP_CPU", lc >= 0 ? std::to_string(lc).c_str() : "off");
    std::vector<Layer> L;
    if (!make_layers(L, {{16, 16}, {16, 17}}, 4)) return 2;

    // ---- the server: requests, a late one, keep-awake, --expert-server-once
    Child ch;
    check(ch.start(port), "the CPU expert server started (a child)");
    rx::Main m;
    std::string err;
    const bool up = connect(m, port, L, err);
    check(up, "the main took the server, sent the geometry and its blobs: all held (" + err + ")");
    if (!up) {
        ch.kill();
        ch.wait(5000);
        std::printf("%d FAILED\n", fails);
        return 1;
    }
    int32_t pong[3] = {-9, -9, -9};
    check(m.ping(1, 56, rx::now_ms() + 5000.0, err, pong) == 1, "a ping's pong");
    if (lc >= 0) check(pong[2] == lc, "the server's serve loop on the link core (CPU " + std::to_string(lc) +
                                          "): pinned to " + std::to_string(pong[2]));

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f), uw(0.05f, 0.4f);
    double worst_ratio = 0;
    bool bits = true, all_in = true;
    int nreq = 0;
    uint32_t seq = 100;
    for (int it = 0; it < 12; ++it) {
        const int li = it % 2, T = it % 3 == 0 ? 4 : 1;
        // T rows, 4 picks each (experts 0..3 in a random order): those of experts 0, 1 to the server, the rest here
        std::vector<rx::ReqRow> all((size_t) T), rem((size_t) T);
        std::vector<std::vector<float>> x((size_t) T, std::vector<float>((size_t) E)), xb = x;
        std::vector<const float*> xp((size_t) T);
        for (int t = 0; t < T; ++t) {
            int ord[4] = {0, 1, 2, 3};
            std::shuffle(ord, ord + 4, rng);
            rx::ReqRow& a = all[(size_t) t];
            a.t = (uint8_t) t;
            a.n = 4;
            for (int i = 0; i < 4; ++i) {
                a.e[i] = (int16_t) ord[i];
                a.w[i] = uw(rng);
            }
            rx::ReqRow& r = rem[(size_t) t];   // (the server's picks first, in their order; the others after)
            r.t = (uint8_t) t;
            r.n = 0;
            for (int i = 0; i < 4; ++i)
                if (a.e[i] < 2) {
                    r.e[r.n] = a.e[i];
                    r.w[r.n++] = a.w[i];
                }
            for (int c = 0; c < E; ++c) {
                x[(size_t) t][(size_t) c] = ux(rng);
                xb[(size_t) t][(size_t) c] = rx::bf2f(rx::f2bf(x[(size_t) t][(size_t) c]));
            }
            xp[(size_t) t] = x[(size_t) t].data();
        }
        if (!m.send_req(++seq, li, T, rem.data(), xp.data(), err)) {
            check(false, "a request sent (" + err + ")");
            break;
        }
        const uint16_t* y = nullptr;
        int srv_us = 0;
        const int got = m.wait_resp(seq, T, rx::now_ms() + 20000.0, &y, srv_us, err);
        all_in = all_in && got == 1;
        if (got != 1) break;
        ++nreq;
        for (int t = 0; t < T; ++t) {
            const Layer& l = L[(size_t) li];
            std::vector<float> want, ref, here;
            lane_sum(l, rem[(size_t) t], 0, rem[(size_t) t].n, xb[(size_t) t].data(), want);   // (the server's)
            for (int c = 0; c < E; ++c) bits = bits && y[(size_t) t * E + c] == rx::f2bf(want[(size_t) c]);
            // here: the other picks (x unrounded); the split sum against all of them here
            rx::ReqRow loc = all[(size_t) t];
            loc.n = 0;
            for (int i = 0; i < 4; ++i)
                if (all[(size_t) t].e[i] >= 2) {
                    loc.e[loc.n] = all[(size_t) t].e[i];
                    loc.w[loc.n++] = all[(size_t) t].w[i];
                }
            lane_sum(l, loc, 0, loc.n, x[(size_t) t].data(), here);
            lane_sum(l, all[(size_t) t], 0, 4, x[(size_t) t].data(), ref);
            std::vector<float> diff((size_t) E), part((size_t) E);
            for (int c = 0; c < E; ++c) {
                part[(size_t) c] = rx::bf2f(y[(size_t) t * E + c]);
                diff[(size_t) c] = here[(size_t) c] + part[(size_t) c] - ref[(size_t) c];
            }
            worst_ratio = std::max(worst_ratio, norm(diff) / std::max(1e-30, norm(part)));
        }
    }
    check(all_in && nreq == 12, "12 requests (T = 1 and 4, both layers) answered");
    check(bits, "each reply is the CPU lane's sum of the server's picks for x in bf16, rounded to bf16, bit for bit");
    char b[160];
    // (x in bf16 moves some of the int8 activations by one step, here and again in h: a few % of the server's part;
    // a wrong pick, weight or row is ~1)
    std::snprintf(b, sizeof b, "the split sum is the all-here one within the server's roundings (|diff| / |its part| "
                  "%.2e < 0.10)", worst_ratio);
    check(worst_ratio < 0.10, b);

    // a late reply: 0 by its deadline; the next request has its own; the late one in take_late
    {
        rx::ReqRow r;
        r.n = 2;
        r.e[0] = 0;
        r.e[1] = 1;
        r.w[0] = r.w[1] = 0.25f;
        std::vector<float> x((size_t) E, 0.5f);
        const float* xp[1] = {x.data()};
        const uint16_t* y = nullptr;
        int us = 0;
        (void) m.take_late();
        const bool s1 = m.send_req(500, 0, 1, &r, xp, err);
        const int g1 = m.wait_resp(500, 1, rx::now_ms() - 1.0, &y, us, err);
        const bool s2 = m.send_req(501, 0, 1, &r, xp, err);
        const int g2 = m.wait_resp(501, 1, rx::now_ms() + 20000.0, &y, us, err);
        const auto late = m.take_late();
        check(s1 && g1 == 0, "a reply past its deadline: 0 (its picks computed here)");
        check(s2 && g2 == 1 && late.size() == 1 && late[0].first == 500,
              "the next request gets its own reply; the late one skipped, in take_late");
    }

    // keep-awake: a Wake within the spin keeps the server's receive spinning; without one it waits in poll
    {
        int32_t p0[3], p1[3], q0[3], q1[3];
        std::string e;
        const bool a0 = m.ping(10, 56, rx::now_ms() + 5000.0, e, p0) == 1;
        const bool early = m.wake(e);   // (just sent: not quiet for kWakeQuietMs)
        sleep_ms(120);
        const bool w1 = m.wake(e);
        sleep_ms(120);
        const bool w2 = m.wake(e);
        sleep_ms(120);
        const bool a1 = m.ping(11, 56, rx::now_ms() + 5000.0, e, p1) == 1;
        check(!early, "no Wake right after a send (kWakeQuietMs)");
        check(a0 && a1 && w1 && w2 && p1[0] == p0[0] + 2, "two Wakes 120 ms apart sent and taken (" +
                                                             std::to_string(p1[0] - p0[0]) + ")");
        check(a0 && a1 && p1[1] == p0[1], "with them the server's receive kept spinning for 360 ms (waits in poll: " +
                                              std::to_string(p1[1] - p0[1]) + ")");
        const bool c0 = m.ping(12, 56, rx::now_ms() + 5000.0, e, q0) == 1;
        sleep_ms(360);
        const bool c1 = m.ping(13, 56, rx::now_ms() + 5000.0, e, q1) == 1;
        check(c0 && c1 && q1[1] > q0[1], "without them it waited in poll after kSpinMs (" + std::to_string(q1[1] - q0[1]) +
                                             " waits)");
    }

    // --expert-server-once: the main leaves, the server exits 0
    m.close();
    const int rc = ch.wait(10000);
    check(rc == 0, "--expert-server-once: the server exited 0 when the main left (" + std::to_string(rc) + ")");
    if (rc < 0) {
        ch.kill();
        ch.wait(5000);
    }

    // ---- a killed server: the wait ends with -1 at once, the picks computed here
    {
        std::vector<Layer> L1;
        make_layers(L1, {{16, 16}}, 2);
        Child c2;
        c2.start(port + 1);
        rx::Main m2;
        std::string e2;
        const bool up2 = connect(m2, port + 1, L1, e2);
        check(up2, "a second server taken (" + e2 + ")");
        if (up2) {
            c2.kill();
            const int krc = c2.wait(5000);
            rx::ReqRow r;
            r.n = 1;
            r.e[0] = 0;
            r.w[0] = 1.0f;
            std::vector<float> x((size_t) E, 0.25f);
            const float* xp[1] = {x.data()};
            const uint16_t* y = nullptr;
            int us = 0;
            const double t0 = rx::now_ms();
            int g = -1;
            if (m2.send_req(1, 0, 1, &r, xp, e2)) g = m2.wait_resp(1, 1, t0 + 10000.0, &y, us, e2);
            const double dt = rx::now_ms() - t0;
            check(krc >= 0 && g == -1 && dt < 2000.0, "a killed server: the request's wait ends with -1 in " +
                                                          std::to_string((int) dt) + " ms (no hang; " + e2 + ")");
        } else {
            c2.kill();
            c2.wait(5000);
        }
    }
    if (fails == 0) std::printf("ALL OK\n");
    else std::printf("%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
