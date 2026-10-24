// Strata. The remote experts' link (src/dsv41/ds4_rx.cpp, rx::Conn) over loopback: a message's arrival time in the
// kernel (SO_TIMESTAMPNS, Linux) is when it was sent, not when recv read it - the engine's round trip to the expert
// server is that time minus the send, however long the CPU lane kept it from reading.  Where the kernel gives no time
// (Windows) arrival_ms is -1 and the round trip falls back to the read.  No GPU, no model.
//   ds4_rx_link_test [PORT (27891)]
#include "../../src/dsv41/ds4_rx.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace rx = strata::ds4::rx;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}
static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

int main(int argc, char** argv) {
    const std::string port = argc > 1 ? argv[1] : "27891";
    rx::Conn a, b;
    std::string ea, eb;
    bool la = false;
    std::thread t([&] { la = a.listen_accept("127.0.0.1:" + port, 10, ea); });
    sleep_ms(200);
    const bool lb = b.dial("127.0.0.1:" + port, 5, eb);
    t.join();
    check(la && lb, ("a loopback link (" + ea + eb + ")").c_str());
    if (!la || !lb) return 1;
    std::vector<uint8_t> p(10288, 7), q;
    rx::Hdr h;
    std::string e;
    // one message, read 30 ms after it was sent
    const double s1 = rx::now_ms();
    b.send(rx::Ping, 1, 0, 0, 0, p.data(), p.size(), e);
    sleep_ms(30);
    const int got = a.recv(h, q, rx::now_ms() + 1000.0, e);
    const double r1 = rx::now_ms(), a1 = a.arrival_ms();
    check(got == 1 && h.seq == 1 && q.size() == p.size(), "the message came whole");
    if (a1 < 0) {
        std::printf("no kernel arrival time here (not Linux): arrival_ms -1, the engine times the read\n");
    } else {
        std::printf("sent %.3f, in the kernel %.3f, read %.3f ms\n", 0.0, a1 - s1, r1 - s1);
        check(a1 >= s1 - 0.5 && a1 <= s1 + 5.0, "its arrival is the send (within 5 ms), not the read");
        check(r1 - a1 >= 25.0, "and 25 ms or more before the read");
        // two messages 20 ms apart, both read after: the second's arrival is its own send's
        const double s2 = rx::now_ms();
        b.send(rx::Ping, 2, 0, 0, 0, p.data(), p.size(), e);
        sleep_ms(20);
        const double s3 = rx::now_ms();
        b.send(rx::Ping, 3, 0, 0, 0, p.data(), p.size(), e);
        sleep_ms(30);
        a.recv(h, q, rx::now_ms() + 1000.0, e);
        const double a2 = a.arrival_ms();
        const int g3 = a.recv(h, q, rx::now_ms() + 1000.0, e);
        const double a3 = a.arrival_ms();
        std::printf("two: the first in the kernel %.3f after its send, the second %.3f after its send\n", a2 - s2, a3 - s3);
        check(g3 == 1 && h.seq == 3 && a3 >= s3 - 0.5 && a3 <= s3 + 5.0, "the second's arrival is its own send");
        check(a2 >= s2 - 0.5 && a2 <= s3 + 5.0, "the first's is no earlier than its send (one read: the later one's)");
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
