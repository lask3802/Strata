// Strata. The remote experts' send gate and reply deadline (rx::Gate, src/dsv41/ds4_rx.hpp): the main process sends a
// window's remote picks only when the expert server's round trip is expected to end no later than the CPU lane
// computing them with its own, and waits for a reply until the later of DS4_RX_TIMEOUT_MS after the lane's part and
// 1.5 x the round trip expected at the request's rows.  No GPU, no socket.  The numbers are the 3070's over 1 GbE
// (rx-poc-report.md): T=1 0.66 ms (1 pick), T=4 2.17 ms; the lane 0.26 ms a pick.
#include "../../src/dsv41/ds4_rx.hpp"

#include <cmath>
#include <cstdio>

namespace rx = strata::ds4::rx;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    {
        rx::Gate g;
        check(g.send(1, 1, 1, 0), "nothing learned yet: the picks go out");
        check(near(g.deadline(10.0, 11.0, 2.0, 1), 13.0), "nothing learned yet: the deadline is the timeout after the lane");
        g.rtt_seen(1, 0.66);
        check(near(g.rtt[1], 0.66), "the first round trip is the estimate");
        g.rtt_seen(1, 1.66);
        check(near(g.rtt[1], 0.76), "then an EWMA of 0.1");
    }
    {
        rx::Gate g;
        g.rtt_seen(1, 0.66);
        g.pick_seen(1, 0.26);
        check(g.send(1, 1, 1, 4), "T=1, 1 remote pick, 4 own: 0.66 <= 1.30 ms, out");
        check(!g.send(1, 1, 1, 0), "T=1, 1 remote pick, none own: 0.66 > 0.26 ms, kept here");
        int out = 0;
        for (int i = 0; i < 63; ++i) out += g.send(1, 1, 1, 0) ? 1 : 0;
        check(out == 2, "kept here in a row: every 32nd goes out (a probe)");
        check(g.send(1, 1, 2, 1), "a window that pays after kept ones: out");
        check(!g.send(1, 1, 1, 0), "and the run counts from there");
    }
    {
        rx::Gate g;   // the T=4 case the plain deadline lost: a window whose picks are almost all remote
        g.rtt_seen(4, 2.17);
        g.pick_seen(4, 0.26);
        const double sent = 100.0, lane_done = 100.1;
        const double dl = g.deadline(sent, lane_done, 2.0, 4);
        check(near(dl, sent + 1.5 * 2.17), "T=4: the deadline is 1.5 x the round trip after the send (3.26 ms), not 2.1");
        check(sent + 2.17 <= dl, "T=4: a reply at the mean round trip is in time");
        check(sent + 2.17 > lane_done + 2.0, "(the lane's part + 2 ms alone would have called it late)");
        check(g.send(4, 4, 8, 4), "T=4, 8 remote picks, 4 own: 2.17 <= 3.12 ms, out");
        check(!g.send(4, 4, 4, 2), "T=4, 4 remote picks, 2 own: 2.17 > 1.56 ms, kept here");
        g.rtt_seen(1, 0.66);
        check(near(g.deadline(sent, 101.0, 2.0, 1), 103.0), "T=1: the lane's part + 2 ms is the later one");
    }
    {
        // the slot's rx-ds arm (2ce60ad7): replies already there when the lane finished its part, timed at the lane's
        // end, made the round trip at 1 row 6.9 ms (the 1.1-6 ms of a window's own picks) and kept 168 of 176 T=1 windows
        rx::Gate g;
        g.pick_seen(1, 0.27);
        g.rtt_seen(1, 6.9, true);
        check(!g.send(1, 1, 1, 4), "an estimate of 6.9 ms keeps a T=1 window here (1.35 ms of lane)");
        g.rtt_seen(1, 9.0, false);
        check(near(g.rtt[1], 6.9), "a reply there already, after 9 ms of lane: no higher");
        for (int i = 0; i < 40; ++i) g.rtt_seen(1, 1.1, false);
        check(g.rtt[1] < 1.35, "replies there already after 1.1 ms of lane: the estimate falls under that");
        check(g.send(1, 1, 1, 4), "and the T=1 window goes out again");
        g.rtt_seen(1, 2.0, true);
        check(g.rtt[1] > 1.1, "a reply the lane waited 2 ms for raises it");
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
