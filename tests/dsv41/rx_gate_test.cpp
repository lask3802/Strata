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
    {
        // one 27 ms reply (the 1 GbE T=4 max) among 0.8 ms ones: the gate's next windows go out as before
        rx::Gate g;
        g.pick_seen(1, 0.27);
        for (int i = 0; i < 10; ++i) g.rtt_seen(1, 0.8);
        check(g.send(1, 1, 1, 4), "0.8 ms learned: a T=1 window with 1 remote pick and 4 own goes out (1.35 ms)");
        g.rtt_seen(1, 27.0);
        check(g.rtt[1] < 0.8 * 1.25, "a 27 ms spike counts as 3 x 0.8 ms: the estimate 0.96, not 3.42");
        bool all = true;
        for (int i = 0; i < 8; ++i) all = all && g.send(1, 1, 1, 4);
        check(all, "and the next windows still go out");
        rx::Gate f;   // (before 4 samples: no clamp - the first ones set the scale)
        f.rtt_seen(1, 0.8);
        f.rtt_seen(1, 27.0);
        check(near(f.rtt[1], 0.9 * 0.8 + 2.7), "the second sample is taken as it is");
    }
    {
        // the link gets slower for good (0.8 -> 8 ms): 4 samples in a row clamped, then the 5th counts as it is
        rx::Gate g;
        for (int i = 0; i < 10; ++i) g.rtt_seen(1, 0.8);
        for (int i = 0; i < 4; ++i) g.rtt_seen(1, 8.0);
        const double four = 0.8 * 1.2 * 1.2 * 1.2 * 1.2;
        check(std::fabs(g.rtt[1] - four) < 1e-9, "4 slow samples in a row: each clamped, the estimate 0.8 x 1.2^4");
        g.rtt_seen(1, 8.0);
        check(std::fabs(g.rtt[1] - (0.9 * four + 0.8)) < 1e-9, "the 5th in a row counts as it is: the link did get slower");
        g.rtt_seen(1, 30.0);
        check(g.rtt[1] < 0.9 * (0.9 * four + 0.8) + 0.3 * (0.9 * four + 0.8) + 1e-9, "and the run starts again: clamped");
        rx::Gate h;   // (a sample within 3 x between ends the run)
        for (int i = 0; i < 10; ++i) h.rtt_seen(1, 0.8);
        for (int i = 0; i < 4; ++i) h.rtt_seen(1, 8.0);
        h.rtt_seen(1, 0.8);
        const double r = h.rtt[1];
        h.rtt_seen(1, 8.0);
        check(std::fabs(h.rtt[1] - 1.2 * r) < 1e-9, "4 slow (clamped), one fast, one slow: the fast one ends the run, the slow one is clamped again");
    }
    {
        // a late request's reply that comes after all (lead's Q3): the book gives its send and rows once
        rx::LateBook b;
        double s = 0;
        int rows = 0;
        b.add(70, 100.0, 4);
        check(!b.take(71, s, rows), "a request that did not go late: none");
        check(b.take(70, s, rows) && near(s, 100.0) && rows == 4, "the late one: its send time and rows");
        check(!b.take(70, s, rows), "once only");
        b.add(5, 1.0, 1);
        b.add(69, 2.0, 2);   // (5 + 64: the same entry, a newer late request)
        check(!b.take(5, s, rows) && b.take(69, s, rows) && near(s, 2.0), "64 later: the newer one has the entry");
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
