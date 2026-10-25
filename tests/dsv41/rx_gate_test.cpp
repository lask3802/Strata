// Strata. The remote experts' send gate and reply deadline (rx::Gate, src/dsv41/ds4_rx.hpp): the main process sends a
// window's remote picks only when the expert server's round trip is expected to end no later than the CPU lane
// computing them with its own, and waits for a reply until the later of the lane's time for the remote picks after
// its part and 1.5 x the round trip expected at the request's rows.  No GPU, no socket.  The numbers are the 3070's
// over 1 GbE (rx-poc-report.md): T=1 0.66 ms (1 pick), T=4 2.17 ms; the lane 0.26 ms a pick.
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
        // (lead 2026-10-25) the wait after the lane's part: as long as the lane would take for the remote picks
        rx::Gate g;
        check(near(g.wait_ms(1, 2, 2.0, false), 2.0), "the lane's time a pick not known yet: the fixed 2 ms");
        g.pick_seen(1, 0.26);
        check(near(g.wait_ms(1, 1, 2.0, false), 0.26), "1 remote pick at 0.26 ms a pick: wait 0.26 ms");
        check(near(g.wait_ms(1, 3, 2.0, false), 0.78), "3 remote picks: 0.78 ms");
        check(near(g.wait_ms(1, 3, 2.0, true), 2.0), "DS4_RX_TIMEOUT_MS set: the fixed wait");
        rx::Gate f;
        f.pick_seen(1, 0.03);
        check(near(f.wait_ms(1, 1, 2.0, false), rx::Gate::kMinWaitMs), "a fast lane: 0.1 ms at least");
        g.pick_seen(4, 0.5);
        check(near(g.wait_ms(4, 2, 2.0, false), 1.0), "T=4: its own time a pick (2 x 0.5 ms)");
        g.rtt_seen(1, 0.7);
        check(near(g.deadline(10.0, 11.0, g.wait_ms(1, 1, 2.0, false), 1), 11.26), "the lane's part + 0.26 ms is the later");
        g.rtt_seen(1, 3.0);   // (0.93 ms: 1.5 x is 1.395 after the send)
        check(near(g.deadline(10.0, 11.0, g.wait_ms(1, 1, 2.0, false), 1), 10.0 + 1.5 * 0.93), "1.5 x the round trip is");
        // the lane's deadline and the link's bound (rx::Backoff counts against the fixed wait's, not the lane's)
        const auto d = g.deadlines(1, 1, 1, 10.0, 12.0, 2.0, false, true);
        check(near(d.first, 12.26) && near(d.second, 14.0), "the lane's deadline 0.26 ms after its part, the bound 2 ms");
        const auto w = g.deadlines(1, 1, 1, 10.0, 12.0, 2.0, false, false);
        check(near(w.first, 14.0) && near(w.second, 14.0), "no kernel arrival times: the fixed wait, the bound is it");
        const auto x = g.deadlines(1, 1, 1, 10.0, 12.0, 2.0, true, true);
        check(near(x.first, 14.0) && near(x.second, 14.0), "DS4_RX_TIMEOUT_MS set: the bound is the deadline");
        const auto y = g.deadlines(4, 2, 4, 10.0, 12.0, 2.0, false, true);
        check(near(y.first, 13.0) && near(y.second, 14.0), "T=4: its own time a pick, the bound still 2 ms");
        const auto z = g.deadlines(4, 6, 4, 10.0, 12.0, 2.0, false, true);
        check(near(z.first, 15.0) && near(z.second, 14.0), "6 picks at T=4: the lane waits 3 ms, past the bound");
    }
    {
        // (lead 2026-10-25) the gate's start: the load's pings seed it, and the clamp holds from the first sample (a
        // cold server's first reply took 178 ms in a slot, and the gate kept 936 windows here until probes brought it
        // down)
        rx::Gate cold;
        cold.rtt_seen(1, 178.0);
        check(near(cold.rtt[1], 178.0), "no seed: a cold first reply is the estimate");
        rx::Gate s;
        s.seed(1, 0.7);
        check(near(s.rtt[1], 0.7), "seeded: the warm-up requests' median is the estimate");
        s.rtt_seen(1, 178.0, true, 0.0);
        check(near(s.rtt[1], 0.9 * 0.7 + 0.1 * 2.1), "seeded: the cold first reply counts as 3 x the seed");
        s.pick_seen(1, 0.26);
        check(s.send(1, 1, 1, 3), "and the next window goes out (0.84 ms against 4 picks x 0.26)");
        rx::Gate z;
        z.seed(1, 0.0);
        z.rtt_seen(1, 5.0);
        check(near(z.rtt[1], 5.0), "a seed of 0 (no warm-up): nothing seeded");
        rx::Gate q;   // (a seed ends a clamped run)
        q.seed(1, 0.7);
        for (int i = 0; i < 4; ++i) q.rtt_seen(1, 100.0, true, 100.0 * i);
        q.seed(1, 0.7);
        q.rtt_seen(1, 100.0, true, 1000.0);
        check(near(q.rtt[1], 0.9 * 0.7 + 0.1 * 2.1), "a seed after 4 clamped: the run starts again, clamped");
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
        // the link gets slower for good (0.8 -> 8 ms): 4 samples in a row clamped, then the 5th counts as it is (sent
        // 20 ms apart: the run spans 80 ms)
        rx::Gate g;
        for (int i = 0; i < 10; ++i) g.rtt_seen(1, 0.8, true, 10.0 * i);
        for (int i = 0; i < 4; ++i) g.rtt_seen(1, 8.0, true, 200.0 + 20.0 * i);
        const double four = 0.8 * 1.2 * 1.2 * 1.2 * 1.2;
        check(std::fabs(g.rtt[1] - four) < 1e-9, "4 slow samples in a row: each clamped, the estimate 0.8 x 1.2^4");
        g.rtt_seen(1, 8.0, true, 280.0);
        check(std::fabs(g.rtt[1] - (0.9 * four + 0.8)) < 1e-9, "the 5th in a row counts as it is: the link did get slower");
        g.rtt_seen(1, 30.0, true, 300.0);
        check(g.rtt[1] < 0.9 * (0.9 * four + 0.8) + 0.3 * (0.9 * four + 0.8) + 1e-9, "and the run starts again: clamped");
        rx::Gate h;   // (a sample within 3 x between ends the run)
        for (int i = 0; i < 10; ++i) h.rtt_seen(1, 0.8);
        for (int i = 0; i < 4; ++i) h.rtt_seen(1, 8.0);
        h.rtt_seen(1, 0.8);
        const double r = h.rtt[1];
        h.rtt_seen(1, 8.0);
        check(std::fabs(h.rtt[1] - 1.2 * r) < 1e-9, "4 slow (clamped), one fast, one slow: the fast one ends the run, the slow one is clamped again");
        // (lead 2026-10-25) a stall bunches its replies: 5 clamped sent within 10 ms stay clamped; 5 over 60 ms: the
        // 5th counts as it is
        const double five = four * 1.2;
        rx::Gate s;
        for (int i = 0; i < 10; ++i) s.rtt_seen(1, 0.8, true, 10.0 * i);
        for (int i = 0; i < 5; ++i) s.rtt_seen(1, 27.0 - i, true, 1000.0 + 2.5 * i);
        check(std::fabs(s.rtt[1] - five) < 1e-9, "a stall: 5 slow sent within 10 ms, all 5 clamped (0.8 x 1.2^5)");
        rx::Gate w;
        for (int i = 0; i < 10; ++i) w.rtt_seen(1, 0.8, true, 10.0 * i);
        for (int i = 0; i < 5; ++i) w.rtt_seen(1, 8.0, true, 1000.0 + 15.0 * i);
        check(std::fabs(w.rtt[1] - (0.9 * four + 0.8)) < 1e-9, "a slower link: 5 slow sent over 60 ms, the 5th as it is");
        rx::Gate o;   // (a late reply's sample comes after later sends': the span is of the sends, any order)
        for (int i = 0; i < 10; ++i) o.rtt_seen(1, 0.8, true, 10.0 * i);
        const double so[5] = {1030.0, 1045.0, 1060.0, 1015.0, 1000.0};
        for (double x : so) o.rtt_seen(1, 8.0, true, x);
        check(std::fabs(o.rtt[1] - (0.9 * four + 0.8)) < 1e-9, "sends out of order over 60 ms: the 5th as it is");
        rx::Gate g2 = g;
        g2.rtt_seen(1, 0.8, true, 310.0);   // (ends g's run of one)
        const double base2 = g2.rtt[1];
        for (int i = 0; i < 5; ++i) g2.rtt_seen(1, 27.0, true, 400.0 + 2.5 * i);
        check(std::fabs(g2.rtt[1] - base2 * 1.2 * 1.2 * 1.2 * 1.2 * 1.2) < 1e-9,
              "a second run in the same gate: its own span (bunched: all 5 clamped)");
        rx::Gate n;   // (no send times: the count alone)
        for (int i = 0; i < 10; ++i) n.rtt_seen(1, 0.8);
        for (int i = 0; i < 5; ++i) n.rtt_seen(1, 8.0);
        check(std::fabs(n.rtt[1] - (0.9 * four + 0.8)) < 1e-9, "no send times: the 5th as it is (the count alone)");
        rx::Gate m;   // (the run's first sample has no send time, the 4 others are bunched)
        for (int i = 0; i < 10; ++i) m.rtt_seen(1, 0.8, true, 10.0 * i);
        m.rtt_seen(1, 8.0);
        for (int i = 0; i < 4; ++i) m.rtt_seen(1, 8.0, true, 1000.0 + 2.5 * i);
        check(std::fabs(m.rtt[1] - five) < 1e-9, "the first without a send time, then 4 within 10 ms: all clamped");
        rx::Gate e;   // (the edge: exactly 50 ms)
        for (int i = 0; i < 10; ++i) e.rtt_seen(1, 0.8, true, 10.0 * i);
        for (int i = 0; i < 5; ++i) e.rtt_seen(1, 8.0, true, 1000.0 + 12.5 * i);
        check(std::fabs(e.rtt[1] - (0.9 * four + 0.8)) < 1e-9, "5 slow over exactly 50 ms: the 5th as it is");
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
        int ly = -2;
        b.add(9, 3.0, 1, 17);
        check(b.take(9, s, rows, &ly) && ly == 17, "and its layer");
        // late for the lane's wait, the link's lateness decided against its bound (lead 2026-10-25: the short wait
        // must not drive the backoff)
        rx::LateBook c;
        bool op = false;
        double bd = -1;
        c.add(20, 10.0, 1, 3, 13.0);
        c.add(21, 11.0, 1, 3, 14.0);
        c.add(22, 12.0, 1, 3, -1.0);   // (decided at once: its deadline was the bound)
        check(c.expire(13.5) == 1, "past seq 20's bound, not 21's: one decided late");
        check(c.expire(13.6) == 0, "and once only");
        check(c.take(20, s, rows, nullptr, &op, &bd) && !op, "seq 20's reply after: already decided");
        check(c.take(21, s, rows, nullptr, &op, &bd) && op && near(bd, 14.0), "seq 21's reply: open, against 14 ms");
        check(c.expire(100.0) == 0, "taken: nothing left to expire");
        check(c.take(22, s, rows, nullptr, &op, &bd) && !op, "decided at once: not open");
        rx::LateBook a;
        check(a.add_late(30, 10.0, 1, 3, 12.3, 14.0) == 0, "late for the lane's deadline, before the bound: open");
        check(a.expire(14.0) == 0 && a.expire(14.01) == 1, "decided late only once the bound has passed");
        check(a.add_late(31, 10.0, 1, 3, 14.0, 14.0) == 1, "the deadline was the bound: decided late at once");
        check(a.take(31, s, rows, nullptr, &op, &bd) && !op, "and not open");
        check(a.add_late(32, 10.0, 1, 3, 12.3, 90.0) == 0, "a bound far out: open");
        check(a.add_late(96, 11.0, 1, 3, 12.3, 14.0) == 1, "64 sends on, its slot still open: decided late");
        check(a.take(96, s, rows, nullptr, &op, &bd) && op && near(bd, 14.0), "and the newer one open in it");
        // a late reply that came: its kernel arrival against its bound, else its read time
        check(!rx::LateBook::past(13.5, 10.0, 16.0, 14.0), "arrived before the bound (read after): not late");
        check(rx::LateBook::past(14.5, 10.0, 16.0, 14.0), "arrived after the bound: late");
        check(rx::LateBook::past(-1.0, 10.0, 16.0, 14.0), "no kernel time, read after the bound: late");
        check(!rx::LateBook::past(-1.0, 10.0, 13.9, 14.0), "no kernel time, read before the bound: not late");
        check(rx::LateBook::past(9.0, 10.0, 16.0, 14.0), "a stamp before the send (the clock stepped): the read's");
        check(!rx::LateBook::past(17.0, 10.0, 13.9, 14.0), "a stamp after the read (the clock stepped): the read's");
        check(!rx::LateBook::past(14.0, 10.0, 16.0, 14.0), "arrived at the bound itself: not late");
        check(!rx::LateBook::past(15.0, 10.0, 16.0, 14.0, true), "a shared stamp past the bound: an upper bound, not late");
        check(rx::LateBook::past(-1.0, 10.0, 16.0, 14.0, true), "shared but no kernel time: the read's");
        rx::LateBook p;
        p.add_late(40, 10.0, 1, 3, 12.3, 14.0);
        check(p.expire(15.0) == 1 && p.add_late(104, 20.0, 1, 3, 22.3, 24.0) == 0,
              "64 on, the slot's one expired already (not taken): not decided again");
        // in time for the lane's deadline: late for the link only past the bound, when the lane's wait went past it
        check(!rx::LateBook::in_time_past(13.0, 14.0, 13.5, 10.0, 13.6), "the lane's wait within the bound: not late");
        check(!rx::LateBook::in_time_past(14.0, 14.0, -1.0, 10.0, 14.001), "the deadline the bound, read just after: not late");
        check(rx::LateBook::in_time_past(15.0, 14.0, 14.5, 10.0, 14.6), "a 3 ms wait, arrived at 14.5: late for the link");
        check(!rx::LateBook::in_time_past(15.0, 14.0, 13.5, 10.0, 14.6), "a 3 ms wait, arrived at 13.5: not late");
    }
    {
        // 8 late of the last 64 (lead 2026-10-25): the link idle for `hold` requests, then tried again with a clean count
        using B = rx::Backoff;
        B k;
        k.hold = 3;
        bool trip = false;
        for (int i = 0; i < 7; ++i) trip = trip || k.seen(true) != B::On;
        check(!trip && !k.idle(), "7 late: the link stays on");
        for (int i = 0; i < 57; ++i) trip = trip || k.seen(false) != B::On;
        trip = trip || k.seen(true) != B::On;
        check(!trip && !k.idle(), "7 late, 57 on time, 1 late: the oldest late one is out of the last 64 (7), on");
        for (int i = 0; i < 6; ++i) trip = trip || k.seen(true) != B::On;
        check(!trip, "7 late in the last 64 again: on");
        k.reply();
        check(k.seen(true) == B::Idle, "the 8th late in the last 64: idle");
        const bool i1 = k.idle(), i2 = k.idle(), i3 = k.idle(), i4 = k.idle();
        check(i1 && i2 && i3 && !i4, "idle for the next 3 requests (hold), then the link is tried again");
        trip = false;
        for (int i = 0; i < 7; ++i) trip = trip || k.seen(true) != B::On;
        check(!trip && !k.idle(), "a clean count after: 7 late do not idle it");
        k.reply();   // (a late reply came after all)
        check(k.seen(true) == B::Idle && k.idle(), "and an 8th does, again, after a reply came: idle, not given up");
        B w;   // (the oldest of the 64 still counts)
        for (int i = 0; i < 7; ++i) w.seen(true);
        for (int i = 0; i < 56; ++i) w.seen(false);
        check(w.seen(true) == B::Idle, "7 late, 56 on time, 1 late: 8 in the last 64, idle");
        B s;   // (a server that stopped answering on a link still open)
        s.hold = 2;
        for (int i = 0; i < 7; ++i) s.seen(true);
        check(s.seen(true) == B::Idle, "the first idle spell: never given up at once");
        s.idle();
        s.idle();
        for (int i = 0; i < 7; ++i) s.seen(true);
        check(s.seen(true) == B::Silent, "8 late again and no reply at all since the spell: given up");
    }
    {
        // the remote experts beside expert parallelism's n_ff slices (U2): refused (lead 2026-10-25, U4a)
        check(rx::slice_refusal(true, 0, -1, 2304) == nullptr, "the whole n_ff (unresolved, -1): taken");
        check(rx::slice_refusal(true, 0, 2304, 2304) == nullptr, "the whole n_ff: taken");
        check(rx::slice_refusal(true, 0, 1024, 2304) != nullptr, "a slice [0, 1024): refused");
        check(rx::slice_refusal(true, 1024, 1280, 2304) != nullptr, "a slice [1024, 2304): refused");
        check(rx::slice_refusal(false, 0, 1024, 2304) == nullptr, "no --remote-experts: the slice is not ours to refuse");
    }
    {
        // the expert cache does not take the expert server's experts into VRAM (lead 2026-10-25: the settle after the
        // prompt moved 67, and the server's share of the lane's picks fell from 20.2% to 16.3%)
        const std::vector<int> hits = {3, 5, 7, 5};
        int slot[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        const float heat[8] = {0, 0, 0, 9.0f, 0, 6.0f, 0, 4.0f};
        const uint8_t none[8] = {0, 0, 0, 0, 0, 0, 0, 0}, held3[8] = {0, 0, 0, 1, 0, 0, 0, 0},
                      all[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        check(rx::cache_in(hits, slot, heat, nullptr) == 3, "no expert server: the hottest hit (3)");
        check(rx::cache_in(hits, slot, heat, none) == 3, "the server holds none of them: the hottest hit (3)");
        check(rx::cache_in(hits, slot, heat, held3) == 5, "the server holds 3: the next hottest (5)");
        check(rx::cache_in(hits, slot, heat, all) == -1, "the server holds all of them: none");
        slot[5] = 0;
        check(rx::cache_in(hits, slot, heat, held3) == 7, "5 in VRAM already, 3 on the server: 7");
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
