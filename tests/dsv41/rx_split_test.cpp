// Strata. Which lane computes which n_ff blocks of a pick under H' (rx::pick_lane, src/dsv41/ds4_rx_split.hpp): every
// kind of pick (the main's GPU or a CPU lane with no slice, the server's GPU set, a slice-able CPU-lane pick) x the
// request sent or not x the reply on time, late or lost x every k, each of a layer's n_ff blocks added exactly once.  The
// harness holds the lanes to what they can do, not to what pick_lane says: the main's GPU computes its picks whole
// whatever their lane, the server computes a whole pick only of the experts it holds and a slice only where the layer is
// sliced, a request not sent has no reply and no fallback.  serve_lane's own flow the same way (rx::split_plan: what of
// a layer's request goes tagged, the lane's range, the fallback's - the engine's choices).  Then the checks around it:
// a server's of a request row (rx::row_refusal: the slice bits, k, the parts) and the load's of a tagged slice
// (ep_slice_refusal, rx::slice_refusal: with H' a VRAM tier or zc slots are no reason, the rest still are).
//   ds4_rx_split_test     exit 0: all passed
#include "../../src/dsv41/ds4_ffslice.hpp"
#include "../../src/dsv41/ds4_rx.hpp"
#include "../../src/dsv41/ds4_rx_split.hpp"

#include <cstdio>
#include <string>

namespace rx = strata::ds4::rx;

static int checks_bad = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++checks_bad;
}

static void case_checks() {
    rx::ReqRow rw;
    rw.n = 3;
    rw.slice = 0x5;   // (picks 0 and 2 slice, 1 whole)
    check(rx::row_refusal(rw, 8, true).empty(), "row_refusal: 3 picks, 2 of them slice, k 8, parts here: taken");
    check(rx::row_refusal(rw, 8 | rx::kReqSets | rx::kReqWrong, true).empty(), "row_refusal: the flags beside k: taken");
    check(!rx::row_refusal(rw, 0, true).empty(), "row_refusal: slice picks without a k: refused");
    check(!rx::row_refusal(rw, rx::kReqSets, true).empty(), "row_refusal: slice picks, a flag but no k: refused");
    check(!rx::row_refusal(rw, 8, false).empty(), "row_refusal: slice picks and no parts here: refused");
    rw.slice = (uint16_t) (0x5 | 0x4 << rx::kPoolBit);   // (pick 2 W: whole on the server, U5c)
    check(rx::row_refusal(rw, 8, true, true).empty(), "row_refusal: a pool bit on a slice pick, a W here: taken");
    check(!rx::row_refusal(rw, 8, true).empty(), "row_refusal: a pool bit and no W here: refused");
    rw.slice = (uint16_t) (0x5 | 0x2 << rx::kPoolBit);
    check(!rx::row_refusal(rw, 8, true, true).empty(), "row_refusal: a pool bit on a whole pick: refused");
    rw.slice = 0x8;
    check(!rx::row_refusal(rw, 8, true).empty(), "row_refusal: a slice bit past the row's picks: refused");
    rw.slice = 0;
    check(rx::row_refusal(rw, 0, false).empty(), "row_refusal: whole picks alone, no k, no parts: taken (U4's rows)");
    rw.n = 0;
    check(!rx::row_refusal(rw, 0, false).empty(), "row_refusal: no picks: refused");
    rw.n = rx::kMaxPicks + 1;
    check(!rx::row_refusal(rw, 0, false).empty(), "row_refusal: more than kMaxPicks picks: refused");

    using strata::ds4::ep_slice_refusal;   // (ff0, ffn, n_ff, iq, vram, zc, mtp, worker, disk_skip, tagged)
    constexpr int FF = 2304;
    check(ep_slice_refusal(0, 2048, FF, true, true, false, false, false, false) != nullptr &&
              ep_slice_refusal(0, 2048, FF, true, false, true, false, false, false) != nullptr,
          "ep_slice_refusal untagged: a VRAM tier or zc slots refuse a slice (U2)");
    check(ep_slice_refusal(0, 2048, FF, true, true, true, false, false, false, true) == nullptr,
          "ep_slice_refusal tagged (H'): a VRAM tier and zc slots taken - their picks are never sent, never sliced");
    check(ep_slice_refusal(0, 2048, FF, false, false, false, false, false, false, true) != nullptr &&
              ep_slice_refusal(0, 2048, FF, true, false, false, true, false, false, true) != nullptr &&
              ep_slice_refusal(0, 2048, FF, true, false, false, false, true, false, true) != nullptr &&
              ep_slice_refusal(0, 2048, FF, true, false, false, false, false, true, true) != nullptr &&
              ep_slice_refusal(0, 2000, FF, true, false, false, false, false, false, true) != nullptr,
          "ep_slice_refusal tagged: no i-quant lane, a drafter layer, a stage worker, DISK_SKIP, no whole blocks: refused");
    check(rx::slice_refusal(true, 0, 2048, FF) != nullptr && rx::slice_refusal(true, 0, 2048, FF, true) == nullptr,
          "rx::slice_refusal: a slice beside the remote experts refused untagged (U4), taken tagged (H')");
}

// serve_lane's flow over rx::split_plan, as the engine runs it: the plan as if the request goes (its tagged picks go on
// the wire), the send, the plan again if it failed; a CPU-lane pick computed by the lane over sp.here, and with the
// reply its [k, nb) when it went tagged, or late or lost sp.late_slice; a pick of the server's set sent whole (the
// reply its [0, nb), late or lost sp.late_whole), or not sent: one of the lane's picks (sp.here).  Every block once
static int case_flow(int NB) {
    int bad = 0, cases = 0;
    const char* reply_name[2] = {"on time", "late or lost"};
    for (int k = 1; k < NB; ++k)
        for (int m = 0; m < 8; ++m)
            for (int reply = 0; reply < 2; ++reply) {
                const bool spl = m & 1, out = m & 2, ok = m & 4;
                const bool wire = rx::split_plan(spl, out, true, k, NB).tagged;   // (the picks rx_send carries tagged)
                const rx::SplitPlan sp = rx::split_plan(spl, out, ok, k, NB);
                const bool sent = out && ok;
                bool wrong = sent != sp.sent || (sp.sent && sp.tagged != wire);
                for (int kind = 0; kind < 2; ++kind) {   // 0: a CPU-lane pick, 1: one of the server's set
                    int times[16] = {};
                    auto add = [&](rx::BlockRange r) {
                        for (int b = r.b0; b < r.b1; ++b) ++times[b];
                    };
                    if (kind == 0) {
                        add(sp.here);
                        if (sent && wire) add(reply == 0 ? rx::BlockRange{k, NB} : sp.late_slice);
                    } else if (sent) {
                        add(reply == 0 ? rx::BlockRange{0, NB} : sp.late_whole);
                    } else {
                        add(sp.here);
                    }
                    for (int b = 0; b < NB; ++b) wrong = wrong || times[b] != 1;
                }
                ++cases;
                if (wrong && ++bad <= 8)
                    std::printf("  split_plan k %d, %s, %s, %s, reply %s: tagged %d sent %d here [%d, %d) late [%d, %d) / [%d, %d)  FAIL\n",
                                k, spl ? "the layer splits" : "no split", out ? "out" : "kept here", ok ? "sent" : "send failed",
                                reply_name[reply], (int) sp.tagged, (int) sp.sent, sp.here.b0, sp.here.b1, sp.late_whole.b0,
                                sp.late_whole.b1, sp.late_slice.b0, sp.late_slice.b1);
            }
    // the ranges as cpu_experts' values: [0, k) and [k, nb) of n_ff 2304 meet and end at n_ff; no blocks (rx off): whole
    bool ff = true;
    for (int k = 1; k < NB; ++k) {
        int a0, an, b0, bn;
        rx::ff_range({0, k}, NB, 2304, a0, an);
        rx::ff_range({k, NB}, NB, 2304, b0, bn);
        ff = ff && a0 == 0 && an == 256 * k && b0 == 256 * k && b0 + bn == 2304;
    }
    int w0, wn, e0, en;
    rx::ff_range({0, 1}, 1, 2304, w0, wn);
    rx::ff_range({0, 0}, NB, 2304, e0, en);
    ff = ff && w0 == 0 && wn == 2304 && en == 0;
    // the host-routed path sends nothing: its CPU-lane picks whole at every k, split layer or not
    bool host = true;
    for (int k = 1; k <= NB; ++k)
        for (int spl = 0; spl < 2; ++spl) {
            int h0, hn;
            rx::ff_range(rx::host_range(spl != 0, k, NB), NB, 2304, h0, hn);
            host = host && h0 == 0 && hn == 2304;
        }
    std::printf("split_plan: %d cases (k 1..8 x the layer splits x out x the send x the reply), %d wrong; ff_range %s; "
                "host_range %s\n", cases, bad, ff ? "ok" : "WRONG", host ? "ok" : "WRONG");
    return bad + (ff ? 0 : 1) + (host ? 0 : 1);
}

int main() {
    constexpr int NB = 9;
    int cases = 0, bad = 0;
    const char* reply_name[3] = {"on time", "late", "lost"};
    for (int k = 1; k < NB; ++k)
        for (int m = 0; m < 16; ++m)
            for (int reply = 0; reply < 3; ++reply) {
                const bool on_main = m & 1, server_has = m & 2, split_ok = m & 4, sent = m & 8;
                const rx::PickLane p = rx::pick_lane(on_main, server_has, split_ok, sent);
                int times[NB] = {};
                bool wrong = false;
                auto add = [&](rx::BlockRange r) {
                    for (int b = r.b0; b < r.b1; ++b) ++times[b];
                };
                // the main before the reply: its GPU computes whole whatever the lane; its CPU lane the lane's range
                add(on_main ? rx::BlockRange{0, NB} : rx::main_range(p, k, NB));
                const bool in_req = sent && rx::pick_sent(p);
                if (rx::pick_sent(p) && !sent) wrong = true;   // (a pick of a request not sent cannot be sent)
                if (in_req && rx::pick_slice(p) && !split_ok) wrong = true;   // (the server holds no slice of the layer)
                if (in_req && !rx::pick_slice(p) && !server_has) wrong = true;   // (nor that expert whole)
                if (in_req) add(reply == 0 ? rx::server_range(p, k, NB) : rx::fallback_range(p, k, NB));
                for (int b = 0; b < NB; ++b) wrong = wrong || times[b] != 1;
                ++cases;
                if (wrong) {
                    ++bad;
                    if (bad <= 12) {
                        std::printf("  k %d, %s%s%s%s, reply %s: lane %d, blocks added", k, on_main ? "on the main's GPU " : "",
                                    server_has ? "the server's GPU holds it " : "", split_ok ? "layer sliced " : "",
                                    sent ? "request sent" : "request not sent", reply_name[reply], (int) p);
                        for (int b = 0; b < NB; ++b) std::printf(" %d", times[b]);
                        std::printf("  FAIL\n");
                    }
                }
            }
    // the lanes as the table of ep-u6-plan.md 0 has them
    bool table = rx::pick_lane(true, true, true, true) == rx::PickLane::kMain &&
                 rx::pick_lane(false, true, true, true) == rx::PickLane::kServerWhole &&
                 rx::pick_lane(false, false, true, true) == rx::PickLane::kSplit &&
                 rx::pick_lane(false, false, true, false) == rx::PickLane::kMainWhole &&
                 rx::pick_lane(false, true, false, false) == rx::PickLane::kMainWhole &&
                 rx::pick_lane(false, false, false, true) == rx::PickLane::kMainWhole;
    // the ranges of a slice pick at k 8: the main [0, 8), the server [8, 9), its fallback the same
    const rx::BlockRange mr = rx::main_range(rx::PickLane::kSplit, 8, NB), sr = rx::server_range(rx::PickLane::kSplit, 8, NB),
                         fr = rx::fallback_range(rx::PickLane::kSplit, 8, NB);
    table = table && mr.b0 == 0 && mr.b1 == 8 && sr.b0 == 8 && sr.b1 == 9 && fr.b0 == 8 && fr.b1 == 9 &&
            rx::main_range(rx::PickLane::kServerWhole, 8, NB).empty() && rx::server_range(rx::PickLane::kMainWhole, 8, NB).empty();
    std::printf("pick_lane: %d cases (k 1..8 x on the main's GPU, the server's set, layer sliced, sent x the reply), %d wrong; "
                "the plan's table %s\n", cases, bad, table ? "ok" : "WRONG");
    const int flow_bad = case_flow(NB);
    case_checks();
    const bool ok = bad == 0 && table && checks_bad == 0 && flow_bad == 0;
    std::printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
