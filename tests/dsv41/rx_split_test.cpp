// Strata. Which lane computes which n_ff blocks of a pick under H' (rx::pick_lane, src/dsv41/ds4_rx_split.hpp): every
// kind of pick (the main's GPU or a CPU lane with no slice, the server's GPU set, a slice-able CPU-lane pick) x the
// request sent or not x the reply on time, late or lost x every k, each of a layer's n_ff blocks added exactly once.  The
// harness holds the lanes to what they can do, not to what pick_lane says: the main's GPU computes its picks whole
// whatever their lane, the server computes a whole pick only of the experts it holds and a slice only where the layer is
// sliced, a request not sent has no reply and no fallback.  Then the checks around it: a server's of a request row
// (rx::row_refusal: the slice bits, k, the parts) and the load's of a tagged slice (ep_slice_refusal, rx::slice_refusal:
// with H' a VRAM tier or zc slots are no reason, the rest still are).
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
    case_checks();
    const bool ok = bad == 0 && table && checks_bad == 0;
    std::printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
