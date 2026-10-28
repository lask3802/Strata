// Strata. Heavy-first deferral's pure logic (src/dsv41/ds4_defer_split.hpp: DS4_EP_DEFER_BELOW / DS4_EP_DEFER_TOP, their
// CHECK, DS4_EP_DEFER_STAT).  What a value says and the line it prints (unset, empty, not a number, NaN, <= 0, >= 1, both
// set, the deferral off, a check beside it, no layer to split); which layers a seq's request splits (never the last, a
// drafter layer, a layer before DS4_EP_DEFER_FROM, a window the deferral does not take); the shares of a token's route
// weights and which CPU-lane picks are heavy (the threshold's >=, the top r with its ties, the lane's picks alone); the
// adds check (every CPU-lane pick's n_ff blocks once, into its part) over serve_lane's flow - rx::split_plan's ranges, the
// reply on time or the fallback, the request sent tagged, untagged or not at all - and the faults it must see; the
// STAT tally and its lines.  No GPU, no model.
//   ds4_defer_split_test     exit 0: all passed
#include "../../src/dsv41/ds4_defer_split.hpp"
#include "../../src/dsv41/ds4_rx_split.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace ds = strata::ds4::dsplit;
namespace rx = strata::ds4::rx;

static int checks_bad = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++checks_bad;
}
static bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

static void case_parse() {
    float w = -1.0f;
    int r = -1;
    check(ds::parse_below(nullptr, w) == ds::Val::kUnset && ds::parse_below("", w) == ds::Val::kUnset,
          "parse_below: unset and empty are unset");
    check(ds::parse_below("abc", w) == ds::Val::kBad && ds::parse_below("0.15x", w) == ds::Val::kBad &&
              ds::parse_below("nan", w) == ds::Val::kBad,
          "parse_below: not a number, trailing text, NaN: not a rule");
    check(ds::parse_below("0", w) == ds::Val::kBad && ds::parse_below("-0.1", w) == ds::Val::kBad,
          "parse_below: <= 0: not a rule");
    check(ds::parse_below("1", w) == ds::Val::kNone && ds::parse_below("2.5", w) == ds::Val::kNone &&
              ds::parse_below("inf", w) == ds::Val::kNone,
          "parse_below: >= 1 (inf too): no pick heavy");
    check(ds::parse_below("0.15", w) == ds::Val::kOn && w == 0.15f, "parse_below: 0.15 -> 0.15");
    check(ds::parse_below("0.2 ", w) == ds::Val::kOn && w == 0.2f, "parse_below: a blank after it taken");
    check(ds::parse_top(nullptr, r) == ds::Val::kUnset && ds::parse_top("", r) == ds::Val::kUnset,
          "parse_top: unset and empty are unset");
    check(ds::parse_top("2", r) == ds::Val::kOn && r == 2, "parse_top: 2 -> 2");
    check(ds::parse_top("0", r) == ds::Val::kNone && ds::parse_top("-3", r) == ds::Val::kNone,
          "parse_top: < 1: no pick heavy");
    check(ds::parse_top("1.5", r) == ds::Val::kBad && ds::parse_top("x", r) == ds::Val::kBad,
          "parse_top: not a whole number: not a rule");
    check(ds::parse_top("99999", r) == ds::Val::kOn && r == 1000, "parse_top: a huge r held to 1000 (all picks)");
}

static void case_setup() {
    ds::Setup s = ds::setup(nullptr, nullptr, true, nullptr, 0, 43);
    check(!s.rule.on() && s.line.empty(), "setup: both unset: off, nothing said");
    s = ds::setup("", "", true, nullptr, 0, 43);
    check(!s.rule.on() && s.line.empty(), "setup: both empty: off, nothing said");
    s = ds::setup("0.1", "2", true, nullptr, 0, 43);
    check(!s.rule.on() && has(s.line, "both set: the split off"), "setup: both set: off (" + s.line + ")");
    s = ds::setup("zz", nullptr, true, nullptr, 0, 43);
    check(!s.rule.on() && has(s.line, "DS4_EP_DEFER_BELOW=zz: not a share") && has(s.line, "the split off"),
          "setup: BELOW not a number: off (" + s.line + ")");
    s = ds::setup("-1", nullptr, true, nullptr, 0, 43);
    check(!s.rule.on() && has(s.line, "the split off"), "setup: BELOW <= 0: off (" + s.line + ")");
    s = ds::setup("1", nullptr, true, nullptr, 0, 43);
    check(!s.rule.on() && has(s.line, ">= 1, no CPU-lane pick is heavy - no split"), "setup: BELOW >= 1: no split (" + s.line + ")");
    s = ds::setup(nullptr, "0", true, nullptr, 0, 43);
    check(!s.rule.on() && has(s.line, "DS4_EP_DEFER_TOP=0: < 1") && has(s.line, "no split"),
          "setup: TOP < 1: no split (" + s.line + ")");
    s = ds::setup(nullptr, "two", true, nullptr, 0, 43);
    check(!s.rule.on() && has(s.line, "not a whole number - the split off"), "setup: TOP not a whole number: off (" + s.line + ")");
    s = ds::setup("0.15", nullptr, false, nullptr, 0, 43);
    check(!s.rule.on() && has(s.line, "ignored - the deferral is off"), "setup: no deferral: ignored (" + s.line + ")");
    s = ds::setup("0.15", nullptr, true, "DS4_EP_SPLIT_VERIFY", 0, 43);
    check(!s.rule.on() && has(s.line, "the split off - DS4_EP_SPLIT_VERIFY is on"), "setup: beside SPLIT_VERIFY: off (" + s.line + ")");
    s = ds::setup(nullptr, "1", true, "DS4_EP_POOL_CUT_CHECK", 0, 43);
    check(!s.rule.on() && has(s.line, "DS4_EP_POOL_CUT_CHECK is on"), "setup: beside POOL_CUT_CHECK: off (" + s.line + ")");
    s = ds::setup("0.15", nullptr, true, nullptr, 42, 43);
    check(!s.rule.on() && has(s.line, "no layer to split"), "setup: DEFER_FROM the last layer: no split (" + s.line + ")");
    s = ds::setup("0.15", nullptr, true, nullptr, 0, 43);
    check(s.rule.kind == ds::Kind::kBelow && s.rule.w == 0.15f && has(s.line, "DS4_EP_DEFER_BELOW=0.15: layers 0-41 split") &&
              has(s.line, ">= 0.15 (heavy) added in the layer") && has(s.line, "layer 42's whole") &&
              has(s.line, "the layer waits for both parts: a quality probe, the speed path is not built yet"),
          "setup: BELOW 0.15 on (" + s.line + ")");
    s = ds::setup(nullptr, "2", true, nullptr, 10, 43);
    check(s.rule.kind == ds::Kind::kTop && s.rule.r == 2 && has(s.line, "DS4_EP_DEFER_TOP=2: layers 10-41 split") &&
              has(s.line, "each token's 2 largest-share CPU-lane picks"),
          "setup: TOP 2 from layer 10 on (" + s.line + ")");
}

static void case_kind() {
    const int nl = 43;
    check(ds::defer_kind(false, false, true, 5, 0, nl, true) == 0, "defer_kind: a window not deferred (T > 1, prompts): 0");
    check(ds::defer_kind(true, true, true, 43, 0, nl, true) == 0, "defer_kind: a drafter layer: 0");
    check(ds::defer_kind(true, false, false, 5, 0, nl, true) == 0, "defer_kind: off DS4_MOE_TOKEN's path: 0");
    check(ds::defer_kind(true, false, true, 9, 10, nl, true) == 0 && ds::defer_kind(true, false, true, 10, 10, nl, true) == 2,
          "defer_kind: DS4_EP_DEFER_FROM=10: layer 9 not deferred, 10 split");
    check(ds::defer_kind(true, false, true, nl - 1, 0, nl, true) == 1 && ds::defer_kind(true, false, true, nl - 2, 0, nl, true) == 2,
          "defer_kind: the last layer deferred whole, the one before split");
    check(ds::defer_kind(true, false, true, 5, 0, nl, false) == 1, "defer_kind: no rule: deferred whole (as before)");
}

static void case_heavy() {
    // shares: invalid ids (-1, past n_exp) are 0 and out of the sum
    const int ids[6] = {3, -1, 5, 300, 7, 9};
    const float w[6] = {0.6f, 9.0f, 0.3f, 9.0f, 0.45f, 0.15f};
    float s[8] = {};
    ds::shares(ids, w, 6, 256, s);
    const float sum = s[0] + s[2] + s[4] + s[5];
    check(s[1] == 0.0f && s[3] == 0.0f && std::fabs(sum - 1.0f) < 1e-6f && std::fabs(s[0] - 0.6f / 1.5f) < 1e-7f,
          "shares: the valid picks' weights over their sum, the invalid ones 0");
    const int z_ids[2] = {1, 2};
    const float z_w[2] = {0.0f, 0.0f};
    float zs[2] = {1, 1};
    ds::shares(z_ids, z_w, 2, 256, zs);
    check(zs[0] == 0.0f && zs[1] == 0.0f, "shares: a zero sum: all 0");

    uint8_t lane[8] = {1, 1, 1, 1, 0, 1, 0, 0}, hv[8];
    const float q[8] = {0.25f, 0.25f, 0.125f, 0.5f, 0.9f, 0.25f, 0.0f, 0.0f};
    ds::Rule b;
    b.kind = ds::Kind::kBelow;
    b.w = 0.25f;
    ds::heavy(b, q, lane, 8, hv);
    check(hv[0] && hv[1] && !hv[2] && hv[3] && !hv[4] && hv[5] && !hv[6],
          "heavy BELOW 0.25: shares == 0.25 heavy (>=), 0.125 light, a pick off the lane never");
    b.w = std::nextafter(0.25f, 1.0f);
    ds::heavy(b, q, lane, 8, hv);
    check(!hv[0] && !hv[1] && hv[3], "heavy BELOW just past 0.25: those light now");
    ds::Rule t;
    t.kind = ds::Kind::kTop;
    t.r = 2;
    ds::heavy(t, q, lane, 8, hv);   // (lane shares 0.25 0.25 0.125 0.5 . 0.25: 3 first, then the lowest index of the 0.25s)
    check(hv[3] && hv[0] && !hv[1] && !hv[5] && !hv[2] && !hv[4], "heavy TOP 2: the largest, then a tie to the lower index");
    t.r = 4;
    ds::heavy(t, q, lane, 8, hv);
    check(hv[3] && hv[0] && hv[1] && hv[5] && !hv[2] && !hv[4], "heavy TOP 4: the three 0.25s join, 0.125 out");
    t.r = 9;
    ds::heavy(t, q, lane, 8, hv);
    check(hv[0] && hv[1] && hv[2] && hv[3] && hv[5] && !hv[4] && !hv[6], "heavy TOP 9 (more than the lane's 5): all of them");
    ds::Rule off;
    ds::heavy(off, q, lane, 8, hv);
    bool none = true;
    for (int i = 0; i < 8; ++i) none = none && !hv[i];
    check(none, "heavy, no rule: none");
}

// serve_lane's adds for one split request (T 1, so pick pk is i), as the engine makes them: the lane's own picks over
// sp.here (tagged slice when sent so), the cut W picks' [0, cut) here, then the reply's rows (a whole pick [0, nb), a
// slice one [k, nb), a W one [cut, nb)) or the fallback's (sp's late ranges, W's [cut, nb)), each into its part
struct Req1 {
    std::vector<int> local, held, wp;   // (picks: the lane's own, the server's GPU's, W's)
};
static std::vector<ds::Add> adds_of(const Req1& q, const uint8_t* hv, bool spl, bool out, bool ok, bool reply, int k,
                                    int nb, int cut) {
    std::vector<ds::Add> a;
    rx::SplitPlan sp = rx::split_plan(spl, out, true, k, nb);
    Req1 r = q;
    if (out && !ok) {   // (a send that failed: everything back on the lane, whole)
        r.local.insert(r.local.end(), r.held.begin(), r.held.end());
        r.local.insert(r.local.end(), r.wp.begin(), r.wp.end());
        r.held.clear();
        r.wp.clear();
        cut = 0;
        sp = rx::split_plan(spl, out, false, k, nb);
    }
    if (!out) {   // (not sent: the server's whole picks here, W's never taken out of the lane's)
        r.local.insert(r.local.end(), r.held.begin(), r.held.end());
        r.local.insert(r.local.end(), r.wp.begin(), r.wp.end());
        r.held.clear();
        r.wp.clear();
    }
    auto put = [&](int pk, rx::BlockRange br) {
        if (!br.empty()) a.push_back(ds::Add{pk, hv[pk] ? 0 : 1, br.b0, br.b1});
    };
    for (int pk : r.local) put(pk, sp.here);
    for (int pk : r.wp) put(pk, rx::BlockRange{0, cut});
    if (!sp.sent) return a;
    for (int pk : r.held) put(pk, reply ? rx::server_range(rx::PickLane::kServerWhole, k, nb) : sp.late_whole);
    if (sp.tagged)
        for (int pk : r.local) put(pk, reply ? rx::server_range(rx::PickLane::kSplit, k, nb) : sp.late_slice);
    for (int pk : r.wp) put(pk, rx::BlockRange{cut, nb});
    return a;
}

static void case_ledger() {
    const int nb = 9, k = 6;
    Req1 q;
    q.local = {1, 3};
    q.held = {0};
    q.wp = {2};
    uint8_t lane[ds::kMaxPk] = {}, hv[ds::kMaxPk] = {};
    for (int pk : {0, 1, 2, 3}) lane[pk] = 1;   // (pick 4: the device's, not the lane's)
    hv[0] = hv[2] = 1;                          // (the server's GPU's and the W pick heavy, the lane's own light)
    std::string why;
    for (int reply = 0; reply < 2; ++reply)
        for (int cut : {0, 3}) {
            const auto a = adds_of(q, hv, true, true, true, reply != 0, k, nb, cut);
            check(ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why),
                  std::string("ledger: sent tagged, ") + (reply ? "the reply" : "the fallback") + ", W cut " +
                      std::to_string(cut) + ": every block once, into its part (" + why + ")");
        }
    check(ds::ledger_ok(lane, hv, ds::kMaxPk, adds_of(q, hv, false, true, true, true, k, nb, 0), nb, &why),
          "ledger: sent untagged (the layer not sliced: the lane's own whole here) (" + why + ")");
    check(ds::ledger_ok(lane, hv, ds::kMaxPk, adds_of(q, hv, true, false, false, false, k, nb, 0), nb, &why),
          "ledger: not sent (the gate, an idle link): all whole here (" + why + ")");
    check(ds::ledger_ok(lane, hv, ds::kMaxPk, adds_of(q, hv, true, true, false, false, k, nb, 3), nb, &why),
          "ledger: the send failed: all whole here (" + why + ")");
    check(ds::ledger_ok(lane, hv, ds::kMaxPk, adds_of(q, hv, false, false, false, false, 1, 1, 0), 1, &why),
          "ledger: no H' (one block a pick) (" + why + ")");
    // the faults it must see
    auto base = adds_of(q, hv, true, true, true, true, k, nb, 3);
    auto a = base;
    a.push_back(a.front());
    check(!ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why) && has(why, "twice"), "ledger: an add twice: seen (" + why + ")");
    a = base;
    a.pop_back();
    check(!ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why) && has(why, "not added"), "ledger: an add missing: seen (" + why + ")");
    a = base;
    a[0].part ^= 1;
    check(!ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why) && has(why, "part"), "ledger: an add in the other part: seen (" + why + ")");
    a = base;
    a.push_back(ds::Add{4, 1, 0, nb});
    check(!ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why) && has(why, "not a CPU-lane pick"),
          "ledger: a pick off the lane added: seen (" + why + ")");
    a = base;
    a.push_back(ds::Add{-1, 1, 0, nb});
    check(!ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why), "ledger: a row's expert not found (pick -1): seen (" + why + ")");
    a = base;
    a.back().b1 = nb + 1;
    check(!ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why) && has(why, "past"), "ledger: a block past nb: seen (" + why + ")");
    // the engine's slice range off by one (the server [k - 1, nb) with this lane's [0, k)): twice
    a = adds_of(q, hv, true, true, true, true, k, nb, 0);
    for (ds::Add& x : a)
        if (x.b0 == k) x.b0 = k - 1;
    check(!ds::ledger_ok(lane, hv, ds::kMaxPk, a, nb, &why), "ledger: the server's slice from k - 1: seen (" + why + ")");
}

static void case_tally() {
    ds::Tally t, t0;
    // one request: six CPU-lane picks of known shares; 2 the server's GPU's, 2 slice, 1 W, 1 here whole (nb 9, k 6, cut 3)
    ds::LanePick p[6];
    const float sh[6] = {0.4f, 0.25f, 0.15f, 0.1f, 0.06f, 0.04f};
    const int kind[6] = {ds::kHeld, ds::kSlice, ds::kSlice, ds::kW, ds::kHeld, ds::kHere};
    for (int i = 0; i < 6; ++i) {
        p[i].i = i;
        p[i].s = sh[i];
        p[i].kind = kind[i];
        p[i].bh = kind[i] == ds::kHeld ? 0 : kind[i] == ds::kSlice ? 6 : kind[i] == ds::kW ? 3 : 9;
        p[i].bs = 9 - p[i].bh;
        p[i].e = kind[i] == ds::kHeld ? 0.0 : (double) (i + 1);
    }
    ds::tally(t, p, 6);
    ds::tally(t, p, 0);   // (a request with no CPU-lane pick: counted, nothing heavy)
    check(t.reqs == 2 && t.picks == 6 && t.by_kind[ds::kHeld] == 2 && t.by_kind[ds::kSlice] == 2, "tally: requests, picks, kinds");
    // w >= 0.05: picks 0-4 (5); 0.10: 0-3; 0.15: 0-2; 0.20: 0-1; top 1: 0; top 2: 0, 1; top 3: 0-2
    const int want_hp[7] = {5, 4, 3, 2, 1, 2, 3};
    bool ok = true;
    for (int r = 0; r < 7; ++r) ok = ok && t.hp[r] == want_hp[r];
    check(ok, "tally: each rule's heavy picks (5 4 3 2 | 1 2 3)");
    check(std::fabs(t.hw[3] - 0.65) < 1e-6 && std::fabs(t.w - 1.0) < 1e-6, "tally: w >= 0.20's weight 0.65 of 1.0");
    // energy: picks 1 (2), 2 (3), 3 (4), 5 (6) - the held ones none; w >= 0.15: picks 0-2 -> 2 + 3 of 15
    check(std::fabs(t.e - 15.0) < 1e-9 && std::fabs(t.he[2] - 5.0) < 1e-9, "tally: w >= 0.15's energy 5 of 15");
    // blocks: here 0 6 6 3 0 9 = 24, server 9 3 3 6 9 0 = 30; top 1 (pick 0, held): here 0, server 9
    check(t.bh == 24 && t.bs == 30 && t.hbh[4] == 0 && t.hbs[4] == 9, "tally: blocks here 24, server 30; top 1's 0 / 9");
    // requests with a heavy server pick: every rule's heavy set holds pick 0 (held) in the first request, none the second
    ok = true;
    for (int r = 0; r < 7; ++r) ok = ok && t.hreq[r] == 1;
    check(ok, "tally: requests with a heavy server pick: 1 of 2 for every rule");
    const double p50 = ds::share_q(t, t0, 0.5), p10 = ds::share_q(t, t0, 0.1), p90 = ds::share_q(t, t0, 0.9);
    check(p10 <= 0.0425 && p10 >= 0.04 && std::fabs(p50 - 0.1025) < 1e-9 && p90 >= 0.4 && p90 <= 0.4025,
          "tally: shares p10 / p50 / p90 at their bins' upper edges (" + std::to_string(p10) + " " + std::to_string(p50) +
              " " + std::to_string(p90) + ")");
    const std::string ln = ds::tally_lines(t, t0);
    check(has(ln, "ep defer stat: 2 requests") && has(ln, "3.00 CPU-lane picks a request") &&
              has(ln, "ep defer stat w>=0.20: heavy picks 33.3% (1.00 a request), weight 65.0%") &&
              has(ln, "ep defer stat top 1: heavy picks 16.7%") && has(ln, "requests with a heavy server pick 50.0%"),
          "tally_lines: the lines");
    check(ds::tally_lines(t, t).empty(), "tally_lines: nothing since the snapshot: no lines");
    std::printf("%s", ln.c_str());
}

int main() {
    case_parse();
    case_setup();
    case_kind();
    case_heavy();
    case_ledger();
    case_tally();
    if (checks_bad == 0) std::printf("ALL OK\n");
    else std::printf("%d FAILED\n", checks_bad);
    return checks_bad == 0 ? 0 : 1;
}
