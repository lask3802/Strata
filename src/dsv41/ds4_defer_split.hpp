// src/dsv41/ds4_defer_split.hpp - heavy-first deferral (DS4_EP_DEFER_BELOW=w / DS4_EP_DEFER_TOP=r, on DS4_EP_DEFER=1):
// a deferred layer's CPU-lane part in two.  The heavy picks' outputs (a share of the token's route weights >= w, or the
// token's r largest-share CPU-lane picks) go into the mailbox slot's part[0], which the layer adds through the experts'
// sum as it would without the deferral; the light ones' into part[1], which the next layer adds after its attention as
// DS4_EP_DEFER adds the whole part today.  Here: what a value says, which picks are heavy, the check that every pick's
// n_ff blocks went into exactly one part once (DS4_EP_DEFER_BELOW_CHECK=1), and DS4_EP_DEFER_STAT's tally of what
// each rule would make heavy.  Pure functions, no state, no CUDA - serve_lane's and defer_split_test's, so the wiring
// is the one tested.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::ds4::dsplit {

constexpr int kTokPicks = 8;                // a token's routed picks at most (ds4_kernels.cuh's kPicks)
constexpr int kMaxPk = 16 * kTokPicks;      // a window's picks, t * kTokPicks + i (kWin x kPicks)
constexpr int kMaxBlocks = 32;              // the n_ff blocks a pick has at most (rx_nb: n_ff / 256)

enum class Kind : int { kOff = 0, kBelow = 1, kTop = 2 };
struct Rule {
    Kind kind = Kind::kOff;
    float w = 0.0f;   // kBelow: a CPU-lane pick is heavy when its share is >= w
    int r = 0;        // kTop: the token's r largest-share CPU-lane picks are heavy
    bool on() const { return kind != Kind::kOff; }
};

/// what a DS4_EP_DEFER_BELOW / DS4_EP_DEFER_TOP text says: unset (or empty); a rule; not one (a text that is not a number,
/// NaN, BELOW <= 0, TOP not a whole number); a rule that makes no pick heavy (BELOW >= 1, TOP < 1)
enum class Val : int { kUnset, kOn, kBad, kNone };
inline Val parse_below(const char* v, float& w) {
    if (v == nullptr || *v == '\0') return Val::kUnset;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    if (end == v) return Val::kBad;
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0' || std::isnan(d) || d <= 0.0) return Val::kBad;
    if (d >= 1.0) return Val::kNone;
    w = (float) d;
    return Val::kOn;
}
inline Val parse_top(const char* v, int& r) {
    if (v == nullptr || *v == '\0') return Val::kUnset;
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v) return Val::kBad;
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return Val::kBad;
    if (x < 1) return Val::kNone;
    r = x > 1000 ? 1000 : (int) x;   // (more than a token's picks: all of them)
    return Val::kOn;
}

/// the switches together: the rule and the line that says so (no "ds4: " prefix; empty: nothing to say).  below, top:
/// the texts; deferred: DS4_EP_DEFER's deferral runs (or its VERIFY's); conflict: a check of the run's own that the split
/// would change (null: none); the deferred layers defer_from .. nl - 1, the last of them never split (its add before the
/// head is exact); wait - what a split layer waits for: kWaitHeavy the heavy part alone (hseq: the speed path),
/// kWaitBoth both (DS4_EP_DEFER_BELOW_WAIT=1, the quality probe: the speed path's numbers), kWaitVerify both
/// (DS4_EP_DEFER_VERIFY: decode exact)
enum Wait : int { kWaitHeavy = 0, kWaitBoth = 1, kWaitVerify = 2 };
struct Setup {
    Rule rule;
    std::string line;
};
inline Setup setup(const char* below, const char* top, bool deferred, const char* conflict, int defer_from, int nl,
                   int wait = kWaitHeavy) {
    Setup s;
    float w = 0.0f;
    int r = 0;
    const Val vb = parse_below(below, w), vt = parse_top(top, r);
    if (vb == Val::kUnset && vt == Val::kUnset) return s;
    const bool is_b = vb != Val::kUnset;
    char b[640];
    if (is_b && vt != Val::kUnset) {
        std::snprintf(b, sizeof b, "DS4_EP_DEFER_BELOW=%.32s and DS4_EP_DEFER_TOP=%.32s both set: the split off (one rule "
                                   "at a time; every CPU-lane part joins late, as DS4_EP_DEFER=1 has it)", below, top);
        s.line = b;
        return s;
    }
    const char* nm = is_b ? "DS4_EP_DEFER_BELOW" : "DS4_EP_DEFER_TOP";
    const char* tx = is_b ? below : top;
    const Val v = is_b ? vb : vt;
    if (v == Val::kBad) {
        std::snprintf(b, sizeof b, "%s=%.32s: %s - the split off", nm, tx,
                      is_b ? "not a share of the route weights in (0, 1)" : "not a whole number");
        s.line = b;
        return s;
    }
    if (v == Val::kNone) {
        std::snprintf(b, sizeof b, "%s=%.32s: %s, no CPU-lane pick is heavy - no split (every CPU-lane part joins late, "
                                   "as DS4_EP_DEFER=1 has it)", nm, tx, is_b ? ">= 1" : "< 1");
        s.line = b;
        return s;
    }
    if (!deferred) {
        std::snprintf(b, sizeof b, "%s=%.32s: ignored - the deferral is off (DS4_EP_DEFER=1 not set, or off for this run)",
                      nm, tx);
        s.line = b;
        return s;
    }
    if (conflict != nullptr) {
        std::snprintf(b, sizeof b, "%s=%.32s: the split off - %s is on (its check would see the split's rows)", nm, tx,
                      conflict);
        s.line = b;
        return s;
    }
    if (defer_from >= nl - 1) {
        std::snprintf(b, sizeof b, "%s=%.32s: no layer to split (the deferral takes the last layer alone, whose add "
                                   "before the head is exact) - no split", nm, tx);
        s.line = b;
        return s;
    }
    char rule[160];
    if (is_b) {
        s.rule.kind = Kind::kBelow;
        s.rule.w = w;
        std::snprintf(rule, sizeof rule, "the CPU-lane picks with a share of the token's route weights >= %g", (double) w);
    } else {
        s.rule.kind = Kind::kTop;
        s.rule.r = r;
        std::snprintf(rule, sizeof rule, "each token's %d largest-share CPU-lane picks", r);
    }
    const char* wt = wait == kWaitBoth   ? "DS4_EP_DEFER_BELOW_WAIT=1: the layer waits for both parts - a quality probe, the "
                                           "speed path's numbers at about DS4_EP_DEFER=0's speed"
                     : wait == kWaitVerify ? "the layer waits for both parts (DS4_EP_DEFER_VERIFY: decode exact)"
                                           : "the layer waits for the heavy part alone (hseq: the speed path)";
    std::snprintf(b, sizeof b, "%s=%.32s: layers %d-%d split their CPU-lane part - %s (heavy) added in the layer, the rest "
                               "a layer late (layer %d's whole, before the head); %s", nm, tx, defer_from, nl - 2, rule,
                  nl - 1, wt);
    s.line = b;
    return s;
}

/// a seq's layer in the deferral (plan_window's defer_of, from run_layer's `late` terms): 0 not deferred, 1 deferred
/// whole (DS4_EP_DEFER as before), 2 split.  win: the window is deferred (wp.defer - DS4_EP_DEFER, an EP window of one row
/// on the async path); mtp: a drafter layer; token_path: DS4_MOE_TOKEN's path; split: a rule is on.  Layers defer_from
/// .. nl - 1 are deferred, the last of them never split
inline int defer_kind(bool win, bool mtp, bool token_path, int il, int defer_from, int nl, bool split) {
    if (!win || mtp || !token_path || il < defer_from) return 0;
    return split && il != nl - 1 ? 2 : 1;
}

/// the shares of a token's k picks: w[i] / the sum of the weights of its valid picks (an expert id in [0, n_exp)),
/// summed in pick order (DISK_SKIP_W's sum); an invalid pick's share, or every one when the sum is not > 0: 0
inline void shares(const int* ids, const float* w, int k, int n_exp, float* s) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i)
        if (ids[i] >= 0 && ids[i] < n_exp) sum += w[i];
    for (int i = 0; i < k; ++i) s[i] = ids[i] >= 0 && ids[i] < n_exp && sum > 0.0f ? w[i] / sum : 0.0f;
}
/// the heavy ones of a token's CPU-lane picks (lane[i] != 0, k picks; the others 0): kBelow - share >= w; kTop - the r
/// largest shares (ties: the lower pick index first; fewer than r CPU-lane picks: all of them); kOff - none
inline void heavy(const Rule& ru, const float* s, const uint8_t* lane, int k, uint8_t* hv) {
    for (int i = 0; i < k; ++i) {
        hv[i] = 0;
        if (!lane[i]) continue;
        if (ru.kind == Kind::kBelow) {
            hv[i] = s[i] >= ru.w;
        } else if (ru.kind == Kind::kTop) {
            int rank = 0;
            for (int j = 0; j < k; ++j) rank += lane[j] && j != i && (s[j] > s[i] || (s[j] == s[i] && j < i));
            hv[i] = rank < ru.r;
        }
    }
}

/// one add into a part: pick pk's n_ff blocks [b0, b1) into part row `part` (0 heavy, 1 light)
struct Add {
    int pk = 0, part = 0, b0 = 0, b1 = 0;
};
/// DS4_EP_DEFER_BELOW_CHECK=1: every CPU-lane pick (lane[pk], npk picks) has each of its nb blocks added exactly once,
/// all into part hv[pk] ? 0 : 1, and nothing else is added (no other pick, no block past nb).  why: the first fault
inline bool ledger_ok(const uint8_t* lane, const uint8_t* hv, int npk, const std::vector<Add>& adds, int nb,
                      std::string* why = nullptr) {
    char b[160];
    auto bad = [&](const char* f, int pk) {
        if (why != nullptr) {
            std::snprintf(b, sizeof b, "pick %d: %s", pk, f);
            *why = b;
        }
        return false;
    };
    if (nb < 1 || nb > kMaxBlocks || npk > kMaxPk) return bad("past the check's sizes", -1);
    uint32_t got[kMaxPk] = {};
    for (const Add& a : adds) {
        if (a.pk < 0 || a.pk >= npk || !lane[a.pk]) return bad("added, not a CPU-lane pick", a.pk);
        if (a.b0 < 0 || a.b1 > nb || a.b0 > a.b1) return bad("blocks past the pick's", a.pk);
        if (a.part != (hv[a.pk] ? 0 : 1)) return bad(hv[a.pk] ? "heavy, added to the light part" : "light, added to the heavy part", a.pk);
        for (int q = a.b0; q < a.b1; ++q) {
            if ((got[a.pk] >> q) & 1u) return bad("a block added twice", a.pk);
            got[a.pk] |= 1u << q;
        }
    }
    const uint32_t all = nb == 32 ? ~0u : (1u << nb) - 1u;
    for (int pk = 0; pk < npk; ++pk)
        if (lane[pk] && got[pk] != all) return bad("a block not added", pk);
    return true;
}

/// DS4_EP_DEFER_STAT=1: the rules counted side by side - shares >= 0.05, 0.10, 0.15, 0.20, then the top 1, 2, 3
constexpr int kRules = 7;
inline Rule stat_rule(int q) {
    static const float kW[4] = {0.05f, 0.10f, 0.15f, 0.20f};
    Rule r;
    if (q < 4) {
        r.kind = Kind::kBelow;
        r.w = kW[q];
    } else {
        r.kind = Kind::kTop;
        r.r = q - 3;
    }
    return r;
}
/// a CPU-lane pick of a counted request: its index in the token, share, where it ran (kHere: whole on this lane;
/// kSlice: this lane [0, k), the server's CPU lane [k, nb); kHeld: the server's GPU, whole; kW: the server's W, this
/// lane [0, cut)), its local slot's energy |w y|^2 (0: none here) and its blocks here and on the server
enum PickKind : int { kHere = 0, kSlice = 1, kHeld = 2, kW = 3 };
struct LanePick {
    int i = 0, kind = kHere, bh = 0, bs = 0;
    float s = 0.0f;
    double e = 0.0;
};
constexpr int kBins = 400;   // (the shares' distribution: bins of 0.0025)
struct Tally {
    int64_t reqs = 0, picks = 0, by_kind[4] = {};
    int64_t bins[kBins] = {};
    double w = 0, e = 0, bh = 0, bs = 0;   // all the CPU-lane picks': shares, energy, blocks here, blocks on the server
    // each rule's heavy picks, the requests with a heavy pick the server computes (blocks there), and the heavy picks'
    // shares, energy, blocks here and blocks on the server
    int64_t hp[kRules] = {}, hreq[kRules] = {};
    double hw[kRules] = {}, he[kRules] = {}, hbh[kRules] = {}, hbs[kRules] = {};
};
/// one request (a token: decode, T 1) and its n CPU-lane picks into t
inline void tally(Tally& t, const LanePick* p, int n) {
    ++t.reqs;
    float s[kTokPicks] = {};
    uint8_t lane[kTokPicks] = {}, hv[kTokPicks] = {};
    int at[kTokPicks];
    for (int j = 0; j < kTokPicks; ++j) at[j] = -1;
    for (int a = 0; a < n; ++a) {
        const LanePick& q = p[a];
        ++t.picks;
        if (q.kind >= 0 && q.kind < 4) ++t.by_kind[q.kind];
        const int bi = q.s >= 1.0f ? kBins - 1 : q.s <= 0.0f ? 0 : (int) (q.s * (float) kBins);
        ++t.bins[bi < kBins ? bi : kBins - 1];
        t.w += q.s;
        t.e += q.e;
        t.bh += q.bh;
        t.bs += q.bs;
        if (q.i >= 0 && q.i < kTokPicks) {
            s[q.i] = q.s;
            lane[q.i] = 1;
            at[q.i] = a;
        }
    }
    for (int r = 0; r < kRules; ++r) {
        heavy(stat_rule(r), s, lane, kTokPicks, hv);
        bool srv = false;
        for (int i = 0; i < kTokPicks; ++i) {
            if (!hv[i] || at[i] < 0) continue;
            const LanePick& q = p[at[i]];
            ++t.hp[r];
            t.hw[r] += q.s;
            t.he[r] += q.e;
            t.hbh[r] += q.bh;
            t.hbs[r] += q.bs;
            srv = srv || q.bs > 0;
        }
        t.hreq[r] += srv;
    }
}
/// the shares' quantile f (0..1) since b0: the upper edge of the bin it falls in
inline double share_q(const Tally& a, const Tally& b0, double f) {
    int64_t n = 0;
    for (int i = 0; i < kBins; ++i) n += a.bins[i] - b0.bins[i];
    if (n <= 0) return 0.0;
    const double want = f * (double) n;
    int64_t c = 0;
    for (int i = 0; i < kBins; ++i) {
        c += a.bins[i] - b0.bins[i];
        if ((double) c >= want && c > 0) return (double) (i + 1) / kBins;
    }
    return 1.0;
}
/// the stat lines since b0 ("" when no request was counted), each "ep defer stat ...\n"
inline std::string tally_lines(const Tally& a, const Tally& b0) {
    const double nr = (double) (a.reqs - b0.reqs);
    if (nr <= 0) return "";
    auto pc = [](double x, double all) { return all > 0 ? 100.0 * x / all : 0.0; };
    const double np = (double) (a.picks - b0.picks);
    char b[720];
    std::string out;
    std::snprintf(b, sizeof b, "ep defer stat: %.0f requests (decode, the deferred layers but the last), %.2f CPU-lane "
                               "picks a request (the server's GPU %.2f, slice %.2f, W %.2f, here whole %.2f); their share of "
                               "the token's route weights p10 %.4f p50 %.4f p90 %.4f (bins of %.4f)\n",
                  nr, np / nr, (double) (a.by_kind[kHeld] - b0.by_kind[kHeld]) / nr,
                  (double) (a.by_kind[kSlice] - b0.by_kind[kSlice]) / nr, (double) (a.by_kind[kW] - b0.by_kind[kW]) / nr,
                  (double) (a.by_kind[kHere] - b0.by_kind[kHere]) / nr, share_q(a, b0, 0.1), share_q(a, b0, 0.5),
                  share_q(a, b0, 0.9), 1.0 / kBins);
    out += b;
    const double w = a.w - b0.w, e = a.e - b0.e, bh = a.bh - b0.bh, bs = a.bs - b0.bs;
    for (int r = 0; r < kRules; ++r) {
        char nm[32];
        const Rule ru = stat_rule(r);
        if (ru.kind == Kind::kBelow) std::snprintf(nm, sizeof nm, "w>=%.2f", (double) ru.w);
        else std::snprintf(nm, sizeof nm, "top %d", ru.r);
        const double hp = (double) (a.hp[r] - b0.hp[r]);
        std::snprintf(b, sizeof b, "ep defer stat %s: heavy picks %.1f%% (%.2f a request), weight %.1f%%, energy %.1f%% "
                                   "(|w y|^2 of the local slots), blocks here %.1f%% / server %.1f%%, requests with a heavy "
                                   "server pick %.1f%%\n",
                      nm, pc(hp, np), hp / nr, pc(a.hw[r] - b0.hw[r], w), pc(a.he[r] - b0.he[r], e),
                      pc(a.hbh[r] - b0.hbh[r], bh), pc(a.hbs[r] - b0.hbs[r], bs), pc((double) (a.hreq[r] - b0.hreq[r]), nr));
        out += b;
    }
    return out;
}

}  // namespace strata::ds4::dsplit
