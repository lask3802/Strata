// src/dsv41/ds4_defer_rows.hpp - a request's rows to the expert server and the adds a split request makes
// (DS4_EP_DEFER_BELOW / TOP, ds4_defer_split.hpp): rx_send's rows (a row a token; a split request's a row a (token,
// part), the heavy ones first), the adds of the lane's own slots and of a reply's rows (DS4_EP_DEFER_BELOW_CHECK=1's
// ledger), and the fallback's picks by part (a reply late or lost).  Pure functions, no state - serve_lane's, rx_send's,
// rx_finish's and defer_split_test's, so the rows and the fallback the engine makes are the ones tested.
#pragma once

#include "ds4_defer_split.hpp"
#include "ds4_rx.hpp"
#include "ds4_rx_split.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace strata::ds4::dsplit {

using Picks = std::vector<std::pair<int, int>>;   // (expert, t * kTokPicks + i)

/// rx_send's rows of a request of T tokens (w: the route weights, [t * kTokPicks + i]): rpk the server's whole picks,
/// spk the slice ones (tagged slice), wpk W's (tagged slice and pool) - each pick into its token's row, in that order, a
/// token's row made by its first pick.  hv (a split request's heavy picks; null: none) - a row a (token, part): first
/// every heavy pick's rows, then every light one's.  dst: each row's part row (its token; a split request's: 0 heavy, 1
/// light).  Picks of tokens past T are left out.  The tokens the rows hold, a bit a token
inline unsigned req_rows(int T, const float* w, const Picks& rpk, const Picks& spk, const Picks& wpk, const uint8_t* hv,
                         std::vector<rx::ReqRow>& rows, std::vector<uint8_t>& dst) {
    rows.clear();
    dst.clear();
    int row_of[2 * rx::kMaxRows];
    for (int& r : row_of) r = -1;
    unsigned toks = 0;
    for (int hp = 0; hp < (hv != nullptr ? 2 : 1); ++hp)
        for (int pass = 0; pass < 3; ++pass)
            for (const auto& q : pass == 0 ? rpk : pass == 1 ? spk : wpk) {
                const int t = q.second / kTokPicks, i = q.second % kTokPicks;
                if (t >= T || t >= rx::kMaxRows) continue;
                if (hv != nullptr && (hv[q.second] != 0) != (hp == 0)) continue;
                const int key = hv != nullptr ? 2 * t + hp : t;
                if (row_of[key] < 0) {
                    row_of[key] = (int) rows.size();
                    rows.emplace_back();
                    rows.back().t = (uint8_t) t;
                    dst.push_back((uint8_t) (hv != nullptr ? hp : t));
                    toks |= 1u << t;
                }
                rx::ReqRow& r = rows[(size_t) row_of[key]];
                if (pass >= 1) r.slice |= (uint16_t) (1u << r.n);
                if (pass == 2) r.slice |= (uint16_t) (1u << (rx::kPoolBit + r.n));
                r.e[r.n] = (int16_t) q.first;
                r.w[r.n] = w[t * kTokPicks + i];
                ++r.n;
            }
    return toks;
}

/// (DS4_EP_DEFER_BELOW_CHECK=1) a reply's rows as adds: each row's picks into its part row (dst), over the server's
/// share - a whole pick [0, nb), a slice one [k, nb), a W one [cut, nb).  A row's expert back to its pick by the token's
/// ids (ids[t * kTokPicks + i], k_exp a token; a token's picks are distinct experts); none: pick -1, which the check
/// refuses
inline void row_adds(const std::vector<rx::ReqRow>& rows, const std::vector<uint8_t>& dst, const int* ids, int k_exp,
                     int k, int cut, int nb, std::vector<Add>& adds) {
    for (size_t j = 0; j < rows.size(); ++j) {
        const rx::ReqRow& r = rows[j];
        for (int n = 0; n < r.n; ++n) {
            int pk = -1;
            for (int i = 0; i < k_exp && i < kTokPicks; ++i)
                if (ids[r.t * kTokPicks + i] == r.e[n]) pk = r.t * kTokPicks + i;
            const bool sl = (r.slice >> n) & 1u, pool = (r.slice >> (rx::kPoolBit + n)) & 1u;
            adds.push_back(Add{pk, j < dst.size() ? (int) dst[j] : -1, !sl ? 0 : pool ? cut : k, nb});
        }
    }
}

/// (DS4_EP_DEFER_BELOW_CHECK=1) the lane's own slots of a split request as adds: slot sl's pick pk_of[sl] into its part,
/// over mr (the lane's range for its own picks: sp.here) for the first nmain slots, [0, wcut) for the cut W picks' after
inline void local_adds(const int* pk_of, int nslots, int nmain, const uint8_t* hv, rx::BlockRange mr, int wcut,
                       std::vector<Add>& adds) {
    for (int sl = 0; sl < nslots; ++sl) {
        const rx::BlockRange br = sl < nmain ? mr : rx::BlockRange{0, wcut};
        if (!br.empty()) adds.push_back(Add{pk_of[sl], hv[pk_of[sl]] ? 0 : 1, br.b0, br.b1});
    }
}

/// A split request's fallback (its reply late or lost: the lane adds exactly the server's share), a part's: the
/// server's whole picks over sp.late_whole, the slice ones over sp.late_slice, W's over [cut, nb) - rx_finish computes
/// part 0's (the heavy picks'), publishes hseq, then part 1's
struct FallPart {
    Picks pk[3];
    rx::BlockRange br[3];
};
inline void fallback_parts(const Picks& rpk, const Picks& spk, const Picks& wpk, const uint8_t* hv, const rx::SplitPlan& sp,
                           int cut, int nb, FallPart out[2]) {
    const Picks* src[3] = {&rpk, &spk, &wpk};
    const rx::BlockRange br[3] = {sp.late_whole, sp.late_slice, rx::BlockRange{cut, nb}};
    for (int p = 0; p < 2; ++p)
        for (int c = 0; c < 3; ++c) {
            out[p].pk[c].clear();
            out[p].br[c] = br[c];
        }
    for (int c = 0; c < 3; ++c)
        for (const auto& q : *src[c]) out[hv[q.second] ? 0 : 1].pk[c].push_back(q);
}
/// (DS4_EP_DEFER_BELOW_CHECK=1) part p's fallback as adds (T 1: pick t * kTokPicks + i is its index)
inline void fallback_adds(const FallPart& f, int p, std::vector<Add>& adds) {
    for (int c = 0; c < 3; ++c)
        for (const auto& q : f.pk[c])
            if (!f.br[c].empty()) adds.push_back(Add{q.second, p, f.br[c].b0, f.br[c].b1});
}

}  // namespace strata::ds4::dsplit
