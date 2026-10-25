// src/dsv41/ds4_rx_lane.hpp - the expert server's CPU lane (expert parallelism H', ep-design.md 1.2): for a request's
// slice picks, the n_ff slice [k, nb) of each expert, read in place from the parts pack (ds4_epparts.hpp: the stored
// [a, nb) of a part file, any k >= a), while the main process's own CPU lane computes [0, k) of the same picks.  The
// arithmetic is the main's CPU lane's for the i-quant experts (Impl::cpu_experts_iq with one node): x quantized once a
// row to the gate / up activation, the slice's gate / up rows (the clamped swiglu into its h), the slice's h quantized
// alone (the same blocks as the whole h's [256 k, ..)), the down rows' values [256 k, ..) at the stored part's stride;
// then each row's weighted sum over its picks in their order.  So a pick's two halves sum to the expert computed whole
// up to the float order (ep_slice_test, rx_lane_test).  No GPU.
#pragma once

#include "ds4_epparts.hpp"
#include "ds4_gguf_cpu.hpp"
#include "ds4_pool.hpp"
#include "ds4_rx.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4::rx {

constexpr int kLaneMaxRows = 16;    // a request's rows at most (ds4_rx.hpp kMaxRows)
constexpr int kLaneMaxPicks = 8;    // picks a row at most (kMaxPicks)
constexpr int kLaneMaxSets = 2;     // ranges a run_sets at most (the hybrid: the slice picks' [k, nb), W's [0, k))

/// One slice pick of a request: its row (the window's token), the expert, the route weight.
struct SlicePick {
    int t = 0, e = 0;
    float w = 0.0f;
};

/// One range of a run_sets: np picks over the n_ff blocks [k0, k1) of `from`'s parts
struct LaneSet {
    int k0 = 0, k1 = 0;
    const ep::Parts* from = nullptr;
    const SlicePick* picks = nullptr;
    int np = 0;
};

class SliceLane {
public:
    struct Types {   // a layer's experts: the gate / up and the down ggml types (0: the layer is not served)
        int tg = 0, td = 0;
    };

    SliceLane() = default;
    SliceLane(const SliceLane&) = delete;
    SliceLane& operator=(const SliceLane&) = delete;

    /// `parts` open (kept, not owned; it outlives the lane); `types` a layer each; the geometry; `limit` the swiglu's
    /// clamp (0: none); `threads` workers on `cpus` (empty: not pinned), each spinning `spin_us` between batches.  False
    /// (err): a served layer the parts do not cover, a part of another type or layout than the layer's, a format this
    /// build has no CPU lane for, parts of another geometry.  Every served expert's three matrices are looked at here.
    bool init(const ep::Parts& parts, const std::vector<Types>& types, int n_embd, int n_ff, float limit, int threads,
              const std::vector<int>& cpus, int spin_us, std::string& err);
    /// the lowest k the stored parts take for every served layer's every expert (blocks(): none - no layer served)
    int lo() const { return lo_; }
    int blocks() const { return nb_; }

    /// y[t * ldy + c] += the sum over row t's picks, in their order, of w x (the expert's n_ff slice [k, nb))(x[t]),
    /// for nr rows (<= kLaneMaxRows) of x (n_embd f32 values, ldx apart) and np picks (<= kLaneMaxPicks a row).  help: the
    /// caller works too (otherwise it only waits: the expert server's link thread stays off the lane's DRAM work).
    /// False (err), y untouched: a layer not served, k outside [lo, nb), a pick out of range, a row's expert twice.
    bool run(int layer, int k, const SlicePick* picks, int np, int nr, const float* x, size_t ldx, float* y, size_t ldy,
             bool help, std::string& err);
    /// the same over the n_ff blocks [k0, k1) (0 <= k0 < k1 <= nb) of `from`'s parts: each pick's three matrices in a part
    /// that holds k0 and runs to k1 or past it, of the layer's types and layout (the hybrid, U5c: a W pick's [0, k) from
    /// the pool, beside run's [k, nb) - the two sum to it whole).  False (err), y untouched, as run's; also a part missing
    bool run_range(int layer, int k0, int k1, const ep::Parts& from, const SlicePick* picks, int np, int nr, const float* x,
                   size_t ldx, float* y, size_t ldy, bool help, std::string& err);
    /// run_range over ns (<= kLaneMaxSets) sets in one pass - the four steps (gate / up, h quantized, down, the sums)
    /// once for all of them, so one wait on the pool a step, not one a set; y += the sets' sums in set order, each set's
    /// picks in their order (the same bits as run_range set by set into one y).  False (err), y untouched, as run_range's
    bool run_sets(int layer, const LaneSet* sets, int ns, int nr, const float* x, size_t ldx, float* y, size_t ldy, bool help,
                  std::string& err);
    /// whether run_range would take expert e of the layer over [k0, k1) from `from` (the layer served, its parts there)
    bool takes(const ep::Parts& from, int layer, int e, int k0, int k1) const {
        ep::View v[3];
        return layer >= 0 && layer < (int) fmt_.size() && fmt_[(size_t) layer].on && views(from, layer, e, k0, k1, v);
    }

    // (read on the thread that runs: not atomic)
    uint64_t bytes() const { return bytes_; }   // weight bytes the runs read
    double busy_ms() const { return busy_ms_; }   // their time, from the first quantization to the last sum
    int64_t runs() const { return runs_; }

private:
    bool init_(const ep::Parts& parts, const std::vector<Types>& types, int n_embd, int n_ff, float limit, int threads,
               const std::vector<int>& cpus, int spin_us, std::string& err);
    // a served layer's expert e: its three matrices' views from k0 in parts that run to k1 or past it, of the layer's
    // types and layout
    bool views(const ep::Parts& from, int layer, int e, int k0, int k1, ep::View* v) const;
    struct Fmt {
        bool on = false;
        int cap = kLaneMaxRows;   // rows an entry (a call of the rows) at most
        Types ty;
        gguf::IqCpu fm;
    };
    struct Ent {
        int e = 0, n = 0, slot0 = 0, set = 0;
        int tok[kLaneMaxRows] = {};
        ep::View v[3];
    };
    const ep::Parts* parts_ = nullptr;
    std::vector<Fmt> fmt_;
    int dim_ = 0, ff_ = 0, nb_ = 0, lo_ = 0;
    float lim_ = 0.0f;
    std::unique_ptr<Pool> pool_;
    // a run's buffers, sized for the most it can take: the rows' activations, then a slot (pick) each its h, its h
    // quantized and its slice's output
    std::vector<uint8_t> aq_, hq_;
    std::vector<float> h_, yd_;
    std::vector<Ent> ents_;
    uint64_t bytes_ = 0;
    double busy_ms_ = 0.0;
    int64_t runs_ = 0;
};

/// Either expert server's slice picks (ds4_rx.hpp, H'): the parts pack (--expert-server-parts DIR) opened once a process,
/// checked at each main process's Geo, then each request's slice picks through a SliceLane whose workers run off the link
/// core - one a physical core (11 on a 12-core 5900X); DS4_RX_LANE_THREADS=N: N of them, past the physical cores every
/// logical CPU off the link core (the A/B of SMT).  The link thread does not work in the lane: it only waits.
class SliceServer {
public:
    /// the parts at `dir`; `link`: the link core (rx::link_cpu(), -1: none) the lane stays off.  False (err): parts it
    /// cannot open, none covered, or an expert's matrix with no part to the last n_ff block.  DS4_RX_PARTS_WARM=1: every
    /// page of the parts read once here (the first tokens' slices not from the disk)
    bool open(const std::string& dir, int link, std::string& err);
    bool on() const { return parts_.is_open(); }
    int lo() const { return on() ? lo_ : 0; }   // (the hello's: the lowest k every covered layer's parts take; 0: none)
    int threads() const { return on() ? threads_ : 0; }
    const std::vector<int>& cpus() const { return cpus_; }
    /// the parts in a line: the pack, the blocks held, the bytes and how many of them are in the page cache (Linux)
    std::string note() const;
    /// a main process's Geo: "" or why these parts do not serve it - another pack (pack_hash), a served layer (K[0] != 0)
    /// they do not cover, gate and up of two types, a part of another type or layout than the layer's (SliceLane::init).
    /// The lane set up for it ("" without parts: nothing to check).  Only a main that splits (g.k 1 .. nb - 1) is
    /// refused: one that does not (DS4_EP_K unset) gets "" and `warn` says why, and the session takes no slice picks
    /// (slices() false: whole picks only, U4)
    std::string geo(const GeoMsg& g, const std::vector<GeoLayer>& layers, std::string& warn);
    bool slices() const { return on() && slices_; }   // (the last Geo set the lane up: slice picks taken)
    /// a request's slice picks (its rows' slice bits; c: kReqK, kReqWrong) of x (nr rows of n_embd floats, ldx apart)
    /// into ys (nr rows, ldy apart: each its slice picks' sum, in their order; 0 for a row with none).  False (err)
    bool run(int layer, int32_t c, const ReqRow* rows, int nr, const float* x, size_t ldx, float* ys, size_t ldy,
             std::string& err);
    double last_ms() const { return last_ms_; }   // the last run's time in the lane
    const SliceLane& lane() const { return lane_; }
    const ep::Parts& parts() const { return parts_; }   // (a fill's [k, nb) beside the main's [0, k): ds4_rx_server.cu)

    // The hybrid (U5c, ds4_rx.hpp protocol 4): DS4_RX_POOL=DIR at open() - a second parts index of the same pack with
    // the [0, k) of the hottest experts (ep_parts.py --pool-only); DS4_RX_POOL_GB=G (6) of them kept in RAM as W.
    bool pool_on() const { return pool_.is_open(); }
    /// a session's W, after its blobs: the pool's experts in heat order with a part [0, k) of every matrix (the last
    /// Geo's k), a layer the lane serves, not `held` (layer << 16 | expert, sorted: the GPU's), up to the budget;
    /// locked in RAM (mlock; where it may not, every page read once).  The ids (layer << 16 | expert; empty: no pool,
    /// a Geo that does not split)
    std::vector<uint32_t> pool_choose(const std::vector<uint32_t>& held);
    bool pools() const { return slices() && !w_.empty(); }   // (the session has a W: pool picks taken)
    std::string pool_note() const { return pool_note_; }   // (the last pool_choose in a line)
    /// the parts' bytes in the page cache now, of all of them (Linux mincore, every page), in a phrase
    std::string cached() const;
    int64_t pool_picks() const { return pool_picks_; }   // (W picks computed whole here)

private:
    ep::Parts parts_;
    SliceLane lane_;
    std::string dir_;
    int lo_ = 0, threads_ = 0;
    std::vector<int> cpus_;
    double last_ms_ = 0.0;
    bool wrong_said_ = false, slices_ = false;
    std::string geo_check(const GeoMsg& g, const std::vector<GeoLayer>& layers);
    ep::Parts pool_;
    std::string pool_dir_, pool_note_;
    double pool_gb_ = 6.0;
    int k_ = 0;
    std::vector<uint32_t> w_;   // (sorted)
    std::vector<std::pair<const void*, size_t>> locked_;   // (W's ranges locked: unlocked at the next pool_choose)
    int64_t pool_picks_ = 0;
    std::vector<SlicePick> pk_, wk_;
};

}  // namespace strata::ds4::rx
