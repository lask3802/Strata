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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4::rx {

constexpr int kLaneMaxRows = 16;    // a request's rows at most (ds4_rx.hpp kMaxRows)
constexpr int kLaneMaxPicks = 8;    // picks a row at most (kMaxPicks)

/// One slice pick of a request: its row (the window's token), the expert, the route weight.
struct SlicePick {
    int t = 0, e = 0;
    float w = 0.0f;
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

    // (read on the thread that runs: not atomic)
    uint64_t bytes() const { return bytes_; }   // weight bytes the runs read
    double busy_ms() const { return busy_ms_; }   // their time, from the first quantization to the last sum
    int64_t runs() const { return runs_; }

private:
    bool init_(const ep::Parts& parts, const std::vector<Types>& types, int n_embd, int n_ff, float limit, int threads,
               const std::vector<int>& cpus, int spin_us, std::string& err);
    struct Fmt {
        bool on = false;
        int cap = kLaneMaxRows;   // rows an entry (a call of the rows) at most
        gguf::IqCpu fm;
    };
    struct Ent {
        int e = 0, n = 0, slot0 = 0;
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

}  // namespace strata::ds4::rx
