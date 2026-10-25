// src/dsv41/ds4_rx_lane.cpp - the expert server's CPU lane over the parts pack's n_ff slices (ds4_rx_lane.hpp).
#include "ds4_rx_lane.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>

namespace strata::ds4::rx {
namespace gg = strata::ds4::gguf;

namespace {
constexpr int kRb = 64;     // rows a job (the main's cpu_experts_iq: RB)
constexpr int kSumCols = 512;   // the weighted sums' columns a job
double lane_ms_now() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// the lowest k a matrix's parts take as run() looks it up (view(k): the first part that holds block k), every k from
// it to nb in a part that runs to nb (overlapping parts: the first one found for k may end before nb); nb: none
int run_lo(const ep::Parts& parts, int l, int e, int kind, int nb) {
    int k = nb;
    ep::View v;
    while (k > 0 && parts.view(l, e, kind, k - 1, v) && v.b == nb) --k;
    return k;
}
}  // namespace

bool SliceLane::init(const ep::Parts& parts, const std::vector<Types>& types, int n_embd, int n_ff, float limit, int threads,
                     const std::vector<int>& cpus, int spin_us, std::string& err) {
    if (init_(parts, types, n_embd, n_ff, limit, threads, cpus, spin_us, err)) return true;
    fmt_.clear();   // (refused: no layer served, every run refused)
    return false;
}

bool SliceLane::init_(const ep::Parts& parts, const std::vector<Types>& types, int n_embd, int n_ff, float limit, int threads,
                      const std::vector<int>& cpus, int spin_us, std::string& err) {
    parts_ = &parts;
    dim_ = n_embd;
    ff_ = n_ff;
    lim_ = limit;
    nb_ = parts.blocks();
    lo_ = 0;
    if (!parts.is_open() || parts.dim() != n_embd || parts.n_ff() != n_ff || parts.unit() != 256 || nb_ <= 0) {
        err = "the parts pack's geometry (dim " + std::to_string(parts.dim()) + ", n_ff " + std::to_string(parts.n_ff()) +
              ", unit " + std::to_string(parts.unit()) + ") is not the main process's (" + std::to_string(n_embd) + ", " +
              std::to_string(n_ff) + ", 256)";
        return false;
    }
    fmt_.assign(types.size(), Fmt{});
    bool any = false;
    for (size_t l = 0; l < types.size(); ++l) {
        const Types& ty = types[l];
        if (ty.tg == 0 && ty.td == 0) continue;
        const std::string ln = "layer " + std::to_string(l);
        if (!parts.covers((int) l) || (int) l >= parts.layers()) {
            err = ln + ": not in the parts pack";
            return false;
        }
        Fmt& f = fmt_[l];
        std::string ferr;
        if (!gg::iq_cpu_fmt(ty.tg, ty.td, n_embd, n_ff, f.fm, ferr)) {
            err = ln + "'s experts: " + ferr;
            return false;
        }
        if (f.fm.act_bytes > (size_t) n_embd * 4 || f.fm.h_bytes > (size_t) n_ff * 4) {
            err = ln + "'s experts: activations larger than f32";
            return false;
        }
        // every expert's three matrices: a stored part that runs to nb, of the layer's type and layout
        for (int e = 0; e < parts.experts(); ++e)
            for (int kind = 0; kind < 3; ++kind) {
                ep::View v;
                const std::string me = ln + " expert " + std::to_string(e) + (kind == 0 ? " gate" : kind == 1 ? " up" : " down");
                if (!parts.view((int) l, e, kind, nb_ - 1, v) || v.b != nb_) {
                    err = me + ": no part holds the last n_ff block";
                    return false;
                }
                const int want = kind < 2 ? ty.tg : ty.td;
                const size_t ub = kind < 2 ? 0 : f.fm.d_row / (size_t) nb_;   // (a down row's bytes a block)
                if (v.type != want || (kind < 2 ? v.row_bytes != f.fm.gu_row || v.stride != v.row_bytes || v.rows != 256u
                                                : v.rows != (uint32_t) n_embd || v.row_bytes != ub || v.stride != ub * (size_t) (v.b - v.a))) {
                    err = me + ": type " + std::to_string(v.type) + " or layout not the layer's (type " + std::to_string(want) + ")";
                    return false;
                }
                lo_ = std::max(lo_, run_lo(parts, (int) l, e, kind, nb_));
            }
        f.cap = gg::tq_cpu_type(ty.tg) || gg::tq_cpu_type(ty.td) ? 8 : kLaneMaxRows;   // (the trellis rows: 8 tokens a call, tq_cpu.hpp kTqMaxTokens - the main's ent_cap)
        f.ty = ty;
        f.on = true;
        any = true;
    }
    if (!any) lo_ = nb_;
    aq_.assign((size_t) kLaneMaxRows * (size_t) n_embd * 4, 0);
    h_.assign((size_t) kLaneMaxRows * kLaneMaxPicks * (size_t) n_ff, 0.0f);
    hq_.assign((size_t) kLaneMaxRows * kLaneMaxPicks * (size_t) n_ff * 4, 0);
    yd_.assign((size_t) kLaneMaxRows * kLaneMaxPicks * (size_t) n_embd, 0.0f);
    ents_.reserve((size_t) kLaneMaxRows * kLaneMaxPicks);
    pool_ = std::make_unique<Pool>(std::max(0, threads), cpus, spin_us);
    return true;
}

bool SliceLane::views(const ep::Parts& from, int layer, int e, int k0, int k1, ep::View* v) const {
    const Fmt& f = fmt_[(size_t) layer];
    const size_t ub = f.fm.d_row / (size_t) nb_;   // (a down row's bytes a block)
    for (int kind = 0; kind < 3; ++kind)
        if (!from.view(layer, e, kind, k0, v[kind]) || v[kind].b < k1 || v[kind].type != (kind < 2 ? f.ty.tg : f.ty.td) ||
            (kind < 2 ? v[kind].row_bytes != f.fm.gu_row
                      : v[kind].rows != (uint32_t) dim_ || v[kind].stride != ub * (size_t) (v[kind].b - v[kind].a)))
            return false;
    return true;
}

bool SliceLane::run(int layer, int k, const SlicePick* picks, int np, int nr, const float* x, size_t ldx, float* y, size_t ldy,
                    bool help, std::string& err) {
    if (layer < 0 || layer >= (int) fmt_.size() || !fmt_[(size_t) layer].on) {
        err = "layer " + std::to_string(layer) + " is not served here";
        return false;
    }
    if (k < lo_ || k >= nb_) {
        err = "n_ff slice [" + std::to_string(k) + ", " + std::to_string(nb_) + "): the parts here hold [" +
              std::to_string(lo_) + ", " + std::to_string(nb_) + ")";
        return false;
    }
    return run_range(layer, k, nb_, *parts_, picks, np, nr, x, ldx, y, ldy, help, err);
}

bool SliceLane::run_range(int layer, int k0, int k1, const ep::Parts& from, const SlicePick* picks, int np, int nr,
                          const float* x, size_t ldx, float* y, size_t ldy, bool help, std::string& err) {
    if (layer < 0 || layer >= (int) fmt_.size() || !fmt_[(size_t) layer].on) {
        err = "layer " + std::to_string(layer) + " is not served here";
        return false;
    }
    if (k0 < 0 || k0 >= k1 || k1 > nb_) {
        err = "n_ff blocks [" + std::to_string(k0) + ", " + std::to_string(k1) + ") outside [0, " + std::to_string(nb_) + ")";
        return false;
    }
    if (nr < 1 || nr > kLaneMaxRows || np < 0 || np > nr * kLaneMaxPicks) {
        err = "a request of " + std::to_string(nr) + " rows and " + std::to_string(np) + " slice picks";
        return false;
    }
    if (np == 0) return true;
    for (int i = 0; i < np; ++i)
        if (picks[i].t < 0 || picks[i].t >= nr || picks[i].e < 0 || picks[i].e >= from.experts()) {
            err = "a slice pick of row " + std::to_string(picks[i].t) + ", expert " + std::to_string(picks[i].e);
            return false;
        }
    const double t0 = lane_ms_now();
    const gg::IqCpu& fm = fmt_[(size_t) layer].fm;
    const int cap = fmt_[(size_t) layer].cap;
    const int E = dim_, ff0 = 256 * k0, ffn = (k1 >= nb_ ? ff_ : 256 * k1) - ff0;
    gg::IqCpu fs;
    size_t d_off = 0;
    if (!gg::iq_cpu_slice(fm, ff0, ffn, fs, d_off, err)) return false;
    // the entries: the picks by expert, then row (more than cap of one expert: the next entry); a row picks an expert
    // once and kLaneMaxPicks experts at most
    std::vector<int> ord((size_t) np);
    int per_row[kLaneMaxRows] = {};
    for (int i = 0; i < np; ++i) {
        ord[(size_t) i] = i;
        if (++per_row[picks[i].t] > kLaneMaxPicks) {
            err = "row " + std::to_string(picks[i].t) + ": more than " + std::to_string(kLaneMaxPicks) + " slice picks";
            return false;
        }
    }
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return picks[a].e != picks[b].e ? picks[a].e < picks[b].e : picks[a].t < picks[b].t; });
    for (size_t i = 1; i < ord.size(); ++i)
        if (picks[ord[i]].e == picks[ord[i - 1]].e && picks[ord[i]].t == picks[ord[i - 1]].t) {
            err = "row " + std::to_string(picks[ord[i]].t) + " picks expert " + std::to_string(picks[ord[i]].e) + " twice";
            return false;
        }
    std::vector<int> slot_of((size_t) np);
    ents_.clear();
    for (size_t a = 0; a < ord.size();) {
        Ent en;
        en.e = picks[ord[a]].e;
        en.slot0 = (int) a;
        // (a part to k1: two that overlap, the first one found may end before it)
        if (!views(from, layer, en.e, k0, k1, en.v) || fm.d_row / (size_t) nb_ * (size_t) (k1 - k0) != fs.d_row) {
            err = "layer " + std::to_string(layer) + " expert " + std::to_string(en.e) + ": no part holds blocks [" + std::to_string(k0) +
                  ", " + std::to_string(k1) + ") of the layer's type and layout";
            return false;
        }
        while (a < ord.size() && picks[ord[a]].e == en.e && en.n < cap) {
            slot_of[(size_t) ord[a]] = (int) a;
            en.tok[en.n++] = picks[ord[a]].t;
            ++a;
        }
        ents_.push_back(en);
    }
    // each row's x as the gate / up activation, once
    bool used[kLaneMaxRows] = {};
    for (int i = 0; i < np; ++i) used[picks[i].t] = true;
    for (int t = 0; t < nr; ++t)
        if (used[t]) gg::iq_cpu_quant(fm.gu_act, x + (size_t) t * ldx, E, aq_.data() + (size_t) t * E * 4);
    const int ne = (int) ents_.size(), GB = (ffn + kRb - 1) / kRb, DB = (E + kRb - 1) / kRb;
    const std::vector<Ent>& ents = ents_;
    const float lim = lim_;
    std::function<void(int)> f = [&](int j) {   // the slice's gate / up rows, the swiglu into each slot's h
        const Ent& en = ents[(size_t) (j / GB)];
        const int r0 = (j % GB) * kRb, r1 = std::min(ffn, r0 + kRb);
        const void* ap[kLaneMaxRows];
        float* hp[kLaneMaxRows];
        for (int i = 0; i < en.n; ++i) {
            ap[i] = aq_.data() + (size_t) en.tok[i] * E * 4;
            hp[i] = h_.data() + (size_t) (en.slot0 + i) * ff_;
        }
        gg::iq_cpu_gate_up(fm, en.v[0].data, en.v[1].data, ap, en.n, hp, r0, r1, lim);
    };
    auto go = [&](int n) {
        pool_->start(n, f);
        if (help || pool_->size() == 1) pool_->help();
        pool_->wait();
    };
    go(ne * GB);
    f = [&](int s) {   // each slot's h quantized alone
        gg::iq_cpu_quant(fm.d_act, h_.data() + (size_t) s * ff_, ffn, hq_.data() + (size_t) s * ff_ * 4);
    };
    go(np);
    f = [&](int j) {   // the down rows over the slice's values, at the stored part's stride
        const Ent& en = ents[(size_t) (j / DB)];
        const int r0 = (j % DB) * kRb, r1 = std::min(E, r0 + kRb);
        const void* hp[kLaneMaxRows];
        float* yp[kLaneMaxRows];
        for (int i = 0; i < en.n; ++i) {
            hp[i] = hq_.data() + (size_t) (en.slot0 + i) * ff_ * 4;
            yp[i] = yd_.data() + (size_t) (en.slot0 + i) * E;
        }
        gg::IqCpu fe = fs;   // (the slice's rows, at the stored part's stride: the part holds [a, nb) of each row)
        fe.d_stride = en.v[2].stride;
        gg::iq_cpu_down(fe, en.v[2].data, hp, en.n, yp, r0, r1);
    };
    go(ne * DB);
    const int SC = (E + kSumCols - 1) / kSumCols;
    f = [&](int j) {   // each row's weighted sum, its picks in their order (a job its columns: the same order)
        const int c0 = j * kSumCols, c1 = std::min(E, c0 + kSumCols);
        for (int i = 0; i < np; ++i) {
            const float w = picks[i].w;
            const float* yd = yd_.data() + (size_t) slot_of[(size_t) i] * E;
            float* yo = y + (size_t) picks[i].t * ldy;
            for (int c = c0; c < c1; ++c) yo[c] += w * yd[c];
        }
    };
    go(SC);
    bytes_ += (uint64_t) ents.size() * (2 * (uint64_t) ffn * fm.gu_row + (uint64_t) E * fs.d_row);
    busy_ms_ += lane_ms_now() - t0;
    ++runs_;
    (void) d_off;
    return true;
}

bool SliceServer::open(const std::string& dir, int link, std::string& err) {
    dir_ = dir;
    if (!parts_.open(dir, err)) return false;
    const int nb = parts_.blocks();
    lo_ = 0;
    bool any = false;
    for (int l = 0; l < parts_.layers(); ++l) {   // (the lowest k every covered layer's every matrix takes: the hello's lo)
        if (!parts_.covers(l)) continue;
        any = true;
        for (int e = 0; e < parts_.experts(); ++e)
            for (int kind = 0; kind < 3; ++kind) {
                ep::View v;
                if (!parts_.view(l, e, kind, nb - 1, v) || v.b != nb) {
                    err = "the parts at " + dir + ": layer " + std::to_string(l) + " expert " + std::to_string(e) +
                          (kind == 0 ? " gate" : kind == 1 ? " up" : " down") + ": no part holds the last n_ff block";
                    parts_.close();
                    return false;
                }
                lo_ = std::max(lo_, run_lo(parts_, l, e, kind, nb));
            }
    }
    if (!any || lo_ < 1) {
        err = "the parts at " + dir + (any ? " start at block 0: the main would have no part of n_ff" : " cover no layer");
        parts_.close();
        return false;
    }
    // the lane's CPUs: off the link core (its SMT siblings too), one a physical core; DS4_RX_LANE_THREADS past them:
    // every logical CPU off the link core
    const std::vector<int> link_sibs = link >= 0 ? os::cpu_siblings(link) : std::vector<int>();
    std::vector<int> phys, logical;
    const int hw = (int) std::max(1u, std::thread::hardware_concurrency());
    for (int c = 0; c < hw && c < 1024; ++c) {
        if (!os::cpu_allowed(c) || std::find(link_sibs.begin(), link_sibs.end(), c) != link_sibs.end()) continue;
        logical.push_back(c);
        if (os::cpu_siblings(c).front() == c) phys.push_back(c);
    }
    const char* tv = std::getenv("DS4_RX_LANE_THREADS");
    const int want = tv != nullptr && *tv != '\0' ? std::max(0, std::atoi(tv)) : (int) phys.size();
    cpus_ = want <= (int) phys.size() ? phys : logical;
    threads_ = std::min(want, (int) cpus_.size());   // (no CPU known: none - the link thread computes the slices)
    cpus_.resize((size_t) threads_);
    if (const char* wv = std::getenv("DS4_RX_PARTS_WARM"); wv != nullptr && *wv == '1') {
        const double t0 = lane_ms_now();
        uint64_t sum = 0, n = 0;
        for (const ep::Parts::File& f : parts_.files()) {
            os::advise(f.fd, 0, f.bytes, true);
            for (uint64_t at = 0; at < f.bytes; at += 4096) sum += f.map[at];   // (each page read once)
            n += f.bytes;
        }
        std::fprintf(stderr, "expert server: the parts' %.2f GB read once in %.1f s (DS4_RX_PARTS_WARM; %llu)\n",
                     (double) n / 1e9, (lane_ms_now() - t0) / 1000.0, (unsigned long long) (sum & 1));
    }
    if (const char* pv = std::getenv("DS4_RX_POOL"); pv != nullptr && *pv != '\0') {   // (the hybrid's pool, U5c)
        pool_dir_ = pv;
        if (!pool_.open(pool_dir_, err)) {
            err = "the pool at " + pool_dir_ + ": " + err;
            parts_.close();
            return false;
        }
        if (pool_.pack_hash() != parts_.pack_hash() || pool_.dim() != parts_.dim() || pool_.n_ff() != parts_.n_ff() ||
            pool_.unit() != parts_.unit() || pool_.layers() != parts_.layers() || pool_.experts() != parts_.experts()) {
            err = "the pool at " + pool_dir_ + " is not of the parts' pack and geometry";
            pool_.close();
            parts_.close();
            return false;
        }
        if (const char* gv = std::getenv("DS4_RX_POOL_GB"); gv != nullptr && *gv != '\0') pool_gb_ = std::max(0.0, std::atof(gv));
    }
    return true;
}

std::vector<uint32_t> SliceServer::pool_choose(const std::vector<uint32_t>& held) {
    for (const auto& r : locked_) os::unlock_range(r.first, r.second);
    locked_.clear();
    w_.clear();
    pool_note_.clear();
    if (!pool_on() || !slices() || k_ < 1 || k_ >= parts_.blocks()) return w_;
    const double t0 = lane_ms_now();
    const uint64_t budget = (uint64_t) (pool_gb_ * 1e9);
    uint64_t bytes = 0;
    std::vector<std::pair<const void*, size_t>> rs;
    for (const ep::Parts::Heat& h : pool_.heat()) {   // (hottest first)
        const uint32_t id = (uint32_t) h.layer << 16 | (uint32_t) h.expert;
        if (std::binary_search(held.begin(), held.end(), id) || !lane_.takes(pool_, h.layer, h.expert, 0, k_)) continue;
        ep::View v[3];
        uint64_t b = 0;
        for (int kind = 0; kind < 3; ++kind) {   // (the stored parts from block 0, whole: [0, k) of them is read)
            pool_.view(h.layer, h.expert, kind, 0, v[kind]);
            b += (uint64_t) v[kind].rows * v[kind].stride;
        }
        if (bytes + b > budget) break;
        bytes += b;
        for (int kind = 0; kind < 3; ++kind) rs.push_back({v[kind].data, (size_t) v[kind].rows * v[kind].stride});
        w_.push_back(id);
    }
    os::prefetch(rs);   // (the kernel's read-ahead over all of W, then each range locked)
    size_t lk = 0;
    uint64_t sum = 0;
    for (const auto& r : rs) {
        if (os::lock_range(r.first, r.second)) {
            locked_.push_back(r);
            ++lk;
            continue;
        }
        for (size_t at = 0; at < r.second; at += 4096) sum += ((const uint8_t*) r.first)[at];   // (each page read once)
    }
    std::sort(w_.begin(), w_.end());
    char b[300];
    std::snprintf(b, sizeof b, "W: %zu of the pool's experts (%s) [0, %d) in RAM, %.2f GB of DS4_RX_POOL_GB %.1f, %s in "
                  "%.1f s (%llu)", w_.size(), pool_dir_.c_str(), k_, (double) bytes / 1e9, pool_gb_,
                  lk == rs.size() ? "locked" : lk == 0 ? "read once (not locked: RLIMIT_MEMLOCK?)" : "partly locked",
                  (lane_ms_now() - t0) / 1000.0, (unsigned long long) (sum & 1));
    pool_note_ = b;
    return w_;
}

std::string SliceServer::note() const {
    if (!on()) return "no parts (whole picks only)";
    uint64_t n = 0, cold = 0;
    for (const ep::Parts::File& f : parts_.files()) {
        n += f.bytes;
        cold += os::cold_bytes(f.map, f.bytes);
    }
    char cache[64] = "";
#if defined(__linux__)
    std::snprintf(cache, sizeof cache, ", %.2f GB of them in the page cache", (double) (n - cold) / 1e9);
#endif
    char b[400];
    std::snprintf(b, sizeof b, "parts %s: n_ff blocks [%d, %d), %.2f GB%s (pack_hash %016llx); the CPU lane %d workers "
                  "off the link core", dir_.c_str(), lo_, parts_.blocks(), (double) n / 1e9, cache,
                  (unsigned long long) parts_.pack_hash(), threads_);
    std::string s = b;
    if (!cpus_.empty()) {
        s += " (CPUs";
        for (int c : cpus_) s += " " + std::to_string(c);
        s += ")";
    }
    if (pool_on()) {
        uint64_t pn = 0;
        for (const ep::Parts::File& f : pool_.files()) pn += f.bytes;
        std::snprintf(b, sizeof b, "; the pool %s: %zu experts, %.2f GB (DS4_RX_POOL_GB %.1f of them kept as W)",
                      pool_dir_.c_str(), pool_.heat().size(), (double) pn / 1e9, pool_gb_);
        s += b;
    }
    return s;
}

std::string SliceServer::geo(const GeoMsg& g, const std::vector<GeoLayer>& layers, std::string& warn) {
    warn.clear();
    slices_ = false;
    k_ = g.k;
    if (!on()) return "";
    const std::string why = geo_check(g, layers);
    if (why.empty()) {
        slices_ = true;
        return "";
    }
    if (g.k >= 1 && g.k < g.n_ff / 256) return why;   // (the main splits - k below its own nb: it needs these parts)
    warn = why + " - the main does not split (k " + std::to_string(g.k) + "): whole picks only";
    return "";
}

std::string SliceServer::geo_check(const GeoMsg& g, const std::vector<GeoLayer>& layers) {
    if (g.pack_hash != parts_.pack_hash()) {
        char b[200];
        std::snprintf(b, sizeof b, "the parts at %s were cut from another pack (pack_hash %016llx, the main's %016llx)",
                      dir_.c_str(), (unsigned long long) parts_.pack_hash(), (unsigned long long) g.pack_hash);
        return b;
    }
    std::vector<SliceLane::Types> ty(layers.size());
    for (size_t l = 0; l < layers.size(); ++l) {
        const GeoLayer& G = layers[l];
        if (G.K[0] == 0) continue;
        if (G.K[0] != G.K[1])
            return "layer " + std::to_string(l) + ": gate and up of two types (" + std::to_string(G.K[0]) + ", " +
                   std::to_string(G.K[1]) + "): the CPU lane takes one";
        ty[l] = {G.K[0], G.K[2]};
    }
    std::string err;
    if (!lane_.init(parts_, ty, g.n_embd, g.n_ff, g.limit, threads_, cpus_, 50000, err))
        return "the parts at " + dir_ + ": " + err;
    return "";
}

bool SliceServer::run(int layer, int32_t c, const ReqRow* rows, int nr, const float* x, size_t ldx, float* ys, size_t ldy,
                      std::string& err) {
    last_ms_ = 0.0;
    const int dim = parts_.dim();
    std::vector<SlicePick>& pk = pk_;
    pk.clear();
    wk_.clear();
    for (int j = 0; j < nr; ++j) {
        for (int c2 = 0; c2 < dim; ++c2) ys[(size_t) j * ldy + (size_t) c2] = 0.0f;
        for (int i = 0; i < rows[j].n && i < kMaxPicks; ++i) {
            if ((rows[j].slice >> i) & 1) pk.push_back({j, rows[j].e[i], rows[j].w[i]});
            if ((rows[j].slice >> (kPoolBit + i)) & 1) wk_.push_back({j, rows[j].e[i], rows[j].w[i]});
        }
    }
    if (pk.empty()) return true;
    if (!wk_.empty()) {   // (W picks: whole here - [0, k) from the pool below, [k, nb) with the slice picks)
        if (!pools()) {
            err = "pool picks, and no W here (DS4_RX_POOL)";
            return false;
        }
        for (const SlicePick& q : wk_)
            if (!std::binary_search(w_.begin(), w_.end(), (uint32_t) layer << 16 | (uint32_t) (uint16_t) q.e)) {
                err = "layer " + std::to_string(layer) + " expert " + std::to_string(q.e) + ": a pool pick not in W";
                return false;
            }
    }
    if (!slices()) {
        err = "slice picks, and no parts here (--expert-server-parts)";
        return false;
    }
    int k = c & kReqK, times = 1;
    if (c & kReqWrong) {   // (the check's own check: [k - 1, nb) where the parts hold it, else [k, nb) added twice)
        if (k - 1 >= lo_) --k;
        else times = 2;
        if (!wrong_said_)
            std::fprintf(stderr, "expert server: DS4_EP_SPLIT_VERIFY=2: the slice picks' sums wrong on purpose (%s)\n",
                         times == 2 ? "[k, nb) twice: the parts hold no block before k" : "from block k - 1");
        wrong_said_ = true;
    }
    const double t0 = lane_ms_now();
    for (int r = 0; r < times; ++r)
        if (!lane_.run(layer, k, pk.data(), (int) pk.size(), nr, x, ldx, ys, ldy, false, err)) return false;
    if (!wk_.empty()) {
        if (!lane_.run_range(layer, 0, c & kReqK, pool_, wk_.data(), (int) wk_.size(), nr, x, ldx, ys, ldy, false, err))
            return false;
        pool_picks_ += (int64_t) wk_.size();
    }
    last_ms_ = lane_ms_now() - t0;
    return true;
}

}  // namespace strata::ds4::rx
