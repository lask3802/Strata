// Strata. The expert server's CPU lane over the parts pack's n_ff slices (rx::SliceLane, src/dsv41/ds4_rx_lane.hpp)
// without a GPU or a model: random legal i-quant experts (ep_slice_test's generator), their parts written here (the
// ep-parts 1 format, a naive cutter), then each request's slice picks through the lane against the same arithmetic on
// the whole expert's bytes in place (gg::iq_cpu_slice: the reference for [k, nb) - the lane must give the same values
// up to the float order of the weighted sum), and against the expert computed whole with its [0, k) (the main's part):
// the two halves sum to it.  Per k from a stored [6, 9) and a stored [8, 9), 1-16 rows, with and without pool workers,
// on the ggml-cpu lane (the i-quants) and the trellis one (ds4_tq_cpu.cpp: 8 tokens a call at most, so an expert of more
// rows is two entries); the refusals (k below what is stored, a type or a layer the parts do not have, a pick out of range).
//   ds4_rx_lane_test     exit 0: all passed
#include "../../src/dsv41/ds4_rx_lane.hpp"

#include "ggml.h"   // the block sizes

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace ep = strata::ds4::ep;
namespace rx = strata::ds4::rx;
namespace fs = std::filesystem;

static constexpr int E = 1024, FF = 2304, NB = FF / 256, NL = 3, NE = 8, RB = 64;
static constexpr float kLim = 10.0f;
static int fails = 0;
#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);               \
            ++fails;                                                       \
        }                                                                  \
    } while (0)

static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; const int e = (int) ((u >> 23) & 0xff) - 127 + 15; const uint32_t m = u & 0x7fffffu;
    uint32_t v = ((uint32_t) e << 10) | (m >> 13); const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
static int tq_bits(int type) { return type == 145 ? 6 : type == 146 ? 7 : 0; }   // TQK6, TQK7 (Strata ids, not ggml's)
static size_t rowb(int type, int k) {
    return tq_bits(type) ? (size_t) (k / 128) * (size_t) (2 + 4 * tq_bits(type)) : ggml_row_size((ggml_type) type, k);
}
// random legal blocks of `type`, rows x k (iq_expert_test's generator; tq_expert_test's for the trellis types: any bits,
// a block's F16 scale first)
static void fill(std::mt19937_64& rng, int type, int rows, int k, uint8_t* p) {
    if (tq_bits(type)) {
        const size_t bb = (size_t) (2 + 4 * tq_bits(type)), nblk = (size_t) rows * (size_t) (k / 128);
        for (size_t i = 0; i < nblk * bb; ++i) p[i] = (uint8_t) rng();
        for (size_t b = 0; b < nblk; ++b) {
            const uint16_t d = f2h(0.05f + 0.0005f * (float) (rng() % 600));
            std::memcpy(p + b * bb, &d, 2);
        }
        return;
    }
    const size_t bb = ggml_type_size((ggml_type) type), nblk = (size_t) rows * (size_t) (k / ggml_blck_size((ggml_type) type));
    for (size_t i = 0; i < nblk * bb; ++i) p[i] = (uint8_t) rng();
    for (size_t b = 0; b < nblk; ++b) {
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* q = p + b * bb;
        if (type == 29) {   // block_iq1_m: its F16 scale is the top nibbles of its four u16 scales
            q[49] = (uint8_t) ((q[49] & 0x0f) | ((d & 0x000f) << 4));
            q[51] = (uint8_t) ((q[51] & 0x0f) | (d & 0x00f0));
            q[53] = (uint8_t) ((q[53] & 0x0f) | ((d & 0x0f00) >> 4));
            q[55] = (uint8_t) ((q[55] & 0x0f) | ((d & 0xf000) >> 8));
        } else {
            std::memcpy(q, &d, 2);
        }
    }
}

static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v == nullptr ? "" : v);
#else
    if (v == nullptr) unsetenv(k);
    else setenv(k, v, 1);
#endif
}

struct Expert {   // gate, up, down as the pack holds them
    std::vector<uint8_t> m[3];
};
static const int kTy[NL][2] = {{29, 16}, {17, 29}, {145, 146}};   // per layer: gate / up, down (IQ1_M, IQ2_XXS, IQ2_XS; TQK6, TQK7)
static size_t mat_rb(int l, int kind) { return kind < 2 ? rowb(kTy[l][0], E) : rowb(kTy[l][1], FF); }
static size_t mat_rows(int kind) { return (size_t) (kind < 2 ? FF : E); }

// ---- the parts writer (the index format of ds4_epparts.hpp; epparts_test.cpp's naive one, for whole i-quant experts)
static uint64_t index_stamp(const std::string& text) {
    uint64_t h = 1469598103934665603ull;
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        if (line.empty() || line.compare(0, 6, "stamp ") == 0) continue;
        for (char c : line) h = (h ^ (uint8_t) c) * 1099511628211ull;
        h = (h ^ (uint8_t) '\n') * 1099511628211ull;
    }
    return h;
}
static std::string hex16(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}
static std::vector<uint8_t> cut(int l, int kind, const std::vector<uint8_t>& whole, int a, int b) {
    const size_t rb = mat_rb(l, kind);
    if (kind < 2) return std::vector<uint8_t>(whole.begin() + (long) ((size_t) a * 256 * rb), whole.begin() + (long) ((size_t) b * 256 * rb));
    const size_t ub = rb / NB;
    std::vector<uint8_t> out;
    for (size_t r = 0; r < (size_t) E; ++r)
        out.insert(out.end(), whole.begin() + (long) (r * rb + (size_t) a * ub), whole.begin() + (long) (r * rb + (size_t) b * ub));
    return out;
}
// a file a set [a, b) of `sets`, each a part of every matrix of every expert of `layers` (in that order: the index's
// part lines too); `down_type_override` the type each down part line says (a test of the lane's checks); `kinds` a set
// each the matrices it holds (bit 0 gate, 1 up, 2 down; none given: all three)
static void write_parts(const std::string& dir, const std::vector<std::vector<Expert>>& ex, const std::vector<std::pair<int, int>>& sets,
                        const std::vector<int>& layers, int down_type_override = -1, const std::vector<int>& kinds = {}) {
    fs::create_directories(dir);
    std::string head = "ep-parts 1\npack_hash 00000000000000aa\nstamp 0000000000000000\nnonce 0000000000000001\n";
    head += "geometry " + std::to_string(E) + " " + std::to_string(FF) + " " + std::to_string(NL) + " " + std::to_string(NE) + " 256 4096\n";
    std::string covered = "covered", files, parts, heat;
    for (int l : layers) covered += " " + std::to_string(l);
    for (int l : layers)
        for (int e = 0; e < NE; ++e) heat += "heat " + std::to_string(l) + " " + std::to_string(e) + " " + std::to_string(100 - e) + "\n";
    std::vector<std::vector<uint8_t>> bins;
    for (size_t fi = 0; fi < sets.size(); ++fi) {
        const int a = sets[fi].first, b = sets[fi].second;
        std::vector<uint8_t> file(4096, 0);
        for (int l : layers)
            for (int e = 0; e < NE; ++e)
                for (int kind = 0; kind < 3; ++kind) {
                    if (!kinds.empty() && !(kinds[fi] & (1 << kind))) continue;
                    file.resize((file.size() + 4095) / 4096 * 4096, 0);
                    const std::vector<uint8_t> p = cut(l, kind, ex[(size_t) l][(size_t) e].m[kind], a, b);
                    const size_t rb = kind == 2 ? p.size() / E : mat_rb(l, kind), rows = p.size() / rb;
                    static const char* kn[3] = {"gate", "up", "down"};
                    const int ty = kind < 2 ? kTy[l][0] : (down_type_override >= 0 ? down_type_override : kTy[l][1]);
                    parts += "part " + std::to_string(l) + " " + std::to_string(e) + " " + kn[kind] + " " + std::to_string(ty) + " " +
                             std::to_string(fi) + " " + std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(rows) + " " +
                             std::to_string(rb) + " " + std::to_string(file.size()) + " " + std::to_string(p.size()) + "\n";
                    file.insert(file.end(), p.begin(), p.end());
                }
        file.resize((file.size() + 4095) / 4096 * 4096, 0);
        files += "file " + std::to_string(fi) + " ep-parts-" + std::to_string(a) + "-" + std::to_string(b) + ".bin " + std::to_string(a) +
                 " " + std::to_string(b) + " " + std::to_string(file.size()) + "\n";
        bins.push_back(std::move(file));
    }
    std::string idx = head + covered + "\n" + files + heat + parts;
    const uint64_t stamp = index_stamp(idx);
    const size_t at = idx.find("stamp ");
    idx = idx.substr(0, at) + "stamp " + hex16(stamp) + idx.substr(idx.find('\n', at));
    for (size_t fi = 0; fi < sets.size(); ++fi) {
        std::vector<uint8_t>& file = bins[fi];
        std::memcpy(file.data(), "EPPARTS1", 8);
        for (int i = 0; i < 8; ++i) file[8 + (size_t) i] = (uint8_t) (stamp >> (8 * i));
        const uint32_t hdr[3] = {(uint32_t) fi, (uint32_t) sets[fi].first, (uint32_t) sets[fi].second};
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 4; ++i) file[16 + (size_t) j * 4 + (size_t) i] = (uint8_t) (hdr[j] >> (8 * i));
        const std::string name = "ep-parts-" + std::to_string(sets[fi].first) + "-" + std::to_string(sets[fi].second) + ".bin";
        std::ofstream(dir + "/" + name, std::ios::binary | std::ios::trunc).write((const char*) file.data(), (std::streamsize) file.size());
    }
    std::ofstream(dir + "/ep-parts.txt", std::ios::binary | std::ios::trunc) << idx;
}
static void write_parts(const std::string& dir, const std::vector<std::vector<Expert>>& ex, int a, const std::vector<int>& layers,
                        int down_type_override = -1) {
    write_parts(dir, ex, {{a, NB}}, layers, down_type_override);
}

// ---- the reference: the CPU lane's steps on the whole expert's bytes in place over [ff0, ff0 + ffn) (ep_slice_test's
// run(), one token): y[c] = the slice's output
static std::vector<float> ref_slice(int l, const Expert& x, const gg::IqCpu& fm, const std::vector<uint8_t>& xa, int ff0, int ffn) {
    gg::IqCpu fs;
    size_t d_off = 0;
    std::string err;
    const bool sliced = ff0 != 0 || ffn != FF;
    if (sliced && !gg::iq_cpu_slice(fm, ff0, ffn, fs, d_off, err)) { std::printf("iq_cpu_slice: %s\n", err.c_str()); ++fails; return {}; }
    const gg::IqCpu& fd = sliced ? fs : fm;
    std::vector<float> h(FF, 0.0f), y(E, 0.0f);
    std::vector<uint8_t> hq(fd.h_bytes);
    const void* ap[1] = {xa.data()};
    float* hr[1] = {h.data()};
    for (int r0 = ff0; r0 < ff0 + ffn; r0 += RB)
        gg::iq_cpu_gate_up(fm, x.m[0].data(), x.m[1].data(), ap, 1, hr, r0, std::min(ff0 + ffn, r0 + RB), kLim);
    gg::iq_cpu_quant(fm.d_act, h.data() + ff0, ffn, hq.data());
    const void* hp[1] = {hq.data()};
    float* yr[1] = {y.data()};
    for (int r0 = 0; r0 < E; r0 += RB) gg::iq_cpu_down(fd, x.m[2].data() + d_off, hp, 1, yr, r0, std::min(E, r0 + RB));
    (void) l;
    return y;
}

struct Req {
    int nr = 0;
    std::vector<rx::SlicePick> picks;
    std::vector<float> x;   // nr x E
};
static Req make_req(std::mt19937_64& rng, int nr) {
    Req q;
    q.nr = nr;
    q.x.resize((size_t) nr * E);
    for (auto& v : q.x) v = ((float) (rng() % 20001) - 10000.0f) * 1e-4f;
    for (int t = 0; t < nr; ++t) {
        int es[NE];
        for (int e = 0; e < NE; ++e) es[e] = e;
        std::shuffle(es, es + NE, rng);
        const int n = 1 + (int) (rng() % 6);   // (1-6 of the row's picks are slice picks)
        for (int i = 0; i < n; ++i) q.picks.push_back({t, es[i], 0.05f + 0.3f * (float) (rng() % 1000) / 1000.0f});
    }
    return q;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937_64 rng(20261025);
    std::vector<std::vector<Expert>> ex((size_t) NL, std::vector<Expert>(NE));
    for (int l = 0; l < NL; ++l)
        for (int e = 0; e < NE; ++e)
            for (int kind = 0; kind < 3; ++kind) {
                Expert& x = ex[(size_t) l][(size_t) e];
                x.m[kind].assign(mat_rows(kind) * mat_rb(l, kind), 0);
                fill(rng, kind < 2 ? kTy[l][0] : kTy[l][1], (int) mat_rows(kind), kind < 2 ? E : FF, x.m[kind].data());
            }
    gg::IqCpu fm[NL];
    std::vector<rx::SliceLane::Types> types((size_t) NL);
    for (int l = 0; l < NL; ++l) {
        std::string err;
        CHECK(gg::iq_cpu_fmt(kTy[l][0], kTy[l][1], E, FF, fm[l], err));
        types[(size_t) l] = {kTy[l][0], kTy[l][1]};
    }
    const fs::path root = fs::temp_directory_path() / ("ds4_rx_lane_test_" + std::to_string((unsigned) (rng() & 0xffffff)));
    write_parts((root / "a6").string(), ex, 6, {0, 1, 2});
    write_parts((root / "a8").string(), ex, 8, {0, 1, 2});

    // ---- the lane's sums against the reference, per store, k, rows, workers
    struct Store { const char* dir; int a; };
    const Store stores[] = {{"a6", 6}, {"a8", 8}};
    double worst_slice = 0, worst_whole = 0;
    int cases = 0;
    for (const Store& s : stores) {
        ep::Parts p;
        std::string err;
        CHECK(p.open((root / s.dir).string(), err));
        if (!p.is_open()) { std::printf("open %s: %s\n", s.dir, err.c_str()); continue; }
        for (int threads : {0, 3}) {
            rx::SliceLane lane;
            CHECK(lane.init(p, types, E, FF, kLim, threads, {}, 50, err));
            if (lane.blocks() != NB) { std::printf("init: %s\n", err.c_str()); continue; }
            CHECK(lane.lo() == s.a);
            const double tl = (double) std::clock();
            const std::vector<int> rows = threads == 0 ? std::vector<int>{1, 3, 16} : std::vector<int>{1, 2, 4, 16};
            for (int k = s.a; k < NB; ++k)
                for (int nr : rows)
                    for (int l = 0; l < NL; ++l) {
                        const Req q = make_req(rng, nr);
                        // (x and y rows LD apart on every second request: the lane must not take them E apart)
                        const size_t LD = (cases & 1) ? (size_t) E + 64 : (size_t) E;
                        std::vector<float> xl((size_t) nr * LD, 1e30f), yl((size_t) nr * LD, 0.0f), y((size_t) nr * E, 0.0f);
                        for (int t = 0; t < nr; ++t) std::memcpy(xl.data() + (size_t) t * LD, q.x.data() + (size_t) t * E, (size_t) E * 4);
                        CHECK(lane.run(l, k, q.picks.data(), (int) q.picks.size(), nr, xl.data(), LD, yl.data(), LD, threads == 0, err));
                        bool pad_ok = true;   // (the gaps between y's rows untouched)
                        for (int t = 0; t < nr; ++t) {
                            std::memcpy(y.data() + (size_t) t * E, yl.data() + (size_t) t * LD, (size_t) E * 4);
                            for (size_t c = (size_t) E; c < LD; ++c) pad_ok = pad_ok && yl[(size_t) t * LD + c] == 0.0f;
                        }
                        CHECK(pad_ok);
                        // the reference: each pick's slice in place, the weighted sum in pick order; and the whole
                        std::vector<double> ys((size_t) nr * E, 0.0), yw((size_t) nr * E, 0.0), ymax((size_t) nr, 1e-30);
                        std::vector<float> ymain((size_t) nr * E, 0.0f);
                        for (const rx::SlicePick& pk : q.picks) {
                            std::vector<uint8_t> xa(fm[l].act_bytes);
                            gg::iq_cpu_quant(fm[l].gu_act, q.x.data() + (size_t) pk.t * E, E, xa.data());
                            const Expert& x = ex[(size_t) l][(size_t) pk.e];
                            const std::vector<float> rs = ref_slice(l, x, fm[l], xa, 256 * k, FF - 256 * k);
                            const std::vector<float> rm = ref_slice(l, x, fm[l], xa, 0, 256 * k);
                            const std::vector<float> rw = ref_slice(l, x, fm[l], xa, 0, FF);
                            if (rs.empty() || rm.empty() || rw.empty()) continue;
                            for (int c = 0; c < E; ++c) {
                                ys[(size_t) pk.t * E + c] += (double) pk.w * rs[(size_t) c];
                                ymain[(size_t) pk.t * E + c] += pk.w * rm[(size_t) c];
                                yw[(size_t) pk.t * E + c] += (double) pk.w * rw[(size_t) c];
                                ymax[(size_t) pk.t] = std::max(ymax[(size_t) pk.t], std::fabs(yw[(size_t) pk.t * E + c]));
                            }
                        }
                        double ds = 0, dw = 0;
                        for (int t = 0; t < nr; ++t)
                            for (int c = 0; c < E; ++c) {
                                const size_t i = (size_t) t * E + c;
                                const double a = std::fabs((double) y[i] - ys[i]) / ymax[(size_t) t];
                                const double b = std::fabs((double) y[i] + (double) ymain[i] - yw[i]) / ymax[(size_t) t];
                                ds = std::isfinite(a) ? std::max(ds, a) : 1e30;
                                dw = std::isfinite(b) ? std::max(dw, b) : 1e30;
                            }
                        worst_slice = std::max(worst_slice, ds);
                        ++cases;
                        worst_whole = std::max(worst_whole, dw);
                        // the slice's values up to the weighted sum's float order; the halves sum to the whole up to the
                        // order of the 9 block terms of each down row (ep_slice_test: a few eps of L1)
                        if (ds > 1e-6 || dw > 1e-4) {
                            std::printf("  store [%d,9) k %d rows %d layer %d threads %d: |lane - slice| %.2e, |main + lane - whole| %.2e of max|y|  FAIL\n",
                                        s.a, k, nr, l, threads, ds, dw);
                            ++fails;
                        }
                    }
            std::printf("  store [%d,9), %d pool workers: k %d..8, rows", s.a, threads, s.a);
            for (int nr : rows) std::printf(" %d", nr);
            std::printf(" (%.1f s; the lane %lld runs, %.2f GB/s)\n", ((double) std::clock() - tl) / CLOCKS_PER_SEC, (long long) lane.runs(),
                        lane.busy_ms() > 0 ? (double) lane.bytes() / lane.busy_ms() / 1e6 : 0.0);
            // a run adds to y (the 3080's sum is in it already): twice the same request is twice the sum
            {
                const Req q = make_req(rng, 2);
                std::vector<float> y1((size_t) 2 * E, 0.0f), y2((size_t) 2 * E, 0.0f);
                CHECK(lane.run(0, NB - 1, q.picks.data(), (int) q.picks.size(), 2, q.x.data(), E, y1.data(), E, true, err));
                CHECK(lane.run(0, NB - 1, q.picks.data(), (int) q.picks.size(), 2, q.x.data(), E, y2.data(), E, true, err));
                CHECK(lane.run(0, NB - 1, q.picks.data(), (int) q.picks.size(), 2, q.x.data(), E, y2.data(), E, true, err));
                double d = 0, m = 1e-30;   // (of max|y|: the second run's sums start from the first's, not from 0)
                for (size_t i = 0; i < y1.size(); ++i) {
                    d = std::max(d, (double) std::fabs(y2[i] - 2.0f * y1[i]));
                    m = std::max(m, (double) std::fabs(y1[i]));
                }
                CHECK(d / m < 1e-6);
            }
            // refusals: y untouched
            {
                const Req q = make_req(rng, 1);
                std::vector<float> y((size_t) E, 7.0f);
                const int k_low = s.a - 1;
                CHECK(!lane.run(0, k_low, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
                CHECK(err.find("the parts here hold [") != std::string::npos);   // (the lane's own refusal, before any part is looked up)
                CHECK(!lane.run(0, NB, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
                CHECK(!lane.run(NL, NB - 1, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
                rx::SlicePick bad[2] = {q.picks[0], {0, NE, 0.5f}};
                CHECK(!lane.run(0, NB - 1, bad, 2, 1, q.x.data(), E, y.data(), E, true, err));
                bad[1] = {1, 0, 0.5f};   // (row 1 of a 1-row request)
                CHECK(!lane.run(0, NB - 1, bad, 2, 1, q.x.data(), E, y.data(), E, true, err));
                CHECK(!lane.run(0, NB - 1, q.picks.data(), (int) q.picks.size(), 0, q.x.data(), E, y.data(), E, true, err));
                bad[1] = q.picks[0];   // (row 0 picks the same expert twice)
                CHECK(!lane.run(0, NB - 1, bad, 2, 1, q.x.data(), E, y.data(), E, true, err) && err.find("twice") != std::string::npos);
                rx::SlicePick nine[9];
                for (int i = 0; i < 9; ++i) nine[i] = {0, i % NE, 0.1f};   // (9 of row 0, within np <= nr x 8 of 2 rows)
                std::vector<float> x2((size_t) 2 * E, 0.5f), y2((size_t) 2 * E, 7.0f);
                CHECK(!lane.run(0, NB - 1, nine, 9, 2, x2.data(), E, y2.data(), E, true, err) && err.find("more than") != std::string::npos);
                for (float v : y2) CHECK(v == 7.0f);
                bool same = true;
                for (float v : y) same = same && v == 7.0f;
                CHECK(same);
                CHECK(lane.run(0, NB - 1, q.picks.data(), 0, 1, q.x.data(), E, y.data(), E, true, err));   // (no slice picks: nothing)
            }
        }
    }
    std::printf("lane against the slice in place: %d requests, max %.2e of max|y|; main [0,k) + lane against whole: max %.2e\n",
                cases, worst_slice, worst_whole);
    // ---- gate / up parts that overlap, [3, 6) and [5, 9), the down's [5, 9) alone (Parts::open takes them; a view of block
    // 5 is the first part, [3, 6)): the lo 6 (the lowest k a view takes to nb: the hello's - not the [5, 9) part's 5, which a
    // main would then send and lose the server for), k 5 refused rather than the slice's gate / up rows read past that
    // part's end; k 6 from [5, 9) alone, the slice's values
    {
        write_parts((root / "ov").string(), ex, {{3, 6}, {5, 9}}, {0, 1, 2}, -1, {3, 7});
        ep::Parts p;
        std::string err;
        CHECK(p.open((root / "ov").string(), err));
        rx::SliceLane lane;
        CHECK(lane.init(p, types, E, FF, kLim, 0, {}, 50, err) && lane.lo() == 6);
        const Req q = make_req(rng, 1);
        std::vector<float> y((size_t) E, 7.0f);
        CHECK(!lane.run(0, 5, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
        bool same = true;
        for (float v : y) same = same && v == 7.0f;
        CHECK(same);
        std::printf("  overlapping parts, k 5: refused (%s)\n", err.c_str());
        std::fill(y.begin(), y.end(), 0.0f);
        CHECK(lane.run(0, 6, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
        std::vector<double> ys((size_t) E, 0.0);
        double ymax = 1e-30, d = 0;
        for (const rx::SlicePick& pk : q.picks) {
            std::vector<uint8_t> xa(fm[0].act_bytes);
            gg::iq_cpu_quant(fm[0].gu_act, q.x.data() + (size_t) pk.t * E, E, xa.data());
            const std::vector<float> rs = ref_slice(0, ex[0][(size_t) pk.e], fm[0], xa, 6 * 256, FF - 6 * 256);
            for (int c = 0; c < E && !rs.empty(); ++c) ys[(size_t) c] += (double) pk.w * rs[(size_t) c];
        }
        for (int c = 0; c < E; ++c) ymax = std::max(ymax, std::fabs(ys[(size_t) c]));
        for (int c = 0; c < E; ++c) d = std::max(d, std::fabs((double) y[(size_t) c] - ys[(size_t) c]) / ymax);
        CHECK(d < 1e-6);
    }
    // ---- the hybrid (U5c): a pool [0, 6) beside the [6, 9) parts.  run_range's [0, 6) from it is the slice in place,
    // and with run's [6, 9) the expert whole; a layer the pool does not cover refused, y untouched; takes() says so
    {
        write_parts((root / "p6").string(), ex, {{0, 6}}, {0, 1, 2});
        write_parts((root / "p6l0").string(), ex, {{0, 6}}, {0});
        ep::Parts tail, pool, pool0;
        std::string err;
        CHECK(tail.open((root / "a6").string(), err));
        CHECK(pool.open((root / "p6").string(), err));
        CHECK(pool0.open((root / "p6l0").string(), err));
        rx::SliceLane lane;
        CHECK(lane.init(tail, types, E, FF, kLim, 2, {}, 50, err));
        double dw = 0, dm = 0;
        for (int l = 0; l < NL; ++l)
            for (int nr : {1, 4, 16}) {
                const Req q = make_req(rng, nr);
                std::vector<float> y((size_t) nr * E, 0.0f), y0((size_t) nr * E, 0.0f);
                CHECK(lane.run(l, 6, q.picks.data(), (int) q.picks.size(), nr, q.x.data(), E, y.data(), E, true, err));
                CHECK(lane.run_range(l, 0, 6, pool, q.picks.data(), (int) q.picks.size(), nr, q.x.data(), E, y0.data(), E, true, err));
                std::vector<double> yw((size_t) nr * E, 0.0), ym((size_t) nr * E, 0.0), ymax((size_t) nr, 1e-30);
                for (const rx::SlicePick& pk : q.picks) {
                    std::vector<uint8_t> xa(fm[l].act_bytes);
                    gg::iq_cpu_quant(fm[l].gu_act, q.x.data() + (size_t) pk.t * E, E, xa.data());
                    const Expert& x = ex[(size_t) l][(size_t) pk.e];
                    const std::vector<float> rw = ref_slice(l, x, fm[l], xa, 0, FF), rm = ref_slice(l, x, fm[l], xa, 0, 6 * 256);
                    for (int c = 0; c < E && !rw.empty() && !rm.empty(); ++c) {
                        yw[(size_t) pk.t * E + c] += (double) pk.w * rw[(size_t) c];
                        ym[(size_t) pk.t * E + c] += (double) pk.w * rm[(size_t) c];
                    }
                }
                for (int t = 0; t < nr; ++t)
                    for (int c = 0; c < E; ++c) ymax[(size_t) t] = std::max(ymax[(size_t) t], std::fabs(yw[(size_t) t * E + c]));
                for (int t = 0; t < nr; ++t)
                    for (int c = 0; c < E; ++c) {
                        const size_t i = (size_t) t * E + c;
                        dw = std::max(dw, std::fabs((double) y[i] + (double) y0[i] - yw[i]) / ymax[(size_t) t]);
                        dm = std::max(dm, std::fabs((double) y0[i] - ym[i]) / ymax[(size_t) t]);
                    }
            }
        CHECK(dw < 1e-4 && dm < 1e-6);
        std::printf("  the pool's [0, 6) through run_range: against the slice in place %.2e, with [6, 9) against whole %.2e\n", dm, dw);
        // run_sets: the slice picks' [6, 9) and some of them (W) [0, 6) in one pass - the bits of run then run_range
        // into one y; with 0 / 2 workers, 1-16 rows
        bool same_bits = true;
        int sets_runs = 0;
        for (int threads : {0, 2}) {
            rx::SliceLane ln;
            CHECK(ln.init(tail, types, E, FF, kLim, threads, {}, 50, err));
            for (int l = 0; l < NL; ++l)
                for (int nr : {1, 3, 16}) {
                    const Req q = make_req(rng, nr);
                    std::vector<rx::SlicePick> wpk;
                    for (size_t i = 0; i < q.picks.size(); i += 2) wpk.push_back(q.picks[i]);
                    std::vector<float> ya((size_t) nr * E, 0.0f), yb((size_t) nr * E, 0.0f);
                    CHECK(ln.run(l, 6, q.picks.data(), (int) q.picks.size(), nr, q.x.data(), E, ya.data(), E, true, err));
                    CHECK(ln.run_range(l, 0, 6, pool, wpk.data(), (int) wpk.size(), nr, q.x.data(), E, ya.data(), E, true, err));
                    const rx::LaneSet s[2] = {{6, 9, &tail, q.picks.data(), (int) q.picks.size()},
                                              {0, 6, &pool, wpk.data(), (int) wpk.size()}};
                    CHECK(ln.run_sets(l, s, 2, nr, q.x.data(), E, yb.data(), E, true, err));
                    same_bits = same_bits && std::memcmp(ya.data(), yb.data(), ya.size() * sizeof(float)) == 0;
                    ++sets_runs;
                }
        }
        CHECK(same_bits);
        std::printf("  run_sets ([6, 9) and W's [0, 6) in one pass) against run + run_range: %s in %d runs\n",
                    same_bits ? "the same bits" : "DIFFER", sets_runs);
        CHECK(lane.takes(pool, 0, 3, 0, 6) && !lane.takes(pool, 0, 3, 0, 7) && !lane.takes(pool, NL, 3, 0, 6));
        CHECK(lane.takes(pool0, 0, 3, 0, 6) && !lane.takes(pool0, 1, 3, 0, 6));
        const Req q = make_req(rng, 1);
        std::vector<float> y((size_t) E, 7.0f);
        CHECK(!lane.run_range(1, 0, 6, pool0, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
        CHECK(!lane.run_range(0, 0, 7, pool, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
        CHECK(!lane.run_range(0, 6, 6, pool, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
        bool same = true;
        for (float v : y) same = same && v == 7.0f;
        CHECK(same);
    }
    // ---- init refusals: a layer the parts do not cover, a down type not the layer's, another geometry
    {
        write_parts((root / "l0").string(), ex, 6, {0});
        write_parts((root / "ty").string(), ex, 6, {0, 1, 2}, 17);
        ep::Parts p;
        std::string err;
        CHECK(p.open((root / "l0").string(), err));
        {
            rx::SliceLane lane;
            CHECK(!lane.init(p, types, E, FF, kLim, 0, {}, 50, err));   // (layers 1, 2 served, not covered)
            std::printf("  refused, layer 1 not covered: %s\n", err.c_str());
            {   // (a lane whose init failed serves nothing - not the layers it had looked at before the refusal)
                std::vector<float> y((size_t) E, 0.0f);
                const Req q = make_req(rng, 1);
                CHECK(!lane.run(0, NB - 1, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));
            }
            std::vector<rx::SliceLane::Types> only0 = {types[0], {0, 0}, {0, 0}};
            rx::SliceLane lane2;
            CHECK(lane2.init(p, only0, E, FF, kLim, 0, {}, 50, err) && lane2.lo() == 6);
            std::vector<float> y((size_t) E, 0.0f);
            const Req q = make_req(rng, 1);
            CHECK(!lane2.run(1, 8, q.picks.data(), (int) q.picks.size(), 1, q.x.data(), E, y.data(), E, true, err));   // (not served)
            rx::SliceLane lane3;
            CHECK(!lane3.init(p, only0, 2 * E, FF, kLim, 0, {}, 50, err));   // (another dim)
            std::vector<rx::SliceLane::Types> none = {{0, 0}, {0, 0}, {0, 0}};
            rx::SliceLane lane4;
            CHECK(lane4.init(p, none, E, FF, kLim, 0, {}, 50, err) && lane4.lo() == NB);
        }
        p.close();
        CHECK(p.open((root / "ty").string(), err));
        rx::SliceLane lane;
        CHECK(!lane.init(p, types, E, FF, kLim, 0, {}, 50, err));
        std::printf("  refused, a down type not the layer's: %s\n", err.c_str());
    }
    // ---- SliceServer::geo: parts that do not serve a Geo (here another pack) refuse a main that splits (k below its
    // own nb); a main that does not (k 0, or its nb - also one whose nb is below the parts' blocks) is taken, warned,
    // with no slice picks
    {
        rx::SliceServer sv;
        std::string err;
        CHECK(sv.open((root / "a6").string(), -1, err) && sv.lo() == 6);
        rx::GeoMsg g;
        g.n_embd = E;
        g.n_layers = NL;
        g.limit = kLim;
        g.pack_hash = 0xab;   // (the parts': 0xaa)
        const std::vector<rx::GeoLayer> gl((size_t) NL);
        struct Case {
            int n_ff, k;
            bool refused;
        };
        const Case cs[4] = {{FF, 6, true}, {FF, NB, false}, {FF, 0, false}, {FF - 256, NB - 1, false}};
        for (const Case& c : cs) {
            g.n_ff = c.n_ff;
            g.k = c.k;
            std::string warn;
            const std::string why = sv.geo(g, gl, warn);
            CHECK(!why.empty() == c.refused && warn.empty() == c.refused && !sv.slices());
            std::printf("  geo, n_ff %d, k %d: %s\n", c.n_ff, c.k, c.refused ? ("refused: " + why).c_str() : ("taken, warned: " + warn).c_str());
        }
    }
    // ---- SliceServer with a pool (DS4_RX_POOL, U5c): W is the pool's experts the GPU does not hold (in RAM), a row's pool
    // pick comes back whole beside a slice pick's [6, 9); a pool pick not in W refused; a Geo whose k the pool does not
    // reach (7: [0, 7)) has no W
    {
        set_env("DS4_RX_POOL", (root / "p6").string().c_str());
        set_env("DS4_RX_POOL_GB", "1");
        rx::SliceServer sv;
        std::string err, warn;
        CHECK(sv.open((root / "a6").string(), -1, err) && sv.pool_on());
        set_env("DS4_RX_POOL", nullptr);
        set_env("DS4_RX_POOL_GB", nullptr);
        rx::GeoMsg g;
        g.n_embd = E;
        g.n_ff = FF;
        g.n_layers = NL;
        g.limit = kLim;
        g.pack_hash = 0xaa;
        g.k = 6;
        std::vector<rx::GeoLayer> gl((size_t) NL);
        for (int l = 0; l < NL; ++l) {
            gl[(size_t) l].K[0] = gl[(size_t) l].K[1] = kTy[l][0];
            gl[(size_t) l].K[2] = kTy[l][1];
        }
        CHECK(sv.geo(g, gl, warn).empty() && sv.slices());
        const uint32_t held = 1u << 16 | 2u;   // (layer 1 expert 2 in the GPU's VRAM)
        const std::vector<uint32_t> w = sv.pool_choose({held});
        CHECK(w.size() == (size_t) (NL * NE - 1) && !std::binary_search(w.begin(), w.end(), held) && sv.pools());
        std::printf("  %s\n", sv.pool_note().c_str());
        for (int l = 0; l < NL; ++l) {
            rx::ReqRow rw;
            rw.n = 3;
            const int es[3] = {1, 3, 2};
            const float ws[3] = {0.4f, 0.25f, 0.6f};
            for (int i = 0; i < 3; ++i) rw.e[i] = (int16_t) es[i], rw.w[i] = ws[i];
            rw.slice = (uint16_t) (0x3u | 1u << rx::kPoolBit);   // (pick 0 W, pick 1 slice, pick 2 the GPU's)
            CHECK(rx::row_refusal(rw, 6, sv.slices(), sv.pools()).empty());
            std::vector<float> x((size_t) E), ys((size_t) E, 7.0f);
            for (auto& v : x) v = ((float) (rng() % 20001) - 10000.0f) * 1e-4f;
            CHECK(sv.run(l, 6, &rw, 1, x.data(), E, ys.data(), E, err));
            std::vector<uint8_t> xa(fm[l].act_bytes);
            gg::iq_cpu_quant(fm[l].gu_act, x.data(), E, xa.data());
            const std::vector<float> r1 = ref_slice(l, ex[(size_t) l][1], fm[l], xa, 0, FF);
            const std::vector<float> r3 = ref_slice(l, ex[(size_t) l][3], fm[l], xa, 6 * 256, FF - 6 * 256);
            double d = 0, m = 1e-30;
            for (int c = 0; c < E && !r1.empty() && !r3.empty(); ++c) {
                const double ref = 0.4 * r1[(size_t) c] + 0.25 * r3[(size_t) c];
                d = std::max(d, std::fabs(ys[(size_t) c] - ref));
                m = std::max(m, std::fabs(ref));
            }
            CHECK(d / m < 1e-4);
            {   // the W pick cut at block 2 (DS4_EP_POOL_CUT): [2, nb) here, the main [0, 2)
                std::vector<float> yc((size_t) E, 7.0f);
                CHECK(sv.run(l, 6 | 2 << rx::kReqCutShift, &rw, 1, x.data(), E, yc.data(), E, err));
                const std::vector<float> r1c = ref_slice(l, ex[(size_t) l][1], fm[l], xa, 2 * 256, FF - 2 * 256);
                double dc = 0, mc = 1e-30, dw = 0;   // (dw: against the whole W pick's sums - not those)
                for (int c = 0; c < E && !r1c.empty() && !r3.empty(); ++c) {
                    const double ref = 0.4 * r1c[(size_t) c] + 0.25 * r3[(size_t) c];
                    dc = std::max(dc, std::fabs(yc[(size_t) c] - ref));
                    mc = std::max(mc, std::fabs(ref));
                    dw = std::max(dw, (double) std::fabs(yc[(size_t) c] - ys[(size_t) c]));
                }
                CHECK(dc / mc < 1e-4 && dw / mc > 1e-3);
                CHECK(!sv.run(l, 6 | 6 << rx::kReqCutShift, &rw, 1, x.data(), E, yc.data(), E, err) &&
                      err.find("not before k") != std::string::npos);
            }
            if (l == 1) {
                rw.slice = (uint16_t) (0x4u | 4u << rx::kPoolBit);   // (expert 2 of layer 1 tagged W: not in W)
                CHECK(!sv.run(l, 6, &rw, 1, x.data(), E, ys.data(), E, err) && err.find("not in W") != std::string::npos);
            }
        }
        g.k = 7;
        CHECK(sv.geo(g, gl, warn).empty() && sv.slices() && sv.pool_choose({}).empty() && !sv.pools());
    }
    std::error_code ec;
    fs::remove_all(root, ec);
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails != 0;
}
