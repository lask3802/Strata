// Strata. The n_ff slices of expert parallelism on the CPU lane of the DeepSeek-V4.1 path (gg::iq_cpu_slice, ds4_gguf_cpu.hpp;
// the byte copies of ds4_ffslice.hpp): an expert computed as slices [0, a), [a, b) .. of its n_ff, each read in place from
// the whole expert's bytes, sums to the expert computed whole up to the order of the float sums.  Per slice it checks
// that the slice's h rows are the whole h's bit for bit, that the slice's h quantized alone is the same activation blocks
// as the whole h's, and that the down over the slice in place (strided rows) gives the same bits as the down over the
// slice copied out to a blob of its own (packed rows: what a device or a parts file would hold); then the sum of the
// slices against the whole, bounded by the products' L1 norm (ggml's decode of the weights, the quantized h); and the
// blob -> slices -> blob round trip byte for byte.  Random legal blocks (any bit pattern with a sane F16 scale), and real
// experts read from a pack (read only) when one is given.  No GPU.
//   ep_slice_test [--type 29,16,17] [--tokens 1,2,3,6,8] [--pack DIR --layers 3,7,17,23 --experts 0,7]
#include "../../src/dsv41/ds4_ffslice.hpp"
#include "../../src/dsv41/ds4_gguf_cpu.hpp"

#include "ggml.h"   // ggml_get_type_traits(type)->to_float, the block sizes

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
using strata::ds4::FfSlice;

static constexpr int E = 5120, FF = 2304, NB = FF / 256, RB = 64;   // the DeepSeek-V4.1-Flash geometry; rows a job
static constexpr float kLim = 10.0f;                                  // swiglu_limit
static int fails = 0;

static std::vector<int> ints(const char* p) {
    std::vector<int> v;
    while (*p) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}
static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; const int e = (int) ((u >> 23) & 0xff) - 127 + 15; const uint32_t m = u & 0x7fffffu;
    uint32_t v = ((uint32_t) e << 10) | (m >> 13); const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
static size_t rowb(int type, int k) { return ggml_row_size((ggml_type) type, k); }
static size_t al256(size_t n) { return (n + 255) / 256 * 256; }

// random legal blocks of `type`, rows x k (iq_expert_test's generator)
static void fill(std::mt19937_64& rng, int type, int rows, int k, uint8_t* p) {
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
// an activation of a ggml type decoded (Q8_K has no to_float: float d, int8 qs[256], int16 bsums[16])
static std::vector<float> act_values(int type, const void* act, int k) {
    std::vector<float> v((size_t) k);
    if (type == GGML_TYPE_Q8_K) {
        const size_t bs = rowb(type, 256);
        for (int b = 0; b < k / 256; ++b) {
            const uint8_t* p = (const uint8_t*) act + (size_t) b * bs;
            float d; std::memcpy(&d, p, 4);
            for (int j = 0; j < 256; ++j) v[(size_t) b * 256 + j] = d * (float) ((const int8_t*) (p + 4))[j];
        }
        return v;
    }
    ggml_get_type_traits((ggml_type) type)->to_float(act, v.data(), k);
    return v;
}

struct Expert {
    int tg = 0, td = 0;
    std::string name;
    std::vector<uint8_t> blob;   // gate, up, down at off[3] (256-aligned, as a layer's layout lays them)
    size_t off[3] = {0, 0, 0};
};
static void layout(Expert& x) {
    x.off[0] = 0;
    x.off[1] = al256((size_t) FF * rowb(x.tg, E));
    x.off[2] = 2 * x.off[1];
    x.blob.assign(x.off[2] + al256((size_t) E * rowb(x.td, FF)), 0);
}

// the CPU lane's steps on one expert for nt tokens, over the slice [ff0, ff0 + ffn) (the whole: 0, FF), as
// Impl::cpu_experts_iq takes them (NH = 1): gate / up rows in jobs of RB, h quantized, the down rows
struct Run {
    std::vector<std::vector<float>> h, y;      // [t][FF] (rows of the slice), [t][E]
    std::vector<std::vector<uint8_t>> hq;      // [t] the down's activation
};
static bool run(const Expert& x, const gg::IqCpu& fm, const std::vector<std::vector<uint8_t>>& xa, int ff0, int ffn, Run& o,
                const uint8_t* packed = nullptr, const size_t* poff = nullptr) {
    const int nt = (int) xa.size();
    gg::IqCpu fs;
    size_t d_off = 0;
    std::string err;
    const bool sliced = ff0 != 0 || ffn != FF;
    if (sliced && !gg::iq_cpu_slice(fm, ff0, ffn, fs, d_off, err)) { std::printf("iq_cpu_slice: %s\n", err.c_str()); return false; }
    const gg::IqCpu& fd = sliced ? fs : fm;
    o.h.assign((size_t) nt, std::vector<float>(FF, 0.0f));
    o.y.assign((size_t) nt, std::vector<float>(E, 0.0f));
    o.hq.assign((size_t) nt, std::vector<uint8_t>(fd.h_bytes));
    std::vector<const void*> ap((size_t) nt), hp((size_t) nt);
    std::vector<float*> hr((size_t) nt), yr((size_t) nt);
    for (int t = 0; t < nt; ++t) { ap[(size_t) t] = xa[(size_t) t].data(); hr[(size_t) t] = o.h[(size_t) t].data(); }
    const uint8_t* b = x.blob.data();
    for (int r0 = ff0; r0 < ff0 + ffn; r0 += RB)
        gg::iq_cpu_gate_up(fm, b + x.off[0], b + x.off[1], ap.data(), nt, hr.data(), r0, std::min(ff0 + ffn, r0 + RB), kLim);
    for (int t = 0; t < nt; ++t) {
        gg::iq_cpu_quant(fm.d_act, o.h[(size_t) t].data() + ff0, ffn, o.hq[(size_t) t].data());
        hp[(size_t) t] = o.hq[(size_t) t].data();
        yr[(size_t) t] = o.y[(size_t) t].data();
    }
    if (packed != nullptr) {   // the down over the slice's own blob: packed rows (a format of n_ff = ffn)
        gg::IqCpu fp;
        if (!gg::iq_cpu_fmt(x.tg, x.td, E, ffn, fp, err)) { std::printf("iq_cpu_fmt %d: %s\n", ffn, err.c_str()); return false; }
        for (int r0 = 0; r0 < E; r0 += RB) gg::iq_cpu_down(fp, packed + poff[2], hp.data(), nt, yr.data(), r0, std::min(E, r0 + RB));
        return true;
    }
    for (int r0 = 0; r0 < E; r0 += RB) gg::iq_cpu_down(fd, b + x.off[2] + d_off, hp.data(), nt, yr.data(), r0, std::min(E, r0 + RB));
    return true;
}

static FfSlice slice_of(const Expert& x, int ff0, int ffn) {
    FfSlice s;
    s.dim = E; s.ff0 = ff0; s.ffn = ffn;
    s.gr = rowb(x.tg, E); s.dr = rowb(x.td, FF); s.ds = rowb(x.td, ffn); s.d_off = rowb(x.td, ff0);
    return s;
}

// one expert, nt tokens, each partition of n_ff into slices (block bounds): the checks above
static void check(const Expert& x, int nt, const std::vector<std::vector<int>>& cuts, std::mt19937_64& rng, double& worst_eps, double& worst_rel) {
    gg::IqCpu fm; std::string err;
    if (!gg::iq_cpu_fmt(x.tg, x.td, E, FF, fm, err)) { std::printf("%s: iq_cpu_fmt: %s\n", x.name.c_str(), err.c_str()); ++fails; return; }
    std::vector<std::vector<uint8_t>> xa((size_t) nt, std::vector<uint8_t>(fm.act_bytes));
    for (int t = 0; t < nt; ++t) {
        std::vector<float> xv(E);
        for (auto& v : xv) v = ((float) (rng() % 20001) - 10000.0f) * 1e-4f;
        gg::iq_cpu_quant(fm.gu_act, xv.data(), E, xa[(size_t) t].data());
    }
    Run whole;
    if (!run(x, fm, xa, 0, FF, whole)) { ++fails; return; }
    // the bound's L1 norms: the products of ggml's decode of the down rows and the h as quantized, a row a token
    std::vector<float> dw((size_t) E * FF);
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) x.td);
    for (int r = 0; r < E; ++r) tr->to_float(x.blob.data() + x.off[2] + (size_t) r * rowb(x.td, FF), &dw[(size_t) r * FF], FF);
    std::vector<std::vector<double>> l1((size_t) nt, std::vector<double>(E, 0.0));
    std::vector<double> ymax((size_t) nt, 0.0);
    for (int t = 0; t < nt; ++t) {
        const std::vector<float> hh = act_values(fm.d_act, whole.hq[(size_t) t].data(), FF);
        for (int r = 0; r < E; ++r) {
            double a = 0;
            for (int c = 0; c < FF; ++c) a += std::fabs((double) dw[(size_t) r * FF + c] * hh[(size_t) c]);
            l1[(size_t) t][(size_t) r] = a;
            ymax[(size_t) t] = std::max(ymax[(size_t) t], (double) std::fabs(whole.y[(size_t) t][(size_t) r]));
        }
    }
    for (const std::vector<int>& cut : cuts) {
        std::string what;
        for (size_t i = 0; i + 1 < cut.size(); ++i) what += (i ? "+" : "") + std::string("[") + std::to_string(cut[i]) + "," + std::to_string(cut[i + 1]) + ")";
        std::vector<std::vector<float>> fsum((size_t) nt, std::vector<float>(E, 0.0f));
        bool h_ok = true, q_ok = true, p_ok = true, nan = false;
        const size_t hb = whole.hq[0].size() / NB;   // bytes of the down's activation a 256 values
        for (size_t i = 0; i + 1 < cut.size(); ++i) {
            const int ff0 = cut[i] * 256, ffn = (cut[i + 1] - cut[i]) * 256;
            Run s;
            if (!run(x, fm, xa, ff0, ffn, s)) { ++fails; return; }
            // the slice's own blob (packed down rows), and the down over it
            const FfSlice fs = slice_of(x, ff0, ffn);
            const size_t po[3] = {0, al256((size_t) ffn * fs.gr), 2 * al256((size_t) ffn * fs.gr)};
            std::vector<uint8_t> pk(po[2] + (size_t) E * fs.ds);
            strata::ds4::ff_blob_to_slice(fs, x.blob.data(), x.off, pk.data(), po);
            for (int r = 0; r < 2; ++r)   // (its gate / up rows, which the packed run below does not read)
                p_ok = p_ok && std::memcmp(pk.data() + po[r], x.blob.data() + x.off[r] + (size_t) ff0 * fs.gr, (size_t) ffn * fs.gr) == 0;
            Run p;
            if (!run(x, fm, xa, ff0, ffn, p, pk.data(), po)) { ++fails; return; }
            for (int t = 0; t < nt; ++t) {
                h_ok = h_ok && std::memcmp(s.h[(size_t) t].data() + ff0, whole.h[(size_t) t].data() + ff0, (size_t) ffn * 4) == 0;
                q_ok = q_ok && s.hq[(size_t) t].size() == hb * (size_t) (ffn / 256) &&
                       std::memcmp(s.hq[(size_t) t].data(), whole.hq[(size_t) t].data() + hb * (size_t) (ff0 / 256), s.hq[(size_t) t].size()) == 0;
                p_ok = p_ok && std::memcmp(s.y[(size_t) t].data(), p.y[(size_t) t].data(), (size_t) E * 4) == 0;
                for (int r = 0; r < E; ++r) fsum[(size_t) t][(size_t) r] += s.y[(size_t) t][(size_t) r];
            }
        }
        double we = 0, wr = 0;   // |sum - whole| in eps of the L1 norm, and of the largest output
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < E; ++r) {
                const double d = std::fabs((double) fsum[(size_t) t][(size_t) r] - whole.y[(size_t) t][(size_t) r]);
                nan = nan || !std::isfinite(d);
                we = std::max(we, d / (FLT_EPSILON * l1[(size_t) t][(size_t) r] + 1e-30));
                wr = std::max(wr, d / (ymax[(size_t) t] + 1e-30));
            }
        worst_eps = std::max(worst_eps, we);
        worst_rel = std::max(worst_rel, wr);
        // a sum of 9 block terms a row (each an exact integer sum times a float scale): reassociating them moves it by a few eps of L1
        const bool ok = h_ok && q_ok && p_ok && !nan && we <= 32.0;
        if (!ok || std::getenv("EP_SLICE_VERBOSE"))
            std::printf("  %-30s %-38s nt %d: h %s, h quantized %s, in place == packed %s, |sum - whole| %.2f eps L1 (max %.2e of max|y|)  %s\n",
                        x.name.c_str(), what.c_str(), nt, h_ok ? "same" : "DIFFER", q_ok ? "same" : "DIFFER", p_ok ? "yes" : "NO", we, wr, ok ? "ok" : "FAIL");
        fails += !ok;
    }
}

// blob -> slices -> blob byte for byte (every byte of the three matrices; the alignment gaps are not the slices')
static void round_trip(const Expert& x, const std::vector<int>& cut) {
    std::vector<uint8_t> back(x.blob.size(), 0xEE);
    for (size_t i = 0; i + 1 < cut.size(); ++i) {
        const FfSlice fs = slice_of(x, cut[i] * 256, (cut[i + 1] - cut[i]) * 256);
        const size_t po[3] = {64, 64 + al256((size_t) fs.ffn * fs.gr) + 32, 64 + 2 * al256((size_t) fs.ffn * fs.gr) + 96};
        std::vector<uint8_t> pk(po[2] + (size_t) E * fs.ds, 0x11);
        strata::ds4::ff_blob_to_slice(fs, x.blob.data(), x.off, pk.data(), po);
        strata::ds4::ff_slice_to_blob(fs, pk.data(), po, back.data(), x.off);
    }
    const size_t len[3] = {(size_t) FF * rowb(x.tg, E), (size_t) FF * rowb(x.tg, E), (size_t) E * rowb(x.td, FF)};
    bool ok = true;
    for (int r = 0; r < 3; ++r) ok = ok && std::memcmp(back.data() + x.off[r], x.blob.data() + x.off[r], len[r]) == 0;
    if (!ok) { std::printf("  %-30s round trip over %zu slices: DIFFER  FAIL\n", x.name.c_str(), cut.size() - 1); ++fails; }
}

// a pack's expert e of layer L (tensors.txt: "file N name", "t name TYPE file offset bytes ..."; ds4.txt's model_dir)
static bool read_real(const std::string& dir, int L, int e, Expert& x) {
    static std::map<std::string, std::vector<std::string>> lines;
    std::vector<std::string>& tl = lines[dir];
    if (tl.empty()) { std::ifstream f(dir + "/tensors.txt"); for (std::string s; std::getline(f, s);) tl.push_back(s); }
    std::string mdir;
    { std::ifstream f(dir + "/ds4.txt"); for (std::string s; std::getline(f, s);) if (s.rfind("model_dir ", 0) == 0) mdir = s.substr(10); }
    std::vector<std::string> files;
    const char* w[3] = {"w1", "w3", "w2"};
    std::string ty[3]; int fi[3] = {-1, -1, -1}; uint64_t of[3] = {0, 0, 0}, by[3] = {0, 0, 0};
    for (const std::string& s : tl) {
        std::istringstream is(s);
        std::string k; is >> k;
        if (k == "file") { int n; std::string nm; is >> n >> nm; if ((int) files.size() <= n) files.resize((size_t) n + 1); files[(size_t) n] = nm; continue; }
        if (k != "t") continue;
        std::string nm; is >> nm;
        for (int r = 0; r < 3; ++r)
            if (nm == "layers." + std::to_string(L) + ".ffn.experts." + std::to_string(e) + "." + w[r] + ".weight") is >> ty[r] >> fi[r] >> of[r] >> by[r];
    }
    auto tid = [](const std::string& n) { for (int t = 0; t < GGML_TYPE_COUNT; ++t) { const char* tn = ggml_type_name((ggml_type) t); if (tn && n == tn) return t; } return -1; };
    for (int r = 0; r < 3; ++r) if (fi[r] < 0 || fi[r] >= (int) files.size()) { std::printf("pack: layer %d expert %d %s not found\n", L, e, w[r]); return false; }
    std::string lt[3];
    for (int r = 0; r < 3; ++r) { lt[r] = ty[r]; std::transform(lt[r].begin(), lt[r].end(), lt[r].begin(), ::tolower); }
    x.tg = tid(lt[0]); x.td = tid(lt[2]);
    if (x.tg < 0 || x.td < 0 || ty[0] != ty[1]) { std::printf("pack: layer %d types %s / %s / %s\n", L, ty[0].c_str(), ty[1].c_str(), ty[2].c_str()); return false; }
    x.name = "L" + std::to_string(L) + " e" + std::to_string(e) + " " + ty[0] + "/" + ty[2];
    layout(x);
    const uint64_t want[3] = {(uint64_t) FF * rowb(x.tg, E), (uint64_t) FF * rowb(x.tg, E), (uint64_t) E * rowb(x.td, FF)};
    for (int r = 0; r < 3; ++r) {
        if (by[r] != want[r]) { std::printf("pack: %s %s: %llu bytes, %llu expected\n", x.name.c_str(), w[r], (unsigned long long) by[r], (unsigned long long) want[r]); return false; }
        const std::string& fn = files[(size_t) fi[r]];
        std::ifstream f(fn[0] == '/' ? fn : mdir + "/" + fn, std::ios::binary);
        f.seekg((std::streamoff) of[r]);
        if (!f.read((char*) x.blob.data() + x.off[r], (std::streamsize) by[r])) { std::printf("pack: %s: cannot read %s\n", x.name.c_str(), fn.c_str()); return false; }
    }
    return true;
}

int main(int argc, char** argv) {
    std::vector<int> types = {29, 16, 17}, toks = {1, 2, 3, 6, 8}, layers = {3, 7, 17, 23}, experts = {0, 7};
    std::string pack;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--type" && i + 1 < argc) types = ints(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) toks = ints(argv[++i]);
        else if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--layers" && i + 1 < argc) layers = ints(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) experts = ints(argv[++i]);
        else { std::printf("usage: ep_slice_test [--type 29,16,17] [--tokens 1,2,3,6,8] [--pack DIR --layers 3,7,17,23 --experts 0,7]\n"); return 2; }
    }
    // the experts: every pair of the types (gate / up, down), then the pack's
    std::vector<Expert> xs;
    std::mt19937_64 rng(20261025);
    for (int tg : types)
        for (int td : types) {
            Expert x; x.tg = tg; x.td = td;
            x.name = std::string("random ") + ggml_type_name((ggml_type) tg) + "/" + ggml_type_name((ggml_type) td);
            layout(x);
            fill(rng, tg, FF, E, x.blob.data() + x.off[0]);
            fill(rng, tg, FF, E, x.blob.data() + x.off[1]);
            fill(rng, td, E, FF, x.blob.data() + x.off[2]);
            xs.push_back(std::move(x));
        }
    if (!pack.empty())
        for (int L : layers)
            for (int e : experts) {
                Expert x;
                if (!read_real(pack, L, e, x)) { ++fails; continue; }
                xs.push_back(std::move(x));
            }
    // the partitions: [0, k) + [k, 9) for k = 1..8, three uneven slices, nine one-block slices
    std::vector<std::vector<int>> cuts;
    for (int k = 1; k < NB; ++k) cuts.push_back({0, k, NB});
    cuts.push_back({0, 2, 6, NB});
    cuts.push_back({0, 1, 2, 3, 4, 5, 6, 7, 8, NB});
    // the rows: ggml-cpu's dot, Strata's own rows always (AVX2; the AVX-512 ones where ds4_gg_cpu.cpp's want_512 takes
    // them: Zen 4 / 5), the default rule
    const char* mode_name[3] = {"ggml-cpu dots", "own rows", "default rule"};
    const int modes[3] = {0, 1, 2};
    for (int mi = 0; mi < 3; ++mi) {
        gg::gg_cpu_set_iq256(modes[mi]);
        for (const Expert& x : xs) {
            double we = 0, wr = 0;
            const int f0 = fails;
            for (int nt : toks) check(x, nt, cuts, rng, we, wr);
            std::printf("%-14s %-30s %zu partitions x %zu token counts: |sum of slices - whole| <= %.2f eps L1, max %.2e of max|y|  %s\n",
                        mode_name[mi], x.name.c_str(), cuts.size(), toks.size(), we, wr, fails == f0 ? "ok" : "FAIL");
        }
    }
    gg::gg_cpu_set_iq256(-1);
    // the iq_cpu_slice refusals: not whole 256 blocks, outside n_ff
    {
        gg::IqCpu fm, fs; size_t d = 0; std::string err;
        gg::iq_cpu_fmt(29, 29, E, FF, fm, err);
        const int bad[][2] = {{128, 256}, {0, 128}, {0, 0}, {2048, 512}, {-256, 512}, {256, 2304}};
        int refused = 0;
        for (const auto& b : bad) refused += !gg::iq_cpu_slice(fm, b[0], b[1], fs, d, err);
        const bool whole = gg::iq_cpu_slice(fm, 0, FF, fs, d, err) && d == 0 && fs.d_row == fm.d_row && fs.h_bytes == fm.h_bytes &&
                           fs.d_stride == fm.d_stride && fs.n_ff == FF;
        std::printf("iq_cpu_slice: %d of 6 bad slices refused, the whole one the whole format %s  %s\n", refused, whole ? "yes" : "NO",
                    refused == 6 && whole ? "ok" : "FAIL");
        fails += !(refused == 6 && whole);
    }
    for (const Expert& x : xs)
        for (const auto& c : cuts) round_trip(x, c);
    {   // even thirds of 768 (the form of a tensor parallel member's share; mx_blob_to_half's own experts are MXFP4)
        for (const Expert& x : xs) round_trip(x, {0, 3, 6, 9});
        std::printf("round trips: %zu experts x %zu partitions + thirds, blob -> slices -> blob\n", xs.size(), cuts.size());
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails != 0;
}
