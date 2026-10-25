// Strata. The remote experts (src/dsv41/ds4_rx.hpp) end to end against a real expert server: this program is the main
// process's end (rx::Main, the engine's own code), the server is strata-dsv41 --expert-server-connect on a GPU (the same
// PC or another).  It sends random legal expert blobs of the i-quant types, then requests of T rows, and checks each
// reply against the CPU lane's arithmetic for the same experts and inputs (ds4_iq_cpu.cpp: what the main process would
// have computed itself) by relative error; then times the round trip of T = 1 requests, and checks that a reply later
// than its deadline is skipped (the next request gets its own).  No model, no GPU here.
//   ds4_rx_test [--listen [BIND:]PORT] [--layers 29/29,16/16,17/17,16/17] [--experts N] [--reqs N] [--rtt N] [--picks K]
//               [--ref N (rows also checked against the double reference)]
//               [--rtt-only 1 (no correctness or late-reply checks: a DS4_RX_NULL server's zeros)]
//               [--ping N (then N pings of 56 B and of 10,320 B each way: the link and both ends' sockets alone)]
//               [--fill 1 (as many blobs as the server's room holds, as the engine sends them: --experts N contents
//                each layer, again under more expert ids)]
//               [--bad-geo 1 (layer 0's down 256 bytes before its blob's end: past it - the server refuses at the Geo)]
//               [--early 1 (a refusal must come before the last blob: one at the BlobsEnd is a failure)]
//               [--write-parts DIR [--parts-a A (8)] (no server: the blobs' parts pack of n_ff blocks [A, 9), all layers,
//                pack_hash 5eed00000000abcd, for the server's --expert-server-parts DIR; then exit 0)]
//               [--parts-k K [--parts-reqs N (60)] (H': the server has that pack; requests also of slice picks at k K,
//                checked against the CPU lane's [K, 9) - the two sets of kReqSets apart, kReqWrong's [K - 1, 9); K below
//                the server's lo: the request refused, exit 0)]  [--parts-hash HEX (another pack's: refused at the Geo, 3)]
//               (no --parts-k against a server with parts: the Geo's k 9, pack_hash 0 - taken, the server warns; U4)
//               [--fills N (EP U5b: then N fills - experts --experts .. + N - 1, of their own bytes, each into a held
//                one's slot; with --parts-k only their [0, K) shipped, the server's parts the rest (a --write-parts with
//                the same --fills).  A request naming an evicted one: Stale; the filled ones' replies against the CPU lane)]
//   (STRATA_STAGE_TOKEN: the same as the server's)
// Exit 0 all good, 1 a failure, 3 the server refused the blobs with an Error (the engine drops it at load).  Two cases
// of the server's load: a full room (the server with DS4_RX_TEST_VRAM_MIB=600 --expert-server-reserve-mib 0, this
// with --fill 1): every blob held and served, exit 0; a type with no token kernel (--layers 16/16,16/13: the second
// layer's down Q5_K): refused at load, exit 3.  Either way never a failure at a request.  A refusal at the Geo (the
// room's VRAM, the geometry: --bad-geo 1) stops the blobs at once: --early 1 checks that.
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_rx.hpp"
#include "../../src/dsv41/ds4_rx_fill_link.hpp"
#include "ep_parts_write.hpp"

#include "ggml.h"   // ggml_get_type_traits(type)->to_float: the double reference's weights

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
namespace gg = strata::ds4::gguf;
namespace rx = strata::ds4::rx;

static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
// random legal blocks of `type` (n bytes, whole blocks): any bit pattern with a sane F16 scale is one (iq_expert_test's)
static void fill_blocks(std::mt19937_64& rng, int type, uint8_t* p0, size_t n) {
    const int bb = iq::block_bytes(type);
    for (size_t i = 0; i < n; ++i) p0[i] = (uint8_t) rng();
    for (size_t b = 0; b < n / (size_t) bb; ++b) {
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* p = p0 + b * (size_t) bb;
        if (type == iq::kIQ1_M) {   // (no d field: its F16 scale is the top nibbles of its four u16 scales)
            p[49] = (uint8_t) ((p[49] & 0x0f) | ((d & 0x000f) << 4));
            p[51] = (uint8_t) ((p[51] & 0x0f) | (d & 0x00f0));
            p[53] = (uint8_t) ((p[53] & 0x0f) | ((d & 0x0f00) >> 4));
            p[55] = (uint8_t) ((p[55] & 0x0f) | ((d & 0xf000) >> 8));
        } else {
            std::memcpy(p, &d, 2);
        }
    }
}

struct Layer {
    int tg = 0, td = 0;
    rx::GeoLayer geo;
    gg::IqCpu fm;
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<std::vector<uint8_t>> xblobs;   // (--fills: the experts n_exp .. n_exp + N - 1, sent by fills only)
};

int main(int argc, char** argv) {
    std::string at = "7861", layers_s = "29/29,16/16,17/17,16/17";
    int n_exp = 8, n_req = 200, n_rtt = 1000, picks = 6, n_ref = 24, rtt_only = 0, n_ping = 0, fill = 0, bad_geo = 0,
        early = 0, parts_a = 8, parts_k = 0, parts_reqs = 60, fills = 0;
    std::string parts_dir;
    uint64_t parts_hash = 0x5eed00000000abcdull;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--listen") at = v;
        else if (a == "--layers") layers_s = v;
        else if (a == "--experts") n_exp = std::max(1, std::atoi(v.c_str()));
        else if (a == "--reqs") n_req = std::atoi(v.c_str());
        else if (a == "--rtt") n_rtt = std::atoi(v.c_str());
        else if (a == "--ref") n_ref = std::atoi(v.c_str());
        else if (a == "--rtt-only") rtt_only = std::atoi(v.c_str());
        else if (a == "--ping") n_ping = std::atoi(v.c_str());
        else if (a == "--fill") fill = std::atoi(v.c_str());
        else if (a == "--bad-geo") bad_geo = std::atoi(v.c_str());
        else if (a == "--early") early = std::atoi(v.c_str());
        else if (a == "--write-parts") parts_dir = v;
        else if (a == "--parts-a") parts_a = std::atoi(v.c_str());
        else if (a == "--parts-k") parts_k = std::atoi(v.c_str());
        else if (a == "--parts-reqs") parts_reqs = std::atoi(v.c_str());
        else if (a == "--parts-hash") parts_hash = std::strtoull(v.c_str(), nullptr, 16);
        else if (a == "--fills") fills = std::max(0, std::atoi(v.c_str()));
        else if (a == "--picks") picks = std::max(1, std::min(rx::kMaxPicks, std::atoi(v.c_str())));
        else { std::printf("unknown option %s\n", a.c_str()); return 2; }
    }
    picks = std::min(picks, n_exp);
    const int E = 5120, FF = 2304;
    const float lim = 10.0f;
    std::vector<Layer> L;
    for (size_t p = 0; p < layers_s.size();) {
        const size_t c = layers_s.find(',', p);
        const std::string one = layers_s.substr(p, c == std::string::npos ? std::string::npos : c - p);
        Layer l;
        l.tg = std::atoi(one.c_str());
        l.td = std::atoi(one.c_str() + one.find('/') + 1);
        L.push_back(l);
        p = c == std::string::npos ? layers_s.size() : c + 1;
    }
    std::mt19937_64 rng(42);
    rx::GeoMsg gm;
    gm.n_embd = E;
    gm.n_ff = FF;
    gm.n_layers = (int) L.size();
    gm.limit = lim;
    std::vector<rx::GeoLayer> gl;
    uint64_t all_b = 0;
    for (Layer& l : L) {
        std::string err;
        if (!gg::iq_cpu_fmt(l.tg, l.td, E, FF, l.fm, err)) {
            std::printf("layer types %s / %s: %s\n", iq::name(l.tg), iq::name(l.td), err.c_str());
            return 2;
        }
        const size_t b[3] = {(size_t) FF * l.fm.gu_row, (size_t) FF * l.fm.gu_row, (size_t) E * l.fm.d_row};
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            l.geo.K[r] = r < 2 ? l.tg : l.td;
            l.geo.off[r] = off;
            off += (b[r] + 255) & ~(size_t) 255;
        }
        l.geo.blob = off;
        gl.push_back(l.geo);
        for (int e = 0; e < n_exp; ++e) {
            std::vector<uint8_t> blob(off, 0);
            for (int r = 0; r < 3; ++r) fill_blocks(rng, l.geo.K[r], blob.data() + l.geo.off[r], b[r]);
            l.blobs.push_back(std::move(blob));
        }
    }
    std::mt19937_64 xrng(4242);   // (the fills' experts from their own seed: the others' bytes as without --fills)
    for (Layer& l : L)
        for (int j = 0; j < fills; ++j) {
            std::vector<uint8_t> blob(l.geo.blob, 0);
            for (int r = 0; r < 3; ++r)
                fill_blocks(xrng, l.geo.K[r], blob.data() + l.geo.off[r],
                            r < 2 ? (size_t) FF * l.fm.gu_row : (size_t) E * l.fm.d_row);
            l.xblobs.push_back(std::move(blob));
        }
    // expert e's bytes: its own (a fill's), else content e % n_exp (--fill: more ids than contents)
    auto blob_of = [&](const Layer& l, int e) -> const uint8_t* {
        return e >= n_exp && e - n_exp < (int) l.xblobs.size() ? l.xblobs[(size_t) (e - n_exp)].data()
                                                                : l.blobs[(size_t) (e % n_exp)].data();
    };
    const int NB = FF / 256;
    if (!parts_dir.empty()) {   // (the same blobs each run: seed 42, the same --layers and --experts)
        ep_test::PartsSrc s;
        s.dim = E;
        s.n_ff = FF;
        s.layers = (int) L.size();
        s.experts = n_exp + fills;
        s.mat = [&](int l, int e, int kind) { return blob_of(L[(size_t) l], e) + L[(size_t) l].geo.off[kind]; };
        s.row_bytes = [&](int l, int kind) { return kind < 2 ? L[(size_t) l].fm.gu_row : L[(size_t) l].fm.d_row; };
        s.type = [&](int l, int kind) { return L[(size_t) l].geo.K[kind]; };
        std::vector<int> all;
        for (int l = 0; l < (int) L.size(); ++l) all.push_back(l);
        if (parts_a < 1 || parts_a >= NB || !ep_test::write_parts(parts_dir, s, parts_a, all, parts_hash)) {
            std::printf("FAILED: the parts pack [%d, %d) not written to %s\n", parts_a, NB, parts_dir.c_str());
            return 1;
        }
        std::printf("parts pack: n_ff blocks [%d, %d) of %zu layers x %d experts in %s (pack_hash %016llx)\n", parts_a, NB,
                    L.size(), n_exp + fills, parts_dir.c_str(), (unsigned long long) parts_hash);
        return 0;
    }
    if (parts_k > 0 && (fill || rtt_only || parts_k >= NB)) {
        std::printf("--parts-k: 1 .. %d, and not with --fill or --rtt-only (the parts hold experts 0 .. --experts - 1)\n", NB - 1);
        return 2;
    }
    if (fills > 0 && (fill || rtt_only || n_exp + fills > 1024)) {
        std::printf("--fills: not with --fill or --rtt-only, and --experts + --fills at most 1024\n");
        return 2;
    }
    if (parts_k > 0) gm.pack_hash = parts_hash;
    gm.k = parts_k > 0 ? parts_k : NB;   // (no --parts-k: a main that does not split - parts that do not fit, a warning)
    std::vector<int> sent(L.size(), n_exp);   // (each layer's expert ids 0 .. sent - 1: content id % n_exp)
    const char* tv = std::getenv("STRATA_STAGE_TOKEN");
    rx::Main m;
    std::string err;
    std::printf("waiting for an expert server on %s ...\n", at.c_str());
    std::fflush(stdout);
    if (!m.accept(at, tv != nullptr ? tv : "", 120, err)) {
        std::printf("FAILED: %s\n", err.c_str());
        return 1;
    }
    std::printf("server %s (%s), %.2f GB of VRAM for blobs\n", m.peer().c_str(), m.name().c_str(), (double) m.room() / 1e9);
    if (fill) {   // a blob a layer in turn while the room holds it
        std::fill(sent.begin(), sent.end(), 0);
        for (bool more = true; more;) {
            more = false;
            for (size_t li = 0; li < L.size(); ++li)
                if (all_b + L[li].geo.blob <= m.room() && sent[li] < 65536) {
                    all_b += L[li].geo.blob;
                    ++sent[li];
                    more = true;
                }
        }
        for (size_t li = 0; li < L.size(); ++li)
            if (sent[li] < picks) {
                std::printf("FAILED: the room holds %d blobs of layer %zu, fewer than %d picks\n", sent[li], li, picks);
                return 1;
            }
    } else {
        for (const Layer& l : L) all_b += (uint64_t) n_exp * l.geo.blob;
    }
    if (all_b > m.room()) {
        std::printf("FAILED: %.2f GB of test blobs do not fit\n", (double) all_b / 1e9);
        return 1;
    }
    if (bad_geo) gl[0].off[2] = gl[0].blob - 256;   // (the blobs keep their real layout: only the geometry lies)
    int n_all = 0;
    for (int s : sent) n_all += s;
    const double t0 = rx::now_ms();
    int nb = 0;
    // the server's Error (the engine drops it at load), after nb of the n_all blobs went out
    auto refused = [&](bool at_end) {
        const bool err_msg = err.rfind("expert server: ", 0) == 0;
        if (!err_msg) {
            std::printf("FAILED: %s\n", err.c_str());
            return 1;
        }
        std::printf("REFUSED at load after %d of %d blobs, %.0f ms after the geometry: %s\n", nb, n_all, rx::now_ms() - t0,
                    err.c_str());
        if (early && at_end) {
            std::printf("FAILED: --early: the refusal came only after every blob\n");
            return 1;
        }
        return 3;
    };
    if (!m.send_geo(gm, gl, err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
    for (size_t li = 0; li < L.size(); ++li)
        for (int e = 0; e < sent[li]; ++e, ++nb) {
            const std::vector<uint8_t>& bl = L[li].blobs[(size_t) (e % n_exp)];
            if (!m.send_blob((int) li, e, bl.data(), bl.size(), err)) return refused(false);
        }
    std::vector<uint8_t> ok;
    if (!m.end_blobs(nb, ok, 120000.0, err)) return refused(true);
    const double t_up = rx::now_ms() - t0;
    int held = 0;
    for (uint8_t v : ok) held += v;
    std::printf("upload: %d of %d blobs (%.1f MB) held, %.2f s (%.0f MB/s)\n", held, nb, (double) all_b / 1e6, t_up / 1000.0,
                (double) all_b / 1e3 / t_up);
    std::vector<std::vector<int>> ids_of(L.size());   // (each layer's held expert ids)
    for (size_t li = 0, i = 0; li < L.size(); ++li)
        for (int e = 0; e < sent[li]; ++e, ++i)
            if (ok[i]) ids_of[li].push_back(e);
    int fails = 0;
    if (held != nb) {
        std::printf("FAILED: not every blob held%s\n", fill ? " (--fill: the requests take the held ones, as the engine's)" : "");
        if (!fill) return 1;
        ++fails;
        for (size_t li = 0; li < L.size(); ++li)
            if ((int) ids_of[li].size() < picks) {
                std::printf("FAILED: layer %zu holds %zu blobs, fewer than %d picks\n", li, ids_of[li].size(), picks);
                return 1;
            }
    }

    // the CPU lane's sum for a row: sum_i w_i expert_i(x), x as the server gets it (bf16)
    auto cpu_row = [&](const Layer& l, const rx::ReqRow& rw, const std::vector<float>& xb, std::vector<double>& y) {
        y.assign((size_t) E, 0.0);
        std::vector<uint8_t> act(l.fm.act_bytes), hact(l.fm.h_bytes);
        gg::iq_cpu_quant(l.fm.gu_act, xb.data(), E, act.data());
        std::vector<float> h(FF), yo(E);
        for (int i = 0; i < rw.n; ++i) {
            const uint8_t* b = blob_of(l, rw.e[i]);
            const void* ap[1] = {act.data()};
            float* hp[1] = {h.data()};
            gg::iq_cpu_gate_up(l.fm, b + l.geo.off[0], b + l.geo.off[1], ap, 1, hp, 0, FF, lim);
            gg::iq_cpu_quant(l.fm.d_act, h.data(), FF, hact.data());
            const void* hq[1] = {hact.data()};
            float* yp[1] = {yo.data()};
            gg::iq_cpu_down(l.fm, b + l.geo.off[2], hq, 1, yp, 0, E);
            for (int c = 0; c < E; ++c) y[(size_t) c] += (double) rw.w[i] * yo[(size_t) c];
        }
    };
    // the double reference: ggml's own decode of the weights, x unrounded (as bf16 here), every sum in double
    auto ref_row = [&](const Layer& l, const rx::ReqRow& rw, const std::vector<float>& xb, std::vector<double>& y) {
        y.assign((size_t) E, 0.0);
        const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) l.tg);
        const ggml_type_traits* td = ggml_get_type_traits((ggml_type) l.td);
        std::vector<float> wr((size_t) E);
        std::vector<double> h((size_t) FF);
        for (int i = 0; i < rw.n; ++i) {
            const uint8_t* b = blob_of(l, rw.e[i]);
            for (int r = 0; r < FF; ++r) {
                double gu[2];
                for (int q = 0; q < 2; ++q) {
                    tg->to_float(b + l.geo.off[q] + (size_t) r * l.fm.gu_row, wr.data(), E);
                    double s = 0;
                    for (int c = 0; c < E; ++c) s += (double) wr[(size_t) c] * xb[(size_t) c];
                    gu[q] = s;
                }
                double g = std::min(gu[0], (double) lim), u = std::min(std::max(gu[1], -(double) lim), (double) lim);
                h[(size_t) r] = g / (1.0 + std::exp(-g)) * u;
            }
            for (int c = 0; c < E; ++c) {
                td->to_float(b + l.geo.off[2] + (size_t) c * l.fm.d_row, wr.data(), FF);
                double s = 0;
                for (int j = 0; j < FF; ++j) s += (double) wr[(size_t) j] * h[(size_t) j];
                y[(size_t) c] += (double) rw.w[i] * s;
            }
        }
    };
    auto rel_err = [&](const double* a, const std::vector<double>& ref) {
        double num = 0, den = 0;
        for (int c = 0; c < E; ++c) {
            num += (a[c] - ref[(size_t) c]) * (a[c] - ref[(size_t) c]);
            den += ref[(size_t) c] * ref[(size_t) c];
        }
        return std::sqrt(num / std::max(den, 1e-30));
    };
    auto make_req = [&](int li, int T, int k, std::vector<rx::ReqRow>& rows, std::vector<std::vector<float>>& xs) {
        rows.assign((size_t) T, rx::ReqRow{});
        xs.assign((size_t) T, std::vector<float>((size_t) E));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int t = 0; t < T; ++t) {
            rows[(size_t) t].t = (uint8_t) t;
            rows[(size_t) t].n = (uint8_t) k;
            std::vector<int> ids = ids_of[(size_t) li];
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int i = 0; i < k; ++i) {
                rows[(size_t) t].e[i] = (int16_t) ids[(size_t) i];
                rows[(size_t) t].w[i] = 0.05f + 0.3f * (float) (rng() % 1000) / 1000.0f;
            }
            for (int c = 0; c < E; ++c) xs[(size_t) t][(size_t) c] = rx::bf2f(rx::f2bf(nd(rng)));   // (bf16 already: both sides the same x)
        }
    };
    std::vector<const float*> xp;
    auto ptrs = [&](const std::vector<std::vector<float>>& xs) {
        xp.clear();
        for (const auto& v : xs) xp.push_back(v.data());
        return xp.data();
    };

    // ---- correctness: random layers, T 1..16, 1..picks picks a row.  Each row against the CPU lane's sum (what the
    // main process computes without the server; the two round the activations differently: q8 a 32 values on the
    // device, Q8_K a 256 on the lane), and the first n_ref rows also both against the double reference: the server
    // must be about as close to it as the lane is
    double worst = 0, sum_rel = 0, ref_s = 0, ref_c = 0, ref_s_max = 0, ref_c_max = 0;
    int nrow = 0, n_ref_done = 0;
    uint32_t seq = 1;
    std::vector<rx::ReqRow> rows;
    std::vector<std::vector<float>> xs;
    for (int q = 0; q < (rtt_only ? 0 : n_req); ++q, ++seq) {
        const int li = (int) (rng() % L.size());
        const int T = q % 4 == 0 ? 1 + (int) (rng() % rx::kMaxRows) : 1;
        const int k = 1 + (int) (rng() % (uint64_t) picks);
        make_req(li, T, k, rows, xs);
        if (!m.send_req(seq, li, T, rows.data(), ptrs(xs), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
        const uint16_t* y = nullptr;
        int us = 0;
        const int got = m.wait_resp(seq, T, rx::now_ms() + 5000.0, &y, us, err);
        if (got != 1) { std::printf("FAILED: request %u: %s\n", seq, got == 0 ? "no reply in 5 s" : err.c_str()); return 1; }
        for (int t = 0; t < T; ++t) {
            std::vector<double> lane, ref;
            cpu_row(L[(size_t) li], rows[(size_t) t], xs[(size_t) t], lane);
            std::vector<double> srv((size_t) E);
            for (int c = 0; c < E; ++c) srv[(size_t) c] = (double) rx::bf2f(y[(size_t) t * E + c]);
            const double rel = rel_err(srv.data(), lane);
            worst = std::max(worst, rel);
            sum_rel += rel;
            ++nrow;
            bool bad = !(rel < 0.15);   // (a wrong expert, weight, offset or row is ~1; the two roundings, a few %)
            if (n_ref_done < n_ref) {
                ref_row(L[(size_t) li], rows[(size_t) t], xs[(size_t) t], ref);
                const double es = rel_err(srv.data(), ref), ec = rel_err(lane.data(), ref);
                ref_s += es;
                ref_c += ec;
                ref_s_max = std::max(ref_s_max, es);
                ref_c_max = std::max(ref_c_max, ec);
                ++n_ref_done;
                bad = bad || !(es <= 1.5 * ec + 0.01);
            }
            if (bad) {
                if (fails < 5)
                    std::printf("row: layer %d (%s/%s) T %d row %d %d picks: relative error %.3g to the lane's\n", li,
                                iq::name(L[(size_t) li].tg), iq::name(L[(size_t) li].td), T, t, k, rel);
                ++fails;
            }
        }
    }
    if (!rtt_only) std::printf("correctness: %d requests, %d rows: relative error (L2) of the server's bf16 sums to the CPU lane's mean "
                "%.2e, max %.2e; on %d rows to the double reference: the server mean %.2e max %.2e, the CPU lane mean "
                "%.2e max %.2e -> %s\n", n_req, nrow, nrow ? sum_rel / nrow : 0.0, worst, n_ref_done,
                n_ref_done ? ref_s / n_ref_done : 0.0, ref_s_max, n_ref_done ? ref_c / n_ref_done : 0.0, ref_c_max,
                fails ? "FAIL" : "ok");

    // ---- H' (--parts-k): rows of whole picks (the server's GPU, the blobs) and slice picks (its CPU lane, the parts'
    // [k, 9)), the slice bits at random and some rows all slice (no whole pick: the GPU's rows skipped).  In turn: one
    // set (the reply the sum of both), kReqSets (the whole picks' sums, then the slice picks'), kReqSets | kReqWrong (the
    // slice from k - 1, or [k, 9) twice where the parts hold no block k - 1).  The whole set against the CPU lane's
    // (the GPU's roundings: 0.15, as above); the slice set against the CPU lane's own steps on the same bytes for x in
    // bf16 - the server's arithmetic, within its bf16 reply (1e-2); an empty set exactly 0
    if (parts_k > 0) {
        const int lo = m.lo();
        std::printf("H': the server's parts from n_ff block %d (its CPU lane %d threads), slice picks at k %d\n", lo,
                    m.lane_threads(), parts_k);
        // sum_i w_i (expert_i's n_ff blocks [b0, b1))(x) over the picks i with `mask` set, in pick order
        auto slice_row = [&](const Layer& l, const rx::ReqRow& rw, unsigned mask, int b0, int b1, const std::vector<float>& xb,
                             std::vector<double>& y) {
            y.assign((size_t) E, 0.0);
            const int ff0 = 256 * b0, ffn = 256 * (b1 - b0);
            gg::IqCpu fs;
            size_t d_off = 0;
            std::string e2;
            if (!gg::iq_cpu_slice(l.fm, ff0, ffn, fs, d_off, e2)) return false;
            std::vector<uint8_t> act(l.fm.act_bytes), hact(l.fm.h_bytes);
            std::vector<float> h((size_t) FF), yo((size_t) E);
            gg::iq_cpu_quant(l.fm.gu_act, xb.data(), E, act.data());
            for (int i = 0; i < rw.n; ++i) {
                if (!((mask >> i) & 1)) continue;
                const uint8_t* b = blob_of(l, rw.e[i]);
                const void* ap[1] = {act.data()};
                float* hp[1] = {h.data()};
                gg::iq_cpu_gate_up(l.fm, b + l.geo.off[0], b + l.geo.off[1], ap, 1, hp, ff0, ff0 + ffn, lim);
                gg::iq_cpu_quant(l.fm.d_act, h.data() + ff0, ffn, hact.data());
                const void* hq[1] = {hact.data()};
                float* yp[1] = {yo.data()};
                gg::iq_cpu_down(fs, b + l.geo.off[2] + d_off, hq, 1, yp, 0, E);
                for (int c = 0; c < E; ++c) y[(size_t) c] += (double) rw.w[i] * yo[(size_t) c];
            }
            return true;
        };
        if (parts_k < lo || lo == 0) {   // (the server refuses the request: its Error, the link closed)
            make_req(0, 1, picks, rows, xs);
            rows[0].slice = 1;
            const uint16_t* y = nullptr;
            int us = 0;
            const bool sent_ok = m.send_req(seq, 0, 1, rows.data(), ptrs(xs), err, parts_k);
            const int got = sent_ok ? m.wait_resp(seq, 1, rx::now_ms() + 5000.0, &y, us, err) : -1;
            const bool refused_ok = got == -1 && err.find("expert server: ") != std::string::npos;
            std::printf("H': slice picks at k %d, the server's parts from %d: %s (%s) -> %s\n", parts_k, lo,
                        got == 1 ? "answered" : got == 0 ? "no reply" : "refused", err.c_str(), refused_ok ? "ok" : "FAIL");
            m.close();
            return refused_ok ? 0 : 1;
        }
        double w_whole = 0, w_slice = 0, w_sum = 0, lane_us_sum = 0;
        int n_mode[3] = {}, n_empty_w = 0, n_empty_s = 0, n_rows = 0, n_bad = 0;
        for (int q = 0; q < parts_reqs; ++q, ++seq) {
            const int li = q % (int) L.size();
            const int T = q % 5 == 0 ? rx::kMaxRows : 1 + q % 4;
            const int k = 1 + (int) (rng() % (uint64_t) picks);
            const int mode = q % 3;   // (0 one set, 1 kReqSets, 2 kReqSets | kReqWrong)
            make_req(li, T, k, rows, xs);
            for (int t = 0; t < T; ++t)
                rows[(size_t) t].slice = (uint16_t) ((q % 4 == 1 || t % 3 == 2 ? ~0u : (unsigned) rng()) & ((1u << k) - 1));
            const int32_t c = parts_k | (mode ? rx::kReqSets : 0) | (mode == 2 ? rx::kReqWrong : 0);
            if (!m.send_req(seq, li, T, rows.data(), ptrs(xs), err, c)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
            const uint16_t* y = nullptr;
            int us = 0, lane_us = 0;
            const int sets = mode ? 2 : 1;
            const int got = m.wait_resp(seq, T, rx::now_ms() + 5000.0, &y, us, err, sets, &lane_us);
            if (got != 1) { std::printf("FAILED: H' request %u: %s\n", seq, got == 0 ? "no reply in 5 s" : err.c_str()); return 1; }
            ++n_mode[mode];
            lane_us_sum += lane_us;
            const int kw = mode == 2 && parts_k - 1 >= lo ? parts_k - 1 : parts_k;   // (the wrong slice's start)
            const double twice = mode == 2 && kw == parts_k ? 2.0 : 1.0;
            for (int t = 0; t < T; ++t) {
                const rx::ReqRow& rw = rows[(size_t) t];
                const unsigned all = (1u << rw.n) - 1, sl = rw.slice;
                std::vector<double> wr, sr;
                slice_row(L[(size_t) li], rw, all & ~sl, 0, NB, xs[(size_t) t], wr);
                if (!slice_row(L[(size_t) li], rw, sl, kw, NB, xs[(size_t) t], sr)) { std::printf("FAILED: iq_cpu_slice\n"); return 1; }
                for (double& v : sr) v *= twice;
                std::vector<double> got_w((size_t) E), got_s((size_t) E);
                for (int cc = 0; cc < E; ++cc) {
                    got_w[(size_t) cc] = (double) rx::bf2f(y[(size_t) t * E + cc]);
                    got_s[(size_t) cc] = mode ? (double) rx::bf2f(y[((size_t) T + t) * E + cc]) : 0.0;
                }
                bool bad = false;
                auto zero = [&](const std::vector<double>& v) {
                    for (double d : v)
                        if (d != 0.0) return false;
                    return true;
                };
                ++n_rows;
                if (mode == 0) {   // (the sum of both)
                    std::vector<double> ref((size_t) E);
                    for (int cc = 0; cc < E; ++cc) ref[(size_t) cc] = wr[(size_t) cc] + sr[(size_t) cc];
                    const double r = rel_err(got_w.data(), ref);
                    const double bound = (sl & all) == all ? 1e-2 : 0.15;   // (all slice: the CPU lane's alone)
                    w_sum = std::max(w_sum, (sl & all) == all ? r : 0.0);
                    bad = !(r < bound);
                } else {
                    if ((sl & all) == all) {
                        ++n_empty_w;
                        bad = !zero(got_w);
                    } else {
                        const double r = rel_err(got_w.data(), wr);
                        w_whole = std::max(w_whole, r);
                        bad = !(r < 0.15);
                    }
                    if (sl == 0) {
                        ++n_empty_s;
                        bad = bad || !zero(got_s);
                    } else {
                        const double r = rel_err(got_s.data(), sr);
                        w_slice = std::max(w_slice, r);
                        bad = bad || !(r < 1e-2);
                    }
                }
                if (bad) {
                    if (n_bad < 5)
                        std::printf("H' row: request %u mode %d layer %d T %d row %d %d picks slice 0x%x: wrong\n", seq, mode, li,
                                    T, t, rw.n, sl);
                    ++n_bad;
                }
            }
        }
        std::printf("H': %d requests (%d one set, %d kReqSets, %d kReqSets | kReqWrong from %d%s), %d rows: the whole set to the "
                    "CPU lane's max %.2e (< 0.15), the slice set max %.2e (< 1e-2), all-slice rows' one set max %.2e (< 1e-2); "
                    "%d empty whole sets, %d empty slice sets, each exactly 0; the server's lane %.0f us a request -> %s\n",
                    parts_reqs, n_mode[0], n_mode[1], n_mode[2], parts_k - 1 >= lo ? parts_k - 1 : parts_k,
                    parts_k - 1 >= lo ? "" : ", twice", n_rows, w_whole, w_slice, w_sum, n_empty_w, n_empty_s,
                    parts_reqs ? lane_us_sum / parts_reqs : 0.0, n_bad ? "FAIL" : "ok");
        fails += n_bad;
    }

    // ---- fills (--fills N, EP U5b): the fills' socket, N fills (layer j % layers: expert n_exp + j into the slot of
    // the layer's held expert j / layers), every FillOk 0; then a request naming an evicted expert must be Stale, and
    // requests whose first pick is a filled expert must match the CPU lane's sums of its own bytes (with --parts-k:
    // the server's blob from the shipped [0, K) and its parts' [K, 9))
    if (fills > 0) {
        rx::FillLink fl;
        if (!fl.open(m, at, 0, tv != nullptr ? tv : "", 2e9, -1, 5000, err)) {
            std::printf("FAILED: fills: %s\n", err.c_str());
            return 1;
        }
        const double tf = rx::now_ms();
        std::vector<std::pair<int, int>> filled, evicted;   // (layer, expert)
        uint64_t shipped = 0;
        for (int j = 0; j < fills; ++j) {
            const int li = j % (int) L.size(), slot = j / (int) L.size();
            if (slot >= (int) ids_of[(size_t) li].size() - picks) {
                std::printf("FAILED: fills: layer %d holds too few experts for %d fills\n", li, fills);
                return 1;
            }
            const Layer& l = L[(size_t) li];
            const uint64_t rb[3] = {(uint64_t) l.fm.gu_row, (uint64_t) l.fm.gu_row, (uint64_t) l.fm.d_row};
            rx::FillLink::Job jb;
            jb.seq = (uint32_t) j + 1;
            jb.layer = li;
            jb.in = n_exp + j;
            jb.out = ids_of[(size_t) li][(size_t) slot];
            jb.k = parts_k;
            jb.geo = rx::FillGeo(l.geo, E, FF, rb);
            for (int r = 0; r < 3; ++r) jb.src[r] = blob_of(l, jb.in) + l.geo.off[r];
            shipped += jb.geo.ship_bytes(jb.k);
            fl.queue(jb);
            filled.push_back({li, jb.in});
            evicted.push_back({li, jb.out});
        }
        std::vector<rx::FillLink::Done> dn;
        while ((int) dn.size() < fills && fl.up() && rx::now_ms() - tf < 10000.0) {
            for (const auto& d : fl.take()) dn.push_back(d);
            if ((int) dn.size() < fills) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const double t_fill = rx::now_ms() - tf;
        int bad_ok = 0;
        for (const auto& d : dn) bad_ok += d.status != 0;
        const bool all_in = (int) dn.size() == fills && bad_ok == 0;
        std::printf("fills: %d of %d FillOks (%d not loaded) in %.1f ms, %.1f MB shipped (%s)%s -> %s\n", (int) dn.size(),
                    fills, bad_ok, t_fill, (double) shipped / 1e6, parts_k > 0 ? "the main's [0, K)" : "whole blobs",
                    fl.up() ? "" : (", the socket: " + fl.why()).c_str(), all_in ? "ok" : "FAIL");
        if (!all_in) return 1;
        for (size_t i = 0; i < filled.size(); ++i) {   // (the held ids now: out gone, in there)
            std::vector<int>& id = ids_of[(size_t) filled[i].first];
            id.erase(std::find(id.begin(), id.end(), evicted[i].second));
            id.push_back(filled[i].second);
        }
        // an evicted expert: Stale, not an Error (the link stays)
        make_req(evicted[0].first, 1, 1, rows, xs);
        rows[0].e[0] = (int16_t) evicted[0].second;
        const uint16_t* y = nullptr;
        int us = 0;
        if (!m.send_req(seq, evicted[0].first, 1, rows.data(), ptrs(xs), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
        const int gs = m.wait_resp(seq, 1, rx::now_ms() + 5000.0, &y, us, err);
        ++seq;
        const bool stale_ok = gs == 0 && m.last_stale();
        std::printf("fills: a request naming evicted expert %d of layer %d: %s -> %s\n", evicted[0].second, evicted[0].first,
                    gs == 0 ? (m.last_stale() ? "Stale" : "no reply") : gs == 1 ? "answered" : err.c_str(),
                    stale_ok ? "ok" : "FAIL");
        if (!stale_ok) return 1;
        // the filled experts' replies: each filled one first in a row's picks, T 1 .. 4
        double fworst = 0;
        int fbad = 0, frows = 0;
        for (size_t i = 0; i < filled.size() * 3; ++i, ++seq) {
            const int li = filled[i % filled.size()].first, T = 1 + (int) (i % 4), k = 1 + (int) (i % (size_t) picks);
            make_req(li, T, k, rows, xs);
            for (int t = 0; t < T; ++t) {
                rx::ReqRow& rw = rows[(size_t) t];
                for (int q = 1; q < rw.n; ++q)
                    if (rw.e[q] == filled[i % filled.size()].second) rw.e[q] = rw.e[0];
                rw.e[0] = (int16_t) filled[i % filled.size()].second;
            }
            if (!m.send_req(seq, li, T, rows.data(), ptrs(xs), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
            const int got = m.wait_resp(seq, T, rx::now_ms() + 5000.0, &y, us, err);
            if (got != 1) { std::printf("FAILED: fills: request %u: %s\n", seq, got == 0 ? "no reply" : err.c_str()); return 1; }
            for (int t = 0; t < T; ++t, ++frows) {
                std::vector<double> lane, srv((size_t) E);
                cpu_row(L[(size_t) li], rows[(size_t) t], xs[(size_t) t], lane);
                for (int c = 0; c < E; ++c) srv[(size_t) c] = (double) rx::bf2f(y[(size_t) t * E + c]);
                const double rel = rel_err(srv.data(), lane);
                if (!(rel <= fworst)) fworst = rel;   // (a NaN shows: a slot of the wrong bytes can give one)
                fbad += !(rel < 0.15);
            }
        }
        std::printf("fills: %d rows with a filled expert first: relative error to the CPU lane's max %.2e (< 0.15) -> %s\n",
                    frows, fworst, fbad ? "FAIL" : "ok");
        fails += fbad;
        fl.close();
    }

    // ---- a late reply is skipped: a request whose deadline has passed, then one that must get its own reply.  Two are
    // outstanding then (DS4_EP_DEFER's case: layer l's reply judged late and still out when layer l + 1's request
    // goes): the late one is a whole window (its reply the slower), the next is sent without reading anything, and the
    // late reply - in first, the server answers in order - must be set aside by its seq (take_late), not taken
    if (!rtt_only) {
        (void) m.take_late();
        make_req(0, rx::kMaxRows, picks, rows, xs);
        const uint32_t late = seq++;
        if (!m.send_req(late, 0, rx::kMaxRows, rows.data(), ptrs(xs), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
        const uint16_t* y = nullptr;
        int us = 0;
        const int g1 = m.wait_resp(late, rx::kMaxRows, rx::now_ms() - 1.0, &y, us, err);
        std::vector<rx::ReqRow> rows2;
        std::vector<std::vector<float>> xs2;
        make_req((int) (1 % L.size()), 1, picks, rows2, xs2);
        const uint32_t next = seq++;
        if (!m.send_req(next, (int) (1 % L.size()), 1, rows2.data(), ptrs(xs2), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
        const int g2 = m.wait_resp(next, 1, rx::now_ms() + 5000.0, &y, us, err);
        double num = 0, den = 0;
        if (g2 == 1) {
            std::vector<double> ref;
            cpu_row(L[1 % L.size()], rows2[0], xs2[0], ref);
            for (int c = 0; c < E; ++c) {
                const double d = (double) rx::bf2f(y[c]) - ref[(size_t) c];
                num += d * d;
                den += ref[(size_t) c] * ref[(size_t) c];
            }
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        bool aside = false;   // (the late reply came before the next one's and was set aside by its seq)
        for (const auto& lt : m.take_late()) aside = aside || lt.first == late;
        const bool good = g1 == 0 && g2 == 1 && rel < 0.15 && aside;
        std::printf("late reply (2 outstanding): the first wait %s, the next request's reply %s (relative error %.2e to "
                    "the lane's), the late one %s -> %s\n",
                    g1 == 0 ? "timed out" : "got it anyway (the server beat a 0 ms deadline)", g2 == 1 ? "came" : "did not come",
                    rel, aside ? "set aside by its seq" : "not seen",
                    good ? "ok" : (g1 == 1 && g2 == 1 && rel < 0.15 ? "ok (not exercised)" : "FAIL"));
        if (g2 != 1 || !(rel < 0.15) || (g1 == 0 && !aside)) ++fails;
    }

    // ---- the round trip of T = 1 requests, `picks` picks each (the decode shape), and T = 4
    auto st = [](std::vector<double> v, const char* what) {
        std::sort(v.begin(), v.end());
        double s = 0;
        for (double x : v) s += x;
        std::printf(" %s mean %.3f p50 %.3f p99 %.3f max %.3f ms;", what, s / v.size(), v[v.size() / 2],
                    v[(size_t) (v.size() * 0.99)], v.back());
    };
    for (int T : {1, 4}) {
        if (n_rtt <= 0) break;   // (--rtt 0: correctness only)
        std::vector<double> rt, srv, cs;
        make_req(0, T, picks, rows, xs);
        const int warm = 50;
        for (int q = 0; q < n_rtt + warm; ++q, ++seq) {
            const double a = rx::now_ms();
            const int li = q % (int) L.size();
            const std::vector<int>& id = ids_of[(size_t) li];
            for (int t = 0; t < T; ++t) rows[(size_t) t].e[0] = (int16_t) id[(size_t) q % id.size()];
            for (int t = 0; t < T; ++t)   // (expert ids distinct within the row)
                for (int i = 1; i < picks; ++i) rows[(size_t) t].e[i] = (int16_t) id[(size_t) (q + i) % id.size()];
            if (!m.send_req(seq, li, T, rows.data(), ptrs(xs), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
            const double a1 = rx::now_ms();
            const uint16_t* y = nullptr;
            int us = 0;
            if (m.wait_resp(seq, T, rx::now_ms() + 5000.0, &y, us, err) != 1) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
            if (q >= warm) {
                rt.push_back(rx::now_ms() - a);
                srv.push_back(us / 1000.0);
                cs.push_back(a1 - a);
            }
        }
        std::printf("round trip T=%d, %d picks a row, %d requests:", T, picks, n_rtt);
        st(rt, "send to reply");
        st(srv, "the server's own");
        st(cs, "this end's send (bf16 x and the send call)");
        std::printf(" %.1f KB out, %.1f KB in a request\n", (T * (sizeof(rx::ReqRow) + E * 2.0) + 32) / 1e3, (T * E * 2.0 + 32) / 1e3);
    }
    // ---- pings: the same round trip with nothing done at the server (an echo), 56 B and a T = 1 request's size
    for (size_t bytes : {(size_t) 24, (size_t) 10288}) {
        if (n_ping <= 0) break;
        std::vector<double> rt;
        for (int q = 0; q < n_ping + 50; ++q, ++seq) {
            const double a = rx::now_ms();
            if (m.ping(seq, bytes, a + 5000.0, err) != 1) { std::printf("FAILED: ping: %s\n", err.c_str()); return 1; }
            if (q >= 50) rt.push_back(rx::now_ms() - a);
        }
        std::printf("ping %zu B each way, %d pings:", bytes + 32, n_ping);
        st(rt, "send to echo");
        std::printf("\n");
    }
    m.close();
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
