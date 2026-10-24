// Strata. The remote experts (src/dsv41/ds4_rx.hpp) end to end against a real expert server: this program is the main
// process's end (rx::Main, the engine's own code), the server is strata-dsv41 --expert-server-connect on a GPU (the same
// PC or another).  It sends random legal expert blobs of the i-quant types, then requests of T rows, and checks each
// reply against the CPU lane's arithmetic for the same experts and inputs (ds4_iq_cpu.cpp: what the main process would
// have computed itself) by relative error; then times the round trip of T = 1 requests, and checks that a reply later
// than its deadline is skipped (the next request gets its own).  No model, no GPU here.
//   ds4_rx_test [--listen [BIND:]PORT] [--layers 29/29,16/16,17/17,16/17] [--experts N] [--reqs N] [--rtt N] [--picks K]
//               [--ref N (rows also checked against the double reference)]
//   (STRATA_STAGE_TOKEN: the same as the server's)
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_rx.hpp"

#include "ggml.h"   // ggml_get_type_traits(type)->to_float: the double reference's weights

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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
};

int main(int argc, char** argv) {
    std::string at = "7861", layers_s = "29/29,16/16,17/17,16/17";
    int n_exp = 8, n_req = 200, n_rtt = 1000, picks = 6, n_ref = 24;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--listen") at = v;
        else if (a == "--layers") layers_s = v;
        else if (a == "--experts") n_exp = std::max(1, std::atoi(v.c_str()));
        else if (a == "--reqs") n_req = std::atoi(v.c_str());
        else if (a == "--rtt") n_rtt = std::atoi(v.c_str());
        else if (a == "--ref") n_ref = std::atoi(v.c_str());
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
            all_b += off;
        }
    }
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
    if (all_b > m.room()) {
        std::printf("FAILED: %.2f GB of test blobs do not fit\n", (double) all_b / 1e9);
        return 1;
    }
    const double t0 = rx::now_ms();
    int nb = 0;
    if (!m.send_geo(gm, gl, err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
    for (size_t li = 0; li < L.size(); ++li)
        for (int e = 0; e < n_exp; ++e, ++nb)
            if (!m.send_blob((int) li, e, L[li].blobs[(size_t) e].data(), L[li].blobs[(size_t) e].size(), err)) {
                std::printf("FAILED: %s\n", err.c_str());
                return 1;
            }
    std::vector<uint8_t> ok;
    if (!m.end_blobs(nb, ok, 120000.0, err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
    const double t_up = rx::now_ms() - t0;
    int held = 0;
    for (uint8_t v : ok) held += v;
    std::printf("upload: %d of %d blobs (%.1f MB) held, %.2f s (%.0f MB/s)\n", held, nb, (double) all_b / 1e6, t_up / 1000.0,
                (double) all_b / 1e3 / t_up);
    if (held != nb) { std::printf("FAILED: not every blob held\n"); return 1; }

    // the CPU lane's sum for a row: sum_i w_i expert_i(x), x as the server gets it (bf16)
    auto cpu_row = [&](const Layer& l, const rx::ReqRow& rw, const std::vector<float>& xb, std::vector<double>& y) {
        y.assign((size_t) E, 0.0);
        std::vector<uint8_t> act(l.fm.act_bytes), hact(l.fm.h_bytes);
        gg::iq_cpu_quant(l.fm.gu_act, xb.data(), E, act.data());
        std::vector<float> h(FF), yo(E);
        for (int i = 0; i < rw.n; ++i) {
            const uint8_t* b = l.blobs[(size_t) rw.e[i]].data();
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
            const uint8_t* b = l.blobs[(size_t) rw.e[i]].data();
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
    auto make_req = [&](int T, int k, std::vector<rx::ReqRow>& rows, std::vector<std::vector<float>>& xs) {
        rows.assign((size_t) T, rx::ReqRow{});
        xs.assign((size_t) T, std::vector<float>((size_t) E));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int t = 0; t < T; ++t) {
            rows[(size_t) t].t = (uint8_t) t;
            rows[(size_t) t].n = (uint8_t) k;
            std::vector<int> ids((size_t) n_exp);
            for (int e = 0; e < n_exp; ++e) ids[(size_t) e] = e;
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
    int fails = 0;
    double worst = 0, sum_rel = 0, ref_s = 0, ref_c = 0, ref_s_max = 0, ref_c_max = 0;
    int nrow = 0, n_ref_done = 0;
    uint32_t seq = 1;
    std::vector<rx::ReqRow> rows;
    std::vector<std::vector<float>> xs;
    for (int q = 0; q < n_req; ++q, ++seq) {
        const int li = (int) (rng() % L.size());
        const int T = q % 4 == 0 ? 1 + (int) (rng() % rx::kMaxRows) : 1;
        const int k = 1 + (int) (rng() % (uint64_t) picks);
        make_req(T, k, rows, xs);
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
    std::printf("correctness: %d requests, %d rows: relative error (L2) of the server's bf16 sums to the CPU lane's mean "
                "%.2e, max %.2e; on %d rows to the double reference: the server mean %.2e max %.2e, the CPU lane mean "
                "%.2e max %.2e -> %s\n", n_req, nrow, nrow ? sum_rel / nrow : 0.0, worst, n_ref_done,
                n_ref_done ? ref_s / n_ref_done : 0.0, ref_s_max, n_ref_done ? ref_c / n_ref_done : 0.0, ref_c_max,
                fails ? "FAIL" : "ok");

    // ---- a late reply is skipped: a request whose deadline has passed, then one that must get its own reply
    {
        make_req(1, picks, rows, xs);
        const uint32_t late = seq++;
        if (!m.send_req(late, 0, 1, rows.data(), ptrs(xs), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
        const uint16_t* y = nullptr;
        int us = 0;
        const int g1 = m.wait_resp(late, 1, rx::now_ms() - 1.0, &y, us, err);
        std::vector<rx::ReqRow> rows2;
        std::vector<std::vector<float>> xs2;
        make_req(1, picks, rows2, xs2);
        const uint32_t next = seq++;
        if (!m.send_req(next, 1, 1, rows2.data(), ptrs(xs2), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
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
        const bool good = g1 == 0 && g2 == 1 && rel < 0.15;
        std::printf("late reply: the first wait %s, the next request's reply %s (relative error %.2e to the lane's) -> %s\n",
                    g1 == 0 ? "timed out" : "got it anyway (the server beat a 0 ms deadline)", g2 == 1 ? "came" : "did not come",
                    rel, good ? "ok" : (g1 == 1 && g2 == 1 && rel < 0.15 ? "ok (not exercised)" : "FAIL"));
        if (g2 != 1 || !(rel < 0.15)) ++fails;
    }

    // ---- the round trip of T = 1 requests, `picks` picks each (the decode shape), and T = 4
    for (int T : {1, 4}) {
        std::vector<double> rt, srv;
        make_req(T, picks, rows, xs);
        const int warm = 50;
        for (int q = 0; q < n_rtt + warm; ++q, ++seq) {
            const double a = rx::now_ms();
            for (int t = 0; t < T; ++t) rows[(size_t) t].e[0] = (int16_t) (q % n_exp);
            const int li = q % (int) L.size();
            for (int t = 0; t < T; ++t)   // (expert ids distinct within the row)
                for (int i = 1; i < picks; ++i) rows[(size_t) t].e[i] = (int16_t) ((q + i) % n_exp);
            if (!m.send_req(seq, li, T, rows.data(), ptrs(xs), err)) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
            const uint16_t* y = nullptr;
            int us = 0;
            if (m.wait_resp(seq, T, rx::now_ms() + 5000.0, &y, us, err) != 1) { std::printf("FAILED: %s\n", err.c_str()); return 1; }
            if (q >= warm) {
                rt.push_back(rx::now_ms() - a);
                srv.push_back(us / 1000.0);
            }
        }
        auto st = [](std::vector<double> v, const char* what) {
            std::sort(v.begin(), v.end());
            double s = 0;
            for (double x : v) s += x;
            std::printf(" %s mean %.3f p50 %.3f p99 %.3f max %.3f ms;", what, s / v.size(), v[v.size() / 2],
                        v[(size_t) (v.size() * 0.99)], v.back());
        };
        std::printf("round trip T=%d, %d picks a row, %d requests:", T, picks, n_rtt);
        st(rt, "send to reply");
        st(srv, "the server's own");
        std::printf(" %.1f KB out, %.1f KB in a request\n", (T * (sizeof(rx::ReqRow) + E * 2.0) + 32) / 1e3, (T * E * 2.0 + 32) / 1e3);
    }
    m.close();
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
