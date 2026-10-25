// src/dsv41/ds4_rx_cpu.cpp - the expert server of ds4_rx.hpp without a GPU, for tests (--expert-server-cpu): the
// same hello, geometry, blobs and request loop as ds4_rx_server.cu (serve_loop, serve_next), each request's rows through
// the CPU lane's own arithmetic for the i-quant experts (ds4_iq_cpu.cpp: x quantized as the lane does, the clamped
// swiglu, the down rows, the weighted sum in pick order), the blobs in RAM.
#include "ds4_rx.hpp"

#include "ds4_gguf_cpu.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace strata::ds4::rx {
namespace gg = strata::ds4::gguf;

namespace {

struct CpuArg {
    const std::string& token;
};

// one main process: 0 when it went away (or was refused)
int cpu_session(Conn& c, void* arg) {
    const std::string& token = ((const CpuArg*) arg)->token;
    std::string err;
    const char* rv = std::getenv("DS4_RX_CPU_ROOM_MIB");
    HelloMsg me;
    std::memcpy(me.token, token.data(), std::min(token.size(), sizeof me.token - 1));
    me.room = (uint64_t) (rv != nullptr && *rv != '\0' ? std::max(0LL, std::atoll(rv)) : 2048LL) << 20;
    std::snprintf(me.name, sizeof me.name, "CPU (tests)");
    if (!c.send(Hello, 0, 0, 0, 0, &me, sizeof me, err)) return 0;
    Hdr h;
    std::vector<uint8_t> p;
    if (c.recv(h, p, now_ms() + 10000.0, err) != 1 || h.type != Hello || p.size() != sizeof(HelloMsg)) {
        std::fprintf(stderr, "expert server (CPU): no hello from %s (%s)\n", c.peer().c_str(),
                     h.type == Error ? std::string(p.begin(), p.end()).c_str() : err.c_str());
        return 0;
    }
    HelloMsg peer;
    std::memcpy(&peer, p.data(), sizeof peer);
    if (std::strncmp(peer.token, me.token, sizeof me.token) != 0) {
        const char why[] = "wrong token (STRATA_STAGE_TOKEN)";
        (void) c.send(Error, 0, 0, 0, 0, why, sizeof why - 1, err);
        std::fprintf(stderr, "expert server (CPU): %s: %s\n", c.peer().c_str(), why);
        return 0;
    }
    if (c.recv(h, p, -1, err) != 1 || h.type != Geo || p.size() < sizeof(GeoMsg)) {
        std::fprintf(stderr, "expert server (CPU): no geometry (%s)\n", err.c_str());
        return 0;
    }
    GeoMsg g;
    std::memcpy(&g, p.data(), sizeof g);
    if (g.n_embd <= 0 || g.n_embd > 8192 || g.n_ff <= 0 || g.n_ff > 8192 || g.n_layers <= 0 || g.n_layers > 256 ||
        p.size() != sizeof(GeoMsg) + (size_t) g.n_layers * sizeof(GeoLayer)) {
        const char why[] = "a geometry this server does not take";
        (void) c.send(Error, 0, 0, 0, 0, why, sizeof why - 1, err);
        return 0;
    }
    std::vector<GeoLayer> layers((size_t) g.n_layers);
    std::memcpy(layers.data(), p.data() + sizeof(GeoMsg), layers.size() * sizeof(GeoLayer));
    std::vector<gg::IqCpu> fm(layers.size());
    std::vector<bool> takes(layers.size(), false);   // (a layer the CPU lane's i-quant path takes)
    for (size_t li = 0; li < layers.size(); ++li) {
        std::string e2;
        takes[li] = layers[li].K[0] != 0 && layers[li].K[0] == layers[li].K[1] &&
                    gg::iq_cpu_fmt(layers[li].K[0], layers[li].K[2], g.n_embd, g.n_ff, fm[li], e2);
    }
    std::fprintf(stderr, "expert server (CPU): main process %s: %d layers, n_embd %d, n_ff %d; %.2f GB of RAM for blobs\n",
                 c.peer().c_str(), g.n_layers, g.n_embd, g.n_ff, (double) me.room / 1e9);
    std::unordered_map<uint32_t, std::vector<uint8_t>> blobs;   // (layer << 16 | expert)
    uint64_t used = 0;
    std::vector<uint8_t> ok;
    for (;;) {
        if (c.recv(h, p, -1, err) != 1) {
            std::fprintf(stderr, "expert server (CPU): the link went during the blobs (%s)\n", err.c_str());
            return 0;
        }
        if (h.type == BlobsEnd) break;
        if (h.type != Blob) continue;
        const int li = h.a, e = h.b;
        const bool good = li >= 0 && li < g.n_layers && e >= 0 && e < 65536 && takes[(size_t) li] &&
                          p.size() == layers[(size_t) li].blob && used + p.size() <= me.room &&
                          blobs.count((uint32_t) li << 16 | (uint32_t) e) == 0;
        if (good) {
            used += p.size();
            blobs[(uint32_t) li << 16 | (uint32_t) e] = std::move(p);
            p.clear();
        }
        ok.push_back(good ? 1 : 0);
    }
    if (h.a != (int32_t) ok.size()) {
        std::fprintf(stderr, "expert server (CPU): %d blobs said, %zu came\n", h.a, ok.size());
        return 0;
    }
    if (!c.send(BlobsOk, 0, (int32_t) ok.size(), 0, 0, ok.data(), ok.size(), err)) return 0;
    std::fprintf(stderr, "expert server (CPU): %zu blobs (%.2f GB) in RAM; serving\n", blobs.size(), (double) used / 1e9);
    const int D = g.n_embd, FF = g.n_ff;
    std::vector<float> x((size_t) D), hh((size_t) FF), yo((size_t) D), y((size_t) D);
    std::vector<uint8_t> act, hact;
    std::vector<uint16_t> yb((size_t) kMaxRows * D);
    int64_t served = 0, pings = 0, wakes = 0;
    for (;;) {
        if (serve_next(c, h, p, pings, wakes, err) != 1) {
            std::fprintf(stderr, "expert server (CPU): main process %s gone (%s); %lld requests served; %lld Wakes taken, "
                                 "%llu waits in poll, %llu reads quick-ACKed\n",
                         c.peer().c_str(), err.c_str(), (long long) served, (long long) wakes,
                         (unsigned long long) c.polls, (unsigned long long) c.quickacks);
            return 0;
        }
        const double tr = now_ms();
        const int li = h.a, nr = h.b;
        const size_t per = sizeof(ReqRow) + (size_t) D * 2;
        std::string why;
        if (li < 0 || li >= g.n_layers || nr < 1 || nr > kMaxRows || p.size() != (size_t) nr * per) why = "a malformed request";
        for (int j = 0; why.empty() && j < nr; ++j) {
            ReqRow rw;
            std::memcpy(&rw, p.data() + (size_t) j * per, sizeof rw);
            if (rw.n < 1 || rw.n > kMaxPicks) {
                why = "a request row with " + std::to_string(rw.n) + " picks";
                break;
            }
            const gg::IqCpu& f = fm[(size_t) li];
            const uint16_t* xb = (const uint16_t*) (p.data() + (size_t) j * per + sizeof(ReqRow));
            for (int i = 0; i < D; ++i) x[(size_t) i] = bf2f(xb[i]);
            act.resize(f.act_bytes);
            hact.resize(f.h_bytes);
            gg::iq_cpu_quant(f.gu_act, x.data(), D, act.data());
            std::fill(y.begin(), y.end(), 0.0f);
            for (int i = 0; i < rw.n; ++i) {
                auto it = blobs.find((uint32_t) li << 16 | (uint32_t) (uint16_t) rw.e[i]);
                if (it == blobs.end()) {
                    why = "layer " + std::to_string(li) + " expert " + std::to_string(rw.e[i]) + " is not here";
                    break;
                }
                const uint8_t* b = it->second.data();
                const GeoLayer& gl = layers[(size_t) li];
                const void* ap[1] = {act.data()};
                float* hp[1] = {hh.data()};
                gg::iq_cpu_gate_up(f, b + gl.off[0], b + gl.off[1], ap, 1, hp, 0, FF, g.limit);
                gg::iq_cpu_quant(f.d_act, hh.data(), FF, hact.data());
                const void* hq[1] = {hact.data()};
                float* yp[1] = {yo.data()};
                gg::iq_cpu_down(f, b + gl.off[2], hq, 1, yp, 0, D);
                for (int k = 0; k < D; ++k) y[(size_t) k] += rw.w[i] * yo[(size_t) k];
            }
            for (int k = 0; k < D; ++k) yb[(size_t) j * D + k] = f2bf(y[(size_t) k]);
        }
        if (!why.empty()) {
            (void) c.send(Error, h.seq, 0, 0, 0, why.data(), why.size(), err);
            std::fprintf(stderr, "expert server (CPU): %s\n", why.c_str());
            return 0;
        }
        if (!c.send(Resp, h.seq, nr, 0, (int32_t) ((now_ms() - tr) * 1000.0), yb.data(), (size_t) nr * D * 2, err)) {
            std::fprintf(stderr, "expert server (CPU): reply failed (%s)\n", err.c_str());
            return 0;
        }
        ++served;
    }
}

}  // namespace

int serve_cpu(const std::string& host_port, const std::string& token, bool once) {
    if (token.empty()) std::fprintf(stderr, "expert server (CPU): WARNING: no STRATA_STAGE_TOKEN\n");
    CpuArg a{token};
    return serve_loop(host_port, once, "expert server (CPU)", cpu_session, &a);
}

}  // namespace strata::ds4::rx
