// src/dsv41/ds4_rx_server.cu - the expert server of ds4_rx.hpp: the main process's blobs in this GPU's VRAM, each
// request's rows through the engine's own token-path kernels for VRAM experts (gg::moe_gate_up / moe_down: q8 inputs,
// the i-quant or MXFP4 dots, the clamped swiglu, the weighted sum in pick order), the sums back as bf16.
#include "ds4_rx.hpp"

#include "ds4_gguf.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unordered_map>

namespace strata::ds4::rx {
namespace gg = strata::ds4::gguf;
namespace x3 = strata::dsv41k::exl3;

namespace {

#define RX_CK(call)                                                                                    \
    do {                                                                                               \
        cudaError_t e_ = (call);                                                                       \
        if (e_ != cudaSuccess) {                                                                       \
            std::fprintf(stderr, "expert server: %s: %s\n", #call, cudaGetErrorString(e_));            \
            return -1;                                                                                 \
        }                                                                                              \
    } while (0)

struct Session {
    GeoMsg geo;
    std::vector<GeoLayer> layers;
    std::vector<x3::ExpertLayout> lay;
    std::unordered_map<uint32_t, void*> blobs;   // (layer << 16 | expert) -> its blob in VRAM
    uint64_t used = 0;
    ~Session() {
        for (auto& b : blobs) cudaFree(b.second);
    }
};

// the buffers of one request (kMaxRows rows), host pinned and device
struct Bufs {
    unsigned long long* h_plan = nullptr;   // [kMaxRows][8]
    float* h_w = nullptr;                   // [kMaxRows][8]
    float* h_x = nullptr;                   // [kMaxRows][n_embd]
    float* h_y = nullptr;
    unsigned long long* d_plan = nullptr;
    float *d_w = nullptr, *d_x = nullptr, *d_h = nullptr, *d_y = nullptr;
    size_t stage = 0;   // bytes of plan + w + x, one copy
};

// one main process: 0 when it went away, -1 on a CUDA error
int session(Conn& c, const std::string& token, int64_t reserve_mib, int dev, cudaStream_t s) {
    std::string err;
    size_t fr = 0, tot = 0;
    RX_CK(cudaMemGetInfo(&fr, &tot));
    HelloMsg me;
    std::memcpy(me.token, token.data(), std::min(token.size(), sizeof me.token - 1));
    me.room = fr > (uint64_t) reserve_mib << 20 ? fr - ((uint64_t) reserve_mib << 20) : 0;
    cudaDeviceProp pr{};
    cudaGetDeviceProperties(&pr, dev);
    std::snprintf(me.name, sizeof me.name, "%s", pr.name);
    if (!c.send(Hello, 0, 0, 0, 0, &me, sizeof me, err)) return 0;
    Hdr h;
    std::vector<uint8_t> p;
    if (c.recv(h, p, now_ms() + 10000.0, err) != 1 || h.type != Hello || p.size() != sizeof(HelloMsg)) {
        std::fprintf(stderr, "expert server: no hello from %s (%s)\n", c.peer().c_str(),
                     h.type == Error ? std::string(p.begin(), p.end()).c_str() : err.c_str());
        return 0;
    }
    HelloMsg peer;
    std::memcpy(&peer, p.data(), sizeof peer);
    if (std::strncmp(peer.token, me.token, sizeof me.token) != 0) {
        const char why[] = "wrong token (STRATA_STAGE_TOKEN)";
        (void) c.send(Error, 0, 0, 0, 0, why, sizeof why - 1, err);
        std::fprintf(stderr, "expert server: %s: %s\n", c.peer().c_str(), why);
        return 0;
    }
    Session ss;
    if (c.recv(h, p, -1, err) != 1 || h.type != Geo || p.size() < sizeof(GeoMsg)) {
        std::fprintf(stderr, "expert server: no geometry (%s)\n", err.c_str());
        return 0;
    }
    std::memcpy(&ss.geo, p.data(), sizeof ss.geo);
    const GeoMsg& g = ss.geo;
    if (g.n_embd <= 0 || g.n_embd > 8192 || g.n_ff <= 0 || g.n_ff > 8192 || g.n_layers <= 0 || g.n_layers > 256 ||
        p.size() != sizeof(GeoMsg) + (size_t) g.n_layers * sizeof(GeoLayer)) {
        const char why[] = "a geometry this server does not take";
        (void) c.send(Error, 0, 0, 0, 0, why, sizeof why - 1, err);
        return 0;
    }
    ss.layers.resize((size_t) g.n_layers);
    std::memcpy(ss.layers.data(), p.data() + sizeof(GeoMsg), ss.layers.size() * sizeof(GeoLayer));
    ss.lay.resize(ss.layers.size());
    for (size_t li = 0; li < ss.layers.size(); ++li) {
        x3::ExpertLayout& l = ss.lay[li];
        l.fmt = gg::kFmtMXFP4;
        for (int r = 0; r < 3; ++r) {
            l.K[r] = ss.layers[li].K[r];
            l.trellis[r] = l.suh[r] = l.svh[r] = (size_t) ss.layers[li].off[r];
        }
    }
    std::fprintf(stderr, "expert server: main process %s: %d layers, n_embd %d, n_ff %d; %.2f GB of VRAM for blobs\n",
                 c.peer().c_str(), g.n_layers, g.n_embd, g.n_ff, (double) me.room / 1e9);
    // the blobs, once
    std::vector<uint8_t> ok;
    const double t0 = now_ms();
    for (;;) {
        if (c.recv(h, p, -1, err) != 1) {
            std::fprintf(stderr, "expert server: the link went during the blobs (%s)\n", err.c_str());
            return 0;
        }
        if (h.type == BlobsEnd) break;
        if (h.type != Blob) continue;
        const int li = h.a, e = h.b;
        bool good = li >= 0 && li < g.n_layers && e >= 0 && e < 65536 && ss.layers[(size_t) li].K[0] != 0 &&
                    p.size() == ss.layers[(size_t) li].blob && ss.used + p.size() <= me.room &&
                    ss.blobs.count((uint32_t) li << 16 | (uint32_t) e) == 0;
        void* d = nullptr;
        if (good && cudaMalloc(&d, p.size()) != cudaSuccess) {
            cudaGetLastError();
            good = false;
        }
        if (good) {
            RX_CK(cudaMemcpy(d, p.data(), p.size(), cudaMemcpyHostToDevice));
            ss.blobs[(uint32_t) li << 16 | (uint32_t) e] = d;
            ss.used += p.size();
        }
        ok.push_back(good ? 1 : 0);
    }
    if (h.a != (int32_t) ok.size()) {
        std::fprintf(stderr, "expert server: %d blobs said, %zu came\n", h.a, ok.size());
        return 0;
    }
    if (!c.send(BlobsOk, 0, (int32_t) ok.size(), 0, 0, ok.data(), ok.size(), err)) return 0;
    std::fprintf(stderr, "expert server: %zu blobs (%.2f GB) in VRAM in %.1f s; serving\n", ss.blobs.size(),
                 (double) ss.used / 1e9, (now_ms() - t0) / 1000.0);
    // the requests
    const int D = g.n_embd, FF = g.n_ff;
    Bufs b;
    b.stage = (size_t) kMaxRows * 8 * 8 + (size_t) kMaxRows * 8 * 4 + (size_t) kMaxRows * D * 4;
    void* hs = nullptr;
    RX_CK(cudaHostAlloc(&hs, b.stage + (size_t) kMaxRows * D * 4, cudaHostAllocDefault));
    b.h_plan = (unsigned long long*) hs;
    b.h_w = (float*) (b.h_plan + kMaxRows * 8);
    b.h_x = b.h_w + kMaxRows * 8;
    b.h_y = b.h_x + (size_t) kMaxRows * D;
    void* ds = nullptr;
    RX_CK(cudaMalloc(&ds, b.stage + (size_t) kMaxRows * D * 4 + (size_t) 8 * FF * 4));
    b.d_plan = (unsigned long long*) ds;
    b.d_w = (float*) (b.d_plan + kMaxRows * 8);
    b.d_x = b.d_w + kMaxRows * 8;
    b.d_y = b.d_x + (size_t) kMaxRows * D;
    b.d_h = b.d_y + (size_t) kMaxRows * D;
    std::vector<uint16_t> yb((size_t) kMaxRows * D);
    int rc = 0;
    int64_t served = 0;
    double busy = 0;
    const int errs0 = gg::launch_errors();
    for (;;) {
        if (c.recv(h, p, -1, err) != 1) {
            std::fprintf(stderr, "expert server: main process %s gone (%s); %lld requests served, %.1f us each\n",
                         c.peer().c_str(), err.c_str(), (long long) served, served > 0 ? busy * 1000.0 / served : 0.0);
            break;
        }
        if (h.type != Req) continue;
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
            for (int i = 0; i < kMaxPicks; ++i) {
                void* d = nullptr;
                if (i < rw.n) {
                    auto it = ss.blobs.find((uint32_t) li << 16 | (uint32_t) (uint16_t) rw.e[i]);
                    if (it == ss.blobs.end()) {
                        why = "layer " + std::to_string(li) + " expert " + std::to_string(rw.e[i]) + " is not here";
                        break;
                    }
                    d = it->second;
                }
                b.h_plan[j * 8 + i] = (unsigned long long) d;
                b.h_w[j * 8 + i] = i < rw.n ? rw.w[i] : 0.0f;
            }
            const uint16_t* xb = (const uint16_t*) (p.data() + (size_t) j * per + sizeof(ReqRow));
            float* xf = b.h_x + (size_t) j * D;
            for (int i = 0; i < D; ++i) xf[i] = bf2f(xb[i]);
        }
        if (!why.empty()) {
            (void) c.send(Error, h.seq, 0, 0, 0, why.data(), why.size(), err);
            std::fprintf(stderr, "expert server: %s\n", why.c_str());
            break;
        }
        RX_CK(cudaMemcpyAsync(b.d_plan, b.h_plan, (size_t) kMaxRows * 8 * 8 + (size_t) kMaxRows * 8 * 4, cudaMemcpyHostToDevice, s));
        RX_CK(cudaMemcpyAsync(b.d_x, b.h_x, (size_t) nr * D * 4, cudaMemcpyHostToDevice, s));
        for (int j = 0; j < nr; ++j) {
            ReqRow rw;
            std::memcpy(&rw, p.data() + (size_t) j * per, sizeof rw);
            x3::MoeArgs ma;
            ma.plan_ptr = b.d_plan + j * 8;
            ma.plan_w = b.d_w + j * 8;
            ma.lay = ss.lay[(size_t) li];
            ma.k = rw.n;
            ma.n_embd = D;
            ma.n_ff = FF;
            ma.limit = g.limit;
            gg::moe_gate_up(ma, b.d_x + (size_t) j * D, b.d_h, s);
            gg::moe_down(ma, b.d_h, nullptr, b.d_y + (size_t) j * D, s);
        }
        RX_CK(cudaMemcpyAsync(b.h_y, b.d_y, (size_t) nr * D * 4, cudaMemcpyDeviceToHost, s));
        RX_CK(cudaStreamSynchronize(s));
        if (gg::launch_errors() != errs0) {
            const char w2[] = "a kernel launch failed on the expert server";
            (void) c.send(Error, h.seq, 0, 0, 0, w2, sizeof w2 - 1, err);
            rc = -1;
            break;
        }
        for (size_t i = 0; i < (size_t) nr * D; ++i) yb[i] = f2bf(b.h_y[i]);
        const double dt = now_ms() - tr;
        if (!c.send(Resp, h.seq, nr, 0, (int32_t) (dt * 1000.0), yb.data(), (size_t) nr * D * 2, err)) {
            std::fprintf(stderr, "expert server: reply failed (%s)\n", err.c_str());
            break;
        }
        busy += dt;
        ++served;
    }
    cudaStreamSynchronize(s);
    cudaFree(ds);
    cudaFreeHost(hs);
    return rc;
}

}  // namespace

int serve(const std::string& host_port, const std::string& token, int64_t reserve_mib, int dev, bool once) {
    RX_CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    RX_CK(cudaSetDevice(dev));
    cudaStream_t s = nullptr;
    RX_CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    if (!gg::prepare(s)) return 1;
    if (token.empty()) std::fprintf(stderr, "expert server: WARNING: no STRATA_STAGE_TOKEN\n");
    std::fprintf(stderr, "expert server: GPU %d, dialing %s (again each second until a main process listens)\n", dev,
                 host_port.c_str());
    bool said = false;
    for (;;) {
        Conn c;
        std::string err;
        if (!c.dial(host_port, 2, err)) {
            if (!said) std::fprintf(stderr, "expert server: %s; trying again each second\n", err.c_str());
            said = true;
            std::this_thread::sleep_for(std::chrono::seconds(1));   // (no main process listening: nothing to spin for)
            continue;
        }
        said = false;
        std::fprintf(stderr, "expert server: connected to %s\n", host_port.c_str());
        const int rc = session(c, token, reserve_mib, dev, s);
        c.close();
        if (rc < 0) return 1;
        if (once) return 0;
    }
}

}  // namespace strata::ds4::rx
