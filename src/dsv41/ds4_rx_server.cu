// src/dsv41/ds4_rx_server.cu - the expert server of ds4_rx.hpp: the main process's blobs in this GPU's VRAM, each
// request's rows through the engine's own token-path kernels for VRAM experts (gg::moe_gate_up / moe_down: q8 inputs,
// the i-quant or MXFP4 dots, the clamped swiglu, the weighted sum in pick order), the sums back as bf16.
#include "ds4_rx.hpp"

#include "ds4_gguf.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
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

// the wall clock as HH:MM:SS.mmm UTC (a slow request's line, to match with the main's late replies)
std::string utc_now() {
    const auto t = std::chrono::system_clock::now();
    const std::time_t s = std::chrono::system_clock::to_time_t(t);
    const int ms = (int) (std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count() % 1000);
    std::tm u{};
#if defined(_WIN32)
    gmtime_s(&u, &s);
#else
    gmtime_r(&s, &u);
#endif
    char b[32];
    std::snprintf(b, sizeof b, "%02d:%02d:%02d.%03d", u.tm_hour, u.tm_min, u.tm_sec, ms);
    return b;
}

constexpr int kMaxDim = 8192;   // (a geometry's n_embd and n_ff at most: the request buffers are this size)

// a main process's session: its geometry, its blobs, the request buffers (all freed with it)
struct Session {
    GeoMsg geo;
    std::vector<GeoLayer> layers;
    std::vector<x3::ExpertLayout> lay;
    std::unordered_map<uint32_t, void*> blobs;   // (layer << 16 | expert) -> its blob in the arena
    // the blobs' VRAM: one cudaMalloc of the room, each blob at a 256 B boundary in it (a cudaMalloc a blob took about
    // 1 MiB more each on the 3080: past the reserve, the request's own memory failed after the BlobsOk)
    void* arena = nullptr;
    uint64_t cap = 0, used = 0;   // (the arena's bytes; the end of its last blob)
    void* hs = nullptr;           // the request buffers, pinned host and device (kMaxDim), taken before the room is read
    void* ds = nullptr;
    void* hm = nullptr;           // (DS4_RX_COPY=map: x, then y, mapped)
    cudaEvent_t evt[3 + 2 * kMaxRows + 1] = {};   // (DS4_RX_EVENTS)
    // (DS4_RX_TEST_VRAM_MIB=N, tests: this session's cudaMallocs as on a GPU with N MiB free, each rounded up to 2 MiB
    // as this PC's driver rounds them, out of memory past it; the room is read from it.  WDDM's own cudaMalloc goes
    // past the free into host memory, so on Windows only this can run out)
    uint64_t budget = 0, spent = 0;
    cudaError_t alloc(void** p, size_t n) {
        const uint64_t r = (n + (2u << 20) - 1) / (2u << 20) * (2u << 20);
        if (budget > 0 && spent + r > budget) return cudaErrorMemoryAllocation;
        const cudaError_t e = cudaMalloc(p, n);
        if (e == cudaSuccess) spent += r;
        return e;
    }
    ~Session() {
        for (cudaEvent_t e : evt)
            if (e != nullptr) cudaEventDestroy(e);
        if (arena != nullptr) cudaFree(arena);
        if (ds != nullptr) cudaFree(ds);
        if (hs != nullptr) cudaFreeHost(hs);
        if (hm != nullptr) cudaFreeHost(hm);
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

enum class Copy { Base, Nr, One, Map };   // (DS4_RX_COPY: how a request's inputs reach the GPU)

// a request of one pick a layer format (the roles' types) on a blob of it, through the requests' kernels and copies,
// before the BlobsOk: what a first request would meet (a kernel that does not load or launch, no memory left past the
// room) the main then sees at load.  "" or why it failed
std::string warm_up(const Session& ss, const Bufs& b, cudaStream_t s) {
    const int D = ss.geo.n_embd;
    const int errs0 = gg::launch_errors();
    std::vector<float> x((size_t) D, 1.0f / 64.0f), y((size_t) D);
    std::vector<size_t> done;   // (layers whose format had its request)
    for (size_t li = 0; li < ss.layers.size(); ++li) {
        const GeoLayer& L = ss.layers[li];
        bool seen = false;
        for (size_t d : done) seen = seen || std::equal(L.K, L.K + 3, ss.layers[d].K);
        if (seen) continue;
        void* blob = nullptr;
        for (const auto& kv : ss.blobs)
            if ((kv.first >> 16) == (uint32_t) li) {
                blob = kv.second;
                break;
            }
        if (blob == nullptr) continue;
        done.push_back(li);
        const unsigned long long plan[8] = {(unsigned long long) blob};
        const float w[8] = {1.0f};
        cudaError_t e = cudaMemcpyAsync(b.d_plan, plan, sizeof plan, cudaMemcpyHostToDevice, s);
        if (e == cudaSuccess) e = cudaMemcpyAsync(b.d_w, w, sizeof w, cudaMemcpyHostToDevice, s);
        if (e == cudaSuccess) e = cudaMemcpyAsync(b.d_x, x.data(), (size_t) D * 4, cudaMemcpyHostToDevice, s);
        if (e == cudaSuccess) {
            x3::MoeArgs ma;
            ma.plan_ptr = b.d_plan;
            ma.plan_w = b.d_w;
            ma.lay = ss.lay[li];
            ma.k = 1;
            ma.n_embd = D;
            ma.n_ff = ss.geo.n_ff;
            ma.limit = ss.geo.limit;
            gg::moe_gate_up(ma, b.d_x, b.d_h, s);
            gg::moe_down(ma, b.d_h, nullptr, b.d_y, s);
            e = cudaMemcpyAsync(y.data(), b.d_y, (size_t) D * 4, cudaMemcpyDeviceToHost, s);
        }
        const cudaError_t es = cudaStreamSynchronize(s);
        if (e == cudaSuccess) e = es;
        if (e != cudaSuccess || gg::launch_errors() != errs0) {
            cudaGetLastError();
            return "the warm-up request of layer " + std::to_string(li) + " (" + gg::type_name(L.K[0]) + " / " +
                   gg::type_name(L.K[1]) + " / " + gg::type_name(L.K[2]) + ") failed: " +
                   (e != cudaSuccess ? cudaGetErrorString(e) : "a kernel did not launch (the server's log says which)");
        }
    }
    return "";
}

// one main process: 0 when it went away, -1 on a CUDA error
int session(Conn& c, const std::string& token, int64_t reserve_mib, int dev, cudaStream_t s) {
    std::string err;
    Session ss;
    // the requests: the rows' plan, weights and x packed, one copy in (DS4_RX_COPY=one, the default: 0.02-0.05 ms less a
    // request than base's two on the 3070).  Diagnostics (env, read at each session):
    //   DS4_RX_NULL=1    a reply of zeros at once: no prep, no CUDA call (the link and both ends' sockets alone)
    //   DS4_RX_COPY=base all kMaxRows rows' plan and weights, then x: two copies (the PoC's first path)
    //               nr   the plan and weights of the request's rows only, packed, then x: two copies
    //               map  the plan and weights packed (one copy); x read and y written by the kernels in mapped host
    //                    memory (no x or y copy)
    //   DS4_RX_EVENTS=1  cudaEvents between the GPU parts (plan in, x in, gate_up, down, y out), printed with the host
    //                    parts' at the session's end
    //   DS4_RX_TEST_VRAM_MIB=N  (tests) the session's VRAM as N MiB free (Session::budget)
    const char* nv = std::getenv("DS4_RX_NULL");
    const bool null_reply = nv != nullptr && *nv == '1';
    const char* cv = std::getenv("DS4_RX_COPY");
    const std::string cm = cv != nullptr && *cv != '\0' ? cv : "one";
    const Copy cp = cm == "nr" ? Copy::Nr : cm == "base" ? Copy::Base : cm == "map" ? Copy::Map : Copy::One;
    const char* ev = std::getenv("DS4_RX_EVENTS");
    const bool evs = ev != nullptr && *ev == '1';
    const bool diag = null_reply || evs || cp != Copy::One;
    const char* bv = std::getenv("DS4_RX_TEST_VRAM_MIB");
    if (bv != nullptr && *bv != '\0') {
        ss.budget = (uint64_t) std::max(0LL, std::atoll(bv)) << 20;
        std::fprintf(stderr, "expert server: test: %s MiB of VRAM (DS4_RX_TEST_VRAM_MIB)\n", bv);
    }
    // the request buffers (kMaxDim: before the geometry) and events, so that the room read next is the blobs' alone (the
    // kernels are loaded: serve())
    const size_t stage = (size_t) kMaxRows * 8 * 8 + (size_t) kMaxRows * 8 * 4 + (size_t) kMaxRows * kMaxDim * 4;
    RX_CK(cudaHostAlloc(&ss.hs, stage + (size_t) kMaxRows * kMaxDim * 4, cudaHostAllocDefault));
    RX_CK(ss.alloc(&ss.ds, stage + (size_t) kMaxRows * kMaxDim * 4 + (size_t) 8 * kMaxDim * 4));
    float* dm = nullptr;
    if (cp == Copy::Map) {
        RX_CK(cudaHostAlloc(&ss.hm, (size_t) 2 * kMaxRows * kMaxDim * 4, cudaHostAllocMapped));
        RX_CK(cudaHostGetDevicePointer((void**) &dm, ss.hm, 0));
    }
    if (evs)
        for (cudaEvent_t& e : ss.evt) RX_CK(cudaEventCreate(&e));
    size_t fr = 0, tot = 0;
    RX_CK(cudaMemGetInfo(&fr, &tot));
    if (ss.budget > 0) fr = std::min<size_t>(fr, ss.budget - ss.spent);
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
    if (c.recv(h, p, -1, err) != 1 || h.type != Geo || p.size() < sizeof(GeoMsg)) {
        std::fprintf(stderr, "expert server: no geometry (%s)\n", err.c_str());
        return 0;
    }
    std::memcpy(&ss.geo, p.data(), sizeof ss.geo);
    const GeoMsg& g = ss.geo;
    if (g.n_embd <= 0 || g.n_embd > kMaxDim || g.n_ff <= 0 || g.n_ff > kMaxDim || g.n_layers <= 0 || g.n_layers > 256 ||
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
    // the blobs, once, into the arena: the room in one piece
    std::string refuse;   // (not empty: the BlobsOk is an Error with it, and the main drops this server at load)
    const cudaError_t ea = me.room > 0 ? ss.alloc(&ss.arena, me.room) : cudaSuccess;
    if (ea != cudaSuccess) {
        cudaGetLastError();
        char w[160];
        std::snprintf(w, sizeof w, "the room's %.2f GB of VRAM in one piece failed (%s): a bigger --expert-server-reserve-mib",
                      (double) me.room / 1e9, cudaGetErrorString(ea));
        refuse = w;
        ss.arena = nullptr;
    }
    ss.cap = ss.arena != nullptr ? me.room : 0;
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
        const uint64_t at = (ss.used + 255) / 256 * 256;
        const bool good = li >= 0 && li < g.n_layers && e >= 0 && e < 65536 && ss.layers[(size_t) li].K[0] != 0 &&
                          p.size() == ss.layers[(size_t) li].blob && at + p.size() <= ss.cap &&
                          ss.blobs.count((uint32_t) li << 16 | (uint32_t) e) == 0;
        if (good) {
            void* d = (uint8_t*) ss.arena + at;
            RX_CK(cudaMemcpy(d, p.data(), p.size(), cudaMemcpyHostToDevice));
            ss.blobs[(uint32_t) li << 16 | (uint32_t) e] = d;
            ss.used = at + p.size();
        }
        ok.push_back(good ? 1 : 0);
    }
    if (h.a != (int32_t) ok.size()) {
        std::fprintf(stderr, "expert server: %d blobs said, %zu came\n", h.a, ok.size());
        return 0;
    }
    const int D = g.n_embd, FF = g.n_ff;
    if (diag)
        std::fprintf(stderr, "expert server: diagnostics: %s, copies %s, events %s\n", null_reply ? "NULL replies" : "replies",
                     cm.c_str(), evs ? "on" : "off");
    Bufs b;   // (the request buffers laid out for this geometry)
    b.stage = (size_t) kMaxRows * 8 * 8 + (size_t) kMaxRows * 8 * 4 + (size_t) kMaxRows * D * 4;
    void* const hs = ss.hs;
    b.h_plan = (unsigned long long*) hs;
    b.h_w = (float*) (b.h_plan + kMaxRows * 8);
    b.h_x = b.h_w + kMaxRows * 8;
    b.h_y = b.h_x + (size_t) kMaxRows * D;
    void* const ds = ss.ds;
    b.d_plan = (unsigned long long*) ds;
    b.d_w = (float*) (b.d_plan + kMaxRows * 8);
    b.d_x = b.d_w + kMaxRows * 8;
    b.d_y = b.d_x + (size_t) kMaxRows * D;
    b.d_h = b.d_y + (size_t) kMaxRows * D;
    void* const hm = ss.hm;
    if (refuse.empty()) refuse = warm_up(ss, b, s);
    if (!refuse.empty()) {
        (void) c.send(Error, 0, 0, 0, 0, refuse.data(), refuse.size(), err);
        std::fprintf(stderr, "expert server: %s; the main process drops this server\n", refuse.c_str());
        return 0;
    }
    if (!c.send(BlobsOk, 0, (int32_t) ok.size(), 0, 0, ok.data(), ok.size(), err)) return 0;
    std::fprintf(stderr, "expert server: %zu blobs (%.2f GB) in VRAM in %.1f s, their warm-up done; serving\n",
                 ss.blobs.size(), (double) ss.used / 1e9, (now_ms() - t0) / 1000.0);
    cudaEvent_t* const evt = ss.evt;
    enum { kPrep, kEnq, kSync, kPost, kAll, kPlanIn, kXIn, kGateUp, kDown, kYOut, kSpan, kParts };
    static const char* const part_name[kParts] = {"host: the rows' prep", "host: the copies' and kernels' enqueue",
                                                  "host: the sync's wait", "host: bf16 and send", "host: the whole request",
                                                  "GPU: plan in", "GPU: x in", "GPU: gate_up (q8 + dots)",
                                                  "GPU: down (q8 + dots + combine)", "GPU: y out", "GPU: first copy to last"};
    std::vector<float> parts[3][kParts];   // (by the request's rows: T = 1, T = 4, any other T)
    int bucket = 0;
    auto keep = [&](int k, double ms) {
        if (parts[bucket][k].size() < (size_t(1) << 20)) parts[bucket][k].push_back((float) (ms * 1000.0));
    };
    std::vector<uint16_t> yb((size_t) kMaxRows * D);
    int rc = 0;
    int64_t served = 0, pings = 0, wakes = 0, slow = 0;   // (slow: requests over 2 ms here)
    double last_end = -1;
    double busy = 0, prep = 0, gpu = 0;
    const int errs0 = gg::launch_errors();
    for (;;) {
        if (serve_next(c, h, p, pings, wakes, err) != 1) {
            const double n = (double) std::max<int64_t>(1, served);
            std::fprintf(stderr, "expert server: main process %s gone (%s); %lld requests served, %.1f us each (the "
                                 "rows' prep %.1f, the GPU from the copy in to the copy out %.1f, bf16 and send %.1f)\n",
                         c.peer().c_str(), err.c_str(), (long long) served, busy * 1000.0 / n, prep * 1000.0 / n,
                         gpu * 1000.0 / n, (busy - prep - gpu) * 1000.0 / n);
            std::fprintf(stderr, "expert server: %lld Wakes taken, %llu waits in poll, %llu reads quick-ACKed\n",
                         (long long) wakes, (unsigned long long) c.polls, (unsigned long long) c.quickacks);
            if (pings > 0) std::fprintf(stderr, "expert server: %lld pings answered\n", (long long) pings);
            std::fprintf(stderr, "expert server: %lld requests over 2 ms here\n", (long long) slow);
            static const char* const bucket_name[3] = {"T=1", "T=4", "T=other"};
            for (int bk = 0; diag && bk < 3; ++bk)
                for (int k = 0; k < kParts; ++k) {
                    std::vector<float>& v = parts[bk][k];
                    if (v.empty()) continue;
                    std::sort(v.begin(), v.end());
                    double s = 0;
                    for (float x : v) s += x;
                    std::fprintf(stderr, "expert server: part %-7s %-36s mean %7.1f p50 %7.1f p99 %7.1f us (%zu)\n",
                                 bucket_name[bk], part_name[k], s / (double) v.size(), (double) v[v.size() / 2],
                                 (double) v[(size_t) ((double) v.size() * 0.99)], v.size());
                }
            break;
        }
        const double tr = now_ms();
        const int li = h.a, nr = h.b;
        bucket = nr == 1 ? 0 : nr == 4 ? 1 : 2;
        const size_t per = sizeof(ReqRow) + (size_t) D * 2;
        std::string why;
        if (li < 0 || li >= g.n_layers || nr < 1 || nr > kMaxRows || p.size() != (size_t) nr * per) why = "a malformed request";
        if (null_reply && why.empty()) {
            std::fill(yb.begin(), yb.begin() + (long) ((size_t) nr * D), (uint16_t) 0);
            const double dt = now_ms() - tr;
            if (!c.send(Resp, h.seq, nr, 0, (int32_t) (dt * 1000.0), yb.data(), (size_t) nr * D * 2, err)) break;
            keep(kAll, now_ms() - tr);
            busy += dt;
            ++served;
            continue;
        }
        // where this request's plan, weights and x go (host and device)
        const bool packed = cp != Copy::Base;
        const size_t pw = (size_t) nr * (8 * 8 + 8 * 4);   // (packed: the rows' plan, then their weights)
        const size_t xoff = (pw + 255) / 256 * 256;
        unsigned long long* hp = packed ? (unsigned long long*) hs : b.h_plan;
        float* hw = packed ? (float*) (hp + (size_t) nr * 8) : b.h_w;
        unsigned long long* dp = packed ? (unsigned long long*) ds : b.d_plan;
        float* dw = packed ? (float*) (dp + (size_t) nr * 8) : b.d_w;
        float* hx = cp == Copy::One ? (float*) ((char*) hs + xoff) : cp == Copy::Map ? (float*) hm : b.h_x;
        float* dx = cp == Copy::One ? (float*) ((char*) ds + xoff) : cp == Copy::Map ? dm : b.d_x;
        float* hy = cp == Copy::Map ? (float*) hm + (size_t) kMaxRows * D : b.h_y;
        float* dy = cp == Copy::Map ? dm + (size_t) kMaxRows * D : b.d_y;
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
                hp[j * 8 + i] = (unsigned long long) d;
                hw[j * 8 + i] = i < rw.n ? rw.w[i] : 0.0f;
            }
            const uint16_t* xb = (const uint16_t*) (p.data() + (size_t) j * per + sizeof(ReqRow));
            float* xf = hx + (size_t) j * D;
            for (int i = 0; i < D; ++i) xf[i] = bf2f(xb[i]);
        }
        if (!why.empty()) {
            (void) c.send(Error, h.seq, 0, 0, 0, why.data(), why.size(), err);
            std::fprintf(stderr, "expert server: %s\n", why.c_str());
            break;
        }
        const double tg = now_ms();
        prep += tg - tr;
        int ne = 0;
        if (evs) RX_CK(cudaEventRecord(evt[ne++], s));
        switch (cp) {
            case Copy::Base:
                RX_CK(cudaMemcpyAsync(b.d_plan, b.h_plan, (size_t) kMaxRows * 8 * 8 + (size_t) kMaxRows * 8 * 4,
                                      cudaMemcpyHostToDevice, s));
                break;
            case Copy::Nr:
            case Copy::Map: RX_CK(cudaMemcpyAsync(ds, hs, pw, cudaMemcpyHostToDevice, s)); break;
            case Copy::One: RX_CK(cudaMemcpyAsync(ds, hs, xoff + (size_t) nr * D * 4, cudaMemcpyHostToDevice, s)); break;
        }
        if (evs) RX_CK(cudaEventRecord(evt[ne++], s));
        if (cp == Copy::Base || cp == Copy::Nr) RX_CK(cudaMemcpyAsync(dx, hx, (size_t) nr * D * 4, cudaMemcpyHostToDevice, s));
        if (evs) RX_CK(cudaEventRecord(evt[ne++], s));
        for (int j = 0; j < nr; ++j) {
            ReqRow rw;
            std::memcpy(&rw, p.data() + (size_t) j * per, sizeof rw);
            x3::MoeArgs ma;
            ma.plan_ptr = dp + j * 8;
            ma.plan_w = dw + j * 8;
            ma.lay = ss.lay[(size_t) li];
            ma.k = rw.n;
            ma.n_embd = D;
            ma.n_ff = FF;
            ma.limit = g.limit;
            gg::moe_gate_up(ma, dx + (size_t) j * D, b.d_h, s);
            if (evs) RX_CK(cudaEventRecord(evt[ne++], s));
            gg::moe_down(ma, b.d_h, nullptr, dy + (size_t) j * D, s);
            if (evs) RX_CK(cudaEventRecord(evt[ne++], s));
        }
        if (cp != Copy::Map) RX_CK(cudaMemcpyAsync(hy, dy, (size_t) nr * D * 4, cudaMemcpyDeviceToHost, s));
        if (evs) RX_CK(cudaEventRecord(evt[ne++], s));
        const double te = now_ms();
        RX_CK(cudaStreamSynchronize(s));
        const double ts = now_ms();
        gpu += ts - tg;
        if (gg::launch_errors() != errs0) {
            const char w2[] = "a kernel launch failed on the expert server";
            (void) c.send(Error, h.seq, 0, 0, 0, w2, sizeof w2 - 1, err);
            rc = -1;
            break;
        }
        for (size_t i = 0; i < (size_t) nr * D; ++i) yb[i] = f2bf(hy[i]);
        const double dt = now_ms() - tr;
        if (!c.send(Resp, h.seq, nr, 0, (int32_t) (dt * 1000.0), yb.data(), (size_t) nr * D * 2, err)) {
            std::fprintf(stderr, "expert server: reply failed (%s)\n", err.c_str());
            break;
        }
        busy += dt;
        ++served;
        const double tq = now_ms(), gap = last_end > 0 ? tr - last_end : -1.0;
        if (diag) {
            const double tz = now_ms();
            keep(kPrep, tg - tr);
            keep(kEnq, te - tg);
            keep(kSync, ts - te);
            keep(kPost, tz - ts);
            keep(kAll, tz - tr);
        }
        if (evs) {
            auto el = [&](int a, int z) {
                float ms = 0;
                cudaEventElapsedTime(&ms, evt[a], evt[z]);
                return (double) ms;
            };
            keep(kPlanIn, el(0, 1));
            keep(kXIn, el(1, 2));
            double gu = 0, dn = 0;
            for (int j = 0; j < nr; ++j) {
                gu += el(2 + 2 * j, 3 + 2 * j);
                dn += el(3 + 2 * j, 4 + 2 * j);
            }
            keep(kGateUp, gu);
            keep(kDown, dn);
            keep(kYOut, el(ne - 2, ne - 1));
            keep(kSpan, el(0, ne - 1));
        }
        if (tq - tr > 2.0 && ++slow <= 100)   // a slow one, its parts (a late reply: here or the network); said after
            std::fprintf(stderr, "expert server: slow request %s UTC seq %u layer %d rows %d: %.2f ms (the rows' prep "
                                 "%.2f, the GPU %.2f, bf16 %.2f, the send %.2f; %.1f ms after the last one's end)%s\n",
                         utc_now().c_str(), h.seq, li, nr, tq - tr, tg - tr, ts - tg, tr + dt - ts, tq - tr - dt, gap,
                         slow == 100 ? " (the last one said; the count at the end)" : "");
        last_end = now_ms();   // (after this one's timing and line: the gap is the main's and the network's)
    }
    cudaStreamSynchronize(s);
    return rc;   // (the buffers, the events and the arena: ~Session)
}

}  // namespace

int serve(const std::string& host_port, const std::string& token, int64_t reserve_mib, int dev, bool once) {
    RX_CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    RX_CK(cudaSetDevice(dev));
    cudaStream_t s = nullptr;
    RX_CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    if (!gg::prepare(s)) return 1;
    // every kernel a request can launch, loaded before any session reads its room (DS4_RX_PRELOAD=0, tests: not, so
    // that CUDA's lazy loading takes their memory at the first launch, which is then the warm-up's)
    const char* lv = std::getenv("DS4_RX_PRELOAD");
    if (!(lv != nullptr && *lv == '0')) {
        size_t f0 = 0, f1 = 0, tot = 0;
        cudaMemGetInfo(&f0, &tot);
        if (!gg::load_moe_kernels()) return 1;
        cudaMemGetInfo(&f1, &tot);
        std::fprintf(stderr, "expert server: the requests' kernels loaded (%.1f MiB of VRAM)\n",
                     f0 > f1 ? (double) (f0 - f1) / 1048576.0 : 0.0);
    }
    if (token.empty()) std::fprintf(stderr, "expert server: WARNING: no STRATA_STAGE_TOKEN\n");
    std::fprintf(stderr, "expert server: GPU %d\n", dev);
    struct Arg {
        const std::string& token;
        int64_t reserve_mib;
        int dev;
        cudaStream_t s;
    } a{token, reserve_mib, dev, s};
    return serve_loop(host_port, once, "expert server", [](Conn& c, void* p) {
        const Arg& x = *(const Arg*) p;
        return session(c, x.token, x.reserve_mib, x.dev, x.s);
    }, &a);
}

}  // namespace strata::ds4::rx
