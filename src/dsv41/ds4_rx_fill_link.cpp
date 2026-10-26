// src/dsv41/ds4_rx_fill_link.cpp - the fills' socket (ds4_rx_fill_link.hpp)
#include "ds4_rx_fill_link.hpp"

#include "ds4_os.hpp"   // (os::pin_thread, cpu_siblings: the fill thread's CPU)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>

namespace strata::ds4::rx {

namespace {

#pragma pack(push, 1)
struct FillHello {   // the server's hello on the fills' socket (EpHdr Hello); the main answers HelloOk, empty, or Error
    uint32_t proto = kProto;
    uint32_t pad = 0;
    uint64_t session = 0;
    char token[64] = {};
};
#pragma pack(pop)

constexpr double kHelloMs = 5000.0;
constexpr uint64_t kPiece = 1u << 20;   // (the fill thread's paced sends)

// (DS4_RX_FILL_GATE) FillGeo::gather and gather_down in copies of at most FillLink::kGatePiece bytes, idle() before each
void copy_pieces(uint8_t* d, const uint8_t* s, uint64_t n, const std::function<void()>& idle) {
    for (uint64_t at = 0; at < n; at += FillLink::kGatePiece) {
        idle();
        std::memcpy(d + at, s + at, (size_t) std::min<uint64_t>(FillLink::kGatePiece, n - at));
    }
}

void gather_down_pieces(const FillGeo& g, const uint8_t* src2, int k, uint8_t* dst, const std::function<void()>& idle) {
    const size_t kb = (size_t) ((uint64_t) k * (g.row[2] / (uint64_t) g.nb));
    uint64_t since = FillLink::kGatePiece;   // (idle() before the first row's copy too)
    for (int i = 0; i < g.rows[2]; ++i) {
        if (since >= FillLink::kGatePiece) {
            idle();
            since = 0;
        }
        std::memcpy(dst + (size_t) i * kb, src2 + (size_t) i * g.row[2], kb);
        since += kb;
    }
}

void gather_pieces(const FillGeo& g, const uint8_t* const src[3], int k, uint8_t* dst, const std::function<void()>& idle) {
    if (!g.split(k)) {
        for (uint64_t at = 0; at < g.blob; at += FillLink::kGatePiece) {
            idle();
            std::memset(dst + at, 0, (size_t) std::min<uint64_t>(FillLink::kGatePiece, g.blob - at));
        }
        for (int r = 0; r < 3; ++r) copy_pieces(dst + g.off[r], src[r], (uint64_t) g.rows[r] * g.row[r], idle);
        return;
    }
    copy_pieces(dst, src[0], (uint64_t) k * 256 * g.row[0], idle);
    copy_pieces(dst + g.ship_off(1, k), src[1], (uint64_t) k * 256 * g.row[1], idle);
    gather_down_pieces(g, src[2], k, dst + g.ship_off(2, k), idle);
}

}  // namespace

FillGeo::FillGeo(const GeoLayer& G, int n_embd, int n_ff, const uint64_t row_bytes[3]) {
    for (int r = 0; r < 3; ++r) {
        off[r] = G.off[r];
        row[r] = row_bytes[r];
        rows[r] = r < 2 ? n_ff : n_embd;
    }
    blob = G.blob;
    nb = std::max(1, n_ff / 256);
}

uint64_t FillGeo::ship_off(int r, int k) const {
    if (r == 0) return 0;
    if (r == 1) return (uint64_t) k * 256 * row[0];
    return (uint64_t) k * 256 * (row[0] + row[1]);
}

uint64_t FillGeo::ship_bytes(int k) const {
    if (!split(k)) return blob;
    return ship_off(2, k) + (uint64_t) rows[2] * (uint64_t) k * (row[2] / (uint64_t) nb);
}

void FillGeo::gather(const uint8_t* const src[3], int k, uint8_t* dst) const {
    if (!split(k)) {
        std::memset(dst, 0, (size_t) blob);
        for (int r = 0; r < 3; ++r) std::memcpy(dst + off[r], src[r], (size_t) (rows[r] * row[r]));
        return;
    }
    std::memcpy(dst, src[0], (size_t) ((uint64_t) k * 256 * row[0]));
    std::memcpy(dst + ship_off(1, k), src[1], (size_t) ((uint64_t) k * 256 * row[1]));
    gather_down(src[2], k, dst + ship_off(2, k));
}

void FillGeo::gather_down(const uint8_t* src2, int k, uint8_t* dst) const {
    const size_t kb = (size_t) ((uint64_t) k * (row[2] / (uint64_t) nb));
    for (int i = 0; i < rows[2]; ++i) std::memcpy(dst + (size_t) i * kb, src2 + (size_t) i * row[2], kb);
}

FillLink::~FillLink() { close(); }

bool FillLink::open(Main& m, const std::string& rx_at, int port, const std::string& token, double pace_bps, int cpu,
                    int wait_ms, std::string& err) {
    std::string host, p;
    if (!net::ep::split_host_port(rx_at, host, p)) {
        err = "expected [BIND:]PORT, not " + rx_at;
        return false;
    }
    const int pt = port > 0 ? port : std::atoi(p.c_str()) + 1;
    const std::string at = (host.empty() ? "" : (host.find(':') != std::string::npos ? "[" + host + "]:" : host + ":")) +
                           std::to_string(pt);
    if (!ls_.open(at, err)) return false;
    port_ = ls_.port();
    std::random_device rd;
    const uint64_t session = (uint64_t) rd() << 32 | (uint64_t) rd();
    if (!m.bulk_hello(port_, session, err)) return false;
    if (!ls_.accept(c_, wait_ms, err)) {
        err = "the expert server did not dial the fills' port " + at + " (" + err + "; a server that takes no fills)";
        ls_.close();
        return false;
    }
    ls_.close();
    net::ep::EpHdr h;
    std::vector<uint8_t> pl;
    c_.max_frame = sizeof(FillHello);
    c_.spin_ms = 0;
    auto refuse = [&](const std::string& why) {
        std::string e2;
        (void) c_.send(net::ep::EpHdr{0, net::ep::Error}, why.data(), why.size(), e2);
        c_.close();
        err = why;
        return false;
    };
    if (c_.recv(h, pl, net::ep::now_ms() + kHelloMs, err) != 1) return refuse("no hello on the fills' socket (" + err + ")");
    FillHello fh;
    if (h.type != net::ep::Hello || pl.size() != sizeof fh) return refuse("not a fills hello");
    std::memcpy(&fh, pl.data(), sizeof fh);
    char mine[64] = {};
    std::memcpy(mine, token.data(), std::min(token.size(), sizeof mine - 1));
    if (fh.proto != kProto || fh.session != session || std::strncmp(fh.token, mine, sizeof mine) != 0)
        return refuse(fh.proto != kProto ? "protocol differs" : fh.session != session ? "another session" : "wrong token");
    if (!c_.send(net::ep::EpHdr{0, net::ep::HelloOk}, nullptr, 0, err)) return false;
    c_.max_frame = 4096;   // (FillOks, Errors)
    pace_ = pace_bps;
    due_ = 0.0;
    c_.cancel = &stop_;
    cpu_ = cpu;
    stop_ = false;
    up_ = true;
    th_ = std::thread([this] { loop_(); });
    if (cpu_ >= 0) os::pin_thread(th_, cpu_);
    return true;
}

void FillLink::queue(const Job& j) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        jobs_.push_back(j);
    }
    cv_.notify_one();
}

std::vector<FillLink::Done> FillLink::take() {
    std::vector<Done> v;
    std::lock_guard<std::mutex> lk(mu_);
    v.swap(done_);
    return v;
}

std::string FillLink::why() {
    std::lock_guard<std::mutex> lk(mu_);
    return why_;
}

std::string FillLink::zc_note() {
    std::lock_guard<std::mutex> lk(mu_);
    return zc_note_;
}

void FillLink::close() {
    stop_ = true;
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    up_ = false;
    c_.close();
    ls_.close();
}

void FillLink::loop_() {
    std::vector<uint8_t> buf, pl;
    std::string err;
    auto fail = [&](const std::string& why) {
        std::lock_guard<std::mutex> lk(mu_);
        if (why_.empty()) why_ = why;
        up_ = false;
    };
    // (gate) until the lane's batches are done: true if it waited.  The count is looked at again each time it changes
    // (the lane's call starts and ends), the cap then; a stop waits for the call under way at most
    auto lane_wait = [this]() -> bool {
        int b = gate->load(std::memory_order_acquire);
        if (b <= 0) return false;
        const uint64_t w0 = net::ep::now_ns();
        ++gate_waits;
        while (b > 0 && !stop_.load(std::memory_order_acquire)) {
            if ((double) (net::ep::now_ns() - w0) / 1e6 >= kGateMaxMs) {
                ++gate_over;
                break;
            }
            gate->wait(b, std::memory_order_acquire);
            b = gate->load(std::memory_order_acquire);
        }
        gate_us += (net::ep::now_ns() - w0) / 1000;
        return true;
    };
    const std::function<void()> idle = [&] { (void) lane_wait(); };
    const uint64_t piece = gate != nullptr ? kGatePiece : kPiece;
    while (!stop_.load(std::memory_order_acquire)) {
        Job j;
        bool have = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!jobs_.empty()) {
                j = jobs_.front();
                jobs_.pop_front();
                have = true;
            }
        }
        if (have) {
            const uint64_t n = j.geo.ship_bytes(j.k);
            // the payload in pieces as gather lays it out: one buffer (zc 0, or a whole blob), or a split fill's gate
            // and up from the pack (the mapping; zc 2: its file) and its downs gathered
            struct Seg {
                const uint8_t* p = nullptr;
                std::intptr_t fd = -1;
                uint64_t foff = 0, n = 0;
                bool pack = false;
            } seg[3];
            int nseg = 1;
            if (zc > 0 && j.geo.split(j.k)) {
                const uint64_t o1 = j.geo.ship_off(1, j.k), o2 = j.geo.ship_off(2, j.k);
                if (buf.size() < n - o2) buf.resize((size_t) (n - o2));
                if (gate != nullptr) gather_down_pieces(j.geo, j.src[2], j.k, buf.data(), idle);
                else j.geo.gather_down(j.src[2], j.k, buf.data());
                const uint64_t len[2] = {o1, o2 - o1};
                for (int r = 0; r < 2; ++r) {
                    seg[r].p = j.src[r];
                    seg[r].fd = zc == 2 ? j.fd[r] : -1;
                    seg[r].foff = j.foff[r];
                    seg[r].n = len[r];
                    seg[r].pack = true;
                }
                seg[2].p = buf.data();
                seg[2].n = n - o2;
                nseg = 3;
            } else {
                if (buf.size() < n) buf.resize((size_t) n);
                if (gate != nullptr) gather_pieces(j.geo, j.src, j.k, buf.data(), idle);
                else j.geo.gather(j.src, j.k, buf.data());
                seg[0].p = buf.data();
                seg[0].n = n;
            }
            net::ep::EpHdr h;
            h.type = net::ep::Fill;
            h.layer = (uint8_t) j.layer;
            h.T = (uint8_t) (j.geo.split(j.k) ? j.k : 0);
            h.seq = j.seq;
            h.aux = (uint32_t) j.in;
            h.epoch = (uint32_t) j.out;
            h.bytes = (uint32_t) n;
            h.t_ns = net::ep::now_ns();
            const uint64_t t0 = net::ep::now_ns();
            bool ok = c_.send_bytes(&h, sizeof h, err);
            // paced here, not by the Conn (its pace spins between 64 KB pieces - beside the link core, on its sibling):
            // 1 MB pieces of each segment, a sleep until each is due; time the socket idled earns no burst.  (gate:
            // 64 KB pieces, each after the lane's batches; a wait for them restarts the pace's clock - no burst either)
            for (int s = 0; ok && s < nseg; ++s)
                for (uint64_t at = 0; ok && at < seg[s].n; at += piece) {
                    const size_t m = (size_t) std::min<uint64_t>(piece, seg[s].n - at);
                    if (pace_ > 0.0) {
                        const double now = net::ep::now_ms();
                        if (due_ > now) std::this_thread::sleep_for(std::chrono::microseconds((int64_t) ((due_ - now) * 1000.0)));
                        due_ = std::max(due_, now) + 1000.0 * (double) m / pace_;
                    }
                    if (gate != nullptr && lane_wait() && pace_ > 0.0) due_ = net::ep::now_ms() + 1000.0 * (double) m / pace_;
                    if (stop_.load(std::memory_order_acquire)) {
                        ok = false;
                        break;
                    }
                    int r = 0;
                    if (seg[s].fd >= 0 && file_ok_) {
                        r = c_.send_file(seg[s].fd, seg[s].foff + at, m, err);
                        if (r == 0) {   // (refused before a byte went: the mapping's way from here on)
                            file_ok_ = false;
                            std::lock_guard<std::mutex> lk(mu_);
                            zc_note_ = err;
                        } else if (r > 0) {
                            filed += m;
                        }
                    }
                    if (r == 0) {
                        r = c_.send_bytes(seg[s].p + at, m, err) ? 1 : -1;
                        if (r > 0 && seg[s].pack) direct += m;
                    }
                    ok = r > 0;
                }
            if (!ok) {
                fail("a fill did not go out (" + err + ")");
                return;
            }
            send_us += (net::ep::now_ns() - t0) / 1000;
            ++sent;
            bytes += n;
        }
        // the FillOks in: at once while fills wait to go out, else a short wait (a queued fill is seen within it)
        for (;;) {
            net::ep::EpHdr h;
            const int got = c_.recv(h, pl, have ? net::ep::now_ms() : net::ep::now_ms() + 2.0, err);
            if (got < 0) {
                fail("the fills' socket failed (" + err + ")");
                return;
            }
            if (got == 0) break;
            if (h.type == net::ep::Error) {
                fail("the expert server: " + std::string(pl.begin(), pl.end()));
                return;
            }
            if (h.type != net::ep::FillOk || pl.size() != 2) continue;
            Done d;
            d.seq = h.seq;
            d.layer = h.layer;
            d.in = (int) h.aux;
            d.out = (int) h.epoch;
            std::memcpy(&d.status, pl.data(), 2);
            (d.status == 0 ? oks : fails)++;
            std::lock_guard<std::mutex> lk(mu_);
            done_.push_back(d);
            have = true;   // (another may be in: look again without waiting)
        }
    }
}

void lane_busy_in(std::atomic<int>& busy) { busy.fetch_add(1, std::memory_order_acq_rel); }

void lane_busy_out(std::atomic<int>& busy) {
    if (busy.fetch_sub(1, std::memory_order_acq_rel) == 1) busy.notify_all();
}

bool fill_dial(net::ep::Conn& c, const std::string& rx_peer, int port, uint64_t session, const std::string& token,
               std::string& err) {
    std::string host, p;
    if (!net::ep::split_host_port(rx_peer, host, p) || host.empty()) {
        err = "no host in " + rx_peer;
        return false;
    }
    const std::string at = (host.find(':') != std::string::npos ? "[" + host + "]:" : host + ":") + std::to_string(port);
    if (!c.dial(at, 5000, err)) return false;
    FillHello fh;
    fh.session = session;
    std::memcpy(fh.token, token.data(), std::min(token.size(), sizeof fh.token - 1));
    if (!c.send(net::ep::EpHdr{0, net::ep::Hello}, &fh, sizeof fh, err)) return false;
    net::ep::EpHdr h;
    std::vector<uint8_t> pl;
    c.spin_ms = 0;
    if (c.recv(h, pl, net::ep::now_ms() + kHelloMs, err) != 1) return false;
    if (h.type != net::ep::HelloOk) {
        err = h.type == net::ep::Error ? std::string(pl.begin(), pl.end()) : "no HelloOk";
        c.close();
        return false;
    }
    c.max_frame = net::ep::kBulkFrame;
    return true;
}

int fill_cpu() {
    if (const char* v = std::getenv("DS4_RX_FILL_CPU"); v != nullptr && *v != '\0') {
        const int cpu = std::atoi(v);
        return cpu >= 0 && os::cpu_allowed(cpu) ? cpu : -1;
    }
    const int l = link_cpu();
    if (l < 0) return -1;
    for (int c : os::cpu_siblings(l))
        if (c != l && os::cpu_allowed(c)) return c;
    return -1;
}

}  // namespace strata::ds4::rx
