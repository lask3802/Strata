// src/dsv41/ds4_rx_fill_link.cpp - the fills' socket (ds4_rx_fill_link.hpp)
#include "ds4_rx_fill_link.hpp"

#include "ds4_os.hpp"   // (os::pin_thread, cpu_siblings: the fill thread's CPU)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
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
    const size_t kb = (size_t) ((uint64_t) k * (row[2] / (uint64_t) nb));
    uint8_t* d = dst + ship_off(2, k);
    for (int i = 0; i < rows[2]; ++i) std::memcpy(d + (size_t) i * kb, src[2] + (size_t) i * row[2], kb);
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
            if (buf.size() < n) buf.resize((size_t) n);
            j.geo.gather(j.src, j.k, buf.data());
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
            // 1 MB pieces, a sleep until each is due; time the socket idled earns no burst
            for (uint64_t at = 0; ok && at < n; at += kPiece) {
                const size_t m = (size_t) std::min<uint64_t>(kPiece, n - at);
                if (pace_ > 0.0) {
                    const double now = net::ep::now_ms();
                    if (due_ > now) std::this_thread::sleep_for(std::chrono::microseconds((int64_t) ((due_ - now) * 1000.0)));
                    due_ = std::max(due_, now) + 1000.0 * (double) m / pace_;
                }
                ok = !stop_.load(std::memory_order_acquire) && c_.send_bytes(buf.data() + at, m, err);
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
