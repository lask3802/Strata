// src/dsv41/ds4_rx.cpp - see ds4_rx.hpp: the sockets, the messages, the main process's end.
#include "ds4_rx.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_M_ARM64EC)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::ds4::rx {

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf2f(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void relax() {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_M_ARM64EC)
    _mm_pause();
#endif
}

namespace {

#if defined(_WIN32)
using sock_t = SOCKET;
constexpr sock_t kBad = INVALID_SOCKET;
constexpr int kSendFlags = 0;
bool net_init() {
    static const bool ok = [] {
        WSADATA d;
        return WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    return ok;
}
int last_err() { return WSAGetLastError(); }
bool would_block(int e) { return e == WSAEWOULDBLOCK; }
void close_sock(sock_t s) { closesocket(s); }
bool nonblock(sock_t s) {
    u_long on = 1;
    return ioctlsocket(s, FIONBIO, &on) == 0;
}
#else
using sock_t = int;
constexpr sock_t kBad = -1;
#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif
bool net_init() { return true; }
int last_err() { return errno; }
bool would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }
void close_sock(sock_t s) { ::close(s); }
bool nonblock(sock_t s) {
    const int fl = fcntl(s, F_GETFL, 0);
    return fl >= 0 && fcntl(s, F_SETFL, fl | O_NONBLOCK) == 0;
}
#endif

std::string err_text(const char* what) { return std::string(what) + " failed (error " + std::to_string(last_err()) + ")"; }

// a connected socket as the link wants it: no Nagle, non-blocking, large buffers
void tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    int buf = 8 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
    nonblock(s);
}

// "HOST:PORT" / "PORT" (host empty)
bool split(const std::string& at, std::string& host, std::string& port) {
    const size_t c = at.rfind(':');
    host = c == std::string::npos ? "" : at.substr(0, c);
    port = c == std::string::npos ? at : at.substr(c + 1);
    return !port.empty() && std::atoi(port.c_str()) > 0 && std::atoi(port.c_str()) < 65536;
}

// wait for s readable (rd) or writable, up to ms (< 0: forever); false: timed out or failed
bool wait_sock(sock_t s, bool rd, int ms) {
#if defined(_WIN32)
    WSAPOLLFD p{};
    p.fd = s;
    p.events = rd ? POLLRDNORM : POLLWRNORM;
    return WSAPoll(&p, 1, ms) > 0;
#else
    pollfd p{};
    p.fd = s;
    p.events = rd ? POLLIN : POLLOUT;
    return ::poll(&p, 1, ms) > 0;
#endif
}

}  // namespace

Conn::~Conn() { close(); }

void Conn::close() {
    if (fd_ >= 0) close_sock((sock_t) fd_);
    fd_ = -1;
    rb_.clear();
}

bool Conn::dial(const std::string& host_port, int timeout_s, std::string& err) {
    close();
    if (!net_init()) {
        err = "the sockets did not start (WSAStartup)";
        return false;
    }
    std::string host, port;
    if (!split(host_port, host, port) || host.empty()) {
        err = "expected HOST:PORT, not " + host_port;
        return false;
    }
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + host;
        return false;
    }
    sock_t s = kBad;
    for (addrinfo* a = res; a != nullptr && s == kBad; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == kBad) continue;
        nonblock(s);
        const int rc = ::connect(s, a->ai_addr, (int) a->ai_addrlen);
        bool up = rc == 0;
        if (!up && (would_block(last_err())
#if defined(_WIN32)
                    || last_err() == WSAEINPROGRESS
#else
                    || last_err() == EINPROGRESS
#endif
                    )) {
            if (wait_sock(s, false, timeout_s * 1000)) {
                int so = 0;
                socklen_t sl = sizeof so;
                up = getsockopt(s, SOL_SOCKET, SO_ERROR, (char*) &so, &sl) == 0 && so == 0;
            }
        }
        if (!up) {
            close_sock(s);
            s = kBad;
        }
    }
    freeaddrinfo(res);
    if (s == kBad) {
        err = "no connection to " + host_port;
        return false;
    }
    tune(s);
    fd_ = (std::intptr_t) s;
    peer_ = host_port;
    return true;
}

bool Conn::listen_accept(const std::string& at, int wait_s, std::string& err) {
    close();
    if (!net_init()) {
        err = "the sockets did not start (WSAStartup)";
        return false;
    }
    std::string host, port;
    if (!split(at, host, port)) {
        err = "expected [BIND:]PORT, not " + at;
        return false;
    }
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    if (getaddrinfo(host.empty() ? nullptr : host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + at;
        return false;
    }
    sock_t l = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (l == kBad) {
        freeaddrinfo(res);
        err = err_text("socket");
        return false;
    }
    int one = 1;
    setsockopt(l, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    const bool bound = ::bind(l, res->ai_addr, (int) res->ai_addrlen) == 0 && ::listen(l, 1) == 0;
    freeaddrinfo(res);
    if (!bound) {
        err = err_text(("listening on " + at).c_str());
        close_sock(l);
        return false;
    }
    if (!wait_sock(l, true, wait_s * 1000)) {
        close_sock(l);
        err = "no expert server connected to " + at + " in " + std::to_string(wait_s) + " s";
        return false;
    }
    sockaddr_storage sa{};
    socklen_t sl = sizeof sa;
    sock_t s = ::accept(l, (sockaddr*) &sa, &sl);
    close_sock(l);
    if (s == kBad) {
        err = err_text("accept");
        return false;
    }
    char hb[64] = {};
    if (sa.ss_family == AF_INET) inet_ntop(AF_INET, &((sockaddr_in*) &sa)->sin_addr, hb, sizeof hb);
    peer_ = hb;
    tune(s);
    fd_ = (std::intptr_t) s;
    return true;
}

bool Conn::send_raw_(const void* p, size_t n, std::string& err) {
    const char* b = (const char*) p;
    const double give_up = now_ms() + 30000.0;   // (a full buffer for 30 s: the peer is gone)
    while (n > 0) {
        const int chunk = (int) std::min<size_t>(n, (size_t) 1 << 20);
        const int k = (int) ::send((sock_t) fd_, b, chunk, kSendFlags);
        if (k > 0) {
            b += k;
            n -= (size_t) k;
            bytes_out += (uint64_t) k;
            continue;
        }
        if (k < 0 && would_block(last_err()) && now_ms() < give_up) {
            wait_sock((sock_t) fd_, false, 100);
            continue;
        }
        err = k == 0 ? "the link closed" : err_text("send");
        close();
        return false;
    }
    return true;
}

bool Conn::send(uint32_t type, uint32_t seq, int32_t a, int32_t b, int32_t c, const void* p, size_t n, std::string& err,
                const void* p2, size_t n2) {
    if (fd_ < 0) {
        err = "not connected";
        return false;
    }
    Hdr h;
    h.type = type;
    h.seq = seq;
    h.a = a;
    h.b = b;
    h.c = c;
    h.bytes = n + n2;
    if (n + n2 <= 64 * 1024) {   // (a small message in one send: one segment on the wire)
        uint8_t one[sizeof(Hdr) + 64 * 1024];
        std::memcpy(one, &h, sizeof h);
        if (n) std::memcpy(one + sizeof h, p, n);
        if (n2) std::memcpy(one + sizeof h + n, p2, n2);
        return send_raw_(one, sizeof h + n + n2, err);
    }
    return send_raw_(&h, sizeof h, err) && (n == 0 || send_raw_(p, n, err)) && (n2 == 0 || send_raw_(p2, n2, err));
}

int Conn::recv(Hdr& h, std::vector<uint8_t>& payload, double deadline, std::string& err) {
    if (fd_ < 0) {
        err = "not connected";
        return -1;
    }
    uint8_t tmp[64 * 1024];
    for (;;) {
        if (rb_.size() >= sizeof(Hdr)) {
            std::memcpy(&h, rb_.data(), sizeof h);
            if (h.magic != kMagic || h.bytes > ((uint64_t) 1 << 32)) {
                err = "a malformed message (the link is out of step)";
                close();
                return -1;
            }
            if (rb_.size() >= sizeof(Hdr) + h.bytes) {
                payload.assign(rb_.begin() + sizeof(Hdr), rb_.begin() + (long) (sizeof(Hdr) + h.bytes));
                rb_.erase(rb_.begin(), rb_.begin() + (long) (sizeof(Hdr) + h.bytes));
                return 1;
            }
        }
        const int k = (int) ::recv((sock_t) fd_, (char*) tmp, (int) sizeof tmp, 0);
        if (k > 0) {
            rb_.insert(rb_.end(), tmp, tmp + k);
            bytes_in += (uint64_t) k;
            last_in_ = now_ms();
            continue;
        }
        if (k == 0 || !would_block(last_err())) {
            err = k == 0 ? "the link closed" : err_text("recv");
            close();
            return -1;
        }
        const double now = now_ms();
        if (deadline >= 0 && now > deadline) return 0;
        if (now - last_in_ > 200.0) {   // quiet: sleep in poll (a reply's wait in decode stays a spin)
            const int ms = deadline < 0 ? 50 : (int) std::min(50.0, deadline - now) + 1;
            (void) wait_sock((sock_t) fd_, true, ms);
            continue;
        }
        relax();
    }
}

bool Main::accept(const std::string& at, const std::string& token, int wait_s, std::string& err) {
    if (!c_.listen_accept(at, wait_s, err)) return false;
    Hdr h;
    std::vector<uint8_t> p;
    if (c_.recv(h, p, now_ms() + 10000.0, err) != 1 || h.type != Hello || p.size() != sizeof(HelloMsg)) {
        if (err.empty()) err = "no hello from the expert server";
        c_.close();
        return false;
    }
    std::memcpy(&peer_hello_, p.data(), sizeof peer_hello_);
    HelloMsg me;
    std::memcpy(me.token, token.data(), std::min(token.size(), sizeof me.token - 1));
    std::snprintf(me.name, sizeof me.name, "main");
    if (peer_hello_.proto != kProto || std::strncmp(peer_hello_.token, me.token, sizeof me.token) != 0) {
        const std::string why = peer_hello_.proto != kProto ? "protocol differs" : "wrong token (STRATA_STAGE_TOKEN)";
        (void) c_.send(Error, 0, 0, 0, 0, why.data(), why.size(), err);
        err = "expert server " + c_.peer() + " refused: " + why;
        c_.close();
        return false;
    }
    return c_.send(Hello, 0, 0, 0, 0, &me, sizeof me, err);
}

bool Main::send_geo(const GeoMsg& g, const std::vector<GeoLayer>& layers, std::string& err) {
    n_embd_ = g.n_embd;
    tx_.resize(sizeof g + layers.size() * sizeof(GeoLayer));
    std::memcpy(tx_.data(), &g, sizeof g);
    if (!layers.empty()) std::memcpy(tx_.data() + sizeof g, layers.data(), layers.size() * sizeof(GeoLayer));
    return c_.send(Geo, 0, 0, 0, 0, tx_.data(), tx_.size(), err);
}

bool Main::send_blob(int layer, int e, const void* blob, size_t bytes, std::string& err) {
    return c_.send(Blob, 0, layer, e, 0, blob, bytes, err);
}

bool Main::end_blobs(int n, std::vector<uint8_t>& ok, double timeout_ms, std::string& err) {
    if (!c_.send(BlobsEnd, 0, n, 0, 0, nullptr, 0, err)) return false;
    Hdr h;
    const int got = c_.recv(h, rx_, now_ms() + timeout_ms, err);
    if (got != 1 || h.type != BlobsOk || rx_.size() != (size_t) n) {
        if (got == 1 && h.type == Error) err = "expert server: " + std::string(rx_.begin(), rx_.end());
        else if (err.empty()) err = "no answer to the blobs";
        c_.close();
        return false;
    }
    ok = rx_;
    return true;
}

bool Main::send_req(uint32_t seq, int layer, int nrows, const ReqRow* rows, const float* const* x, std::string& err) {
    const size_t per = sizeof(ReqRow) + (size_t) n_embd_ * 2;
    tx_.resize((size_t) nrows * per);
    for (int j = 0; j < nrows; ++j) {
        uint8_t* o = tx_.data() + (size_t) j * per;
        std::memcpy(o, &rows[j], sizeof(ReqRow));
        uint16_t* xb = (uint16_t*) (o + sizeof(ReqRow));
        for (int i = 0; i < n_embd_; ++i) xb[i] = f2bf(x[j][i]);
    }
    return c_.send(Req, seq, layer, nrows, 0, tx_.data(), tx_.size(), err);
}

int Main::wait_resp(uint32_t seq, int nrows, double deadline, const uint16_t** y, int& srv_us, std::string& err) {
    for (;;) {
        Hdr h;
        const int got = c_.recv(h, rx_, deadline, err);
        if (got <= 0) return got;
        if (h.type == Error) {
            err = "expert server: " + std::string(rx_.begin(), rx_.end());
            c_.close();
            return -1;
        }
        if (h.type != Resp) continue;
        if (h.seq != seq) continue;   // (an earlier request's, late: its picks were computed here)
        if (h.a != nrows || rx_.size() != (size_t) nrows * (size_t) n_embd_ * 2) {
            err = "a reply of the wrong size";
            c_.close();
            return -1;
        }
        *y = (const uint16_t*) rx_.data();
        srv_us = h.c;
        return 1;
    }
}

}  // namespace strata::ds4::rx
