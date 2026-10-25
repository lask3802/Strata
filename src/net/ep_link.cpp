// src/net/ep_link.cpp - see ep_link.hpp: the sockets, the hellos, both ends' rules.
#include "ep_link.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
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
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#endif

namespace strata::net::ep {

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
int64_t wall_ns() {
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
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
constexpr int kSendFlags = 0, kRecvFlags = 0;   // (the socket itself is non-blocking)
bool net_init() {
    static const bool ok = [] {
        WSADATA d;
        return WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    return ok;
}
int last_err() { return WSAGetLastError(); }
bool would_block(int e) { return e == WSAEWOULDBLOCK; }
bool in_progress(int e) { return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS; }
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
constexpr int kRecvFlags = MSG_DONTWAIT;
bool net_init() { return true; }
int last_err() { return errno; }
bool would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }
bool in_progress(int e) { return e == EINPROGRESS || e == EINTR; }
void close_sock(sock_t s) { ::close(s); }
bool nonblock(sock_t s) {
    const int fl = fcntl(s, F_GETFL, 0);
    return fl >= 0 && fcntl(s, F_SETFL, fl | O_NONBLOCK) == 0;
}
#endif

std::string err_text(const char* what) { return std::string(what) + " failed (error " + std::to_string(last_err()) + ")"; }

template <typename T> void set_opt(sock_t s, int level, int name, T v) { setsockopt(s, level, name, (const char*) &v, sizeof v); }

// a connected socket as the link wants it: stage_link's tune_socket (no Nagle, keepalive that notices a vanished peer
// in about a minute, 8 MB buffers), and non-blocking
void tune(sock_t s) {
    set_opt<int>(s, IPPROTO_TCP, TCP_NODELAY, 1);
    set_opt<int>(s, SOL_SOCKET, SO_KEEPALIVE, 1);
#if defined(SO_NOSIGPIPE)
    set_opt<int>(s, SOL_SOCKET, SO_NOSIGPIPE, 1);
#endif
#if defined(TCP_KEEPIDLE)
    set_opt<int>(s, IPPROTO_TCP, TCP_KEEPIDLE, 30);
#endif
#if defined(TCP_KEEPINTVL)
    set_opt<int>(s, IPPROTO_TCP, TCP_KEEPINTVL, 10);
#endif
#if defined(TCP_KEEPCNT)
    set_opt<int>(s, IPPROTO_TCP, TCP_KEEPCNT, 3);
#endif
    set_opt<int>(s, SOL_SOCKET, SO_SNDBUF, 8 << 20);
    set_opt<int>(s, SOL_SOCKET, SO_RCVBUF, 8 << 20);
    nonblock(s);
}

// "HOST:PORT" / "PORT" (host empty)
bool split(const std::string& at, std::string& host, std::string& port) {
    const size_t c = at.rfind(':');
    host = c == std::string::npos ? "" : at.substr(0, c);
    port = c == std::string::npos ? at : at.substr(c + 1);
    const int p = std::atoi(port.c_str());
    return !port.empty() && p >= 0 && p < 65536;
}

// s readable (rd) or writable within ms (< 0: forever); false: not (timed out or failed)
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

bool same_token(const EpHello& a, const EpHello& b) { return std::memcmp(a.token, b.token, sizeof a.token) == 0; }

// what both hellos must agree on, whichever side checks
std::string common_check(const EpHello& mine, const EpHello& p) {
    if (p.magic != kMagic) return "not an EP link's hello";
    if (p.proto != kProto)
        return "EP protocol " + std::to_string(p.proto) + ", this side speaks " + std::to_string(kProto);
    if (!same_token(mine, p)) return "wrong token (STRATA_STAGE_TOKEN)";
    if (p.role == mine.role) return std::string("both sides are ") + (mine.role == 0 ? "the main" : "a worker");
    if (p.channel != mine.channel) return "the hello is for the other socket (request / bulk)";
    if (p.pack_hash != mine.pack_hash) return "a different pack (pack_hash differs)";
    struct F {
        const char* name;
        int32_t a, b;
    };
    const F fs[] = {{"n_embd", mine.n_embd, p.n_embd},       {"layers", mine.n_layers, p.n_layers},
                    {"experts", mine.n_experts, p.n_experts}, {"picks", mine.k_picks, p.k_picks},
                    {"n_ff", mine.n_ff, p.n_ff},             {"block", mine.block, p.block},
                    {"max T", mine.max_t, p.max_t},          {"bf16 rows", mine.rows_bf16, p.rows_bf16}};
    for (const F& f : fs)
        if (f.a != f.b) return std::string(f.name) + " " + std::to_string(f.b) + ", this side's " + std::to_string(f.a);
    if (p.n_layers < 1 || p.n_layers > kMaxLayers || p.max_t < 1 || p.max_t > kMaxT || p.k_picks < 1 || p.n_embd < 1 ||
        p.block < 1 || p.n_ff % p.block != 0 || p.n_ff / p.block > 255)
        return "a geometry the link cannot carry";
    return "";
}

}  // namespace

std::string slices_cover(const std::vector<EpSlice>& t, const std::vector<int>& k, int blocks) {
    if (k.size() != t.size()) return "the slice table has " + std::to_string(t.size()) + " layers, k " + std::to_string(k.size());
    for (size_t l = 0; l < t.size(); ++l) {
        if (k[l] < 0 || k[l] > blocks) return "layer " + std::to_string(l) + ": k " + std::to_string(k[l]) + " out of range";
        if (k[l] == blocks) continue;   // (nothing of this layer asked of the worker)
        if (t[l].lo >= t[l].hi || t[l].lo > k[l] || t[l].hi != blocks)
            return "layer " + std::to_string(l) + ": the worker stores blocks [" + std::to_string(t[l].lo) + ", " +
                   std::to_string(t[l].hi) + "), the main asks for [" + std::to_string(k[l]) + ", " + std::to_string(blocks) + ")";
    }
    return "";
}

std::string main_check(const EpHello& mine, const std::vector<EpSlice>& my_table, const std::vector<int>& k,
                       const EpHello& w, const std::vector<EpSlice>& wt) {
    std::string why = common_check(mine, w);
    if (!why.empty()) return why;
    if (w.channel != 0) return "";
    if (wt.size() != (size_t) mine.n_layers || my_table.size() != (size_t) mine.n_layers)
        return "the slice table has " + std::to_string(wt.size()) + " layers";
    for (size_t l = 0; l < wt.size(); ++l)
        for (int m = 0; m < 3; ++m)
            if (wt[l].type[m] != my_table[l].type[m] || wt[l].row_bytes[m] != my_table[l].row_bytes[m])
                return "layer " + std::to_string(l) + ": matrix " + std::to_string(m) + " is type " + std::to_string(wt[l].type[m]) +
                       " with " + std::to_string(wt[l].row_bytes[m]) + "-byte rows, this side's " + std::to_string(my_table[l].type[m]) +
                       " / " + std::to_string(my_table[l].row_bytes[m]);
    return slices_cover(wt, k, mine.n_ff / mine.block);
}

std::string worker_check(const EpHello& mine, const EpHello& m) { return common_check(mine, m); }

// ---------------------------------------------------------------------------------------------------- Conn
Conn::~Conn() { close(); }

void Conn::close() {
    if (fd_ >= 0) close_sock((sock_t) fd_);
    fd_ = -1;
    rb_.clear();
    rd_ = 0;
    stamps_ = false;
    rx_ns_ = last_rx_ns_ = 0;
}

void Conn::adopt(std::intptr_t fd, const std::string& peer) {
    close();
    tune((sock_t) fd);
    fd_ = fd;
    peer_ = peer;
    last_frame_ = now_ms();
}

bool Conn::dial(const std::string& host_port, int timeout_ms, std::string& err) {
    close();
    if (!net_init()) {
        err = "the sockets did not start (WSAStartup)";
        return false;
    }
    std::string host, port;
    if (!split(host_port, host, port) || host.empty() || std::atoi(port.c_str()) == 0) {
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
        bool up = ::connect(s, a->ai_addr, (int) a->ai_addrlen) == 0;
        if (!up && in_progress(last_err()) && wait_sock(s, false, timeout_ms)) {
            int so = 0;
            socklen_t sl = sizeof so;
            up = getsockopt(s, SOL_SOCKET, SO_ERROR, (char*) &so, &sl) == 0 && so == 0;
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
    adopt((std::intptr_t) s, host_port);
    return true;
}

bool Conn::busy_poll(int us, std::string& err) {
#if defined(SO_BUSY_POLL)
    if (fd_ < 0) {
        err = "not connected";
        return false;
    }
    if (setsockopt((sock_t) fd_, SOL_SOCKET, SO_BUSY_POLL, (const char*) &us, sizeof us) != 0) {
        err = err_text("SO_BUSY_POLL");
        return false;
    }
    int got = -1;
    socklen_t sl = sizeof got;
    getsockopt((sock_t) fd_, SOL_SOCKET, SO_BUSY_POLL, (char*) &got, &sl);
    if (got != us) {
        err = "SO_BUSY_POLL reads back " + std::to_string(got);
        return false;
    }
    return true;
#else
    (void) us;
    err = "no SO_BUSY_POLL on this system";
    return false;
#endif
}

bool Conn::rx_stamps(std::string& err) {
#if defined(SO_TIMESTAMPNS)
    if (fd_ < 0) {
        err = "not connected";
        return false;
    }
    const int on = 1;
    if (setsockopt((sock_t) fd_, SOL_SOCKET, SO_TIMESTAMPNS, (const char*) &on, sizeof on) != 0) {
        err = err_text("SO_TIMESTAMPNS");
        return false;
    }
    stamps_ = true;
    return true;
#else
    err = "no SO_TIMESTAMPNS on this system";
    return false;
#endif
}

int Conn::read_(uint8_t* p, size_t len) {
#if defined(SO_TIMESTAMPNS)
    if (stamps_) {   // (TCP gives the stamp of the last segment this read took)
        iovec iov{p, len};
        alignas(cmsghdr) char ctl[CMSG_SPACE(sizeof(timespec))];
        msghdr m{};
        m.msg_iov = &iov;
        m.msg_iovlen = 1;
        m.msg_control = ctl;
        m.msg_controllen = sizeof ctl;
        const int k = (int) ::recvmsg((sock_t) fd_, &m, kRecvFlags);
        if (k > 0) rx_ns_ = 0;   // (a read without a stamp leaves none, never the one before)
        if (k > 0)
            for (cmsghdr* c = CMSG_FIRSTHDR(&m); c != nullptr; c = CMSG_NXTHDR(&m, c))
                if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_TIMESTAMPNS) {
                    timespec ts;
                    std::memcpy(&ts, CMSG_DATA(c), sizeof ts);
                    rx_ns_ = (int64_t) ts.tv_sec * 1000000000 + ts.tv_nsec;
                }
        return k;
    }
#endif
    return (int) ::recv((sock_t) fd_, (char*) p, (int) len, kRecvFlags);
}

bool Conn::send_raw_(const uint8_t* p, size_t len, std::string& err) {
    const double t0 = now_ms(), give_up = t0 + 30000.0;   // (a full buffer for 30 s: the peer is gone)
    size_t sent = 0;
    while (sent < len) {
        size_t want = len - sent;
        if (pace_bps > 0.0) {   // (a paced socket: 64 KB at a time, never ahead of the rate)
            want = std::min<size_t>(want, (size_t) 64 << 10);
            const double due = t0 + 1000.0 * (double) sent / pace_bps;
            for (double now = now_ms(); now < due; now = now_ms())
                if (due - now > 2.0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                else relax();
        }
        const int k = (int) ::send((sock_t) fd_, (const char*) p + sent, (int) std::min<size_t>(want, (size_t) 1 << 30), kSendFlags);
        if (k > 0) {
            sent += (size_t) k;
            stat.bytes_out += (uint64_t) k;
            continue;
        }
        if (k < 0 && would_block(last_err()) && now_ms() < give_up) {
            (void) wait_sock((sock_t) fd_, false, 100);
            continue;
        }
        err = k == 0 ? "the link closed" : err_text("send");
        close();
        return false;
    }
    return true;
}

bool Conn::send(EpHdr h, const void* p, size_t n, std::string& err, const void* p2, size_t n2) {
    if (fd_ < 0) {
        err = "not connected";
        return false;
    }
    h.magic = kMagic;
    h.bytes = (uint32_t) (n + n2);
    h.t_ns = now_ns();
    tx_.resize(sizeof h + n + n2);
    std::memcpy(tx_.data(), &h, sizeof h);
    if (n) std::memcpy(tx_.data() + sizeof h, p, n);
    if (n2) std::memcpy(tx_.data() + sizeof h + n, p2, n2);
    if (!send_raw_(tx_.data(), tx_.size(), err)) return false;
    ++stat.frames_out;
    last_frame_ = now_ms();
    return true;
}

int Conn::recv(EpHdr& h, std::vector<uint8_t>& payload, double deadline, std::string& err) {
    if (fd_ < 0) {
        err = "not connected";
        return -1;
    }
    uint8_t tmp[64 * 1024];
    for (;;) {
        const size_t have = rb_.size() - rd_;
        if (have >= sizeof(EpHdr)) {
            std::memcpy(&h, rb_.data() + rd_, sizeof h);
            if (h.magic != kMagic || h.bytes > max_frame) {
                err = h.magic != kMagic ? "a frame with a bad magic (the link is out of step)"
                                        : "a frame of " + std::to_string(h.bytes) + " bytes, over " + std::to_string(max_frame);
                close();
                return -1;
            }
            if (have >= sizeof(EpHdr) + h.bytes) {
                const uint8_t* b = rb_.data() + rd_ + sizeof(EpHdr);
                payload.assign(b, b + h.bytes);
                rd_ += sizeof(EpHdr) + h.bytes;
                if (rd_ == rb_.size()) {
                    rb_.clear();
                    rd_ = 0;
                }
                ++stat.frames_in;
                last_frame_ = now_ms();
                last_rx_ns_ = rx_ns_;
                return 1;
            }
        }
        const int k = read_(tmp, sizeof tmp);
        if (k > 0) {
            if (rd_ > 0 && rd_ == rb_.size()) {
                rb_.clear();
                rd_ = 0;
            }
            rb_.insert(rb_.end(), tmp, tmp + k);
            stat.bytes_in += (uint64_t) k;
#if defined(TCP_QUICKACK)
            set_opt<int>((sock_t) fd_, IPPROTO_TCP, TCP_QUICKACK, 1);   // (the kernel turns it off again by itself)
#endif
            continue;
        }
        if (k == 0 || !would_block(last_err())) {
            err = (k < 0 ? err_text("recv") : std::string("the link closed")) + (buffered() > 0 ? " mid-frame" : "");
            close();
            return -1;
        }
        const double now = now_ms();
        if (deadline >= 0 && now >= deadline) return 0;
        if (spin_ms > 0 && now - last_frame_ < spin_ms) {   // (inside the budget: spin, the core stays awake)
            ++stat.spins;
            relax();
            continue;
        }
        ++stat.polls;   // (quiet: wait in poll, at most 50 ms at a time)
        const double left = deadline < 0 ? 50.0 : std::min(50.0, deadline - now);
        (void) wait_sock((sock_t) fd_, true, (int) std::max(0.0, left + 0.999));
    }
}

// ---------------------------------------------------------------------------------------------------- Listener
Listener::~Listener() { close(); }

void Listener::close() {
    if (fd_ >= 0) close_sock((sock_t) fd_);
    fd_ = -1;
}

bool Listener::open(const std::string& at, std::string& err) {
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
    const sock_t l = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (l == kBad) {
        freeaddrinfo(res);
        err = err_text("socket");
        return false;
    }
    set_opt<int>(l, SOL_SOCKET, SO_REUSEADDR, 1);
    const bool bound = ::bind(l, res->ai_addr, (int) res->ai_addrlen) == 0 && ::listen(l, 4) == 0;
    freeaddrinfo(res);
    if (!bound) {
        err = err_text(("listening on " + at).c_str());
        close_sock(l);
        return false;
    }
    sockaddr_in sa{};
    socklen_t sl = sizeof sa;
    getsockname(l, (sockaddr*) &sa, &sl);
    port_ = ntohs(sa.sin_port);
    fd_ = (std::intptr_t) l;
    return true;
}

bool Listener::accept(Conn& c, int wait_ms, std::string& err) {
    if (fd_ < 0) {
        err = "not listening";
        return false;
    }
    if (!wait_sock((sock_t) fd_, true, wait_ms)) {
        err = "no connection on port " + std::to_string(port_) + " in " + std::to_string(wait_ms) + " ms";
        return false;
    }
    sockaddr_storage sa{};
    socklen_t sl = sizeof sa;
    const sock_t s = ::accept((sock_t) fd_, (sockaddr*) &sa, &sl);
    if (s == kBad) {
        err = err_text("accept");
        return false;
    }
    char hb[64] = {};
    if (sa.ss_family == AF_INET) inet_ntop(AF_INET, &((sockaddr_in*) &sa)->sin_addr, hb, sizeof hb);
    c.adopt((std::intptr_t) s, hb);
    return true;
}

// ---------------------------------------------------------------------------------------------------- the hellos
namespace {

bool send_error(Conn& c, const std::string& why) {
    std::string e;
    EpHdr h;
    h.type = Error;
    return c.send(h, why.data(), why.size(), e);
}

// a hello frame (type want) into hello and table; false (err) for anything else, an Error's text included
bool read_hello(Conn& c, uint16_t want, EpHello& hello, std::vector<EpSlice>* table, double timeout_ms, const char* who,
                std::string& err) {
    EpHdr h;
    std::vector<uint8_t> p;
    const int got = c.recv(h, p, now_ms() + timeout_ms, err);
    if (got == 0) err = std::string("no hello from ") + who + " in " + std::to_string((int) timeout_ms) + " ms";
    if (got != 1) return false;
    if (h.type == Error) {
        err = std::string(who) + " refused: " + std::string(p.begin(), p.end());
        return false;
    }
    if (h.type != want || p.size() < sizeof(EpHello)) {
        err = std::string("not a hello from ") + who;
        return false;
    }
    std::memcpy(&hello, p.data(), sizeof hello);
    if (table != nullptr) {
        const size_t extra = p.size() - sizeof(EpHello);
        if (extra % sizeof(EpSlice) != 0) {
            err = std::string("a malformed slice table from ") + who;
            return false;
        }
        table->resize(extra / sizeof(EpSlice));
        if (extra) std::memcpy(table->data(), p.data() + sizeof(EpHello), extra);
    }
    return true;
}

bool send_hello(Conn& c, uint16_t type, const EpHello& hello, const std::vector<EpSlice>* table, std::string& err) {
    EpHdr h;
    h.type = type;
    return c.send(h, &hello, sizeof hello, err, table && !table->empty() ? table->data() : nullptr,
                  table ? table->size() * sizeof(EpSlice) : 0);
}

}  // namespace

bool hello_worker(Conn& c, const EpHello& mine, const std::vector<EpSlice>& table, Peer& main, std::string& err,
                  double timeout_ms) {
    if (!send_hello(c, Hello, mine, &table, err)) return false;
    if (!read_hello(c, HelloOk, main.hello, &main.table, timeout_ms, "the main", err)) {
        c.close();
        return false;
    }
    const std::string why = worker_check(mine, main.hello);
    if (!why.empty()) {
        (void) send_error(c, why);
        err = "the main " + c.peer() + " refused here: " + why;
        c.close();
        return false;
    }
    EpHdr ok;
    ok.type = HelloOk;
    return c.send(ok, nullptr, 0, err);
}

bool hello_main(Conn& c, EpHello& mine, const std::vector<EpSlice>& table, const std::vector<int>& k, Peer& worker,
                std::string& err, double timeout_ms) {
    if (!read_hello(c, Hello, worker.hello, &worker.table, timeout_ms, "the worker", err)) {
        c.close();
        return false;
    }
    const std::string why = main_check(mine, table, k, worker.hello, worker.table);
    if (!why.empty()) {
        (void) send_error(c, why);
        err = "worker " + c.peer() + " refused: " + why;
        c.close();
        return false;
    }
    if (mine.session == 0) mine.session = std::random_device{}() | ((uint64_t) std::random_device{}() << 32) | 1u;
    if (!send_hello(c, HelloOk, mine, &table, err)) return false;
    EpHdr h;
    std::vector<uint8_t> p;
    const int got = c.recv(h, p, now_ms() + timeout_ms, err);
    if (got == 1 && h.type == HelloOk && p.empty()) return true;
    if (got == 1) err = h.type == Error ? "the worker refused: " + std::string(p.begin(), p.end()) : "no agreement from the worker";
    else if (got == 0) err = "no agreement from the worker in " + std::to_string((int) timeout_ms) + " ms";
    c.close();
    return false;
}

bool bulk_worker(Conn& c, const EpHello& mine, uint64_t session, std::string& err, double timeout_ms) {
    EpHello b = mine;
    b.channel = 1;
    b.session = session;
    if (!send_hello(c, Hello, b, nullptr, err)) return false;
    EpHello m;
    if (!read_hello(c, HelloOk, m, nullptr, timeout_ms, "the main (bulk)", err)) {
        c.close();
        return false;
    }
    return true;
}

bool bulk_main(Conn& c, const EpHello& mine, std::string& err, double timeout_ms) {
    EpHello w;
    if (!read_hello(c, Hello, w, nullptr, timeout_ms, "the worker (bulk)", err)) {
        c.close();
        return false;
    }
    EpHello b = mine;
    b.channel = 1;
    std::string why = common_check(b, w);
    if (why.empty() && w.session != mine.session) why = "the bulk socket names another session";
    if (!why.empty()) {
        (void) send_error(c, why);
        err = "bulk socket from " + c.peer() + " refused: " + why;
        c.close();
        return false;
    }
    c.max_frame = (size_t) 1 << 26;   // (a fill carries an expert's blob)
    return send_hello(c, HelloOk, b, nullptr, err);
}

// ---------------------------------------------------------------------------------------------------- the main's end
void MainLink::start(const EpHello& mine) {
    n_embd_ = mine.n_embd;
    k_ = mine.k_picks;
    n_layers_ = mine.n_layers;
    out_.clear();
    up_ = req.ok();
    any_sent_ = fill_bad_ = false;
    last_seq_ = ready_ = filling_ = 0;
    fills_out_ = in_a_row_ = 0;
}

void MainLink::down_(const std::string& why, std::string& err) {
    err = why;
    if (up_) ++st.downs;
    up_ = false;
    out_.clear();
    req.close();
    bulk.close();
}

int MainLink::take_(double deadline, Out& o, Reply& r, std::string& err) {
    for (;;) {
        EpHdr h;
        const int got = req.recv(h, rx_, deadline, err);
        if (got <= 0) {
            if (got < 0) down_(err, err);
            return got;
        }
        if (h.type == Pong) continue;   // (a late probe's)
        if (h.type == Error || h.type == Bye) {
            down_(h.type == Bye ? "the worker said bye" : "the worker: " + std::string(rx_.begin(), rx_.end()), err);
            return -1;
        }
        if (h.type != Resp || out_.empty() || h.seq != out_.front().seq) {
            down_(h.type != Resp ? "a frame of type " + std::to_string(h.type) + " where a reply was due"
                                 : "a reply for seq " + std::to_string(h.seq) + " while " +
                                       (out_.empty() ? std::string("none was out") : "seq " + std::to_string(out_.front().seq) + " was the oldest"),
                  err);
            return -1;
        }
        o = out_.front();
        uint16_t mask = 0, status = 0;
        if (rx_.size() >= 4) {
            std::memcpy(&mask, rx_.data(), 2);
            std::memcpy(&status, rx_.data() + 2, 2);
        }
        int rows = 0;
        for (int t = 0; t < 16; ++t) rows += (mask >> t) & 1;
        if (rx_.size() < 4 || (mask >> o.T) != 0 || rx_.size() != 4 + (size_t) rows * n_embd_ * 2) {
            down_("a reply of the wrong size for seq " + std::to_string(h.seq), err);
            return -1;
        }
        out_.pop_front();
        if (o.dropped) ++st.late_dropped;   // (an earlier request's, computed here by now)
        r.seq = h.seq;
        r.rowmask = mask;
        r.status = status;
        r.lane_us = h.aux;
        r.rows = (const uint16_t*) (rx_.data() + 4);
        return 1;
    }
}

int MainLink::drain_(std::string& err) {
    while (up_ && !out_.empty() && out_.front().dropped) {
        Out o;
        Reply r;
        const int got = take_(now_ms(), o, r, err);   // (no wait: only what is in already)
        if (got <= 0) return got;
    }
    return 1;
}

int MainLink::send_req(uint32_t seq, int layer, int T, uint32_t epoch, uint32_t flags, const EpPick* picks,
                       const uint16_t* x, std::string& err) {
    if (up_ && drain_(err) < 0) return -1;   // (late replies that came in since: their room back)
    const bool full = (int) out_.size() >= kOutstanding;
    const char* why = !up_                          ? "the link is down"
                      : full                        ? "two requests out already"
                      : any_sent_ && seq <= last_seq_ ? "a seq not after the last one sent"
                      : T < 1 || T > kMaxT || layer < 0 || layer >= n_layers_ ? "a layer or T out of range"
                      : epoch > ready_                ? "an epoch the worker has not loaded"
                                                      : nullptr;
    if (why != nullptr) {
        err = why;
        ++st.not_sent;
        // (no room because both requests out missed their deadlines and no reply came since: one more miss)
        if (full && out_.front().dropped && out_.back().dropped && ++in_a_row_ >= kDownAfter) {
            down_(std::to_string(kDownAfter) + " replies in a row missed their deadline", err);
            return 0;
        }
        return 0;
    }
    EpHdr h;
    h.type = Req;
    h.layer = (uint8_t) layer;
    h.T = (uint8_t) T;
    h.seq = seq;
    h.epoch = epoch;
    h.aux = flags;
    if (!req.send(h, picks, (size_t) T * k_ * sizeof(EpPick), err, x, (size_t) T * n_embd_ * 2)) {
        down_(err, err);
        return -1;
    }
    out_.push_back({seq, T, false});
    any_sent_ = true;
    last_seq_ = seq;
    ++st.sent;
    return 1;
}

int MainLink::wait_resp(uint32_t seq, double deadline, Reply& r, std::string& err) {
    if (!up_) {
        err = "the link is down";
        return -1;
    }
    auto it = out_.begin();
    while (it != out_.end() && it->dropped) ++it;
    if (it == out_.end() || it->seq != seq) {   // (a caller's mistake: the link itself is fine)
        err = "seq " + std::to_string(seq) + " is not the oldest request awaited";
        return -1;
    }
    for (;;) {
        Out o;
        const int got = take_(deadline, o, r, err);
        if (got < 0) return -1;
        if (got == 0) {
            for (Out& x : out_)
                if (x.seq == seq) x.dropped = true;   // (computed here now; its reply is dropped when it comes)
            ++st.timeouts;
            if (++in_a_row_ >= kDownAfter) {
                down_(std::to_string(kDownAfter) + " replies in a row missed their deadline", err);
                return -1;
            }
            err = "no reply by the deadline";
            return 0;
        }
        if (o.dropped) continue;
        in_a_row_ = 0;
        ++st.replies;
        return 1;
    }
}

bool MainLink::wake(std::string& err) {
    if (!up_) {
        err = "the link is down";
        return false;
    }
    EpHdr h;
    h.type = Ping;
    h.aux = kPingWake;
    if (req.send(h, nullptr, 0, err)) return true;
    down_(err, err);
    return false;
}

int MainLink::ping(uint32_t seq, size_t bytes, double deadline, uint32_t& held_ns, std::string& err,
                   std::vector<uint8_t>* payload, uint32_t aux) {
    if (!up_ || !out_.empty()) {
        err = !up_ ? "the link is down" : "a request is out";
        return -1;
    }
    if (payload == nullptr) tx_.assign(bytes, 0);
    EpHdr h;
    h.type = Ping;
    h.seq = seq;
    h.aux = aux & ~kPingWake;
    const std::vector<uint8_t>& p = payload != nullptr ? *payload : tx_;
    if (!req.send(h, p.data(), p.size(), err)) {
        down_(err, err);
        return -1;
    }
    for (;;) {
        EpHdr g;
        const int got = req.recv(g, rx_, deadline, err);
        if (got < 0) {
            down_(err, err);
            return -1;
        }
        if (got == 0) return 0;
        if (g.type == Pong && g.seq == seq) {
            held_ns = g.aux;
            if (payload != nullptr) *payload = rx_;
            return 1;
        }
        if (g.type == Error) {
            down_("the worker: " + std::string(rx_.begin(), rx_.end()), err);
            return -1;
        }
    }
}

void MainLink::bye() {
    std::string e;
    EpHdr h;
    h.type = Bye;
    if (req.ok()) (void) req.send(h, nullptr, 0, e);
    up_ = false;
    out_.clear();
    req.close();
    bulk.close();
}

bool MainLink::send_fill(int layer, int e_in, int e_out, uint32_t epoch, const void* blob, size_t n, std::string& err) {
    if (epoch != ready_ + 1 || (fills_out_ > 0 && epoch != filling_)) {
        err = "a fill for epoch " + std::to_string(epoch) + " while " + std::to_string(ready_) + " is loaded";
        return false;
    }
    EpHdr h;
    h.type = Fill;
    h.layer = (uint8_t) layer;
    h.epoch = epoch;
    h.aux = (uint32_t) e_in;
    EpFill f;
    f.e_in = e_in;
    f.e_out = e_out;
    if (!bulk.send(h, &f, sizeof f, err, blob, n)) return false;
    if (fills_out_ == 0) fill_bad_ = false;   // (a new batch for the epoch)
    filling_ = epoch;
    ++fills_out_;
    ++st.fills;
    return true;
}

int MainLink::poll_fill_ok(double deadline, int& layer, int& e_in, uint32_t& epoch, uint16_t& status, std::string& err) {
    EpHdr h;
    const int got = bulk.recv(h, brx_, deadline, err);
    if (got <= 0) return got;
    if (h.type != FillOk || brx_.size() != 2 || fills_out_ == 0 || h.epoch != filling_) {
        err = h.type == Error ? "the worker (bulk): " + std::string(brx_.begin(), brx_.end()) : "a bulk frame out of step";
        bulk.close();
        return -1;
    }
    layer = h.layer;
    e_in = (int) h.aux;
    epoch = h.epoch;
    std::memcpy(&status, brx_.data(), 2);
    if (status != 0) {
        ++st.fill_fails;
        fill_bad_ = true;
    }
    // (every fill of the batch loaded: the epoch may be used; one not loaded: it may not, and the next batch is for the
    // same epoch again)
    if (--fills_out_ == 0 && !fill_bad_) ready_ = filling_;
    return 1;
}

// ---------------------------------------------------------------------------------------------------- the worker's end
void WorkerLink::start(const EpHello& mine) {
    n_embd_ = mine.n_embd;
    k_ = mine.k_picks;
    n_layers_ = mine.n_layers;
    pending_ = any_ = false;
    last_seq_ = 0;
    epoch_used = 0;
    epoch_acked.store(0);
}

int WorkerLink::refuse_(const std::string& why, std::string& err) {
    err = why;
    if (req.ok()) (void) send_error(req, why);
    req.close();
    return -1;
}

int WorkerLink::next(Frame& f, double deadline, std::string& err) {
    if (pending_) {
        err = "the last request was not answered";
        return -1;
    }
    for (;;) {
        EpHdr h;
        const int got = req.recv(h, rx_, deadline, err);
        if (got <= 0) return got;
        const double t_in = now_ms();
        if (h.type == Ping) {
            if (h.aux & kPingWake) {
                ++st.wakes;
                if (on_wake) on_wake();
                continue;
            }
            ++st.pings;
            if (on_ping) on_ping(h, rx_);
            EpHdr p;
            p.type = Pong;
            p.seq = h.seq;
            p.aux = (uint32_t) std::min(4.0e9, (now_ms() - t_in) * 1.0e6);
            if (!req.send(p, rx_.data(), rx_.size(), err)) return -1;
            continue;
        }
        if (h.type == Bye) {
            err = "the main said bye";
            req.close();
            return -1;
        }
        if (h.type == Error) {
            err = "the main: " + std::string(rx_.begin(), rx_.end());
            req.close();
            return -1;
        }
        if (h.type != Req) return refuse_("a frame of type " + std::to_string(h.type) + " on the request socket", err);
        if (any_ && h.seq <= last_seq_)
            return refuse_("request seq " + std::to_string(h.seq) + " after " + std::to_string(last_seq_), err);
        if (h.T < 1 || h.T > kMaxT || h.layer >= n_layers_) return refuse_("a request's layer or T out of range", err);
        const uint32_t acked = epoch_acked.load(std::memory_order_acquire);
        if (h.epoch < epoch_used || h.epoch > acked)
            return refuse_("a request tagged with epoch " + std::to_string(h.epoch) + " (in use " + std::to_string(epoch_used) +
                               ", acked " + std::to_string(acked) + ")",
                           err);
        const size_t np = (size_t) h.T * k_ * sizeof(EpPick), nx = (size_t) h.T * n_embd_ * 2;
        if (rx_.size() != np + nx) return refuse_("a request of the wrong size", err);
        epoch_used = h.epoch;
        any_ = true;
        last_seq_ = h.seq;
        pending_ = true;
        ++st.reqs;
        f.h = h;
        f.T = h.T;
        f.picks = (const EpPick*) rx_.data();
        f.x = (const uint16_t*) (rx_.data() + np);
        return 1;
    }
}

bool WorkerLink::send_resp(uint32_t seq, uint16_t rowmask, uint16_t status, const uint16_t* rows, uint32_t lane_us,
                           std::string& err) {
    if (!pending_ || seq != last_seq_) {
        err = "a reply for seq " + std::to_string(seq) + ", not the request in hand";
        return false;
    }
    pending_ = false;
    int n = 0;
    for (int t = 0; t < 16; ++t) n += (rowmask >> t) & 1;
    uint16_t head[2] = {rowmask, status};
    EpHdr h;
    h.type = Resp;
    h.seq = seq;
    h.aux = lane_us;
    return req.send(h, head, sizeof head, err, rows, (size_t) n * n_embd_ * 2);
}

int WorkerLink::serve_fill(double deadline, const Loader& load, std::string& err) {
    EpHdr h;
    const int got = bulk.recv(h, brx_, deadline, err);
    if (got <= 0) return got;
    if (h.type == Bye || h.type == Error) {
        err = h.type == Bye ? "the main said bye (bulk)" : "the main (bulk): " + std::string(brx_.begin(), brx_.end());
        bulk.close();
        return -1;
    }
    const uint32_t acked = epoch_acked.load(std::memory_order_acquire);
    std::string why;
    if (h.type != Fill || brx_.size() < sizeof(EpFill)) why = "not a fill on the bulk socket";
    else if (h.epoch != acked && h.epoch != acked + 1)
        why = "a fill for epoch " + std::to_string(h.epoch) + " (acked " + std::to_string(acked) + ")";
    else if (h.layer >= n_layers_) why = "a fill for layer " + std::to_string(h.layer);
    if (!why.empty()) {
        err = why;
        (void) send_error(bulk, why);
        bulk.close();
        return -1;
    }
    EpFill fm;
    std::memcpy(&fm, brx_.data(), sizeof fm);
    const uint16_t status = load ? load(h.layer, fm.e_in, fm.e_out, h.epoch, brx_.data() + sizeof fm, brx_.size() - sizeof fm) : 1;
    ++st.fills;
    if (status == 0) epoch_acked.store(h.epoch, std::memory_order_release);
    EpHdr ok;
    ok.type = FillOk;
    ok.layer = h.layer;
    ok.epoch = h.epoch;
    ok.aux = (uint32_t) fm.e_in;
    return bulk.send(ok, &status, sizeof status, err);
}

}  // namespace strata::net::ep
