// src/net/ep_link.cpp - see ep_link.hpp: the sockets, the hellos, both ends' rules.
#include "ep_link.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
// Winsock's SO_REUSEADDR binds over a port another process is listening on (either one taking the connection);
// SO_EXCLUSIVEADDRUSE refuses that, as Linux does (stage_link's kListenOpt)
constexpr int kListenOpt = SO_EXCLUSIVEADDRUSE;
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
constexpr int kListenOpt = SO_REUSEADDR;   // (a port in TIME_WAIT only; never another listener's)
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

constexpr size_t kCompactAt = (size_t) 1 << 20;   // a read buffer's consumed bytes kept at most (past them: erased)
constexpr size_t kMaxHello = sizeof(EpHello) + (size_t) kMaxLayers * sizeof(EpSlice);   // a hello frame's payload at most

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

namespace {
std::atomic<int> g_token_warnings{0};
void warn_no_token(const EpHello& mine) {
    const std::string w = no_token_warning(mine);
    if (w.empty()) return;
    std::fprintf(stderr, "%s\n", w.c_str());
    g_token_warnings.fetch_add(1);
}
}  // namespace

std::string no_token_warning(const EpHello& mine) {
    if (mine.token[0] != 0) return "";
    return mine.role == 0 ? "ep link - WARNING: no STRATA_STAGE_TOKEN, any host that reaches this port can act as the EP worker"
                          : "ep link - WARNING: no STRATA_STAGE_TOKEN, any host this worker dials can act as the main";
}
int token_warnings() { return g_token_warnings.load(); }

bool split_host_port(const std::string& at, std::string& host, std::string& port) {
    if (!at.empty() && at[0] == '[') {   // [V6]:PORT
        const size_t e = at.find(']');
        if (e == std::string::npos || e + 1 >= at.size() || at[e + 1] != ':') return false;
        host = at.substr(1, e - 1);
        port = at.substr(e + 2);
    } else {
        const size_t c = at.rfind(':');
        if (c != std::string::npos && at.find(':') != c) return false;   // (a bare IPv6 address: brackets needed)
        host = c == std::string::npos ? "" : at.substr(0, c);
        port = c == std::string::npos ? at : at.substr(c + 1);
    }
    if (port.empty() || port.find_first_not_of("0123456789") != std::string::npos) return false;
    const int p = std::atoi(port.c_str());
    return p >= 0 && p < 65536;
}

std::string FillBatch::take(uint32_t e, int i, int nn, uint32_t acked) {
    if (e != acked + 1)
        return "a fill for epoch " + std::to_string(e) + " (acked " + std::to_string(acked) + ": only " + std::to_string(acked + 1) + " is filled)";
    if (nn < 1 || i < 0 || i >= nn) return "a fill " + std::to_string(i) + " of " + std::to_string(nn);
    if (i == 0) {   // (a batch starts: the first, or a retry of one that did not load whole)
        epoch = e;
        n = nn;
        next = 0;
        bad = false;
    } else if (e != epoch || nn != n || i != next) {
        return "fill " + std::to_string(i) + " of " + std::to_string(nn) + " for epoch " + std::to_string(e) + " out of its batch's order (next: " +
               std::to_string(next) + " of " + std::to_string(n) + " for epoch " + std::to_string(epoch) + ")";
    }
    return "";
}

bool FillBatch::done(bool loaded) {
    if (!loaded) bad = true;
    ++next;
    return next == n && !bad;
}

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

void Conn::reset() {
    if (fd_ >= 0) {
        linger l{};
        l.l_onoff = 1;
        l.l_linger = 0;
        setsockopt((sock_t) fd_, SOL_SOCKET, SO_LINGER, (const char*) &l, sizeof l);
    }
    close();
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
    if (!split_host_port(host_port, host, port) || host.empty() || std::atoi(port.c_str()) == 0) {
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
        if (k > 0 && drop_stamps > 0) --drop_stamps;
        else if (k > 0)
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
            } else if (rd_ >= kCompactAt) {   // (reads that end inside a frame, one after another: drop what was taken)
                rb_.erase(rb_.begin(), rb_.begin() + (std::ptrdiff_t) rd_);
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
    if (!split_host_port(at, host, port)) {
        err = "expected [BIND:]PORT, not " + at;
        return false;
    }
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = host.find(':') != std::string::npos ? AF_INET6 : AF_INET;   // (no BIND: every IPv4 address)
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
    set_opt<int>(l, SOL_SOCKET, kListenOpt, 1);
    const bool bound = ::bind(l, res->ai_addr, (int) res->ai_addrlen) == 0 && ::listen(l, 4) == 0;
    freeaddrinfo(res);
    if (!bound) {
        err = err_text(("listening on " + at).c_str());
        close_sock(l);
        return false;
    }
    sockaddr_storage sa{};
    socklen_t sl = sizeof sa;
    getsockname(l, (sockaddr*) &sa, &sl);
    port_ = ntohs(sa.ss_family == AF_INET6 ? ((sockaddr_in6*) &sa)->sin6_port : ((sockaddr_in*) &sa)->sin_port);
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
    if (sa.ss_family == AF_INET6) inet_ntop(AF_INET6, &((sockaddr_in6*) &sa)->sin6_addr, hb, sizeof hb);
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

// a hello frame (type want) into hello and table: 1; 0 failed with nothing to tell the peer (the link failed, no
// hello in time, the peer's own Error: err); 2 refused, err to be told to the peer.  The frame may be no larger than
// a hello (a peer no token admitted yet: its sizes are not trusted); its magic and protocol are judged before its
// size, so an older or newer peer is refused for its protocol.
int read_hello(Conn& c, uint16_t want, EpHello& hello, std::vector<EpSlice>* table, double timeout_ms, const char* who,
               std::string& err) {
    EpHdr h;
    std::vector<uint8_t> p;
    const size_t cap = c.max_frame;
    c.max_frame = kMaxHello;
    const int got = c.recv(h, p, now_ms() + timeout_ms, err);
    c.max_frame = cap;
    if (got == 0) err = std::string("no hello from ") + who + " in " + std::to_string((int) timeout_ms) + " ms";
    if (got != 1) return 0;
    if (h.type == Error) {
        err = std::string(who) + " refused: " + std::string(p.begin(), p.end());
        return 0;
    }
    uint32_t head[2] = {0, 0};   // (magic, proto: an EpHello's first 8 bytes)
    if (p.size() >= sizeof head) std::memcpy(head, p.data(), sizeof head);
    if (h.type != want) err = std::string("a frame of type ") + std::to_string(h.type) + " where the hello of " + who + " was due";
    else if (p.size() < sizeof head || head[0] != kMagic) err = std::string("not an EP link's hello from ") + who;
    else if (head[1] != kProto) err = "EP protocol " + std::to_string(head[1]) + ", this side speaks " + std::to_string(kProto);
    else if (p.size() < sizeof(EpHello)) err = std::string("a short hello from ") + who;
    else if (table != nullptr && (p.size() - sizeof(EpHello)) % sizeof(EpSlice) != 0) err = std::string("a malformed slice table from ") + who;
    else if (table == nullptr && p.size() != sizeof(EpHello)) err = std::string("a bulk hello with a table from ") + who;
    else {
        std::memcpy(&hello, p.data(), sizeof hello);
        if (table != nullptr) {
            const size_t extra = p.size() - sizeof(EpHello);
            table->resize(extra / sizeof(EpSlice));
            if (extra) std::memcpy(table->data(), p.data() + sizeof(EpHello), extra);
        }
        return 1;
    }
    return 2;
}

// a refusal: said to the peer where the socket takes it, then closed
bool refuse(Conn& c, const std::string& why, const std::string& what, std::string& err) {
    (void) send_error(c, why);
    err = what + why;
    c.close();
    return false;
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
    warn_no_token(mine);
    if (!send_hello(c, Hello, mine, &table, err)) return false;
    const int got = read_hello(c, HelloOk, main.hello, &main.table, timeout_ms, "the main", err);
    if (got == 2) return refuse(c, err, "the main " + c.peer() + " refused here: ", err);
    if (got != 1) {
        c.close();
        return false;
    }
    const std::string why = worker_check(mine, main.hello);
    if (!why.empty()) return refuse(c, why, "the main " + c.peer() + " refused here: ", err);
    EpHdr ok;
    ok.type = HelloOk;
    return c.send(ok, nullptr, 0, err);
}

bool hello_main(Conn& c, EpHello& mine, const std::vector<EpSlice>& table, const std::vector<int>& k, Peer& worker,
                std::string& err, double timeout_ms) {
    warn_no_token(mine);
    const int got = read_hello(c, Hello, worker.hello, &worker.table, timeout_ms, "the worker", err);
    if (got == 2) return refuse(c, err, "worker " + c.peer() + " refused: ", err);
    if (got != 1) {
        c.close();
        return false;
    }
    const std::string why = main_check(mine, table, k, worker.hello, worker.table);
    if (!why.empty()) return refuse(c, why, "worker " + c.peer() + " refused: ", err);
    // (a fresh session on every connection: a bulk socket left from an earlier one is not this one's)
    mine.session = std::random_device{}() | ((uint64_t) std::random_device{}() << 32) | 1u;
    if (!send_hello(c, HelloOk, mine, &table, err)) return false;
    EpHdr h;
    std::vector<uint8_t> p;
    const int agreed = c.recv(h, p, now_ms() + timeout_ms, err);
    if (agreed == 1 && h.type == HelloOk && p.empty()) return true;
    if (agreed == 1) err = h.type == Error ? "the worker refused: " + std::string(p.begin(), p.end()) : "no agreement from the worker";
    else if (agreed == 0) err = "no agreement from the worker in " + std::to_string((int) timeout_ms) + " ms";
    c.close();
    return false;
}

bool bulk_worker(Conn& c, const EpHello& mine, uint64_t session, std::string& err, double timeout_ms) {
    EpHello b = mine;
    b.channel = 1;
    b.session = session;
    if (!send_hello(c, Hello, b, nullptr, err)) return false;
    EpHello m;
    const int got = read_hello(c, HelloOk, m, nullptr, timeout_ms, "the main (bulk)", err);
    if (got == 2) return refuse(c, err, "the main (bulk) refused here: ", err);
    if (got != 1) {
        c.close();
        return false;
    }
    std::string why = worker_check(b, m);
    if (why.empty() && m.session != session) why = "the main's bulk answer names another session";
    if (!why.empty()) return refuse(c, why, "the main (bulk) refused here: ", err);
    c.max_frame = kBulkFrame;   // (a fill carries an expert's blob)
    c.spin_ms = 0.0;            // (keep-awake is for the request socket: a bulk thread waits in poll)
    return true;
}

bool bulk_main(Conn& c, const EpHello& mine, std::string& err, double timeout_ms) {
    EpHello w;
    std::vector<EpSlice> t;   // (a request socket's hello carries one: it is told a worker is connected already)
    const int got = read_hello(c, Hello, w, &t, timeout_ms, "the worker (bulk)", err);
    if (got == 2) return refuse(c, err, "bulk socket from " + c.peer() + " refused: ", err);
    if (got != 1) {
        c.close();
        return false;
    }
    EpHello b = mine;
    b.channel = 1;
    std::string why = w.channel == 0 && w.magic == kMagic ? "a worker is already connected" : common_check(b, w);
    if (why.empty() && w.session != mine.session) why = "unknown session (the bulk socket names another link's)";
    if (!why.empty()) return refuse(c, why, "bulk socket from " + c.peer() + " refused: ", err);
    c.max_frame = kBulkFrame;   // (FillOks only come in, but both ends agree)
    c.spin_ms = 0.0;
    return send_hello(c, HelloOk, b, nullptr, err);
}

// ---------------------------------------------------------------------------------------------------- the main's end
MainLink::~MainLink() { stop_fills_(); }

// (mine: the hello hello_main sent, with the session it chose: accept_bulk matches the bulk socket by it)
void MainLink::start(const EpHello& mine) {
    stop_fills_();
    bulk.close();
    mine_ = mine;
    n_embd_ = mine.n_embd;
    k_ = mine.k_picks;
    n_layers_ = mine.n_layers;
    out_.clear();
    up_ = req.ok();
    any_sent_ = false;
    last_seq_ = last_epoch_ = 0;
    in_a_row_ = 0;
    q_epoch_ = 0;
    q_n_ = q_next_ = 0;
    jobs_.clear();
    bulk_why_.clear();
    stop_ = false;
    bulk_failed_ = false;
    ready_ = 0;
    pending_ = 0;
}

void MainLink::stop_fills_() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (fill_th_.joinable()) fill_th_.join();
}

bool MainLink::accept_bulk(Listener& ls, int wait_ms, std::string& err) {
    if (!up_) {
        err = "the link is down";
        return false;
    }
    if (fill_th_.joinable()) {
        err = "the bulk socket is up already";
        return false;
    }
    const double end = now_ms() + wait_ms;
    std::string last;
    for (;;) {
        const double left = end - now_ms();
        if (left <= 0.0) {
            err = "no bulk socket for this link in " + std::to_string(wait_ms) + " ms" + (last.empty() ? "" : " (last: " + last + ")");
            return false;
        }
        if (!ls.accept(bulk, (int) std::ceil(left), err)) {   // (it waited what was left, or the listener failed)
            if (!last.empty()) err += " (last: " + last + ")";
            return false;
        }
        // (each connection's hello within what is left; a stranger's costs the wait, never the link)
        if (bulk_main(bulk, mine_, last, std::max(1.0, std::min(10000.0, end - now_ms())))) break;
    }
    stop_ = false;
    fill_th_ = std::thread([this] { fill_loop_(); });
    return true;
}

bool MainLink::refuse_extra(Listener& ls, int wait_ms) {
    Conn c;
    std::string e;
    if (!ls.accept(c, wait_ms, e)) return false;
    EpHello h;
    std::vector<EpSlice> t;
    (void) read_hello(c, Hello, h, &t, 1000.0, "another worker", e);   // (read first: closing on unread bytes is a reset)
    (void) send_error(c, "a worker is already connected");
    c.close();
    return true;
}

void MainLink::down_(const std::string& why, std::string& err) {
    err = why;
    if (up_) ++st.downs;
    up_ = false;
    out_.clear();
    req.close();
    {   // (the fill thread closes the bulk socket itself as it ends)
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
}

bool MainLink::bulk_ok_(std::string& err) {
    if (!bulk_failed_.load(std::memory_order_acquire)) return true;
    std::string why;
    {
        std::lock_guard<std::mutex> g(mu_);
        why = bulk_why_;
    }
    if (up_) down_("the bulk socket: " + why, err);
    else err = "the link is down";
    return false;
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
    if (up_ && !bulk_ok_(err)) {
        ++st.not_sent;
        return 0;
    }
    if (up_ && drain_(err) < 0) return -1;   // (late replies that came in since: their room back)
    // (a worker that holds the oldest request this long is gone, FIN or not: no reply will free the slots)
    if (up_ && !out_.empty() && now_ms() - out_.front().t_sent > kStaleMs) {
        down_("the oldest request out (seq " + std::to_string(out_.front().seq) + ") is over " + std::to_string((int) kStaleMs) +
                  " ms old",
              err);
        ++st.not_sent;
        return 0;
    }
    const bool full = (int) out_.size() >= kOutstanding;
    const char* why = !up_                                                    ? "the link is down"
                      : full                                                  ? "two requests out already"
                      : any_sent_ && seq <= last_seq_                         ? "a seq not after the last one sent"
                      : T < 1 || T > kMaxT || layer < 0 || layer >= n_layers_ ? "a layer or T out of range"
                      : epoch > epoch_ready()                                 ? "an epoch the worker has not loaded"
                      : epoch < last_epoch_                                   ? "an epoch before the last one sent"
                                                                              : nullptr;
    if (why != nullptr) {   // (no room is not a miss: the requests out are, each when its deadline passes)
        err = why;
        ++st.not_sent;
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
    out_.push_back({seq, T, false, now_ms()});
    any_sent_ = true;
    last_seq_ = seq;
    last_epoch_ = epoch;
    ++st.sent;
    return 1;
}

int MainLink::wait_resp(uint32_t seq, double deadline, Reply& r, std::string& err) {
    if (up_ && !bulk_ok_(err)) return -1;
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
                down_(std::to_string(kDownAfter) + " requests in a row dropped at their deadline", err);
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
    if (up_ && !bulk_ok_(err)) return false;
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
    stop_fills_();
    std::string e;
    EpHdr h;
    h.type = Bye;
    if (req.ok()) (void) req.send(h, nullptr, 0, e);
    up_ = false;
    out_.clear();
    req.close();
    bulk.close();
}

bool MainLink::queue_fill(int layer, int e_in, int e_out, uint32_t epoch, int i, int n, const void* blob, size_t bytes,
                          std::string& err) {
    if (up_ && !bulk_ok_(err)) return false;
    const uint32_t ready = epoch_ready();
    if (!up_) err = "the link is down";
    else if (!fill_th_.joinable()) err = "no bulk socket";
    else if (layer < 0 || layer >= n_layers_) err = "a fill for layer " + std::to_string(layer) + " (the link has " + std::to_string(n_layers_) + ")";
    else if (n < 1 || n > 0xffff || i < 0 || i >= n) err = "a fill " + std::to_string(i) + " of " + std::to_string(n);
    else if (i > 0 && (q_epoch_ != epoch || q_n_ != n || q_next_ != i))
        err = "fill " + std::to_string(i) + " of " + std::to_string(n) + " for epoch " + std::to_string(epoch) + " out of its batch's order";
    else if (i == 0 && q_next_ < q_n_) err = "a new batch while fill " + std::to_string(q_next_) + " of " + std::to_string(q_n_) + " is still to come";
    else if (i == 0 && fills_pending() > 0) err = "a new batch while the last one's fills are out";
    else if (i == 0 && epoch != ready + 1)
        err = "a fill for epoch " + std::to_string(epoch) + " while " + std::to_string(ready) + " is loaded (only " + std::to_string(ready + 1) + " is filled)";
    else if (i == 0 && epoch > last_epoch_ + 1)
        err = "a fill for epoch " + std::to_string(epoch) + " before a request tagged " + std::to_string(epoch - 1) +
              " was sent (the worker frees epoch " + std::to_string(epoch - 2) + "'s slots then)";
    else {
        if (i == 0) {
            q_epoch_ = epoch;
            q_n_ = n;
        }
        q_next_ = i + 1;
        Job j;
        j.layer = layer;
        j.e_in = e_in;
        j.e_out = e_out;
        j.i = i;
        j.n = n;
        j.epoch = epoch;
        j.blob = blob;
        j.bytes = bytes;
        pending_.fetch_add(1);
        {
            std::lock_guard<std::mutex> g(mu_);
            jobs_.push_back(j);
        }
        cv_.notify_all();
        return true;
    }
    return false;
}

void MainLink::fail_(const std::string& why) {
    {
        std::lock_guard<std::mutex> g(mu_);
        if (bulk_why_.empty()) bulk_why_ = why;
    }
    bulk_failed_.store(true, std::memory_order_release);
}

// The fill thread: the queued fills out on the bulk socket (paced by bulk.pace_bps), two ahead of their FillOks at
// most; each FillOk in order, the batch's count kept here (one not loaded: the epoch is not ready, the next batch
// fills it again).  It ends at stop_ or on the first failure (fail_: the decode thread takes the link down), and
// closes the bulk socket as it ends.
void MainLink::fill_loop_() {
    std::deque<Job> out;   // sent, no FillOk yet
    bool bad = false;      // a fill of the current batch not loaded
    double last_ok = now_ms();
    std::string err;
    for (;;) {
        Job j;
        bool have = false;
        {
            std::unique_lock<std::mutex> g(mu_);
            if (out.empty()) cv_.wait_for(g, std::chrono::milliseconds(50), [this] { return stop_.load() || !jobs_.empty(); });
            if (stop_) break;
            if (out.size() < 2 && !jobs_.empty()) {
                j = jobs_.front();
                jobs_.pop_front();
                have = true;
            }
        }
        if (have) {
            EpHdr h;
            h.type = Fill;
            h.layer = (uint8_t) j.layer;
            h.epoch = j.epoch;
            h.aux = (uint32_t) j.e_in;
            EpFill f;
            f.e_in = j.e_in;
            f.e_out = j.e_out;
            f.i = (uint16_t) j.i;
            f.n = (uint16_t) j.n;
            if (!bulk.send(h, &f, sizeof f, err, j.blob, j.bytes)) {
                fail_(err);
                break;
            }
            if (out.empty()) last_ok = now_ms();
            out.push_back(j);
            st.fills.fetch_add(1);
            continue;
        }
        if (out.empty()) continue;
        EpHdr h;
        const int got = bulk.recv(h, brx_, now_ms() + 20.0, err);
        if (got < 0) {
            fail_(err);
            break;
        }
        if (got == 0) {
            if (now_ms() - last_ok > 30000.0) {
                fail_("no FillOk in 30 s");
                break;
            }
            continue;
        }
        const Job& o = out.front();
        if (h.type != FillOk || brx_.size() != 2 || h.epoch != o.epoch || h.layer != (uint8_t) o.layer || h.aux != (uint32_t) o.e_in) {
            fail_(h.type == Error ? "the worker: " + std::string(brx_.begin(), brx_.end()) : "a bulk frame out of step");
            break;
        }
        uint16_t status = 0;
        std::memcpy(&status, brx_.data(), 2);
        if (o.i == 0) bad = false;
        if (status != 0) {
            st.fill_fails.fetch_add(1);
            bad = true;
        }
        if (o.i == o.n - 1 && !bad) ready_.store(o.epoch, std::memory_order_release);   // (the whole batch loaded)
        const Job done = o;
        out.pop_front();
        last_ok = now_ms();
        pending_.fetch_sub(1);
        if (on_fill_ok) on_fill_ok(done.layer, done.e_in, done.epoch, status);
    }
    bulk.close();
}

// ---------------------------------------------------------------------------------------------------- the worker's end
void WorkerLink::start(const EpHello& mine) {
    n_embd_ = mine.n_embd;
    k_ = mine.k_picks;
    n_layers_ = mine.n_layers;
    pending_ = any_ = false;
    last_seq_ = 0;
    batch_ = FillBatch();
    epoch_used.store(0);
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
        const uint32_t acked = epoch_acked.load(std::memory_order_acquire), used = epoch_used.load(std::memory_order_relaxed);
        if (h.epoch < used || h.epoch > acked)
            return refuse_("a request tagged with epoch " + std::to_string(h.epoch) + " (in use " + std::to_string(used) +
                               ", acked " + std::to_string(acked) + ")",
                           err);
        const size_t np = (size_t) h.T * k_ * sizeof(EpPick), nx = (size_t) h.T * n_embd_ * 2;
        if (rx_.size() != np + nx) return refuse_("a request of the wrong size", err);
        epoch_used.store(h.epoch, std::memory_order_release);
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
    EpFill fm;
    if (brx_.size() >= sizeof fm) std::memcpy(&fm, brx_.data(), sizeof fm);
    std::string why;
    if (h.type != Fill || brx_.size() < sizeof fm) why = "not a fill on the bulk socket";
    else if (h.layer >= n_layers_) why = "a fill for layer " + std::to_string(h.layer);
    else why = batch_.take(h.epoch, fm.i, fm.n, epoch_acked.load(std::memory_order_acquire));
    if (!why.empty()) {
        err = why;
        (void) send_error(bulk, why);
        bulk.close();
        return -1;
    }
    const uint16_t status = load ? load(h.layer, fm.e_in, fm.e_out, h.epoch, brx_.data() + sizeof fm, brx_.size() - sizeof fm) : 1;
    ++st.fills;
    if (batch_.done(status == 0)) epoch_acked.store(h.epoch, std::memory_order_release);   // (the whole batch loaded)
    EpHdr ok;
    ok.type = FillOk;
    ok.layer = h.layer;
    ok.epoch = h.epoch;
    ok.aux = (uint32_t) fm.e_in;
    return bulk.send(ok, &status, sizeof status, err) ? 1 : -1;
}

}  // namespace strata::net::ep
