// src/net/stage_link.cpp - see include/strata/net/stage_link.hpp.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace strata::net {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

#if !defined(_WIN32)

// Seconds a read or write may block before the link counts as dead (STRATA_REMOTE_TIMEOUT_S; a prompt chunk on a
// slow worker can take several seconds, a first window after a long prompt a few more).
int io_timeout_s() {
    static const int s = [] {
        const char* v = std::getenv("STRATA_REMOTE_TIMEOUT_S");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n > 0 ? n : 300;
    }();
    return s;
}

void tune_socket(int fd) {
    int one = 1;
    (void) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    (void) setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    int buf = 8 << 20;
    (void) setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    (void) setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
    timeval tv{};
    tv.tv_sec = io_timeout_s();
    (void) setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    (void) setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
}

bool send_all(int fd, const void* p, size_t n, std::string& err) {
    const char* c = (const char*) p;
    while (n > 0) {
        const ssize_t w = ::send(fd, c, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            err = std::string("remote stage: send failed: ") + std::strerror(errno);
            return false;
        }
        c += w;
        n -= (size_t) w;
    }
    return true;
}

bool recv_all(int fd, void* p, size_t n, std::string& err) {
    char* c = (char*) p;
    while (n > 0) {
        const ssize_t r = ::recv(fd, c, n, 0);
        if (r == 0) { err = "remote stage: the peer closed the connection"; return false; }
        if (r < 0) {
            if (errno == EINTR) continue;
            err = std::string("remote stage: receive failed: ") +
                  (errno == EAGAIN || errno == EWOULDBLOCK ? std::string("timed out") : std::string(std::strerror(errno)));
            return false;
        }
        c += r;
        n -= (size_t) r;
    }
    return true;
}

// header + up to two payload parts, as one logical message
bool send_msg(int fd, const StageHeader& h, const void* p1, size_t n1, const void* p2, size_t n2, std::string& err) {
    return send_all(fd, &h, sizeof h, err) && (n1 == 0 || send_all(fd, p1, n1, err)) &&
           (n2 == 0 || send_all(fd, p2, n2, err));
}

bool recv_header(int fd, StageHeader& h, std::string& err) {
    if (!recv_all(fd, &h, sizeof h, err)) return false;
    if (h.magic != kStageMagic) { err = "remote stage: bad message magic (not a Strata stage peer?)"; return false; }
    return true;
}

bool drain(int fd, uint64_t n, std::string& err) {
    char tmp[65536];
    while (n > 0) {
        const size_t k = (size_t) std::min<uint64_t>(n, sizeof tmp);
        if (!recv_all(fd, tmp, k, err)) return false;
        n -= k;
    }
    return true;
}

bool hello_matches(const StageHello& a, const StageHello& b, std::string& err) {
    char buf[512];
    if (a.protocol != b.protocol || a.n_embd != b.n_embd || a.hc != b.hc || a.n_layers != b.n_layers ||
        a.n_expert != b.n_expert || a.layer_begin != b.layer_begin || a.handoff_floats != b.handoff_floats ||
        a.kv_type != b.kv_type) {
        std::snprintf(buf, sizeof buf,
                      "remote stage: the two sides differ (protocol %u/%u, n_embd %d/%d, hc %d/%d, layers %d/%d, "
                      "experts %d/%d, first remote layer %d/%d, hand-off floats %lld/%lld, K/V type %d/%d)",
                      a.protocol, b.protocol, a.n_embd, b.n_embd, a.hc, b.hc, a.n_layers, b.n_layers, a.n_expert,
                      b.n_expert, a.layer_begin, b.layer_begin, (long long) a.handoff_floats,
                      (long long) b.handoff_floats, a.kv_type, b.kv_type);
        err = buf;
        return false;
    }
    return true;
}

#endif

}  // namespace

// ------------------------------------------------------------------------------------------------ StageClient

StageClient::~StageClient() { close(); }

void StageClient::close() {
#if !defined(_WIN32)
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
}

bool StageClient::connect(const std::string& host_port, const StageHello& mine, StageHello& peer, std::string& err) {
#if defined(_WIN32)
    (void) host_port; (void) mine; (void) peer;
    err = "remote stage: not supported on Windows";
    return false;
#else
    const size_t colon = host_port.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= host_port.size()) {
        err = "remote stage: expected HOST:PORT, got \"" + host_port + "\"";
        return false;
    }
    const std::string host = host_port.substr(0, colon), port = host_port.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res); rc != 0) {
        err = "remote stage: cannot resolve " + host_port + ": " + gai_strerror(rc);
        return false;
    }
    int fd = -1;
    for (addrinfo* ai = res; ai != nullptr && fd < 0; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) != 0) {
            ::close(fd);
            fd = -1;
        }
    }
    freeaddrinfo(res);
    if (fd < 0) { err = "remote stage: cannot connect to " + host_port + ": " + std::strerror(errno); return false; }
    tune_socket(fd);
    fd_ = fd;
    peer_ = host_port;
    StageHeader h;
    h.type = (uint32_t) StageMsg::Hello;
    h.bytes = sizeof mine;
    StageHeader r;
    if (!send_msg(fd_, h, &mine, sizeof mine, nullptr, 0, err) || !recv_header(fd_, r, err)) { close(); return false; }
    if (r.type == (uint32_t) StageMsg::Error) {
        std::string m((size_t) r.bytes, '\0');
        std::string e2;
        (void) recv_all(fd_, m.data(), m.size(), e2);
        err = "remote stage: the worker refused: " + m;
        close();
        return false;
    }
    if (r.type != (uint32_t) StageMsg::HelloOk || r.bytes != sizeof peer || !recv_all(fd_, &peer, sizeof peer, err)) {
        if (err.empty()) err = "remote stage: unexpected reply to hello";
        close();
        return false;
    }
    if (!hello_matches(mine, peer, err)) { close(); return false; }
    return true;
#endif
}

bool StageClient::roundtrip_(const StageHeader& h, const void* p1, size_t n1, const void* p2, size_t n2, StageMsg want,
                             void* reply, size_t reply_bytes, std::string& err) {
#if defined(_WIN32)
    (void) h; (void) p1; (void) n1; (void) p2; (void) n2; (void) want; (void) reply; (void) reply_bytes;
    err = "remote stage: not supported on Windows";
    return false;
#else
    if (fd_ < 0) { err = "remote stage: not connected"; return false; }
    StageHeader r;
    const auto t0 = Clock::now();
    if (!send_msg(fd_, h, p1, n1, p2, n2, err)) { close(); return false; }
    const auto t1 = Clock::now();
    if (!recv_header(fd_, r, err)) { close(); return false; }
    const auto t2 = Clock::now();
    last_send_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_wait_ms_ = std::chrono::duration<double, std::milli>(t2 - t1).count();
    stats_.ms_send += last_send_ms_;
    stats_.ms_wait += last_wait_ms_;
    stats_.bytes_out += sizeof h + n1 + n2;
    if (r.type == (uint32_t) StageMsg::Error) {
        std::string m((size_t) r.bytes, '\0');
        std::string e2;
        (void) recv_all(fd_, m.data(), m.size(), e2);
        err = "remote stage (worker): " + m;
        return false;   // the connection stays: the worker reported and waits for the next message
    }
    if (r.type != (uint32_t) want || r.bytes != reply_bytes) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "remote stage: unexpected reply (type %u, %llu bytes; wanted type %u, %zu bytes)",
                      r.type, (unsigned long long) r.bytes, (unsigned) want, reply_bytes);
        err = buf;
        close();
        return false;
    }
    if (reply_bytes > 0 && !recv_all(fd_, reply, reply_bytes, err)) { close(); return false; }
    last_recv_ms_ = ms_since(t2);
    stats_.ms_recv += last_recv_ms_;
    stats_.bytes_in += sizeof r + reply_bytes;
    return true;
#endif
}

// STRATA_REMOTE_TIMING=1: one line per prompt chunk and every 64th window, with the round trip's three parts
static bool remote_timing() {
    static const bool on = [] { const char* v = std::getenv("STRATA_REMOTE_TIMING"); return v && v[0] == '1'; }();
    return on;
}

bool StageClient::run(int T, const int32_t* tokens, int64_t pos0, const float* rows_in, size_t in_floats,
                      float* rows_out, size_t out_floats, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto t0 = Clock::now();
    StageHeader h;
    h.type = (uint32_t) StageMsg::Run;
    h.a = T;
    h.b = pos0;
    h.bytes = (uint64_t) T * sizeof(int32_t) + in_floats * sizeof(float);
    const bool ok = roundtrip_(h, tokens, (size_t) T * sizeof(int32_t), rows_in, in_floats * sizeof(float),
                               StageMsg::RunOk, rows_out, out_floats * sizeof(float), err);
    ++stats_.runs;
    const double ms = ms_since(t0);
    stats_.ms_run += ms;
    if (remote_timing() && (stats_.runs & 63) == 0)
        std::fprintf(stderr, "strata remote: window T=%d at %lld: %.2f ms (send %.2f, worker %.2f, receive %.2f); "
                             "%llu windows, mean %.2f ms (send %.2f, worker %.2f, receive %.2f)\n",
                     T, (long long) pos0, ms, last_send_ms_, last_wait_ms_, last_recv_ms_,
                     (unsigned long long) stats_.runs, stats_.ms_run / (double) stats_.runs,
                     stats_.ms_send / (double) (stats_.runs + stats_.prefills),
                     stats_.ms_wait / (double) (stats_.runs + stats_.prefills),
                     stats_.ms_recv / (double) (stats_.runs + stats_.prefills));
    return ok;
}

bool StageClient::commit(int n_keep, std::string& err) {
#if defined(_WIN32)
    (void) n_keep;
    err = "remote stage: not supported on Windows";
    return false;
#else
    std::lock_guard<std::mutex> lk(mu_);
    if (fd_ < 0) { err = "remote stage: not connected"; return false; }
    StageHeader h;
    h.type = (uint32_t) StageMsg::Commit;
    h.a = n_keep;
    if (!send_msg(fd_, h, nullptr, 0, nullptr, 0, err)) { close(); return false; }
    stats_.bytes_out += sizeof h;
    ++stats_.commits;
    return true;
#endif
}

bool StageClient::prefill(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, const float* rows_in,
                          size_t row_floats, float* rows_out, int64_t skip, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto t0 = Clock::now();
    skip = std::max<int64_t>(0, std::min<int64_t>(skip, T));
    StageHeader h;
    h.type = (uint32_t) StageMsg::Prefill;
    h.a = T;
    h.b = pos0;
    h.c = (flags & 0xff) | (skip << 8);
    const size_t in_bytes = (size_t) T * row_floats * sizeof(float);
    const size_t out_bytes = (size_t) (T - skip) * row_floats * sizeof(float);
    h.bytes = (uint64_t) T * sizeof(int64_t) + in_bytes;
    const bool ok = roundtrip_(h, tokens, (size_t) T * sizeof(int64_t), rows_in, in_bytes, StageMsg::PrefillOk,
                               rows_out + (size_t) skip * row_floats, out_bytes, err);
    ++stats_.prefills;
    const double ms = ms_since(t0);
    stats_.ms_prefill += ms;
    if (remote_timing())
        std::fprintf(stderr, "strata remote: chunk T=%lld at %lld: %.0f ms (send %.0f, worker %.0f, receive %.0f; %lld rows back)\n",
                     (long long) T, (long long) pos0, ms, last_send_ms_, last_wait_ms_, last_recv_ms_,
                     (long long) (T - skip));
    return ok;
}

bool StageClient::reset(std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    StageHeader h;
    h.type = (uint32_t) StageMsg::Reset;
    const bool ok = roundtrip_(h, nullptr, 0, nullptr, 0, StageMsg::ResetOk, nullptr, 0, err);
    ++stats_.resets;
    return ok;
}

// ------------------------------------------------------------------------------------------------ serve_stage

int serve_stage(int port, const StageHandlers& h, const StageBuffers& buf, StageStats& stats, const bool* stop) {
#if defined(_WIN32)
    (void) port; (void) h; (void) buf; (void) stats; (void) stop;
    std::fprintf(stderr, "strata stage worker: not supported on Windows\n");
    return 1;
#else
    const int lfd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (lfd < 0) { std::fprintf(stderr, "strata stage worker: socket: %s\n", std::strerror(errno)); return 1; }
    int one = 1, zero = 0;
    (void) setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    (void) setsockopt(lfd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);
    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_any;
    addr.sin6_port = htons((uint16_t) port);
    if (::bind(lfd, (sockaddr*) &addr, sizeof addr) != 0 || ::listen(lfd, 1) != 0) {
        std::fprintf(stderr, "strata stage worker: cannot listen on port %d: %s\n", port, std::strerror(errno));
        ::close(lfd);
        return 1;
    }
    std::fprintf(stderr, "strata stage worker: listening on port %d\n", port);
    std::fflush(stderr);
    std::vector<int32_t> tok32;
    std::vector<int64_t> tok64;
    while (stop == nullptr || !*stop) {
        pollfd pfd{lfd, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 1000);
        if (pr <= 0) continue;
        sockaddr_storage peer{};
        socklen_t plen = sizeof peer;
        const int fd = ::accept(lfd, (sockaddr*) &peer, &plen);
        if (fd < 0) continue;
        char host[128] = "?";
        (void) getnameinfo((sockaddr*) &peer, plen, host, sizeof host, nullptr, 0, NI_NUMERICHOST);
        tune_socket(fd);
        // no read timeout while idle between requests: the main process may wait for its next prompt for hours
        timeval tv0{};
        (void) setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv0, sizeof tv0);
        std::fprintf(stderr, "strata stage worker: connection from %s\n", host);
        std::fflush(stderr);
        std::string err, pending_err;   // pending_err: a failed one-way message, reported on the next reply
        bool greeted = false;
        auto reply_error = [&](const std::string& m) -> bool {
            StageHeader e;
            e.type = (uint32_t) StageMsg::Error;
            e.bytes = m.size();
            std::string se;
            return send_msg(fd, e, m.data(), m.size(), nullptr, 0, se);
        };
        for (;;) {
            StageHeader m;
            if (!recv_header(fd, m, err)) break;
            const auto t0 = Clock::now();
            stats.bytes_in += sizeof m + m.bytes;
            const StageMsg type = (StageMsg) m.type;
            if (!greeted && type != StageMsg::Hello) {
                (void) reply_error("hello first");
                break;
            }
            bool ok = true, keep = true;
            switch (type) {
            case StageMsg::Hello: {
                StageHello theirs, mine;
                if (m.bytes != sizeof theirs || !recv_all(fd, &theirs, sizeof theirs, err)) { keep = false; break; }
                std::string he;
                if (!h.hello(theirs, mine, he) || !hello_matches(theirs, mine, he)) {
                    (void) reply_error(he);
                    keep = false;
                    break;
                }
                StageHeader r;
                r.type = (uint32_t) StageMsg::HelloOk;
                r.bytes = sizeof mine;
                if (!send_msg(fd, r, &mine, sizeof mine, nullptr, 0, err)) keep = false;
                greeted = true;
                std::fprintf(stderr, "strata stage worker: main process %s: layers %d.. here, context %lld, chunk %lld\n",
                             host, theirs.layer_begin, (long long) theirs.max_context, (long long) theirs.chunk);
                std::fflush(stderr);
                break;
            }
            case StageMsg::Run: {
                const int T = (int) m.a;
                const size_t rows = (size_t) T * (size_t) buf.handoff_floats;
                if (T < 1 || rows > buf.run_floats || m.bytes != (uint64_t) T * 4 + rows * 4) {
                    (void) drain(fd, m.bytes, err);
                    ok = reply_error("bad window size");
                    break;
                }
                tok32.resize((size_t) T);
                if (!recv_all(fd, tok32.data(), (size_t) T * 4, err) || !recv_all(fd, buf.run_in, rows * 4, err)) {
                    keep = false;
                    break;
                }
                std::string e;
                if (!pending_err.empty()) { e = pending_err; pending_err.clear(); ok = false; }
                else ok = h.run(T, tok32.data(), m.b, e);
                if (!ok) { ok = reply_error(e); break; }
                StageHeader r;
                r.type = (uint32_t) StageMsg::RunOk;
                r.bytes = rows * 4;
                if (!send_msg(fd, r, buf.run_out, rows * 4, nullptr, 0, err)) keep = false;
                stats.bytes_out += sizeof r + rows * 4;
                ++stats.runs;
                stats.ms_run += ms_since(t0);
                break;
            }
            case StageMsg::Commit: {
                std::string e;
                if (pending_err.empty() && !h.commit((int) m.a, e)) pending_err = e;
                ++stats.commits;
                break;
            }
            case StageMsg::Prefill: {
                const int64_t T = m.a;
                const size_t rows = (size_t) T * (size_t) buf.pf_row_floats;
                if (T < 1 || rows > buf.pf_floats || m.bytes != (uint64_t) T * 8 + rows * 4) {
                    (void) drain(fd, m.bytes, err);
                    ok = reply_error("bad prompt chunk size");
                    break;
                }
                tok64.resize((size_t) T);
                if (!recv_all(fd, tok64.data(), (size_t) T * 8, err) || !recv_all(fd, buf.pf_in, rows * 4, err)) {
                    keep = false;
                    break;
                }
                std::string e;
                const int64_t skip = std::max<int64_t>(0, std::min<int64_t>(m.c >> 8, T));
                if (!pending_err.empty()) { e = pending_err; pending_err.clear(); ok = false; }
                else ok = h.prefill(tok64.data(), T, m.b, m.c & 0xff, e);
                if (!ok) { ok = reply_error(e); break; }
                if (remote_timing())
                    std::fprintf(stderr, "strata stage worker: chunk T=%lld at %lld: received + read in %.0f ms\n",
                                 (long long) T, (long long) m.b, ms_since(t0));
                const size_t back = (size_t) (T - skip) * (size_t) buf.pf_row_floats;
                StageHeader r;
                r.type = (uint32_t) StageMsg::PrefillOk;
                r.bytes = back * 4;
                if (!send_msg(fd, r, buf.pf_out + (size_t) skip * (size_t) buf.pf_row_floats, back * 4, nullptr, 0, err))
                    keep = false;
                stats.bytes_out += sizeof r + back * 4;
                ++stats.prefills;
                stats.ms_prefill += ms_since(t0);
                break;
            }
            case StageMsg::Reset: {
                std::string e;
                pending_err.clear();
                if (!h.reset(e)) { ok = reply_error(e); break; }
                StageHeader r;
                r.type = (uint32_t) StageMsg::ResetOk;
                if (!send_msg(fd, r, nullptr, 0, nullptr, 0, err)) keep = false;
                ++stats.resets;
                break;
            }
            default:
                (void) drain(fd, m.bytes, err);
                ok = reply_error("unknown message type " + std::to_string(m.type));
                break;
            }
            if (!keep || !ok) break;
        }
        ::close(fd);
        std::fprintf(stderr, "strata stage worker: %s disconnected%s%s\n", host, err.empty() ? "" : ": ", err.c_str());
        std::fflush(stderr);
        if (h.disconnected) h.disconnected();
    }
    ::close(lfd);
    return 0;
#endif
}

}  // namespace strata::net
