// src/net/stage_link.cpp - see include/strata/net/stage_link.hpp.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>
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

// STRATA_REMOTE_TIMING=1: one line per prompt chunk and every 64th window, the worker's own time and the link's
bool remote_timing() {
    static const bool on = [] { const char* v = std::getenv("STRATA_REMOTE_TIMING"); return v && v[0] == '1'; }();
    return on;
}

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

// the next reply: `want` with exactly reply_bytes of payload, or the worker's error (the connection then stays)
bool StageClient::read_reply_(StageMsg want, void* reply, size_t reply_bytes, int64_t& worker_us, std::string& err) {
#if defined(_WIN32)
    (void) want; (void) reply; (void) reply_bytes; (void) worker_us;
    err = "remote stage: not supported on Windows";
    return false;
#else
    StageHeader r;
    if (!recv_header(fd_, r, err)) { close(); return false; }
    if (r.type == (uint32_t) StageMsg::Error) {
        std::string m((size_t) r.bytes, '\0');
        std::string e2;
        (void) recv_all(fd_, m.data(), m.size(), e2);
        err = "remote stage (worker): " + m;
        return false;
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
    worker_us = r.a;
    stats_.bytes_in += sizeof r + reply_bytes;
    return true;
#endif
}

bool StageClient::run(int T, const int32_t* tokens, int64_t pos0, const float* rows_in, size_t in_floats,
                      float* rows_out, size_t out_floats, std::string& err) {
#if defined(_WIN32)
    (void) T; (void) tokens; (void) pos0; (void) rows_in; (void) in_floats; (void) rows_out; (void) out_floats;
    err = "remote stage: not supported on Windows";
    return false;
#else
    std::lock_guard<std::mutex> ls(send_mu_);
    std::lock_guard<std::mutex> lr(recv_mu_);
    if (fd_ < 0) { err = "remote stage: not connected"; return false; }
    const auto t0 = Clock::now();
    StageHeader h;
    h.type = (uint32_t) StageMsg::Run;
    h.a = T;
    h.b = pos0;
    h.bytes = (uint64_t) T * sizeof(int32_t) + in_floats * sizeof(float);
    if (!send_msg(fd_, h, tokens, (size_t) T * sizeof(int32_t), rows_in, in_floats * sizeof(float), err)) {
        close();
        return false;
    }
    stats_.bytes_out += sizeof h + h.bytes;
    int64_t wus = 0;
    const bool ok = read_reply_(StageMsg::RunOk, rows_out, out_floats * sizeof(float), wus, err);
    const double ms = ms_since(t0);
    ++stats_.runs;
    stats_.ms_run += ms;
    stats_.ms_run_worker += (double) wus / 1000.0;
    if (remote_timing() && (stats_.runs & 63) == 0)
        std::fprintf(stderr, "strata remote: %llu windows: mean %.2f ms a round trip = worker %.2f + link %.2f "
                             "(last: T=%d at %lld, %.2f = %.2f + %.2f)\n",
                     (unsigned long long) stats_.runs, stats_.ms_run / (double) stats_.runs,
                     stats_.ms_run_worker / (double) stats_.runs,
                     (stats_.ms_run - stats_.ms_run_worker) / (double) stats_.runs, T, (long long) pos0, ms,
                     (double) wus / 1000.0, ms - (double) wus / 1000.0);
    return ok;
#endif
}

bool StageClient::commit(int n_keep, std::string& err) {
#if defined(_WIN32)
    (void) n_keep;
    err = "remote stage: not supported on Windows";
    return false;
#else
    std::lock_guard<std::mutex> lk(send_mu_);
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

bool StageClient::prefill_send(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, const float* rows_in,
                               size_t row_floats, int64_t skip, std::string& err) {
#if defined(_WIN32)
    (void) tokens; (void) T; (void) pos0; (void) flags; (void) rows_in; (void) row_floats; (void) skip;
    err = "remote stage: not supported on Windows";
    return false;
#else
    std::lock_guard<std::mutex> lk(send_mu_);
    if (fd_ < 0) { err = "remote stage: not connected"; return false; }
    skip = std::max<int64_t>(0, std::min<int64_t>(skip, T));
    StageHeader h;
    h.type = (uint32_t) StageMsg::Prefill;
    h.a = T;
    h.b = pos0;
    h.c = (flags & 0xff) | (skip << 8);
    const size_t in_bytes = (size_t) T * row_floats * sizeof(float);
    h.bytes = (uint64_t) T * sizeof(int64_t) + in_bytes;
    {
        std::lock_guard<std::mutex> lq(q_mu_);
        sent_at_.push_back(Clock::now());
    }
    if (!send_msg(fd_, h, tokens, (size_t) T * sizeof(int64_t), rows_in, in_bytes, err)) { close(); return false; }
    stats_.bytes_out += sizeof h + h.bytes;
    return true;
#endif
}

bool StageClient::prefill_recv(float* rows_out, size_t row_floats, int64_t T, int64_t skip, std::string& err) {
    std::lock_guard<std::mutex> lk(recv_mu_);
    if (fd_ < 0) { err = "remote stage: not connected"; return false; }
    skip = std::max<int64_t>(0, std::min<int64_t>(skip, T));
    int64_t wus = 0;
    const bool ok = read_reply_(StageMsg::PrefillOk, rows_out + (size_t) skip * row_floats,
                                (size_t) (T - skip) * row_floats * sizeof(float), wus, err);
    Clock::time_point t0 = Clock::now();
    {
        std::lock_guard<std::mutex> lq(q_mu_);
        if (!sent_at_.empty()) {
            t0 = sent_at_.front();
            sent_at_.pop_front();
        }
    }
    const double ms = ms_since(t0);
    ++stats_.prefills;
    stats_.ms_prefill += ms;
    if (remote_timing())
        std::fprintf(stderr, "strata remote: chunk T=%lld: %.0f ms from its send to its rows = worker %.0f + link and "
                             "queue %.0f (%lld rows back)\n",
                     (long long) T, ms, (double) wus / 1000.0, ms - (double) wus / 1000.0, (long long) (T - skip));
    return ok;
}

bool StageClient::reset(std::string& err) {
#if defined(_WIN32)
    err = "remote stage: not supported on Windows";
    return false;
#else
    std::lock_guard<std::mutex> ls(send_mu_);
    std::lock_guard<std::mutex> lr(recv_mu_);
    if (fd_ < 0) { err = "remote stage: not connected"; return false; }
    StageHeader h;
    h.type = (uint32_t) StageMsg::Reset;
    if (!send_msg(fd_, h, nullptr, 0, nullptr, 0, err)) { close(); return false; }
    int64_t wus = 0;
    ++stats_.resets;
    return read_reply_(StageMsg::ResetOk, nullptr, 0, wus, err);
#endif
}

// ------------------------------------------------------------------------------------------------ serve_stage

#if !defined(_WIN32)
namespace {

// One message as the receiving thread read it.  A prompt chunk's rows are already in buf.pf_in[slot].
struct Inbox {
    StageHeader h;
    std::vector<int32_t> tok32;
    std::vector<int64_t> tok64;
    std::vector<float> rows;    ///< a window's rows (copied to run_in by the serving thread)
    StageHello hello;
    int slot = -1;
    std::string bad;            ///< the message was malformed: reply this error
    bool closed = false;        ///< the connection ended (err says why)
    std::string err;
};

}  // namespace
#endif

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

        // ---- the receiving thread: whole messages into a queue, a prompt chunk's rows straight into a free pf_in
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Inbox> q;
        bool slot_busy[2] = {false, false};
        std::atomic<bool> quit{false};
        std::thread reader([&] {
            for (;;) {
                Inbox in;
                if (!recv_header(fd, in.h, in.err)) {
                    in.closed = true;
                } else {
                    const StageMsg type = (StageMsg) in.h.type;
                    if (type == StageMsg::Hello) {
                        if (in.h.bytes != sizeof in.hello) {
                            in.bad = "bad hello";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else if (!recv_all(fd, &in.hello, sizeof in.hello, in.err)) {
                            in.closed = true;
                        }
                    } else if (type == StageMsg::Run) {
                        const int64_t T = in.h.a;
                        const size_t rows = (size_t) std::max<int64_t>(T, 0) * (size_t) buf.handoff_floats;
                        if (T < 1 || rows > buf.run_floats || in.h.bytes != (uint64_t) T * 4 + rows * 4) {
                            in.bad = "bad window size";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else {
                            in.tok32.resize((size_t) T);
                            in.rows.resize(rows);
                            if (!recv_all(fd, in.tok32.data(), (size_t) T * 4, in.err) ||
                                !recv_all(fd, in.rows.data(), rows * 4, in.err))
                                in.closed = true;
                        }
                    } else if (type == StageMsg::Prefill) {
                        const int64_t T = in.h.a;
                        const size_t rows = (size_t) std::max<int64_t>(T, 0) * (size_t) buf.pf_row_floats;
                        if (T < 1 || rows > buf.pf_floats || in.h.bytes != (uint64_t) T * 8 + rows * 4) {
                            in.bad = "bad prompt chunk size";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else {
                            {   // a free buffer: the serving thread releases one when it has read that chunk
                                std::unique_lock<std::mutex> lk(mu);
                                cv.wait(lk, [&] { return !slot_busy[0] || !slot_busy[1] || quit.load(); });
                                if (quit.load()) break;
                                in.slot = !slot_busy[0] ? 0 : 1;
                                slot_busy[in.slot] = true;
                            }
                            in.tok64.resize((size_t) T);
                            if (!recv_all(fd, in.tok64.data(), (size_t) T * 8, in.err) ||
                                !recv_all(fd, buf.pf_in[in.slot], rows * 4, in.err))
                                in.closed = true;
                        }
                    } else if (type != StageMsg::Commit && type != StageMsg::Reset) {
                        in.bad = "unknown message type " + std::to_string(in.h.type);
                        if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                    }
                }
                const bool closed = in.closed;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(in));
                }
                cv.notify_all();
                if (closed || quit.load()) break;
            }
        });

        // ---- the serving thread (this one): the messages in order
        std::string err, pending_err;   // pending_err: a failed one-way message, reported on the next reply
        bool greeted = false;
        // a prompt chunk's reply goes out on its own thread while the next chunk is read; every later send waits
        // for it (the replies keep their order), and so does the reuse of its output buffer
        std::future<bool> reply_out;
        std::string reply_err;
        int out_buf = 0;
        auto replies_out = [&]() -> bool {
            if (!reply_out.valid()) return true;
            if (reply_out.get()) return true;
            err = reply_err;
            return false;
        };
        auto reply_error = [&](const std::string& m) -> bool {
            if (!replies_out()) return false;
            StageHeader e;
            e.type = (uint32_t) StageMsg::Error;
            e.bytes = m.size();
            std::string se;
            return send_msg(fd, e, m.data(), m.size(), nullptr, 0, se);
        };
        for (;;) {
            Inbox in;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return !q.empty(); });
                in = std::move(q.front());
                q.pop_front();
            }
            if (in.closed) { err = in.err; break; }
            auto release = [&] {
                if (in.slot < 0) return;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    slot_busy[in.slot] = false;
                }
                in.slot = -1;
                cv.notify_all();
            };
            const StageMsg type = (StageMsg) in.h.type;
            stats.bytes_in += sizeof in.h + in.h.bytes;
            if (!greeted && type != StageMsg::Hello) { (void) reply_error("hello first"); release(); break; }
            if (!in.bad.empty()) {
                release();
                if (!reply_error(in.bad)) break;
                continue;
            }
            bool keep = true;
            const auto t0 = Clock::now();
            switch (type) {
            case StageMsg::Hello: {
                StageHello mine;
                std::string he;
                if (!h.hello(in.hello, mine, he) || !hello_matches(in.hello, mine, he)) {
                    (void) reply_error(he);
                    keep = false;
                    break;
                }
                StageHeader r;
                r.type = (uint32_t) StageMsg::HelloOk;
                r.bytes = sizeof mine;
                if (!replies_out() || !send_msg(fd, r, &mine, sizeof mine, nullptr, 0, err)) keep = false;
                greeted = true;
                std::fprintf(stderr, "strata stage worker: main process %s: layers %d.. here, context %lld, chunk %lld\n",
                             host, in.hello.layer_begin, (long long) in.hello.max_context, (long long) in.hello.chunk);
                std::fflush(stderr);
                break;
            }
            case StageMsg::Run: {
                const int T = (int) in.h.a;
                std::memcpy(buf.run_in, in.rows.data(), in.rows.size() * 4);
                std::string e;
                bool ok;
                if (!pending_err.empty()) { e = pending_err; pending_err.clear(); ok = false; }
                else ok = h.run(T, in.tok32.data(), in.h.b, e);
                if (!ok) { keep = reply_error(e); break; }
                StageHeader r;
                r.type = (uint32_t) StageMsg::RunOk;
                r.a = (int64_t) (ms_since(t0) * 1000.0);   // the worker's own time, microseconds
                r.bytes = in.rows.size() * 4;
                if (!replies_out() || !send_msg(fd, r, buf.run_out, in.rows.size() * 4, nullptr, 0, err)) keep = false;
                stats.bytes_out += sizeof r + r.bytes;
                ++stats.runs;
                stats.ms_run += ms_since(t0);
                break;
            }
            case StageMsg::Commit: {
                std::string e;
                if (pending_err.empty() && !h.commit((int) in.h.a, e)) pending_err = e;
                ++stats.commits;
                break;
            }
            case StageMsg::Prefill: {
                const int64_t T = in.h.a;
                const int64_t skip = std::max<int64_t>(0, std::min<int64_t>(in.h.c >> 8, T));
                std::string e;
                bool ok;
                // this chunk's output buffer: the one whose reply went out two chunks ago - the reply in flight (the
                // last chunk's) uses the other
                float* const out = buf.pf_out[out_buf];
                if (!pending_err.empty()) { e = pending_err; pending_err.clear(); ok = false; }
                else ok = h.prefill(in.tok64.data(), T, in.h.b, in.h.c & 0xff, buf.pf_in[in.slot], out, e);
                release();   // the chunk is read (its rows were uploaded before `prefill` returned)
                if (!ok) { keep = reply_error(e); break; }
                const double work_ms = ms_since(t0);
                if (remote_timing())
                    std::fprintf(stderr, "strata stage worker: chunk T=%lld at %lld: read in %.0f ms\n", (long long) T,
                                 (long long) in.h.b, work_ms);
                const size_t back = (size_t) (T - skip) * (size_t) buf.pf_row_floats;
                if (!replies_out()) { keep = false; break; }   // the previous reply is out: its buffer is free next time
                StageHeader r;
                r.type = (uint32_t) StageMsg::PrefillOk;
                r.a = (int64_t) (work_ms * 1000.0);
                r.bytes = back * 4;
                const float* src = out + (size_t) skip * (size_t) buf.pf_row_floats;
                reply_err.clear();
                reply_out = std::async(std::launch::async, [fd, r, src, back, &reply_err] {
                    return send_msg(fd, r, src, back * 4, nullptr, 0, reply_err);
                });
                out_buf ^= 1;
                stats.bytes_out += sizeof r + back * 4;
                ++stats.prefills;
                stats.ms_prefill += work_ms;
                break;
            }
            case StageMsg::Reset: {
                std::string e;
                pending_err.clear();
                if (!h.reset(e)) { keep = reply_error(e); break; }
                StageHeader r;
                r.type = (uint32_t) StageMsg::ResetOk;
                if (!replies_out() || !send_msg(fd, r, nullptr, 0, nullptr, 0, err)) keep = false;
                ++stats.resets;
                break;
            }
            default:
                keep = reply_error("unknown message type " + std::to_string(in.h.type));
                break;
            }
            release();
            if (!keep) break;
        }
        (void) replies_out();   // a reply still going out ends first (or fails on the closed socket)
        // end the receiving thread: shutting the socket down wakes a blocked recv
        quit.store(true);
        (void) ::shutdown(fd, SHUT_RDWR);
        cv.notify_all();
        reader.join();
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
