// src/dsv41/ds4_rx.hpp - remote VRAM experts (a proof of concept): another PC's GPU holds some of the routed experts
// the main process's CPU lane would compute, and computes them for it while the lane computes the rest.
//
//   expert server:  strata-dsv41 --expert-server-connect HOST:PORT [--expert-server-reserve-mib N] [--gpus I]
//   main process:   strata-dsv41 --pack ... --remote-experts [BIND:]PORT ...
//
// The server needs no pack and dials out to the main process (no inbound port on the server's PC).  The main process
// listens at its load, after its own placement, for at most DS4_RX_WAIT_S seconds (120): no server, it runs as it
// would without.  Both sides send STRATA_STAGE_TOKEN in their hello and refuse a different one.  The server says how
// much VRAM it has (free less the reserve); the main process sends the geometry, then the blobs of the hottest experts
// by its profile that are neither in its own VRAM nor on a CPU-only layer, as many as fit, once.  In decode, per layer
// request of the CPU lane (serve_lane): the picks the server holds go out as soon as the routing is known (T rows of
// x in bf16, up to 8 picks and weights each), the lane computes the others, then waits for the reply (T rows of
// sum_i w_i expert_i(x) in bf16) and adds it to the same part before the answer's seq.  A reply later than
// DS4_RX_TIMEOUT_MS (2) after the lane's own picks are done: those picks are computed on the lane (the late reply is
// skipped when it comes); 8 late replies of the last 64: the lane computes everything for the next DS4_RX_BACKOFF (256)
// requests, then the link is tried again with a clean count; a lost link, or no reply at all from one such idle spell
// to the next: the lane computes everything from then on.
// Never a hang.
// Keep-awake: each end's receive spins while bytes came in the last kSpinMs; a window's start sends a Wake when the
// link has been quiet for kWakeQuietMs (the server's spin restarts before the window's first request), and serve_lane
// does not nap while the link is up and its last request is under kSpinMs old.  Both ends' link threads run on the
// link core (DS4_EP_CPU, 2), the main's CPU lane workers off it and its SMT sibling.  DS4_EP_WAKE=0: no Wake and the
// lane's 2 ms naps (A/B).
//
// The wire: a 32-byte header (magic, type, seq, a, b, c, payload bytes), then the payload; TCP_NODELAY, rows bf16.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds4::rx {

constexpr uint32_t kMagic = 0x58524453u;   // "SDRX"
constexpr uint32_t kProto = 1;
constexpr int kMaxRows = 16;               // a request's rows at most (kWin)
constexpr int kMaxPicks = 8;               // picks a row at most (kPicks)

enum Msg : uint32_t {
    Hello = 1,      // payload HelloMsg (each side sends one; the server first)
    Geo = 2,        // main -> server: payload GeoMsg, then n_layers x GeoLayer
    Blob = 3,       // main -> server: a = layer, b = expert; payload the blob (GeoLayer's layout)
    BlobsEnd = 4,   // main -> server: a = the blobs sent
    BlobsOk = 5,    // server -> main: payload uint8 ok[blobs] (1: in its VRAM)
    Req = 6,        // main -> server: seq, a = layer, b = rows; payload rows x (ReqRow, then bf16 x[n_embd])
    Resp = 7,       // server -> main: seq, a = rows, c = the server's time (us); payload rows x bf16 y[n_embd]
    Error = 9,      // either way: payload the message
    Ping = 10,      // main -> server (tests: the link alone): seq; the server answers Pong with the same payload at once
    Pong = 11,      // (a = the Wakes the server took, b = its waits in poll (the low 31 bits), c = the one CPU its
                    // serve loop may run on, -1: not pinned)
    Wake = 12,      // main -> server: no payload, no reply; a window starts (its bytes restart the server's spin)
};

constexpr double kSpinMs = 200.0;      // a receive spins while bytes came this recently, then waits in poll
constexpr double kWakeQuietMs = 100.0; // a Wake goes out only when the main sent nothing for this long (half the spin)

/// The link core (DS4_EP_CPU, 2 by default; -1 or "off": none): the logical CPU the thread that drives the link runs
/// on, on either PC - the main's service thread, the server's serve loop.  -1 also where this PC has no such CPU
int link_cpu();
/// `cores` less the link CPU's core (`link`: its SMT siblings, os::cpu_siblings), in their order: the CPU lane's
/// workers stay off the core the link thread spins on
std::vector<int> off_link(const std::vector<int>& cores, const std::vector<int>& link);
/// serve_lane, with no request in: may it nap (20 us sleeps) after idle_ms without one - after 2 ms, but while the
/// link is up not before kSpinMs (the server spins as long: a window's first request is not held up by the lane)
inline bool lane_naps(double idle_ms, bool link_up) { return idle_ms > (link_up ? kSpinMs : 2.0); }

#pragma pack(push, 1)
struct Hdr {
    uint32_t magic = kMagic, type = 0, seq = 0;
    int32_t a = 0, b = 0, c = 0;
    uint64_t bytes = 0;
};
struct HelloMsg {
    uint32_t proto = kProto;
    char token[64] = {};
    uint64_t room = 0;    // (the server) bytes of VRAM it can give the blobs
    char name[52] = {};   // the GPU's / the host's name (informational)
};
struct GeoMsg {
    int32_t n_embd = 0, n_ff = 0, n_layers = 0;
    float limit = 0.0f;   // the swiglu's clamp (0: none)
};
struct GeoLayer {         // a layer's experts: the roles' ggml types (0: the layer is not served), each role's offset in
    int32_t K[3] = {0, 0, 0};   // a blob (gate | up | down, 256-aligned) and the blob's bytes
    uint64_t off[3] = {0, 0, 0};
    uint64_t blob = 0;
};
struct ReqRow {           // a request's row: the window's token t, n picks (expert ids, route weights)
    uint8_t t = 0, n = 0;
    uint16_t pad = 0;
    int16_t e[kMaxPicks] = {};
    float w[kMaxPicks] = {};
};
#pragma pack(pop)
static_assert(sizeof(Hdr) == 32, "the header's size");
static_assert(sizeof(HelloMsg) == 128, "the hello's size");

/// The main's send gate and a reply's deadline, from EWMAs (0: none yet) of the round trip at a request's rows and of
/// the CPU lane's time a pick at the window's T.  A window's remote picks go out only when the round trip is expected
/// to end no later than the lane computing them with its own (none learned yet: out; every 32nd window kept in a row
/// at a T: out anyway, a probe that keeps the estimate current).  The deadline: timeout_ms after the lane's own part,
/// but not before 1.5 x the round trip expected at those rows.
struct Gate {
    double rtt[kMaxRows + 1] = {}, pick[kMaxRows + 1] = {};
    int run[kMaxRows + 1] = {}, seen[kMaxRows + 1] = {}, over[kMaxRows + 1] = {};
    double run_lo[kMaxRows + 1] = {}, run_hi[kMaxRows + 1] = {};   // (the clamped run's first and last send, ms)
    static constexpr double kSlowerMs = 50.0;
    static void ew(double& a, double v) { a = a > 0.0 ? 0.9 * a + 0.1 * v : v; }
    // a reply's round trip: from the kernel's arrival time where there is one (Linux; a late reply's too, when it
    // comes), else a reply the lane waited for (a late one: the time waited, a lower bound).  Without it, one already
    // there when the lane finished its own part says only that the round trip was at most that long - the time is the
    // lane's, not the link's -: it may lower the estimate, never raise it.  After 4 samples a sample counts as 3 x the
    // estimate at most (one spike moves it by 20%; the late count is the reply's own time), but not after 4 in a row
    // were and their sends span kSlowerMs or more: then the link did get slower, and the sample counts as it is (with
    // sends sparser than that, 8 of 64 late may idle the link first; the run spans it after).  A stall bunches its
    // replies within a few ms: those stay clamped.  sent < 0 (no send time): the count alone
    void rtt_seen(int rows, double ms, bool waited = true, double sent = -1.0) {
        if (!waited && rtt[rows] > 0.0 && ms >= rtt[rows]) return;
        if (seen[rows] >= 4 && ms > 3.0 * rtt[rows]) {
            if (over[rows] == 0) {
                run_lo[rows] = run_hi[rows] = sent;
            } else if (sent >= 0.0) {
                run_lo[rows] = run_lo[rows] < 0.0 || sent < run_lo[rows] ? sent : run_lo[rows];
                run_hi[rows] = sent > run_hi[rows] ? sent : run_hi[rows];
            }
            const bool spans = sent < 0.0 || run_hi[rows] - run_lo[rows] >= kSlowerMs;
            if (over[rows] < 4 || !spans) {
                ms = 3.0 * rtt[rows];
                ++over[rows];
            } else {
                over[rows] = 0;
            }
        } else {
            over[rows] = 0;
        }
        ew(rtt[rows], ms);
        ++seen[rows];
    }
    void pick_seen(int T, double ms_a_pick) { ew(pick[T], ms_a_pick); }
    bool send(int T, int rows, size_t n_remote, size_t n_local) {
        const double rt = rtt[rows], c = pick[T];
        if (rt <= 0.0 || c <= 0.0 || rt <= c * (double) (n_remote + n_local) || ++run[T] >= 32) {
            run[T] = 0;
            return true;
        }
        return false;
    }
    double deadline(double sent, double lane_done, double timeout_ms, int rows) const {
        const double a = lane_done + timeout_ms, b = sent + 1.5 * rtt[rows];
        return a > b ? a : b;
    }
};

/// The main's late requests (seq mod 64: a later one 64 on takes the slot): a reply that comes after all gives the gate the round trip it had
struct LateBook {
    struct E {
        uint32_t seq = 0;
        double sent = -1;
        int rows = 0, layer = -1;
    };
    E e[64];
    void add(uint32_t seq, double sent, int rows, int layer = -1) { e[seq % 64] = {seq, sent, rows, layer}; }
    /// seq's send time, rows and layer (false: not a late request here, or taken already)
    bool take(uint32_t seq, double& sent, int& rows, int* layer = nullptr) {
        E& x = e[seq % 64];
        if (x.sent < 0 || x.seq != seq) return false;
        sent = x.sent;
        rows = x.rows;
        if (layer != nullptr) *layer = x.layer;
        x.sent = -1;
        return true;
    }
};

/// The late replies of the main's last 64 requests: 8 of them idle the link for the next `hold` requests (the lane
/// computes their picks), then it is tried again with a clean count.  Given up for good only when no reply at all
/// (late or not) came from one idle spell to the next: a server that stopped answering on a link still open
struct Backoff {
    enum { On = 0, Idle = 1, Silent = 2 };
    uint64_t bits = 0;
    int hold = 256, left = 0;
    bool spell = false, heard = false;   // (an idle spell before, a reply read since it)
    /// once a request that has remote picks: true = keep them here (the link idle)
    bool idle() {
        if (left <= 0) return false;
        --left;
        return true;
    }
    void reply() { heard = true; }   // (a reply read: on time, or a late one that came after all)
    /// a sent request's reply on time or late: Idle = this late one began an idle spell, Silent = and none came since
    /// the last one (give the link up)
    int seen(bool late) {
        bits = bits << 1 | (late ? 1u : 0u);
        int n = 0;
        for (uint64_t b = bits; b != 0; b &= b - 1) ++n;
        if (n < 8) return On;
        const bool silent = spell && !heard;
        bits = 0;
        left = hold;
        spell = true;
        heard = false;
        return silent ? Silent : Idle;
    }
};

uint16_t f2bf(float f);   // round to nearest even (NaN kept a NaN)
float bf2f(uint16_t h);
double now_ms();          // steady clock
void relax();             // a spin-wait's pause

/// One TCP connection, either end.  Sends block (a full socket buffer waits for room); receives poll the socket
/// (non-blocking) until a whole message is in, spinning while bytes came in the last 200 ms (a link that has gone quiet
/// waits in poll, so an idle server does not burn a core), until a deadline.
class Conn {
public:
    Conn() = default;
    ~Conn();
    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;
    /// HOST:PORT; false (err) when it does not connect within timeout_s
    bool dial(const std::string& host_port, int timeout_s, std::string& err);
    /// [BIND:]PORT: listen, take the first connection within wait_s, stop listening
    bool listen_accept(const std::string& at, int wait_s, std::string& err);
    void close();
    bool ok() const { return fd_ >= 0; }
    bool send(uint32_t type, uint32_t seq, int32_t a, int32_t b, int32_t c, const void* p, size_t n, std::string& err,
              const void* p2 = nullptr, size_t n2 = 0);
    /// 1: a whole message in h / payload; 0: none by deadline (steady ms; < 0: no deadline); -1: the link failed (err)
    int recv(Hdr& h, std::vector<uint8_t>& payload, double deadline, std::string& err);
    std::string peer() const { return peer_; }
    /// the last message recv gave: when the kernel took in the read that completed it (steady ms; SO_TIMESTAMPNS,
    /// Linux), -1 where the kernel says nothing (Windows)
    double arrival_ms() const { return arrival_; }
    /// when a message last went out (steady ms; 0: none yet)
    double last_out_ms() const { return last_out_; }
    uint64_t bytes_out = 0, bytes_in = 0;
    uint64_t polls = 0;       // receives' waits in poll (quiet past kSpinMs)
    uint64_t quickacks = 0;   // reads followed by TCP_QUICKACK (Linux: the peer's next send is not held for our ACK)

private:
    bool send_raw_(const void* p, size_t n, std::string& err);
    std::intptr_t fd_ = -1;
    std::string peer_;
    std::vector<uint8_t> rb_;   // bytes read, not yet a whole message
    double last_in_ = 0;        // when bytes last came in (steady ms)
    double last_out_ = 0;       // when a message last went out
    double in_at_ = -1, arrival_ = -1;   // the kernel's time of the last read; of the last message's
};

/// The main process's end: the server's hello, the geometry and blobs, then requests and their replies.
class Main {
public:
    /// listen at `at` ([BIND:]PORT) for wait_s seconds; check the server's hello, send ours
    bool accept(const std::string& at, const std::string& token, int wait_s, std::string& err);
    uint64_t room() const { return peer_hello_.room; }
    std::string name() const { return std::string(peer_hello_.name, strnlen_(peer_hello_.name, sizeof peer_hello_.name)); }
    std::string peer() const { return c_.peer(); }
    bool send_geo(const GeoMsg& g, const std::vector<GeoLayer>& layers, std::string& err);
    bool send_blob(int layer, int e, const void* blob, size_t bytes, std::string& err);
    /// the end of the blobs: the server's ok[] (one a blob sent, in order)
    bool end_blobs(int n, std::vector<uint8_t>& ok, double timeout_ms, std::string& err);
    /// rows[j] and its x (n_embd floats, sent as bf16)
    bool send_req(uint32_t seq, int layer, int nrows, const ReqRow* rows, const float* const* x, std::string& err);
    /// request seq's reply: 1 and its rows (nrows x n_embd bf16, valid until the next call) in *y, the server's time
    /// in srv_us; 0: not in by the deadline (steady ms; it is skipped when it comes); -1: the link failed (err)
    int wait_resp(uint32_t seq, int nrows, double deadline, const uint16_t** y, int& srv_us, std::string& err);
    /// the last reply's arrival in the kernel (Conn::arrival_ms: steady ms, -1 unknown)
    double arrival_ms() const { return c_.arrival_ms(); }
    /// the replies wait_resp skipped since the last call (late ones, their picks computed here): seq and the kernel's
    /// arrival time (steady ms, -1 unknown), for the round trip they had
    std::vector<std::pair<uint32_t, double>> take_late() {
        std::vector<std::pair<uint32_t, double>> v;
        v.swap(late_);
        return v;
    }
    /// tests: a Ping of `bytes` payload and its Pong (1; 0 none by the deadline; -1 the link failed); the Pong's a, b, c
    /// in pong[0..2] (the server's Wakes taken, waits in poll, pinned CPU) when given
    int ping(uint32_t seq, size_t bytes, double deadline, std::string& err, int32_t* pong = nullptr);
    /// a window starts: a Wake when nothing went out for kWakeQuietMs (the server's spin may have run out); true if sent
    bool wake(std::string& err);
    uint64_t wakes = 0;   // Wakes sent
    void close() { c_.close(); }
    bool ok() const { return c_.ok(); }
    uint64_t bytes_out() const { return c_.bytes_out; }
    uint64_t bytes_in() const { return c_.bytes_in; }

private:
    static size_t strnlen_(const char* s, size_t n) {
        size_t i = 0;
        while (i < n && s[i] != '\0') ++i;
        return i;
    }
    Conn c_;
    HelloMsg peer_hello_;
    int n_embd_ = 0;
    std::vector<uint8_t> tx_, rx_;
    std::vector<std::pair<uint32_t, double>> late_;   // (take_late)
    std::vector<uint16_t> xb_;
};

/// The expert server (--expert-server-connect): dials `host_port` (again each second until a main process takes it),
/// serves it until it goes away, frees the blobs, dials again (once: return after one main process).  GPU `dev`, VRAM
/// free less reserve_mib for blobs.  Returns nonzero on a CUDA error.
int serve(const std::string& host_port, const std::string& token, int64_t reserve_mib, int dev, bool once);
/// tests (--expert-server-cpu): the same server with the CPU lane's arithmetic (ds4_iq_cpu.cpp, the i-quant types) in
/// place of a GPU, the blobs in RAM (room: DS4_RX_CPU_ROOM_MIB, 2048)
int serve_cpu(const std::string& host_port, const std::string& token, bool once);

/// Either server's parts.  serve_loop: dial `host_port` (again each second until a main process takes it), run
/// `session` on the connection (0: the main went away; < 0: an error, returned as 1), again unless `once`; the calling
/// thread on link_cpu() meanwhile (its mask restored at the return).  serve_next: the next request (1, in h / p),
/// answering Pings and taking Wakes on the way (the counts in *wakes); -1: the link went (err)
int serve_loop(const std::string& host_port, bool once, const char* what, int (*session)(Conn&, void*), void* arg);
int serve_next(Conn& c, Hdr& h, std::vector<uint8_t>& p, int64_t& pings, int64_t& wakes, std::string& err);

}  // namespace strata::ds4::rx
