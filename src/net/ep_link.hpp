// src/net/ep_link.hpp - the expert-parallel link (dsv41 EP, results/dsv41-perf-20261025/ep-design.md section 3): the
// main PC and the worker PC exchange each layer's routed-expert work on a TCP connection of their own, beside (never
// with) the remote stage's stage_link.  One frame a layer each way, so the link is built for latency, not throughput:
//
//   - a frame is a 32-byte header and its payload in one buffer, sent with one send() (stage_link sends the header and
//     each payload part on their own);
//   - the receiving thread reads the socket itself, non-blocking, spinning (no reader thread, no condition variable,
//     no wake-up on the critical path) while frames came within its spin budget (50 ms), then waiting in poll() - so
//     an idle link does not hold a core, and a decode keeps both PCs' cores awake (section 3.3);
//   - the main sends at most two requests ahead (n+1 deferral), the worker serves them in order, and a reply that
//     misses its deadline is computed on the main instead (the late reply is dropped when it comes; three in a row:
//     the link is down) - the main never waits on the worker past the deadline and never aborts a request for it;
//   - a Wake (a Ping that wants no Pong) gets the worker spinning again before a window's first request;
//   - placement fills (Fill / FillOk, epochs: section 3.7) go on a second, paced socket, so a 9 MB expert never sits
//     in front of a layer's request.
//
// Both sides say hello first (EpHello, a fixed 256 bytes so an older peer's is read whole and refused for its
// protocol): the same pack (pack_hash), the same geometry, the shared secret (STRATA_STAGE_TOKEN, never printed), and
// the worker's slice table - the main checks that the worker stores every block [k, n_ff / block) it will be asked
// for.  The worker dials, the main listens (one listening main can take more than one helper).
//
// Not wired into the engine here (U4).  Tests and the token-shaped probe: tests/dsv41/ep_link_test.cpp.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace strata::net::ep {

constexpr uint32_t kMagic = 0x4b4c5045u;   // "EPLK" as the bytes go on the wire
constexpr uint32_t kProto = 1;
constexpr int kMaxT = 16;                  // a frame's rows at most (the engine's kWin)
constexpr int kMaxLayers = 64;
constexpr int kOutstanding = 2;            // requests the main may have out at once
constexpr double kTimeoutMs = 50.0;        // a reply's deadline by default (DS4_EP_TIMEOUT_MS; a normal one: 0.4-0.7)
constexpr double kSpinMs = 50.0;           // a receiver's spin budget after the last frame, either way
constexpr int kDownAfter = 3;              // timeouts in a row: the link is down
constexpr uint16_t kDefaultPort = 7870;    // (not the stage link's 7840 / 7841, nor the agent's 7850)

enum Type : uint16_t {
    Hello = 1,     // worker -> main: EpHello, then (request socket) n_layers x EpSlice
    HelloOk = 2,   // main -> worker: EpHello, then n_layers x EpSlice (the main's types); worker -> main: empty (it agrees)
    Req = 3,       // main -> worker: layer, T, seq, epoch, aux flags; T x k_picks x EpPick, then T x n_embd bf16 x
    Resp = 4,      // worker -> main: seq echoed, aux = the worker's lane us; uint16 rowmask, uint16 status, popcount x n_embd bf16
    Fill = 5,      // main -> worker (bulk socket): layer, epoch, aux = expert in; EpFill, then the blob (0 bytes: from the pool)
    FillOk = 6,    // worker -> main (bulk socket): layer, epoch, aux = expert in; uint16 status (0: loaded)
    Ping = 7,      // either way; aux kPingWake: a Wake, no Pong
    Pong = 8,      // the Ping's seq and payload, aux = the ns it was held
    Error = 9,     // the reason as text; the sender closes
    Bye = 10,      // a clean close
};
constexpr uint32_t kPingWake = 1u;      // Ping aux: wake up and spin; no Pong wanted
constexpr uint32_t kReqDeferred = 1u;   // Req aux bits (section 3.2)
constexpr uint32_t kReqTrace = 2u;

enum Lane : uint8_t { LaneSkip = 0, LaneSlice = 1, LaneWhole = 2 };   // a pick's owner, tagged by the main

#pragma pack(push, 1)
struct EpHdr {
    uint32_t magic = kMagic;
    uint16_t type = 0;
    uint8_t layer = 0;
    uint8_t T = 0;        // rows, 1..kMaxT (Req / Resp)
    uint32_t seq = 0;     // Req: the mailbox seq; Resp / Pong echo it
    uint32_t epoch = 0;   // the worker GPU's placement the picks were tagged under (Req, Fill, FillOk)
    uint32_t bytes = 0;   // payload after the header
    uint32_t aux = 0;     // Req: flags; Resp: the worker's lane us; Fill / FillOk: the expert loaded; Ping: kPingWake; Pong: ns held
    uint64_t t_ns = 0;    // the sender's steady clock (timing only)
};
struct EpPick {
    int16_t e = -1;
    uint8_t lane = LaneSkip;
    uint8_t pad = 0;
    float w = 0.0f;
};
/// a layer of the slice table: the blocks [lo, hi) of n_ff the worker stores (lo == hi: none), and each matrix's
/// (gate, up, down) ggml type and the bytes of one whole row in the source pack
struct EpSlice {
    uint8_t lo = 0, hi = 0;
    uint8_t type[3] = {0, 0, 0};
    uint8_t pad[3] = {0, 0, 0};
    uint32_t row_bytes[3] = {0, 0, 0};
};
struct EpHello {
    uint32_t magic = kMagic, proto = kProto;
    uint32_t role = 0;      // 0 the main, 1 a worker
    uint32_t channel = 0;   // 0 the request socket, 1 the bulk socket
    uint64_t pack_hash = 0;    // the source pack's fingerprint (ds4_stage.hpp's pack_hash); a worker's: its parts' source
    uint64_t parts_hash = 0;   // the worker's ep-parts.txt (informational)
    uint64_t session = 0;      // the main's HelloOk: names the connection; a bulk socket's hello carries it back
    int32_t n_embd = 0, n_layers = 0, n_experts = 0, k_picks = 0, n_ff = 0, block = 0, max_t = 0, rows_bf16 = 0;
    int32_t gpu_slots = 0;     // a worker's GPU: whole-expert slots a layer (sizes the main's remote tier)
    int32_t pad0 = 0;
    uint64_t slot_bytes = 0;   // bytes a slot
    char build[40] = {};       // the engine's build id (informational)
    char token[64] = {};       // the shared secret: each side refuses another one
    uint8_t reserved[64] = {};
};
struct EpFill {
    int32_t e_in = -1, e_out = -1;   // the expert loaded, the one whose slot it takes (-1: a free slot)
};
#pragma pack(pop)
static_assert(sizeof(EpHdr) == 32, "the header's size");
static_assert(sizeof(EpPick) == 8, "a pick's size");
static_assert(sizeof(EpSlice) == 20, "a slice row's size");
// one size across protocols: an older peer's hello is read whole and refused for its protocol, not as malformed
static_assert(sizeof(EpHello) == 256, "the hello's size");

uint16_t f2bf(float f);   // round to nearest even, as the GPU kernels round (a NaN stays a NaN)
float bf2f(uint16_t b);
double now_ms();          // steady clock
uint64_t now_ns();
void relax();             // a spin-wait's pause

/// The hello checks, as pure functions (the tests call them; the handshakes below use them).  "" when the peer is
/// acceptable, else why not (never with the token in it).
/// main_check: the worker's hello and slice table against the main's own; k[l]: the blocks [0, k[l]) of layer l the
/// main computes itself (k[l] == n_ff / block: none asked of the worker).
std::string main_check(const EpHello& mine, const std::vector<EpSlice>& my_table, const std::vector<int>& k,
                       const EpHello& worker, const std::vector<EpSlice>& worker_table);
/// worker_check: the main's hello against the worker's own
std::string worker_check(const EpHello& mine, const EpHello& main);
/// the blocks [k, n_ff / block) of every layer within what the table stores (k may change at run time: check again)
std::string slices_cover(const std::vector<EpSlice>& worker_table, const std::vector<int>& k, int blocks);

/// One TCP connection of the link.
class Conn {
public:
    Conn() = default;
    ~Conn();
    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;
    /// HOST:PORT; false (err) when it does not connect within timeout_ms
    bool dial(const std::string& host_port, int timeout_ms, std::string& err);
    void adopt(std::intptr_t fd, const std::string& peer);   // (Listener::accept's)
    void close();
    bool ok() const { return fd_ >= 0; }
    std::string peer() const { return peer_; }
    /// one frame, header and payload(s) in one buffer and one send() (more only while the socket's buffer is full);
    /// h.bytes and h.t_ns are set here.  false: the link failed (err; closed)
    bool send(EpHdr h, const void* p, size_t n, std::string& err, const void* p2 = nullptr, size_t n2 = 0);
    /// 1: a whole frame in h and payload; 0: none by the deadline (steady ms; < 0: none); -1: the link failed (err;
    /// closed): closed by the peer (mid-frame or not), a bad magic, a frame over max_frame
    int recv(EpHdr& h, std::vector<uint8_t>& payload, double deadline, std::string& err);
    /// SO_BUSY_POLL us on this socket (Linux; false and why elsewhere or when refused)
    bool busy_poll(int us, std::string& err);
    size_t buffered() const { return rb_.size() - rd_; }   // bytes read, not yet a whole frame
    /// bytes as they are, no header (tests: a peer that breaks the framing)
    bool send_bytes(const void* p, size_t len, std::string& err) { return send_raw_((const uint8_t*) p, len, err); }
    double last_frame_ms() const { return last_frame_; }

    double spin_ms = kSpinMs;      // spin this long after the last frame either way, then wait in poll (0: always poll)
    size_t max_frame = 1u << 22;   // payload bytes a frame may carry (the bulk socket: more)
    double pace_bps = 0.0;         // send at most this many bytes a second, 64 KB at a time (0: no pacing; the bulk socket)
    struct Counters {
        uint64_t bytes_in = 0, bytes_out = 0, frames_in = 0, frames_out = 0;
        uint64_t polls = 0;        // waits in poll() (past the spin budget, or spin_ms 0)
        uint64_t spins = 0;        // spin passes (a pause each)
    } stat;

private:
    bool send_raw_(const uint8_t* p, size_t len, std::string& err);
    std::intptr_t fd_ = -1;
    std::string peer_;
    std::vector<uint8_t> rb_, tx_;
    size_t rd_ = 0;            // rb_'s consumed bytes
    double last_frame_ = 0.0;  // steady ms of the last whole frame, in or out
};

/// A listening socket (the main's): connections are taken one at a time.
class Listener {
public:
    ~Listener();
    /// [BIND:]PORT (PORT 0: any free one, port() says which)
    bool open(const std::string& at, std::string& err);
    int port() const { return port_; }
    /// the next connection within wait_ms into c; false (err) when none came
    bool accept(Conn& c, int wait_ms, std::string& err);
    void close();

private:
    std::intptr_t fd_ = -1;
    int port_ = 0;
};

/// The handshakes on a connected socket.  The worker speaks first; the main answers HelloOk with its own hello (a
/// fresh session id in it) or an Error with the reason; the worker checks that and confirms with an empty HelloOk.
/// A bulk socket: the worker dials again and sends its hello with channel 1 and the session; the main answers HelloOk.
struct Peer {
    EpHello hello;
    std::vector<EpSlice> table;
};
bool hello_worker(Conn& c, const EpHello& mine, const std::vector<EpSlice>& table, Peer& main, std::string& err,
                  double timeout_ms = 10000.0);
bool hello_main(Conn& c, EpHello& mine, const std::vector<EpSlice>& table, const std::vector<int>& k, Peer& worker,
                std::string& err, double timeout_ms = 10000.0);
bool bulk_worker(Conn& c, const EpHello& mine, uint64_t session, std::string& err, double timeout_ms = 10000.0);
bool bulk_main(Conn& c, const EpHello& mine, std::string& err, double timeout_ms = 10000.0);

/// A reply as the main reads it: rows valid until the next call on the link.
struct Reply {
    uint32_t seq = 0;
    uint16_t rowmask = 0, status = 0;
    uint32_t lane_us = 0;
    const uint16_t* rows = nullptr;   // popcount(rowmask) x n_embd bf16, the rowmask's rows in order
};

/// The main's end: requests out and their replies, Wakes, the probe's Pings, fills on the bulk socket.
class MainLink {
public:
    Conn req, bulk;
    /// after hello_main on req: the geometry the frames are sized by, a clean state
    void start(const EpHello& mine);
    bool up() const { return up_; }
    /// seq's request (T rows of k_picks picks, then T x n_embd bf16 x): 1 sent; 0 not sent - compute it here (two out
    /// already, the link down, an epoch not acked, a seq not after the last one sent: err says which); -1 the send
    /// failed (the link is down now)
    int send_req(uint32_t seq, int layer, int T, uint32_t epoch, uint32_t flags, const EpPick* picks, const uint16_t* x,
                 std::string& err);
    /// seq's reply, the oldest request still awaited (they are awaited in the order sent): 1 in r; 0 not by the
    /// deadline - compute it here (its reply is dropped when it comes; kDownAfter in a row: the link is down); -1 the
    /// link is down (err): closed, an Error from the worker, a reply out of order or of the wrong size
    int wait_resp(uint32_t seq, double deadline, Reply& r, std::string& err);
    bool wake(std::string& err);
    /// the probe: a Ping of `bytes` and its Pong (1; 0 none by the deadline; -1 the link failed).  No request out.
    int ping(uint32_t seq, size_t bytes, double deadline, uint32_t& held_ns, std::string& err,
             std::vector<uint8_t>* payload = nullptr, uint32_t aux = 0);
    void bye();
    int outstanding() const { return (int) out_.size(); }

    /// the bulk socket: a fill of epoch `epoch` (the next one, or the one being filled), blob n bytes (0: the worker
    /// loads it from its pool)
    bool send_fill(int layer, int e_in, int e_out, uint32_t epoch, const void* blob, size_t n, std::string& err);
    /// the next FillOk: 1 (layer, expert, epoch, status), 0 none by the deadline, -1 the bulk link failed
    int poll_fill_ok(double deadline, int& layer, int& e_in, uint32_t& epoch, uint16_t& status, std::string& err);
    /// the newest epoch every fill of which the worker loaded: the epoch requests may be tagged with
    uint32_t epoch_ready() const { return ready_; }

    struct Stats {
        uint64_t sent = 0, replies = 0, timeouts = 0, late_dropped = 0, not_sent = 0, downs = 0, fills = 0, fill_fails = 0;
    } st;

private:
    struct Out {
        uint32_t seq = 0;
        int T = 0;
        bool dropped = false;
    };
    void down_(const std::string& why, std::string& err);
    int take_(double deadline, Out& o, Reply& r, std::string& err);   // the next reply, the oldest request's (popped)
    int drain_(std::string& err);                                    // late replies already in, without waiting
    std::deque<Out> out_;
    bool up_ = false, any_sent_ = false, fill_bad_ = false;
    uint32_t last_seq_ = 0, ready_ = 0, filling_ = 0;
    int fills_out_ = 0, in_a_row_ = 0, n_embd_ = 0, k_ = 0, n_layers_ = 0;
    std::vector<uint8_t> tx_, rx_, brx_;
};

/// The worker's end: requests in order, the replies, Pings / Wakes, fills on the bulk socket (from its own thread).
class WorkerLink {
public:
    Conn req, bulk;
    void start(const EpHello& mine);
    struct Frame {
        EpHdr h;
        int T = 0;
        const EpPick* picks = nullptr;   // T x k_picks
        const uint16_t* x = nullptr;     // T x n_embd bf16
    };
    /// the next request: 1 in f, valid until send_resp, which must come before the next call; 0 none by the deadline;
    /// -1 the link failed or the main broke a rule (err; an Error frame sent where the link still takes one): a seq not
    /// after the last one, an epoch not acked, a wrong size, a layer or T out of range.  Pings are answered here (a Pong
    /// of the same payload, after on_ping), a Wake calls on_wake; neither returns.
    int next(Frame& f, double deadline, std::string& err);
    bool send_resp(uint32_t seq, uint16_t rowmask, uint16_t status, const uint16_t* rows, uint32_t lane_us,
                   std::string& err);
    std::function<void()> on_wake;
    std::function<void(const EpHdr&, std::vector<uint8_t>&)> on_ping;   // may rewrite the Pong's payload
    /// the bulk socket's loop body: a Fill to `load` (status 0: loaded), its FillOk.  1 one handled, 0 none by the
    /// deadline, -1 failed (err): an epoch neither the acked one nor the next
    using Loader = std::function<uint16_t(int layer, int e_in, int e_out, uint32_t epoch, const uint8_t* blob, size_t n)>;
    int serve_fill(double deadline, const Loader& load, std::string& err);
    std::atomic<uint32_t> epoch_acked{0};   // the newest epoch with a FillOk sent
    uint32_t epoch_used = 0;                 // the newest epoch a request was tagged with

    struct Stats {
        uint64_t reqs = 0, pings = 0, wakes = 0, fills = 0;
    } st;

private:
    int refuse_(const std::string& why, std::string& err);
    bool pending_ = false, any_ = false;
    uint32_t last_seq_ = 0;
    int n_embd_ = 0, k_ = 0, n_layers_ = 0;
    std::vector<uint8_t> rx_, tx_, brx_;
};

}  // namespace strata::net::ep
