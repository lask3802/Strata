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
// DS4_RX_TIMEOUT_MS (20) after its send: those picks are computed on the lane (the late reply is skipped when it
// comes); a lost link, or 8 late replies in a row: the lane computes everything from then on.  Never a hang.
//
// The wire: a 32-byte header (magic, type, seq, a, b, c, payload bytes), then the payload; TCP_NODELAY, rows bf16.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
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
};

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

uint16_t f2bf(float f);   // round to nearest even (NaN kept a NaN)
float bf2f(uint16_t h);
double now_ms();          // steady clock
void relax();             // a spin-wait's pause

/// One TCP connection, either end.  Sends block (a full socket buffer waits for room); receives poll the socket
/// (non-blocking) until a whole message is in, spinning, until a deadline.
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
    uint64_t bytes_out = 0, bytes_in = 0;

private:
    bool send_raw_(const void* p, size_t n, std::string& err);
    std::intptr_t fd_ = -1;
    std::string peer_;
    std::vector<uint8_t> rb_;   // bytes read, not yet a whole message
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
    std::vector<uint16_t> xb_;
};

/// The expert server (--expert-server-connect): dials `host_port` (again each second until a main process takes it),
/// serves it until it goes away, frees the blobs, dials again (once: return after one main process).  GPU `dev`, VRAM
/// free less reserve_mib for blobs.  Returns nonzero on a CUDA error.
int serve(const std::string& host_port, const std::string& token, int64_t reserve_mib, int dev, bool once);

}  // namespace strata::ds4::rx
