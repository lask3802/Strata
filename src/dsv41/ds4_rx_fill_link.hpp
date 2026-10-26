// src/dsv41/ds4_rx_fill_link.hpp - the remote experts' fills on the wire (EP U5b, DS4_RX_FILL=1; ds4_rx_fill.hpp is
// the pure part, which fills to make).  A second, paced socket beside the rx link (net::ep::Conn: a fill is MBs, so it never
// sits in front of a layer's request): the main listens on it (the rx port + 1, DS4_RX_FILL_PORT), says so on the rx
// link (BulkHello: the port, a session), and the expert server dials it and says hello (the session and the token).
// Then the main's fill thread sends Fills - an expert's bytes into the slot of one the server holds - paced, and reads
// their FillOks; the decode thread only queues fills and takes the FillOks at a token boundary.
//
// A Fill: EpHdr type Fill, layer, T = k, seq, aux = in, epoch = out; the payload the expert's blob (k 0), or with H'
// (DS4_EP_K = k) only what the server's parts do not hold: gate and up rows [0, k x 256), then each down row's first k
// blocks (FillGeo::gather) - the server puts the parts' [k, nb) beside them.  FillOk: seq, layer, aux = in, epoch =
// out, payload uint16 status (0: loaded; else refused, `in` stays the lane's).
#pragma once

#include "ds4_rx.hpp"
#include "../net/ep_link.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds4::rx {

/// A layer's blob as a fill sees it: the three matrices at off[] (gate and up n_ff rows of row[0] / row[1] bytes, down
/// n_embd rows of row[2]: nb blocks of 256 inputs each)
struct FillGeo {
    uint64_t off[3] = {}, blob = 0;
    uint64_t row[3] = {};
    int rows[3] = {};
    int nb = 1;
    FillGeo() = default;
    FillGeo(const GeoLayer& G, int n_embd, int n_ff, const uint64_t row_bytes[3]);
    bool split(int k) const { return k > 0 && k < nb; }
    /// what a fill of k ships (split(k): the main's [0, k) of each matrix; else the blob)
    uint64_t ship_bytes(int k) const;
    /// the payload from the three matrices (each contiguous, as the pack has them)
    void gather(const uint8_t* const src[3], int k, uint8_t* dst) const;
    /// (split(k)) where the shipped [0, k) of matrix r begins in the payload, its bytes a row and its rows
    uint64_t ship_off(int r, int k) const;
    /// (split(k)) the payload's down part alone: each down row's first k blocks, ship_bytes(k) - ship_off(2, k) bytes.
    /// The gate and up parts before it are the matrices' first ship_off(1, k) and ship_off(2, k) - ship_off(1, k)
    /// bytes as the pack has them (DS4_RX_FILL_ZC: sent from there, not gathered)
    void gather_down(const uint8_t* src2, int k, uint8_t* dst) const;
};

/// The main's end.  open() and queue() / take() on the decode thread; the fill thread owns the socket after open().
class FillLink {
public:
    FillLink() = default;
    ~FillLink();
    FillLink(const FillLink&) = delete;
    FillLink& operator=(const FillLink&) = delete;
    /// listen on rx_at's address at port (0: rx_at's + 1), BulkHello on m, the server's dial and hello within wait_ms,
    /// then the fill thread on `cpu` (-1: not pinned), its sends paced at pace_bps.  false (err): no fills (the server
    /// takes none - an older one, the CPU server - or the hello failed)
    bool open(Main& m, const std::string& rx_at, int port, const std::string& token, double pace_bps, int cpu,
              int wait_ms, std::string& err);
    struct Job {
        uint32_t seq = 0;
        int layer = -1, in = -1, out = -1, k = 0;
        const uint8_t* src[3] = {};   // in's matrices (valid until its FillOk, or the link's end)
        std::intptr_t fd[3] = {-1, -1, -1};   // (zc 2) the pack files src[] map, and where in them src[] begins
        uint64_t foff[3] = {};
        FillGeo geo;
    };
    void queue(const Job& j);
    struct Done {
        uint32_t seq = 0;
        int layer = -1, in = -1, out = -1;
        uint16_t status = 0;
    };
    /// the FillOks in since the last call
    std::vector<Done> take();
    bool up() const { return up_.load(std::memory_order_acquire); }
    std::string why();   // (after up() went false)
    void close();        // the fill thread ended, the socket closed (queued fills dropped: their slots stay out)
    int port() const { return port_; }

    std::atomic<uint64_t> sent{0}, bytes{0}, oks{0}, fails{0};
    std::atomic<uint64_t> send_us{0};   // the fill thread's time in send (the pace's)
    /// (DS4_RX_FILL_ZC, set before open) how a split fill's gate and up go out: 0 gathered into a buffer with the downs
    /// (U5b's); 1 sent straight from the mapping (no gather: the socket's copy alone); 2 through sendfile from the pack's
    /// file (no copy through this process; refused: 1's way from then on, zc_note says why).  The downs are gathered
    /// in every mode (n_embd rows of k blocks each, strided)
    int zc = 0;
    std::atomic<uint64_t> direct{0}, filed{0};   // gate / up bytes sent from the mapping, through sendfile
    std::string zc_note();                       // (zc 2) why sendfile went off, "" while it is on
    /// (DS4_RX_FILL_GATE=1, set before open) the CPU lane's batches under way (the lane's count, its owner's: > 0 while a
    /// call's gate / up and down batches run): the fill thread copies and sends only while it is 0 - each piece of
    /// kGatePiece bytes (the gather's copies too) waits for it (an atomic wait, woken by the lane's last batch's end;
    /// cut off kGateMaxMs after it began, then the piece goes anyway).  Off (null): 1 MB pieces, the gather in one go
    std::atomic<int>* gate = nullptr;
    static constexpr uint64_t kGatePiece = 64u << 10;
    static constexpr double kGateMaxMs = 2.0;
    std::atomic<uint64_t> gate_waits{0}, gate_us{0}, gate_over{0};   // pieces that found the lane busy, their wait, cut off

private:
    void loop_();
    net::ep::Conn c_;
    net::ep::Listener ls_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;      // (under mu_)
    std::vector<Done> done_;    // (under mu_)
    std::string why_;           // (under mu_)
    std::string zc_note_;       // (under mu_)
    bool file_ok_ = true;       // (the fill thread's: sendfile not refused yet)
    std::atomic<bool> stop_{false}, up_{false};
    int cpu_ = -1, port_ = 0;
    double pace_ = 0.0, due_ = 0.0;   // (the fill thread's: bytes a second; when the next piece may go, steady ms)
};

/// (DS4_RX_FILL_GATE) the lane's side of FillLink::gate: a call's batches begin / end; the last to end wakes the fill
/// thread (here, not inline in a .cu: the atomic's wait and notify are C++20, the CUDA sources C++17)
void lane_busy_in(std::atomic<int>& busy);
void lane_busy_out(std::atomic<int>& busy);

/// The server's end: dial the main's fills port (its host is the rx link's peer), the hello.  false (err)
bool fill_dial(net::ep::Conn& c, const std::string& rx_peer, int port, uint64_t session, const std::string& token,
               std::string& err);

/// the fill threads' CPU: DS4_RX_FILL_CPU, else the link core's SMT sibling (link_cpu(); -1: not pinned)
int fill_cpu();

}  // namespace strata::ds4::rx
