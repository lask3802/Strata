// include/strata/net/stage_link.hpp - a layer split's later stage on another machine, over TCP (remote-stage fork).
//
// The main process runs layers [0, K) and the head; a worker process (strata --serve --stage-worker PORT
// --stage-begin K, on another PC) runs layers [K, n_layers) without the head.  They exchange exactly what an
// in-process layer split hands from one GPU to the next:
//   - a verify window: T rows of Verifier::handoff_floats (the residual streams, the pending write and the inject)
//     each way, then the accepted count (one way, no reply);
//   - a prompt chunk: T rows of hc * n_embd floats (the residual streams) each way.
// The worker keeps its own session (its layers' K/V, GDN state), expert cache and CPU expert pool; the main process
// sends a reset at every fresh prompt.  Linux only (POSIX sockets); elsewhere every call fails with a message.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace strata::net {

inline constexpr uint32_t kStageMagic = 0x4d525453u;   // "STRM"
inline constexpr uint32_t kStageProtocol = 1;

enum class StageMsg : uint32_t {
    Hello = 1,
    HelloOk = 2,
    Run = 3,        // a = T, b = pos0; payload: int32 tokens[T], float rows[T * handoff_floats]
    RunOk = 4,      // payload: float rows[T * handoff_floats]
    Commit = 5,     // a = n_keep; no reply (an error comes back on the next reply)
    Prefill = 6,    // a = T, b = pos0, c = flags (bit 0: the prompt is one chunk) | skip << 8; payload: int64 tokens[T],
                    // float rows[T * D]
    PrefillOk = 7,  // payload: float rows[(T - skip) * D], the chunk's rows from row `skip` (the main process needs
                    // no earlier ones: the drafter's window does not reach them)
    Reset = 8,      // a fresh prompt from position 0
    ResetOk = 9,
    Error = 10,     // payload: the message
};

#pragma pack(push, 1)
struct StageHeader {
    uint32_t magic = kStageMagic;
    uint32_t type = 0;
    int64_t a = 0, b = 0, c = 0;
    uint64_t bytes = 0;   // payload bytes after the header
};

/// What both sides must agree on; exchanged once per connection, each side checks the other's.
struct StageHello {
    uint32_t protocol = kStageProtocol;
    int32_t n_embd = 0, hc = 0, n_layers = 0, n_expert = 0;
    int32_t layer_begin = 0;      ///< the worker's first layer (the main process's K)
    int32_t max_t = 0;            ///< the widest verify window
    int32_t kv_type = 0;          ///< the K/V element type id (both sides must store the same)
    int64_t max_context = 0;
    int64_t chunk = 0;            ///< the largest prompt chunk the main process sends
    int64_t handoff_floats = 0;   ///< floats per verify-window row
    char build[64] = {};          ///< engine build id (informational)
};
#pragma pack(pop)

/// Running totals of one side's traffic (STRATA_REMOTE_TIMING=1 prints them).
struct StageStats {
    uint64_t runs = 0, commits = 0, prefills = 0, resets = 0;
    uint64_t bytes_out = 0, bytes_in = 0;
    double ms_run = 0, ms_prefill = 0;   ///< wall time of the round trips (main side) / of the work (worker side)
    /// main side, split of the round trips: sending, waiting for the reply's header (the worker's work), receiving
    double ms_send = 0, ms_wait = 0, ms_recv = 0;
};

/// The main process's end: one connection, used by one thread at a time (a mutex guards it).
class StageClient {
public:
    StageClient() = default;
    ~StageClient();
    StageClient(const StageClient&) = delete;
    StageClient& operator=(const StageClient&) = delete;

    /// "host:port".  Sends `mine`, receives the worker's hello into `peer` and checks it against `mine`.
    bool connect(const std::string& host_port, const StageHello& mine, StageHello& peer, std::string& err);
    bool connected() const { return fd_ >= 0; }
    void close();

    /// One verify window: send T tokens and T rows (in_floats floats), receive T rows (out_floats floats).
    bool run(int T, const int32_t* tokens, int64_t pos0, const float* rows_in, size_t in_floats, float* rows_out,
             size_t out_floats, std::string& err);
    /// The accepted count of the last window (one way).
    bool commit(int n_keep, std::string& err);
    /// One prompt chunk: send T tokens and T rows of `row_floats`, receive rows [skip, T) into rows_out + skip rows.
    bool prefill(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, const float* rows_in, size_t row_floats,
                 float* rows_out, int64_t skip, std::string& err);
    /// A fresh prompt: the worker zeroes its session.
    bool reset(std::string& err);

    const StageStats& stats() const { return stats_; }
    std::string peer_name() const { return peer_; }

private:
    bool roundtrip_(const StageHeader& h, const void* p1, size_t n1, const void* p2, size_t n2, StageMsg want,
                    void* reply, size_t reply_bytes, std::string& err);
    int fd_ = -1;
    std::string peer_;
    std::mutex mu_;
    StageStats stats_;
    double last_send_ms_ = 0, last_wait_ms_ = 0, last_recv_ms_ = 0, last_worker_ms_ = 0, win_worker_ms_ = 0;
};

/// The worker's end.  Every handler runs on the serving thread; `rows_in`/`rows_out` are the buffers the server
/// was given (the caller's pinned / mapped memory) - a handler reads its input there and writes its output there.
struct StageHandlers {
    std::function<bool(const StageHello& main_hello, StageHello& mine, std::string& err)> hello;
    std::function<bool(int T, const int32_t* tokens, int64_t pos0, std::string& err)> run;   // in: run_in, out: run_out
    std::function<bool(int n_keep, std::string& err)> commit;
    std::function<bool(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, std::string& err)> prefill;   // in: pf_in, out: pf_out (all T rows)
    std::function<bool(std::string& err)> reset;
    std::function<void()> disconnected;   ///< the main process went away (the session is stale)
};

struct StageBuffers {
    float* run_in = nullptr;    ///< max_t * handoff_floats
    float* run_out = nullptr;
    size_t run_floats = 0;      ///< capacity of each, floats
    float* pf_in = nullptr;     ///< chunk * D
    float* pf_out = nullptr;
    size_t pf_floats = 0;
    int64_t handoff_floats = 0; ///< floats per verify row
    int64_t pf_row_floats = 0;  ///< floats per prompt row (D)
};

/// Listens on `port` (all interfaces) and serves one main process at a time until `stop` is set or listening fails.
/// Returns 0 when stopped, nonzero on a listen error (with the reason on stderr).
int serve_stage(int port, const StageHandlers& h, const StageBuffers& buf, StageStats& stats, const bool* stop);

}  // namespace strata::net
