// src/dsv41/ds4_stage.hpp - the DeepSeek engine's remote stage (docs/REMOTE_STAGE.md), the parts with no GPU: the
// model pack's fingerprint both sides' hellos carry, what each side refuses in the other's hello, the main process's
// prompt chunks over the link and the worker's buffers for them.  The link itself is Strata's
// (src/net/stage_link.cpp), the engine's side of it ds4_main.cpp / ds4_model.cu.
//
// v1: windows of T <= 16 rows each way, and prompt chunks (Prefill) of up to the hellos' chunk rows - each side's
// chunked prompt path on its own layers, the main process's next chunk through its layers while the worker reads the
// one before - and no checkpoints: a request that does not continue the conversation in place starts it again from
// token 0.  The rows are fp32 - each carries the indexer's int32 picks and candidate blocks beside the streams, which
// the link's own bf16 would not keep - or, with STRATA_REMOTE_ROWS=bf16 on both sides, the engine's compact rows: the
// streams as bf16, pre and the picks exact (rows_pack), for windows only - the link's Prefill would convert a chunk's
// whole rows, so with bf16 rows neither side has chunks (prompts in windows).
//
// A draft's check (DSpark on the main process) is a window like any other: the worker keeps its layers' backups for
// every window of 2 rows or more, and the main process's Commit says how many of its rows stand (the rest undone there
// as one process undoes them); the drafter's taps of the worker's layers come back in the reply (reply_tap_at) - in
// the streams block, so with bf16 rows they come back bf16 as well.  A prompt chunk is never a draft's check: it is a
// Prefill, kept whole, with no Commit and no backups.
//
// A chunk's rows are a window's fp32 hand-off rows (Model::StageGeo, full_floats a row).  To the worker they go block
// by block as a window's do (T x hc x dim floats, then T x 8, T x topk_stride, T x cand_stride); the reply comes back
// row by row - row r at r x full_floats: its streams (hc x dim), then its pre (8), the rest of the row unused - so
// that the reply's skip leaves out whole rows (the main process reads only the rows its head needs).
#pragma once

#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace strata::ds4::stage {

/// row_formats' capability bit both sides of a DeepSeek link set (hello_matches compares neither it nor the other
/// bits 24-30): the worker's reply is reply_tap_at's layout.  An earlier build's worker sent its last layer's streams
/// and pre back, which a main process of this one would read as the normed rows - each side refuses a peer without it.
/// (Bit 30 is the compact rows' capability, kRowsBf16Cap.)
inline constexpr uint32_t kRowsNormed = 1u << 29;

inline uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char* p = (const unsigned char*) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/// The pack's fingerprint: its tensor index and its geometry (tensors.txt, ds4.txt), as the Qwen engine's hello
/// hashes its own index - each file's name, then its bytes.  ds4.txt's model_dir line is left out: it says where this
/// PC keeps the model's files, which two PCs need not share.  0: neither file could be read.
inline uint64_t pack_hash(const std::string& pack) {
    uint64_t h = 0;
    for (const char* f : {"/tensors.txt", "/ds4.txt"}) {
        std::ifstream in(pack + f, std::ios::binary);
        if (!in) continue;
        const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        h = fnv1a(f, std::char_traits<char>::length(f), h == 0 ? 1469598103934665603ull : h);
        const bool geometry = std::string(f) == "/ds4.txt";
        for (size_t a = 0; a < body.size();) {   // (line by line, each with its newline)
            size_t b = body.find('\n', a);
            b = b == std::string::npos ? body.size() : b + 1;
            if (!(geometry && body.compare(a, 10, "model_dir ") == 0)) h = fnv1a(body.data() + a, b - a, h);
            a = b;
        }
    }
    return h;
}

/// The compact rows of STRATA_REMOTE_ROWS=bf16.  The link's own bf16 (StageRows::BF16) converts every float of a
/// message, which would round the int32 picks; this engine's hello lists fp32 for the link and carries this capability
/// bit (row_formats bits 24-30 are free capability bits; the link reads only the format bits and kStageCapKvElem), and
/// the engine packs the rows itself before the link sends them.  Both sides must have it or neither: a hello with it
/// also has a narrower hand-off row (RowGeo::compact), so the link's own check refuses a mismatch as well.
inline constexpr uint32_t kRowsBf16Cap = 1u << 30;

/// A window of T rows crosses block by block (Model::StageGeo): the streams (T x `h` fp32), pre (T x `pre` fp32), the
/// indexer's picks (T x `topk` int32), then the candidate blocks (T x `cand` int32).  Compact: only the streams as bf16
/// (T x h / 2 words); pre, the picks and the candidates as they are.
struct RowGeo {
    int64_t h = 0;                // the streams: hc x dim
    int64_t pre = 8;              // pre, fp32 either way
    int64_t topk = 0, cand = 0;   // int32 per row
    int64_t exact() const { return pre + topk + cand; }   // words of a row that travel as they are
    int64_t full() const { return h + exact(); }          // words of a row, fp32
    int64_t compact() const { return h / 2 + exact(); }   // words of a row, bf16 streams + the rest
    bool compact_ok() const { return h > 0 && h % 2 == 0; }   // (a row's bf16 part is whole words)
};

/// T rows in the fp32 block layout `full` (T x full() words) to the compact layout `wire` (T x compact() words).  Only
/// the streams are converted; pre and the ints are copied as bytes.
inline void rows_pack(const RowGeo& g, int T, const float* full, float* wire) {
    const size_t nh = (size_t) T * (size_t) g.h, ne = (size_t) T * (size_t) g.exact();
    net::stage_rows_encode(net::StageRows::BF16, full, 1, nh, wire);
    std::memcpy(wire + nh / 2, full + nh, ne * sizeof(float));
}

/// The compact layout back to the fp32 block layout: the streams and pre always, the ints when `ints` (a reply's ints
/// are not read)
inline void rows_unpack(const RowGeo& g, int T, const float* wire, float* full, bool ints = true) {
    const size_t nh = (size_t) T * (size_t) g.h, np = (size_t) T * (size_t) g.pre,
                 ne = (size_t) T * (size_t) g.exact();
    net::stage_rows_decode(net::StageRows::BF16, wire, 1, nh, full);
    std::memcpy(full + nh, wire + nh / 2, (ints ? ne : np) * sizeof(float));
}

/// A prompt chunk's flags (StageMsg::Prefill's, bits 0-7; the Qwen engine's bit 0 is not used here): the last chunk of
/// the main process's run - the worker settles its expert cache after it, as one process does after a prompt
inline constexpr int kChunkLast = 2;

/// The refusal when the main process's chunks (`sends` rows at most) do not fit the worker's (`takes`; 0: windows
/// only), empty when they do - said the same on both sides
inline std::string chunk_refusal(int64_t sends, int64_t takes, int max_t) {
    if (sends <= takes) return {};
    if (takes <= 0)
        return "the main process sends prompt chunks of up to " + std::to_string(sends) + " tokens: the worker reads "
               "windows of " + std::to_string(max_t) + " only (its chunked prompt path is off: --prefill-chunk 0, "
               "DS4_WINDOWS, or its buffers did not fit) - --prefill-chunk 0 on the main process, or chunks on the worker";
    return "the main process sends prompt chunks of up to " + std::to_string(sends) + " tokens, the worker takes " +
           std::to_string(takes) + " at most - --prefill-chunk " + std::to_string(takes) +
           " or less on the main process, or more on the worker";
}

/// What this engine's worker cannot serve in a main process's hello (one of the same protocol; hello_matches has the
/// geometry, the layers and the pack): false and why.  `mine`: the worker's own hello (its chunk: the most rows a
/// prompt chunk may have here), `max_context` its context.
inline bool worker_accepts(const net::StageHello& theirs, const net::StageHello& mine, int64_t max_context,
                           std::string& err) {
    if (theirs.chunk < 0) {
        err = "the main process's hello says prompt chunks of " + std::to_string(theirs.chunk) + " tokens";
        return false;
    }
    if (const std::string r = chunk_refusal(theirs.chunk, mine.chunk, mine.max_t); !r.empty()) {
        err = r;
        return false;
    }
    if (theirs.vision != 0) {
        err = "the main process reads images: a DeepSeek remote stage reads text only";
        return false;
    }
    if (theirs.row_formats & ~(uint32_t) (1u << (uint32_t) net::StageRows::F32) & net::kStageRowsAll) {
        err = "the main process asks the link for rows other than fp32: a DeepSeek row carries the indexer's int32 "
              "picks, which the link's own bf16 would round (this engine packs bf16 rows itself: its hello says so)";
        return false;
    }
    const bool want = (theirs.row_formats & kRowsBf16Cap) != 0, have = (mine.row_formats & kRowsBf16Cap) != 0;
    if (want != have) {
        err = want ? "the main process sends bf16 rows (STRATA_REMOTE_ROWS=bf16) and this worker reads fp32: start the "
                     "worker with STRATA_REMOTE_ROWS=bf16 too"
                   : "this worker reads bf16 rows (STRATA_REMOTE_ROWS=bf16) and the main process sends fp32: set "
                     "STRATA_REMOTE_ROWS=bf16 on the main process too";
        return false;
    }
    if (want && theirs.chunk > 0) {   // (the link's Prefill converts whole rows: the compact rows are the windows' only)
        err = "the main process sends bf16 rows (STRATA_REMOTE_ROWS=bf16) and prompt chunks of " +
              std::to_string(theirs.chunk) + " tokens: a chunk's rows cross as fp32 only - the same build on both PCs";
        return false;
    }
    if (theirs.ckpt_max != 0) {
        err = "the main process keeps conversation checkpoints on its worker: this worker keeps none";
        return false;
    }
    if (theirs.max_context > max_context || theirs.max_t > mine.max_t || theirs.max_t < 1) {
        err = "the main process's context (" + std::to_string(theirs.max_context) + ") or window (" +
              std::to_string(theirs.max_t) + ") is larger than this worker's (" + std::to_string(max_context) + ", " +
              std::to_string(mine.max_t) + ")";
        return false;
    }
    if ((theirs.row_formats & kRowsNormed) == 0) {
        err = "the main process is an earlier build's (it reads the last layer's streams back): this worker replies "
              "with the normed rows and the drafter's taps - the same build on both PCs";
        return false;
    }
    return true;
}

/// What the main process cannot use in its worker's hello (its HelloOk, after the worker accepted this side's;
/// StageClient::connect has checked the rest): false and why.  An earlier build's worker, which replies with its last
/// layer's streams; and its chunks larger than the worker's - a worker that accepted them says it can take them, this
/// is the same check at this end, for a worker whose own check is not this one.
inline bool main_accepts(const net::StageHello& worker, const net::StageHello& mine, std::string& err) {
    if ((worker.row_formats & kRowsNormed) == 0) {
        err = "remote stage: the worker is an earlier build's (it replies with its last layer's streams): this main "
              "process reads the normed rows and the drafter's taps - the same build on both PCs";
        return false;
    }
    if (const std::string r = chunk_refusal(mine.chunk, worker.chunk, worker.max_t); !r.empty()) {
        err = "remote stage: refused: " + r;
        return false;
    }
    return true;
}

/// The worker's reply to a window of T rows, inside the streams block of the request's layout (its first T x hc x dim
/// floats, the part every row format carries as floats), block by block: its last layer's rows through the final norm
/// (T x dim: what the head reads), then the drafter's taps - the streams' mean entering each target layer, T rows of
/// nt x dim, target after target in a row (the main process's h_taps layout).  The rest of the reply is not read.  A
/// target the worker does not hold leaves its slot unwritten.
inline size_t reply_tap_at(int T, int dim, int nt, int t, int j) {
    return (size_t) T * (size_t) dim + ((size_t) t * (size_t) nt + (size_t) j) * (size_t) dim;
}
/// whether a reply with nt targets' taps fits a row's streams (hc x dim floats)
inline bool reply_fits(int dim, int nt, int64_t streams) {
    return (int64_t) dim * (1 + (int64_t) nt) <= streams;
}
/// (the main process) the taps of the targets from layer `end` on - the worker's - from a reply of T rows into taps
/// (T rows of targets.size() x dim); the others' slots stay as this process's own layers wrote them
inline void taps_from_reply(const float* reply, int T, int dim, const std::vector<int>& targets, int end, float* taps) {
    const int nt = (int) targets.size();
    const size_t x = (size_t) T * (size_t) dim;   // (the taps after the normed rows: their own layout from there)
    for (int j = 0; j < nt; ++j) {
        if (targets[(size_t) j] < end) continue;
        for (int t = 0; t < T; ++t) {
            const size_t at = reply_tap_at(T, dim, nt, t, j);
            std::copy(reply + at, reply + at + dim, taps + (at - x));
        }
    }
}

/// A Commit of n_keep after the worker's last window of last_t rows (0: none since the last Commit): All keeps it
/// whole, Part keeps its first n_keep rows (the rest undone from the backups a window of 2 or more rows keeps), Bad
/// is an error (err says why).
enum class Keep { All, Part, Bad };
inline Keep commit_kind(int n_keep, int last_t, std::string& err) {
    if (last_t > 0 && n_keep == last_t) return Keep::All;
    if (last_t > 1 && n_keep >= 1 && n_keep < last_t) return Keep::Part;
    err = last_t <= 0 ? "a commit of " + std::to_string(n_keep) + " rows with no window run since the last one"
                      : "a commit of " + std::to_string(n_keep) + " rows of a window of " + std::to_string(last_t);
    return Keep::Bad;
}

/// The worker's Run and Commit (ds4_main.cpp's handlers; M: Model, or a test's stand-in with its stage_run and
/// stage_rollback).  `last_t`: the last window's rows until its Commit (0: none).
template <class M>
bool worker_run(M& m, const int32_t* tokens, int T, int64_t p0, int& last_t, std::string& err) {
    last_t = 0;
    if (!m.stage_run(tokens, T, p0, err)) return false;
    last_t = T;
    return true;
}
/// The worker's Prefill (a prompt chunk, any number of rows up to the hellos' chunk): kept whole - no Commit follows
/// it, and none after it undoes the window before it (last_t 0: such a Commit is Bad, not a rollback)
template <class M>
bool worker_prefill(M& m, const int64_t* tokens, int T, int64_t p0, bool last, int skip, const float* in, float* out,
                    int& last_t, std::string& err) {
    last_t = 0;
    return m.stage_prefill(tokens, T, p0, last, skip, in, out, err);
}
template <class M>
bool worker_commit(M& m, int n_keep, int& last_t, std::string& err) {
    const Keep k = commit_kind(n_keep, last_t, err);
    last_t = 0;
    if (k == Keep::Part) return m.stage_rollback(n_keep, err);
    return k == Keep::All;
}

/// The main process's prompt chunks over its link: each chunk's send on a thread of its own, one in flight at a time
/// in order (net::StageRelay - a chunk's rows stay as they are until the next send() or wait() returns), its reply
/// read by recv() in the same order, on any one thread.
class ChunkLink {
public:
    /// `row_floats`: a hand-off row's fp32 floats (Model::StageGeo::full_floats)
    ChunkLink(net::StageClient& link, int64_t row_floats) : link_(link), relay_(link), row_(row_floats) {}
    /// chunk rows `rows` (T, block by block) of tokens at p0 to the worker; `skip`: the rows its reply leaves out
    bool send(const int32_t* tokens, int T, int64_t p0, int flags, const float* rows, int skip, std::string& err) {
        tok_.assign(tokens, tokens + (T > 0 ? T : 0));
        return relay_.send(tok_.data(), T, p0, flags, rows, (size_t) row_, skip, err);
    }
    /// the oldest unanswered chunk's reply: rows [skip, T) into `rows` (that chunk's row 0), row by row
    bool recv(float* rows, int T, int skip, std::string& err) {
        return link_.prefill_recv(rows, (size_t) row_, T, skip, err);
    }
    /// the last send out (or its error)
    bool wait(std::string& err) { return relay_.wait(err); }

private:
    net::StageClient& link_;
    net::StageRelay relay_;
    int64_t row_ = 0;
    std::vector<int64_t> tok_;   // (StageRelay keeps its own copy once it has sent)
};

/// The worker's buffers for net::serve_stage: a window's rows (run_in / run_out: max_t rows of handoff_floats, the
/// link's row - the compact one with bf16 rows) and, with prompt chunks (chunk > 0), two chunks' rows in and two out
/// (pf: pf_in[0], pf_in[1], pf_out[0], pf_out[1], chunk rows each of full_floats, the fp32 row: chunks are fp32 only)
inline net::StageBuffers worker_buffers(float* run_in, float* run_out, int max_t, float* const pf[4], int64_t chunk,
                                        int64_t handoff_floats, int64_t full_floats) {
    net::StageBuffers b;
    b.run_in = run_in;
    b.run_out = run_out;
    b.run_floats = (size_t) max_t * (size_t) handoff_floats;
    b.handoff_floats = handoff_floats;
    b.pf_row_floats = full_floats;   // (pf_floats 0: no prompt chunks - every one is refused as too large)
    if (chunk > 0 && pf != nullptr && pf[0] != nullptr && pf[1] != nullptr && pf[2] != nullptr && pf[3] != nullptr) {
        b.pf_in[0] = pf[0];
        b.pf_in[1] = pf[1];
        b.pf_out[0] = pf[2];
        b.pf_out[1] = pf[3];
        b.pf_floats = (size_t) chunk * (size_t) full_floats;
    }
    return b;
}

}  // namespace strata::ds4::stage
