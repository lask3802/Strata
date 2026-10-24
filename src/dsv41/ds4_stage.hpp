// src/dsv41/ds4_stage.hpp - the DeepSeek engine's remote stage (docs/REMOTE_STAGE.md), the parts with no GPU: the
// model pack's fingerprint both sides' hellos carry, what each side refuses in the other's hello, the main process's
// prompt chunks over the link and the worker's buffers for them.  The link itself is Strata's
// (src/net/stage_link.cpp), the engine's side of it ds4_main.cpp / ds4_model.cu.
//
// v1: windows of T <= 16 rows each way, and prompt chunks (Prefill) of up to the hellos' chunk rows - each side's
// chunked prompt path on its own layers, the main process's next chunk through its layers while the worker reads the
// one before.  The rows fp32 - each carries the indexer's int32 picks and candidate blocks beside the streams, which
// bf16 would not keep - and no checkpoints: a request that does not continue the conversation in place starts it
// again from token 0.
//
// A chunk's rows are a window's hand-off rows (Model::StageGeo, handoff_floats a row).  To the worker they go block
// by block as a window's do (T x hc x dim floats, then T x 8, T x topk_stride, T x cand_stride); the reply comes back
// row by row - row r at r x handoff_floats: its streams (hc x dim), then its pre (8), the rest of the row unused - so
// that the reply's skip leaves out whole rows (the main process reads only the rows its head needs).
#pragma once

#include "strata/net/stage_link.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace strata::ds4::stage {

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
        err = "the main process asks for rows other than fp32 (STRATA_REMOTE_ROWS): a DeepSeek row carries the "
              "indexer's int32 picks, which only fp32 keeps";
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
    return true;
}

/// What the main process cannot use in its worker's hello (its HelloOk, after the worker accepted this side's): its
/// chunks larger than the worker's.  A worker that accepted them says it can take them; this is the same check at
/// this end, for a worker whose own check is not this one.
inline bool main_accepts(const net::StageHello& worker, const net::StageHello& mine, std::string& err) {
    if (const std::string r = chunk_refusal(mine.chunk, worker.chunk, worker.max_t); !r.empty()) {
        err = "remote stage: refused: " + r;
        return false;
    }
    return true;
}

/// The main process's prompt chunks over its link: each chunk's send on a thread of its own, one in flight at a time
/// in order (net::StageRelay - a chunk's rows stay as they are until the next send() or wait() returns), its reply
/// read by recv() in the same order, on any one thread.
class ChunkLink {
public:
    /// `row_floats`: a hand-off row's floats (Model::StageGeo::handoff_floats)
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

/// The worker's buffers for net::serve_stage: a window's rows (run_in / run_out: max_t rows) and, with prompt chunks
/// (chunk > 0), two chunks' rows in and two out (pf: pf_in[0], pf_in[1], pf_out[0], pf_out[1], chunk rows each) - every
/// row handoff_floats wide
inline net::StageBuffers worker_buffers(float* run_in, float* run_out, int max_t, float* const pf[4], int64_t chunk,
                                        int64_t handoff_floats) {
    net::StageBuffers b;
    b.run_in = run_in;
    b.run_out = run_out;
    b.run_floats = (size_t) max_t * (size_t) handoff_floats;
    b.handoff_floats = handoff_floats;
    b.pf_row_floats = handoff_floats;   // (pf_floats 0: no prompt chunks - every one is refused as too large)
    if (chunk > 0 && pf != nullptr && pf[0] != nullptr && pf[1] != nullptr && pf[2] != nullptr && pf[3] != nullptr) {
        b.pf_in[0] = pf[0];
        b.pf_in[1] = pf[1];
        b.pf_out[0] = pf[2];
        b.pf_out[1] = pf[3];
        b.pf_floats = (size_t) chunk * (size_t) handoff_floats;
    }
    return b;
}

}  // namespace strata::ds4::stage
