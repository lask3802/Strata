// src/dsv41/ds4_stage.hpp - the DeepSeek engine's remote stage (docs/REMOTE_STAGE.md), the parts with no GPU: the
// model pack's fingerprint both sides' hellos carry, and what a worker refuses in its main process's hello.  The link
// itself is Strata's (src/net/stage_link.cpp), the engine's side of it ds4_main.cpp / ds4_model.cu.
//
// v1: the windows only (T <= 16 rows each way, a prompt in windows of 16), the rows fp32 - each carries the
// indexer's int32 picks and candidate blocks beside the streams, which bf16 would not keep - and no checkpoints: a
// request that does not continue the conversation in place starts it again from token 0.  A draft's check (DSpark on
// the main process) is a window like any other: the worker keeps its layers' backups for every window of 2 rows or
// more, and the main process's Commit says how many of its rows stand (the rest undone there as one process undoes
// them); the drafter's taps of the worker's layers come back in the reply (reply_tap_at).
#pragma once

#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace strata::ds4::stage {

/// row_formats' capability bit both sides of a DeepSeek link set (hello_matches compares neither it nor the other
/// bits 24-30): the worker's reply is reply_tap_at's layout.  An earlier build's worker sent its last layer's streams
/// and pre back, which a main process of this one would read as the normed rows - each side refuses a peer without it.
/// (Bit 30 is the compact rows' capability on dsv41-u4-rows.)
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

/// What this engine's worker cannot serve in a main process's hello (one of the same protocol; hello_matches has the
/// geometry, the layers and the pack): false and why.  `mine`: the worker's own hello, `max_context` its context.
inline bool worker_accepts(const net::StageHello& theirs, const net::StageHello& mine, int64_t max_context,
                           std::string& err) {
    if (theirs.chunk != 0) {
        err = "the main process sends prompt chunks of " + std::to_string(theirs.chunk) +
              " tokens: this worker reads windows of " + std::to_string(mine.max_t) + " only";
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
    if ((theirs.row_formats & kRowsNormed) == 0) {
        err = "the main process is an earlier build's (it reads the last layer's streams back): this worker replies "
              "with the normed rows and the drafter's taps - the same build on both PCs";
        return false;
    }
    return true;
}

/// What a main process cannot use in its worker's hello (StageClient::connect has checked the rest): false and why.
inline bool main_accepts(const net::StageHello& peer, std::string& err) {
    if ((peer.row_formats & kRowsNormed) == 0) {
        err = "remote stage: the worker is an earlier build's (it replies with its last layer's streams): this main "
              "process reads the normed rows and the drafter's taps - the same build on both PCs";
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
template <class M>
bool worker_commit(M& m, int n_keep, int& last_t, std::string& err) {
    const Keep k = commit_kind(n_keep, last_t, err);
    last_t = 0;
    if (k == Keep::Part) return m.stage_rollback(n_keep, err);
    return k == Keep::All;
}

}  // namespace strata::ds4::stage
