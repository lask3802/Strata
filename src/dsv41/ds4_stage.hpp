// src/dsv41/ds4_stage.hpp - the DeepSeek engine's remote stage (docs/REMOTE_STAGE.md), the parts with no GPU: the
// model pack's fingerprint both sides' hellos carry, and what a worker refuses in its main process's hello.  The link
// itself is Strata's (src/net/stage_link.cpp), the engine's side of it ds4_main.cpp / ds4_model.cu.
//
// v1: the windows only (T <= 16 rows each way, a prompt in windows of 16) and no checkpoints: a request that does not
// continue the conversation in place starts it again from token 0.  The rows are fp32 - each carries the indexer's
// int32 picks and candidate blocks beside the streams, which the link's own bf16 would not keep - or, with
// STRATA_REMOTE_ROWS=bf16 on both sides, the engine's compact rows: the streams as bf16, pre and the picks exact (rows_pack).
#pragma once

#include "strata/net/stage_link.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

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

/// The compact rows of STRATA_REMOTE_ROWS=bf16.  The link's own bf16 (StageRows::BF16) converts every float of a
/// message, which would round the int32 picks; this engine's hello lists fp32 for the link and carries this capability
/// bit (row_formats bits 24-30 are free capability bits; the link reads only the format bits and kStageCapKvElem), and
/// the engine packs the rows itself before the link sends them.  Both sides must have it or neither: a hello with it
/// also has a narrower hand-off row (RowGeo::wire), so the link's own check refuses a mismatch as well.
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

}  // namespace strata::ds4::stage
