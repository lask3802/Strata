// src/dsv41/ds4_stage.hpp - the DeepSeek engine's remote stage (docs/REMOTE_STAGE.md), the parts with no GPU: the
// model pack's fingerprint both sides' hellos carry, and what a worker refuses in its main process's hello.  The link
// itself is Strata's (src/net/stage_link.cpp), the engine's side of it ds4_main.cpp / ds4_model.cu.
//
// v1: the windows only (T <= 16 rows each way, a prompt in windows of 16), the rows fp32 - each carries the
// indexer's int32 picks and candidate blocks beside the streams, which bf16 would not keep.  Checkpoints and slots
// (STRATA_DS4_SLOT_KEEP=1 on the main process) ride the link's CkptSave / CkptRestore (below); without them a request
// that does not continue the conversation in place starts it again from token 0.
#pragma once

#include "ds4_partstore.hpp"
#include "strata/net/stage_link.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <utility>
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
    if (theirs.max_context > max_context || theirs.max_t > mine.max_t || theirs.max_t < 1) {
        err = "the main process's context (" + std::to_string(theirs.max_context) + ") or window (" +
              std::to_string(theirs.max_t) + ") is larger than this worker's (" + std::to_string(max_context) + ", " +
              std::to_string(mine.max_t) + ")";
        return false;
    }
    return true;
}

// ---- conversation checkpoints and slots across the link.  The link's CkptSave / CkptRestore carry an id, a position L
// and the ids the main process holds (keep[]); the DeepSeek engine gives the id a meaning of its own, as its state is
// not a Qwen engine's: the engine's four checkpoints (the end of the last prompt, the position before it, the anchor, the
// shared start - Model's kChecks) are rings and partial groups both ends keep by role at the same positions, and a slot
// is the whole of a conversation's state, which the worker keeps in a PartStore under the main process's slot id.
//   CkptSave    kCkRole0 + r   the worker takes its checkpoint of role r (L is its position, as the link checks)
//               kCkPin         the worker pins (checkpoint 1 also becomes the anchor; L is its position)
//               >= kSlotId0    the worker sets its part of the conversation aside under that id
//   CkptRestore kCkAt          the worker goes back to its checkpoint at L (the main process chose it)
//               >= kSlotId0    the worker takes its part of that slot back (L: the slot's length)
// keep[] holds slot ids only (the checkpoints are kept by role, not by id): the worker drops the parts not in it.
constexpr int kCkRoles = 4;   // the model's kChecks
constexpr int64_t kCkRole0 = 1;
constexpr int64_t kCkPin = kCkRole0 + kCkRoles;
constexpr int64_t kCkAt = kCkPin + 1;
constexpr int64_t kSlotId0 = 16;
/// the slots the main process keeps in all (RAM and folder) when its worker keeps a part of each: keep[] is the link's
/// to carry (kStageCkptMax ids), and the hello's ckpt_max says so
constexpr int kMainSlotsMax = 32;
static_assert(kMainSlotsMax + 1 <= net::kStageCkptMax, "the main process's slots and one being saved fit keep[]");

enum class CkOp { Role, Pin, At, Slot, Bad };
/// what a CkptSave id asks for (*role: for Role)
inline CkOp ck_save_op(int64_t id, int* role = nullptr) {
    if (id >= kCkRole0 && id < kCkRole0 + kCkRoles) {
        if (role != nullptr) *role = (int) (id - kCkRole0);
        return CkOp::Role;
    }
    if (id == kCkPin) return CkOp::Pin;
    return id >= kSlotId0 ? CkOp::Slot : CkOp::Bad;
}
/// what a CkptRestore id asks for
inline CkOp ck_restore_op(int64_t id) {
    if (id == kCkAt) return CkOp::At;
    return id >= kSlotId0 ? CkOp::Slot : CkOp::Bad;
}

/// The worker's answers to Reset (the main process starts again; `keep`: the slots it still has) and to the link ending
/// (a new main process is a new id space: nothing is kept)
template <class M>
void worker_reset(M& m, PartStore& parts, const std::vector<int64_t>& keep) {
    m.reset();
    parts.prune(keep);
}
template <class M>
void worker_disconnected(M& m, PartStore& parts) {
    m.reset();
    parts.clear();
}

/// The worker's answer to a CkptSave over a model `m` (Model, or a stand-in in the tests) that has pos(), checkpoint(role),
/// pin(), slot_save(bytes, err) and keeps its slot parts in `parts`: 1 kept, 0 not kept (nothing changed here - the main
/// process goes on without it), -1 an error in `e` (the session is lost).  `note`: a line for the log, if any.
/// The link has checked that L is the position the worker's session is at.
template <class M>
int worker_ck_save(M& m, PartStore& parts, int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e,
                   std::string& note) {
    int role = 0;
    switch (ck_save_op(id, &role)) {
    case CkOp::Role:
        if (L != m.pos()) {
            e = "a checkpoint at " + std::to_string(L) + ", this worker is at " + std::to_string(m.pos());
            return -1;
        }
        m.checkpoint(role);
        parts.prune(keep);
        return 1;
    case CkOp::Pin:
        m.pin();
        parts.prune(keep);
        return 1;
    case CkOp::Slot: {
        std::vector<uint8_t> buf;
        size_t n = 0;
        std::string why;
        bool put = false;
        try {
            if (m.slot_save(buf, e) == 0) {
                note = "slot " + std::to_string(id) + " not kept (" + e + ")";
                e.clear();
                return 0;
            }
            n = buf.size();
            put = parts.put(id, L, std::move(buf), why);
        } catch (const std::bad_alloc&) {   // (the part does not fit in memory: the main process reads that conversation again)
            why = "out of memory";
        }
        if (!put) {
            note = "slot " + std::to_string(id) + " (" + std::to_string(n >> 20) + " MB) not kept (" + why + ")";
            return 0;
        }
        std::vector<int64_t> k = keep;
        k.push_back(id);
        parts.prune(k);
        note = "slot " + std::to_string(id) + " kept (" + std::to_string(L) + "-token conversation, " +
               std::to_string(n >> 20) + " MB); " + std::to_string(parts.ram_parts()) + " in RAM, " +
               std::to_string(parts.disk_parts()) + " in the folder";
        return 1;
    }
    default:
        e = "checkpoint id " + std::to_string(id) + " is none this worker keeps";
        return -1;
    }
}

/// The worker's answer to a CkptRestore (also over a model with restorable(limit), restore(limit) and slot_load(data,
/// bytes, n_pos, err)): true done; false with e - the start kStageCkptMissing when it does not hold what was asked for
/// and nothing changed (the main process then reads the conversation from token 0), else the session is lost.  `note`: a
/// line for the log, if any.
template <class M>
bool worker_ck_restore(M& m, PartStore& parts, int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e,
                       std::string& note) {
    switch (ck_restore_op(id)) {
    case CkOp::At:
        if (m.restorable(L) != L) {   // (it would go to an earlier one, or none: nothing is changed)
            e = std::string(net::kStageCkptMissing) + "the checkpoint at " + std::to_string(L);
            return false;
        }
        if (!m.restore(L) || m.pos() != L) {
            e = "going back to " + std::to_string(L) + " failed";
            return false;
        }
        parts.prune(keep);
        note = "went back to token " + std::to_string(L);
        return true;
    case CkOp::Slot: {
        std::vector<uint8_t> buf;
        std::string why;
        PartStore::Get got = PartStore::Get::Failed;
        try {
            got = parts.get(id, L, buf, why);
        } catch (const std::bad_alloc&) {
            why = "out of memory taking part " + std::to_string(id) + " back";
        }
        switch (got) {
        case PartStore::Get::Missing:
            e = std::string(net::kStageCkptMissing) + std::to_string(id);
            return false;
        case PartStore::Get::Failed:
            e = why;
            return false;
        case PartStore::Get::Ok:
            break;
        }
        if (!m.slot_load(buf.data(), buf.size(), L, e)) return false;   // (a load that failed part way reset the state)
        parts.prune(keep);
        note = "slot " + std::to_string(id) + " taken back (" + std::to_string(L) + "-token conversation)";
        return true;
    }
    default:
        e = "checkpoint id " + std::to_string(id) + " is none this worker keeps";
        return false;
    }
}

}  // namespace strata::ds4::stage
