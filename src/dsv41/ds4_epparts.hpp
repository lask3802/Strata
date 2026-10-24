// Strata. The expert-parallel parts pack, read side (tools/dsv41/ep_parts.py writes it; ep-design.md 4.4): the part files
// mapped read-only and the index (ep-parts.txt) parsed once, so a CPU lane or a pool fill reads an expert's n_ff slice
// in place, and a whole expert is put back together from its parts.
//
// An expert is three ggml matrices (gate w1, up w3, down w2).  n_ff is cut in blocks of `unit` (256) values:
//   gate, up  rows [unit a, unit b) of n_ff rows, contiguous
//   down      dim rows, each cut to its blocks [a, b) and stored back to back (row_bytes = (b - a) unit_bytes)
// A part is one matrix of one expert for one block range [a, b), in a file, 4096-aligned.  Several ranges of one expert
// (a worker's [6, 9) and the pool's [0, 6)) live in different files; `view()` hands out the slice [k, b) of whichever part
// holds k (a CPU lane reads the stored [6, 9) from any k >= 6 with no copy), `assemble()` puts every matrix back as the
// pack holds it (byte for byte) when the parts tile [0, nb).
//
// Read only, no device: tests/dsv41/epparts_test.cpp.  The index's pack_hash is ds4_stage.hpp's pack_hash of the pack the
// parts were cut from; the caller compares it with its own pack's (a worker refuses a main process with another pack).
#pragma once

#include "ds4_os.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::ds4::ep {

enum Kind : int { kGate = 0, kUp = 1, kDown = 2 };

/// One matrix of one expert, a slice of n_ff blocks [a, b), as stored: rows x row_bytes bytes at `data`.
struct View {
    const uint8_t* data = nullptr;
    uint32_t rows = 0;         // gate, up: (b - k) unit rows; down: dim rows
    uint32_t row_bytes = 0;    // bytes of one row of the slice [k, b)
    uint64_t stride = 0;       // bytes from one row to the next (gate, up: row_bytes; down: the stored part's row_bytes)
    int type = 0;              // ggml type id
    int a = 0, b = 0;          // the stored range; the view covers [k, b)
};

class Parts {
public:
    struct Entry {
        int layer = 0, expert = 0, kind = 0, type = 0, file = 0, a = 0, b = 0;
        uint32_t rows = 0, row_bytes = 0;
        uint64_t offset = 0, bytes = 0;
    };
    struct File {
        std::string name;
        int a = 0, b = 0;
        uint64_t bytes = 0;
        uint8_t* map = nullptr;
        os::fd_t fd = os::kBadFd;
    };
    struct Heat {
        int layer = 0, expert = 0;
        uint32_t count = 0;
    };

    Parts() = default;
    ~Parts() { close(); }
    Parts(const Parts&) = delete;
    Parts& operator=(const Parts&) = delete;

    /// `dir`: where ep-parts.txt and the files it names are.  False: `err` says what is wrong (a missing or short file,
    /// a part past its file's end, an index line it cannot read).
    bool open(const std::string& dir, std::string& err) {
        close();
        std::ifstream in(dir + "/ep-parts.txt", std::ios::binary);
        if (!in) return fail(err, dir + "/ep-parts.txt: not found");
        std::string line;
        if (!std::getline(in, line) || strip(line) != "ep-parts 1") return fail(err, "ep-parts.txt: not an ep-parts 1 index");
        for (long ln = 2; std::getline(in, line); ++ln) {
            strip(line);
            if (line.empty()) continue;
            std::istringstream s(line);
            std::string tag;
            s >> tag;
            const std::string where = "ep-parts.txt line " + std::to_string(ln) + ": ";
            if (tag == "pack_hash") {
                std::string h;
                s >> h;
                char* end = nullptr;
                hash_ = std::strtoull(h.c_str(), &end, 16);
                if (h.empty() || *end != '\0') return fail(err, where + "bad pack_hash");
            } else if (tag == "geometry") {
                if (!(s >> dim_ >> ff_ >> nl_ >> ne_ >> unit_ >> align_) || unit_ <= 0 || ff_ % unit_ != 0 || nl_ <= 0 || ne_ <= 0)
                    return fail(err, where + "bad geometry");
            } else if (tag == "file") {
                File f;
                int i = 0;
                if (!(s >> i >> f.name >> f.a >> f.b >> f.bytes) || i != (int) files_.size() || f.a < 0 || f.a >= f.b)
                    return fail(err, where + "bad file line");
                files_.push_back(std::move(f));
            } else if (tag == "heat") {
                Heat h;
                if (!(s >> h.layer >> h.expert >> h.count)) return fail(err, where + "bad heat line");
                heat_.push_back(h);
            } else if (tag == "part") {
                Entry e;
                std::string kind;
                if (!(s >> e.layer >> e.expert >> kind >> e.type >> e.file >> e.a >> e.b >> e.rows >> e.row_bytes >> e.offset >> e.bytes))
                    return fail(err, where + "bad part line");
                e.kind = kind == "gate" ? kGate : kind == "up" ? kUp : kind == "down" ? kDown : -1;
                if (e.kind < 0 || e.layer < 0 || e.layer >= nl_ || e.expert < 0 || e.expert >= ne_ || e.file < 0 ||
                    e.file >= (int) files_.size() || e.a < 0 || e.a >= e.b || e.b * unit_ > ff_ ||
                    e.bytes != (uint64_t) e.rows * e.row_bytes ||
                    e.rows != (uint32_t) (e.kind == kDown ? dim_ : (e.b - e.a) * unit_) ||
                    e.a != files_[(size_t) e.file].a || e.b != files_[(size_t) e.file].b)
                    return fail(err, where + "part disagrees with the geometry or its file");
                if (e.offset + e.bytes > files_[(size_t) e.file].bytes) return fail(err, where + "part past its file's end");
                entries_.push_back(e);
            } else {
                return fail(err, where + "unknown tag " + tag);
            }
        }
        if (dim_ <= 0 || files_.empty()) return fail(err, "ep-parts.txt: no geometry or no files");
        std::sort(entries_.begin(), entries_.end(), [](const Entry& x, const Entry& y) {
            return key(x) != key(y) ? key(x) < key(y) : x.a < y.a;
        });
        for (File& f : files_) {
            const std::string path = dir + "/" + f.name;
            f.fd = os::open_ro(path);
            if (f.fd == os::kBadFd) return fail(err, path + ": cannot open");
            if (os::file_size(f.fd) != f.bytes) return fail(err, path + ": " + std::to_string(os::file_size(f.fd)) + " bytes, the index says " + std::to_string(f.bytes));
            f.map = (uint8_t*) os::map_ro(f.fd, f.bytes);
            if (f.map == nullptr) return fail(err, path + ": cannot map");
        }
        return true;
    }

    void close() {
        for (File& f : files_) {
            os::unmap(f.map, f.bytes);
            os::close_fd(f.fd);
        }
        files_.clear();
        entries_.clear();
        heat_.clear();
        hash_ = 0;
        dim_ = ff_ = nl_ = ne_ = 0;
    }

    bool is_open() const { return !files_.empty() && files_[0].map != nullptr; }
    uint64_t pack_hash() const { return hash_; }
    int dim() const { return dim_; }
    int n_ff() const { return ff_; }
    int unit() const { return unit_; }
    int blocks() const { return unit_ > 0 ? ff_ / unit_ : 0; }   // n_ff in units (9)
    int layers() const { return nl_; }
    int experts() const { return ne_; }
    const std::vector<File>& files() const { return files_; }
    const std::vector<Entry>& entries() const { return entries_; }
    /// every listed expert, hottest first (the profile's counts; the order the files were written in)
    const std::vector<Heat>& heat() const { return heat_; }

    /// Matrix `kind` of expert (layer, e) for the blocks [k, b) of whichever part holds k; false: no part does.
    /// `out.data` points into the mapping (valid until close()).  For gate and up it points at the rows of block k; for
    /// down at the row 0's block k, and row r of the slice is at data + r * stride.
    bool view(int layer, int e, int kind, int k, View& out) const {
        const Entry* en = find(layer, e, kind, k);
        if (en == nullptr) return false;
        const uint8_t* base = files_[(size_t) en->file].map + en->offset;
        const uint32_t unit_bytes = en->row_bytes / (uint32_t) (en->b - en->a);   // (down; gate/up: not used below)
        out.type = en->type;
        out.a = en->a;
        out.b = en->b;
        if (kind == kDown) {
            out.data = base + (uint64_t) (k - en->a) * unit_bytes;
            out.rows = en->rows;
            out.row_bytes = (uint32_t) (en->b - k) * unit_bytes;
            out.stride = en->row_bytes;
        } else {
            out.data = base + (uint64_t) (k - en->a) * (uint64_t) unit_ * en->row_bytes;
            out.rows = (uint32_t) (en->b - k) * (uint32_t) unit_;
            out.row_bytes = en->row_bytes;
            out.stride = en->row_bytes;
        }
        return true;
    }

    /// whether the parts of all three matrices tile [0, blocks()) with no gap and no overlap
    bool whole(int layer, int e) const {
        for (int kind = 0; kind < 3; ++kind)
            if (!tiles(layer, e, kind)) return false;
        return true;
    }

    /// bytes of the whole matrix as the pack holds it (0: the parts do not tile it)
    uint64_t whole_bytes(int layer, int e, int kind) const {
        if (!tiles(layer, e, kind)) return 0;
        uint64_t row = 0, rows = 0, n = 0;
        for (const Entry* p = lower(layer, e, kind); p != entries_.data() + entries_.size() && key(*p) == make_key(layer, e, kind); ++p) {
            if (kind == kDown) row += p->row_bytes, rows = p->rows;
            else n += p->bytes;
        }
        return kind == kDown ? row * rows : n;
    }

    /// Matrix `kind` of the whole expert into `dst` (whole_bytes() bytes), as the pack holds it.  False: the parts do not tile it.
    bool assemble(int layer, int e, int kind, uint8_t* dst) const {
        if (!tiles(layer, e, kind)) return false;
        const Entry* p = lower(layer, e, kind);
        const Entry* end = entries_.data() + entries_.size();
        if (kind != kDown) {
            for (; p != end && key(*p) == make_key(layer, e, kind); ++p) {   // (rows are in n_ff order: parts sorted by a)
                std::memcpy(dst, files_[(size_t) p->file].map + p->offset, p->bytes);
                dst += p->bytes;
            }
            return true;
        }
        uint64_t full = 0;
        for (const Entry* q = p; q != end && key(*q) == make_key(layer, e, kind); ++q) full += q->row_bytes;
        uint64_t at = 0;
        for (; p != end && key(*p) == make_key(layer, e, kind); ++p) {
            const uint8_t* src = files_[(size_t) p->file].map + p->offset;
            for (uint32_t r = 0; r < p->rows; ++r) std::memcpy(dst + (uint64_t) r * full + at, src + (uint64_t) r * p->row_bytes, p->row_bytes);
            at += p->row_bytes;
        }
        return true;
    }

    /// The three matrices into `blob` at the byte offsets off[0..2] (an engine blob: gate | up | down, each at its own
    /// aligned place).  False: the parts do not tile the expert.
    bool assemble_blob(int layer, int e, uint8_t* blob, const uint64_t off[3]) const {
        if (!whole(layer, e)) return false;
        for (int kind = 0; kind < 3; ++kind)
            if (!assemble(layer, e, kind, blob + off[kind])) return false;
        return true;
    }

private:
    static uint64_t make_key(int layer, int e, int kind) { return ((uint64_t) layer << 40) | ((uint64_t) e << 8) | (uint64_t) kind; }
    static uint64_t key(const Entry& x) { return make_key(x.layer, x.expert, x.kind); }
    static std::string& strip(std::string& s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
        return s;
    }
    bool fail(std::string& err, const std::string& why) {
        err = why;
        close();
        return false;
    }
    const Entry* lower(int layer, int e, int kind) const {
        const uint64_t k = make_key(layer, e, kind);
        return std::lower_bound(entries_.data(), entries_.data() + entries_.size(), k, [](const Entry& x, uint64_t v) { return key(x) < v; });
    }
    const Entry* find(int layer, int e, int kind, int k) const {
        const Entry* end = entries_.data() + entries_.size();
        for (const Entry* p = lower(layer, e, kind); p != end && key(*p) == make_key(layer, e, kind); ++p)
            if (p->a <= k && k < p->b) return p;
        return nullptr;
    }
    bool tiles(int layer, int e, int kind) const {
        const Entry* end = entries_.data() + entries_.size();
        int at = 0;
        for (const Entry* p = lower(layer, e, kind); p != end && key(*p) == make_key(layer, e, kind); ++p) {
            if (p->a != at) return false;
            at = p->b;
        }
        return at == blocks();
    }

    std::vector<File> files_;
    std::vector<Entry> entries_;   // sorted by (layer, expert, kind, a)
    std::vector<Heat> heat_;
    uint64_t hash_ = 0;
    int dim_ = 0, ff_ = 0, nl_ = 0, ne_ = 0, unit_ = 0, align_ = 0;
};

}  // namespace strata::ds4::ep
