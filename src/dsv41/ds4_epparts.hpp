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
// Every part file starts with a header page (EPPARTS1, the run's stamp, the file's number and range) and the index carries
// the same stamp, a hash of its own lines (one of them a random per-run nonce, so a re-cut that changed only the weights
// has another stamp too): open() refuses files from another run than the index (a re-cut with another heat order has
// the same sizes), an index that was edited, a part that overflows or leaves its file, and parts of one
// matrix that do not agree on type and row layout (they would not join).  `covered` lists the layers that have parts; the
// others have no view.  The index is read in order: a line may use only what the lines before it said.
//
// Read only, no device: tests/dsv41/epparts_test.cpp.  The index's pack_hash is ds4_stage.hpp's pack_hash of the pack the
// parts were cut from; the caller compares it with its own pack's (a worker refuses a main process with another pack).
#pragma once

#include "ds4_os.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
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

    /// `dir`: where ep-parts.txt and the files it names are.  False: `err` says what is wrong (a missing or short file, a
    /// file from another run than the index, a part past its file's end, an index line it cannot read).
    ///
    /// The index is read line by line and a line may use only what the lines before it said: geometry, covered and the
    /// files come before the heat and the parts (the writer emits them in that order); a line that breaks it is refused.
    bool open(const std::string& dir, std::string& err) {
        close();
        std::ifstream in(dir + "/ep-parts.txt", std::ios::binary);
        if (!in) return fail(err, dir + "/ep-parts.txt: not found");
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        uint64_t fnv = 1469598103934665603ull;   // (the stamp checks every line but its own, each '\n' ended, no '\r')
        bool first = true, have_stamp = false, have_nonce = false, have_geometry = false, have_covered = false;
        long ln = 0;
        for (size_t at = 0; at < text.size();) {
            size_t nl = text.find('\n', at);
            if (nl == std::string::npos) nl = text.size();
            std::string line = text.substr(at, nl - at);
            at = nl + 1;
            ++ln;
            strip(line);
            if (line.empty()) continue;
            const std::string where = "ep-parts.txt line " + std::to_string(ln) + ": ";
            if (first) {
                if (line != "ep-parts 1") return fail(err, "ep-parts.txt: not an ep-parts 1 index");
                first = false;
            }
            if (line.compare(0, 6, "stamp ") != 0) {
                for (char c : line) fnv = (fnv ^ (uint8_t) c) * 1099511628211ull;
                fnv = (fnv ^ (uint8_t) '\n') * 1099511628211ull;
            }
            std::istringstream s(line);
            std::string tag;
            s >> tag;
            if (tag == "ep-parts") {
                std::string v;
                if (!(s >> v) || v != "1" || !done(s)) return fail(err, where + "bad header");
            } else if (tag == "pack_hash" || tag == "stamp" || tag == "nonce") {
                std::string h;
                char* end = nullptr;
                if (!(s >> h) || !done(s)) return fail(err, where + "bad " + tag);
                const uint64_t v = std::strtoull(h.c_str(), &end, 16);
                if (h.empty() || h[0] == '-' || h[0] == '+' || *end != '\0') return fail(err, where + "bad " + tag);
                if (tag == "stamp") stamp_ = v, have_stamp = true;
                else if (tag == "nonce") nonce_ = v, have_nonce = true;
                else hash_ = v;
            } else if (tag == "geometry") {
                uint64_t g[6] = {};
                for (uint64_t& x : g)
                    if (!unum(s, x) || x > 0x7fffffffu) return fail(err, where + "bad geometry");
                if (!done(s)) return fail(err, where + "bad geometry");
                dim_ = (int) g[0], ff_ = (int) g[1], nl_ = (int) g[2], ne_ = (int) g[3], unit_ = (int) g[4], align_ = (int) g[5];
                if (dim_ <= 0 || ff_ <= 0 || unit_ <= 0 || ff_ % unit_ != 0 || nl_ <= 0 || ne_ <= 0 || align_ <= 0 || kHeader % align_ != 0)
                    return fail(err, where + "bad geometry");
                have_geometry = true;
            } else if (tag == "covered") {
                if (!have_geometry) return fail(err, where + "covered before geometry");
                uint64_t l = 0;
                while (unum(s, l)) {
                    if (l >= (uint64_t) nl_) return fail(err, where + "covered layer out of range");
                    covered_.push_back((int) l);
                }
                if (!done(s)) return fail(err, where + "bad covered line");
                std::sort(covered_.begin(), covered_.end());
                have_covered = true;
            } else if (tag == "file") {
                if (!have_geometry) return fail(err, where + "file before geometry");
                File f;
                uint64_t i = 0, a = 0, b = 0, bytes = 0;
                if (!unum(s, i) || !(s >> f.name) || !unum(s, a) || !unum(s, b) || !unum(s, bytes) || !done(s) || i != files_.size() ||
                    a >= b || b * (uint64_t) unit_ > (uint64_t) ff_ || bytes < kHeader)
                    return fail(err, where + "bad file line");
                f.a = (int) a, f.b = (int) b, f.bytes = bytes;
                files_.push_back(std::move(f));
            } else if (tag == "heat") {
                uint64_t l = 0, e = 0, c = 0;
                if (!have_geometry || !unum(s, l) || !unum(s, e) || !unum(s, c) || !done(s) || l >= (uint64_t) nl_ || e >= (uint64_t) ne_ || c > 0xffffffffu)
                    return fail(err, where + "bad heat line");
                heat_.push_back({(int) l, (int) e, (uint32_t) c});
            } else if (tag == "part") {
                Entry e;
                std::string kind;
                uint64_t v[10] = {};   // layer expert type file a b rows row_bytes offset bytes
                if (!have_geometry || !have_covered)
                    return fail(err, where + "part before the geometry and covered lines (an index of the old format has no covered line: "
                                              "write the parts again)");
                if (!unum(s, v[0]) || !unum(s, v[1]) || !(s >> kind) || !unum(s, v[2]) || !unum(s, v[3]) ||
                    !unum(s, v[4]) || !unum(s, v[5]) || !unum(s, v[6]) || !unum(s, v[7]) || !unum(s, v[8]) || !unum(s, v[9]) || !done(s))
                    return fail(err, where + "bad part line");
                e.kind = kind == "gate" ? kGate : kind == "up" ? kUp : kind == "down" ? kDown : -1;
                if (e.kind < 0 || v[0] >= (uint64_t) nl_ || v[1] >= (uint64_t) ne_ || v[2] > 0x7fffffffu || v[3] >= files_.size() || v[4] >= v[5] ||
                    v[5] > 0x7fffffffu || v[6] > 0xffffffffu || v[7] > 0xffffffffu)
                    return fail(err, where + "part has a field out of range");
                e.layer = (int) v[0], e.expert = (int) v[1], e.type = (int) v[2], e.file = (int) v[3], e.a = (int) v[4], e.b = (int) v[5];
                e.rows = (uint32_t) v[6], e.row_bytes = (uint32_t) v[7], e.offset = v[8], e.bytes = v[9];
                const File& f = files_[(size_t) e.file];
                if (!std::binary_search(covered_.begin(), covered_.end(), e.layer)) return fail(err, where + "part of a layer the index does not cover");
                if ((uint64_t) e.b * (uint64_t) unit_ > (uint64_t) ff_ || e.a != f.a || e.b != f.b || e.bytes != (uint64_t) e.rows * e.row_bytes ||
                    e.row_bytes == 0 || e.rows != (uint32_t) (e.kind == kDown ? (int64_t) dim_ : (int64_t) (e.b - e.a) * unit_) ||
                    (e.kind == kDown && e.row_bytes % (uint32_t) (e.b - e.a) != 0))
                    return fail(err, where + "part disagrees with the geometry or its file");
                if (e.offset < kHeader || e.offset % (uint64_t) align_ != 0) return fail(err, where + "part inside the header page or not aligned");
                if (e.offset > f.bytes || e.bytes > f.bytes - e.offset) return fail(err, where + "part past its file's end");
                entries_.push_back(e);
            } else {
                return fail(err, where + "unknown tag " + tag);
            }
        }
        if (first) return fail(err, "ep-parts.txt: empty");
        if (!have_geometry || !have_covered || files_.empty()) return fail(err, "ep-parts.txt: no geometry, covered line or files");
        if (!have_stamp || !have_nonce)
            return fail(err, "ep-parts.txt: an index of the old format (no stamp or nonce line): write the parts again");
        if (stamp_ != fnv) return fail(err, "ep-parts.txt: its stamp does not match its own lines (edited or damaged)");
        std::sort(entries_.begin(), entries_.end(), [](const Entry& x, const Entry& y) {
            return key(x) != key(y) ? key(x) < key(y) : x.a < y.a;
        });
        for (size_t i = 0; i < files_.size(); ++i) {
            File& f = files_[i];
            const std::string path = dir + "/" + f.name;
            f.fd = os::open_ro(path);
            if (f.fd == os::kBadFd) return fail(err, path + ": cannot open");
            if (os::file_size(f.fd) != f.bytes) return fail(err, path + ": " + std::to_string(os::file_size(f.fd)) + " bytes, the index says " + std::to_string(f.bytes));
            f.map = (uint8_t*) os::map_ro(f.fd, f.bytes);
            if (f.map == nullptr) return fail(err, path + ": cannot map");
            // the header page names the run: EPPARTS1, the stamp (u64), the file's number, a, b (u32), all little endian
            auto le = [&](size_t off, int n) {
                uint64_t x = 0;
                for (int k = n - 1; k >= 0; --k) x = x << 8 | f.map[off + (size_t) k];
                return x;
            };
            if (std::memcmp(f.map, "EPPARTS1", 8) != 0) return fail(err, path + ": not a parts file (no EPPARTS1 header)");
            if (le(8, 8) != stamp_ || le(16, 4) != i || le(20, 4) != (uint64_t) f.a || le(24, 4) != (uint64_t) f.b)
                return fail(err, path + ": its header is from another run than ep-parts.txt (stale index or files)");
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
        covered_.clear();
        hash_ = stamp_ = nonce_ = 0;
        dim_ = ff_ = nl_ = ne_ = unit_ = align_ = 0;
    }

    bool is_open() const { return !files_.empty() && files_[0].map != nullptr; }
    uint64_t pack_hash() const { return hash_; }
    uint64_t stamp() const { return stamp_; }   // the run that wrote the index and its files
    uint64_t nonce() const { return nonce_; }   // that run's random number (part of what the stamp hashes)
    /// whether the index has parts for the layer (a parts pack of a few layers answers false for the others)
    bool covers(int layer) const { return std::binary_search(covered_.begin(), covered_.end(), layer); }
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
    static constexpr uint64_t kHeader = 4096;   // a part file's first page: its header (EPPARTS1, stamp, file, a, b)
    /// the next token as a plain decimal number (no sign, no letters, fits 64 bits)
    static bool unum(std::istringstream& s, uint64_t& v) {
        std::string t;
        if (!(s >> t) || t.size() > 20) return false;
        for (char c : t)
            if (c < '0' || c > '9') return false;
        errno = 0;
        v = std::strtoull(t.c_str(), nullptr, 10);
        return errno == 0;
    }
    static bool done(std::istringstream& s) {   // no token left on the line
        std::string x;
        return !(s >> x);
    }
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
        if (!covers(layer)) return nullptr;
        const Entry* end = entries_.data() + entries_.size();
        for (const Entry* p = lower(layer, e, kind); p != end && key(*p) == make_key(layer, e, kind); ++p)
            if (p->a <= k && k < p->b) return p;
        return nullptr;
    }
    // the parts of one matrix cover [0, blocks()) with no gap or overlap, and agree on the type and the row geometry
    // (down: the bytes of one unit of a row; gate, up: the bytes of a row) - parts of two different layouts never join
    bool tiles(int layer, int e, int kind) const {
        if (!covers(layer)) return false;
        const Entry* end = entries_.data() + entries_.size();
        int at = 0;
        const Entry* head = nullptr;
        for (const Entry* p = lower(layer, e, kind); p != end && key(*p) == make_key(layer, e, kind); ++p) {
            if (p->a != at) return false;
            if (head == nullptr) head = p;
            if (p->type != head->type) return false;
            if (kind == kDown ? (p->rows != head->rows || p->row_bytes / (uint32_t) (p->b - p->a) != head->row_bytes / (uint32_t) (head->b - head->a))
                              : p->row_bytes != head->row_bytes)
                return false;
            at = p->b;
        }
        return at == blocks();
    }

    std::vector<File> files_;
    std::vector<Entry> entries_;   // sorted by (layer, expert, kind, a)
    std::vector<Heat> heat_;
    std::vector<int> covered_;   // sorted
    uint64_t hash_ = 0, stamp_ = 0, nonce_ = 0;
    int dim_ = 0, ff_ = 0, nl_ = 0, ne_ = 0, unit_ = 0, align_ = 0;
};

}  // namespace strata::ds4::ep
