// The EP parts pack's read side (src/dsv41/ds4_epparts.hpp) without a GPU or a model.
//   ds4_epparts_test                       synthetic packs: parts written here (a naive per-row cutter, not
//                                          tools/dsv41/ep_parts.py's), read back; exit 0: all passed
//   ds4_epparts_test PACK PARTS [LAYERS]   a pack and the parts ep_parts.py wrote from it (read-only on both): every part
//                                          equals the pack's own slice, every whole expert assembles to the pack's bytes,
//                                          the index's pack_hash is ds4_stage.hpp's for PACK; LAYERS: "3" or "0-9,12";
//                                          "none": only the index load time (the files may be sparse)
#include "ds4_epparts.hpp"
#include "ds4_stage.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ep = strata::ds4::ep;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);               \
            ++fails;                                                       \
        }                                                                  \
    } while (0)

namespace {

constexpr int kUnit = 256, kAlign = 4096;

struct Ty {
    const char* name;
    int id, bytes;   // a 256-value block's bytes
};
const Ty kTypes[] = {{"IQ1_M", 29, 56}, {"IQ2_XXS", 16, 66}, {"IQ2_XS", 17, 74}, {"MXFP4", 39, 136}};

uint64_t rng_state = 0x9e3779b97f4a7c15ull;
uint8_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint8_t) (rng_state >> 24);
}
std::vector<uint8_t> noise(size_t n) {
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = rnd();
    return v;
}

// one synthetic expert: gate / up / down as the pack holds them
struct Expert {
    std::vector<uint8_t> m[3];
};
struct Geo {
    int dim = 512, ff = 2304, nl = 4, ne = 5;
    int nb() const { return ff / kUnit; }
    int ty[4][3] = {{0, 0, 1}, {1, 1, 2}, {2, 2, 0}, {3, 3, 3}};   // per layer: gate, up, down -> kTypes index
    size_t row_bytes(int l, int kind) const { return (size_t) (kind == 2 ? ff : dim) / kUnit * kTypes[ty[l][kind]].bytes; }
    size_t rows(int kind) const { return (size_t) (kind == 2 ? dim : ff); }
};

std::map<std::pair<int, int>, Expert> make_experts(const Geo& g) {
    std::map<std::pair<int, int>, Expert> out;
    for (int l = 0; l < g.nl; ++l)
        for (int e = 0; e < g.ne; ++e)
            for (int k = 0; k < 3; ++k) out[{l, e}].m[k] = noise(g.rows(k) * g.row_bytes(l, k));
    return out;
}

// the naive cutter: part [a, b) of a matrix, row by row
std::vector<uint8_t> cut(const Geo& g, int l, int kind, const std::vector<uint8_t>& whole, int a, int b) {
    const size_t rb = g.row_bytes(l, kind), ub = rb / (kind == 2 ? (size_t) g.nb() : (size_t) (g.dim / kUnit));
    std::vector<uint8_t> out;
    if (kind < 2) {
        out.assign(whole.begin() + (long) ((size_t) a * kUnit * rb), whole.begin() + (long) ((size_t) b * kUnit * rb));
        return out;
    }
    for (size_t r = 0; r < g.rows(2); ++r)
        out.insert(out.end(), whole.begin() + (long) (r * rb + (size_t) a * ub), whole.begin() + (long) (r * rb + (size_t) b * ub));
    return out;
}

struct SetSpec {
    int a, b;
    std::vector<std::pair<int, int>> experts;   // heat order
};

// FNV-1a over every line but the stamp's own, each '\n' ended (the index's own check)
uint64_t index_stamp(const std::string& text) {
    uint64_t h = 1469598103934665603ull;
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(at, nl - at);
        at = nl + 1;
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.compare(0, 6, "stamp ") == 0) continue;
        for (char c : line) h = (h ^ (uint8_t) c) * 1099511628211ull;
        h = (h ^ (uint8_t) '\n') * 1099511628211ull;
    }
    return h;
}
std::string hex16(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}
// the index text with its stamp line made right again (after a test edited a line on purpose)
std::string restamp(const std::string& text) {
    const size_t at = text.find("stamp ");
    const size_t nl = text.find('\n', at);
    return text.substr(0, at) + "stamp " + hex16(index_stamp(text)) + text.substr(nl);
}
void put_le(std::vector<uint8_t>& v, size_t off, uint64_t x, int n) {
    for (int i = 0; i < n; ++i) v[off + (size_t) i] = (uint8_t) (x >> (8 * i));
}

// writes the files and the index for `sets` (the naive way: whole files in memory, row by row cuts).  `salt` changes the
// heat counts, so two runs over the same experts differ in their index (and so their stamp) but not in any size; `nonce`
// alone does the same when nothing else differs (a re-cut with the same layout whose weights changed).
void write_parts(const std::string& dir, const Geo& g, const std::map<std::pair<int, int>, Expert>& ex, const std::vector<SetSpec>& sets,
                 uint64_t hash, int salt = 0, uint64_t nonce = 1) {
    fs::create_directories(dir);
    std::string head = "ep-parts 1\npack_hash " + hex16(hash) + "\nstamp 0000000000000000\nnonce " + hex16(nonce) + "\n";
    head += "geometry " + std::to_string(g.dim) + " " + std::to_string(g.ff) + " " + std::to_string(g.nl) + " " + std::to_string(g.ne) + " 256 4096\n";
    std::string files_txt, parts, heat;
    std::set<std::pair<int, int>> heated, layers;
    std::vector<std::vector<uint8_t>> files;
    std::vector<std::string> names;
    for (size_t fi = 0; fi < sets.size(); ++fi) {
        const SetSpec& s = sets[fi];
        const std::string name = "ep-parts-" + std::to_string(s.a) + "-" + std::to_string(s.b) + ".bin";
        std::vector<uint8_t> file(kAlign, 0);   // the header page, filled in once the stamp is known
        for (auto [l, e] : s.experts) {
            layers.insert({l, 0});
            for (int kind = 0; kind < 3; ++kind) {
                file.resize((file.size() + kAlign - 1) / kAlign * kAlign, 0);
                const std::vector<uint8_t> p = cut(g, l, kind, ex.at({l, e}).m[kind], s.a, s.b);
                const size_t rb = kind == 2 ? p.size() / g.rows(2) : g.row_bytes(l, kind), rows = p.size() / rb;
                static const char* kn[3] = {"gate", "up", "down"};
                parts += "part " + std::to_string(l) + " " + std::to_string(e) + " " + kn[kind] + " " + std::to_string(kTypes[g.ty[l][kind]].id) +
                         " " + std::to_string(fi) + " " + std::to_string(s.a) + " " + std::to_string(s.b) + " " + std::to_string(rows) + " " +
                         std::to_string(rb) + " " + std::to_string(file.size()) + " " + std::to_string(p.size()) + "\n";
                file.insert(file.end(), p.begin(), p.end());
            }
            if (heated.insert({l, e}).second)
                heat += "heat " + std::to_string(l) + " " + std::to_string(e) + " " + std::to_string(1000 - (int) heated.size() + salt) + "\n";
        }
        file.resize((file.size() + kAlign - 1) / kAlign * kAlign, 0);
        files_txt += "file " + std::to_string(fi) + " " + name + " " + std::to_string(s.a) + " " + std::to_string(s.b) + " " + std::to_string(file.size()) + "\n";
        files.push_back(std::move(file));
        names.push_back(name);
    }
    std::string covered = "covered";
    for (auto& l : layers) covered += " " + std::to_string(l.first);
    const std::string idx = restamp(head + covered + "\n" + files_txt + heat + parts);
    const uint64_t stamp = index_stamp(idx);
    for (size_t fi = 0; fi < files.size(); ++fi) {
        std::memcpy(files[fi].data(), "EPPARTS1", 8);
        put_le(files[fi], 8, stamp, 8);
        put_le(files[fi], 16, fi, 4);
        put_le(files[fi], 20, (uint64_t) sets[fi].a, 4);
        put_le(files[fi], 24, (uint64_t) sets[fi].b, 4);
        std::ofstream f(dir + "/" + names[fi], std::ios::binary | std::ios::trunc);
        f.write((const char*) files[fi].data(), (std::streamsize) files[fi].size());
    }
    std::ofstream f(dir + "/ep-parts.txt", std::ios::binary | std::ios::trunc);
    f << idx;
}

std::vector<std::pair<int, int>> all_experts(const Geo& g) {
    std::vector<std::pair<int, int>> v;
    for (int e = 0; e < g.ne; ++e)
        for (int l = 0; l < g.nl; ++l) v.push_back({l, e});
    return v;
}

// ---- parts -> whole == the original bytes, for every type and every split point k
void roundtrip_tests(const fs::path& root) {
    const Geo g;
    const auto ex = make_experts(g);
    for (int k = 1; k < g.nb(); ++k) {
        const std::string dir = (root / ("k" + std::to_string(k))).string();
        // the worker's tail [k, nb) of every expert, the pool's head [0, k) of the odd ones (heat order: reversed index order)
        std::vector<std::pair<int, int>> head, tail = all_experts(g);
        for (auto x : tail)
            if (x.second % 2 == 1) head.push_back(x);
        std::reverse(head.begin(), head.end());
        write_parts(dir, g, ex, {{k, g.nb(), tail}, {0, k, head}}, 0x1234abcd5678ef01ull);
        ep::Parts p;
        std::string err;
        CHECK(p.open(dir, err));
        if (!p.is_open()) {
            std::printf("  k %d: %s\n", k, err.c_str());
            continue;
        }
        CHECK(p.pack_hash() == 0x1234abcd5678ef01ull);
        CHECK(p.dim() == g.dim && p.n_ff() == g.ff && p.layers() == g.nl && p.experts() == g.ne && p.blocks() == g.nb());
        CHECK(p.files().size() == 2);
        CHECK(p.heat().size() == tail.size());
        for (int l = 0; l < g.nl; ++l)
            for (int e = 0; e < g.ne; ++e) {
                const bool odd = e % 2 == 1;
                CHECK(p.whole(l, e) == odd);   // (the even ones have only the tail)
                for (int kind = 0; kind < 3; ++kind) {
                    const std::vector<uint8_t>& orig = ex.at({l, e}).m[kind];
                    CHECK(p.whole_bytes(l, e, kind) == (odd ? orig.size() : 0));
                    std::vector<uint8_t> got(orig.size(), 0xEE);
                    CHECK(p.assemble(l, e, kind, got.data()) == odd);
                    if (odd) CHECK(got == orig);
                    // the slice [k, nb) read in place, row by row against the original
                    ep::View v;
                    CHECK(p.view(l, e, kind, k, v));
                    CHECK(v.type == kTypes[g.ty[l][kind]].id && v.a == k && v.b == g.nb());
                    const std::vector<uint8_t> want = cut(g, l, kind, orig, k, g.nb());
                    CHECK(v.rows * (size_t) v.row_bytes == want.size());
                    bool same = true;
                    for (uint32_t r = 0; r < v.rows; ++r) same = same && std::memcmp(v.data + (size_t) r * v.stride, want.data() + (size_t) r * v.row_bytes, v.row_bytes) == 0;
                    CHECK(same);
                    // a view that starts inside the stored part: every k2 of (k, nb) of [k, nb) (the worker reading a larger
                    // k in place, U6: a stored [6, 9) read at 7 and 8); the down rows stay the stored part's apart
                    const uint64_t stored_stride = v.stride;
                    for (int k2 = k + 1; k2 < g.nb(); ++k2) {
                        CHECK(p.view(l, e, kind, k2, v) && v.a == k && v.b == g.nb());
                        CHECK(v.stride == (kind == ep::kDown ? stored_stride : (uint64_t) v.row_bytes));
                        const std::vector<uint8_t> w1 = cut(g, l, kind, orig, k2, g.nb());
                        CHECK(v.rows * (size_t) v.row_bytes == w1.size());
                        bool s1 = true;
                        for (uint32_t r = 0; r < v.rows; ++r) s1 = s1 && std::memcmp(v.data + (size_t) r * v.stride, w1.data() + (size_t) r * v.row_bytes, v.row_bytes) == 0;
                        CHECK(s1);
                    }
                    // below the stored a: refused where only the tail is stored (the even ones), never a view into the head
                    if (!odd) CHECK(!p.view(l, e, kind, k - 1, v));
                    else CHECK(p.view(l, e, kind, k - 1, v) && v.a == 0 && v.b == k);
                    CHECK(!p.view(l, e, kind, g.nb(), v));   // (past the end)
                }
                if (odd) {   // the engine blob: gate | up | down, each at its own aligned place
                    auto r256 = [](size_t n) { return (uint64_t) ((n + 255) / 256 * 256); };
                    uint64_t offs[3] = {256, 0, 0};
                    offs[1] = offs[0] + r256(ex.at({l, e}).m[0].size()) + 256;
                    offs[2] = offs[1] + r256(ex.at({l, e}).m[1].size()) + 256;
                    std::vector<uint8_t> blob(offs[2] + ex.at({l, e}).m[2].size() + 256, 0xAB);
                    CHECK(p.assemble_blob(l, e, blob.data(), offs));
                    for (int kind = 0; kind < 3; ++kind)
                        CHECK(std::memcmp(blob.data() + offs[kind], ex.at({l, e}).m[kind].data(), ex.at({l, e}).m[kind].size()) == 0);
                    CHECK(blob[0] == 0xAB && blob[255] == 0xAB);   // (the gaps untouched)
                } else {
                    uint8_t one[8];
                    const uint64_t offs[3] = {0, 0, 0};
                    CHECK(!p.assemble_blob(l, e, one, offs));
                }
            }
    }
}

// ---- three parts per matrix, a gap, an overlap
void tiling_tests(const fs::path& root) {
    const Geo g;
    const auto ex = make_experts(g);
    const std::string dir = (root / "three").string();
    write_parts(dir, g, ex, {{0, 2, all_experts(g)}, {2, 5, all_experts(g)}, {5, 9, {{0, 0}, {1, 1}}}}, 7);
    ep::Parts p;
    std::string err;
    CHECK(p.open(dir, err));
    for (auto [l, e] : all_experts(g)) {
        const bool tiled = (l == 0 && e == 0) || (l == 1 && e == 1);
        CHECK(p.whole(l, e) == tiled);
        if (!tiled) continue;
        for (int kind = 0; kind < 3; ++kind) {
            std::vector<uint8_t> got(ex.at({l, e}).m[kind].size());
            CHECK(p.assemble(l, e, kind, got.data()));
            CHECK(got == ex.at({l, e}).m[kind]);
        }
    }
    p.close();
    // a gap: [0,2) and [5,9)
    write_parts((root / "gap").string(), g, ex, {{0, 2, {{0, 0}}}, {5, 9, {{0, 0}}}}, 7);
    CHECK(p.open((root / "gap").string(), err));
    CHECK(!p.whole(0, 0));
    uint8_t buf[4];
    CHECK(!p.assemble(0, 0, 0, buf));
    ep::View v;
    CHECK(p.view(0, 0, 0, 1, v) && v.b == 2);
    CHECK(!p.view(0, 0, 0, 3, v));   // (the gap)
    p.close();
    // an overlap: [0,6) and [4,9)
    write_parts((root / "overlap").string(), g, ex, {{0, 6, {{0, 0}}}, {4, 9, {{0, 0}}}}, 7);
    CHECK(p.open((root / "overlap").string(), err));
    CHECK(!p.whole(0, 0));
    p.close();
}

// ---- an index or a file that does not hold together is refused with a reason, never read
std::string slurp(const std::string& f) {
    std::ifstream in(f, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
void spit(const std::string& f, const std::string& body) { std::ofstream(f, std::ios::binary | std::ios::trunc) << body; }
// an index edited on purpose, with its stamp made right and the part files' headers following it (so it still opens)
void spit_index(const std::string& dir, const std::string& text) {
    const std::string idx = restamp(text);
    spit(dir + "/ep-parts.txt", idx);
    uint8_t le[8];
    for (int i = 0; i < 8; ++i) le[i] = (uint8_t) (index_stamp(idx) >> (8 * i));
    std::istringstream in(idx);
    for (std::string line; std::getline(in, line);) {
        std::istringstream s(line);
        std::string tag, i, name;
        s >> tag >> i >> name;
        if (tag != "file") continue;
        std::fstream f(dir + "/" + name, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(8);
        f.write((const char*) le, 8);
    }
}

// the index text with token `idx` of its last part line replaced (the stamp made right again, so only that field is wrong)
std::string edit_last_part(const std::string& text, std::vector<std::pair<size_t, std::string>> edits, const std::string& append = "") {
    const size_t at = text.rfind("part ");
    std::istringstream in(text.substr(at));
    std::vector<std::string> tok((size_t) 12);
    for (auto& t : tok) in >> t;
    for (auto& [idx, value] : edits) tok[idx] = value;
    std::string line;
    for (size_t i = 0; i < tok.size(); ++i) line += (i ? " " : "") + tok[i];
    return restamp(text.substr(0, at) + line + append + "\n");
}

void refusal_tests(const fs::path& root) {
    const Geo g;
    const auto ex = make_experts(g);
    const std::string dir = (root / "ref").string();
    write_parts(dir, g, ex, {{6, 9, all_experts(g)}}, 99);
    ep::Parts p;
    std::string err;
    CHECK(p.open(dir, err));
    CHECK(p.stamp() != 0 && p.covers(0) && p.covers(g.nl - 1) && !p.covers(g.nl));
    p.close();
    CHECK(!p.open((root / "nowhere").string(), err) && !err.empty());
    // a file one byte short
    {
        const std::string f = dir + "/ep-parts-6-9.bin";
        const auto sz = fs::file_size(f);
        fs::resize_file(f, sz - 1);
        CHECK(!p.open(dir, err) && err.find("bytes") != std::string::npos);
        fs::resize_file(f, sz);
        CHECK(p.open(dir, err));
        p.close();
    }
    const std::string good = slurp(dir + "/ep-parts.txt");
    auto with = [&](const std::string& body) {
        spit(dir + "/ep-parts.txt", body);
        return p.open(dir, err);
    };
    // each of these changes one thing, and its stamp is made right again, so the reason is the field's own
    auto refused = [&](const std::string& body, const char* why) {
        const bool ok = !with(restamp(body));
        const bool said = err.find(why) != std::string::npos;
        if (!said) std::printf("  wanted '%s', got '%s'\n", why, err.c_str());
        return ok && said;
    };
    CHECK(refused("ep-parts 2\n" + good.substr(good.find('\n') + 1), "not an ep-parts 1 index"));
    CHECK(refused(good + "bogus 1\n", "unknown tag"));
    CHECK(refused(good.substr(0, good.rfind("part ")) + "part 0 0 down 29 0 6 9 1 1 4096 1\n", "disagrees"));
    CHECK(refused(edit_last_part(good, {{3, "sideways"}}), "out of range"));            // the kind
    CHECK(refused(edit_last_part(good, {{10, "17592186044416"}}), "past"));                 // an offset past the file's end
    CHECK(refused(edit_last_part(good, {{10, "18446744073709547520"}}), "past"));        // ... or one that would wrap offset + bytes
    CHECK(refused(edit_last_part(good, {{10, "-4096"}}), "bad part line"));              // ... or a negative one
    CHECK(refused(edit_last_part(good, {{10, "4097"}}), "not aligned"));
    CHECK(refused(edit_last_part(good, {{10, "0"}}), "header page"));
    CHECK(refused(edit_last_part(good, {{11, "1"}}), "disagrees"));                      // bytes != rows x row_bytes
    CHECK(refused(edit_last_part(good, {{9, "199"}, {11, "101888"}}), "disagrees"));    // down: row_bytes not a multiple of (b - a), the rest right
    CHECK(refused(edit_last_part(good, {{1, "5"}}), "out of range"));                    // a layer the geometry lacks
    CHECK(refused(edit_last_part(good, {{2, "5"}}), "out of range"));                    // an expert it lacks
    CHECK(refused(edit_last_part(good, {{0, "part"}}, " extra"), "bad part line"));      // a token too many
    CHECK(refused(good + "heat 0 0 1 2\n", "bad heat line"));
    {   // a part of a layer the covered line leaves out
        std::string t = good;
        const size_t at = t.find("covered ");
        t.replace(at, t.find('\n', at) - at, "covered 0 1");
        CHECK(refused(t, "does not cover"));
    }
    {   // lines in the wrong order: a part before the geometry
        const size_t gat = good.find("geometry ");
        const size_t gend = good.find('\n', gat) + 1;
        std::string t = good.substr(0, gat) + good.substr(good.rfind("part ")) + good.substr(gat, gend - gat);
        CHECK(!with(restamp(t)));
    }
    // an index edited by hand (the stamp no longer matches), and files and index from different runs
    {
        std::string t = good;
        const size_t at = t.find("heat ");
        t[t.find(' ', t.find(' ', at + 5) + 1) + 1] = '7';   // the first heat line's count
        CHECK(!with(t) && err.find("stamp") != std::string::npos);
        CHECK(with(good));
        p.close();
    }
    {
        const std::string other = (root / "ref2").string();
        write_parts(other, g, ex, {{6, 9, all_experts(g)}}, 99, 1);   // another run: same sizes, other heat counts
        CHECK(fs::file_size(other + "/ep-parts-6-9.bin") == fs::file_size(dir + "/ep-parts-6-9.bin"));
        CHECK(slurp(other + "/ep-parts.txt") != good);
        const std::string mine = slurp(dir + "/ep-parts-6-9.bin");
        spit(dir + "/ep-parts-6-9.bin", slurp(other + "/ep-parts-6-9.bin"));    // the new run's file under the old index
        CHECK(!p.open(dir, err) && err.find("another run") != std::string::npos);
        spit(dir + "/ep-parts-6-9.bin", mine);
        CHECK(p.open(dir, err));
        p.close();
        spit(dir + "/ep-parts.txt", slurp(other + "/ep-parts.txt"));            // ... and the other way round
        CHECK(!p.open(dir, err) && err.find("another run") != std::string::npos);
        spit(dir + "/ep-parts.txt", good);
        // a file with no header
        std::string raw = mine;
        raw[0] = 'X';
        spit(dir + "/ep-parts-6-9.bin", raw);
        CHECK(!p.open(dir, err) && err.find("EPPARTS1") != std::string::npos);
        spit(dir + "/ep-parts-6-9.bin", mine);
        CHECK(p.open(dir, err));
        p.close();
    }
    // two files: the SECOND one from another run (here only the nonce differs: same layout, same counts, same sizes), or with
    // the wrong file number or range in its header, is refused too - every file is checked, not the first alone
    {
        const std::string two = (root / "two").string(), two2 = (root / "two2").string();
        const std::vector<SetSpec> sp = {{6, 9, all_experts(g)}, {0, 6, {{0, 0}, {1, 1}}}};
        write_parts(two, g, ex, sp, 99);
        write_parts(two2, g, ex, sp, 99, 0, 2);
        CHECK(slurp(two + "/ep-parts.txt") != slurp(two2 + "/ep-parts.txt"));
        CHECK(fs::file_size(two + "/ep-parts-0-6.bin") == fs::file_size(two2 + "/ep-parts-0-6.bin"));
        CHECK(p.open(two, err));
        CHECK(p.nonce() == 1);
        p.close();
        const std::string second = two + "/ep-parts-0-6.bin", mine = slurp(second);
        spit(second, slurp(two2 + "/ep-parts-0-6.bin"));
        CHECK(!p.open(two, err) && err.find("ep-parts-0-6.bin") != std::string::npos && err.find("another run") != std::string::npos);
        for (size_t at : {16, 20, 24}) {   // the file number, a, b
            std::string raw = mine;
            raw[at] = (char) (raw[at] ^ 1);
            spit(second, raw);
            CHECK(!p.open(two, err) && err.find("ep-parts-0-6.bin") != std::string::npos && err.find("another run") != std::string::npos);
        }
        spit(second, mine);
        CHECK(p.open(two, err));
        p.close();
    }
    // an index of the old format (before the stamp, the nonce or the covered line) says so
    for (const char* drop : {"stamp ", "nonce ", "covered "}) {
        std::string t;
        std::istringstream in(good);
        for (std::string line; std::getline(in, line);)
            if (line.compare(0, std::strlen(drop), drop) != 0) t += line + "\n";
        const bool ok = !with(std::string(drop) == "stamp " ? t : restamp(t));
        if (err.find("old format") == std::string::npos) std::printf("  wanted 'old format', got '%s'\n", err.c_str());
        CHECK(ok && err.find("old format") != std::string::npos);
    }
    CHECK(with(good));
    p.close();
    // CRLF index (a pack written on Windows): read the same
    {
        std::string crlf;
        for (char c : good) {
            if (c == '\n') crlf += '\r';
            crlf += c;
        }
        CHECK(with(crlf));
        CHECK(p.pack_hash() == 99);
        p.close();
    }
    CHECK(with(good));
    p.close();
}

// ---- parts that cannot be joined are not: another type in one part, another layer set
void layout_tests(const fs::path& root) {
    const Geo g;
    const auto ex = make_experts(g);
    // two sets for one expert, the second one's type changed in the index (a layer of another layout cut into the same pack)
    const std::string dir = (root / "mixed").string();
    write_parts(dir, g, ex, {{6, 9, all_experts(g)}, {0, 6, {{0, 0}}}}, 5);
    ep::Parts p;
    std::string err;
    CHECK(p.open(dir, err) && p.whole(0, 0) && p.whole(0, 1) == false);
    p.close();
    const std::string good = slurp(dir + "/ep-parts.txt");
    uint8_t buf[8];
    // the pool's down part says IQ2_XS (17), the worker's says IQ2_XXS (16): the index opens, the parts do not join
    spit_index(dir, edit_last_part(good, {{4, "17"}}));
    CHECK(p.open(dir, err) || (std::printf("  open: %s\n", err.c_str()), false));
    CHECK(!p.whole(0, 0) && !p.assemble(0, 0, ep::kDown, buf) && p.whole(0, 1) == false);
    ep::View v0;
    CHECK(p.view(0, 0, ep::kDown, 2, v0) && v0.type == 17 && p.view(0, 0, ep::kDown, 7, v0) && v0.type == 16);
    p.close();
    // ... and a part whose bytes a unit are not the other's (row_bytes 390 = 6 x 65, the worker's 3 x 66)
    spit_index(dir, edit_last_part(good, {{9, "390"}, {11, "199680"}}));
    CHECK(p.open(dir, err) || (std::printf("  open: %s\n", err.c_str()), false));
    CHECK(!p.whole(0, 0));
    p.close();
    // a parts pack of one layer: the others have no view, no whole, no assemble
    const std::string one = (root / "onelayer").string();
    std::vector<std::pair<int, int>> l1;
    for (int e = 0; e < g.ne; ++e) l1.push_back({1, e});
    write_parts(one, g, ex, {{0, 9, l1}}, 5);
    CHECK(p.open(one, err));
    CHECK(p.covers(1) && !p.covers(0) && !p.covers(2));
    ep::View v;
    CHECK(p.view(1, 0, ep::kGate, 4, v) && !p.view(0, 0, ep::kGate, 4, v) && !p.view(2, 0, ep::kGate, 4, v));
    CHECK(p.whole(1, 0) && !p.whole(0, 0));
    CHECK(!p.assemble(0, 0, ep::kGate, buf));
    p.close();
}

// ---- the real pack (args)
struct Tens {
    std::string dtype;
    int file = 0;
    uint64_t off = 0, bytes = 0;
};

std::vector<int> parse_layers(const std::string& s, int nl) {
    std::vector<int> out;
    if (s.empty() || s == "all") {
        for (int l = 0; l < nl; ++l) out.push_back(l);
        return out;
    }
    size_t at = 0;
    while (at < s.size()) {
        const size_t c = s.find(',', at);
        const std::string part = s.substr(at, c == std::string::npos ? c : c - at);
        const size_t d = part.find('-');
        if (d == std::string::npos) out.push_back(std::atoi(part.c_str()));
        else
            for (int l = std::atoi(part.substr(0, d).c_str()); l <= std::atoi(part.substr(d + 1).c_str()); ++l) out.push_back(l);
        if (c == std::string::npos) break;
        at = c + 1;
    }
    return out;
}

int real_pack(const std::string& pack, const std::string& parts, const std::string& layers_arg) {
    std::map<std::string, std::string> kv;
    {
        std::ifstream f(pack + "/ds4.txt");
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
        }
    }
    std::map<int, std::string> files;
    std::map<std::string, Tens> tens;
    {
        std::ifstream f(pack + "/tensors.txt");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream s(line);
            std::string tag;
            s >> tag;
            if (tag == "file") {
                int i;
                std::string n;
                s >> i >> n;
                files[i] = n;
            } else if (tag == "t") {
                std::string name;
                Tens t;
                s >> name >> t.dtype >> t.file >> t.off >> t.bytes;
                tens[name] = t;
            }
        }
    }
    const int dim = std::atoi(kv["hidden_size"].c_str()), ff = std::atoi(kv["moe_intermediate_size"].c_str());
    const int nl = std::atoi(kv["num_hidden_layers"].c_str());
    ep::Parts p;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    const bool opened = p.open(parts, err);
    const double open_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!opened) {
        std::printf("FAIL open %s: %s\n", parts.c_str(), err.c_str());
        return 1;
    }
    CHECK(p.dim() == dim && p.n_ff() == ff && p.layers() == nl);
    CHECK(p.pack_hash() == strata::ds4::stage::pack_hash(pack));
    CHECK(p.pack_hash() != 0);
    std::printf("index: %zu parts in %zu files, %zu experts in heat order, pack_hash %016llx\n", p.entries().size(), p.files().size(),
                p.heat().size(), (unsigned long long) p.pack_hash());
    std::printf("index load (parse, open and map the files): %.1f ms\n", open_ms);
    if (layers_arg == "none") return fails == 0 ? 0 : 1;   // (the load time only: the files may be sparse)
    const std::vector<int> layers = parse_layers(layers_arg, nl);
    const std::set<int> want(layers.begin(), layers.end());
    std::map<int, std::ifstream> fh;
    auto original = [&](int l, int e, int kind) {
        static const char* role[3] = {"w1", "w3", "w2"};
        const Tens& t = tens.at("layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + "." + role[kind] + ".weight");
        std::ifstream& f = fh[t.file];
        if (!f.is_open()) f.open(kv["model_dir"] + "/" + files.at(t.file), std::ios::binary);
        std::vector<uint8_t> v((size_t) t.bytes);
        f.clear();
        f.seekg((std::streamoff) t.off);
        f.read((char*) v.data(), (std::streamsize) t.bytes);
        return v;
    };
    size_t n_parts = 0, n_whole = 0;
    std::set<std::pair<int, int>> done;
    for (const auto& en : p.entries()) {
        if (!want.count(en.layer) || !done.insert({en.layer, en.expert}).second) continue;
        for (int kind = 0; kind < 3; ++kind) {
            const std::vector<uint8_t> orig = original(en.layer, en.expert, kind);
            const size_t rows = kind == 2 ? (size_t) dim : (size_t) ff, rb = orig.size() / rows, ub = kind == 2 ? rb / (size_t) p.blocks() : 0;
            for (int k = 0; k < p.blocks(); ++k) {
                ep::View v;
                if (!p.view(en.layer, en.expert, kind, k, v)) continue;
                ++n_parts;
                // the slice [k, b) of the original, independently: gate/up rows k*256.., down row by row
                bool same = true;
                if (kind < 2) {
                    same = v.rows == (uint32_t) (v.b - k) * 256 && v.row_bytes == rb;
                    for (uint32_t r = 0; same && r < v.rows; ++r) same = std::memcmp(v.data + (size_t) r * v.stride, orig.data() + ((size_t) k * 256 + r) * rb, rb) == 0;
                } else {
                    same = v.rows == (uint32_t) dim && v.row_bytes == (uint32_t) (v.b - k) * ub;
                    for (uint32_t r = 0; same && r < v.rows; ++r) same = std::memcmp(v.data + (size_t) r * v.stride, orig.data() + r * rb + (size_t) k * ub, v.row_bytes) == 0;
                }
                if (!same) std::printf("FAIL layer %d expert %d kind %d view k=%d\n", en.layer, en.expert, kind, k);
                CHECK(same);
            }
            if (p.whole_bytes(en.layer, en.expert, kind) != 0) {
                std::vector<uint8_t> got(orig.size(), 0xCD);
                CHECK(p.whole_bytes(en.layer, en.expert, kind) == orig.size());
                CHECK(p.assemble(en.layer, en.expert, kind, got.data()));
                const bool same = got == orig;
                if (!same) std::printf("FAIL layer %d expert %d kind %d: assembled bytes differ\n", en.layer, en.expert, kind);
                CHECK(same);
                if (kind == 0) ++n_whole;
            }
        }
    }
    std::printf("real pack: %zu experts looked at, %zu slice views (every k that has a part: the stored start and the in-place k inside it), %zu whole experts assembled\n", done.size(), n_parts, n_whole);
    return fails == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3) {
        const int rc = real_pack(argv[1], argv[2], argc > 3 ? argv[3] : "all");
        std::printf("%s\n", rc == 0 ? "ds4_epparts_test (real pack): all passed" : "ds4_epparts_test (real pack): FAILED");
        return rc;
    }
    // (a folder of its own: the clock in nanoseconds, and a taken name is skipped - two runs in one second do not meet)
    fs::path root;
    for (unsigned long long n = (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();; ++n) {
        root = fs::temp_directory_path() / ("ds4_epparts_test_" + std::to_string(n));
        if (fs::create_directories(root)) break;
    }
    roundtrip_tests(root);
    tiling_tests(root);
    refusal_tests(root);
    layout_tests(root);
    std::error_code ec;
    fs::remove_all(root, ec);
    std::printf("%s\n", fails == 0 ? "ds4_epparts_test: all passed" : "ds4_epparts_test: FAILED");
    return fails == 0 ? 0 : 1;
}
