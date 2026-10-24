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
const Ty kTypes[] = {{"IQ1_M", 29, 56}, {"IQ2_XXS", 16, 66}, {"IQ2_XS", 17, 74}};

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
    int dim = 512, ff = 2304, nl = 3, ne = 5;
    int nb() const { return ff / kUnit; }
    int ty[3][3] = {{0, 0, 1}, {1, 1, 2}, {2, 2, 0}};   // per layer: gate, up, down -> kTypes index
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

// writes the files and the index for `sets`; `drop` experts are left out of the first set only
void write_parts(const std::string& dir, const Geo& g, const std::map<std::pair<int, int>, Expert>& ex, const std::vector<SetSpec>& sets,
                 uint64_t hash) {
    fs::create_directories(dir);
    std::string idx = "ep-parts 1\npack_hash " + [&] {
        char b[32];
        std::snprintf(b, sizeof b, "%016llx", (unsigned long long) hash);
        return std::string(b);
    }() + "\n";
    idx += "geometry " + std::to_string(g.dim) + " " + std::to_string(g.ff) + " " + std::to_string(g.nl) + " " + std::to_string(g.ne) + " 256 4096\n";
    std::string parts, heat;
    std::set<std::pair<int, int>> heated;
    for (size_t fi = 0; fi < sets.size(); ++fi) {
        const SetSpec& s = sets[fi];
        const std::string name = "ep-parts-" + std::to_string(s.a) + "-" + std::to_string(s.b) + ".bin";
        std::vector<uint8_t> file;
        for (auto [l, e] : s.experts) {
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
            if (heated.insert({l, e}).second) heat += "heat " + std::to_string(l) + " " + std::to_string(e) + " " + std::to_string(1000 - (int) heated.size()) + "\n";
        }
        file.resize((file.size() + kAlign - 1) / kAlign * kAlign, 0);
        idx += "file " + std::to_string(fi) + " " + name + " " + std::to_string(s.a) + " " + std::to_string(s.b) + " " + std::to_string(file.size()) + "\n";
        std::ofstream f(dir + "/" + name, std::ios::binary | std::ios::trunc);
        f.write((const char*) file.data(), (std::streamsize) file.size());
    }
    std::ofstream f(dir + "/ep-parts.txt", std::ios::binary | std::ios::trunc);
    f << idx << heat << parts;
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
                    // a view that starts inside the stored part: k + 1 of [k, nb) (the worker reading a larger k in place)
                    if (k + 1 < g.nb()) {
                        CHECK(p.view(l, e, kind, k + 1, v) && v.a == k);
                        const std::vector<uint8_t> w1 = cut(g, l, kind, orig, k + 1, g.nb());
                        CHECK(v.rows * (size_t) v.row_bytes == w1.size());
                        bool s1 = true;
                        for (uint32_t r = 0; r < v.rows; ++r) s1 = s1 && std::memcmp(v.data + (size_t) r * v.stride, w1.data() + (size_t) r * v.row_bytes, v.row_bytes) == 0;
                        CHECK(s1);
                    }
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
void refusal_tests(const fs::path& root) {
    const Geo g;
    const auto ex = make_experts(g);
    const std::string dir = (root / "ref").string();
    write_parts(dir, g, ex, {{6, 9, all_experts(g)}}, 99);
    ep::Parts p;
    std::string err;
    CHECK(p.open(dir, err));
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
    // an index with a part past its file's end, a bad kind, a bad header
    auto slurp = [&](const std::string& f) {
        std::ifstream in(f, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const std::string good = slurp(dir + "/ep-parts.txt");
    auto with = [&](const std::string& body) {
        std::ofstream(dir + "/ep-parts.txt", std::ios::binary | std::ios::trunc) << body;
        return p.open(dir, err);
    };
    CHECK(!with("ep-parts 2\n" + good.substr(good.find('\n') + 1)));
    {   // the last part's offset moved past its file's end
        const size_t at = good.rfind("part ");
        std::istringstream in(good.substr(at));
        std::vector<std::string> tok((size_t) 12);
        for (auto& t : tok) in >> t;
        tok[10] = "99999999999";
        std::string line;
        for (size_t i = 0; i < tok.size(); ++i) line += (i ? " " : "") + tok[i];
        CHECK(!with(good.substr(0, at) + line + "\n") && err.find("past") != std::string::npos);
    }
    CHECK(!with(good + "part 0 0 sideways 29 0 6 9 1 1 0 1\n"));
    CHECK(!with(good + "bogus 1\n"));
    // CRLF index (a pack written on Windows): read the same
    {
        std::string crlf;
        for (char c : good) {
            if (c == '\n') crlf += '\r';
            crlf += c;
        }
        CHECK(with(crlf));
        CHECK(p.pack_hash() == 99);
    }
    CHECK(with(good));
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
                break;   // (one view per part is enough: k is the part's first block, the next parts follow)
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
    std::printf("real pack: %zu experts looked at, %zu part views, %zu whole experts assembled\n", done.size(), n_parts, n_whole);
    return fails == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3) {
        const int rc = real_pack(argv[1], argv[2], argc > 3 ? argv[3] : "all");
        std::printf("%s\n", rc == 0 ? "ds4_epparts_test (real pack): all passed" : "ds4_epparts_test (real pack): FAILED");
        return rc;
    }
    const fs::path root = fs::temp_directory_path() / ("ds4_epparts_test_" + std::to_string((unsigned long long) std::time(nullptr)));
    fs::create_directories(root);
    roundtrip_tests(root);
    tiling_tests(root);
    refusal_tests(root);
    std::error_code ec;
    fs::remove_all(root, ec);
    std::printf("%s\n", fails == 0 ? "ds4_epparts_test: all passed" : "ds4_epparts_test: FAILED");
    return fails == 0 ? 0 : 1;
}
