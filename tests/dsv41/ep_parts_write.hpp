// Strata. A test's parts pack (src/dsv41/ds4_epparts.hpp's format, as tools/dsv41/ep_parts.py writes it): one file of
// the n_ff blocks [a, nb) of every matrix of every expert of some layers, cut from the whole experts' bytes - gate and up
// their rows [256 a, n_ff), down each row's values of those blocks.  For rx_loop_test (the CPU expert server) and
// rx_test (the GPU one): their random experts' parts.  (rx_lane_test has its own, of several files and odd layouts.)
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>
#include <vector>

namespace ep_test {

/// the experts: the geometry, and each matrix's bytes (gate / up n_ff rows of n_embd, down n_embd rows of n_ff), its
/// row's bytes and its ggml type
struct PartsSrc {
    int dim = 0, n_ff = 0, layers = 0, experts = 0;
    std::function<const uint8_t*(int layer, int e, int kind)> mat;
    std::function<size_t(int layer, int kind)> row_bytes;
    std::function<int(int layer, int kind)> type;
};

inline uint64_t index_stamp(const std::string& text) {   // (the index's lines but its stamp's, FNV-1a: ep_parts.py's)
    uint64_t h = 1469598103934665603ull;
    for (size_t at = 0; at < text.size();) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        if (line.empty() || line.compare(0, 6, "stamp ") == 0) continue;
        for (char c : line) h = (h ^ (uint8_t) c) * 1099511628211ull;
        h = (h ^ (uint8_t) '\n') * 1099511628211ull;
    }
    return h;
}

/// dir/ep-parts.txt and dir/ep-parts-A-B.bin: blocks [a, b) of `layers` (b < 0: nb), pack_hash `hash`.  False: not
/// written
inline bool write_parts(const std::string& dir, const PartsSrc& s, int a, const std::vector<int>& layers, uint64_t hash,
                        int b = -1) {
    const int nb = s.n_ff / 256;
    if (b < 0) b = nb;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    char hx[40];
    std::snprintf(hx, sizeof hx, "%016llx", (unsigned long long) hash);
    std::string idx = std::string("ep-parts 1\npack_hash ") + hx + "\nstamp 0000000000000000\nnonce 0000000000000001\n";
    idx += "geometry " + std::to_string(s.dim) + " " + std::to_string(s.n_ff) + " " + std::to_string(s.layers) + " " +
           std::to_string(s.experts) + " 256 4096\ncovered";
    for (int l : layers) idx += " " + std::to_string(l);
    idx += "\n";
    std::string heat, parts;
    std::vector<uint8_t> file(4096, 0);
    for (int l : layers)
        for (int e = 0; e < s.experts; ++e) {
            heat += "heat " + std::to_string(l) + " " + std::to_string(e) + " " + std::to_string(100 + s.experts - e) + "\n";
            for (int kind = 0; kind < 3; ++kind) {
                const uint8_t* m = s.mat(l, e, kind);
                const size_t whole = s.row_bytes(l, kind);
                size_t rb = 0, rows = 0;
                std::vector<uint8_t> p;
                if (kind < 2) {   // (gate, up: the rows of blocks [a, b))
                    rb = whole;
                    rows = (size_t) (b - a) * 256;
                    p.assign(m + (size_t) a * 256 * rb, m + (size_t) b * 256 * rb);
                } else {          // (down: each row's values of blocks [a, b))
                    const size_t ub = whole / (size_t) nb;
                    rb = ub * (size_t) (b - a);
                    rows = (size_t) s.dim;
                    for (size_t r = 0; r < rows; ++r) p.insert(p.end(), m + r * whole + (size_t) a * ub, m + r * whole + (size_t) a * ub + rb);
                }
                file.resize((file.size() + 4095) / 4096 * 4096, 0);
                static const char* kn[3] = {"gate", "up", "down"};
                parts += "part " + std::to_string(l) + " " + std::to_string(e) + " " + kn[kind] + " " + std::to_string(s.type(l, kind)) +
                         " 0 " + std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(rows) + " " + std::to_string(rb) +
                         " " + std::to_string(file.size()) + " " + std::to_string(p.size()) + "\n";
                file.insert(file.end(), p.begin(), p.end());
            }
        }
    file.resize((file.size() + 4095) / 4096 * 4096, 0);
    const std::string name = "ep-parts-" + std::to_string(a) + "-" + std::to_string(b) + ".bin";
    idx += "file 0 " + name + " " + std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(file.size()) + "\n" + heat + parts;
    const uint64_t stamp = index_stamp(idx);
    char sx[40];
    std::snprintf(sx, sizeof sx, "%016llx", (unsigned long long) stamp);
    const size_t at = idx.find("stamp ");
    idx = idx.substr(0, at) + "stamp " + sx + idx.substr(idx.find('\n', at));
    std::memcpy(file.data(), "EPPARTS1", 8);
    for (int i = 0; i < 8; ++i) file[8 + (size_t) i] = (uint8_t) (stamp >> (8 * i));
    const uint32_t hdr[3] = {0u, (uint32_t) a, (uint32_t) b};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 4; ++i) file[16 + (size_t) j * 4 + (size_t) i] = (uint8_t) (hdr[j] >> (8 * i));
    std::ofstream(dir + "/" + name, std::ios::binary | std::ios::trunc).write((const char*) file.data(), (std::streamsize) file.size());
    std::ofstream(dir + "/ep-parts.txt", std::ios::binary | std::ios::trunc) << idx;
    return (bool) std::ifstream(dir + "/ep-parts.txt");
}

}  // namespace ep_test
