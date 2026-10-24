// src/dsv41/ds4_u9.hpp - switches for the decode GPU-part changes (strata-winbuild-lab results/dsv41-perf-20261025/
// gpu-part.md): each change gives the same bits as before and is on by default until measured.
//   DS4_U9=0                         all of them off: the 747981ba path
//   DS4_U9_OFF=quant,hc,...          these off (names below)
// Header-only: the test targets link ds4_gguf.cu / ds4_iq.cu without the model.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::ds4::u9 {

enum : unsigned {
    kQuant = 1u << 0,    // gg::mv: a GEMV input rounded to int8 once (jobs with the same x), contiguous ones in one launch
    kHc = 1u << 1,       // hc_mix / hc_split on a side stream (only hc_post needs them)
    kRoute = 1u << 2,    // the router chain: BF16 gate weights, one fused GEMV for the gate and the lookahead
    kTail = 1u << 3,     // the VRAM experts' down before the wait for the CPU lane
    kQ5k = 1u << 4,      // the Q5_K GEMV with its rows staged in shared memory
    kIxfast = 1u << 5,   // the indexer's short-context path
};

inline unsigned& mask() {   // (a test sets it to compare both paths in one process)
    static unsigned m = [] {
        static const struct {
            const char* name;
            unsigned bit;
        } items[] = {{"quant", kQuant}, {"hc", kHc}, {"route", kRoute}, {"tail", kTail}, {"q5k", kQ5k}, {"ixfast", kIxfast}};
        unsigned v = 0;
        for (const auto& it : items) v |= it.bit;
        const char* all = std::getenv("DS4_U9");
        if (all != nullptr && std::atoi(all) == 0) v = 0;
        const char* off = std::getenv("DS4_U9_OFF");
        for (const char* p = off; p != nullptr && *p != '\0';) {
            const char* e = std::strchr(p, ',');
            const size_t len = e != nullptr ? (size_t) (e - p) : std::strlen(p);
            bool known = len == 0;
            for (const auto& it : items)
                if (std::strlen(it.name) == len && std::strncmp(it.name, p, len) == 0) {
                    v &= ~it.bit;
                    known = true;
                }
            if (!known) std::fprintf(stderr, "ds4 u9: DS4_U9_OFF names '%.*s', not a switch\n", (int) len, p);
            p = e != nullptr ? e + 1 : nullptr;
        }
        std::fprintf(stderr, "ds4 u9: on:");
        for (const auto& it : items)
            if (v & it.bit) std::fprintf(stderr, " %s", it.name);
        std::fprintf(stderr, "%s\n", v == 0 ? " none (the 747981ba path)" : "");
        return v;
    }();
    return m;
}
inline bool on(unsigned item) { return (mask() & item) != 0; }

}  // namespace strata::ds4::u9
