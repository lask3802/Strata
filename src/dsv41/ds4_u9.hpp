// src/dsv41/ds4_u9.hpp - switches for the decode GPU-part changes (strata-winbuild-lab results/dsv41-perf-20261025/
// gpu-part.md): each change gives the same bits as before and is on by default until measured.
//   DS4_U9=0                         all of them off: the 747981ba path
//   DS4_U9_OFF=quant,hc,...          these off (names below)
// DS4_ATTN_SPLIT (ds4_kernels.cu, other bits in the last place) keeps its own env; unset, it follows defaults() below.
// So DS4_U9_OFF naming all six still leaves the split on - not DS4_U9=0's bits; add DS4_ATTN_SPLIT=0 for those.
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

inline bool defaults() {   // false: DS4_U9=0 (and HIP without DS4_U9=1) - the switches' defaults are the old path
    const char* all = std::getenv("DS4_U9");
#if defined(STRATA_USE_HIP)
    return all != nullptr && std::atoi(all) == 1;   // (measured on CUDA only: HIP keeps the old path unless DS4_U9=1)
#else
    return all == nullptr || std::atoi(all) != 0;
#endif
}

// q5k by the device: the staged Q5_K GEMV is 11-15% faster a grid on the 2080 Ti (sm75) but 10-20% slower on a 3070
// (sm86), so with q5k on it runs on compute capability 7.5 only (ds4_iq.cu). DS4_U9_Q5K=1 / =0: every device / none.
inline int& q5k_arch() {   // -1: by the device (a test sets 1)
    static int v = [] {
        const char* e = std::getenv("DS4_U9_Q5K");
        return e == nullptr ? -1 : std::atoi(e) != 0 ? 1 : 0;
    }();
    return v;
}

inline unsigned& mask() {   // (a test sets it to compare both paths in one process)
    static unsigned m = [] {
        static const struct {
            const char* name;
            unsigned bit;
        } items[] = {{"quant", kQuant}, {"hc", kHc}, {"route", kRoute}, {"tail", kTail}, {"q5k", kQ5k}, {"ixfast", kIxfast}};
        unsigned v = 0;
        for (const auto& it : items) v |= it.bit;
        if (!defaults()) v = 0;
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
        int shown = 0;
        for (const auto& it : items) {   // (q5k on sm75 only unless DS4_U9_Q5K says: DS4_U9_Q5K=0 leaves it out)
            if ((v & it.bit) == 0 || (it.bit == kQ5k && q5k_arch() == 0)) continue;
            std::fprintf(stderr, " %s%s", it.name, it.bit == kQ5k && q5k_arch() < 0 ? " (sm75)" : "");
            ++shown;
        }
        std::fprintf(stderr, "%s\n", shown == 0 ? " none (the 747981ba path)" : "");
        return v;
    }();
    return m;
}
inline bool on(unsigned item) { return (mask() & item) != 0; }

}  // namespace strata::ds4::u9
