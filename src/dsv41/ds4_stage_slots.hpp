// src/dsv41/ds4_stage_slots.hpp - DS4_STAGE_N=N: the prompt flow's pinned staging with more slots than its VRAM ring
// has.  The flow reads an expert into staging slot (file items so far) % S and DMAs it into the ring once the read is
// done, at most ring_n ahead of the compute; a read waits only for its slot's last DMA, so the reads run at most
// S + ring_n experts ahead of the compute - 72 with S = ring_n (36) while a layer streams ~330, the readers idle while
// a chunk's attention runs.  A larger S lets them go further; the ring (VRAM) stays as it is.  Unset, empty, not a
// whole number, or N <= ring_n: S = ring_n, as before.  A device off the flow keeps ring_n.  Pure function, no state
// (tests/dsv41/stage_slots_test.cpp).
#pragma once

#include <cerrno>
#include <cstdlib>

namespace strata::ds4 {

/// the most slots DS4_STAGE_N may ask for (a typo's worth of digits is not an allocation to try; the pinned allocation
/// that does not fit falls back to ring_n anyway)
constexpr int kStageSlotsMax = 1 << 20;

/// staging slots for a device: env is DS4_STAGE_N's text (nullptr: unset), ring_n the ring's slots, on_flow whether the
/// device streams its prompt through the flow
inline int stage_slots(const char* env, int ring_n, bool on_flow) {
    if (!on_flow || env == nullptr || *env == '\0') return ring_n;
    char* end = nullptr;
    errno = 0;
    const long long n = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0' || errno != 0) return ring_n;
    if (n <= (long long) ring_n) return ring_n;
    return n > kStageSlotsMax ? kStageSlotsMax : (int) n;
}

}  // namespace strata::ds4
