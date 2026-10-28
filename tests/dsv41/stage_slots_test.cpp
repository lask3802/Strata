// Strata. DS4_STAGE_N (ds4_stage_slots.hpp): the flow's staging slots - N when that is more than the ring's slots and
// the device is on the flow; ring_n otherwise (unset, empty, not a whole number, N <= ring_n, a device off the flow).
// No GPU, no model.
#include "../../src/dsv41/ds4_stage_slots.hpp"

#include <cstdio>
#include <initializer_list>

namespace ds4 = strata::ds4;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

int main() {
    const int R = 36;
    check(ds4::stage_slots(nullptr, R, true) == R, "unset: ring_n");
    check(ds4::stage_slots("", R, true) == R, "empty: ring_n");
    for (const char* bad : {"abc", "216x", "x216", " ", "2 16", "1e3", "0x100", "-", "+"})
        check(ds4::stage_slots(bad, R, true) == R, bad);
    check(ds4::stage_slots("0", R, true) == R, "0: ring_n");
    check(ds4::stage_slots("-216", R, true) == R, "negative: ring_n");
    check(ds4::stage_slots("35", R, true) == R, "below ring_n: ring_n");
    check(ds4::stage_slots("36", R, true) == R, "ring_n itself: ring_n");
    check(ds4::stage_slots("37", R, true) == 37, "ring_n + 1: 37");
    check(ds4::stage_slots("216", R, true) == 216, "216: 216");
    check(ds4::stage_slots("+216", R, true) == 216, "+216: 216");
    check(ds4::stage_slots("216", 72, true) == 216 && ds4::stage_slots("216", 216, true) == 216 &&
              ds4::stage_slots("216", 300, true) == 300,
          "ring_n 72 / 216 / 300: max(ring_n, N)");
    check(ds4::stage_slots("216", R, false) == R, "a device off the flow: ring_n");
    check(ds4::stage_slots(nullptr, R, false) == R, "unset, off the flow: ring_n");
    check(ds4::stage_slots("1048576", R, true) == ds4::kStageSlotsMax, "the cap itself");
    check(ds4::stage_slots("1048577", R, true) == ds4::kStageSlotsMax, "past the cap: the cap");
    check(ds4::stage_slots("99999999999", R, true) == ds4::kStageSlotsMax, "past int: the cap");
    check(ds4::stage_slots("99999999999999999999999", R, true) == R, "past long long: unparsable, ring_n");
    std::printf(fails == 0 ? "ALL OK\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
