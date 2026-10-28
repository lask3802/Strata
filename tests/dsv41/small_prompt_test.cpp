// Strata. DS4_SMALL_PROMPT / DS4_SMALL_WIN (ds4_small_prompt.hpp): which feeds take the small windows (2-N tokens,
// one row of logits at most, no remote stage; 0 off: none, whatever the feed) and DS4_SMALL_WIN's rows held to
// 1..min(kWin, DS4_ASYNC_T).  No GPU, no model.
#include "../../src/dsv41/ds4_small_prompt.hpp"

#include <cstdio>

namespace ds4 = strata::ds4;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

int main() {
    {
        bool any = false;   // off (0, the default; 1 and below too): no feed, of any size or logits
        for (int sn : {-5, 0, 1})
            for (int n = 0; n <= 4096; ++n)
                for (int lr = 0; lr <= 2; ++lr)
                    for (int rw = 0; rw < 2; ++rw) any = any || ds4::small_prompt(n, lr, sn, rw != 0);
        check(!any, "off: no feed takes the small windows");
    }
    check(ds4::small_prompt(2, 1, 32, false), "2 tokens, N 32: small");
    check(ds4::small_prompt(19, 1, 32, false), "19 tokens (the 2026-10-27 prompt), N 32: small");
    check(ds4::small_prompt(32, 1, 32, false), "N tokens: small");
    check(!ds4::small_prompt(33, 1, 32, false), "N + 1 tokens: today's path");
    check(!ds4::small_prompt(1, 1, 32, false), "1 token: today's path (a window of 1 already)");
    check(!ds4::small_prompt(0, 0, 32, false), "no tokens: today's path");
    check(ds4::small_prompt(20, 0, 32, false), "no logits (a prompt's earlier feed): small");
    check(!ds4::small_prompt(20, 2, 32, false), "every row's logits (2 and more): the chunked path's");
    check(!ds4::small_prompt(20, 20, 32, false), "every row's logits (all 20): the chunked path's");
    check(!ds4::small_prompt(20, 1, 32, true), "a remote stage or its worker: today's path");
    check(ds4::small_prompt(2, 1, 2, false) && !ds4::small_prompt(3, 1, 2, false), "N 2: only 2 tokens");
    {
        bool ok = true;   // on: exactly 2 <= n <= N, logits_rows <= 1, not remote
        for (int sn : {2, 8, 31, 300})
            for (int n = 0; n <= 400; ++n)
                for (int lr = 0; lr <= 3; ++lr)
                    for (int rw = 0; rw < 2; ++rw)
                        ok = ok && ds4::small_prompt(n, lr, sn, rw != 0) == (n >= 2 && n <= sn && lr <= 1 && rw == 0);
        check(ok, "on: exactly 2 <= n <= N, at most 1 row of logits, no remote stage");
    }
    // DS4_SMALL_WIN (kWin 16)
    check(ds4::small_win(1, 16, 8) == 1, "1 (the default), DS4_ASYNC_T 8: 1");
    check(ds4::small_win(8, 16, 8) == 8, "8, DS4_ASYNC_T 8: 8");
    check(ds4::small_win(9, 16, 8) == 8, "9, DS4_ASYNC_T 8: clamped to 8 (past it the windows are synchronous)");
    check(ds4::small_win(16, 16, 32) == 16, "16, DS4_ASYNC_T 32: 16");
    check(ds4::small_win(20, 16, 32) == 16, "20, DS4_ASYNC_T 32: clamped to kWin");
    check(ds4::small_win(0, 16, 8) == 1 && ds4::small_win(-3, 16, 8) == 1, "0 and below: clamped to 1");
    check(ds4::small_win(4, 16, 0) == 1 && ds4::small_win(4, 16, -1) == 1, "DS4_ASYNC_T 0 or below: 1");
    std::printf(fails == 0 ? "ALL OK\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
