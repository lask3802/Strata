// src/dsv41/ds4_small_prompt.hpp - DS4_SMALL_PROMPT=N: a prompt of 2-N tokens through windows of DS4_SMALL_WIN rows
// (default 1), the decode's own size - the async window the expert server serves, its fills, and at 1 row DS4_EP_DEFER's
// deferral - instead of the flow (every touched expert streamed to the GPU whole: 19 tokens 1954 ms, 1934 experts, on
// 2026-10-27) or windows of 16 (past DS4_ASYNC_T, so synchronous: ~2.6 s each).  Before pick_path and before the
// n < chunk_min windows of 16, in Model::feed and Model::feed_mark; pick_path learns nothing from it.  0 (the default):
// off.  Pure functions, no state (tests/dsv41/small_prompt_test.cpp).
#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::ds4 {

/// a feed of n tokens, logits_rows rows of logits (more than 1: every row's, the chunked path's), takes the small
/// windows?  small_n: DS4_SMALL_PROMPT (0 off); remote_or_worker: a remote stage's main process or its worker (their
/// prompts cross the link, which has paths of its own)
inline bool small_prompt(int n, int logits_rows, int small_n, bool remote_or_worker) {
    return n >= 2 && n <= small_n && logits_rows <= 1 && !remote_or_worker;
}

/// DS4_SMALL_WIN's rows: t held to 1..min(max_rows, async_max_t) (max_rows: kWin; async_max_t: DS4_ASYNC_T, the
/// decode's async windows - below 1 the decode is not async, and the windows are 1 row)
inline int small_win(int t, int max_rows, int async_max_t) {
    return std::max(1, std::min(t, std::min(max_rows, async_max_t)));
}

/// a whole number from an environment text: strtol, the whole text read (blanks after it allowed), in [lo, hi]
enum class EnvInt { kUnset, kOk, kNotNumber, kRange };
inline EnvInt env_int(const char* s, long lo, long hi, long& v) {
    if (s == nullptr || *s == '\0') return EnvInt::kUnset;
    errno = 0;
    char* end = nullptr;
    const long x = std::strtol(s, &end, 10);
    if (end == s) return EnvInt::kNotNumber;
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return EnvInt::kNotNumber;
    if (errno == ERANGE || x < lo || x > hi) return EnvInt::kRange;
    v = x;
    return EnvInt::kOk;
}

/// DS4_SMALL_PROMPT's N from its text: 0 off.  Unset, empty and 0 are off without a word; 1 (no prompt of 2 or more
/// tokens fits), a text that is not a whole number and one out of 0..INT_MAX are off with `note` saying why; past cap
/// (pick_path's kPickMax: from there on it always takes the flow) N is held to cap, with a note.  note: a line for
/// stderr ("" when there is nothing to say)
inline int small_prompt_n(const char* s, int cap, std::string& note) {
    note.clear();
    long v = 0;
    char b[256];
    switch (env_int(s, 0, INT_MAX, v)) {
        case EnvInt::kUnset: return 0;
        case EnvInt::kNotNumber:
            std::snprintf(b, sizeof b, "DS4_SMALL_PROMPT=%.64s: off - not a whole number (N is the most tokens, 2 or more)", s);
            note = b;
            return 0;
        case EnvInt::kRange:
            std::snprintf(b, sizeof b, "DS4_SMALL_PROMPT=%.64s: off - out of range (0 to %d)", s, INT_MAX);
            note = b;
            return 0;
        case EnvInt::kOk: break;
    }
    if (v == 0) return 0;
    if (v == 1) {
        note = "DS4_SMALL_PROMPT=1: off - no prompt of 2 or more tokens fits (N is the most tokens: 2 or more)";
        return 0;
    }
    if (v > cap) {
        std::snprintf(b, sizeof b, "DS4_SMALL_PROMPT=%ld: clamped to %d (past it a prompt takes the flow: pick_path's "
                                   "kPickMax)", v, cap);
        note = b;
        return cap;
    }
    return (int) v;
}

/// DS4_SMALL_WIN's rows as asked (default 1; small_win holds them): unset and empty are 1 without a word, a text that is
/// not a whole number or one past int is 1 with `note` saying so
inline int small_win_ask(const char* s, std::string& note) {
    note.clear();
    long v = 1;
    switch (env_int(s, INT_MIN, INT_MAX, v)) {
        case EnvInt::kUnset: return 1;
        case EnvInt::kOk: return (int) v;
        case EnvInt::kNotNumber:
        case EnvInt::kRange: break;
    }
    char b[200];
    std::snprintf(b, sizeof b, "DS4_SMALL_WIN=%.64s: not a whole number (or past an int): windows of 1", s);
    note = b;
    return 1;
}

}  // namespace strata::ds4
