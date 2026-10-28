// src/dsv41/ds4_small_prompt.hpp - DS4_SMALL_PROMPT=N: a prompt of 2-N tokens through windows of DS4_SMALL_WIN rows
// (default 1), the decode's own size - the async window the expert server serves, its fills, and at 1 row DS4_EP_DEFER's
// deferral - instead of the flow (every touched expert streamed to the GPU whole: 19 tokens 1954 ms, 1934 experts, on
// 2026-10-27) or windows of 16 (past DS4_ASYNC_T, so synchronous: ~2.6 s each).  Before pick_path and before the
// n < chunk_min windows of 16, in Model::feed and Model::feed_mark; pick_path learns nothing from it.  0 (the default):
// off.  Pure functions, no state (tests/dsv41/small_prompt_test.cpp).
#pragma once

#include <algorithm>

namespace strata::ds4 {

/// a feed of n tokens, logits_rows rows of logits (more than 1: every row's, the chunked path's), takes the small
/// windows?  small_n: DS4_SMALL_PROMPT (0 off); remote_or_worker: a remote stage's main process or its worker (their
/// prompts cross the link, which has paths of its own)
inline bool small_prompt(int n, int logits_rows, int small_n, bool remote_or_worker) {
    return small_n >= 2 && n >= 2 && n <= small_n && logits_rows <= 1 && !remote_or_worker;
}

/// DS4_SMALL_WIN's rows: t held to 1..min(max_rows, async_max_t) (max_rows: kWin; async_max_t: DS4_ASYNC_T, the
/// decode's async windows - below 1 the decode is not async, and the windows are 1 row)
inline int small_win(int t, int max_rows, int async_max_t) {
    return std::max(1, std::min(t, std::min(max_rows, async_max_t)));
}

}  // namespace strata::ds4
