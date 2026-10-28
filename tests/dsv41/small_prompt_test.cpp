// Strata. DS4_SMALL_PROMPT / DS4_SMALL_WIN (ds4_small_prompt.hpp): which feeds take the small windows (2-N tokens,
// one row of logits at most, no remote stage; 0 off: none, whatever the feed) and DS4_SMALL_WIN's rows held to
// 1..min(kWin, DS4_ASYNC_T); the env texts' parsing; and the prompt's settle (rx::PromptSettle) over the feeds a small
// prompt makes - one settle a prompt, whatever its windows are.  No GPU, no model.
#include "../../src/dsv41/ds4_rx_fill.hpp"
#include "../../src/dsv41/ds4_small_prompt.hpp"

#include <climits>
#include <cstdio>
#include <string>

namespace ds4 = strata::ds4;
namespace rx = strata::ds4::rx;

// the engine's calls on rx::PromptSettle, in the order Model makes them: forward()'s window_end after every window
// (feed_small's: ep true, win_prompt true; feed_windows': ep false, win_prompt true; the decode's: ep true, win_prompt
// false), a window feed's end (after_feed, then cache_settle's settled()), a chunked run's settle, and a public feed
// (Model::feed / feed_mark) that asked for logits ending the prompt (next())
struct Engine {
    rx::PromptSettle ps;
    bool rx_on = true;
    int settles = 0;
    void settle() {
        ps.settled();
        ++settles;
    }
    void small(int n, int rows, int win = 1) {   // feed_small
        for (int a = 0; a < n; a += win) ps.window_end(true, true);
        if (ps.after_feed(rx_on, rows)) settle();
    }
    void windows(int n, int rows) {   // feed_windows
        for (int a = 0; a < n; a += 16) ps.window_end(false, true);
        if (ps.after_feed(rx_on, rows)) settle();
    }
    void chunked() { settle(); }
    void decode() { ps.window_end(true, false); }
    void public_end(int rows) {
        if (rows > 0) ps.next();
    }
};

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
    // the env texts: DS4_SMALL_PROMPT (cap 2048, pick_path's kPickMax)
    {
        std::string note;
        auto n = [&](const char* s) { return ds4::small_prompt_n(s, 2048, note); };
        check(n(nullptr) == 0 && note.empty(), "DS4_SMALL_PROMPT unset: off, no word");
        check(n("") == 0 && note.empty(), "DS4_SMALL_PROMPT empty: off, no word");
        check(n("0") == 0 && note.empty(), "DS4_SMALL_PROMPT=0: off, no word");
        check(n("1") == 0 && !note.empty(), "DS4_SMALL_PROMPT=1: off, with a word (no prompt of 2 fits)");
        check(n("2") == 2 && note.empty() && n("32") == 32 && note.empty() && n(" 32") == 32,
              "DS4_SMALL_PROMPT=2, 32, ' 32': as asked, no word");
        check(n("2048") == 2048 && note.empty(), "DS4_SMALL_PROMPT=2048 (kPickMax): as asked");
        check(n("2049") == 2048 && !note.empty(), "DS4_SMALL_PROMPT=2049: clamped to 2048, with a word");
        check(n("100000") == 2048 && note.find("100000") != std::string::npos, "DS4_SMALL_PROMPT=100000: clamped, the word names it");
        check(n("abc") == 0 && !note.empty(), "DS4_SMALL_PROMPT=abc: off, with a word");
        check(n("12abc") == 0 && !note.empty(), "DS4_SMALL_PROMPT=12abc: off, with a word (the whole text is read)");
        check(n("3.5") == 0 && !note.empty(), "DS4_SMALL_PROMPT=3.5: off, with a word");
        check(n("-4") == 0 && !note.empty(), "DS4_SMALL_PROMPT=-4: off, with a word (out of range)");
        check(n("99999999999999999999999") == 0 && !note.empty(), "DS4_SMALL_PROMPT past a long: off, with a word (no overflow)");
        check(n("2147483648") == 0 && !note.empty(), "DS4_SMALL_PROMPT=2^31: off, with a word (past an int)");
        std::string a, b;
        ds4::small_prompt_n("abc", 2048, a);
        ds4::small_prompt_n("-4", 2048, b);
        check(a != b, "the word says why: not a number, out of range");
    }
    // DS4_SMALL_WIN's text
    {
        std::string note;
        auto w = [&](const char* s) { return ds4::small_win_ask(s, note); };
        check(w(nullptr) == 1 && note.empty() && w("") == 1 && note.empty(), "DS4_SMALL_WIN unset or empty: 1, no word");
        check(w("4") == 4 && note.empty() && w("0") == 0 && note.empty() && w("-3") == -3 && note.empty(),
              "DS4_SMALL_WIN=4, 0, -3: as asked (small_win holds them)");
        check(w("x") == 1 && !note.empty(), "DS4_SMALL_WIN=x: 1, with a word");
        check(w("4x") == 1 && !note.empty(), "DS4_SMALL_WIN=4x: 1, with a word");
        check(w("99999999999999999999999") == 1 && !note.empty(), "DS4_SMALL_WIN past a long: 1, with a word");
        check(w("-2147483649") == 1 && !note.empty(), "DS4_SMALL_WIN past an int: 1, with a word");
    }
    // the prompt's settle: once a prompt, over the feeds of a small prompt (Engine's calls, in Model's order)
    {
        Engine e;   // a small prompt whole (Model::feed: 19 tokens, windows of 1, the last with logits)
        e.small(19, 1);
        check(e.settles == 1 && e.ps.done, "a small prompt whole: one settle, at its end (done until the feed ends)");
        e.public_end(1);
        check(!e.ps.done, "... and its public feed's end: the next prompt settles again");
    }
    {
        Engine e;   // feed_mark's shape: n - 1 tokens without logits, the checkpoint, the last token with
        e.small(18, 0);
        check(e.settles == 0 && !e.ps.done, "feed_mark's n - 1 tokens, no logits: no settle yet, none reset either");
        e.small(1, 1);
        e.public_end(1);
        check(e.settles == 1, "... then its last token: one settle");
    }
    {
        Engine e;   // a chunked part (settled), then a small tail (feed_mark's two feeds)
        e.chunked();
        e.small(11, 0);
        e.small(1, 1);
        e.public_end(1);
        check(e.settles == 1, "a chunked part that settled and a small tail: one settle (the chunked part's), not two");
    }
    {
        Engine e;   // the same with windows of 4 and a tail longer than one window
        e.chunked();
        e.small(11, 0, 4);
        e.small(1, 1, 4);
        e.public_end(1);
        check(e.settles == 1, "... windows of 4: still one");
    }
    {
        Engine e;   // a decode window in between: the next prompt settles again
        e.small(19, 1);
        e.public_end(1);
        e.decode();
        e.decode();
        check(!e.ps.done, "after a decode window: the next prompt settles again");
        e.small(5, 1);
        e.public_end(1);
        check(e.settles == 2, "a small prompt, a decode, a small prompt: two settles");
    }
    {
        Engine e;   // a small prompt with the decode's windows before it (the conversation's turns)
        e.small(7, 1);
        e.public_end(1);
        for (int i = 0; i < 3; ++i) e.decode();
        e.chunked();
        e.small(3, 1);
        e.public_end(1);
        check(e.settles == 2, "prompt, decode, chunked + small tail: two settles in all (one a prompt)");
    }
    {
        Engine e;   // feed_windows' windows (ep false) and a chunked run's: as before
        e.windows(40, 1);
        e.public_end(1);
        check(e.settles == 1, "windows of 16 (ep false): one settle, as before");
        e.chunked();
        e.decode();
        check(!e.ps.done, "a decode window after a chunked settle: done again");
    }
    {
        rx::PromptSettle ps;   // window_end by its arguments
        ps.settled();
        ps.window_end(false, false);
        check(ps.done, "window_end(ep false, win_prompt false): kept");
        ps.window_end(false, true);
        check(ps.done, "window_end(ep false, win_prompt true): kept");
        ps.window_end(true, true);
        check(ps.done, "window_end(ep true, win_prompt true): kept (a small prompt's window)");
        ps.window_end(true, false);
        check(!ps.done, "window_end(ep true, win_prompt false): reset (a decode window)");
    }
    std::printf(fails == 0 ? "ALL OK\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
