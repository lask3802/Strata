// Strata. DS4_FORCE_TOKENS (ds4_force.hpp): the CLI's decode emits a file's ids instead of its own picks, for speed
// A/Bs on one token path (lead 2026-10-25: the remote-experts arms took paths of their own, so their later tokens
// routed experts the base arms never read).  The file's ids come out exactly, in order; its own picks that differ are
// counted; a file shorter than --n or with something not an id is refused.  No GPU, no model.
#include "../../src/dsv41/ds4_force.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ds4 = strata::ds4;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}
static std::string put(const char* name, const std::string& text) {
    const std::string p = (std::filesystem::temp_directory_path() / name).string();
    std::ofstream(p, std::ios::binary) << text;
    return p;
}

int main() {
    std::string err;
    {
        ds4::ForcedTokens f;
        const std::string p = put("ds4_force_a.ids", "343,5582,1,\n");   // (as the decode prints them)
        check(f.load(p, 3, err) && f.ids == std::vector<int>{343, 5582, 1}, "the decode's own output: 3 ids");
        check(f.on(), "on");
        // a decode of 3: its own picks 343, 7, 1 - the file's come out, the one that differed counted
        std::vector<int> out;
        const int own[3] = {343, 7, 1};
        for (size_t i = 0; i < 3; ++i) out.push_back(f.take(i, own[i]));
        check(out == f.ids, "the file's ids come out exactly");
        check(f.differed == 1 && f.first == 1, "one own pick differed, at position 1");
        check(f.take(3, 99) == 99, "past the file: the own pick");
        check(f.differed == 1, "and not counted");
    }
    {
        ds4::ForcedTokens f;
        const std::string p = put("ds4_force_b.ids", "1, 2\r\n3\t4");
        check(f.load(p, 4, err) && f.ids == std::vector<int>{1, 2, 3, 4}, "spaces, CR LF, tabs between");
        ds4::ForcedTokens g;
        check(!g.load(p, 5, err) && !g.on(), "fewer than --n: refused");
        check(err.find("fewer than --n 5") != std::string::npos, "and says so");
        ds4::ForcedTokens h;
        check(!h.load(put("ds4_force_c.ids", "1,2,x,3"), 1, err), "something not an id: refused");
        check(!h.load(put("ds4_force_d.ids", "1,-2"), 1, err), "a negative id: refused");
        check(!h.load((std::filesystem::temp_directory_path() / "ds4_force_none.ids").string(), 1, err),
              "no file: refused");
        ds4::ForcedTokens e;
        check(!e.load(put("ds4_force_e.ids", ""), 1, err), "an empty file for --n 1: refused");
    }
    {
        ds4::ForcedTokens f;   // every own pick the file's: none counted
        check(f.load(put("ds4_force_f.ids", "5,6"), 2, err), "two ids");
        f.take(0, 5);
        f.take(1, 6);
        check(f.differed == 0 && f.first == -1, "the same picks: none differed");
    }
    std::printf(fails == 0 ? "ALL OK\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
