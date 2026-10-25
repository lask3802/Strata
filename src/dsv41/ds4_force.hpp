// src/dsv41/ds4_force.hpp - DS4_FORCE_TOKENS=<file>: the CLI's decode (--ids ... --n N) emits a file's ids instead of
// its own picks, the prompt as given, so that the arms of a speed A/B whose bf16 sums run in another order (the remote
// experts, expert parallelism) and so take token paths of their own route the same experts as the baseline.  The file:
// ids as the decode prints them (1,2,3, - commas, spaces or new lines between); fewer than N: refused.  Each own pick
// that differed is counted, the first one's position kept.  Plain decode only (no drafts: their verification picks
// the tokens).
#pragma once

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::ds4 {

struct ForcedTokens {
    std::vector<int> ids;
    long long differed = 0, first = -1;   // (own picks not the file's; the first one's position)
    bool on() const { return !ids.empty(); }
    /// path's ids, n of them at least
    bool load(const std::string& path, int n, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            err = "DS4_FORCE_TOKENS: cannot read " + path;
            return false;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string s = ss.str();
        std::vector<int> v;
        const char* p = s.c_str();
        for (;;) {
            while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
            if (*p == 0) break;
            char* e = nullptr;
            const long x = std::strtol(p, &e, 10);
            if (e == p || x < 0) {
                err = "DS4_FORCE_TOKENS: " + path + " has something not an id at byte " + std::to_string(p - s.c_str());
                return false;
            }
            v.push_back((int) x);
            p = e;
        }
        if ((long long) v.size() < (long long) n) {
            err = "DS4_FORCE_TOKENS: " + path + " has " + std::to_string(v.size()) + " ids, fewer than --n " +
                  std::to_string(n);
            return false;
        }
        ids = std::move(v);
        return true;
    }
    /// the token at decode position pos: the file's (own counted when it is not that)
    int take(size_t pos, int own) {
        if (pos >= ids.size()) return own;
        if (own != ids[pos]) {
            if (differed++ == 0) first = (long long) pos;
        }
        return ids[pos];
    }
};

}  // namespace strata::ds4
