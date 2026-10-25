// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_main.cpp - maya-ds4: DeepSeek-V4.1-Flash on Maya.
//
//   maya-ds4 --pack <model>/maya-ds4 --ids 0,128803,... --n 16          greedy tokens after the ids
//   maya-ds4 --pack ... --ids ... --dump-logits out.f32                every position's logits (the oracle's check)
//   maya-ds4 --pack ... --serve                                       Maya's engine protocol on stdin / stdout
//   maya-ds4 --pack ... --stage-begin K --stage-end E ...             only layers [K, E): the embedding into K, the head
//                                                                     after E - 1 (a split's memory plan, by itself)
//   maya-ds4 --pack ... --stage-worker PORT --stage-begin K           a remote stage's worker: layers K to the last for
//                                                                     the main process below, on PORT (no head)
//   maya-ds4 --pack ... --remote-stage HOST:PORT --stage-end K ...    layers 0 to K - 1 and the head here, the rest on
//                                                                     the worker (docs/REMOTE_STAGE.md; any mode above)
//   maya-ds4 --pack ... --remote-experts [BIND:]PORT ...              (a proof of concept, ds4_rx.hpp) some of the CPU
//                                                                     lane's experts on an expert server's GPU
//   maya-ds4 --expert-server-connect HOST:PORT [--expert-server-reserve-mib N] [--expert-server-once] [--gpus I]
//            [--expert-server-cpu]                                     (tests: the CPU lane's arithmetic, no GPU)
//                                                                     that server: no pack, it dials the main process
//
// --serve speaks what serve/server.py's StrataEngine expects: INFO lines, then `READY <context> stop`; per request
// `GEN <max_new> [key=value ...] <id,id,...>` answered by `T <id>` lines and `DONE <generated> <prompt> <prompt ms>
// <decode ms> <stop|length|cancel> 0 0 <reused> 0 0`; `STOP` ends a request early, `QUIT` the engine.  A prompt that
// continues the state (the conversation so far) reuses it; one that only shares the last prompt's start goes back to
// that prompt's end; anything else starts over.  `GENI <max_new> [keys] <file> <ids>` carries the prompt's pictures
// (the file: see read_pics); the state is shared only where the same picture sits at the same place.  With
// STRATA_DS4_SLOT_KEEP=1 the conversations are kept when another one takes the model (a slot each: the state, the
// compressed KV included), and across runs in STRATA_DS4_SLOT_DIR - so a request that does not continue the state but
// does continue a kept conversation reads only its new tokens.  With --remote-stage the same switch keeps the checkpoints
// and the slots across the link: the worker takes its layers' part of each under the main process's id (CkptSave /
// CkptRestore, ds4_stage.hpp), a slot or checkpoint is used only where both sides hold it, and one the worker does not
// hold (restarted, dropped for its own bounds STRATA_DS4_SLOT_CAP / _GB / _DIR, set at the worker too) is read from token 0;
// the main process's folder is then <STRATA_DS4_SLOT_DIR or <pack>/slots>/split-K and holds nothing across runs.
// `pin=N` on a GEN marks the first N prompt tokens as a start other requests share (a system prompt and its tools): the
// read stops there for a checkpoint, so a request behind the same start - another agent - goes back to it instead of
// reading it again (also without slots).  STRATA_DS4_REUSE=0 turns every reuse off (each request reads its whole prompt).
#include "ds4_force.hpp"
#include "ds4_model.hpp"
#include "ds4_os.hpp"
#include "ds4_packinfo.hpp"
#include "ds4_partstore.hpp"
#include "ds4_rx.hpp"
#include "ds4_slotmeta.hpp"
#include "ds4_stage.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the bytes this process had the storage read for it (Linux's /proc/self/io; -1 where there is none): the Monitor's
// disk reads a token, the experts read in place from the model's files included
long long io_read_bytes() {
#ifdef __linux__
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (f == nullptr) return -1;
    char line[128];
    long long v = -1;
    while (std::fgets(line, sizeof line, f) != nullptr)
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) break;
    std::fclose(f);
    return v;
#else
    return -1;
#endif
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}

void usage() {
    std::fprintf(stderr,
                 "maya-ds4 --pack <model>/maya-ds4 [--gpus 0,1,...] [--max-context N] [--reserve-mib N]\n"
                 "           [--vram-experts F] [--cpu-threads N] [--expert-profile FILE|none] [--ssd-cache GB] [--ssd-policy lru|lfu|lfulru] [--no-expert-cache] [--verbose]\n"
                 "           [--tp N] [--threads on|off] [--prefill-chunk N] [--dspark on|off|none|auto] [--dspark-block N] [--dspark-conf F]\n"
                 "           [--vision on|off] [--mmap-experts on|off|auto] [--stage-begin K] [--stage-end E]\n"
                 "           [--remote-stage HOST:PORT --stage-end K] [--remote-experts [BIND:]PORT]\n"
                 "           (--ids I,J,... [--images FILE] [--n N] [--dump-logits FILE] | --serve\n"
                 "           | --stage-worker PORT [--stage-bind ADDR] --stage-begin K\n"
                 "           | --encode-image FILE.rgb --size WxH --dump-image OUT\n"
                 "           | --pack-info)\n"
                 "maya-ds4 --expert-server-connect HOST:PORT [--expert-server-reserve-mib N] [--expert-server-once] [--gpus I]\n"
                 "           [--expert-server-cpu (tests: no GPU)]\n");
}

// stdin lines on a thread, so a request can see STOP while it decodes
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    void start() {
        std::thread([this] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                std::lock_guard<std::mutex> lk(mu);
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    stop.store(true);
                    continue;
                }
                if (l == "QUIT") stop.store(true);
                q.push_back(l);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            stop.store(true);
            cv.notify_all();
        }).detach();
    }
    bool next(std::string& l) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
};

// The drafts between two stats: each draft position (1st, 2nd, ..) - how often it was verified with every draft
// before it kept, and how often it was kept then
std::string per_position(const strata::ds4::Stats& s0, const strata::ds4::Stats& s1) {
    std::string pp = "per position (kept / reached):";
    for (int i = 0; i < 8; ++i) {
        const long long r = s1.draft_reached[i] - s0.draft_reached[i], k = s1.draft_kept[i] - s0.draft_kept[i];
        if (r <= 0) break;
        char b[64];
        std::snprintf(b, sizeof b, " %d: %lld/%lld %.0f%%", i + 1, k, r, 100.0 * (double) k / (double) r);
        pp += b;
    }
    return pp;
}

static std::FILE* g_round_log = nullptr;
static std::string g_round_tag;
static std::vector<float> g_round_conf;
static int g_round_nd = 0;

// The drafts' cut on the product of their confidences (DeepInfer's cumulative rule): a draft is verified while the chance
// that it and every draft before it are kept stays at least this - a long window whose later drafts are likely lost is
// not verified whole.  STRATA_DS4_DSPARK_CUM=F, a request's dspark_cum key; 0: off, only the floor on each draft's own
// confidence (dspark_conf).  Off by default (not measured faster on code: H100 / owner PC); Project Maya runs 0.5 (a Radeon 8065S: decode +6.5%, the same tokens)
float dspark_cum_default() {
    static const float v = [] {
        const char* e = std::getenv("STRATA_DS4_DSPARK_CUM");
        return e != nullptr ? (float) std::atof(e) : 0.0f;
    }();
    return v;
}

// One round of decoding after `tok` (the token at pos(), not fed yet): with the drafter, its drafts while their
// confidence holds (at most `room` - 1 of them), verified together; else a plain step.  `out` gets the tokens that follow
// tok (the accepted drafts, then the trunk's next), the last of them the next round's tok.
bool round(strata::ds4::Model& m, int tok, int room, bool spec, float conf_min, float temperature, float top_p, int top_k,
           float min_p, unsigned long long seed, unsigned long long& counter, std::vector<int>& out, int& offered,
           int& accepted, std::string& err, float cum_min = -1.0f) {
    out.clear();
    offered = accepted = 0;
    if (spec && room > 1 && m.draft_block() > 0) {
        std::vector<int> drafts;
        std::vector<float> conf;
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
        g_round_conf = conf;
        g_round_nd = (int) drafts.size();
        int keep = 0;
        if (cum_min < 0) cum_min = dspark_cum_default();
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < room - 1 && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        if (keep > 0) {
            drafts.resize((size_t) keep);
            int acc = 0, next = -1;
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            if (g_round_log != nullptr) {   // DS4_ROUND_LOG: tag, drafts offered, accepted, the confidences
                std::fprintf(g_round_log, "%s\t%d\t%d", g_round_tag.c_str(), keep, acc);
                for (int i = 0; i < keep; ++i) std::fprintf(g_round_log, "\t%.4f", g_round_conf[(size_t) i]);
                std::fprintf(g_round_log, "\n");
            }
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = keep;
            accepted = acc;
            return true;
        }
    }
    if (!m.step(tok, true, err)) return false;
    out.push_back(m.sample(temperature, top_p, top_k, min_p, seed, counter++));
    return true;
}

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    long long pin = 0;      // pin=N: the first N prompt tokens are a start other requests share (a system prompt and its
                            // tools): the state at N is kept as a checkpoint, so another agent behind the same start
                            // goes back to it instead of reading it again
    float cum = -1.0f;      // dspark_cum=F: the cut on the product of the confidences (-1: the engine's, dspark_cum_default)
    std::string file;   // GENI: the pictures
    std::vector<int> ids;
};

// A prompt's picture: where its span starts, its pixels (the server's resize: RGB rows, both sides multiples of the
// patch size), a hash of them; once encoded, its span's rows
struct Pic {
    int64_t start = 0;
    int width = 0, height = 0;
    uint64_t hash = 0;
    size_t len = 0;   // the span (0: not known yet)
    std::vector<uint8_t> rgb;
};

// GENI's file: int32 'DSI1', int32 count, then per picture int64 start, int32 width, int32 height and its
// height x width x 3 RGB bytes, in prompt order
bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open the pictures' file " + path;
        return false;
    }
    char magic[4] = {};
    int32_t n = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DSI1", 4) == 0 && std::fread(&n, 4, 1, f) == 1 &&
              n >= 0 && n <= 4096;
    for (int i = 0; ok && i < n; ++i) {
        Pic p;
        int32_t wh[2] = {};
        ok = std::fread(&p.start, 8, 1, f) == 1 && std::fread(wh, 4, 2, f) == 2 && wh[0] > 0 && wh[1] > 0 &&
             (int64_t) wh[0] * wh[1] <= (64 << 20);
        if (!ok) break;
        p.width = wh[0], p.height = wh[1];
        p.rgb.resize((size_t) p.width * p.height * 3);
        ok = std::fread(p.rgb.data(), 1, p.rgb.size(), f) == p.rgb.size();
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the size and the pixels
        for (int32_t x : wh)
            for (int b = 0; b < 4; ++b) h = (h ^ (uint8_t) (x >> (8 * b))) * 1099511628211ull;
        for (uint8_t b : p.rgb) h = (h ^ b) * 1099511628211ull;
        p.hash = h;
        v.push_back(std::move(p));
    }
    std::fclose(f);
    if (!ok) err = path + ": not a pictures' file (DSI1)";
    return ok;
}

// the encoded spans, the latest few (a conversation sends its pictures with every turn)
struct SpanCache {
    std::deque<std::pair<uint64_t, std::shared_ptr<const std::vector<uint16_t>>>> q;
    std::shared_ptr<const std::vector<uint16_t>> get(strata::ds4::Model& m, const Pic& p, std::string& err) {
        for (size_t i = 0; i < q.size(); ++i)
            if (q[i].first == p.hash) {
                auto r = q[i].second;
                q.erase(q.begin() + (long) i);
                q.emplace_front(p.hash, r);
                return r;
            }
        auto rows = std::make_shared<std::vector<uint16_t>>();
        if (!m.encode_image(p.rgb.data(), p.width, p.height, *rows, err)) return nullptr;
        q.emplace_front(p.hash, rows);
        if (q.size() > 8) q.pop_back();
        return rows;
    }
};

bool parse_gen(const std::string& line, Req& r, bool geni) {
    const char* p = line.c_str() + (geni ? 5 : 4);
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    if (e == p) return false;
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p && *p != ' ') ++p;
        if (p == s) break;
        const std::string tok(s, (size_t) (p - s));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos && geni && r.file.empty()) {
            r.file = tok;
            continue;
        }
        if (eq == std::string::npos) {
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf") r.conf = std::strtof(v.c_str(), nullptr);
        else if (k == "pin") r.pin = std::atoll(v.c_str());
        else if (k == "dspark_cum" || k == "spec_cum") r.cum = std::strtof(v.c_str(), nullptr);
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

// ---- the remote stage (docs/REMOTE_STAGE.md, ds4_stage.hpp): this PC runs layers 0 to K - 1 and the head, a worker
// PC the rest - each window's rows, and each prompt chunk's, go there after layer K - 1 and come back before the head
// This side's hello: the trunk, the layers where the link cuts it, the windows, the prompt chunks (the most rows this
// side sends, or takes: its chunked path's chunk; 0 windows only), the compact bf16 rows (kRowsBf16Cap and their
// narrower hand-off row; STRATA_REMOTE_ROWS=fp32: fp32 rows), no K/V type to compare (the hand-off carries no attention
// K/V), the reply's layout (kRowsNormed), the checkpoints the main process has the worker keep (slots: ckpt,
// STRATA_DS4_SLOT_KEEP=1), the pack's fingerprint
strata::net::StageHello stage_hello(const strata::ds4::Model& m, const std::string& pack, bool worker, bool ckpt = false) {
    const strata::ds4::Model::StageGeo sg = m.stage_geo();
    strata::net::StageHello me;
    me.n_embd = sg.dim;
    me.hc = sg.hc;
    me.n_layers = sg.n_layers;
    me.n_expert = sg.n_expert;
    me.layer_begin = worker ? sg.layer_begin : sg.layer_end;
    me.max_t = sg.max_t;
    me.kv_type = -1;
    // (the link's format is fp32 either way; the compact rows are the engine's own and say so in a capability bit)
    me.row_formats = strata::ds4::stage::hello_rows(sg.rows_bf16);
    me.max_context = m.max_context();
    me.chunk = sg.chunk;
    me.handoff_floats = sg.handoff_floats;
    me.ckpt_max = !worker && ckpt ? strata::ds4::stage::kMainSlotsMax + 1 : 0;
    me.pack_hash = strata::ds4::stage::pack_hash(pack);
    std::snprintf(me.build, sizeof me.build, "strata-dsv41 remote-stage %s", worker ? "worker" : "main");
    return me;
}

// STRATA_REMOTE_TIMING=1 (read as the Qwen engine reads it): the per-stage timing lines of docs/REMOTE_STAGE.md - the
// link's own ("strata remote: 64 windows: mean ... = worker ... + link ...", from stage_link.cpp), the main process's
// time between windows and the worker's own layers' time, every 64 windows, in the same words the tuner reads
bool remote_timing() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_REMOTE_TIMING");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

// STRATA_STAGE_TOKEN (the shared secret, as the Qwen engine reads it); false: too long
bool stage_token(std::string& tok) {
    const char* v = std::getenv("STRATA_STAGE_TOKEN");
    tok = v != nullptr ? v : "";
    if (tok.size() < sizeof(strata::net::StageHello::token)) return true;
    std::fprintf(stderr, "maya-ds4: STRATA_STAGE_TOKEN is longer than %zu characters\n",
                 sizeof(strata::net::StageHello::token) - 1);
    return false;
}

// The main process's end: one link, made at the start and again when a reset finds it lost (the worker or its PC
// started again, the network went away while idle), with the same hello.  A worker that does not listen yet, a
// network that does not reach it or a name that does not resolve yet is tried every 2 s until STRATA_REMOTE_CONNECT_S
// (180 s; 0: one try) have passed; a refusal (token, model, layers, context, chunks) does not change by waiting.  While
// a request waits (serve), `INFO remote_stage_wait=S` every 10 s tells the server the engine is not frozen, and a STOP
// ends the wait.  Every new connection starts with a Reset: the worker's session from position 0.  Prompt chunks
// (me.chunk > 0) go through `chunks`, one send in flight on its own thread.
struct RemoteMain {
    std::string at;
    strata::net::StageClient link;
    strata::net::StageHello me, peer;
    const std::atomic<bool>* stop = nullptr;   // (serve: its STOP flag - a request's wait ends on it)
    const strata::ds4::Model* model = nullptr;   // (the timing lines: the compact rows' conversions)
    // (remote_timing) this process's share of a window: from one window's rows coming back to the next one's send
    double last_end = 0, gap_ms = 0;
    strata::ds4::Model::StageConv conv0;   // (the conversions at the last line)
    int64_t gaps = 0;
    std::unique_ptr<strata::ds4::stage::ChunkLink> chunks;
    // the prompt chunks' rows over this link (stage::chunk_sent: me.chunk, or the worker's when smaller), told to the
    // model at each link made (chunk_to: Model::set_stage_chunk)
    int64_t sent_chunk = 0;
    std::function<void(int64_t)> chunk_to;
    // (slots: STRATA_DS4_SLOT_KEEP=1) the slot ids this side holds, which the worker keeps its parts of and no others;
    // serve keeps it current.  epoch: the links made so far - a worker connected to again (restarted, or taken over
    // by another main process) holds none of them
    std::vector<int64_t> keep;
    uint64_t epoch = 0;

    // a chunk's send still on its way finishes (or fails) before the link is used again or closed; what it failed of
    // was that run's (its model run is gone)
    void chunks_done() {
        std::string ignored;
        if (chunks) (void) chunks->wait(ignored);
    }
    bool connect(std::string& e) {
        chunks_done();
        static const int wait_s = [] {
            const char* v = std::getenv("STRATA_REMOTE_CONNECT_S");
            return v != nullptr && *v != '\0' ? std::max(0, std::atoi(v)) : 180;
        }();
        const double t0 = now_ms();
        auto spent = [&] { return (now_ms() - t0) / 1000.0; };
        std::mutex mu;
        std::condition_variable cv;
        bool done = false;
        std::thread beat;
        if (stop != nullptr)   // (in a request: the heartbeat)
            beat = std::thread([&] {
                std::unique_lock<std::mutex> lk(mu);
                while (!cv.wait_for(lk, std::chrono::seconds(10), [&] { return done; })) {
                    std::printf("INFO remote_stage_wait=%lld\n", (long long) spent());
                    std::fflush(stdout);
                }
            });
        bool said = false, ok = false;
        for (;;) {
            link.close();
            const int left = wait_s > 0 ? std::min(strata::net::io_timeout_s(), std::max(1, (int) std::ceil(wait_s - spent()))) : 0;
            if (link.connect(at, me, peer, e, left, stop)) {
                ++epoch;   // (the worker holds no part of anything from before, whether or not this reset comes through)
                if (!strata::ds4::stage::main_accepts(peer, me, e)) {   // (a refusal: no waiting mends it, and no
                    link.close();                                        // link a later reset could take as up)
                    break;
                }
                sent_chunk = strata::ds4::stage::chunk_sent(me.chunk, peer.chunk);
                if (me.chunk > 0) {
                    if (sent_chunk < me.chunk)
                        std::fprintf(stderr, "maya ds4: remote stage: prompt chunks of %lld rows: the worker takes %lld "
                                             "at most (this process's own: %lld)\n", (long long) sent_chunk,
                                     (long long) peer.chunk, (long long) me.chunk);
                    if (chunk_to) chunk_to(sent_chunk);
                }
                ok = link.reset({}, e);
                break;
            }
            const bool cancelled = strata::net::stage_wait_cancelled(e) || (stop != nullptr && stop->load());
            if (cancelled || !strata::net::stage_link_lost(e) || spent() + 2 > (double) wait_s) break;
            if (!said) {
                std::fprintf(stderr, "maya ds4: remote stage: waiting for the worker at %s (%s), up to %d s "
                                     "(STRATA_REMOTE_CONNECT_S)\n", at.c_str(), e.c_str(), wait_s);
                said = true;
            }
            for (int i = 0; i < 8 && !(stop != nullptr && stop->load()); ++i)   // 2 s, or until a STOP
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (beat.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu);
                done = true;
            }
            cv.notify_all();
            beat.join();
        }
        return ok;
    }
    // the worker's session from position 0 (Model::reset): connecting again when the link is lost
    bool reset(std::string& e) {
        chunks_done();
        if (link.connected() && link.reset(keep, e)) return true;
        if (link.connected() && !strata::net::stage_link_lost(e)) return false;   // (the worker's own error)
        const std::string lost = e.empty() ? "remote stage: the link is down" : e;
        if (!connect(e)) {
            e = lost + "; connecting again: " + e;
            return false;
        }
        std::fprintf(stderr, "maya ds4: remote stage: the link to %s was lost (%s): connected again\n", at.c_str(),
                     lost.c_str());
        return true;
    }
    // the worker's checkpoint of a role, the pin, a slot's part (CkptSave: ids as ds4_stage.hpp has them) at position L;
    // `stored` false: it did not keep it.  A slot's id is in keep[] by now, as the main process holds it
    bool ck_save(int64_t id, int64_t L, bool& stored, std::string& e) {
        std::vector<int64_t> k = keep;
        if (id >= strata::ds4::stage::kSlotId0 && std::find(k.begin(), k.end(), id) == k.end()) k.push_back(id);
        return link.ckpt_save(id, L, k, stored, e);
    }
    // the worker back at a checkpoint (CkptRestore) or with a slot's part; `missing`: it does not hold it, nothing changed
    bool ck_restore(int64_t id, int64_t L, bool& missing, std::string& e) {
        return link.ckpt_restore(id, L, keep, missing, e);
    }
    // a window: its rows there and back (the model commits it: Model::set_stage_link)
    bool run(int T, const int32_t* tokens, int64_t p0, const float* out, float* in, std::string& e) {
        const bool timing = remote_timing();
        if (timing && last_end != 0) {
            const double d = now_ms() - last_end;
            if (d < 1000.0) gap_ms += d;   // (the same request: a gap of a second is the next prompt)
            if (d < 1000.0 && ++gaps % 64 == 0) {
                std::fprintf(stderr, "strata remote: main process between windows %.2f ms each (64 windows: head, "
                                     "layers 0-K)\n", gap_ms / 64);
                if (model != nullptr && model->stage_geo().rows_bf16) {
                    // (each mean over the windows it converted since the last line: a window's pack and its reply's
                    // unpack are not the same count at a line, and a prompt's first window has no gap)
                    const strata::ds4::Model::StageConv c = model->stage_conv();
                    const int64_t no = std::max<int64_t>(1, c.out_n - conv0.out_n), ni = std::max<int64_t>(1, c.in_n - conv0.in_n);
                    std::fprintf(stderr, "strata remote: rows bf16: %lld bytes a row each way; pack %.3f ms (%lld windows) + "
                                         "unpack %.3f ms (%lld windows) a window (part of the time between windows)\n",
                                 (long long) (me.handoff_floats * 4), (c.out_ms - conv0.out_ms) / (double) no, (long long) no,
                                 (c.in_ms - conv0.in_ms) / (double) ni, (long long) ni);
                    conv0 = c;
                }
                gap_ms = 0;
            }
        }
        const size_t n = (size_t) T * (size_t) me.handoff_floats;
        const bool ok = link.run(T, tokens, p0, out, n, in, n, e);
        last_end = timing ? now_ms() : 0;
        return ok;
    }
    // the last window's first n_keep rows kept there (all of a window kept whole; a draft's check: what it accepted)
    bool commit(int n_keep, std::string& e) { return link.commit(n_keep, e); }
};
RemoteMain* g_remote = nullptr;

// The worker's end (--stage-worker PORT): serves one main process at a time on the port instead of stdin.  A
// conversation starts with a Reset; its windows and prompt chunks arrive in order and run here one at a time, each at
// the position this worker is at (a chunk on the chunked prompt path, while the link reads the next one into the
// other of its two buffers).  Each window's Commit keeps all its rows or (a draft's check) the first n_keep - the rest
// undone from the backups a window of 2 or more rows keeps (ds4_stage.hpp's commit_kind); a chunk is kept whole, with
// no Commit and no backups.  Returns when listening fails.
int stage_worker(strata::ds4::Model& m, const std::string& pack, const std::string& bind, int port) {
    std::string token;
    if (!stage_token(token)) return 1;
    const strata::net::StageHello me = stage_hello(m, pack, true);
    const strata::ds4::Model::StageGeo sg = m.stage_geo();
    int last_t = 0;   // (the last window's rows, until its Commit)
    // the worker's parts of the main process's slots (ds4_stage.hpp, ds4_partstore.hpp), bounded as the main process
    // bounds its own: STRATA_DS4_SLOT_CAP (4) parts in RAM, the rest in STRATA_DS4_SLOT_DIR (default <pack>/slots-stage),
    // STRATA_DS4_SLOT_GB (16) of them at most.  CAP and GB both 0: this worker keeps no slots (the main process then
    // reads a conversation that left the model from its first token)
    const auto env_number = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
    strata::ds4::stage::PartStore parts(
        std::getenv("STRATA_DS4_SLOT_DIR") != nullptr ? std::string(std::getenv("STRATA_DS4_SLOT_DIR")) : pack + "/slots-stage",
        (size_t) std::max(0.0, env_number("STRATA_DS4_SLOT_CAP", 4)),
        (uint64_t) (std::max(0.0, env_number("STRATA_DS4_SLOT_GB", 16)) * 1073741824.0));
    strata::net::StageHandlers hs;
    hs.hello = [&](const strata::net::StageHello& theirs, strata::net::StageHello& mine, std::string& e) {
        mine = me;   // (the worker's own hello carries no token)
        if (theirs.protocol != strata::net::kStageProtocol) return true;   // (the link refuses it for its number)
        return strata::ds4::stage::worker_accepts(theirs, me, m.max_context(), e);
    };
    // (remote_timing) every 64 windows: this worker's own time for a window - its rows unpacked, its layers, its rows
    // packed - and the conversions' share of it
    double own_ms = 0;
    strata::ds4::Model::StageConv conv0;   // (the conversions at the last line)
    int64_t windows = 0;
    // (--remote-experts) the expert server's share of this worker's CPU lane, printed when a main process leaves: its
    // windows since it came ("a token" in those lines is a window here)
    strata::ds4::Stats rx0 = m.stats();
    int64_t rx_windows = 0;
    hs.run = [&](int T, const int32_t* tokens, int64_t p0, std::string& e) {
        const double t0 = now_ms();
        if (!strata::ds4::stage::worker_run(m, tokens, T, p0, last_t, e)) return false;
        ++rx_windows;
        if (remote_timing()) {
            own_ms += now_ms() - t0;
            if (++windows % 64 == 0) {
                const strata::ds4::Model::StageConv c = m.stage_conv();
                const int64_t no = std::max<int64_t>(1, c.out_n - conv0.out_n), ni = std::max<int64_t>(1, c.in_n - conv0.in_n);
                std::fprintf(stderr, "strata stage worker: windows: own layers %.2f ms each (64 windows; rows in %.2f, "
                                     "rows out %.2f)\n", own_ms / 64, (c.in_ms - conv0.in_ms) / (double) ni,
                             (c.out_ms - conv0.out_ms) / (double) no);
                own_ms = 0;
                conv0 = c;
            }
        }
        return true;
    };
    hs.commit = [&](int n_keep, std::string& e) { return strata::ds4::stage::worker_commit(m, n_keep, last_t, e); };
    hs.prefill = [&](const int64_t* tokens, int64_t T, int64_t p0, int64_t flags, int64_t skip, const float* in,
                     float* out, std::string& e) {
        return strata::ds4::stage::worker_prefill(m, tokens, (int) T, p0, (flags & strata::ds4::stage::kChunkLast) != 0,
                                                  (int) skip, in, out, last_t, e);
    };
    hs.reset = [&](const std::vector<int64_t>& keep, std::string&) {
        strata::ds4::stage::worker_reset(m, parts, keep);
        m.save_profile();
        last_t = 0;
        return true;
    };
    hs.disconnected = [&] {
        if (rx_windows > 0) {
            const std::string rl = m.rx_lines(rx0, (double) rx_windows);
            if (!rl.empty()) std::fprintf(stderr, "strata stage worker: %lld windows; a window each below:\n%s",
                                          (long long) rx_windows, rl.c_str());
        }
        rx0 = m.stats();
        rx_windows = 0;
        strata::ds4::stage::worker_disconnected(m, parts);   // (a new main process is a new id space)
        m.save_profile();
        last_t = 0;
    };
    // the main process's checkpoints and slots (CkptSave / CkptRestore, ids as ds4_stage.hpp has them).  serve_stage has
    // checked the position of a save and tracks it across a restore; a restore forgets the last window (last_t)
    strata::ds4::stage::worker_ckpt_handlers(m, parts, last_t, hs, [](const std::string& note) {
        std::fprintf(stderr, "maya ds4: remote stage worker: %s\n", note.c_str());
    });
    float* const pf[4] = {m.stage_chunk_rows(0), m.stage_chunk_rows(1), m.stage_chunk_rows(2), m.stage_chunk_rows(3)};
    strata::net::StageBuffers bufs = strata::ds4::stage::worker_buffers(m.stage_rows_in(), m.stage_rows_out(), sg.max_t,
                                                                       pf, sg.chunk, sg.handoff_floats);
    // (the link's framing is fp32: the engine packs its rows itself - the hello's log line says which)
    bufs.engine_rows = sg.rows_bf16 ? "engine-packed bf16" : "engine rows fp32";
    strata::net::StageStats wst;
    std::fprintf(stderr, "maya ds4: remote stage worker: layers %d-%d, no head, windows of %d, prompt chunks of %d at most%s, "
                 "rows %s, port %d\n", sg.layer_begin, sg.layer_end - 1, sg.max_t, sg.chunk,
                 sg.chunk > 0 ? "" : " (none: windows only)",
                 sg.rows_bf16 ? "bf16 (the streams; the picks exact)" : "fp32 (STRATA_REMOTE_ROWS=fp32)", port);
    std::fflush(stderr);
    return strata::net::serve_stage(bind, port, token, hs, bufs, wst, nullptr);
}

int serve(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min, const std::string& pack) {
    std::vector<int> eos = parse_ids(eos_list);
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    std::printf("READY %lld stop\n", (long long) m.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "maya ds4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <pictures> <ids>; STOP, QUIT\n");
    Lines in;
    in.start();
    if (g_remote != nullptr) g_remote->stop = &in.stop;   // (a request's wait for the worker ends on a STOP)
    std::string line;
    unsigned long long counter = 0;
    std::vector<Pic> held;   // the pictures the state holds (start, hash, span)
    SpanCache spans;
    const int img_tok = m.image_token();
    // CONVERSATION SLOTS: a request that does not continue the conversation in the model first sets that one aside
    // (Model::slot_save: its window rings, the compressors' partial groups and every source's compressed KV and index
    // keys), then takes back the kept conversation this prompt does continue (slot_load) - chats, and an agent's side
    // requests, that interleave no longer re-read each other's prompts (70k and 185k tokens alternating: 75-200 s a
    // switch at ~930 tok/s).  STRATA_DS4_SLOT_KEEP=1 turns them on and keeps them across runs, in STRATA_DS4_SLOT_DIR
    // (default <pack>/slots): a slot is a .bin (the state) and a .meta (this model's identity, the conversation's
    // tokens and its size), both written under a temporary name first; a slot whose .meta is missing, foreign or does
    // not match its file, or unused for STRATA_DS4_SLOT_DAYS (7) days, is removed at start.  STRATA_DS4_SLOT_CAP (4) is
    // the conversations kept in RAM; past it the least recently used is written to the folder and read back from there
    // (RAM first, the folder past the cap), and the engine's end writes the ones in RAM to the folder, so the next run
    // reads on from them.  STRATA_DS4_SLOT_GB (16) caps the folder's .bin files: at start, and after every write, the
    // folder's least recently used conversation is removed while they are over it (never the one just written, so a
    // conversation bigger than the cap stays).  STRATA_DS4_SLOT_MIN (1024) is the shortest conversation set aside (a
    // shorter one is read again in about the time its slot would take).  Text only: a conversation that holds pictures
    // is not set aside, and a prompt with pictures takes none back.
    const auto env_num = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
    const bool slot_keep =
        std::getenv("STRATA_DS4_SLOT_KEEP") != nullptr && std::atoi(std::getenv("STRATA_DS4_SLOT_KEEP")) != 0;
    const bool reuse = std::getenv("STRATA_DS4_REUSE") == nullptr || std::atoi(std::getenv("STRATA_DS4_REUSE")) != 0;
    int slot_max = slot_keep ? std::max(0, (int) env_num("STRATA_DS4_SLOT_CAP", 4)) : 0;
    if (slot_max > 0 && m.ranged() && !m.stage_ckpt_on()) {   // (a slot holds every layer's state; the folder's are left as they are)
        std::fprintf(stderr, "ds4 slots: off - this process holds only some of the layers (STRATA_DS4_SLOT_KEEP ignored)\n");
        slot_max = 0;
    }
    // a remote stage's main process (Options::stage_ckpt): a slot is this process's layers' part and the worker's, kept
    // there under an id of its own (ds4_stage.hpp; the worker's parts end with its link, so a slot is used only where
    // both sides hold it, and the folder is this mode's own, emptied at the start and at the end)
    const bool rslots = slot_max > 0 && g_remote != nullptr && m.stage_ckpt_on();
    const size_t slot_min = (size_t) std::max(1.0, env_num("STRATA_DS4_SLOT_MIN", 1024));
    const double slot_days = env_num("STRATA_DS4_SLOT_DAYS", 7);
    const uint64_t slot_budget = (uint64_t) (env_num("STRATA_DS4_SLOT_GB", 16) * 1073741824.0);
    namespace fs = std::filesystem;
    std::filesystem::path slot_dir;
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                            // empty: the state is in data (RAM)
        uint64_t bytes = 0, used = 0;
        std::shared_ptr<std::vector<uint8_t>> data;   // the RAM slot's bytes (null: a file slot)
        std::vector<int64_t> cks;                     // the positions short of its end it can be taken back at
        int64_t id = 0;                               // a remote stage: the worker's part of it (0: none)
    };
    std::vector<Slot> slots;
    uint64_t slot_clock = 0, slot_seq = 0;
    int64_t remote_slot_seq = strata::ds4::stage::kSlotId0 - 1;
    uint64_t slots_epoch = rslots ? g_remote->epoch : 0;
    const std::string slot_identity = m.slot_identity();
    const auto note = [&](const std::string& msg) {
        std::fprintf(stderr, "ds4 slots: %s\n", msg.c_str());
        std::printf("NOTE sessions: %s\n", msg.c_str());
        std::fflush(stdout);
    };
    const auto gb = [](uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    };
    const auto meta_of = [](const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; };
    const auto meta_write = [&](const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes,
                                const std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::write(path, slot_identity, toks, bytes, cks);
    };
    // a .meta of this model -> its tokens and the slot's size (false: foreign, damaged or not this model's)
    const auto meta_read = [&](const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes,
                               std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::read(path, slot_identity, m.max_context(), toks, bytes, cks);
    };
    const auto slot_age_days = [](const std::string& p) {
        std::error_code ec;
        const auto t = fs::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(fs::file_time_type::clock::now() - t).count() / 86400.0;
    };
    // (a remote stage) the ids the worker keeps its parts of: the slots held
    const auto sync_keep = [&]() {
        if (!rslots) return;
        g_remote->keep.clear();
        for (const Slot& s : slots)
            if (s.id != 0) g_remote->keep.push_back(s.id);
    };
    const auto slot_drop = [&](size_t i) {
        if (slots[i].data)   // a RAM slot: its buffer simply dies
            slots[i].data.reset();
        else {
            std::error_code ec;
            std::filesystem::remove(slots[i].path, ec);
            std::filesystem::remove(meta_of(slots[i].path), ec);
        }
        slots.erase(slots.begin() + (long) i);
        sync_keep();
    };
    // (a remote stage) a worker connected to again - restarted, or taken over by another main process - holds none of
    // the parts: the slots go (their prompts are read from token 0, as without them)
    const auto slots_stale = [&]() {
        if (!rslots || g_remote->epoch == slots_epoch) return;
        slots_epoch = g_remote->epoch;
        if (slots.empty()) return;
        const size_t n = slots.size();
        while (!slots.empty()) slot_drop(slots.size() - 1);
        note("the worker was connected again and holds no part of the " + std::to_string(n) +
             " kept conversation(s): they are removed");
    };
    const auto slot_count = [&](int& ram, int& disk) {
        ram = disk = 0;
        for (const Slot& s : slots) (s.data ? ram : disk) += 1;
    };
    // the folder's conversations (their .bin; a .meta beside each is a few KB: identity and tokens)
    const auto slot_folder_bytes = [&]() {
        uint64_t total = 0;
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {
            const std::string fn = de.path().filename().string();
            if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0) {
                const uint64_t bytes = (uint64_t) std::filesystem::file_size(de.path(), ec);
                if (!ec) total += bytes;
            }
        }
        return total;
    };
    // the folder over STRATA_DS4_SLOT_GB: its least recently used conversation goes first, and never `keep` (the one just
    // written), so a conversation bigger than the cap stays
    const auto slot_trim = [&](const std::string& keep) {
        const uint64_t before = slot_folder_bytes();
        size_t dropped = 0;
        while (slot_folder_bytes() > slot_budget) {
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (!slots[i].data && slots[i].path != keep && (lru == SIZE_MAX || slots[i].used < slots[lru].used))
                    lru = i;
            if (lru == SIZE_MAX) break;
            slot_drop(lru);
            ++dropped;
        }
        if (dropped > 0) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "the folder held %s (over STRATA_DS4_SLOT_GB, %s): %zu least recently used "
                          "conversation(s) removed, %s left", gb(before).c_str(), gb(slot_budget).c_str(), dropped,
                          gb(slot_folder_bytes()).c_str());
            note(msg);
        }
    };
    // a RAM slot into the folder (its bytes, then its .meta); false: not written (its bytes go, then)
    const auto slot_write_file = [&](Slot& s) -> bool {
        std::error_code ec;
        const auto space = std::filesystem::space(slot_dir, ec);
        if (!ec && space.available < ((uint64_t) 8 << 30)) {   // (8 GB free, the GLM engine's guard)
            std::fprintf(stderr, "ds4 slots: under 8 GB free in %s - the conversation is not written\n",
                         slot_dir.string().c_str());
            return false;
        }
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path =
            (slot_dir / ("slot-" + std::to_string(stamp) + "-" + std::to_string(++slot_seq) + ".bin")).string();
        {
            std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
            f.write((const char*) s.data->data(), (std::streamsize) s.data->size());
            f.flush();
            if (!f) {
                std::filesystem::remove(path + ".tmp", ec);
                std::fprintf(stderr, "ds4 slots: writing %s.tmp failed (disk full?)\n", path.c_str());
                return false;
            }
        }
        fs::rename(path + ".tmp", path, ec);
        if (ec || !meta_write(meta_of(path), s.tokens, s.data->size(), s.cks)) {
            std::filesystem::remove(path + ".tmp", ec);
            std::filesystem::remove(path, ec);
            std::fprintf(stderr, "ds4 slots: writing %s failed\n", path.c_str());
            return false;
        }
        s.path = path;
        s.bytes = s.data->size();
        s.data.reset();
        slot_trim(path);   // (the folder over its cap: the least recently used conversation goes, not this one)
        return true;
    };
    // the conversation in the model into a slot (the state is lost otherwise: another one takes the model)
    const auto slot_put = [&](const char* why) {
        if (slot_max <= 0) return;
        const std::vector<int32_t>& t = m.tokens();
        if (t.size() < slot_min) return;   // (a fresh engine's 0-token state is no conversation: nothing to set aside)
        slots_stale();
        const auto t0 = std::chrono::steady_clock::now();
        auto buf = std::make_shared<std::vector<uint8_t>>();
        std::string e;
        const int64_t rid = rslots ? ++remote_slot_seq : 0;
        const uint64_t bytes = m.slot_save(*buf, e, rid);   // (a state unchanged since it was taken refuses here: nothing
        if (bytes == 0) {                             // changes, its slot stays)
            if (e != "it is already set aside")
                std::fprintf(stderr, "ds4 slots: setting the conversation aside failed (%s) - it is read from token 0 "
                                     "when it comes back\n", e.c_str());
            return;
        }
        for (size_t i = slots.size(); i-- > 0;)   // an earlier state of this conversation is superseded
            if (slots[i].tokens.size() <= t.size() && std::equal(slots[i].tokens.begin(), slots[i].tokens.end(), t.begin()))
                slot_drop(i);
        slots.push_back({t, "", bytes, ++slot_clock, nullptr, m.check_positions(), rid});
        slots.back().data = buf;
        if (rslots) {   // the worker's parts are the link's to carry: only so many slots in all, the least recently used goes
            while (slots.size() > (size_t) strata::ds4::stage::kMainSlotsMax) {
                size_t lru = 0;
                for (size_t i = 1; i < slots.size(); ++i)
                    if (slots[i].used < slots[lru].used) lru = i;
                slot_drop(lru);
            }
            sync_keep();
        }
        while (true) {   // past the cap: the least recently used conversation in RAM to the folder, or gone
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            if (ram <= slot_max) break;
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (slots[i].data && (lru == SIZE_MAX || slots[i].used < slots[lru].used)) lru = i;
            if (lru == SIZE_MAX) break;
            if (!slot_write_file(slots[lru])) slot_drop(lru);
        }
        int ram = 0, disk = 0;
        slot_count(ram, disk);
        char msg[256];
        std::snprintf(msg, sizeof msg, "saved a %zu-token conversation%s (%s in RAM in %.1f s); %d in RAM, %d in %s",
                      t.size(), why, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ram, disk,
                      slot_dir.string().c_str());
        note(msg);
    };
    // the kept conversation this prompt shares the most of - past `floor` tokens, what the model itself can go back
    // to - back in the model: the tokens it covers (0: none).  A conversation the prompt continues whole comes back
    // whole (`whole`); one it leaves part way - a retried prompt, a history the client renders again without the
    // earlier reasoning - comes back at its latest checkpoint the prompt still agrees with (the end of a prompt, the
    // position before it, the anchor where its run of continued prompts began).
    const auto slot_take = [&](const std::vector<int>& prompt, size_t floor, bool& whole) -> size_t {
        whole = false;
        if (slot_max <= 0 || prompt.size() < 2) return 0;
        slots_stale();
        std::vector<strata::ds4::slotmeta::Cand> cands;
        for (const Slot& sl : slots) cands.push_back({&sl.tokens, &sl.cks});
        size_t len = 0;
        const size_t best = strata::ds4::slotmeta::pick(cands, prompt, floor, &len);
        if (best == SIZE_MAX) return 0;
        const size_t whole_len = slots[best].tokens.size();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ram = slots[best].data != nullptr;
        std::string e;
        bool back = false;
        if (ram) back = m.slot_load(slots[best].data->data(), slots[best].data->size(), (int64_t) whole_len, e, slots[best].id);
        else {
            std::ifstream f(slots[best].path, std::ios::binary);
            std::vector<uint8_t> file;
            if (f) {
                f.seekg(0, std::ios::end);
                const std::streamoff n = f.tellg();
                f.seekg(0, std::ios::beg);
                if (n > 0) {
                    file.resize((size_t) n);
                    if (f.read((char*) file.data(), n))
                        back = m.slot_load(file.data(), file.size(), (int64_t) whole_len, e, slots[best].id);
                    else e = "reading " + slots[best].path + " failed";
                } else e = slots[best].path + " is empty";
            } else e = "cannot read " + slots[best].path;
        }
        if (!back) {
            note("taking a conversation back failed (" + (e.empty() ? std::string("it did not load") : e) +
                 ") - reading its prompt again");
            slot_drop(best);
            return 0;
        }
        if (len < whole_len && !(m.restore((int64_t) len) && (size_t) m.pos() == len)) {
            note("going back to token " + std::to_string(len) + " of a kept conversation failed - reading its prompt again");
            m.reset();
            return 0;
        }
        whole = len == whole_len;
        slots[best].used = ++slot_clock;
        if (!ram) {   // its age counts from its last use (a file slot's only)
            std::error_code ec;
            fs::last_write_time(meta_of(slots[best].path), fs::file_time_type::clock::now(), ec);
        }
        char msg[256];
        std::string part;
        if (!whole) part = " at token " + std::to_string(len) + " of " + std::to_string(whole_len);
        std::snprintf(msg, sizeof msg, "took back a %zu-token conversation%s from %s (%.1f s); %zu new tokens to read",
                      len, part.c_str(), ram ? "RAM" : "the folder",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                      prompt.size() - len);
        note(msg);
        return len;
    };
    if (slot_max > 0) {
        slot_dir = std::getenv("STRATA_DS4_SLOT_DIR") ? std::filesystem::path(std::getenv("STRATA_DS4_SLOT_DIR"))
                                                      : std::filesystem::path(pack) / "slots";
        if (rslots)   // (this mode's own folder: its .meta files carry the layers' range, no single process's slot is one)
            slot_dir /= "split-" + std::to_string(m.stage_geo().layer_end);
        std::error_code ec;
        std::filesystem::create_directories(slot_dir, ec);
        if (!std::filesystem::is_directory(slot_dir, ec)) {
            std::fprintf(stderr, "ds4 slots: %s is not writable - conversations are not set aside\n",
                         slot_dir.string().c_str());
            slot_max = 0;
        } else if (rslots) {   // (the worker's parts ended with the link they were made on: nothing here can be taken back)
            size_t gone = 0;
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec))
                if (de.path().filename().string().rfind("slot-", 0) == 0 && std::filesystem::remove(de.path(), ec)) ++gone;
            if (gone > 0) note("folder " + slot_dir.string() + ": " + std::to_string(gone) + " file(s) of an earlier run removed");
        } else {
            struct Found {
                std::string bin;
                std::vector<int32_t> toks;
                uint64_t bytes;
                std::vector<int64_t> cks;
                fs::file_time_type t;
            };
            std::vector<Found> found;
            size_t refused = 0;
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a previous run's slots
                const std::string fn = de.path().filename().string(), p = de.path().string();
                if (fn.rfind("slot-", 0) != 0) continue;
                const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
                const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
                if (!is_bin && !is_meta) {   // a .tmp an interrupted write left
                    std::filesystem::remove(de.path(), ec);
                    continue;
                }
                if (!is_meta) continue;      // a .bin is judged by its .meta
                const std::string bin = p.substr(0, p.size() - 5) + ".bin";
                Found fd{bin, {}, 0, {}, fs::last_write_time(de.path(), ec)};
                const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
                if (meta_read(p, fd.toks, fd.bytes, fd.cks) && have == fd.bytes && fd.bytes > 0 && slot_age_days(p) <= slot_days)
                    found.push_back(std::move(fd));
                else {
                    ++refused;
                    std::filesystem::remove(p, ec);
                    std::filesystem::remove(bin, ec);
                }
            }
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a .bin without its .meta
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    std::filesystem::remove(de.path(), ec);
            }
            std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
            uint64_t total = 0;
            for (auto& fd : found) {
                total += fd.bytes;
                slots.push_back({std::move(fd.toks), fd.bin, fd.bytes, ++slot_clock, nullptr, std::move(fd.cks)});
            }
            std::string msg = "folder " + slot_dir.string() + ": " + std::to_string(slots.size()) +
                              " conversation(s) kept from earlier runs (" + gb(total) + ")";
            if (refused) msg += ", " + std::to_string(refused) + " not this model's or incomplete - removed";
            note(msg);
            slot_trim("");   // (the folder over STRATA_DS4_SLOT_GB: the least recently used conversation goes first)
        }
    }
    if (slot_max > 0) {
#if !defined(_WIN32)
        // (the server ends the engine with QUIT and waits for it; a SIGTERM to the whole process group - systemd
        // stopping a service, a supervisor killing the server's group - must not end it mid-write, and a server that
        // is already gone must not end it with SIGPIPE when it writes its last lines)
        std::signal(SIGTERM, SIG_IGN);
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGPIPE, SIG_IGN);
#endif
    }
    while (in.next(line)) {
        if (line == "QUIT") break;
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r, geni)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        const double t0 = now_ms();
        std::vector<Pic> pics;
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
                std::fflush(stdout);
                continue;
            }
        }
        if (!reuse) {   // (STRATA_DS4_REUSE=0, for A/Bs and tests: every request is a fresh read of its whole prompt)
            m.reset();
            held.clear();
        }
        if (g_remote != nullptr) {   // (a remote stage whose link went down while idle: its worker holds nothing of the
            std::string e;           // conversation - read it again from token 0, connecting again at the reset)
            if (!g_remote->link.alive(e)) {
                std::fprintf(stderr, "maya ds4: remote stage: %s\n", e.empty() ? "the link is down" : e.c_str());
                m.reset();
                held.clear();
            }
            slots_stale();
        }
        const auto& st = m.tokens();
        const size_t n_state = st.size();
        size_t c = 0;
        while (c < n_state && c < r.ids.size() && st[c] == r.ids[c]) ++c;
        // the same tokens with another picture (or none) do not share the state from that picture's span on
        auto same = [](const std::vector<Pic>& v, const Pic& q) {
            for (const Pic& x : v)
                if (x.start == q.start && x.hash == q.hash) return true;
            return false;
        };
        for (const Pic& q : pics)
            if ((size_t) q.start < c && !same(held, q)) c = (size_t) q.start;
        for (const Pic& q : held)
            if ((size_t) q.start < c && !same(pics, q)) c = (size_t) q.start;
        size_t from = 0;
        bool took = false, continued = false;   // (continued: the conversation goes on - its anchor stays)
        // the conversation continues, reused in place - a conversation has to be in the model first: a fresh engine
        // (the start of a run, so every restart) holds 0 tokens, and every prompt "continues" that to token 0; taken
        // as a continuation, the take below would never run for the folder's conversations, the case they are for
        if ((n_state > 0 || slot_max <= 0) && c == n_state && n_state < r.ids.size()) {   // (slots off: as before)
            from = n_state;
            continued = true;
        }
        else {
            // CONVERSATION SLOTS: this prompt does not continue the conversation in the model - that one is set aside
            // first (another one takes the model, and its state is lost otherwise), then the kept conversation this
            // prompt does continue is taken back - if it goes further back than the checkpoints the state itself holds
            if (held.empty()) slot_put("");
            const int64_t limit = (int64_t) std::min(c, r.ids.size() - 1);
            const int64_t rs = m.restorable(limit);
            bool whole = false;
            const size_t back = pics.empty() ? slot_take(r.ids, rs > 0 ? (size_t) rs : 0, whole) : 0;
            if (back > 0) {
                from = back;
                took = true;
                continued = whole;
            } else if (rs >= 0 && m.restore(limit))
                from = (size_t) m.pos();
            else m.reset();
        }
        if (took || from == 0) held.clear();   // (a slot is text only: the state that comes back holds no pictures)
        if ((int64_t) r.ids.size() + std::max(0LL, r.max_new) > m.max_context()) {
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", r.ids.size(),
                        r.max_new, (long long) m.max_context());
            std::fflush(stdout);
            continue;
        }
        std::string err;
        bool ok = true;
        // the pictures whose spans the prompt reads (the rest the state holds): encoded (or the cache's), checked
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            const double te = now_ms();
            int encoded = 0;
            for (Pic& q : pics) {
                for (const Pic& x : held)
                    if (x.start == q.start && x.hash == q.hash) q.len = x.len;
                if (q.len > 0 && (size_t) q.start + q.len <= from) continue;
                std::shared_ptr<const std::vector<uint16_t>> rows = spans.get(m, q, err);
                if (!rows) {
                    ok = false;
                    break;
                }
                ++encoded;
                q.len = rows->size() / (size_t) m.dim();
                bool fits = (size_t) q.start + q.len <= r.ids.size() - 1;
                for (size_t i = 0; fits && i < q.len; ++i) fits = r.ids[(size_t) q.start + i] == img_tok;
                if (!fits) {
                    err = "a picture's span (" + std::to_string(q.len) + " tokens at " + std::to_string(q.start) +
                          ") is not a run of image tokens in the prompt";
                    ok = false;
                    break;
                }
                imgs.push_back({q.start, rows});
            }
            if (ok) {
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
        }
        // the prompt in windows (PP <read> <total> <ms> <tok/s> <reuse> every 64 tokens), a checkpoint before its last
        // token and one after it: the next turn re-renders this one, and shares its start up to one of them
        const size_t total = r.ids.size();
        size_t done = from, next_pp = from + 64;
        m.set_progress([&](int64_t d) {   // (a run of chunks: each one through every layer)
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done + (size_t) d, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        // (pin=N: the read stops at N for a checkpoint of the shared start; only when this read crosses N, and N leaves
        //  the prompt's last token to the usual end)
        const size_t pin_at = (r.pin >= 256 && (size_t) r.pin + 1 < total) ? (size_t) r.pin : 0;
        // a STOP while the prompt is read ends it there (the engine's run of chunks at a layer or chunk, windows between
        // feeds) and goes back to the state before this prompt - kept here first, in checkpoint 1, which the prompt's
        // own end overwrites as before - so the next request still shares the conversation up to it
        if (from > 0 && from < total) m.checkpoint(1);
        m.set_cancel([&in] { return in.stop.load(); });
        bool stopped = false;
        while (ok && done < total) {
            if (in.stop.load()) {
                stopped = true;
                break;
            }
            size_t n = std::min<size_t>((size_t) m.feed_unit(), total - done);
            if (pin_at > done && pin_at < done + n) n = pin_at - done;
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = done + n == total;
            if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + done, (int) n - 1, 0, err));
                if (ok) m.checkpoint(1);
                ok = ok && m.feed(r.ids.data() + done + n - 1, 1, 1, err);
            } else
                ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + done, (int) n, 0, err);
            done += n;
            if (ok && pin_at != 0 && done == pin_at) m.checkpoint(3);
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP %zu %zu %.1f %.1f %zu\n", done, r.ids.size(), ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
        }
        m.set_cancel(nullptr);
        m.set_images({});
        if (stopped || (!ok && m.cancelled())) {
            const bool back = from > 0 && m.restore((int64_t) from) && (size_t) m.pos() == from;
            if (!back) {
                m.reset();
                held.clear();
            }
            std::fprintf(stderr, "ds4 serve: stopped while reading the prompt of %zu tokens, after %.0f ms%s\n",
                         total, now_ms() - t0, back ? "; the conversation before it is kept" : "");
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), now_ms() - t0, back ? from : (size_t) 0);
            std::fflush(stdout);
            continue;
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            if (g_remote != nullptr) {   // (a remote stage: the worker again within the request - its wait's heartbeat
                m.reset();               // and its STOP are this request's)
                held.clear();
            }
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            if (g_remote == nullptr) {
                m.reset();
                held.clear();
            }
            continue;
        }
        held = std::move(pics);
        m.checkpoint(0);
        if (!continued && slot_max > 0) m.pin();   // (a new run of prompts: where it may be rendered again from)
        const double t1 = now_ms();
        long long produced = 0;
        const char* finish = "length";
        const long long limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
        long long d_off = 0, d_acc = 0;
        bool cool = false;
        int tok = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, counter++);
        std::vector<int> got;
        // STAT for the Monitor every 8 tokens and at the end, over this answer so far: the GLM engine's keys (speed, the
        // VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB a token the storage
        // read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
        const strata::ds4::Stats s_dec = m.stats();
        const long long io_dec = io_read_bytes();
        auto stat_line = [&]() {
            const strata::ds4::Stats& s = m.stats();
            const double ms = now_ms() - t1;
            const double fed = (double) std::max<int64_t>(1, s.tokens - s_dec.tokens);
            const double ex = (double) (s.experts - s_dec.experts), cx = (double) (s.cpu_experts - s_dec.cpu_experts);
            std::string out = "STAT";
            char b[160];
            std::snprintf(b, sizeof b, " tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f ram=%.3f promo=%.3f engram_ms=%.3f",
                          ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                          ex > 0 ? 1.0 - cx / ex : 1.0, cx / fed, (double) (s.ram_experts - s_dec.ram_experts) / fed,
                          (double) (s.swaps - s_dec.swaps) / fed, (s.engram_ms - s_dec.engram_ms) / fed);
            out += b;
            const long long io = io_read_bytes();
            if (io >= 0 && io_dec >= 0) {
                std::snprintf(b, sizeof b, " disk_mb=%.3f", (double) (io - io_dec) / 1e6 / fed);
                out += b;
            }
            if (d_off > 0) {
                std::snprintf(b, sizeof b, " drafts=%.3f", (double) d_acc / (double) d_off);
                out += b;
            }
            if (s.skipped > s_dec.skipped) {   // (STRATA_DS4_DISK_SKIP_W: experts left out a token)
                std::snprintf(b, sizeof b, " skipped=%.3f", (double) (s.skipped - s_dec.skipped) / fed);
                out += b;
            }
            std::printf("%s\n", out.c_str());
        };
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            std::printf("T %d\n", tok);
            ++produced;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            bool end = false;
            for (int e : eos) end = end || tok == e;
            if (end) {
                finish = "stop";
                break;
            }
            if (produced >= limit || m.pos() >= m.max_context()) break;
            if (got.empty()) {   // the next round: the drafts (if any) and the token after them
                const int room = (int) std::min<long long>(limit - produced, m.max_context() - m.pos());
                int off = 0, acc = 0;
                // a round whose drafts were all below the confidence floor: the next one does without the drafter
                // (a request's spec / dspark_conf keys over the engine's: the speed calibration measures both ways)
                const bool try_spec = (r.spec < 0 ? spec : r.spec != 0) && !cool;
                if (!round(m, tok, room, try_spec, r.conf >= 0 ? r.conf : conf_min, r.temperature, r.top_p, r.top_k,
                           r.min_p, r.seed, counter, got,
                           off, acc, err, r.cum)) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    finish = "cancel";
                    if (g_remote != nullptr) {   // (a remote stage: the worker's state is unknown - the next request
                        m.reset();               // reads from token 0)
                        held.clear();
                    }
                    break;
                }
                d_off += off;
                d_acc += acc;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        // (a stop within accepted drafts leaves the state past it: the next prompt shares less with it, and goes back to
        //  the checkpoint at this prompt's end)
        stat_line();
        const double t2 = now_ms();
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                    d_acc, d_off, from);
        std::fflush(stdout);
        const auto& s = m.stats();
        std::fprintf(stderr,
                     "maya ds4 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s); CPU lane %.1f%% of the experts\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0,
                     s.experts ? 100.0 * (double) s.cpu_experts / (double) s.experts : 0.0);
        if (d_off > 0) std::fprintf(stderr, "maya ds4 serve: drafts %lld of %lld kept; %s\n", d_acc, d_off,
                                    per_position(s_dec, s).c_str());
        m.save_profile();
    }
    // the engine ends (QUIT or the end of its input): the conversation in the model is set aside, and every one in
    // RAM is written to the folder, so the next run reads on from them
    if (rslots) {   // (the worker's parts end with its link: nothing here would be taken back by the next run)
        while (!slots.empty()) slot_drop(slots.size() - 1);
    } else if (slot_max > 0) {
        if (held.empty()) slot_put(" as the engine ended");
        size_t written = 0;
        while (true) {   // (a write can take the folder's least recently used conversation away: the list is asked again)
            size_t i = SIZE_MAX;
            for (size_t k = 0; k < slots.size(); ++k)
                if (slots[k].data) {
                    i = k;
                    break;
                }
            if (i == SIZE_MAX) break;
            if (slot_write_file(slots[i])) ++written;
            else slot_drop(i);   // (its state could not be written: there is nothing more to do at the engine's end)
        }
        if (written > 0) {
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            char msg[256];
            std::snprintf(msg, sizeof msg, "wrote %zu conversation(s) to %s; %d there in all", written,
                          slot_dir.string().c_str(), disk);
            note(msg);
        }
    }
    m.save_profile();
    return 0;
}


// ---- Strata addition: teacher-forced scoring (the quality gate) -------------------------------------------------
// --score IDS.json --score-out RUN.tsv: every segment of a tools/quant_quality.py ids file (name + ids) is read through
// the model from a fresh state; per position the log-probability of the text's next token and the 64 most likely
// tokens go out in the STRATA_LOGPOS row format that `quant_quality.py compare` reads (natural logs, full-vocabulary
// log-softmax in double).  RUN.tsv.seg.json lists each segment's first line and size.
struct ScoreSeg {
    std::string name;
    std::vector<int> ids;
};

bool read_score_ids(const std::string& path, std::vector<ScoreSeg>& segs, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = path + ": cannot open";
        return false;
    }
    std::string s;
    char buf[1 << 16];
    size_t k;
    while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k);
    std::fclose(f);
    size_t p = 0;
    for (;;) {
        const size_t n0 = s.find("\"name\"", p);
        if (n0 == std::string::npos) break;
        const size_t q0 = s.find('"', s.find(':', n0) + 1);
        const size_t q1 = s.find('"', q0 + 1);
        const size_t i0 = s.find("\"ids\"", q1);
        if (q0 == std::string::npos || q1 == std::string::npos || i0 == std::string::npos) break;
        const size_t b0 = s.find('[', i0), b1 = s.find(']', b0);
        if (b0 == std::string::npos || b1 == std::string::npos) break;
        ScoreSeg g;
        g.name = s.substr(q0 + 1, q1 - q0 - 1);
        g.ids = parse_ids(s.substr(b0 + 1, b1 - b0 - 1));
        segs.push_back(std::move(g));
        p = b1;
    }
    if (segs.empty()) {
        err = path + ": no segments (expected quant_quality.py's ids file)";
        return false;
    }
    return true;
}

int score(strata::ds4::Model& m, const std::string& ids_path, const std::string& out_path, int window, int eos) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    // DS4_SCORE_PREFIX=N (a test of the long-prompt layer skip): the segments are joined into one stream, its first N
    // tokens are read as one prompt (logits of the last row only: the path a long prompt takes, where the skip applies)
    // and only the positions after them are scored, in windows
    const long long prefix_n = std::getenv("DS4_SCORE_PREFIX") ? std::atoll(std::getenv("DS4_SCORE_PREFIX")) : 0;
    if (prefix_n > 0 && segs.size() > 1) {
        ScoreSeg all;
        all.name = "concat";
        for (const ScoreSeg& sg : segs) all.ids.insert(all.ids.end(), sg.ids.begin(), sg.ids.end());
        segs.assign(1, all);
    }
    std::FILE* out = std::fopen(out_path.c_str(), "wb");
    std::FILE* sj = std::fopen((out_path + ".seg.json").c_str(), "wb");
    if (out == nullptr || sj == nullptr) {
        std::fprintf(stderr, "strata-dsv41: cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(sj, "{\"segments\": [");
    const int V = m.vocab();
    long long line = 0;
    std::vector<float> lg;
    std::vector<int> order;
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        const long long first = line;
        double sum_nll = 0.0;
        long long top1 = 0, cnt = 0;
        size_t a0 = 0;
        if (prefix_n > 0 && (size_t) prefix_n + 1 < ids.size()) {
            a0 = (size_t) prefix_n;
            if (!m.feed(ids.data(), (int) a0, 1, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
        }
        for (size_t a = a0; a + 1 < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - 1 - a);   // (positions a .. a+T-1 are scored)
            if (!m.feed(ids.data() + a, T, T, err) || !m.logits(lg) || lg.size() < (size_t) T * (size_t) V) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.empty() ? "no logits" : err.c_str());
                return 1;
            }
            for (int t = 0; t < T; ++t) {
                const float* r = lg.data() + (size_t) t * (size_t) V;
                float mx = r[0];
                for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
                double z = 0.0;
                for (int v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
                const double lz = std::log(z) + (double) mx;
                order.resize((size_t) V);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](int x, int y) {
                    return r[x] > r[y] || (r[x] == r[y] && x < y);
                });
                const int tgt = ids[a + (size_t) t + 1];
                const double lt = (double) r[tgt] - lz;
                const double la = (double) r[order[0]] - lz;
                const double le = eos >= 0 && eos < V ? (double) r[eos] - lz : -1e30;
                // logp(target) with the end token left out of the distribution, as the engine's row has it
                const double lt_ne = tgt == eos ? lt : std::log(std::max(1e-300, std::exp(lt) / std::max(1e-300, 1.0 - std::exp(le))));
                std::fprintf(out, "%zu\t%d\t%.6f\t%d\t%.6f\t%d\t%.6f\t%.6f", a + (size_t) t, tgt, lt, order[0], la,
                             order[0] == tgt ? 1 : 0, le, lt_ne);
                for (int j = 0; j < 64; ++j) std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) r[order[(size_t) j]] - lz);
                std::fprintf(out, "\n");
                ++line;
                ++cnt;
                top1 += order[0] == tgt;
                sum_nll -= lt;
            }
            a += (size_t) T;
        }
        std::fprintf(sj, "%s{\"name\": \"%s\", \"first_line\": %lld, \"end_line\": %lld, \"n\": %lld, \"ppl\": %.6f, \"top1\": %.6f}", si ? ", " : "",
                     segs[si].name.c_str(), first, first + cnt, cnt, std::exp(sum_nll / (double) std::max(1LL, cnt)),
                     (double) top1 / (double) std::max(1LL, cnt));
        std::fprintf(stderr, "score %s: %lld positions, ppl %.4f, top-1 %.4f (%.0f s)\n", segs[si].name.c_str(), cnt,
                     std::exp(sum_nll / (double) std::max(1LL, cnt)), (double) top1 / (double) std::max(1LL, cnt),
                     (now_ms() - t0) / 1000.0);
        std::fflush(out);
    }
    std::fprintf(sj, "]}\n");
    std::fclose(sj);
    std::fclose(out);
    return 0;
}
// ---- Strata H100 addition: profile run -------------------------------------------------------------------------
// --profile-run IDS.json: every segment is fed through the model from a fresh state (no logits); the engine's expert
// profile (--expert-profile FILE, counts a layer and expert) and DS4_TRACE collect the routing.  The profile is saved
// every 10 segments.  --window sets the chunk (a chunked prompt path above 16).
int profile_run(strata::ds4::Model& m, const std::string& ids_path, int window) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    const double t0 = now_ms();
    long long tot = 0;
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            // (no logits: with --remote-experts its windows never settle the cache - offline, run without a server)
            if (!m.feed(ids.data() + a, T, 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        tot += (long long) ids.size();
        std::fprintf(stderr, "profile-run %zu/%zu %s: %zu tokens, %lld in all, %.0f s, %.1f tok/s\n", si + 1, segs.size(),
                     segs[si].name.c_str(), ids.size(), tot, (now_ms() - t0) / 1000.0, 1000.0 * (double) tot / (now_ms() - t0));
        if ((si + 1) % 10 == 0) m.save_profile();
    }
    m.save_profile();
    return 0;
}
// ---- Strata H100 addition: DSpark acceptance log ---------------------------------------------------------------
// --spec-eval IDS.json --n N: each segment is a prompt: fed from reset, then N greedy tokens decoded with the drafter, every
// draft verified (--dspark-conf 0); DS4_ROUND_LOG=file gets a line a round (segment name, drafts, accepted, confidences).
int spec_eval(strata::ds4::Model& m, const std::string& ids_path, int n, int window, float conf_min, bool spec) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    if (const char* f = std::getenv("DS4_ROUND_LOG")) g_round_log = std::fopen(f, "wb");
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        g_round_tag = segs[si].name;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            const bool last = a + (size_t) T == ids.size();
            if (!m.feed(ids.data() + a, T, last ? 1 : 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        unsigned long long counter = 0;
        int tok = m.sample(0.0f, 1.0f, 1, 0.0f, 1, counter++);
        std::vector<int> got;
        int emitted = 1;
        long long rounds = 0, acc_all = 0;
        bool stop = false;
        while (emitted < n && !stop) {
            int off = 0, acc = 0;
            if (!round(m, tok, n - emitted, spec, conf_min, 0.0f, 1.0f, 1, 0.0f, 1, counter, got, off, acc, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            acc_all += acc;
            for (int t : got) {
                ++emitted;
                if (t == 1) stop = true;
            }
            tok = got.back();
        }
        std::fprintf(stderr, "spec-eval %zu/%zu %s: prompt %zu, %d tokens, %lld rounds, %.2f tokens a round, accepted %lld (%.0f s)\n", si + 1,
                     segs.size(), segs[si].name.c_str(), ids.size(), emitted, rounds, (double) (emitted - 1) / (double) std::max(1LL, rounds),
                     acc_all, (now_ms() - t0) / 1000.0);
        if (g_round_log != nullptr) std::fflush(g_round_log);
    }
    return 0;
}
}  // namespace

#ifndef DS4_BUILD_ID
#define DS4_BUILD_ID "unknown"
#endif

int main(int argc, char** argv) {
    strata::ds4::os::fine_timer();
    strata::ds4::Options o;
    std::string ids_s, dump, eos = "1";
    int n = 0, window = 0;   // (0: the model's feed unit - a whole prompt on the chunked path; 16 for the scoring runs)
    bool spec = true;
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false;
    bool pack_info_mode = false;
    std::string images, enc_rgb, enc_out;
    std::string score_ids, score_out, prof_ids, spec_ids;
    std::string remote_at, stage_bind;   // (a remote stage: the worker's HOST:PORT; a worker's address to listen on)
    int worker_port = 0;
    bool worker_mode = false;
    std::string rx_at;          // (--expert-server-connect: the main process's HOST:PORT)
    int64_t rx_reserve = 512;   // (its VRAM left free, MiB: the arena slot's 512 held 1106 blobs on the 3080)
    const char* rx_flag = nullptr;   // (an expert server's flag given: it needs --expert-server-connect)
    bool rx_once = false;
    bool rx_cpu = false;        // (--expert-server-cpu, tests: the CPU lane's arithmetic, no GPU)
    int enc_w = 0, enc_h = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = val("--pack");
        else if (a == "--gpus") {
            for (int g : parse_ids(val("--gpus"))) o.gpus.push_back(g);
        } else if (a == "--max-context") o.max_context = std::atoll(val("--max-context").c_str());
        else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val("--reserve-mib").c_str());
        else if (a == "--vram-experts") o.vram_experts = std::atof(val("--vram-experts").c_str());
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--expert-profile") o.profile = val("--expert-profile");
        else if (a == "--ssd-cache") o.ssd_cache_gb = std::atof(val("--ssd-cache").c_str());
        else if (a == "--ssd-policy") {
            const std::string v = val("--ssd-policy");
            o.ssd_policy = v == "lfu" ? 1 : v == "lfulru" ? 2 : 0;
        }
        else if (a == "--eos") eos = val("--eos");
        else if (a == "--no-expert-cache") o.expert_cache = false;
        else if (a == "--dspark") {
            const std::string v = val("--dspark");
            spec = v != "off";
            o.dspark = v != "none";   // none: not even loaded
            o.dspark_auto = v == "auto";
        } else if (a == "--dspark-conf") o.dspark_conf = (float) std::atof(val("--dspark-conf").c_str());
        else if (a == "--dspark-block") o.dspark_block = std::max(0, std::atoi(val("--dspark-block").c_str()));
        else if (a == "--tp") o.tp = std::max(1, std::atoi(val("--tp").c_str()));
        else if (a == "--prefill-chunk") o.prefill_chunk = std::max(0, std::atoi(val("--prefill-chunk").c_str()));
        else if (a == "--threads") o.threads = val("--threads") != "off";
        else if (a == "--vision") o.vision = val("--vision") != "off";
        else if (a == "--mmap-experts") {
            const std::string v = val("--mmap-experts");
            o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
        } else if (a == "--images") images = val("--images");
        else if (a == "--encode-image") enc_rgb = val("--encode-image");
        else if (a == "--dump-image") enc_out = val("--dump-image");
        else if (a == "--size") {
            const std::string v = val("--size");
            enc_w = std::atoi(v.c_str());
            enc_h = v.find('x') != std::string::npos ? std::atoi(v.c_str() + v.find('x') + 1) : 0;
        }
        else if (a == "--temperature") temp = (float) std::atof(val("--temperature").c_str());
        else if (a == "--seed") cli_seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
        else if (a == "--verbose" || a == "-v") {
            o.verbose = true;
            std::fprintf(stderr, "ds4: build %s (%s)\n", DS4_BUILD_ID, STRATA_VERSION);
        } else if (a == "--version") {
            std::printf("strata-dsv41 %s build %s\n", STRATA_VERSION, DS4_BUILD_ID);
            return 0;
        }
        else if (a == "--ids") ids_s = val("--ids");
        else if (a == "--n") n = std::atoi(val("--n").c_str());
        else if (a == "--window") window = std::max(1, std::atoi(val("--window").c_str()));   // (> 16: the chunked prompt path)
        else if (a == "--dump-logits") dump = val("--dump-logits");
        else if (a == "--score") score_ids = val("--score");
        else if (a == "--score-out") score_out = val("--score-out");
        else if (a == "--profile-run") prof_ids = val("--profile-run");
        else if (a == "--spec-eval") spec_ids = val("--spec-eval");
        else if (a == "--serve") serve_mode = true;
        else if (a == "--pack-info") pack_info_mode = true;
        else if (a == "--layer-split") (void) val("--layer-split");   // the server's (this engine balances its own split)
        else if (a == "--stage-begin") o.stage_begin = std::atoi(val("--stage-begin").c_str());
        else if (a == "--stage-end") o.stage_end = std::atoi(val("--stage-end").c_str());
        else if (a == "--remote-stage") remote_at = val("--remote-stage");
        else if (a == "--stage-worker") {
            worker_mode = true;
            worker_port = std::atoi(val("--stage-worker").c_str());
        }
        else if (a == "--stage-bind") stage_bind = val("--stage-bind");
        else if (a == "--remote-experts") o.remote_experts = val("--remote-experts");
        else if (a == "--expert-server-connect") rx_at = val("--expert-server-connect");
        else if (a == "--expert-server-reserve-mib") {
            rx_reserve = std::atoll(val("--expert-server-reserve-mib").c_str());
            rx_flag = "--expert-server-reserve-mib";
        } else if (a == "--expert-server-once") {
            rx_once = true;
            rx_flag = "--expert-server-once";
        } else if (a == "--expert-server-cpu") {
            rx_cpu = true;
            rx_flag = "--expert-server-cpu";
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (rx_flag != nullptr && rx_at.empty()) {   // (not silently a main process: a server's flag, its address left out)
        std::fprintf(stderr, "%s is an expert server's flag: it needs --expert-server-connect HOST:PORT\n", rx_flag);
        usage();
        return 2;
    }
    if (!rx_at.empty()) {   // the expert server (ds4_rx.hpp): no pack, the main process sends what it computes
        std::string token;
        if (!stage_token(token)) return 1;
        if (rx_cpu) return strata::ds4::rx::serve_cpu(rx_at, token, rx_once);
        return strata::ds4::rx::serve(rx_at, token, rx_reserve, o.gpus.empty() ? 0 : o.gpus[0], rx_once);
    }
    if (pack_info_mode) {   // the pack's description only: no device, no weights, no lock
        if (o.pack.empty()) {
            usage();
            return 2;
        }
        return strata::ds4::pack_info(o.pack, stdout);
    }
    // a remote stage (ds4_stage.hpp): the worker (--stage-worker PORT --stage-begin K) serves its main process instead
    // of a mode of its own; the main process (--remote-stage HOST:PORT --stage-end K) runs any mode
    {
        std::string why;
        if (worker_mode && !remote_at.empty()) why = "--stage-worker or --remote-stage, not both";
        else if (worker_mode && (worker_port < 1 || worker_port > 65535)) why = "--stage-worker PORT: 1 to 65535";
        else if (worker_mode && (o.stage_begin < 1 || o.stage_end >= 0))
            why = "--stage-worker needs --stage-begin K (1 or more) and runs to the last layer (no --stage-end: no relay)";
        else if (worker_mode && (!ids_s.empty() || !score_ids.empty() || !prof_ids.empty() || !spec_ids.empty()))
            why = "--stage-worker serves its main process only (not with --ids, --score, --profile-run or --spec-eval)";
        else if (!remote_at.empty() && (o.stage_end < 1 || o.stage_begin != 0))
            why = "--remote-stage needs --stage-end K: this PC runs layers 0 to K - 1 and the head, the worker the rest";
        else if ((worker_mode || !remote_at.empty()) && !enc_rgb.empty()) why = "--encode-image: not with a remote stage";
        else if (!stage_bind.empty() && !worker_mode) why = "--stage-bind is a worker's (--stage-worker)";
        if (!why.empty()) {
            std::fprintf(stderr, "maya-ds4: %s\n", why.c_str());
            return 2;
        }
        o.stage_worker = worker_mode;
        o.remote_stage = !remote_at.empty();
        // a remote stage's rows, either side: the compact ones (ds4_stage.hpp rows_pack) unless STRATA_REMOTE_ROWS=fp32
        // (debugging; both sides alike).  The link's own bf16 would round the int32 picks each row carries, so the
        // engine packs the rows itself and the hello says so
        o.stage_bf16 = worker_mode || o.remote_stage;
        if (const char* v = std::getenv("STRATA_REMOTE_ROWS"); (worker_mode || o.remote_stage) && v != nullptr && *v != '\0') {
            strata::net::StageRows f = strata::net::StageRows::F32;
            if (!strata::net::stage_rows_parse(v, f)) {
                std::fprintf(stderr, "maya-ds4: STRATA_REMOTE_ROWS=%s: expected fp32 or bf16\n", v);
                return 2;
            }
            o.stage_bf16 = f == strata::net::StageRows::BF16;
        }
        // the kept conversations (STRATA_DS4_SLOT_KEEP=1; STRATA_DS4_REUSE=0 turns every reuse off) across the link
        const char* keep_env = std::getenv("STRATA_DS4_SLOT_KEEP");
        const char* reuse_env = std::getenv("STRATA_DS4_REUSE");
        o.stage_ckpt = o.remote_stage && serve_mode && keep_env != nullptr && std::atoi(keep_env) != 0 &&
                       (reuse_env == nullptr || std::atoi(reuse_env) != 0);
    }
    if (o.pack.empty() || (!serve_mode && !worker_mode && ids_s.empty() && enc_rgb.empty() && score_ids.empty() && prof_ids.empty() && spec_ids.empty())) {
        usage();
        return 2;
    }
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.vision_only = !enc_rgb.empty();
    if (o.vision_only && (o.stage_begin != 0 || o.stage_end >= 0)) {
        std::fprintf(stderr, "maya-ds4: --encode-image reads no layers: not with --stage-begin / --stage-end\n");
        return 2;
    }
    RemoteMain remote;   // (before the model: the model's hooks call it until the model is gone)
    strata::ds4::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    if (o.stage_worker) return stage_worker(m, o.pack, stage_bind, worker_port);
    if (o.remote_stage) {
        std::string token;
        if (!stage_token(token)) return 1;
        remote.at = remote_at;
        remote.me = stage_hello(m, o.pack, false, o.stage_ckpt);
        std::memcpy(remote.me.token, token.data(), token.size());
        remote.chunk_to = [&m](int64_t c) { m.set_stage_chunk((int) c); };
        if (!remote.connect(err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "maya ds4: remote stage %s: layers %d-%d there (context %lld, rows %s, prompt chunks of %lld "
                             "here, %lld at most there)\n", remote_at.c_str(),
                     remote.peer.layer_begin, remote.peer.n_layers - 1, (long long) remote.peer.max_context,
                     m.stage_geo().rows_bf16 ? "bf16 (the streams; the picks exact)" : "fp32", (long long) remote.sent_chunk,
                     (long long) remote.peer.chunk);
        remote.model = &m;
        g_remote = &remote;
        if (o.stage_ckpt)
            m.set_stage_ckpt({[&remote](int64_t id, int64_t L, bool& stored, std::string& e) { return remote.ck_save(id, L, stored, e); },
                              [&remote](int64_t id, int64_t L, bool& missing, std::string& e) {
                                  return remote.ck_restore(id, L, missing, e);
                              }});
        m.set_stage_link([&remote](int T, const int32_t* tokens, int64_t p0, const float* out, float* in,
                                   std::string& e) { return remote.run(T, tokens, p0, out, in, e); },
                         [&remote](int n_keep, std::string& e) { return remote.commit(n_keep, e); },
                         [&remote](std::string& e) { return remote.reset(e); });
        if (remote.me.chunk > 0) {   // (the chunked prompt path's chunks: over the link as they leave layer K - 1)
            const strata::ds4::Model::StageGeo sg = m.stage_geo();
            strata::ds4::stage::RowGeo rg;   // (the hand-off row: the compact one on the link with bf16 rows)
            rg.h = (int64_t) sg.hc * sg.dim;
            rg.topk = sg.topk;
            rg.cand = sg.cand;
            remote.chunks = std::make_unique<strata::ds4::stage::ChunkLink>(remote.link, rg, sg.rows_bf16);
            strata::ds4::Model::StageChunks sc;
            sc.send = [&remote](const int32_t* tokens, int T, int64_t p0, bool last, const float* rows, int skip,
                                std::string& e) {
                return remote.chunks->send(tokens, T, p0, last ? strata::ds4::stage::kChunkLast : 0, rows, skip, e);
            };
            sc.recv = [&remote](float* rows, int T, int skip, std::string& e) { return remote.chunks->recv(rows, T, skip, e); };
            sc.wait = [&remote](std::string& e) { return remote.chunks->wait(e); };
            sc.abort = [&remote] {   // (the link shut: a send or a reply on its way fails now; the next reset connects)
                remote.link.abort();
                remote.chunks_done();
            };
            m.set_stage_chunks(std::move(sc));
        }
    }
    if (const char* ff = std::getenv("DS4_FORCE_TOKENS");
        ff != nullptr && *ff != 0 && (serve_mode || !prof_ids.empty() || !spec_ids.empty() || !score_ids.empty()))
        std::fprintf(stderr, "maya-ds4: DS4_FORCE_TOKENS is the CLI decode's (--ids ... --n N): not this mode's, left "
                             "out\n");
    if (serve_mode) return serve(m, eos, spec, o.dspark_conf, o.pack);
    const int window_given = window;
    if (window == 0) window = 16;   // (the scoring / profiling runs; the plain run below takes the model's feed unit)
    if (!prof_ids.empty()) return profile_run(m, prof_ids, window);
    if (!spec_ids.empty()) return spec_eval(m, spec_ids, n, window, o.dspark_conf, spec);
    if (!score_ids.empty()) return score(m, score_ids, score_out.empty() ? score_ids + ".tsv" : score_out, window, std::atoi(eos.c_str()));

    if (!enc_rgb.empty()) {   // the vision tower alone: a picture's span rows (BF16 [rows][dim]) into --dump-image
        std::vector<uint8_t> px((size_t) std::max(0, enc_w) * std::max(0, enc_h) * 3);
        std::FILE* f = std::fopen(enc_rgb.c_str(), "rb");
        const bool got = f != nullptr && !px.empty() && std::fread(px.data(), 1, px.size(), f) == px.size();
        if (f != nullptr) std::fclose(f);
        if (!got) {
            std::fprintf(stderr, "maya-ds4: %s: expected %d x %d RGB bytes (--size WxH)\n", enc_rgb.c_str(), enc_w, enc_h);
            return 1;
        }
        std::vector<uint16_t> span;
        for (int k = 0; k < 2; ++k) {   // (the second: warm)
            const double te = now_ms();
            if (!m.encode_image(px.data(), enc_w, enc_h, span, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "vision: %d x %d pixels -> a span of %zu rows in %.0f ms\n", enc_w, enc_h,
                         span.size() / (size_t) m.dim(), now_ms() - te);
        }
        std::FILE* o2 = enc_out.empty() ? nullptr : std::fopen(enc_out.c_str(), "wb");
        if (o2 != nullptr) {
            std::fwrite(span.data(), 2, span.size(), o2);
            std::fclose(o2);
        }
        return 0;
    }
    if (!ids_s.empty() && ids_s[0] == '@') {   // --ids @file: the ids in a file (a command line is 32 KB on Windows)
        std::ifstream idf(ids_s.substr(1));
        std::stringstream ss;
        ss << idf.rdbuf();
        ids_s = ss.str();
    }
    const std::vector<int> ids = parse_ids(ids_s);
    strata::ds4::ForcedTokens forced;   // (DS4_FORCE_TOKENS: the decode emits a file's ids - ds4_force.hpp)
    if (const char* ff = std::getenv("DS4_FORCE_TOKENS"); ff != nullptr && *ff != 0 && n > 0) {
        if (!forced.load(ff, n, err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "forced tokens: the decode emits the first %d of the %zu ids in %s (plain decode, no "
                             "drafts)\n", n, forced.ids.size(), ff);
    }
    if (!images.empty()) {   // --images: a GENI file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        if (!read_pics(images, pics, err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::vector<strata::ds4::Image> imgs;
        for (const Pic& q : pics) {
            auto rows = std::make_shared<std::vector<uint16_t>>();
            const double te = now_ms();
            if (!m.encode_image(q.rgb.data(), q.width, q.height, *rows, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "picture at %lld: %d x %d pixels, %zu tokens, %.0f ms\n", (long long) q.start, q.width,
                         q.height, rows->size() / (size_t) m.dim(), now_ms() - te);
            imgs.push_back({q.start, rows});
        }
        m.set_images(std::move(imgs));
    }
    std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    std::vector<float> lg;
    if (window_given == 0) window = df != nullptr ? 16 : std::max(16, std::min(m.feed_unit(), 1 << 30));
    const double t0 = now_ms();
    for (size_t a = 0; a < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
        const bool last = a + (size_t) T == ids.size();
        if (!m.feed(ids.data() + a, T, df != nullptr ? T : (last ? 1 : 0), err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
        a += (size_t) T;
    }
    m.set_images({});
    const double t1 = now_ms();
    std::fprintf(stderr, "prompt: %zu tokens in %.0f ms (%.2f tok/s, windows of %d)\n", ids.size(), t1 - t0,
                 1000.0 * (double) ids.size() / (t1 - t0), window);
    const strata::ds4::Stats s0 = m.stats();
    if (s0.eg_rows > 0)
        std::fprintf(stderr, "prompt's Engram rows: %lld, %.1f%% without a read (repeats and the row cache), %lld read\n",
                     (long long) s0.eg_rows, 100.0 * (double) s0.eg_hits / (double) s0.eg_rows, (long long) s0.eg_reads);
    const long long io0 = io_read_bytes();
    std::vector<int> out;
    std::vector<double> tok_ms;   // (DS4_TOKLOG: each token's time since the one before - the latency ledger)
    double t_prev = t1;
    unsigned long long counter = 0;
    long long d_off = 0, d_acc = 0, rounds = 0;
    bool cool = false;
    if (n > 0) {
        int tok = m.sample(temp, 1.0f, temp > 0 ? 64 : 1, 0.0f, cli_seed, counter++);
        if (forced.on()) tok = forced.take(0, tok);
        std::vector<int> got;
        for (;;) {
            out.push_back(tok);
            {
                const double tn = now_ms();
                tok_ms.push_back(tn - t_prev);
                t_prev = tn;
            }
            std::printf("%d,", tok);
            std::fflush(stdout);
            if ((tok == 1 && std::getenv("DS4_NO_EOS") == nullptr) || (int) out.size() >= n) break;   // (DS4_NO_EOS: decode n tokens whatever)
            if (got.empty()) {
                int off = 0, acc = 0;
                const bool try_spec = spec && !cool && !forced.on();
                if (!round(m, tok, n - (int) out.size(), try_spec, o.dspark_conf, temp, 1.0f, temp > 0 ? 64 : 1, 0.0f,
                           cli_seed, counter, got, off, acc, err)) {
                    std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                    return 1;
                }
                d_off += off;
                d_acc += acc;
                ++rounds;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
            if (forced.on()) tok = forced.take(out.size(), tok);
        }
        std::printf("\n");
        if (forced.on())
            std::fprintf(stderr, "forced tokens: %lld of the decode's %zu own picks were not the file's (the first at "
                                 "position %lld)\n", forced.differed, out.size(), forced.first);
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
    }
    if (spec && rounds > 0) {
        const auto& s1 = m.stats();
        const double nv = (double) std::max<int64_t>(1, s1.drafts_rounds - s0.drafts_rounds);
        std::fprintf(stderr, "dspark: %lld rounds, %lld drafts offered, %lld accepted (%.2f tokens a round); drafting "
                             "%.2f ms a round, verifying %.1f ms in all (%lld windows)\n",
                     rounds, d_off, d_acc, (double) out.size() / (double) rounds, (s1.draft_ms - s0.draft_ms) / nv,
                     s1.verify_ms - s0.verify_ms, (long long) (s1.drafts_offered > 0 ? 0 : 0));
        std::fprintf(stderr, "dspark %s\n", per_position(s0, s1).c_str());
    }
    if (std::getenv("DS4_TOKLOG") != nullptr && tok_ms.size() > 8) {   // the decode's tokens after the first (which carries the prompt's tail)
        std::vector<double> v(tok_ms.begin() + 1, tok_ms.end());
        std::sort(v.begin(), v.end());
        std::fprintf(stderr, "token latency ms: p50 %.0f p95 %.0f p99 %.0f max %.0f mean %.0f (%zu tokens)\n", v[v.size() / 2],
                     v[(size_t) ((double) v.size() * 0.95)], v[(size_t) ((double) v.size() * 0.99)], v.back(),
                     std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size(), v.size());
    }
    const double t2 = now_ms();
    const auto& s = m.stats();
    const double nt = std::max<double>(1.0, (double) (s.tokens - s0.tokens));
    std::fprintf(stderr, "decode: %zu tokens in %.0f ms (%.2f tok/s); CPU lane: %lld of %lld experts; the cache moved %lld "
                         "(%lld in all)\n", out.size(), t2 - t1,
                 out.size() > 1 ? 1000.0 * (double) (out.size() - 1) / (t2 - t1) : 0.0,
                 (long long) (s.cpu_experts - s0.cpu_experts), (long long) (s.experts - s0.experts),
                 (long long) (s.swaps - s0.swaps), (long long) s.swaps);
    std::fprintf(stderr, "decode per token: layers %.2f ms (of it: routing waits %.2f, CPU lane %.2f, the rest host launches), "
                         "head wait %.2f, Engram reads %.2f\n",
                 (s.layer_ms - s0.layer_ms) / nt, (s.gpu_wait_ms - s0.gpu_wait_ms) / nt, (s.cpu_ms - s0.cpu_ms) / nt,
                 (s.head_ms - s0.head_ms) / nt, (s.engram_ms - s0.engram_ms) / nt);
    std::fprintf(stderr, "%s", m.rx_lines(s0, nt).c_str());   // (--remote-experts)
    if (s.eg_rows > s0.eg_rows) {   // the Engram rows: how many the row cache served, the reads' time behind the layers
        const double rows = (double) (s.eg_rows - s0.eg_rows);
        std::fprintf(stderr, "Engram per token: %.0f rows, %.1f%% without a read (the row cache holds %lld of %lld), "
                             "%.1f read, the reads %.2f ms from asking to the last one (left to wait: %.2f)\n",
                     rows / nt, 100.0 * (double) (s.eg_hits - s0.eg_hits) / rows, (long long) s.eg_cached,
                     (long long) s.eg_cache_cap, (double) (s.eg_reads - s0.eg_reads) / nt, (s.eg_io_ms - s0.eg_io_ms) / nt,
                     (s.engram_ms - s0.engram_ms) / nt);
    }
    const long long io1 = io_read_bytes();
    if (io0 >= 0 && io1 >= 0)
        std::fprintf(stderr, "decode reads from the disk: %.1f MB a token\n", (double) (io1 - io0) / 1e6 / nt);
#if defined(__linux__)
    if (io0 >= 0 && io1 >= 0 && s.rx_held > 0) {   // (--remote-experts: the server's experts the lane read, the rest)
        const double eb = (double) s.rx_upload_bytes / (double) s.rx_held / 1e6, all = (double) (io1 - io0) / 1e6 / nt,
                     here = (double) (s.rx_here_cold - s0.rx_here_cold) * eb / nt,
                     ahead = (double) (s.rx_ahead_cold - s0.rx_ahead_cold) * eb / nt;
        std::fprintf(stderr, "decode reads from the disk, split (estimates: a whole %.1f MB expert each one sampled out of "
                             "the page cache): the server's experts the lane computed after all %.1f MB a token, the "
                             "server's experts the lookahead asked for early %.1f MB a token (%.1f asked a token), all the "
                             "others %.1f MB a token\n",
                     eb, here, ahead, (double) (s.rx_ahead - s0.rx_ahead) / nt,
                     all > here + ahead ? all - here - ahead : 0.0);
    }
#endif
    if (s.skipped > s0.skipped)   // STRATA_DS4_DISK_SKIP_W
        std::fprintf(stderr, "decode left out %.1f experts a token not in RAM (%.1f%% of the routed)\n",
                     (double) (s.skipped - s0.skipped) / nt,
                     100.0 * (double) (s.skipped - s0.skipped) / std::max<double>(1.0, (double) (s.experts - s0.experts)));
    if (s.store_slots_max > 0) {   // --ssd-cache
        const double h = (double) (s.store_hits - s0.store_hits), m = (double) (s.store_misses - s0.store_misses);
        std::fprintf(stderr,
                     "store: %.1f%% of the CPU lane's experts were in RAM (%.1f hits, %.1f misses a token); read %.1f MB a token "
                     "(demand and guesses), the CPU lane waited %.1f ms a token for them; %lld prefetched, %lld used, %lld given up "
                     "unused; slots %lld of %lld, %lld evictions, %lld failed reads; the GPU computed %.1f experts a token from the store "
                     "(%lld slots visible)\n",
                     100.0 * h / std::max(1.0, h + m), h / nt, m / nt, (double) (s.store_bytes - s0.store_bytes) / 1e6 / nt,
                     (s.store_wait_ms - s0.store_wait_ms) / nt, (long long) (s.store_prefetched - s0.store_prefetched),
                     (long long) (s.store_pf_used - s0.store_pf_used), (long long) (s.store_pf_wasted - s0.store_pf_wasted),
                     (long long) s.store_slots, (long long) s.store_slots_max, (long long) (s.store_evictions - s0.store_evictions),
                     (long long) (s.store_fails - s0.store_fails), (double) (s.store_zc - s0.store_zc) / nt,
                     (long long) s.store_zc_visible);
    }
    if (s.ahead_picks > s0.ahead_picks) {   // lookahead
        const double p = (double) (s.ahead_picks - s0.ahead_picks), c = (double) (s.ahead_cold - s0.ahead_cold);
        std::fprintf(stderr, "lookahead: the guesses a layer before held %.1f%% of the picks (their top k %.1f%%, two layers "
                             "before %.1f%%); file-read picks asked for early %.1f%% (%lld of %lld); %.1f experts asked "
                             "for early a token\n",
                     100.0 * (double) (s.ahead_hit - s0.ahead_hit) / p, 100.0 * (double) (s.ahead_hit_k - s0.ahead_hit_k) / p,
                     100.0 * (double) (s.ahead_hit2 - s0.ahead_hit2) / p,
                     c > 0 ? 100.0 * (double) (s.ahead_cold_hit - s0.ahead_cold_hit) / c : 0.0,
                     (long long) (s.ahead_cold_hit - s0.ahead_cold_hit), (long long) (s.ahead_cold - s0.ahead_cold),
                     (double) (s.ahead_pf - s0.ahead_pf) / nt);
        std::string rk = "lookahead: picked by the guess's rank (1 ..):";
        for (int i = 0; i < 16; ++i) {
            const long long n = (long long) (s.ahead_rank_n[i] - s0.ahead_rank_n[i]);
            if (n <= 0) break;
            char b[24];
            std::snprintf(b, sizeof b, " %.0f%%", 100.0 * (double) (s.ahead_rank_hit[i] - s0.ahead_rank_hit[i]) / (double) n);
            rk += b;
        }
        std::fprintf(stderr, "%s\n", rk.c_str());
    }
    if (s.ep_late > s0.ep_late)   // DS4_EP_DEFER
        std::fprintf(stderr, "ep defer: %lld layers' CPU-lane parts added a layer late (%.1f a token; DS4_PROF's wait for "
                             "the CPU lane is then the next layer's)\n", (long long) (s.ep_late - s0.ep_late),
                     (double) (s.ep_late - s0.ep_late) / (double) std::max<size_t>(1, out.size()));
    if (s.ep_verify_n > s0.ep_verify_n) {   // DS4_EP_DEFER_VERIFY
        // float order (estimated: a few ulp of the terms; the slot measures it); any layer above it is a wiring bug or
        // a non-finite value (inf) - and under VERIFY=2 (the wrong comb) the check must see layers above it
        const float bound = 1e-5f;
        int last = -1, worst = -1, above = 0, seen = 0;
        for (int i = 0; i < 64; ++i)
            if (s.ep_verify[i] != 0.0f) {
                last = i;
                ++seen;
                if (worst < 0 || s.ep_verify[i] > s.ep_verify[worst]) worst = i;
                if (!(s.ep_verify[i] <= bound)) ++above;
            }
        // (VERIFY=2: every layer's add but the head's takes the wrong comb, so all of them but the last must be above)
        const char* verdict = s.ep_verify_wrong ? (above > 0 && above + 1 >= seen ? "the wrong comb seen: the check works"
                                                   : above > 0 ? "the wrong comb seen in some layers only: partly blind, "
                                                                 "unless the others had no CPU-lane part in the run"
                                                               : "the wrong comb NOT seen: the check is blind")
                              : above  ? "MISMATCH (wiring, or inf)"
                              : seen   ? "ok"
                                       : "ok (no layer differed at all: exact, or not checked)";
        std::string ln;
        for (int i = 0; i <= last; ++i) {
            char b[32];
            std::snprintf(b, sizeof b, " %d:%.1e", i, s.ep_verify[i]);
            ln += b;
        }
        std::fprintf(stderr, "ep defer verify: %lld checks of the deferred streams against the exact ones (the same "
                             "attention: float order only); the largest |d| / (1 + |exact|) by the layer whose part it "
                             "was (the last: before the head; since the load):%s; worst %.2e at layer %d; bound %.0e: "
                             "%d of the %d layers that differed above it - %s%s\n",
                     (long long) (s.ep_verify_n - s0.ep_verify_n), ln.c_str(), worst >= 0 ? s.ep_verify[worst] : 0.0f,
                     worst, (double) bound, above, seen, s.ep_verify_wrong ? "DS4_EP_DEFER_VERIFY=2: " : "", verdict);
    }
    if (s.prof_tokens > s0.prof_tokens) {   // DS4_PROF
        const double pt = (double) (s.prof_tokens - s0.prof_tokens);
        double sum = 0;
        for (int i = 0; i < 6; ++i) sum += (s.prof_ms[i] - s0.prof_ms[i]) / pt;
        std::fprintf(stderr, "GPU per token (DS4_PROF): %.2f ms - between layers %.2f, attention %.2f, routing + shared "
                             "expert %.2f, experts gate/up %.2f, waiting for the CPU lane %.2f, down + post %.2f\n", sum,
                     (s.prof_ms[0] - s0.prof_ms[0]) / pt, (s.prof_ms[1] - s0.prof_ms[1]) / pt,
                     (s.prof_ms[2] - s0.prof_ms[2]) / pt, (s.prof_ms[3] - s0.prof_ms[3]) / pt,
                     (s.prof_ms[4] - s0.prof_ms[4]) / pt, (s.prof_ms[5] - s0.prof_ms[5]) / pt);
    }
    if (df != nullptr) std::fclose(df);
    m.save_profile();
    return 0;
}
