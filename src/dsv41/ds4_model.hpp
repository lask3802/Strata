// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_model.hpp - DeepSeek-V4.1-Flash on Maya: an EXL3 model (tools/ds4/pack.py's pack beside its files) over
// the machine's GPUs, layer-split, with the routed experts that do not fit in VRAM on the CPU lane.
//
// Windows of up to 16 tokens through every layer (a prompt in windows, a decode step a window of one): each query of a
// window sees exactly what the reference's decode path gives it (DeepSeek's inference/model.py at start_pos > 0) -
// its own 128-token window, the compressed groups completed by its position.  The state a token leaves is the
// 128-token window ring of every layer, the compressors' partial groups, the compressed KV and index keys of the KV
// source layers (append-only), and the Engram n-gram history; checkpoint() / restore() keep the first two (the rest is
// truncated by position) so a conversation can go back to the end of its last prompt.  A conversation that leaves the
// model altogether (another chat answered in between) is a slot: slot_save / slot_load take the same state - the
// compressed KV and index keys included, which a checkpoint leaves in place - through a file or a host buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4 {

struct Options {
    std::string pack;                 // <model>/maya-ds4
    int64_t max_context = 32768;
    std::vector<int> gpus;            // empty: every visible GPU
    int64_t reserve_mib = 600;        // VRAM left free on each GPU
    double vram_experts = 1.0;        // the share of the expert budget to use (tests: less, so the CPU lane works)
    int cpu_threads = 0;              // 0: the physical cores
    std::string profile;              // expert hit counts (read to place the hottest experts in VRAM, written back)
    bool expert_cache = true;         // experts move between VRAM and the CPU lane as they are used
    // the CPU lane's experts read in place from the model's files, the page cache keeping what fits, instead of
    // copied into RAM: 1 on, 0 off, -1 (auto) where the RAM this process can get does not hold them
    int mmap_experts = -1;
    // the experts the VRAM leaves over are cached in RAM by a store of their own (ds4_store.hpp): whole experts read unbuffered into
    // slots, a replacement policy (0 lru, 1 lfu, 2 lfu then lru), filled on demand and by the lookahead's guesses.  GB of RAM at
    // most (never past the RAM the process can still get less 8 GB); 0: off, the OS page cache as before
    double ssd_cache_gb = -1.0;   // < 0: auto (the RAM the process can get less a floor, at most 48 GB; off below 4 GB), 0: off
    int ssd_policy = 0;
    // dspark_auto: the drafter is loaded only where the routed experts fit the GPUs (more than 1.5x what they have, the verify rounds
    // read 3.3x the bytes of a token and measured no faster than plain decoding, and the drafter's 7 GB of experts take the VRAM the
    // trunk's hottest would have had); --dspark on / off / none says it outright
    bool dspark_auto = true;
    bool dspark = true;              // load the DSpark drafter (the checkpoint's mtp.* blocks) for speculative decoding
    int dspark_block = 0;             // at most this many drafts a round (--dspark-block N; 0: the pack's block size)
    float dspark_conf = 0.5f;         // a draft is verified while its confidence (sigmoid) is at least this
    // tensor parallelism: runs of tp GPUs of one kind (the same memory) take the same layers, each a share of the
    // heads, of the output projection and of every expert's FFN, summed through host-mapped memory (a GPU unlike the
    // next stays alone); tp > 1 runs with threads (--tp 1 --threads off: one host thread, layer split only)
    int tp = 2;
    bool threads = false;             // a host thread per GPU enqueues its layers (the hand-offs host-mapped)
    // prompts in chunks of this many tokens through each layer at once (threads): the GEMMs on the tensor cores, the
    // CPU lane's experts streamed to the GPUs a chunk at a time instead of computed on the host; 0: windows of 16;
    // -1: 2048, or 8192 with the experts read in place (every chunk streams them all from the files); a remote stage's
    // two sides 2048 (the main process sends the worker's when that is smaller: set_stage_chunk)
    int prefill_chunk = -1;
    bool vision = true;               // the vision tower (the checkpoint's vision.* / aligner.*), when it has one
    bool vision_only = false;         // (tests) load the vision tower alone: encode_image() and nothing else
    bool verbose = false;
    // a range of the trunk's layers: this process holds layers [stage_begin, stage_end) only (-1: to the last), the
    // others no device, no weights and no memory - the embedding into the first, the head after the last.  Prompts
    // then run in windows of 16, without DSpark, pictures or slots (tensor parallel and checkpoints stay)
    int stage_begin = 0;
    int stage_end = -1;
    // a remote stage (docs/REMOTE_STAGE.md, src/dsv41/ds4_stage.hpp), a range's two sides on two PCs.  remote_stage:
    // the main process of one, layers [0, stage_end) - its last layer's rows go over the link (set_stage_link) and the
    // worker's normed rows come back for the head (with DSpark: and the drafter's taps of the worker's layers; the
    // drafter is here).  stage_worker: the worker, layers [stage_begin, last] - its rows come from the link
    // (stage_run), its last layer's go back through the final norm; no embedding table, no head, no drafter.
    // Either: conversation checkpoints and slots only with stage_ckpt (the worker as the main process asks); prompts on
    // the chunked path as one process runs them (a chunk's rows over
    // the link: set_stage_chunks / stage_prefill), except with the drafter here: then in windows
    bool remote_stage = false;
    bool stage_worker = false;
    // (either side) the link's rows compact: the streams as bf16, pre and the indexer's picks exact (ds4_stage.hpp
    // rows_pack) - a window's rows and a prompt chunk's.  ds4_main.cpp sets it for either side unless
    // STRATA_REMOTE_ROWS=fp32 (both sides alike: the hello refuses a pair that differs)
    bool stage_bf16 = false;
    // remote_stage: the main process keeps its conversation checkpoints and slots across the link (STRATA_DS4_SLOT_KEEP=1,
    // set_stage_ckpt): each is taken on the worker too, under the main process's id, and used only where both hold it.
    // Off: no checkpoints, every request that does not continue the conversation in place starts from token 0
    bool stage_ckpt = false;
    // remote VRAM experts (ds4_rx.hpp, a proof of concept): [BIND:]PORT this process listens on at its load for an
    // expert server (strata-dsv41 --expert-server-connect), which then computes the CPU lane's picks it holds; empty: off
    std::string remote_experts;
};

/// A picture in the prompt: its span's rows (every position of the span carries the image token), BF16 [span][dim]
struct Image {
    int64_t start = 0;
    std::shared_ptr<const std::vector<uint16_t>> rows;
};

struct Stats {
    int64_t tokens = 0;
    int64_t experts = 0, cpu_experts = 0;   // routed experts run, and how many of them on the CPU lane
    double cpu_ms = 0.0, gpu_wait_ms = 0.0;   // the CPU lane's time; the host's waits for the routing
    double lane_bytes = 0.0;   // the expert bytes that time read (an entry its blob, a slice's share of it: H')
    double layer_ms = 0.0, head_ms = 0.0, engram_ms = 0.0;   // the layers' host time in all; the head's wait; Engram reads
    int64_t eg_rows = 0, eg_hits = 0, eg_reads = 0;   // Engram rows asked for, served without a read, read from the files
    double eg_io_ms = 0.0;                            // the windows' Engram reads, issue to last done (mostly hidden)
    int64_t eg_cached = 0, eg_cache_cap = 0;          // the row cache's rows held, its capacity
    int64_t swaps = 0;                // experts the cache moved into VRAM
    // the cache settles (cache_settle calls) since the load: with the expert server one a prompt (rx::PromptSettle; a
    // chunked run settles each run, so a prompt read in two - pin=N - settles twice); verbose: a line each
    int64_t prompt_settles = 0;
    int64_t skipped = 0;              // routed experts STRATA_DS4_DISK_SKIP_W left out (light, not in RAM)
    int64_t ram_experts = 0, ram_moves = 0;   // CPU-lane experts read from the RAM tier; experts moved into it
    double prof_ms[6] = {};          // DS4_PROF: the GPU's time in decode by stage (between layers, attention, routing +
    int64_t prof_tokens = 0;         // shared expert, experts' gate / up, the wait for the CPU lane, down + post)
    // lookahead (decode windows of one): the picks of layers guessed a layer before (the next layer's gate on this
    // layer's FFN input), of them in the guess's top k / in all its guesses / in the guess two layers before; the picks
    // read from the files, of them prefetched early; experts prefetched early
    int64_t ahead_picks = 0, ahead_hit_k = 0, ahead_hit = 0, ahead_hit2 = 0;
    int64_t ahead_cold = 0, ahead_cold_hit = 0, ahead_pf = 0;
    int64_t ahead_rank_n[16] = {}, ahead_rank_hit[16] = {};   // the next-layer guesses by rank, and how many were picked
    // the RAM store (--ssd-cache), totals since the load: reads asked for and found there (hits), missed (misses), started ahead
    // of use (prefetched) of which used / given up unused, slots given up, bytes read, ms the CPU lane waited for the reads
    int64_t store_hits = 0, store_misses = 0, store_prefetched = 0, store_pf_used = 0, store_pf_wasted = 0, store_evictions = 0;
    int64_t store_bytes = 0, store_fails = 0;
    int64_t store_zc = 0, store_zc_visible = 0;   // experts the GPU computed from the store (zero-copy over PCIe); slots it may read
    double store_wait_ms = 0.0, store_busy_ms = 0.0;
    int64_t store_slots = 0, store_slots_max = 0;
    int64_t drafts_offered = 0, drafts_accepted = 0, drafts_rounds = 0;
    // per draft position (the 1st .. 8th of a round): verified with all the drafts before it accepted, and accepted
    int64_t draft_reached[8] = {}, draft_kept[8] = {};
    double draft_ms = 0.0, verify_ms = 0.0;   // the drafter's time, the verification windows' time
    // remote experts (--remote-experts): requests sent, the CPU lane's picks served remotely, and the server's picks
    // computed here after all (late replies, a lost link: in cpu_experts too), late replies; the lane's wait for the
    // replies after its own part, the round trips (send to reply read), the server's own time and the fallbacks'
    // compute, ms in all; the link's bytes; the load's upload (experts held, bytes, ms)
    int64_t rx_reqs = 0, rx_picks = 0, rx_fallback = 0, rx_late = 0;
    double rx_wait_ms = 0.0, rx_rtt_ms = 0.0, rx_srv_ms = 0.0, rx_fallback_ms = 0.0;
    int64_t rx_bytes_out = 0, rx_bytes_in = 0;
    int64_t rx_held = 0, rx_upload_bytes = 0;
    double rx_upload_ms = 0.0;
    // the link's idle spells after 8 late of 64 (DS4_RX_BACKOFF requests each) and the requests they kept here; the
    // late requests whose replies came after all, those of them read with a later reply (no arrival of their own) and
    // those with their own (rx_late_arr)
    int64_t rx_backoffs = 0, rx_idle = 0, rx_late_came = 0, rx_late_shared = 0, rx_late_timed = 0;
    int64_t rx_late_l[64] = {};   // late replies by layer (the first 64)
    // the server's picks the lane computed after all (kept here by the send gate or an idle spell, late replies, a lost
    // link), and their experts that were not all in the page cache then (Linux: mincore; read from the disk)
    int64_t rx_here = 0, rx_here_cold = 0;
    // the server's experts the lookahead asked for early all the same (madvise WILLNEED: read from the disk when out of
    // the page cache, whoever computes them), and those not all in it then (Linux)
    int64_t rx_ahead = 0, rx_ahead_cold = 0;
    int64_t rx_late_link = 0;   // (late replies past the link's bound, the fixed wait's deadline: rx::Backoff's count)
    // by the window's T (1..16): requests sent, late replies, and windows the send gate kept here (the server's round
    // trip at that T expected to cost more than the lane computing its picks)
    int64_t rx_req_t[17] = {}, rx_late_t[17] = {}, rx_skip_t[17] = {};
    // the link's keep-awake (ds4_rx.hpp): Wakes sent at a window's start, and the CPU lane's requests that found
    // serve_lane napping (its 20 us sleeps: after 2 ms idle, kSpinMs while the link is up)
    int64_t rx_wakes = 0, lane_nap_reqs = 0;
    // expert parallelism H' (DS4_EP_K): the slice picks the server's replies held ([k, nb)), the requests with them and
    // the server's CPU lane's time for them (its replies' b, ms in all), the slice picks whose [k, nb) the lane computed
    // after all (a reply late or lost)
    int64_t rx_split_picks = 0, rx_split_reqs = 0, rx_split_fallback = 0;
    double rx_srv_lane_ms = 0.0;
    // the hybrid (EP U5c, the server's DS4_RX_POOL): the server's W (experts), the W picks its replies held whole, and
    // those the lane computed whole after all (a reply late or lost)
    // those of W the balance kept here (DS4_EP_POOL_BAL: the reply would have come in after this lane's part)
    int64_t rx_pool_w = 0, rx_pool_picks = 0, rx_pool_fallback = 0, rx_pool_kept = 0;
    // the W picks' n_ff blocks the server computed (DS4_EP_POOL_CUT: [cut, nb) of each), and the decode requests (T=1,
    // sent) by the W blocks of [0, k) they moved there (0, 1-3, 4-6, 7-12, 13+): requests, and in ms in all the send to
    // this lane's part done, to the reply's arrival in the kernel (the requests it is known for: rx_wb_na) and this
    // lane's wait after its part
    int64_t rx_pool_blocks = 0;
    int64_t rx_wb_n[5] = {}, rx_wb_na[5] = {};
    double rx_wb_part[5] = {}, rx_wb_arr[5] = {}, rx_wb_wait[5] = {};
    // the sent decode requests (T=1) by their CPU-lane picks (W's among them; 9: 9 or more): requests, the lane's part
    // (ms in all: its own picks' [0, k) and the cut W picks' [0, cut)) and the W blocks of [0, k) they moved there
    int64_t rx_np_n[10] = {};
    double rx_np_ms[10] = {}, rx_np_wb[10] = {};
    // DS4_EP_POOL_CUT_CHECK=1: the requests whose cut W picks' pass was checked against two calls, and those not the same
    int64_t rx_cut_chk = 0, rx_cut_bad = 0;
    // DS4_RX_FILL=1 (EP U5b): fills queued, loaded by the server, refused; their bytes sent and the time in the paced
    // sends; the requests the server answered Stale (an expert mid-fill)
    int64_t rx_fills = 0, rx_fill_ok = 0, rx_fill_fail = 0, rx_fill_bytes = 0, rx_stale = 0;
    double rx_fill_ms = 0.0;
    // DS4_EP_SPLIT_VERIFY: the rows checked (a reply's slice sum against this lane's), the largest |d| / |ref| by layer
    // since the load; =2: the server's slice wrong on purpose (the check must see it)
    int64_t rx_split_n = 0;
    float rx_split_err[64] = {};
    bool rx_split_wrong = false;
    int64_t ep_late = 0;   // DS4_EP_DEFER: the layers whose mailbox part the next layer (or the head) added
    // DS4_EP_DEFER_VERIFY: the checks made (a layer's deferred streams against the exact ones), and by the layer whose
    // part it was the largest relative difference since the load (the last layer's: before the head)
    int64_t ep_verify_n = 0;
    float ep_verify[64] = {};
    bool ep_verify_wrong = false;   // DS4_EP_DEFER_VERIFY=2: the wrong comb on purpose (the check must see it)
};

class Model {
public:
    Model();
    ~Model();
    bool load(const Options& o, std::string& err);

    /// Feeds `token` at position pos(); with logits, the next token's logits are left on the head's device.
    bool step(int token, bool logits, std::string& err);
    /// Feeds n tokens (windows of up to 16, or a long run in chunks of prefill_chunk); the last `logits_rows` of them get
    /// logits (logits_rows > 16: only a run of one chunk or window).
    bool feed(const int* tokens, int n, int logits_rows, std::string& err);
    /// feed() of a prompt's end that also keeps the state before its last token in checkpoint `slot` (a prompt's
    /// last token then rides along in the run of chunks instead of a window of its own); the same as
    /// feed(n - 1), checkpoint(slot), feed(1)
    bool feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err);
    /// The next token from the last feed's last logit row (greedy when temperature <= 0).
    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);
    /// The last feed's logit rows on the host (rows x vocab floats).
    bool logits(std::vector<float>& out);

    /// The DSpark drafter (speculative decoding): its block size (0: not loaded), drafts for the positions after `y`
    /// (y: the token at pos(), not fed yet) with their confidences, and the verification of [y, drafts]: how many of the
    /// drafts stand and the token after them (the state then holds y and the accepted drafts).  The drafts are the
    /// drafter's argmax; a sampled request passes its chain (the same temperature / top_p / top_k / min_p / seed /
    /// counter as its verify()) so that with DS4_DSPARK_SAMPLE=1 they are drawn from the drafter's distribution, as
    /// DeepSeek's reference does.  verify() keeps a draft d with probability min(1, p(d) / q(d)) over the request's
    /// chain - the speculative-sampling rule, so every token is still distributed as the model's own draw.
    int draft_block() const;
    bool draft(int y, std::vector<int>& drafts, std::vector<float>& conf, std::string& err, float temperature = 0.0f,
               float top_p = 1.0f, int top_k = 0, float min_p = 0.0f, uint64_t seed = 0, uint64_t counter = 0);
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err);

    /// Tokens a feed call takes at a time to use the chunked prompt path (all of a prompt: its chunks run as a
    /// pipeline; 16 without it).
    int feed_unit() const;
    /// Called from feed() while a run of chunks goes through the layers, with the tokens done so far.
    void set_progress(std::function<void(int64_t)> f);
    /// Asked while a run of chunks goes through the layers: true (a Stop) ends it early - one GPU after the layer it is
    /// on, several at a chunk all of them reach - and feed() returns false with cancelled() true.  The state past the
    /// prompt's start is then not usable: restore() a checkpoint kept before the feed, or reset().
    void set_cancel(std::function<bool()> f);
    bool cancelled() const;   // the last feed() / feed_mark() ended early on a Stop
    int64_t pos() const;
    int64_t max_context() const;
    int vocab() const;
    int dim() const;
    const std::vector<int32_t>& tokens() const;   // what the state has consumed, in order

    void reset();
    /// Keep the state at the current position in checkpoint slot 0 or 1 (the end of a prompt, the position before it);
    /// restore(limit) goes back to the latest one at or before limit (-1: none).
    void checkpoint(int slot = 0);
    bool restore(int64_t limit);
    int64_t restorable(int64_t limit) const;
    /// Checkpoint 1 also becomes the anchor (kept until a restore goes before it): the position before the end of the
    /// first prompt of a run that continued in place, where a re-rendered history (earlier reasoning dropped) splits.
    void pin();
    /// The positions restore() can go back to, latest first (a slot keeps them: it can be taken back at any of them).
    std::vector<int64_t> check_positions() const;

    /// The conversation in the model into `out` (slot_save): its position, its tokens, every window ring (the DSpark
    /// blocks' included), the compressors' partial groups and each KV source's compressed KV and index keys up to it,
    /// the pieces the state a prompt continues is made of.  0 and err: nothing was set aside (it holds pictures, there
    /// is no conversation, or the conversation is already set aside - unchanged since).
    /// `remote_id` (a remote stage's main process, Options::stage_ckpt; non-zero): the worker's layers' part of the slot
    /// is taken first, under that id, and the slot is this process's layers' part - 0 and err when the worker did not
    /// keep it.
    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err, int64_t remote_id = 0);
    /// `data` (a slot's bytes) back into the model: the state at the slot's position n_pos, as slot_save took it -
    /// what a prompt that continues the slot's tokens reads on from (the caller checked the tokens are a prefix).
    /// `remote_id`: the worker puts its part back first.  False when it does not hold it (nothing changed on either
    /// side); after a link or worker failure, or when this process's own load fails past that point, the conversation
    /// is reset (both sides).
    bool slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err, int64_t remote_id = 0);
    /// What a saved conversation must match: this engine's version, this model's geometry and files and the drafter's
    /// blocks - a slot naming anything else is not this model's state (a file's .meta says so; slot_load refuses).
    std::string slot_identity() const;

    /// The vision tower: the token every position of an image's span carries (-1: no tower).
    int image_token() const;
    /// An image's pixels (RGB rows, width x height, both multiples of the patch size: the server's resize) -> the
    /// rows of its span, BF16 [n_h * (n_w + 1) + 2][dim]: image_start, n_h rows of (n_w aligner rows, image_newline),
    /// image_end.
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err);
    /// The pictures of the prompt about to be fed ({}: none): positions inside a span read its rows (and route with
    /// the image bias, and take no part in the Engram n-grams); the rest the embedding table's.
    void set_images(std::vector<Image> images);

    /// A layer range (Options::stage_begin / stage_end): a slot would hold only some layers' state, so there are none
    /// (slot_save / slot_load refuse) - except the two ends of a remote stage that keep checkpoints (stage_ckpt_on), where
    /// each holds its own layers' part.
    bool ranged() const;
    /// A remote stage's main process that keeps checkpoints and slots across the link, or a worker (which does as it
    /// is asked): checkpoint / restore / pin / slot_save / slot_load work on this process's layers.
    bool stage_ckpt_on() const;
    /// The main process's end of that (Options::stage_ckpt): the worker takes (`save`) or puts back (`restore`) the
    /// state of an id, at position L (the worker is there on a save).  The ids are ds4_stage.hpp's: a checkpoint's role,
    /// the pin, a restore at L, or a slot's.  `stored` false (a save): the worker did not keep it; `missing` (a
    /// restore, false returned): it does not hold it and nothing changed there.  Any other failure: the worker's
    /// session is lost - the conversation is reset (a checkpoint: the next window fails until it is).
    struct StageCkpt {
        std::function<bool(int64_t id, int64_t L, bool& stored, std::string& err)> save;
        std::function<bool(int64_t id, int64_t L, bool& missing, std::string& err)> restore;
    };
    void set_stage_ckpt(StageCkpt c);
    /// What a remote stage's hello carries: the trunk's geometry, this process's layers [layer_begin, layer_end), the
    /// widest window and a hand-off row's floats - the streams (hc x dim), pre (8), the indexer's picks (topk_stride
    /// int32) and the candidate blocks (cand_stride int32).  A window's T rows cross block by block: T x hc x dim
    /// floats, then T x 8, then T x topk_stride, then T x cand_stride (T x handoff_floats in all).  The worker's reply
    /// has the same layout; the main process reads its streams block (ds4_stage.hpp's reply_tap_at).  With stage_bf16
    /// the rows on the link are the compact ones (ds4_stage.hpp RowGeo::compact(): handoff_floats words a row;
    /// full_floats the fp32 row) and the layout is: the streams as bf16, pre, the picks and candidates as they are.
    /// chunk: the most rows of a prompt chunk this process sends (the main process) or takes (the worker) over the link
    /// - its chunked prompt path's chunk, rows of handoff_floats on the link as a window's; 0: windows only.
    struct StageGeo {
        int dim = 0, hc = 0, n_layers = 0, n_expert = 0, layer_begin = 0, layer_end = 0, max_t = 0;
        int64_t handoff_floats = 0;
        int64_t full_floats = 0;   // (an fp32 row) hc x dim + 8 + topk + cand
        int64_t topk = 0, cand = 0;   // int32 a row
        bool rows_bf16 = false;
        int chunk = 0;
    };
    StageGeo stage_geo() const;
    /// The main process's link (remote_stage).  `run`: a window's T tokens at p0 and its rows (`out`) to the worker,
    /// the worker's reply back into `in` (T x handoff_floats each; ds4_stage.hpp's reply_tap_at layout).  `commit`:
    /// how many of the window's rows the worker keeps - all of a window kept whole, right after its run; a draft's
    /// check (verify) the ones it accepted, after the verdict.  `reset`: the worker's session from position 0, as
    /// reset() starts this one (a failure is kept: the next window at position 0 tries it again, any other fails).  A
    /// failed run or commit leaves the conversation unusable until reset().
    using StageRun = std::function<bool(int T, const int32_t* tokens, int64_t p0, const float* out, float* in,
                                        std::string& err)>;
    void set_stage_link(StageRun run, std::function<bool(int n_keep, std::string& err)> commit,
                        std::function<bool(std::string& err)> reset);
    /// The worker's side (stage_worker): a window's rows land in stage_rows_in() (pinned, kWin x handoff_floats);
    /// stage_run() runs this process's layers on them at p0 (which must be pos()) and leaves the reply in
    /// stage_rows_out(): the last layer's rows through the final norm, then the drafter's taps of the target layers
    /// here.  A window of 2 or more rows keeps its layers' backups: stage_rollback() then keeps its first n_keep rows
    /// and undoes the rest (a draft's check the main process rejected part of), as verify() does in one process.
    float* stage_rows_in();
    float* stage_rows_out();
    bool stage_run(const int32_t* tokens, int T, int64_t p0, std::string& err);
    bool stage_rollback(int n_keep, std::string& err);
    /// The compact rows' conversions so far (stage_bf16): the time and the windows converted, rows in = fp32 rows from the
    /// link's, rows out = to the link's (the main process: its window's rows out, the reply's back in; the worker: the
    /// reverse).  Each count is of its own conversions, so a mean divides by its own.
    struct StageConv {
        double in_ms = 0, out_ms = 0;
        int64_t in_n = 0, out_n = 0;
    };
    StageConv stage_conv() const;
    /// The main process's prompt chunks (remote_stage, StageGeo::chunk > 0; ds4_stage.hpp's ChunkLink): each chunk's
    /// rows go to the worker as the chunk leaves this process's last layer, while the next chunk runs here.  `send`:
    /// T tokens at p0 and their rows (T x full_floats, block by block as a window's; the compact ones are ChunkLink's
    /// to make), on a thread of its own - the rows stay as they are until the next send or `wait` returns; `last`: the
    /// run's last chunk; `skip`: the rows the reply leaves out.  `recv`: the oldest chunk's reply, rows [skip, T) row
    /// by row into `rows` (that chunk's row 0; row r at r x full_floats - its streams, then its pre, fp32).  `wait`:
    /// the last send out.  `abort`: a run that failed
    /// with chunks on the way - the link is shut, and the next reset connects again.
    struct StageChunks {
        std::function<bool(const int32_t* tokens, int T, int64_t p0, bool last, const float* rows, int skip,
                           std::string& err)> send;
        std::function<bool(float* rows, int T, int skip, std::string& err)> recv;
        std::function<bool(std::string& err)> wait;
        std::function<void()> abort;
    };
    void set_stage_chunks(StageChunks c);
    /// (the main process, once a link is made) its prompt chunks of at most n rows - the worker's when that is smaller
    /// than this process's own (stage::chunk_sent); StageGeo::chunk stays this process's own, what its hello says
    void set_stage_chunk(int n);
    /// The worker's side of a prompt chunk (stage_worker, StageGeo::chunk > 0): T tokens at p0 (which must be pos())
    /// and their rows `in` (block by block) through this process's layers on its chunked prompt path; its last layer's
    /// rows [skip, T) row by row into `out` (StageChunks::recv's layout).  `last`: the main process's last chunk of a run.
    /// With stage_bf16 `in` and `out` are the link's compact rows (rows_pack's, reply_rows_pack's): the chunk runs on
    /// two fp32 buffers of its own, unpacked into and packed from here.
    bool stage_prefill(const int64_t* tokens, int T, int64_t p0, bool last, int skip, const float* in, float* out,
                       std::string& err);
    /// The worker's prompt-chunk buffers the link lends (StageBuffers::pf_in[0], pf_in[1], pf_out[0], pf_out[1] for i
    /// 0-3): StageGeo::chunk x handoff_floats each (the link's row), pageable, its pages committed as they are first
    /// written; nullptr without chunks.
    float* stage_chunk_rows(int i);

    void save_profile() const;
    const Stats& stats() const;
    /// (--remote-experts) the remote experts' lines since s0 (per token of nt): picks served there, the round trips
    /// and the lane's waits (mean, p50, p99), bytes; "" without an expert server
    std::string rx_lines(const Stats& s0, double nt) const;
    std::string describe() const;
    /// INFO / VRAM lines for the server's Monitor.
    std::vector<std::string> info_lines() const;

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::ds4
