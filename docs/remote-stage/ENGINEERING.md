# Remote stage: design, process and measurements

How the layer split across PCs works inside, how it got to its present speed, every number measured on it, and what
is still open. Using it: [../REMOTE_STAGE.md](../REMOTE_STAGE.md). Code: `include/strata/net/stage_link.hpp` (the
protocol), `src/net/stage_link.cpp` (the sockets), the `remote_main` / `stage_worker` / `stage_relay` parts of
`src/program/generate.cpp`, `Verifier::set_remote` / `set_no_head` (`src/core/verify.cpp`), `Prefill::remote_send` /
`remote_recv` (`src/prefill/prefill.cpp`), `ArenaExpertSource::set_layer_range` (`src/core/expert_source.cpp`), and
`tools/stage_node.py`, `tools/stage_tune.py`, `tools/stage_ship.py`.

## Design

### Where the network sits

An in-process layer split already hands a token from one stage to the next once per verify window (a pinned,
mapped buffer of T rows that the next stage reads) and hands each prompt chunk's rows to the next stage on a
`std::async` task while the first stage reads the next chunk. The remote stage puts the network exactly there. No
CUDA graph, kernel or per-layer path changes; only the object at the end of the hand-off is a TCP link instead of the
next stage.

The split chosen: the **main process** runs layers [0, K), the output head, sampling and the draft (MTP) layer; the
**worker** runs [K, 48) and sends back the residual after its last layer instead of running the head. A window
therefore goes main -> worker -> main. This keeps everything a request touches besides plain layers on the main PC
(the API, sampling, the drafter, the PLE table that layer 1 reads, penalties, logprobs); the worker has no notion of a
request. The price is the rows coming back every window (as large as the rows going out). The alternative of putting
the head and the drafter on the worker would have saved that return trip and moved most of the request state.

### The protocol

One TCP connection per main process. Every message is a 40-byte header (`magic` "STRM", `type`, three int64 fields
`a b c`, payload `bytes`) and a payload. Protocol version 3.

| message | header fields | payload | reply |
|---|---|---|---|
| Hello | | `StageHello` (below) | HelloOk with the worker's hello, or Error |
| Run (a verify window) | a = T, b = first position | int32 tokens[T], fp32 rows[T x 12,804] | RunOk: a = the worker's time (us), rows[T x 12,804] |
| Commit | a = accepted count | none | none (an error comes back on the next reply) |
| Prefill (a prompt chunk) | a = T, b = first position, c = flags (bit 0: the prompt is one chunk) and `skip` << 8 | int64 tokens[T], fp32 rows[T x 10,240] | PrefillOk: a = the worker's time (us), rows[(T - skip) x 10,240] |
| Reset (a fresh prompt) | | none | ResetOk |
| Error | | the message | |

**The hello** carries what both sides must agree on, and each side checks the other's: the protocol, `n_embd`, `hc`,
`n_layers`, `n_expert`, the first remote layer (K), the floats per window row, the K/V element type, and a
fingerprint of the pack (FNV-1a over the pack's `index.txt` and `native_experts.txt`). The worker also refuses a main
process whose prompt chunk, context or window is larger than its own, and a wrong `STRATA_STAGE_TOKEN` (the token is
the hello's last field; the worker's own hello carries none). A peer that sends no hello within 10 s is dropped; after
the hello the worker waits for messages without a time limit (the next prompt may be hours away), while sends and the
main process's receives time out after `STRATA_REMOTE_TIMEOUT_S` (300 s). Sockets: `TCP_NODELAY`, 8 MiB buffers,
keepalive after 30 s idle, 3 probes 10 s apart.

**Bytes, Qwen3.8-Flash-Next** (n_embd 2560, 4 hyper-connection streams, everything fp32):

- a window row is `hc * n_embd + n_embd + hc` = 12,804 floats = **51,216 B**: the four residual streams, the pending
  write of the last layer and the inject. A window of T tokens sends T rows and T token ids and gets T rows back.
  With `--spec 4`, windows of 1 to 6 tokens were captured: up to ~307 KB each way.
- a prompt row is `hc * n_embd` = 10,240 floats = **40,960 B** plus an 8-byte token id. A 5888-token chunk is 241 MB
  out. Only the rows the drafter reads come back: the drafter attends to the last `--mtp-window` (32,768) cells, so
  the main process asks for rows from position n - 32,768 - 128 on (`Prefill::set_remote_rows_from`). For the 90,921-
  token prompt that is 3.72 GB out and 1.35 GB back; for a prompt shorter than the window, every row comes back.
  Without a drafter no prompt rows come back.
- a commit is the header alone.

### Pipelining the prompt

The first version ran each chunk's send, the worker's read and the reply one after the other. Now:

- **main process:** a chunk's rows are copied into one of two pinned hand-off buffers and sent on a task, in order,
  one send at a time; the replies are taken on another task chain in the same order (receive into a pinned buffer,
  upload, then the usual per-chunk callback that builds the drafter's K/V). The main process reads chunk c + 1 while
  chunk c is on the wire or in the worker.
- **worker:** a receiving thread reads whole messages and puts a chunk's rows straight into one of two pinned input
  buffers (it waits when both are in use); the serving thread handles one message at a time in arrival order; a
  chunk's reply goes out on its own thread from one of two output buffers while the serving thread reads the next
  chunk.

### The window path

- **main:** stage 0 (`ver`, layers [0, K)) writes its hand-off as for an in-process split. The next "stage" is a
  head-only verifier (`set_stage(n_layers, -1)`) whose `set_remote(run, commit)` hooks send the window to the worker
  and wait for its rows before running the head; its commit sends the accepted count (one way) before committing
  locally.
- **worker:** a verifier over [K, 48) with `set_no_head(true)`: after the last layer it folds that layer's pending
  write into the residual (as the unsplit window does before the head) and writes the hand-off instead of running the
  head. Before a window it gives back the prompt path's loan from its cache (below) and applies the adaptive tier's
  pending swaps; after every `--adapt-every` commits it runs the adaptive tier on its own layers.

Decode is sequential: a window's time is the main process's layers, head and drafter, plus the round trip, plus the
worker's layers. A second PC adds VRAM for the expert cache and a second RAM pool; it does not run at the same time
as the first on one conversation.

### More than two stages (relay workers)

A relay worker (`--stage-end K2 --stage-next HOST:PORT`) runs [K, K2) and is a client of the next worker as the main
process is its client: windows go on synchronously and the next worker's rows are this worker's reply; prompt chunks
go on through a `StageRelay` (one send in flight, in order), and the next worker's reply to a chunk is read and
forwarded on the replying thread. A reset and a commit go on to the next worker first. The main process sees one
worker. A relay worker connects to the next one when it starts (30 tries, 2 s apart) and again at a reset if that
link broke.

### Each process holds only its own layers

| what | how | log line |
|---|---|---|
| experts in RAM | `ArenaExpertSource::set_layer_range(lo, hi)`: one contiguous range of the arena layout is allocated, pinned and read from the GGUF; the base pointer is shifted back by the range's first byte, so every `blob_offset()` of a layer in range is unchanged; `blob()` is null outside it | `expert arena: layers 0-26 only (29.99 of 53.32 GiB; remote stage)` |
| dense weights | the tensors of other layers are skipped, as `STRATA_STAGE_TRIM=1` does for an in-process split | `the dense weights of layers 0-26 only` |
| expert cache | the profile is cut to the process's layers before the cache is filled | `13824 of the 24576 pairs are this process's layers` |
| session | `session_init(lo, hi)`: only its own layers' K/V and GDN state | |
| PLE | the worker has no PLE run (layer 1 always runs in the main process, since K >= 2) | `no PLE here` |
| head | not loaded on the worker | |

**Layer sizes by pack** (the experts only, from each pack's `native_experts.txt`: blob bytes x 512; computed, not a
runtime measurement; each layer's dense tensors come on top):

| pack | experts per layer | all 48 layers | heavier layers |
|---|---:|---:|---|
| RVN IQ3_S | 1.111 GiB | 53.32 GiB (the engine's log agrees) | none |
| Unsloth UD-IQ4_XS | 1.111-1.660 GiB | 55.43 GiB | 2, 4, 30, 46, 47 |
| Unsloth UD-Q4_K_XL | 1.465-1.904 GiB | 71.73 GiB | 2, 4, 30, 46, 47 |

So a split of a UD pack is balanced by bytes, not by layer count (the tuner in progress does that). In the RVN IQ3_S
GGUF, shard 1 holds layer 0 with the token embedding and the output head, shard 2 the PLE table (26.8 GB), and
shards 3-8 layers 1-47 (from the shards' headers): a worker needs its layers' part of shards 3-8, the small non-layer
tensors and every shard's header, which is what `tools/stage_ship.py` sends (not yet tried end to end).

The range needs the loader that reads a native pack's experts from its GGUF: a pack with `experts.bin`, a shared
arena or `STRATA_ARENA_MMAP=1` would need offsets the range does not have, so the first two refuse at start and the
third is ignored for a range. `--mmap-experts` uses another expert source that gets no range (it maps the whole pack;
not tested).

### Lending and refilling on the worker

On one card the prompt path borrows expert-cache slots for its buffers and refills them before decode. The worker does
the same: at a prompt's first chunk it lends slots for the main process's chunk size, and the first window after the
prompt refills them. The lend and refill code was moved out of the request loop so the worker's handlers can call it;
the main process uses it as before.

### Timing lines

With `STRATA_REMOTE_TIMING=1`: per chunk, a sending stage's own time (`strata prefill: layers A-B, chunk T=... own N
ms, waited M ms for the previous send`), the main process's view of each reply (`from its send to its rows = worker
+ link and queue`) and the worker's own time per chunk; every 64 windows, the round trip split into worker and link,
the main process's own share between windows, and on a worker its own layers' and the next worker's share. The worker
puts its own time in every reply's header, so the main process can split a round trip without synchronized clocks.

## How it got here

All on the rig below, RVN IQ3_S, K=24, the 32K prompt, chunk 2048 (the host's CI load during these runs was heavy,
100-200% CPU in Unity builds, and not recorded per run):

| version | prefill tok/s | decode tok/s | 3080 decode hit rate |
|---|---:|---:|---:|
| first: chunks one after the other, adaptive tier off in both processes | 448 | 38.4 | 55% |
| send and receive pipelined | 554 | 38.8 | - |
| + the worker replies on its own thread, + the adaptive tier on in both processes | 754 | 49.3 | 77-79% |
| the 3080 alone, the same chunk | 444 | 37.6 | 59-60% |

Timing at that point (K=24, chunk 2048, 32K): the worker needed 2.4-2.9 s per chunk, the slow stage of the prompt; a
decode window took ~53 ms, of which the round trip was ~26 ms (worker 23 + link 3).

What went wrong on the way, and why:

- **The adaptive tier was off.** The first versions set `--adapt-every 0` for a remote stage (each process's tier
  swaps against its own cache only, and that was not wired up yet). The caches then kept only what the profile put
  there: 55% hits on the 3080. With the tier on in both processes, 77-79%. The decode gain in that step (38.8 ->
  49.3) came with the hit rate; the worker's threaded replies, in the same step, act on the prompt path.
- **Chunks in series.** Send, worker, reply and the main process's next chunk did not overlap: 448 tok/s. Pipelining
  them gave 554, and the worker replying on its own thread 754 (with the tier).
- **2048-token chunks.** With every split point the split read long prompts slower than the 3080 alone at its auto
  chunk (best 846 vs 1,120 tok/s at 100K). The time per layer of a chunk grows much more slowly than the chunk (on
  the 3080, 96 ms at 2048 tokens and 125 ms at 8192), so small chunks pay it more often (table below). 5888 and 8192
  fixed it.
- **The worker's own prompt buffers.** The worker first allocated its prompt path's buffers next to its cache; it now
  borrows them from its cache and refills them before the next window, as the main process does (commit 5c2247e).
  Not measured on its own.
- **A restart that did not wait.** The first K=28 run found only 496 MiB of VRAM on the 2080 Ti: the previous worker,
  with ~30 GB of pinned experts, was still unpinning (it takes more than 30 s). The lab's worker script now waits for
  the process to exit and the VRAM to come back; the node agent waits for the process to exit (up to 180 s).
- **8192-token chunks without a reserve.** The 3080 had 65 MiB of VRAM free after loading, and the long prompts
  failed with `verify: instantiate: out of memory` or an engine exit (the server restarted the engine for the next
  request, which failed the same way). With `--vram-reserve-mib 1200` it ran. The tuner now retries a chunk with the
  reserve the engine's start-up line asks for.
- **A review pass** (commit a0c4279) added the token, `--stage-bind`, the pack fingerprint, the 10 s hello timeout,
  keepalive, a clean shutdown of a broken link, refusals of control vectors and helper-GPU caches, and `--kv-grow`
  off (the commit message says refused; the code turns it off without an error). The token refusal and the hello
  timeout were probed: a wrong token got `Error "wrong token"`, a silent peer was closed after 10.3 s.

## Measurements

### The rig

| | main PC | worker PC |
|---|---|---|
| GPU | RTX 3080 10 GB (sm_86), PCIe 4.0 slot negotiating **x8** (a damaged slot or card, accepted), probe 13.0 GB/s | RTX 2080 Ti 11 GB (sm_75), PCIe 3.0 x16, probe 13.2 GB/s |
| CPU | Ryzen 9 5900XT, 16 cores / 32 threads (AVX2, no AVX-512) | Ryzen 9 5900X, 12 cores / 24 threads (AVX2, no AVX-512) |
| RAM | 4 x 32 GB DDR4-3200 | 4 x 16 GB DDR4-2400 |
| OS | Ubuntu 24.04 LXC container (32 cores, 96 GB limit) on Proxmox VE 9.2, kernel 7.0.2 | Proxmox VE 9.0 host, kernel 6.14.8 (the worker ran on the host) |
| other load | the host ran CI jobs (Unity builds, a VM) during the runs; recorded as "other CPU %" | none recorded |

Link: the worker PC's Intel I211 1 GbE and the main PC's I225-V 2.5 GbE through a switch; 117 MB/s measured (a
64 MiB HTTP POST from the main PC to the worker's node agent), ping 0.13-0.56 ms (mean 0.40). One engine build
(CUDA 13.0, driver 580.178.04 on the main PC, sm_75 + sm_86) on both.

Model and settings: RVN-Qwen3.8-Flash-Next IQ3_S (8 GGUF shards; a native pack from `tools/iq_pack.py`, experts read
from the GGUF), `--expert-cache auto --spec 4 --spec-min-p 0.5` with the stock draft layer, `--max-context 131072
--kv int8 --kv-resident 32768`, no conversation cache. Requests greedy, reasoning off. "Short decode": the mean of
nine short prompts (code, prose, Chinese; 200-256-token answers). The long prompts: 7,521 / 30,167 / 57,383 / 90,921
tokens of English text with a 320-token answer; speeds from the engine's own `timings`. One run per row. "Other CPU
%": the summed CPU of the busiest non-Strata processes on the main PC's host (top, every 10 s; 100 = one core).

### The sweep (2026-10-09, 11:33-12:41 UTC)

| configuration | short decode | 8K prefill / decode | 32K | 64K | 100K | other CPU % |
|---|---:|---:|---:|---:|---:|---:|
| 3080 alone, `--prefill auto` (5888) | 39.9 | 791 / 38.1 | 1,118 / 37.2 | 1,146 / 39.0 | 1,120 / 38.0 | 34 |
| K=20, chunk 2048 | 51.7 | 491 / 53.2 | 604 / 50.0 | 609 / 52.1 | 600 / 52.0 | 29 |
| K=24, chunk 2048 | 51.3 | 593 / 54.9 | 771 / 50.7 | 764 / 48.1 | 750 / 47.7 | 103 |
| K=26, chunk 2048 | 50.7 | 629 / 51.8 | 816 / 48.7 | 865 / 47.9 | 846 / 47.7 | 109 |
| K=28, chunk 2048 | 50.2 | 607 / 48.0 | 761 / 46.0 | 798 / 46.1 | 812 / 45.3 | 19 |
| K=26, chunk 4096 | 52.3 | 740 / 50.1 | 1,211 / 47.8 | 1,330 / 49.2 | 1,328 / 45.6 | 30 |
| K=24, chunk 5888 | 54.0 | 652 / 51.1 | 1,226 / 50.2 | 1,381 / 50.6 | 1,374 / 47.8 | 26 |
| K=26, chunk 5888 | 53.5 | 655 / 48.1 | 1,231 / 49.7 | 1,474 / 47.7 | 1,521 / 45.6 | 18 |
| K=28, chunk 5888 | 49.4 | 690 / 47.7 | 1,217 / 45.9 | 1,503 / 45.9 | 1,591 / 45.7 | 12 |
| K=26, chunk 8192, `--vram-reserve-mib 1200` | 52.5 | 589 / 46.0 | 1,225 / 47.5 | 1,485 / 47.2 | 1,633 / 45.4 | 44 |
| K=26, chunk 8192, no reserve | 58.4* | out of VRAM | | | | 10 |

\* three code prompts only (the run failed at the next one); code decodes fastest, so it is not comparable to the
nine-prompt means above.

The 3080's decode hit rates at chunk 2048: K=20 86.7%, K=24 81.3%, K=26 77.5-80.6%. The first K=28 attempt failed
(the restart above); the K=28 row is the second.

Request time, a nominal 8,000 / 32,000 / 100,000-token prompt plus a 500-token answer (prompt / prefill + 500 /
decode, computed from the table): 3080 alone 23.2 / 42.1 / 102.4 s; K=24 at 5888 22.1 / 36.1 / 83.2; K=26 at 5888
22.6 / 36.1 / 76.7; K=28 at 5888 22.1 / 37.2 / 73.8; K=26 at 8192 with the reserve 24.5 / 36.6 / 72.3; K=26 at 2048
22.4 / 49.5 / 128.7.

### The time per layer of a prompt chunk

From the timing lines (each stage's own time for a full chunk, divided by its layers); different runs and splits:

| chunk | 3080, ms per layer | per token per layer | 2080 Ti, ms per layer | per token per layer | source |
|---:|---:|---:|---:|---:|---|
| 2048 | ~96 | ~47 us | ~112 | ~55 us | the sweep, K=24 |
| 4096 | 89 | 22 us | 120 | 29 us | tuner, K=22 |
| 5888 | 106 | 18 us | 130-134 | 22-23 us | tuner, K=26 and K=27 |
| 8192 | 125 | 15 us | 160 | 20 us | tuner, K=28 with the reserve |

A layer's experts are 1.11 GiB (RVN IQ3_S); streaming that over a ~13 GB/s link takes ~85-92 ms (an estimate from
the probe's bandwidth), close to the 2048-token time. That fits a prompt path that streams most of each layer's
experts once per chunk whatever its size, which is why the chunk matters more than the split point. The 3080 alone
at `--prefill auto` already used 5888; whether `--prefill auto:32768` would raise it was not measured (by the auto
rule it is held by the VRAM the cache can lend, so probably not).

### Where the decode time goes

- K=24, chunk 2048, 32K: a window ~53 ms; the round trip ~26 ms = worker 23 + link 3.
- K=27, chunk 5888, ~30K (tuner): the round trip 18-21 ms = worker 15.5-17.9 + link 2.8-3.2; the main process
  between windows (head, drafter, layers 0-26) 26-46 ms. The tuner's per-stage window times over all its runs: the
  3080's share 23-37 ms (22-28 layers), the 2080 Ti's 15-20 ms (20-26 layers).

The link is ~3 ms of a 50-60 ms window at 1 GbE.

### The tuner's first run (2026-10-09, 12:51-13:04 UTC)

`tools/stage_tune.py tune`, two nodes (the 3080 main process, the 2080 Ti worker), goal: a 32,000-token prompt
(29,802 tokens as tokenized) and a 500-token answer, chunks 4096 / 5888 / 8192, at most 9 runs:

| stages | chunk | prefill | decode | short decode | request s | per stage: chunk ms / window ms | other CPU % |
|---|---:|---:|---:|---:|---:|---|---:|
| 0-21, 22-47 (by VRAM) | 4096 | 1,035 | 49.9 | 51.6 | 40.9 | 1,964 / 23.3; 3,132 / 20.1 | 25 |
| 0-27, 28-47 (balanced) | 4096 | 1,174 | 41.5 | 46.1 | 39.3 | 2,686 / 32.0; 2,123 / 15.1 | 58 |
| 0-27, 28-47 | 5888 | 1,221 | 43.7 | 49.1 | 37.6 | 3,031 / 31.7; 2,630 / 15.3 | 35 |
| 0-27, 28-47 | 8192 | out of VRAM (65 MiB free; the engine asked for 1,147 MiB) | | | | | |
| 0-27, 28-47, reserve 1,247 | 8192 | 1,235 | 42.2 | 46.2 | 37.8 | 3,495 / 36.5; 3,195 / 16.4 | 47 |
| **0-25, 26-47** | 5888 | 1,230 | 48.7 | 49.3 | **36.3** | 2,753 / 28.6; 2,857 / 15.3 | 14 |
| 0-23, 24-47 | 5888 | 1,230 | 44.0 | 49.7 | 37.4 | 2,533 / 26.4; 3,200 / 17.1 | 26 |
| 0-24, 25-47 | 5888 | 1,199 | 47.8 | 50.4 | 37.2 | 2,657 / 28.5; 3,235 / 16.1 | 20 |
| 0-26, 27-47 | 5888 | 1,239 | 45.9 | 50.2 | 36.7 | 2,862 / 31.2; 2,813 / 15.1 | 15 |

The balance step used the first run's cost per layer (89 ms on the 3080, 120 ms on the 2080 Ti per 4096-token chunk)
and moved K from 22 to 28; at 5888 the walk came back to K=26. It agrees with the sweep (K=26 at 5888: 36.1 s there,
36.3 s here). Decode is noisier than prefill between runs: K=24 at 5888 decoded the ~30K prompt's answer at 50.2
tok/s in the sweep and 44.0 here, with 26% other CPU both times and a different prompt text; not explained.

### Three stages, function only

Main [0, 26) on the 3080; relay worker A [26, 37) and worker B [37, 48) both on the one 2080 Ti with 1,200 cache slots
each; chunk 2048. The answers were coherent. 8K 575 / 42.3, 32K 759 / 44.4 tok/s prefill / decode. Per 2048-token
chunk the main process's 26 layers took ~2.2 s and A's 11 layers ~1.3 s. Two workers shared one card and its PCIe
link, so this says nothing about three PCs.

## Lessons

- **The prompt chunk decides the split's prompt speed.** At 2048 no split beat one card (100K: 600-846 vs 1,120
  tok/s); at 5888-8192 every split from 32K on did (100K: 1,374-1,633). A layer of a chunk cost the 3080 96 ms at
  2048 tokens and 125 ms at 8192 (the table above): four times the tokens for a third more time.
- **Keep the adaptive tier on in every stage.** Off in both processes: 55% hits on the 3080 and 38.4 tok/s decode;
  on: 77-79% and 49.3.
- **Balance by measured time per layer, and again when the chunk changes.** At 2048 the best K was 26 (both stages
  ~2.5 s a chunk); measured costs at 4096 put the balance at 28; at 5888 the request time was lowest at 26, while
  100K prompts read fastest at 28. For the UD packs, whose layers differ by up to 1.5x in expert bytes, balance by
  bytes (computed, not measured).
- **A big chunk takes VRAM from the decode cache.** At 5888 (K=27) the 3080's prompt path borrowed 1,223 cache
  slots (2.65 GiB) and 153 MiB stayed free; at 8192 65 MiB stayed free and the first long prompt ran out. Decode at
  8192 with the reserve was 45.4-47.5 tok/s against 45.6-49.7 at 5888 (K=26).
- **Wait for pinned memory to be released between restarts** (more than 30 s for ~30 GB).
- **Plain TCP on a LAN is enough.** 1 GbE carried 117 MB/s; the link was ~3 ms of a 50-60 ms decode window. No
  special protocol, no RDMA.
- **Measure on a quiet host, or record the load.** The other CPU on the main PC's host ranged from 10% to 109% between
  runs of the sweep.
- **One split serves both phases.** The best K for long prompts (28) is not the best for decode (24-26). Different
  splits per phase would need the K/V and GDN state of the moved layers to cross the network.

## Estimates (not measured)

- **More PCs add cache, not speed per window.** A window runs the stages one after the other, so a third PC adds VRAM
  for the cache and another round trip; it pays only where the cards before it still miss many experts.
- **Over the internet**, decode would need about 1 Gbps and a round trip under ~17 ms per hop to stay ahead of one
  card on this rig (the split's window is that much shorter than the 3080's alone). Prefill needs bandwidth: 40,960 B
  per token each way.
- **Moving prompt state instead of rows:** reading the whole prompt on the main PC and sending the worker's layers'
  K/V and GDN state would be ~7.8 KB per token, about a fifth of the residual hand-off; the main PC would then need
  every layer's experts and stream all of them for the prompt. It would also allow a different split for prefill and
  for decode.
- **A relay on 1 GbE carries each chunk twice** (in from the main process and out to the next worker, and the
  replies back the same way): for prompts within the drafter's window that caps a three-PC chain at ~1,400 tok/s. A
  ring (main -> A -> B -> main) would not double it.

## Open work

- **The default path:** the branch moves the lend and refill code out of the request loop and adds stage hooks to
  `Verifier` and `Prefill`; a byte-identical check of the default path against upstream has not been run.
- **Builds:** CUDA on Linux only. HIP not built; Windows not built (`stage_link.cpp` has Windows stubs that refuse);
  the SYCL port keeps its own copies of `generate.cpp`, `verify.cpp`, `expert_source.cpp` and `prefill.cpp` (made by
  `sycl/tools/migrate.sh`), which this branch does not touch, so the Intel engine has no remote stage. The shared
  header `include/strata/prefill/prefill.hpp` changed under the SYCL copy of `prefill.cpp`; that build was not tried.
- **Windows sockets** (Winsock) for a native Windows worker; WSL2 not tested yet.
- **Correctness check** against the in-process split (`--layer-split K --split-device 0`) with the caches fixed: not
  done. So far the start of every benchmark answer (the 160 characters the benchmark keeps) was read: coherent and on
  topic in every configuration.
- **Reconnecting** without restarting the main engine (a broken link now ends the main engine at the next prompt's
  reset, and the server starts it again; read from the code, not tested); keeping conversation checkpoints across
  stages.
- **A ring topology** for three or more PCs on 1 GbE.
- **A bf16 hand-off** (half the traffic); its effect on the answers would need measuring.
- **Other quantizations** (official IQ3_S, IQ3_XXS, IQ2_XS, UD-IQ4_XS, UD-Q4_K_XL): not tested.
- **`tools/stage_ship.py` end to end** (a worker started from shipped, sparse files): not run.
- **The tuner's workload profile and `max_context` check:** in progress.
- **Three PCs with a real third GPU** (an RTX 3070 in a Windows 11 PC through WSL2): not run.
- **A faster link:** only 1 GbE measured.
