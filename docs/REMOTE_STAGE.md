# Strata across two or more PCs (remote stage)

> **Experimental and opt-in.** Linux only, the Qwen3.8-Flash-Next family only, measured on one pair of PCs with one
> model (RVN IQ3_S). Nothing changes unless the engine is started with `--remote-stage` or `--stage-worker`.
> How it works inside, every measurement and what is still open: [remote-stage/ENGINEERING.md](remote-stage/ENGINEERING.md).

A layer split ([MULTI_GPU.md](MULTI_GPU.md)) runs layers 0 to K-1 on one card and K onward on the next, with one
hand-off per verify window. The remote stage puts the later layers in a **worker process on another PC** and does
that hand-off over TCP:

- the **main PC** runs layers 0 to K-1, the output head, the draft (MTP) layer, sampling and the API server;
- a **worker PC** runs layers K to 47 without the head (`strata --serve --stage-worker PORT --stage-begin K`);
- each PC keeps an expert cache for **its own layers only**, its own CPU expert pool, its own part of the session
  (the K/V of its attention layers, the state of its GDN layers), and loads only its own layers' experts into RAM.

What it buys: a second card's VRAM for the expert cache and a second PC's RAM for the experts. What it costs: one
network round trip per verify window, and every prompt chunk's rows over the network.

## When it helps (measured)

RVN-Qwen3.8-Flash-Next IQ3_S, `--kv int8 --kv-resident 32768 --max-context 131072 --spec 4` with the draft layer,
no conversation cache, greedy, one run per row, 2026-10-09. Main PC: RTX 3080 10 GB (its slot runs at PCIe 4.0 **x8**,
13 GB/s measured), Ryzen 9 5900XT, 128 GB DDR4-3200, Linux container on Proxmox. Worker PC: RTX 2080 Ti 11 GB (PCIe
3.0 x16, 13.2 GB/s), Ryzen 9 5900X, 64 GB DDR4-2400, Proxmox host. Link: 1 GbE, 117 MB/s measured. K is the
number of layers on the 3080; the chunk is `--prefill` on both PCs. Prompt lengths: 7,521 / 30,167 / 57,383 /
90,921 tokens with a 320-token answer; "short decode" is the mean of nine 200-256-token answers to short prompts.
The main PC's host also ran CI jobs during the runs: "other CPU %" is their load (100 = one core of 32).

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
| K=26, chunk 8192, no reserve | - | out of VRAM at the first long prompt | | | | 10 |

- **Decode is 18-44% faster than the 3080 alone** in every row (100K: 45.3-52.0 vs 38.0 tok/s; short prompts
  49.4-54.0 vs 39.9). The two caches hold more of the experts: the 3080's decode hit rate was 77-87% with the
  split, 59-60% alone (both at chunk 2048).
- **Long prompts need a large chunk.** With 2048-token chunks every split read prompts slower than the 3080 alone
  (100K: 600-846 vs 1,120 tok/s). With 5888 or 8192 the split is faster from 32K on (100K: 1,374-1,633 vs 1,120).
- **Short prompts lose.** At 8K the split reads 589-740 tok/s against 791 alone (at 5888: 655 vs 791). The faster
  decode about makes up for it over a whole request (below).
- **The best K moves with the chunk.** At 2048 the 2080 Ti's stage is the slow one and K=26 is best; at 5888 the
  3080's extra speed shows and K=28 reads 100K prompts fastest while K=24-26 decode faster.

A request's time from these numbers (a nominal prompt of 8,000 / 32,000 / 100,000 tokens plus a 500-token answer,
prompt / prefill + 500 / decode; computed, not timed):

| configuration | 8K + 500 | 32K + 500 | 100K + 500 |
|---|---:|---:|---:|
| 3080 alone (auto chunk) | 23.2 s | 42.1 s | 102.4 s |
| K=26, chunk 2048 | 22.4 s | 49.5 s | 128.7 s |
| K=24, chunk 5888 | 22.1 s | 36.1 s | 83.2 s |
| K=26, chunk 5888 | 22.6 s | 36.1 s | 76.7 s |
| K=28, chunk 5888 | 22.1 s | 37.2 s | 73.8 s |
| K=26, chunk 8192 + reserve 1200 | 24.5 s | 36.6 s | 72.3 s |

The layer-split tuner (below), run on the same pair right after with its own 29,802-token prompt, picked K=26 at
chunk 5888 (prefill 1,230, decode 48.7 tok/s, 36.3 s for 32K + 500), the same choice as the table above.

Only this one model and this one pair of PCs have been measured. Other quantizations, other cards, three PCs and
links other than 1 GbE are **not tested** (see [What it works with](#what-it-works-with)).

## What you need

- **Linux on every PC** (the link uses POSIX sockets; a Windows build refuses `--remote-stage` and `--stage-worker`
  with "not supported on Windows"). WSL2 is the intended route for a Windows PC; not tested yet.
- **An NVIDIA card the engine supports on every PC** (RTX 20 or newer). Measured: an RTX 3080 (main) and an RTX 2080
  Ti (worker) running one build of this branch (CUDA 13.0, sm_75 + sm_86).
- **The same model on every PC:** the same pack directory (both sides compare a fingerprint of its `index.txt` and
  `native_experts.txt` and refuse a mismatch), every GGUF shard of the model (the engine checks that each shard exists
  and reads the headers; a worker then reads its own layers' tensors and none of the PLE table), and the expert
  profile. The fingerprint covers the pack's index files, not the weights: two models with the same layout and file
  names would pass it.
- **A native pack that takes its experts from the GGUF** (no `experts.bin` in the pack directory, no
  `--shared-expert-arena`): only that loader can hold a range of layers. Anything else stops at start with "a layer
  range needs a native pack read from its GGUF".
- **RAM for its own layers' experts** on each PC, pinned: 1.11 GiB per layer for RVN IQ3_S (53.32 GiB for all 48), so
  at K=26 about 28.9 GiB on the main PC and 24.4 GiB on the worker (computed; at K=27 the main process's log said
  29.99 GiB).
- **A network both PCs share.** Measured on 1 GbE through a switch (117 MB/s, ping 0.13-0.56 ms).

## Two PCs

**1. A shared secret.** Put the same string (up to 63 characters) in `STRATA_STAGE_TOKEN` on both PCs. A worker
started without one prints a warning and takes orders from any host that reaches its port.

**2. Start the worker first** (the worker PC, here 192.0.2.11, running layers 26-47):

```
export STRATA_STAGE_TOKEN="$(cat ~/strata/stage-token)"
export LD_LIBRARY_PATH=~/strata/bin          # where the engine's CUDA libraries are, if not installed system-wide
~/strata/bin/strata --serve --stage-worker 7841 --stage-begin 26 --stage-bind 192.0.2.11 \
  --pack ~/strata/packs/rvn-iq3_s \
  --native ~/strata/models/rvn-iq3s/RVN-Qwen3.8-Flash-Next-IQ3_S-00001-of-00008.gguf \
  --ple-gguf ~/strata/models/rvn-iq3s/RVN-Qwen3.8-Flash-Next-IQ3_S-00002-of-00008.gguf \
  --expert-profile ~/strata/data/expert-profile.bin --expert-cache auto --prefill 5888 \
  --spec 4 --max-context 131072 --kv int8 --kv-resident 32768
```

It is ready when the log says `strata stage worker: listening on 192.0.2.11:7841 (a token is required)`. It runs
without stdin and serves one main process at a time; when the main process goes away it waits for the next one.
The worker does not read the PLE table (that runs in layer 1, always on the main PC), but the argument checks still
want `--ple-gguf` or a shard that holds it.

**3. Then the main PC.** Add these to the model's config (`strata-*.json`, `"args"`), and the token to its `"env"`:

```json
"args": [ "...the model's usual args...",
          "--prefill", "5888",
          "--layer-split", "26", "--split-device", "0", "--remote-stage", "192.0.2.11:7841" ],
"env": { "STRATA_STAGE_TOKEN": "the same secret" }
```

Then start the server as usual. `--layer-split K --split-device 0` keeps the head on this card and sends layers K
onward to the worker. Keep the split in `"args"` as above (that is how it was run; the config's `"layer_split"` key
is for cards in one PC), and drop `--conversation-cache-mib`: the remote stage turns the conversation cache off
anyway. The main process connects once, at start, and stops if the worker is not listening. The log then says:

```
strata generate: remote stage: this process holds layers 0-25, the rest on 192.0.2.11:7841
strata serve: layer split: layers 0-25 (CUDA0), 26-47 (192.0.2.11:7841), the head (CUDA0), one hand-off per window
strata serve: remote stage 192.0.2.11:7841: layers 26-47 there (prompt chunk 5888, context 131072)
```

**What must agree** (each side checks the other's hello and says which number differs): the model pack, `--kv`, and
K (`--layer-split K` on the main PC, `--stage-begin K` on the worker). The worker's `--prefill`, `--max-context` and
`--spec` must be at least the main process's. Everything else is each PC's own choice: `--expert-cache`,
`--kv-resident`, `--vram-reserve-mib`, the CPU threads.

## More than two PCs (relay workers)

A middle worker runs a range of layers and hands its rows to the next worker; the next worker's reply comes back
through it: main -> A -> B -> A -> main. The main process is configured exactly as for two PCs (it sees one worker).
Start the **last** worker first; a relay worker connects to the next one when it starts (it retries for a minute).

```
# PC B, layers 37-47:
strata --serve --stage-worker 7842 --stage-begin 37 ...model and context args as above...
# PC A, layers 26-36, relaying to B:
strata --serve --stage-worker 7841 --stage-begin 26 --stage-end 37 --stage-next 192.0.2.12:7842 ...
# main PC, layers 0-25: --layer-split 26 --split-device 0 --remote-stage <A's address>:7841
```

Tested for function only: main [0,26) on the 3080, A [26,37) and B [37,48) **both on the one 2080 Ti** (1,200 cache
slots each); the answers were coherent. Its speed (8K 575 / 42.3, 32K 759 / 44.4 tok/s prefill / decode, chunk 2048)
says nothing about three PCs, because two workers shared one card. Two workers on one card need a fixed
`--expert-cache N` each: `auto` gives the first one all the free VRAM.

## Options

| flag or variable | where | what it does |
|---|---|---|
| `--remote-stage HOST:PORT` | main | the (first) worker's address. Needs `--serve` and `--layer-split K --split-device 0` |
| `--layer-split K --split-device 0` | main | K is the first layer the worker runs (2 or more): this PC runs 0 to K-1 and the head |
| `--stage-worker PORT` | worker | serve a main process on this TCP port instead of stdin. Needs `--serve`, not with `--layer-split` |
| `--stage-begin K` | worker | the first layer this worker runs: the main process's K (or the previous relay's `--stage-end`) |
| `--stage-end K2`, `--stage-next HOST:PORT` | relay worker | run layers K to K2-1 and hand the rest to the worker at HOST:PORT. Both or neither |
| `--stage-bind ADDR` | worker | the address to listen on (default: every interface, IPv4 and IPv6) |
| `STRATA_STAGE_TOKEN` | every PC | the shared secret in the hello (up to 63 characters). A worker without one warns and accepts anyone |
| `STRATA_REMOTE_TIMING=1` | any | per-stage timing lines: each prompt chunk, and every 64 windows (see below) |
| `STRATA_REMOTE_TIMEOUT_S` | any | seconds a send or receive may block before the link counts as dead (default 300). After the hello, a worker waits for the next message without a limit |

Fixed: a peer that connects but sends no hello within 10 s is dropped; TCP keepalive notices a peer that vanished
without closing (power, cable) in about a minute.

**Timing lines** (`STRATA_REMOTE_TIMING=1`), what the tuner reads. The main process's lines below are from the
tuner's K=27, chunk 5888 run; the worker's show the format:

```
strata prefill: layers 0-26, chunk T=5888 at 5888: own 2862 ms, waited 0 ms for the previous send     (main, relay)
strata remote: chunk T=5888: 7005 ms from its send to its rows = worker 2704 + link and queue 4301 (5888 rows back)
strata remote: 64 windows: mean 20.15 ms a round trip = worker 17.25 + link 2.90 (last: ...)             (main)
strata remote: main process between windows 30.94 ms each (64 windows: head, drafter, layers 0-K)       (main)
strata stage worker: chunk T=<tokens> at <position>: read in <ms> ms                                   (worker)
strata stage worker: windows: own layers <ms> ms[, next worker <ms> ms] each (64 windows)              (worker)
```

"link and queue" of a chunk includes the time it waited behind earlier chunks: the chunks are pipelined.

## Picking K and the chunk

What the measurements above say (one pair of PCs; re-measure on yours):

- **Use a large prompt chunk on both PCs** (5888 or more), not the 2048 default of a plain split. Each chunk streams
  the experts it routes to over PCIe, layer by layer (at these sizes most of a layer's 1.11 GiB); a layer of a chunk
  took about as long at 4096 tokens as at 2048 (3080: 89 and 96 ms), so a larger chunk spreads that cost over more
  tokens.
- **A larger chunk takes VRAM from the main card's decode cache.** At 8192 the 3080 had 65 MiB free after loading
  and ran out of memory at the first long prompt; `--vram-reserve-mib 1200` fixed it. The engine's start-up line
  `N MiB of VRAM free with everything loaded - LOW ... add --vram-reserve-mib M` gives the number.
- **Balance by each stage's measured time per layer, and again after changing the chunk.** At 2048, a layer of a
  prompt chunk took ~96 ms on the 3080 and ~112 ms on the 2080 Ti; at 4096, 89 and 120 ms.
- **For packs whose layers differ in size** (the Unsloth UD packs: layers 2, 4, 30, 46 and 47 hold up to 1.5x the
  expert bytes of the others), balance by bytes, not by layer count. Computed from the packs' expert tables, not
  measured with the remote stage.

## Tools

Three Python scripts (standard library only) in `tools/`. Their docstrings are the full usage.

**`tools/stage_node.py`: a node agent** on each worker PC. It starts and stops that PC's stage workers on request
and reports the PC (GPU, RAM, CPU) and the link speed toward other nodes. Every request carries the token in an
`X-Stage-Token` header.

```
python3 tools/stage_node.py node.json
```

```json
{ "exe": "/srv/stage/bin/strata", "lib_dirs": ["/srv/stage/bin"],
  "pack": "/srv/stage/packs/rvn-iq3_s",
  "native": "/srv/stage/models/rvn-iq3s/RVN-Qwen3.8-Flash-Next-IQ3_S-00001-of-00008.gguf",
  "ple_gguf": "/srv/stage/models/rvn-iq3s/RVN-Qwen3.8-Flash-Next-IQ3_S-00002-of-00008.gguf",
  "expert_profile": "/srv/stage/data/expert-profile.bin",
  "bind": "192.0.2.11", "agent_port": 7840, "log_dir": "/srv/stage",
  "args": ["--spec", "4", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"],
  "token_file": "/srv/stage/stage-token",
  "data_dir": "/srv/stage/shipped" }
```

Endpoints: `GET /info`, `POST /start` (`{"begin": K, "end": K2, "next": "host:port", "port": 7841, "prefill": 5888,
"cache": "auto", "extra": [...]}`; it answers when the worker listens, or with the log's tail when it exits),
`POST /stop`, `GET /log`, `POST /sink` and `POST /send` (a link test), `POST /put` and `GET /ranges` (for
`stage_ship.py`). A stop waits up to 180 s for the worker to exit: a worker with ~30 GB of pinned experts took more
than 30 s to unpin, and a worker started before that finds no free VRAM.

**`tools/stage_tune.py`: the layer-split tuner**, run on the main PC. It measures for real: for every configuration
it restarts the workers (through their agents) and the main server, reads one long prompt and three short ones, and
takes each stage's own times from the timing lines. The score is one request's time, prompt tokens / prefill +
answer tokens / decode.

```
python3 tools/stage_tune.py tune  cluster.json OUTDIR    # measure and search; writes OUTDIR/main.json, nodes.json, report.md
python3 tools/stage_tune.py apply cluster.json OUTDIR    # start the workers as nodes.json says, then serve OUTDIR/main.json
```

```json
{ "main": {"config": "/srv/strata/Strata/strata-rvn-iq3_s.json", "exe": "/srv/strata/Strata-fork/build/strata",
           "cwd": "/srv/strata/Strata-fork", "python": "/srv/strata/Strata/.venv/bin/python", "port": 8080,
           "api_key_file": "/srv/strata/api-key", "extra": []},
  "nodes": [{"name": "2080ti", "agent": "http://192.0.2.11:7840", "host": "192.0.2.11", "port": 7841}],
  "token_file": "/srv/strata/stage-token", "corpus": "/srv/strata/corpus.txt",
  "goal": {"prompt_tokens": 32000, "answer_tokens": 500},
  "chunks": [4096, 5888, 8192], "max_runs": 9 }
```

The nodes run in the order listed (every node but the last is a relay). The search: a first split by VRAM; the split
that gives every stage the same measured prompt time; the larger chunks at the better of the two (a chunk that runs
out of VRAM is tried once more with the reserve the engine asks for, plus 100 MiB); then each split point two and then
one layer either way while that wins. On the pair above, 9 configurations took 12 minutes and picked K=26 at chunk
5888. `main.json` is a server config with the token in its `"env"`; keep it private.

Being added (another change in progress, not in this branch's tuner yet): a workload profile (several prompt lengths
with weights, scored by the weighted mean request time) measured for the best few configurations, `measure_tokens`
for the search prompt, `max_context` (for example 131072, or the model's 262144) with a final check that the pick
reads a prompt near it, balancing by each layer's bytes from the GGUF, and shipping a node its layers first
(`"ship": true`).

**`tools/stage_ship.py`: send a node only its layers.** From the PC that holds the whole model it sends, to a node
agent's `data_dir`: every GGUF shard's header and the tensors of the node's layers at their own offsets, in files of
the shards' full sizes (sparse on the node, so the loader and the pack's index work unchanged); the non-layer
tensors except the PLE table (`per_layer_token_embd`, ~26.8 GB for RVN IQ3_S), which only a node that runs layer 1
would need, and no worker does; the pack directory, the expert profile and optionally the engine's directory. What
the node already has is skipped, and every 64 MiB piece's sha256 is compared on arrival.

```
python3 tools/stage_ship.py --agent http://192.0.2.11:7840 --token-file /srv/strata/stage-token \
    --model-dir /srv/strata/models/rvn-iq3s --pack-dir /srv/strata/Strata-data/packs/rvn-iq3_s \
    --profile /srv/strata/Strata/data/expert-profile.bin --layers 26-47
```

**Not yet run end to end**: no worker has been started from shipped files so far (the measured worker had the whole
model copied). Its docstring says the node's paths are printed at the end; the script does not print them yet.

## What is turned off or refused

- **Refused at start** with a remote stage: `--batch`, `--pipeline-windows`, `--vision`, `--peer-device`,
  `--control-vector` (and the speed projection, which is one), `--expert-cache-remote`.
- **Turned off without an error:** the prompt cache, the conversation cache and mid-prompt checkpoints (each PC
  holds only its layers' state, so every prompt is read from token 0), and `--kv-grow`.
- **Not checked** with a remote stage: `STRATA_PREFILL_HELP`, `--adapt-async 1`, `--resident-experts`,
  `--mmap-experts` (the mapped expert source gets no layer range; it maps the whole pack), `STRATA_PF_FUSED=1`.

## Security

- **The link is plain TCP, not encrypted.** Every verify window and prompt chunk carries the token ids and the
  layers' hidden states: whoever can read the traffic can read the conversation. Use it on a LAN you trust; across
  anything else put it inside a tunnel (not measured).
- **The token is a shared secret in the hello, also in clear.** It keeps other hosts on the LAN from driving a
  worker; it is no protection against someone who sees the traffic. Bind the worker to the LAN address
  (`--stage-bind`) and firewall the port.
- **A node agent's token is a full credential:** with it a request can start the engine with any extra arguments and
  write files under `data_dir`. Treat it as a password and keep the agent's port on the LAN.
- A worker serves one main process at a time; a connection that sends no hello within 10 s is dropped.

## What it works with

| | status |
|---|---|
| Model family | Qwen3.8-Flash-Next only (48 layers, 512 experts top-10, 4 hyper-connection streams, n_embd 2560). The hello checks the geometry; the tuner assumes 48 layers |
| RVN IQ3_S, native pack read from its GGUF | **measured** (this page) |
| Other quantizations and models of the family (the official IQ3_S, IQ3_XXS, IQ2_XS, Coder, Swift 1.5, Unsloth UD-IQ4_XS and UD-Q4_K_XL) | **not tested**. The hand-off is fp32 rows whatever the quantization, the arena range is cut by layer from the pack's expert table and the splitter works by tensor name; nothing in the code is specific to IQ3_S. A pack that has an `experts.bin` is refused (next row) |
| A pack with `experts.bin`, or `--shared-expert-arena` | refused at start (the layer range needs the GGUF loader) |
| Linux, NVIDIA (CUDA) | measured: Ubuntu 24.04 in an LXC container (main), Proxmox VE 9.0 host, kernel 6.14 (worker) |
| WSL2 | not tested (planned: an RTX 3070 in a Windows 11 PC, mirrored networking) |
| Windows (native) | refused: "not supported on Windows" |
| AMD (HIP) | not built or tested |
| Intel (SYCL) | not available: the SYCL port's own copy of the engine does not have it |
| Links | measured on 1 GbE only |

## Troubleshooting

| message or symptom | what to do |
|---|---|
| `verify: instantiate: out of memory` at the first long prompt; at start `N MiB of VRAM free with everything loaded - LOW ... add --vram-reserve-mib M` | add `--vram-reserve-mib M` (the tuner adds 100 more) or use a smaller `--prefill` |
| a worker restarted right after a stop fails (here: 496 MiB of VRAM free) | the previous worker is still unpinning (more than 30 s with ~30 GB of experts); wait until its process is gone and `nvidia-smi` shows the memory free |
| `remote stage: the worker refused: wrong token (STRATA_STAGE_TOKEN)` | the same `STRATA_STAGE_TOKEN` on both PCs |
| `remote stage: the two sides differ (protocol ..., first remote layer ..., K/V type ..., model pack ...)` | a build of this branch on both PCs (the protocol number), the same pack, the same `--kv`, and `--stage-begin` = the main process's K |
| `the main process reads prompts in chunks of N tokens, this worker's chunk is M (start it with --prefill N)` | the worker's `--prefill` must be at least the main process's chunk |
| `the main process's context (N) or window (T) is larger than this worker's` | the worker's `--max-context` and `--spec` must be at least the main process's |
| `remote stage: cannot connect to HOST:PORT` | start the worker first and wait for `listening on`; check `--stage-bind`, the address and the firewall |
| `ArenaExpertSource: a layer range needs a native pack read from its GGUF (no experts.bin, no shared arena)` | a pack directory with `experts.bin` (a start with `STRATA_ARENA_MMAP=1` writes one): use a copy without it; drop `--shared-expert-arena` |
| `strata generate: remote stage: it does not support ...` | remove that flag (see above) |
| a request fails with a link error | the next request's prompt stops the main engine ("the link failed earlier") and the server starts it again, which connects anew; the worker waits for the new connection. Read from the code, not tested |
| decode slower than expected | look at `decode expert cache hit rate` on both PCs. Keep the adaptive tier on everywhere (do not set `--adapt-every 0`): a first version had it off in both processes, and the 3080's hit rate was 55% instead of 77-79% |
| numbers that move between runs | record the other load on both PCs and repeat runs: the same configuration (K=24, chunk 5888, a ~30K prompt with different text) decoded at 50.2 and 44.0 tok/s twenty minutes apart, with light load both times |
