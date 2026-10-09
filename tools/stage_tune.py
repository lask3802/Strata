"""The remote-stage layer-split tuner: measures this PC (the main process: the first layers, the head and the drafter)
and every node (a PC running tools/stage_node.py with a stage worker) once, then recommends the split points and the
prompt chunk, and writes the configs that run them.

    python3 tools/stage_tune.py tune  cluster.json OUTDIR     measure, search, write OUTDIR/{main.json,nodes.json,report.md}
    python3 tools/stage_tune.py apply cluster.json OUTDIR     start the nodes' workers as OUTDIR/nodes.json says
                                                              (then serve OUTDIR/main.json as any config)

cluster.json:
    {
      "main": {"config": "/srv/strata/Strata/strata-rvn-iq3_s.json",   the model's server config (args, key, sampling)
               "exe": "/srv/strata/Strata-fork/build/strata", "cwd": "/srv/strata/Strata-fork",
               "python": "/srv/strata/Strata/.venv/bin/python", "port": 8080,
               "api_key_file": "/srv/strata/api-key", "extra": []},
      "nodes": [{"name": "2080ti", "agent": "http://192.0.2.11:7840", "host": "192.0.2.11", "port": 7841}],
      "token_file": "/srv/strata/stage-token",
      "corpus": "/srv/strata/corpus.txt",
      "goal": {"prompt_tokens": 32000, "answer_tokens": 500},
      "chunks": [4096, 5888, 8192],          the prompt chunks to try (the first one is the safe start)
      "max_runs": 10, "init_splits": null    e.g. [26] or [20, 34]; default: by VRAM
    }

The stages run in the order the nodes are listed: main -> node 1 -> node 2 -> ...; every node but the last is a
relay worker (its rows go on to the next node, the next node's reply comes back through it).

What it measures, per configuration (every process restarted): one prompt of goal.prompt_tokens with a 320-token
answer, then three short prompts with 256-token answers; and from the logs (STRATA_REMOTE_TIMING=1) each stage's own
time per prompt chunk and per decode window.  The score is a request's time: prompt_tokens / prefill + answer_tokens /
decode (the long prompt's decode).  The search: a first split by VRAM; a split that balances the stages' measured
per-layer prompt cost; the larger prompt chunks at the better of the two (a larger chunk costs VRAM the decode cache
would use - when the engine says "add --vram-reserve-mib N" the chunk is tried once more with it); then, at the best
chunk, each split point two and then one layer either way while that wins (the best split moves with the chunk: a
larger chunk makes the faster card's share pay).  Measured on a busy PC the numbers move a few percent between runs:
the report lists every run.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

N_LAYERS = 48              # Qwen3.8-Flash-Next (every variant Strata runs)
MIN_LAYERS = 2             # the fewest layers a stage may hold
CHARS_PER_TOKEN = 3.0      # this tokenizer on English prose (bench_ctx.py)
LONG_ANSWER = 320
SHORT_PROMPTS = ("Write a Python function that merges two sorted lists into one sorted list, with a docstring and "
                 "two tests.",
                 "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
                 "List twelve European capitals with one sentence about each.")
SHORT_ANSWER = 256


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- nodes (tools/stage_node.py)
class Node:
    def __init__(self, spec: dict, token: str):
        self.name = spec.get("name", spec["host"])
        self.agent = spec["agent"].rstrip("/")
        self.host = spec["host"]
        self.port = int(spec.get("port", 7841))
        self.cache = spec.get("cache", "auto")
        self.extra = list(spec.get("extra", []))
        self.token = token
        self.info: dict = {}

    @property
    def stage_addr(self) -> str:
        return f"{self.host}:{self.port}"

    def call(self, path: str, body: dict | None = None, timeout: float = 900, data: bytes | None = None) -> dict:
        headers = {"X-Stage-Token": self.token}
        if data is None and body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.agent + path, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
        except (urllib.error.URLError, OSError) as e:
            return {"ok": False, "error": f"{self.agent}: {e}"}

    def log(self, lines: int = 4000) -> list[str]:
        return self.call(f"/log?port={self.port}&lines={lines}").get("log", [])


# ---------------------------------------------------------------- the main process (serve/server.py)
class Main:
    def __init__(self, spec: dict, token: str, outdir: Path):
        self.spec = spec
        self.token = token
        self.outdir = outdir
        self.base = json.loads(Path(spec["config"]).read_text(encoding="utf-8"))
        self.port = int(spec.get("port", 8080))
        kf = spec.get("api_key_file")
        self.key = Path(kf).read_text().strip() if kf and Path(kf).exists() else os.environ.get("STRATA_API_KEY", "")
        self.proc: subprocess.Popen | None = None
        self.log_path = outdir / "main-engine.log"

    def config(self, splits: list[int], chunk: int, first: Node, timing: bool, extra=()) -> dict:
        args = list(self.base["args"])
        if "--conversation-cache-mib" in args:   # the remote stage turns it off; say so in the config too
            i = args.index("--conversation-cache-mib")
            del args[i:i + 2]
        if "--prefill" in args:
            args[args.index("--prefill") + 1] = str(chunk)
        else:
            args += ["--prefill", str(chunk)]
        args += ["--layer-split", str(splits[0]), "--split-device", "0", "--remote-stage", first.stage_addr]
        args += [str(a) for a in self.spec.get("extra", [])] + [str(a) for a in extra]
        cfg = dict(self.base)
        env = dict(cfg.get("env") or {})
        env["STRATA_STAGE_TOKEN"] = self.token
        if timing:
            env["STRATA_REMOTE_TIMING"] = "1"
        cfg.update(exe=self.spec["exe"], cwd=self.spec["cwd"], args=args, env=env,
                   log=str(self.log_path), model_name=self.base.get("model_name", "strata") + "-stages")
        return cfg

    def start(self, cfg: dict) -> tuple[bool, str]:
        self.stop()
        path = self.outdir / "main-run.json"
        path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if self.log_path.exists():
            self.log_path.unlink()
        out = open(self.outdir / "main-server.out", "wb")
        self.proc = subprocess.Popen([self.spec["python"], "serve/server.py", "--engine", "strata", "--config",
                                      str(path), "--port", str(self.port)], cwd=self.spec["cwd"],
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        out.close()
        t0 = time.time()
        while time.time() - t0 < 900:
            time.sleep(3)
            if self.proc.poll() is not None:
                return False, tail(self.outdir / "main-server.out", 30)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    h = json.load(r)
                if h.get("loaded"):
                    return True, ""
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if self.log_path.exists():   # an engine that exited reports it in the server's output
                srv = (self.outdir / "main-server.out").read_text(encoding="utf-8", errors="replace")
                if "stopped unexpectedly" in srv or "engine exited" in srv:
                    return False, tail(self.outdir / "main-server.out", 30)
        return False, "not loaded after 900 s"

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        self.proc = None
        exe = self.spec["exe"]
        for _ in range(120):   # the engine (the server's child) frees the GPU before the next start
            if subprocess.run(["pgrep", "-f", exe], capture_output=True).returncode != 0:
                break
            time.sleep(1)

    def chat(self, content: str, max_tokens: int) -> dict:
        body = {"model": "strata", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "temperature": 0.0}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(),
                                     headers)
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            return {"error": e.read().decode(errors="replace")[:300]}
        except (urllib.error.URLError, OSError) as e:
            return {"error": str(e)}
        tm = resp.get("timings") or {}
        return {"prompt_n": tm.get("prompt_n"), "prefill_tps": tm.get("prompt_per_second"),
                "gen_n": tm.get("predicted_n"), "decode_tps": tm.get("predicted_per_second"),
                "text": (resp["choices"][0]["message"].get("content") or "")[:120]}


def tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# ---------------------------------------------------------------- logs -> per-stage times
RE_PF_OWN = re.compile(r"strata prefill: layers (\d+)-(\d+), chunk T=(\d+) at (\d+): own (\d+) ms")
RE_WK_CHUNK = re.compile(r"strata stage worker: chunk T=(\d+) at (\d+): read in (\d+) ms")
RE_WK_WIN = re.compile(r"strata stage worker: windows: own layers ([\d.]+) ms")
RE_MAIN_WIN = re.compile(r"strata remote: main process between windows ([\d.]+) ms")
RE_RT = re.compile(r"strata remote: \d+ windows: mean ([\d.]+) ms a round trip = worker ([\d.]+) \+ link ([\d.]+)")
RE_VRAM = re.compile(r"(\d+) MiB of VRAM free with everything loaded")
RE_RESERVE = re.compile(r"add --vram-reserve-mib (\d+)")


def median_or_none(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def stage_times(lines: list[str], chunk: int, last: bool) -> dict:
    """A stage's own prompt-chunk time (full chunks only) and decode-window time from its log."""
    pf = [float(m.group(5)) for m in map(RE_PF_OWN.search, lines) if m and int(m.group(3)) == chunk]
    if last or not pf:   # the last worker reports the chunk it read (a relay its sending stage's line)
        pf = pf or [float(m.group(3)) for m in map(RE_WK_CHUNK.search, lines) if m and int(m.group(1)) == chunk]
    win = [float(m.group(1)) for m in map(RE_WK_WIN.search, lines) if m]
    win = win or [float(m.group(1)) for m in map(RE_MAIN_WIN.search, lines) if m]
    vram = [int(m.group(1)) for m in map(RE_VRAM.search, lines) if m]
    return {"pf_chunk_ms": median_or_none(pf), "window_ms": median_or_none(win[1:] or win),
            "vram_free_mib": vram[-1] if vram else None}


# ---------------------------------------------------------------- one configuration
class Tuner:
    def __init__(self, cluster: dict, outdir: Path):
        self.cl = cluster
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        tf = cluster.get("token_file")
        self.token = (Path(tf).read_text().strip() if tf else os.environ.get("STRATA_STAGE_TOKEN", "")).strip()
        if not self.token:
            sys.exit("stage_tune: no token (cluster.json's token_file or STRATA_STAGE_TOKEN)")
        self.nodes = [Node(n, self.token) for n in cluster["nodes"]]
        self.main = Main(cluster["main"], self.token, outdir)
        goal = cluster.get("goal", {})
        self.prompt_tokens = int(goal.get("prompt_tokens", 32000))
        self.answer_tokens = int(goal.get("answer_tokens", 500))
        self.corpus = Path(cluster["corpus"]).read_text(encoding="utf-8", errors="replace")
        self.runs: list[dict] = []
        self.runs_path = outdir / "runs.jsonl"

    def stages(self, splits: list[int]) -> list[tuple[int, int]]:
        b = [0] + list(splits) + [N_LAYERS]
        return [(b[i], b[i + 1]) for i in range(len(b) - 1)]

    def start_nodes(self, splits: list[int], chunk: int) -> tuple[bool, str]:
        for n in self.nodes:
            n.call("/stop", {})
        ranges = self.stages(splits)[1:]
        for i in reversed(range(len(self.nodes))):   # the last first: a relay connects to the next when it starts
            n, (lo, hi) = self.nodes[i], ranges[i]
            req = {"begin": lo, "port": n.port, "prefill": chunk, "cache": n.cache, "extra": n.extra}
            if i + 1 < len(self.nodes):
                req.update(end=hi, next=self.nodes[i + 1].stage_addr)
            r = n.call("/start", req)
            if not r.get("ok"):
                return False, f"{n.name}: " + (r.get("error") or "") + "\n" + "\n".join(r.get("log", [])[-15:])
            say(f"  {n.name}: layers {lo}-{hi - 1} up in {r.get('seconds')} s")
        return True, ""

    def measure(self, splits: list[int], chunk: int, extra=()) -> dict:
        for r in self.runs:   # measured already
            if r["key"] == [list(splits), chunk, list(extra)]:
                return r
        say(f"measure: splits {splits} (stages {self.stages(splits)}), chunk {chunk} {' '.join(extra)}")
        run = {"key": [list(splits), chunk, list(extra)], "splits": list(splits), "chunk": chunk, "extra": list(extra),
               "stages": self.stages(splits), "ok": False}
        ok, why = self.start_nodes(splits, chunk)
        if ok:
            ok, why = self.main.start(self.main.config(splits, chunk, self.nodes[0], timing=True, extra=extra))
        if not ok:
            run["error"] = why.strip()[-1500:]
            if self.main.log_path.exists():   # an engine that ran out of VRAM says how much headroom it wants
                res = [int(m.group(1)) for m in map(RE_RESERVE.search, self.main.log_path.read_text(
                    encoding="utf-8", errors="replace").splitlines()) if m]
                if res:
                    run["suggested_reserve_mib"] = res[-1]
            self.main.stop()
            say(f"  does not run: {why.strip().splitlines()[-1] if why.strip() else '?'}")
            self.record(run)
            return run
        n = int(self.prompt_tokens * CHARS_PER_TOKEN)
        doc = (self.corpus * (1 + n // max(1, len(self.corpus))))[:n]
        long = self.main.chat("Here is a long document. After reading it, write a detailed technical summary of its "
                              "main ideas in about 250 words.\n\n<document>\n" + doc + "\n</document>", LONG_ANSWER)
        shorts = [self.main.chat(p, SHORT_ANSWER) for p in SHORT_PROMPTS] if not long.get("error") else []
        run["long"] = long
        run["short_decode"] = median_or_none([s["decode_tps"] for s in shorts if s.get("decode_tps")])
        if long.get("error") or not long.get("prefill_tps"):
            run["error"] = long.get("error", "no timings")
            say(f"  failed: {run['error'][:200]}")
        else:
            run["ok"] = True
            run["prefill_tps"] = long["prefill_tps"]
            run["decode_tps"] = long["decode_tps"]
            run["request_s"] = self.prompt_tokens / long["prefill_tps"] + self.answer_tokens / long["decode_tps"]
        # each stage's own times
        main_lines = self.main.log_path.read_text(encoding="utf-8", errors="replace").splitlines() \
            if self.main.log_path.exists() else []
        st = [stage_times(main_lines, chunk, last=False)]
        for i, nd in enumerate(self.nodes):
            st.append(stage_times(nd.log(), chunk, last=i + 1 == len(self.nodes)))
        for s, (lo, hi) in zip(st, run["stages"]):
            s["layers"] = hi - lo
        run["per_stage"] = st
        res = [int(m.group(1)) for m in map(RE_RESERVE.search, main_lines) if m]
        if res:
            run["suggested_reserve_mib"] = res[-1]
        if run["ok"]:
            say(f"  prefill {run['prefill_tps']:.0f} tok/s, decode {run['decode_tps']:.1f} (short "
                f"{run['short_decode'] or 0:.1f}), request {run['request_s']:.1f} s; stages: " +
                ", ".join(f"{s['layers']}L pf {s['pf_chunk_ms'] or 0:.0f} ms win {s['window_ms'] or 0:.1f} ms"
                          for s in st))
        self.main.stop()
        self.record(run)
        return run

    def record(self, run: dict) -> None:
        self.runs.append(run)
        with open(self.runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run) + "\n")

    # ------------------------------------------------------------ the search
    def initial_splits(self) -> list[int]:
        if self.cl.get("init_splits"):
            return [int(x) for x in self.cl["init_splits"]]
        # by VRAM: each stage's share of the layers follows its card's memory (the main process keeps ~1.5 GB for
        # the head, the drafter and the windows)
        vram = [self.main_vram_mib() - 1500] + [max(1000, (n.info.get("gpus") or [{}])[0].get("vram_mib", 8000) - 500)
                                                 for n in self.nodes]
        return self.splits_from_weights(vram)

    def main_vram_mib(self) -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout
            return int(out.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return 10000

    def splits_from_weights(self, w: list[float]) -> list[int]:
        """Layers in proportion to the weights, every stage at least MIN_LAYERS, as split points."""
        tot = sum(w)
        sizes = [max(MIN_LAYERS, round(N_LAYERS * x / tot)) for x in w]
        while sum(sizes) > N_LAYERS:
            sizes[sizes.index(max(sizes))] -= 1
        while sum(sizes) < N_LAYERS:
            sizes[sizes.index(min(sizes))] += 1
        out, acc = [], 0
        for s in sizes[:-1]:
            acc += s
            out.append(acc)
        return out

    def balanced(self, run: dict) -> list[int] | None:
        """The split that gives every stage the same prompt time, from the run's per-layer costs."""
        costs = []
        for s in run.get("per_stage", []):
            if not s.get("pf_chunk_ms") or not s.get("layers"):
                return None
            costs.append(s["pf_chunk_ms"] / s["layers"])
        return self.splits_from_weights([1.0 / c for c in costs])

    def valid(self, splits: list[int]) -> bool:
        b = [0] + splits + [N_LAYERS]
        return all(b[i + 1] - b[i] >= MIN_LAYERS for i in range(len(b) - 1))

    def best(self, chunk: int | None = None) -> dict | None:
        ok = [r for r in self.runs if r["ok"] and (chunk is None or r["chunk"] == chunk)]
        return min(ok, key=lambda r: r["request_s"]) if ok else None

    def tune(self) -> dict | None:
        max_runs = int(self.cl.get("max_runs", 10))
        chunks = [int(c) for c in self.cl.get("chunks", [4096, 5888, 8192])]
        c0 = chunks[0]
        say("nodes:")
        for n in self.nodes:
            n.info = n.call("/info", timeout=30)
            if not n.info.get("ok"):
                sys.exit(f"stage_tune: {n.name} ({n.agent}) does not answer: {n.info.get('error')}")
            g = (n.info.get("gpus") or [{}])[0]
            say(f"  {n.name}: {g.get('name')} {g.get('vram_mib')} MiB, PCIe gen {g.get('pcie_gen')} "
                f"x{g.get('pcie_width')}, RAM {n.info.get('ram', {}).get('total_mib')} MiB, {n.info.get('cpus')} CPUs")
        self.links()
        r = self.measure(self.initial_splits(), c0)
        if not r["ok"]:
            say("the first split does not run; trying an even one")
            r = self.measure(self.splits_from_weights([1.0] * (len(self.nodes) + 1)), c0)
            if not r["ok"]:
                return None
        # 1. balance the stages' prompt time
        bal = self.balanced(r)
        if bal and bal != r["splits"] and self.valid(bal):
            self.measure(bal, c0)
        # 2. the prompt chunk, at the best split so far
        splits = self.best(c0)["splits"]
        for c in chunks[1:]:
            r = self.measure(splits, c)
            if not r["ok"] and r.get("suggested_reserve_mib"):
                extra = ["--vram-reserve-mib", str(r["suggested_reserve_mib"] + 100)]
                say(f"  the engine asks for more VRAM headroom: once more with {' '.join(extra)}")
                r = self.measure(splits, c, extra)
            if not r["ok"]:
                say(f"  chunk {c} does not run at this split; larger ones are not tried")
                break
        best = self.best()
        chunk, extra = best["chunk"], best["extra"]
        # 3. the split points at that chunk: two, then one layer either way, while that wins
        for step in (2, 1):
            improved = True
            while improved and len(self.runs) < max_runs:
                improved = False
                cur = min((x for x in self.runs if x["ok"] and x["chunk"] == chunk and x["extra"] == extra),
                          key=lambda x: x["request_s"])
                for i in range(len(cur["splits"])):
                    for d in (-step, step):
                        cand = list(cur["splits"])
                        cand[i] += d
                        if not self.valid(cand) or len(self.runs) >= max_runs:
                            continue
                        r = self.measure(cand, chunk, extra)
                        if r["ok"] and r["request_s"] < cur["request_s"]:
                            improved = True
                            break
                    if improved:
                        break
        return self.best()

    def links(self) -> None:
        """main -> node 1 -> node 2 ...: each hop's throughput (64 MiB through the agents) and round trip."""
        say("links:")
        prev = None
        for n in self.nodes:
            t = []
            for _ in range(5):
                t0 = time.perf_counter()
                n.call("/info", timeout=30)
                t.append((time.perf_counter() - t0) * 1000)
            if prev is None:
                data = b"\0" * (64 << 20)
                t0 = time.perf_counter()
                r = n.call("/sink", data=data, timeout=300)
                mbs = len(data) / (time.perf_counter() - t0) / 1e6 if r.get("ok") else None
                frm = "main"
            else:
                r = prev.call("/send", {"url": n.agent, "mib": 64}, timeout=300)
                mbs = r.get("mb_s")
                frm = prev.name
            say(f"  {frm} -> {n.name}: {mbs or 0:.0f} MB/s, agent round trip {min(t):.1f} ms")
            prev = n

    def write(self, best: dict) -> None:
        main_cfg = self.main.config(best["splits"], best["chunk"], self.nodes[0], timing=False, extra=best["extra"])
        (self.outdir / "main.json").write_text(json.dumps(main_cfg, indent=1), encoding="utf-8")
        ranges = self.stages(best["splits"])[1:]
        nodes = []
        for i, (n, (lo, hi)) in enumerate(zip(self.nodes, ranges)):
            req = {"name": n.name, "agent": n.agent, "begin": lo, "port": n.port, "prefill": best["chunk"],
                   "cache": n.cache, "extra": n.extra}
            if i + 1 < len(self.nodes):
                req.update(end=hi, next=self.nodes[i + 1].stage_addr)
            nodes.append(req)
        (self.outdir / "nodes.json").write_text(json.dumps(nodes, indent=1), encoding="utf-8")
        lines = ["# Layer-split tuning", "",
                 f"Goal: a {self.prompt_tokens}-token prompt and a {self.answer_tokens}-token answer "
                 f"(score: the request's seconds).", "",
                 "| stages (layers) | chunk | prefill tok/s | decode tok/s | short decode | request s | per stage: "
                 "prompt chunk ms / window ms |", "|---|---|---|---|---|---|---|"]
        for r in self.runs:
            st = " ".join(f"{a}-{b - 1}" for a, b in r["stages"])
            if not r["ok"]:
                lines.append(f"| {st} | {r['chunk']} | - | - | - | does not run | |")
                continue
            per = "; ".join(f"{s['pf_chunk_ms'] or 0:.0f} / {s['window_ms'] or 0:.1f}" for s in r["per_stage"])
            mark = " **best**" if r is best else ""
            lines.append(f"| {st}{mark} | {r['chunk']} | {r['prefill_tps']:.0f} | {r['decode_tps']:.1f} | "
                         f"{r['short_decode'] or 0:.1f} | {r['request_s']:.1f} | {per} |")
        lines += ["", "Run it: `python3 tools/stage_tune.py apply cluster.json OUTDIR` starts the workers, then serve "
                  "`OUTDIR/main.json` (serve/server.py --config).", ""]
        (self.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def apply(cluster: dict, outdir: Path) -> None:
    tf = cluster.get("token_file")
    token = (Path(tf).read_text().strip() if tf else os.environ.get("STRATA_STAGE_TOKEN", "")).strip()
    nodes = json.loads((outdir / "nodes.json").read_text(encoding="utf-8"))
    for spec in reversed(nodes):   # the last first: a relay connects to the next when it starts
        n = Node({"name": spec["name"], "agent": spec["agent"], "host": "-", "port": spec["port"]}, token)
        req = {k: spec[k] for k in ("begin", "end", "next", "port", "prefill", "cache", "extra") if k in spec}
        r = n.call("/start", req)
        if not r.get("ok"):
            sys.exit(f"{n.name}: " + (r.get("error") or "") + "\n" + "\n".join(r.get("log", [])[-15:]))
        say(f"{n.name}: layers {spec['begin']}-{spec.get('end', N_LAYERS) - 1} up in {r.get('seconds')} s")
    say(f"the workers run; serve {outdir / 'main.json'} (serve/server.py --engine strata --config ...)")


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("tune", "apply"):
        sys.exit(__doc__)
    cluster = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    outdir = Path(sys.argv[3])
    if sys.argv[1] == "apply":
        apply(cluster, outdir)
        return
    t = Tuner(cluster, outdir)
    try:
        best = t.tune()
    finally:
        t.main.stop()
    if best is None:
        sys.exit("stage_tune: no configuration ran; see " + str(t.runs_path))
    t.write(best)
    say(f"best: stages {t.stages(best['splits'])}, chunk {best['chunk']}: prefill {best['prefill_tps']:.0f}, "
        f"decode {best['decode_tps']:.1f}, request {best['request_s']:.1f} s -> {outdir}/main.json, nodes.json, "
        f"report.md")


if __name__ == "__main__":
    main()
