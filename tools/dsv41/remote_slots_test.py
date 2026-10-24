#!/usr/bin/env python3
"""tools/dsv41/remote_slots_test.py - the kept conversations (STRATA_DS4_SLOT_KEEP=1) across strata-dsv41's remote stage,
on one Linux PC and one GPU, over the loopback: a worker (--stage-worker, layers K to the last) and a main process
(--remote-stage 127.0.0.1:PORT --stage-end K, serve) as tools/dsv41/remote_loopback.sh starts them.  It needs a GPU.

    python3 tools/dsv41/remote_slots_test.py --exe build/strata-dsv41 --pack /data/pack2.00 --out /tmp/rs
           [--ids-dir /tmp/slot-ids] [--k 20] [--port 7841] [--ctx 20000] [--gpu 0] [--max-new 24]
           [--engine-args "--vram-experts 0.3 --mmap-experts on --ssd-cache 0 --no-expert-cache"]

Two agents, A and B, behind the same system prompt (the prompts are slot_keep_test.py's) take turns A1 B1 A2:
  single   one process, STRATA_DS4_REUSE=0 (every request a fresh read): the reference tokens, and the prompts every
           other arm sends byte for byte.
  splitref the split (worker + main), STRATA_DS4_REUSE=0: what the split prints without any reuse.
  split    the split with STRATA_DS4_SLOT_KEEP=1 and the shared start marked (pin): B1 takes the model from A (A's state
           is set aside, the worker keeping its part under the main's slot id), and A2 takes A's conversation back.
  lost     the same, but the worker is restarted between B1 and A2: it holds no part of A's slot, so A2 must be read
           from token 0 (the main drops the slots when the link was made again), and still print the same tokens.
  miss     the worker keeps one part only (STRATA_DS4_SLOT_CAP=1, STRATA_DS4_SLOT_GB=0): A, B, then an unrelated C make
           it let A's part go while the main process still lists A: A2 meets "not held" on the worker, nothing is
           changed on either side, and the prompt goes on from the shared start (or token 0), with the same tokens.
Checks (exit 0 when all pass, 1 when one fails, 2 when the engines did not start):
  - A2 reuses its whole earlier conversation (prompt - the new message - 1 tokens) in `split`, i.e. the main's DONE
    line reports the hit, and the worker's log says it kept slot N and took it back (both sides hit);
  - `split` prints `splitref`'s tokens (the slot mechanism changes nothing: a check) and `single`'s tokens (told, not
    a check: each process places its own experts, so a near tie can differ between the split and one process - the
    KL of remote_loopback.sh's IDS check is the finer measure);
  - `lost`, `miss`: A2 reuses less than the slot would give and prints the same tokens as `splitref`.
The GPU is shared by the worker and the main process (VRAM share --vram-experts, the worker loads first); `single`
runs with the worker stopped.  Run it where STRATA_STAGE_TOKEN may be made up (it is, unless set).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slot_keep_test as SK   # noqa: E402  (the prompts, the serve protocol's Engine, run())


class Worker:
    """`strata-dsv41 --stage-worker`: started, waited for its "remote stage worker:" line, stopped."""

    def __init__(self, exe: str, args: list[str], out: Path, name: str, env: dict):
        self.cmd, self.env = [exe, *args], env
        self.err_path = out / f"{name}.err"
        self.out_path = out / f"{name}.out"
        self.proc = None

    def start(self, timeout: float = 1800.0) -> None:
        env = dict(os.environ)
        env.update(self.env)
        with open(self.out_path, "wb") as so, open(self.err_path, "wb") as se:
            self.proc = subprocess.Popen(self.cmd, stdout=so, stderr=se, env=env)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the worker stopped while loading (see {self.err_path})")
            if "remote stage worker:" in self.err_path.read_text(encoding="utf-8", errors="replace"):
                return
            time.sleep(2)
        raise RuntimeError("the worker did not come up")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def log(self) -> list[str]:
        try:
            return self.err_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []


def main() -> int:
    ap = argparse.ArgumentParser(description="slots across the remote stage, serve level, one GPU, loopback")
    ap.add_argument("--exe", default=str(SK.ROOT / "build" / "strata-dsv41"))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True, help="logs and the slot folders go here")
    ap.add_argument("--ids-dir", default="")
    ap.add_argument("--k", type=int, default=20, help="the worker's first layer")
    ap.add_argument("--port", type=int, default=7841)
    ap.add_argument("--ctx", type=int, default=20000)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--len-sys", type=int, default=6000)
    ap.add_argument("--len-task", type=int, default=300)
    ap.add_argument("--len-turn", type=int, default=150)
    ap.add_argument("--engine-args", default="--vram-experts 0.3 --mmap-experts on --ssd-cache 0 --no-expert-cache",
                    help="flags for every process (worker, main and the single one)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(a.exe):
        print(f"{a.exe}: no engine - build it first", flush=True)
        return 2
    ids_dir = Path(a.ids_dir or tempfile.mkdtemp(prefix="slot-ids-"))
    try:
        P = SK.prompts(ids_dir, a.pack, a.len_sys, a.len_task, a.len_turn, 2)
    except (OSError, ImportError, ValueError) as e:
        print(f"the prompts cannot be made: {e}", flush=True)
        return 2
    PIN = len(P["sys"])
    eng = shlex.split(a.engine_args)
    thr = str(max(1, (os.cpu_count() or 2) // 2))
    base_env = {"CUDA_VISIBLE_DEVICES": a.gpu, "DS4_ENGRAM_CACHE_MB": os.environ.get("ENGRAM_MB", "64"),
                "STRATA_STAGE_TOKEN": os.environ.get("STRATA_STAGE_TOKEN", f"slots-{os.getpid()}-{time.time_ns()}")}
    os.environ.update(base_env)   # (SK.run() starts the main process with this environment)
    common = ["--cpu-threads", thr, *eng]
    split_args = [*common, "--max-context", str(a.ctx), "--remote-stage", f"127.0.0.1:{a.port}", "--stage-end", str(a.k)]
    single_args = [*common, "--max-context", str(a.ctx), "--prefill-chunk", "0"]

    def worker(name: str, extra: dict | None = None) -> Worker:
        return Worker(a.exe, ["--pack", a.pack, *common, "--vision", "off", "--max-context", str(a.ctx), "--stage-worker",
                              str(a.port), "--stage-bind", "127.0.0.1", "--stage-begin", str(a.k)],
                      out, name, {**base_env, "STRATA_DS4_SLOT_DIR": str(out / f"{name}-slots"), **(extra or {})})

    ok = True

    def check(what: str, cond: bool) -> None:
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
        ok = ok and cond

    # A1 B1 A2: an agent's prompt is its earlier prompt + the answer it got (the `single` arm's) + the new message
    ref: dict = {}

    def prompt(x: str, i: int):
        def make(ans, x=x, i=i):
            src = ref if ref else ans
            p = P["sys"] + P["task" + x]
            for j in range(1, i):
                p = p + src[f"{x}{j}"]["tokens"] + P[f"turn{x}{j}"]
            return p
        return make

    steps = [("A1", prompt("A", 1)), ("B1", prompt("B", 1)), ("A2", prompt("A", 2))]
    wk = None
    try:
        print(f"prompts: system+tools {PIN} tok, task {len(P['taskA'])}, turn {len(P['turnA1'])}; K={a.k}, pin={PIN}",
              flush=True)
        print("single: one process, every request a fresh read (the reference)", flush=True)
        os.environ["DS4_CHUNK_MIN"] = "100000"   # (as remote_loopback.sh's single run: one read, no chunks)
        try:
            single, _, load_s = SK.run(a.exe, a.pack, single_args, str(out / "single-slots"), False, steps, a.max_new,
                                       pin=PIN, reuse=False)
        finally:
            del os.environ["DS4_CHUNK_MIN"]
        print(f"        load {load_s:.0f} s", flush=True)
        ref.update(single)

        wk = worker("worker")
        wk.start()
        print("splitref: the split, every request a fresh read", flush=True)
        splitref, _, _ = SK.run(a.exe, a.pack, split_args, str(out / "splitref-slots"), False, steps, a.max_new,
                                pin=PIN, reuse=False)

        print("split: the split with slots on: A1, B1 (A set aside), A2 (A taken back)", flush=True)
        n_log = len(wk.log())
        split, notes, _ = SK.run(a.exe, a.pack, split_args, str(out / "split-slots"), True, steps, a.max_new, pin=PIN)
        want = split["A2"]["prompt_len"] - len(P["turnA1"]) - 1
        for k in ("A1", "B1", "A2"):
            print(f"        {k}: prompt {split[k]['prompt_len']:>6}  first token {split[k]['ttft_ms'] / 1000:6.2f} s  "
                  f"reused {split[k]['reused']:>6}", flush=True)
        wlog = wk.log()[n_log:]
        kept = [l for l in wlog if " kept (" in l and "slot" in l]
        back = [l for l in wlog if "taken back" in l]
        check(f"split: B1 reads only its own task (reused {split['B1']['reused']} of {PIN})", split["B1"]["reused"] == PIN)
        check(f"split: A2 reuses its whole conversation (main reports {split['A2']['reused']} of {want})",
              split["A2"]["reused"] == want)
        check(f"split: the worker kept a slot ({len(kept)} 'kept' line(s) in its log)", len(kept) >= 1)
        check(f"split: the worker took it back ({len(back)} 'taken back' line(s))", len(back) >= 1)
        check("split: the same tokens as the split without reuse", all(split[k]["tokens"] == splitref[k]["tokens"] for k in split))
        diff = [k for k in split if split[k]["tokens"] != single[k]["tokens"]]
        print("        tokens against one process: " + ("the same" if not diff else f"{diff} differ (placement; not a failure)"),
              flush=True)
        for l in kept + back:
            print("        worker: " + l.strip(), flush=True)

        c_prompt = P["sys"] + P["turnB1"] + P["turnA1"]   # (an unrelated conversation behind the same system prompt)

        def scripted(label: str, seq: list, restart_after: str | None) -> dict:
            """One main process driven by hand (SK.run() starts and ends one process, and the worker has to go or
            change while the main process stays): the requests of `seq`, the worker restarted after `restart_after`."""
            nonlocal wk
            env = {"STRATA_DS4_SLOT_KEEP": "1", "STRATA_DS4_SLOT_DIR": str(out / f"{label}-slots"),
                   "STRATA_DS4_SLOT_MIN": "256", "STRATA_DS4_REUSE": "1"}
            e = SK.Engine(a.exe, ["--pack", a.pack, "--dspark", "off", "--threads", "on", "--vision", "off",
                                  "--expert-profile", "none", *split_args, "--serve"], env)
            got: dict = {}
            try:
                e.start(3600.0)
                for name, make in seq:
                    r = e.request(make(got), a.max_new, pin=PIN)
                    if "error" in r:   # (the first request after a restart may find the link dead: asked again it connects)
                        print(f"        {name} the first time: {r['error']}; asked again", flush=True)
                        r = e.request(make(got), a.max_new, pin=PIN)
                    got[name] = r
                    if name == restart_after:
                        wk.stop()
                        wk = worker("worker-" + label + "-2")
                        wk.start()
            finally:
                try:
                    e.quit()
                except RuntimeError as ex:
                    print("   ", ex, flush=True)
                e.stderr_lines()
            for name, _ in seq:
                if name in got and "reused" in got[name]:
                    print(f"        {name}: prompt {got[name]['prompt_len']:>6}  reused {got[name]['reused']}", flush=True)
            got["notes"] = e.notes
            return got

        print("lost: the worker is restarted between B1 and A2: it holds no part of A's slot", flush=True)
        lost = scripted("lost", steps, "B1")
        check(f"lost: A2 does not use the slot (reused {lost['A2'].get('reused')} of the {want} it would give)",
              lost["A2"].get("reused") is not None and lost["A2"]["reused"] < want)
        check("lost: A2 still prints the same tokens as the split without reuse",
              lost["A2"].get("tokens") == splitref["A2"]["tokens"])
        check("lost: the main process told it removed the slots", any("holds no part" in n for n in lost["notes"]))

        print("miss: the worker keeps one part only: A's goes when C's conversation is set aside, A2 finds it gone", flush=True)
        wk.stop()
        wk = worker("worker-miss", {"STRATA_DS4_SLOT_CAP": "1", "STRATA_DS4_SLOT_GB": "0"})
        wk.start()
        miss = scripted("miss", [steps[0], steps[1], ("C1", lambda _: c_prompt), steps[2]], None)
        check(f"miss: A2 does not get the slot back (reused {miss['A2'].get('reused')} of the {want} it would give)",
              miss["A2"].get("reused") is not None and miss["A2"]["reused"] < want)
        check("miss: A2 still prints the same tokens as the split without reuse",
              miss["A2"].get("tokens") == splitref["A2"]["tokens"])
        check("miss: the main process went on (no error reached the request)", "error" not in miss["A2"])
    except (RuntimeError, OSError, KeyError) as ex:
        print(f"FAIL: {ex!r}", flush=True)
        ok = False
    finally:
        if wk is not None:
            wk.stop()
    print("PASS: the kept conversations work across the split" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
