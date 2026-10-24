"""Ship a stage worker's share of a model to a node (tools/stage_node.py): only the bytes its layers need.

A node that runs layers [lo, hi) reads only those layers' tensors from the model's GGUF shards (the expert arena
loads its own layers, the dense weights come from the pack).  This sends, from the PC that holds the whole model:
  - every shard's header and the tensors of layers lo..hi-1, at their own offsets in files of the shards' sizes
    (sparse on the node: the other layers are holes), plus the small non-layer tensors (token embedding, output);
    the per-layer token embeddings (the PLE, ~27 GB) only to a node that runs layer 1 - a worker never does;
  - the pack directory, the expert profile and (optional) the engine's directory, whole.
What the node has already (its /ranges) is not sent again, and every piece's sha256 is compared on arrival.

    python3 tools/stage_ship.py --agent http://192.0.2.12:7840 --token-file /opt/strata/stage-token \\
        --model-dir /data/models/<model> --pack-dir /data/packs/<pack> \\
        --profile /opt/strata/data/expert-profile.bin [--bin-dir /opt/strata/build/bin] \\
        --layers 36-47 [--data-dir <the node's data_dir>] [--parallel 4]

The pieces (64 MiB) overlap: a reader thread reads and hashes them ahead, and --parallel senders (default 4) put
them, each on its own HTTP/1.1 keep-alive connection, so the disk, the hashing, the link and the node's writes run
at once and a slow link stays busy.  At most parallel + 3 pieces are in memory (448 MiB with 4).  The first piece
that fails (an HTTP error, a sha256 that differs) stops the run; what arrived is in the node's /ranges, so the same
command resumes.

On the node the files land under its data_dir: models/<model dir name>/, packs/<pack name>/, data/<profile>,
bin/.  A /start that names the model ({"model": {"dir": ..., "native": ..., "pack": ...}}, as stage_tune.py sends)
runs it from there; --data-dir prints the paths for a node.json that should run it by default.

DeepSeek-V4.1 (--engine dsv41): the model is the pack (tools/dsv41/pack_gguf.py), whose tensors.txt names the weights'
files under ds4.txt's model_dir.  A worker for layers LO-HI opens every file and reads only its layers' tensors, so
those bytes are sent at their own offsets (the shards of other layers, the Engram and the drafter: empty sparse files of
their sizes), and the pack folder whole; --model-dir names the weights' folder when ds4.txt's is not this PC's.

    python3 tools/stage_ship.py --engine dsv41 --agent http://192.0.2.12:7840 --token-file /opt/strata/stage-token \\
        --pack-dir /data/packs/<pack> [--bin-dir /opt/strata/build/bin] --layers 20- [--data-dir <the node's data_dir>]

--layers LO- runs to the trunk's last layer.  The node agent writes its own folder into the pack's ds4.txt when /start
runs the worker (tools/stage_node.py, "engine": "dsv41").
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import posixpath
import queue
import re
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

PIECE = 64 << 20
READ_AHEAD = 2   # pieces read and hashed ahead of the senders


class Item:
    """One file to ship: the byte ranges the node needs of it (None: the whole file, sent again when it changed)."""

    def __init__(self, local: Path, rel: str, ranges: list[list[int]] | None = None):
        self.local = local
        self.rel = rel
        self.total = local.stat().st_size
        self.whole = ranges is None
        self.ranges = merge(ranges if ranges is not None else [[0, self.total]])
        self.fp: str | None = None   # a sparse file's source: what its byte ranges on the node belong to
        self.mode = "755" if local.stat().st_mode & stat.S_IXUSR else None

    @property
    def nbytes(self) -> int:
        return sum(n for _, n in self.ranges)


def merge(rs: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for off, n in sorted(rs):
        if n <= 0:
            continue
        if out and off <= out[-1][0] + out[-1][1]:
            out[-1][1] = max(out[-1][1], off + n - out[-1][0])
        else:
            out.append([off, n])
    return out


def subtract(want: list[list[int]], have: list[list[int]]) -> list[list[int]]:
    """want minus have (both merged)."""
    out = []
    for off, n in want:
        cur, end = off, off + n
        for h0, hn in have:
            h1 = h0 + hn
            if h1 <= cur or h0 >= end:
                continue
            if h0 > cur:
                out.append([cur, h0 - cur])
            cur = max(cur, h1)
            if cur >= end:
                break
        if cur < end:
            out.append([cur, end - cur])
    return out


def gguf_items(model_dir: Path, lo: int, hi: int) -> list[Item]:
    """Every shard: its header, layers [lo, hi), the non-layer tensors but the PLE (unless layer 1 is in range)."""
    items = []
    for path in sorted(model_dir.glob("*.gguf")):
        g = GGUFFile(path)
        rs = [[0, g.data_start]]
        for t in g.tensors:
            m = re.match(r"blk\.(\d+)\.", t.name)
            if m:
                keep = lo <= int(m.group(1)) < hi
            else:
                keep = not t.name.startswith("per_layer_token_embd") or lo <= 1 < hi
            if keep:
                n = t.expected_bytes()
                if n is None:
                    raise SystemExit(f"{path.name}: {t.name}: unknown size ({t.type_name})")
                rs.append([g.data_start + t.offset, n])
        it = Item(path, f"models/{model_dir.name}/{path.name}", rs)
        with open(path, "rb") as f:   # the header names every tensor's offset: same header, same layout
            it.fp = hashlib.sha256(f.read(g.data_start)).hexdigest()[:32] + f"-{it.total}"
        items.append(it)
    return items


def dsv41_pack_files(pack_dir: Path) -> tuple[dict[str, str], dict[int, str], list[tuple[str, int, int, int]]]:
    """A DeepSeek-V4.1 pack's description (tools/dsv41/pack_gguf.py): ds4.txt's keys, tensors.txt's "file I NAME" lines
    (the weights' files, relative to ds4.txt's model_dir - the shared Engram and drafter files as "../<dir>/x.gguf") and
    its tensors as (name, file, offset, bytes)."""
    kv: dict[str, str] = {}
    for line in (pack_dir / "ds4.txt").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition(" ")
        if v:
            kv[k] = v
    files: dict[int, str] = {}
    tensors: list[tuple[str, int, int, int]] = []
    with open(pack_dir / "tensors.txt", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[0] == "file":
                files[int(p[1])] = p[2]
            elif len(p) >= 6 and p[0] == "t":
                tensors.append((p[1], int(p[3]), int(p[4]), int(p[5])))
    if not files or not tensors:
        raise SystemExit(f"{pack_dir}: tensors.txt names no files or no tensors")
    return kv, files, tensors


def dsv41_wanted(name: str, lo: int, hi: int) -> bool:
    """Whether a worker for layers [lo, hi) reads this tensor (src/dsv41/ds4_model.cu: tensor_here, and the loads it
    skips): the trunk layers' own - with the Engram tables' tensors of an Engram layer among them (the layers 1 and 14,
    so none for a worker from layer 15 on) - and the few outside the layers; not the embedding table (the main process
    embeds), the head (it keeps the logits), or the DSpark drafter (mtp.*: a range runs without it)."""
    if name.startswith("layers."):
        try:
            return lo <= int(name.split(".", 2)[1]) < hi
        except ValueError:
            return False
    return not (name == "embed.weight" or name.startswith(("head.", "mtp.")))


def dsv41_items(pack_dir: Path, model_dir: Path | None, lo: int, hi: int | None = None) -> list[Item]:
    """Every file the pack names, in the node's models/<model dir name>/ at the names tensors.txt gives: the bytes of
    the tensors a worker for layers [lo, hi) reads at their own offsets, in files of the originals' sizes (sparse on the
    node); a file with no such tensor (the Engram's, the drafter's, a trunk shard of other layers) as an empty file of
    its size, because the engine opens and maps every file tensors.txt names.  The pack's folder is sent whole."""
    kv, files, tensors = dsv41_pack_files(pack_dir)
    nl = int(kv.get("num_hidden_layers", "0"))
    hi = nl if hi is None else hi
    if not 0 < lo < hi <= nl:
        raise SystemExit(f"layers [{lo}, {hi}): a worker needs a range of the trunk's {nl} layers, from layer 1 on")
    model_dir = model_dir or Path(kv.get("model_dir", ""))
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir}: the weights' folder (ds4.txt's model_dir, or --model-dir) is not here")
    ranges: dict[int, list[list[int]]] = {i: [] for i in files}
    for name, i, off, n in tensors:
        if dsv41_wanted(name, lo, hi):
            ranges[i].append([off, n])
    items = []
    for i, fn in sorted(files.items()):
        rel = posixpath.normpath(posixpath.join("models", model_dir.name, fn.replace("\\", "/")))
        if not rel.startswith("models/") or rel.count("/") < 2:
            raise SystemExit(f"{fn}: tensors.txt names a file outside the models folder")
        it = Item(model_dir / fn, rel, ranges[i])
        with open(it.local, "rb") as f:   # no header to fingerprint (the engine reads none): the first MiB and the size
            it.fp = hashlib.sha256(f.read(1 << 20)).hexdigest()[:32] + f"-{it.total}"
        items.append(it)
    return items


def tree_items(src: Path, prefix: str) -> list[Item]:
    if src.is_file():
        return [Item(src, f"{prefix}/{src.name}")]
    return [Item(p, f"{prefix}/{src.name}/{p.relative_to(src).as_posix()}")
            for p in sorted(src.rglob("*")) if p.is_file()]


class Agent:
    """A node agent.  Every request goes to it directly, never through an HTTP proxy from the environment (the
    /puts use http.client, which knows no proxies; the agent is a LAN or tunnel address a proxy cannot reach)."""
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        u = urllib.parse.urlsplit(self.url)
        self.scheme, self.netloc, self.base = u.scheme, u.netloc, u.path

    def sha256(self, rel: str) -> dict:
        req = urllib.request.Request(f"{self.url}/sha256?path={urllib.parse.quote(rel)}",
                                     headers={"X-Stage-Token": self.token})
        with self.direct.open(req, timeout=600) as r:
            return json.load(r)

    def ranges(self, rel: str) -> dict:
        req = urllib.request.Request(f"{self.url}/ranges?path={urllib.parse.quote(rel)}",
                                     headers={"X-Stage-Token": self.token})
        with self.direct.open(req, timeout=60) as r:
            return json.load(r)

    def connect(self) -> http.client.HTTPConnection:
        """A connection for /put, kept alive between pieces (an agent that answers HTTP/1.0 closes it after each
        request: http.client then opens the next one itself)."""
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.netloc, timeout=600)

    def put(self, conn: http.client.HTTPConnection, item: Item, off: int, data: bytes, sha: str | None = None) -> dict:
        """`sha`: the piece's sha256 - an agent that knows the parameter records the range only when the bytes
        that arrived hash to it (an older one ignores it; the answer's sha256 is compared here either way)."""
        q = f"path={urllib.parse.quote(item.rel)}&offset={off}&total={item.total}"
        if item.fp:
            q += f"&fp={urllib.parse.quote(item.fp)}"
        if item.mode:
            q += f"&mode={item.mode}"
        if sha:
            q += f"&sha256={sha}"
        view = memoryview(data)
        for attempt in (0, 1):
            reused = conn.sock is not None
            try:
                # the body 1 MiB at a time, with its length given (no chunked encoding): the socket's timeout is
                # then a stall's, not the whole 64 MiB's - one piece of `parallel` on a slow link takes long
                conn.request("POST", f"{self.base}/put?{q}",
                             body=(view[i:i + (1 << 20)] for i in range(0, len(view), 1 << 20)),
                             headers={"X-Stage-Token": self.token, "Content-Type": "application/octet-stream",
                                      "Content-Length": str(len(view))})
                r = conn.getresponse()
                body = r.read()
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                if reused and attempt == 0:   # a kept-alive connection the agent had closed: once more, anew
                    continue
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if r.status != 200:
                return {"ok": False, "error": f"HTTP {r.status}: {body.decode(errors='replace')[:300]}"}
            try:
                return json.loads(body)
            except ValueError:
                return {"ok": False, "error": f"not JSON: {body[:300]!r}"}
        return {"ok": False, "error": "unreachable"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ship(agent: Agent, items: list[Item], say=print, parallel: int = 4) -> int:
    """Send what the node lacks; returns the bytes sent.  A reader thread reads and hashes the pieces READ_AHEAD
    ahead, `parallel` senders put them (each on its own connection): at most parallel + READ_AHEAD + 1 pieces in
    memory.  The first piece that fails stops the run (SystemExit) once the pieces in flight are answered."""
    todo = []
    for it in items:
        have = agent.ranges(it.rel)
        same = have.get("size") == it.total and (it.fp is None or have.get("fp") in (None, it.fp))
        got = merge(have.get("ranges") or []) if same else []   # another source: everything again
        miss = subtract(it.ranges, got)
        if it.total == 0:   # an empty file: one empty put creates it, unless the node has it (size None: missing)
            miss = [] if have.get("size") == 0 else [[0, 0]]
        elif not it.ranges:   # a file the node opens but reads none of (a DeepSeek pack's): created at its size, a hole
            miss = [] if same and have.get("size") == it.total else [[0, 0]]
        if it.whole and not miss and it.total > 0:   # a whole file the node has: the same bytes?
            remote = agent.sha256(it.rel)
            if remote.get("sha256") != file_sha256(it.local):
                miss = list(it.ranges)
        todo.append((it, miss))
    need = sum(n for _, miss in todo for _, n in miss)
    say(f"ship: {sum(it.nbytes for it in items) / 2**30:.2f} GiB wanted, {need / 2**30:.2f} GiB to send "
        f"({len(items)} files)")
    pieces = []   # (item, its range's index in left, offset, bytes)
    left = []     # each range: its pieces not yet answered (a progress line when none is left)
    for it, miss in todo:
        for off, n in miss:
            pos, end = off, off + n
            left.append(0)
            while True:
                k = min(PIECE, end - pos)
                pieces.append((it, len(left) - 1, pos, k))
                left[-1] += 1
                pos += k
                if pos >= end:
                    break
    if not pieces:
        return 0
    parallel = max(1, parallel)
    work: queue.Queue = queue.Queue(READ_AHEAD)   # (piece, its bytes, their sha256); None: no more
    done: queue.Queue = queue.Queue()             # (piece, None) answered, or (None, the error)
    stop = threading.Event()

    def offer(x) -> bool:
        while not stop.is_set():
            try:
                work.put(x, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def reader() -> None:
        f, cur = None, None
        try:
            for p in pieces:
                it, _, pos, n = p
                if it is not cur:
                    if f is not None:
                        f.close()
                    f, cur = open(it.local, "rb"), it
                f.seek(pos)
                data = f.read(n)
                if len(data) != n:
                    raise ValueError(f"{it.local} ended at {pos + len(data)} (it changed while shipping)")
                if not offer((p, data, hashlib.sha256(data).hexdigest())):
                    return
                del data
        except BaseException as e:   # noqa: BLE001 - every failure ends the run with its reason
            done.put((None, f"ship: {e}"))
        finally:
            if f is not None:
                f.close()
            for _ in range(parallel):
                offer(None)

    def sender() -> None:
        conn = None
        try:
            conn = agent.connect()
            while not stop.is_set():
                try:
                    x = work.get(timeout=0.5)
                except queue.Empty:
                    continue
                if x is None:
                    return
                p, data, sha = x
                del x
                it, _, pos, _ = p
                r = agent.put(conn, it, pos, data, sha)
                del data
                if not r.get("ok") or r.get("sha256") != sha:
                    done.put((None, f"ship: {it.rel} at {pos}: {r.get('error') or 'sha256 differs'}"))
                    return
                done.put((p, None))
        except BaseException as e:   # noqa: BLE001 - not a thread that ends without a word
            done.put((None, f"ship: {e}"))
        finally:
            if conn is not None:
                conn.close()

    threads = [threading.Thread(target=reader, daemon=True)]
    threads += [threading.Thread(target=sender, daemon=True) for _ in range(parallel)]
    for t in threads:
        t.start()
    sent, t0, todo_n, err = 0, time.time(), len(pieces), None
    try:
        while todo_n and err is None:
            try:
                p, err = done.get(timeout=1)   # (a wait with a timeout: Ctrl+C gets through on Windows too)
            except queue.Empty:
                continue
            if p is None:
                continue
            it, k, _, n = p
            todo_n -= 1
            sent += n
            left[k] -= 1
            if left[k] == 0:
                el = time.time() - t0
                say(f"  {it.rel}: {sent / 2**30:.2f} of {need / 2**30:.2f} GiB, "
                    f"{sent / max(el, 1e-6) / 1e6:.0f} MB/s")
    except BaseException:
        stop.set()   # Ctrl+C: not waiting for the pieces in flight (the threads end with the process)
        raise
    stop.set()       # done, or a failure: the pieces in flight are answered first
    if err:          # said now: a stalled piece in flight can take its timeout before the run ends
        say(f"{err} - waiting for the pieces in flight")
    for t in threads:   # a join with a timeout: Ctrl+C gets through on Windows while a sender waits on a node
        while t.is_alive():
            t.join(1)
    if err:
        raise SystemExit(err)
    return sent


def plan(model_dir: Path, pack_dir: Path, profile: Path, bin_dir: Path | None, lo: int, hi: int) -> list[Item]:
    items = gguf_items(model_dir, lo, hi)
    items += tree_items(pack_dir, "packs")
    items += tree_items(profile, "data")
    if bin_dir is not None:
        items += [Item(p, f"bin/{p.name}") for p in sorted(bin_dir.iterdir()) if p.is_file()]
    return items


def plan_dsv41(pack_dir: Path, model_dir: Path | None, bin_dir: Path | None, lo: int, hi: int | None) -> list[Item]:
    """What a DeepSeek-V4.1 worker for layers [lo, hi) needs: its weights' bytes and the pack (its expert profile is in
    it: <pack>/expert-profile.bin)."""
    items = dsv41_items(pack_dir, model_dir, lo, hi)
    items += tree_items(pack_dir, "packs")
    if bin_dir is not None:
        items += [Item(p, f"bin/{p.name}") for p in sorted(bin_dir.iterdir()) if p.is_file()]
    return items


def node_paths_dsv41(data_dir: str, model_name: str, pack_dir: Path, begin: int) -> dict:
    """node.json's paths for a shipped DeepSeek-V4.1 pack, and the /start that runs its worker."""
    win = bool(re.match(r"[A-Za-z]:", data_dir)) or "\\" in data_dir
    sep = "\\" if win else "/"
    d = data_dir.rstrip("/\\")
    j = lambda *parts: sep.join([d, *parts])   # noqa: E731
    return {"node.json": {"dsv41_exe": j("bin", "strata-dsv41.exe" if win else "strata-dsv41"), "lib_dirs": [j("bin")],
                          "dsv41_pack": j("packs", pack_dir.name)},
            "/start": {"engine": "dsv41", "begin": begin, "model": {"dir": model_name, "pack": pack_dir.name}}}


def node_paths(data_dir: str, model_dir: Path, pack_dir: Path, profile: Path) -> dict:
    """node.json's paths for what plan() ships."""
    shards = sorted(p.name for p in model_dir.glob("*.gguf"))
    win = bool(re.match(r"[A-Za-z]:", data_dir)) or "\\" in data_dir   # a Windows node: its own separator, .exe
    sep = "\\" if win else "/"
    d = data_dir.rstrip("/\\")
    j = lambda *parts: sep.join([d, *parts])   # noqa: E731
    return {"exe": j("bin", "strata.exe" if win else "strata"), "lib_dirs": [j("bin")],
            "pack": j("packs", pack_dir.name), "native": j("models", model_dir.name, shards[0]),
            "ple_gguf": j("models", model_dir.name, shards[1]) if len(shards) > 1 else "",
            "expert_profile": j("data", profile.name)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("strata", "dsv41"), default="strata",
                    help="the model's engine: strata (GGUF shards + a native pack, the default) or dsv41 (DeepSeek-V4.1)")
    ap.add_argument("--agent")
    ap.add_argument("--token-file")
    ap.add_argument("--model-dir", type=Path, help="strata: the GGUF shards' folder; dsv41: the weights' folder when "
                                                   "the pack's ds4.txt names another PC's")
    ap.add_argument("--pack-dir", required=True, type=Path)
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--bin-dir", type=Path)
    ap.add_argument("--layers", required=True, help="LO-HI, both included (the node's first and last layer); dsv41 "
                                                    "also LO- (to the trunk's last layer)")
    ap.add_argument("--data-dir", help="the node's data_dir: print the node.json paths for what was shipped")
    ap.add_argument("--parallel", type=int, default=4,
                    help="pieces in flight at once, each on its own connection (default 4; 1: one at a time)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the files and the bytes the node would need, ask no agent, send nothing")
    a = ap.parse_args()
    if a.parallel < 1:
        ap.error("--parallel: 1 or more")
    missing = [f for f, v in (("--agent", True if a.dry_run else a.agent),
                              ("--model-dir", a.model_dir if a.engine == "strata" else True),
                              ("--profile", a.profile if a.engine == "strata" else True)) if v is None or v == ""]
    if missing:
        ap.error("the following arguments are required: " + ", ".join(missing))
    token = (Path(a.token_file).read_text().strip() if a.token_file else os.environ.get("STRATA_STAGE_TOKEN", ""))
    lo, _, hi = a.layers.partition("-")
    if a.engine == "dsv41":
        lo, hi = int(lo), (int(hi) + 1 if hi else None)
        items = plan_dsv41(a.pack_dir, a.model_dir, a.bin_dir, lo, hi)
    else:
        lo, hi = int(lo), int(hi)   # (as before: "LO-HI" or an error)
        items = plan(a.model_dir, a.pack_dir, a.profile, a.bin_dir, lo, hi + 1)
    if a.dry_run:
        for it in items:
            print(f"{it.nbytes / 2**30:10.3f} GiB of {it.total / 2**30:9.3f} GiB  {it.rel}")
        print(f"total {sum(it.nbytes for it in items) / 2**30:.3f} GiB wanted ({len(items)} files)")
        return
    t0 = time.time()
    sent = ship(Agent(a.agent, token), items, parallel=a.parallel)
    print(f"ship: {sent / 2**30:.2f} GiB in {time.time() - t0:.0f} s")
    if a.data_dir:
        if a.engine == "dsv41":
            kv, _, _ = dsv41_pack_files(a.pack_dir)
            md = (a.model_dir or Path(kv.get("model_dir", ""))).name
            print("node paths:", json.dumps(node_paths_dsv41(a.data_dir, md, a.pack_dir, lo), indent=1))
        else:
            print("node.json paths:", json.dumps(node_paths(a.data_dir, a.model_dir, a.pack_dir, a.profile), indent=1))


if __name__ == "__main__":
    main()
