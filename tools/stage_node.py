"""A remote-stage node agent: starts and stops this PC's Strata stage workers for the layer-split tuner
(tools/stage_tune.py), and reports the PC's GPU, RAM and CPU and the link speed toward other nodes.  Python's
standard library only, so it runs wherever the engine runs (Linux, WSL).

    python tools/stage_node.py node.json

node.json (paths are this PC's):
    {
      "exe": "/srv/stage/bin/strata",       the engine (a build of the remote-stage fork)
      "lib_dirs": ["/srv/stage/bin"],       added to LD_LIBRARY_PATH (PATH on Windows)
      "pack": ".../packs/rvn-iq3_s", "native": "...-00001-of-00008.gguf", "ple_gguf": "...-00002-of-00008.gguf",
      "expert_profile": ".../expert-profile.bin",
      "bind": "192.0.2.11",                     the address the agent and the workers listen on
      "agent_port": 7840,
      "log_dir": "/srv/stage",
      "args": ["--spec", "4", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"],
      "token_file": "/srv/stage/stage-token",  or STRATA_STAGE_TOKEN in the environment
      "data_dir": "/srv/stage/shipped",       where tools/stage_ship.py writes (optional)
      "drop_cache": true,                            default: drop the model files from the page cache
      "allow_put_bin": false,                        let /put write executables (bin/, mode): the engine shipped
      "extra_allow": []                              engine flags a /start may add beyond the default list
    }

What the token allows: a holder can start and stop this PC's workers with the engine and model files node.json
names, and the flags in EXTRA_ALLOWED (or "extra_allow"); with "data_dir", write files under it; with
"allow_put_bin", also the engine there (that is running code of their choice - for a node you ship the engine to).
The token and the traffic are plain text: a LAN you trust, or a VPN tunnel.

The page cache: a worker reads its layers' experts through the file cache and copies them into pinned memory, and
a shipped file passes through the cache too - left there, a 10-layer worker would take its experts' RAM twice (and
under WSL the VM keeps the cache from Windows).  So the agent drops the files it wrote and the files a worker read
from the cache once the worker listens (posix_fadvise DONTNEED: no root; mapped pages stay).  "drop_cache": false
keeps them (a faster restart with the same layers).

Every request carries the shared token in the X-Stage-Token header (the workers get it as STRATA_STAGE_TOKEN), so
only the tuner and the main process drive this PC.  Endpoints (JSON in and out):
    GET  /info                 the GPU (nvidia-smi), RAM, CPU and the workers running here
    POST /start                {"begin": K, "end": K2?, "next": "host:port"?, "port": 7841, "prefill": 5888,
                                "cache": "auto"|slots, "extra": [...]} - stops the worker on that port, starts this one
                               and answers when it listens (or with its log's tail when it exits)
    POST /stop                 {"port": 7841} (no port: every worker)
    GET  /log?port=P&lines=N   a worker's log tail
    POST /sink                 reads the body; answers its bytes and the seconds it took (a link test toward here)
    POST /send                 {"url": "http://other:7840", "mib": 64} - sends to another node's /sink
    POST /put?path=P&offset=N&total=T[&mode=755]
                               writes the body at byte N of data_dir/P (a file of T bytes, sparse where nothing
                               was written) and answers the body's sha256; P.ranges.json records what is written
    GET  /ranges?path=P        the byte ranges of data_dir/P written so far ([[offset, length], ...]) and its size
    GET  /sha256?path=P        the sha256 of data_dir/P (a whole file shipped again only when it changed)
The shipped model files keep every offset of the originals, so a node that holds only its layers' bytes (and
every shard's header) loads them as from the whole file - tools/stage_ship.py sends only those.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# engine flags a /start may add (the tuner's): no flag that names a file to write or read
EXTRA_ALLOWED = {"--max-context", "--stage-bind", "--vram-reserve-mib", "--vram-reserve-later-mib", "--spec",
                 "--kv", "--kv-resident", "--pcie-frac", "--pool-workers", "--adapt-every", "--adapt-swaps",
                 "--adapt-decay", "--spec-min-p"}
LISTEN_TIMEOUT_S = 600     # loading a worker's experts: up to ~30 GB from disk on a cold cache
STOP_TIMEOUT_S = 180       # a worker with ~30 GB of pinned experts takes a while to unpin


def load_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    token = os.environ.get("STRATA_STAGE_TOKEN", "")
    if not token and cfg.get("token_file"):
        token = Path(cfg["token_file"]).read_text(encoding="utf-8").strip()
    if not token:
        sys.exit("stage_node: no token (STRATA_STAGE_TOKEN or node.json's token_file)")
    cfg["_token"] = token
    return cfg


def gpu_info() -> list[dict]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    q = "name,memory.total,memory.free,memory.used,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    try:
        out = subprocess.run([smi, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 7:
            continue
        gpus.append({"name": f[0], "vram_mib": int(f[1]), "free_mib": int(f[2]), "used_mib": int(f[3]),
                     "pcie_gen": f[4], "pcie_width": f[5], "pcie_width_max": f[6]})
    return gpus


def ram_info() -> dict:
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0]) // 1024
        return {"total_mib": mem.get("MemTotal"), "available_mib": mem.get("MemAvailable")}
    except OSError:
        return {}


class Workers:
    """The workers this agent started, by port."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.procs: dict[int, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()   # one /start at a time (two on a port would orphan the first)

    def log_path(self, port: int) -> Path:
        return Path(self.cfg.get("log_dir", ".")) / f"stage-worker-{port}.log"

    def stop(self, port: int | None) -> list[int]:
        with self.lock:
            ports = [port] if port is not None else list(self.procs)
            stopped = []
            for p in ports:
                proc = self.procs.pop(p, None)
                if proc is None or proc.poll() is not None:
                    continue
                proc.terminate()
                try:
                    proc.wait(STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(30)
                stopped.append(p)
            return stopped

    def start(self, req: dict) -> dict:
        with self.start_lock:
            return self._start(req)

    def _start(self, req: dict) -> dict:
        port = int(req.get("port", 7841))
        allowed = EXTRA_ALLOWED | set(self.cfg.get("extra_allow", []))
        extra = [str(a) for a in req.get("extra", [])]
        bad = [a for a in extra if a.startswith("-") and a not in allowed]
        if bad:
            return {"ok": False, "error": f"flags not allowed here: {bad} (node.json extra_allow)"}
        if (req.get("end") is None) != (req.get("next") is None):
            return {"ok": False, "error": "a relay needs both end and next"}
        self.stop(port)
        c = self.cfg
        args = [c["exe"], "--serve", "--stage-worker", str(port), "--stage-begin", str(int(req["begin"]))]
        if req.get("end") is not None:
            args += ["--stage-end", str(int(req["end"])), "--stage-next", str(req["next"])]
        args += ["--pack", c["pack"], "--native", c["native"], "--ple-gguf", c["ple_gguf"],
                 "--expert-profile", c["expert_profile"], "--expert-cache", str(req.get("cache", "auto")),
                 "--prefill", str(int(req.get("prefill", 2048))), "--stage-bind", c["bind"]]
        args += [str(a) for a in c.get("args", [])] + extra
        env = dict(os.environ)
        env["STRATA_STAGE_TOKEN"] = c["_token"]
        env["STRATA_REMOTE_TIMING"] = "1"
        libs = os.pathsep.join(c.get("lib_dirs", []))
        if libs:
            var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[var] = libs + (os.pathsep + env[var] if env.get(var) else "")
        log = self.log_path(port)
        fh = open(log, "wb")
        kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, env=env, **kw)
        fh.close()
        with self.lock:
            self.procs[port] = proc
        t0 = time.time()
        while time.time() - t0 < LISTEN_TIMEOUT_S:
            time.sleep(1)
            text = log.read_text(encoding="utf-8", errors="replace")
            if "listening on" in text:
                if c.get("drop_cache", True):   # its experts are pinned now: the cached file pages are a second copy
                    for f in model_files(c):
                        drop_cache(f)
                return {"ok": True, "port": port, "seconds": round(time.time() - t0, 1), "log": tail(text, 25)}
            if proc.poll() is not None:
                with self.lock:
                    self.procs.pop(port, None)
                return {"ok": False, "port": port, "exit": proc.returncode, "log": tail(text, 40)}
        self.stop(port)
        return {"ok": False, "port": port, "error": f"not listening after {LISTEN_TIMEOUT_S} s",
                "log": tail(log.read_text(encoding="utf-8", errors="replace"), 40)}

    def running(self) -> list[int]:
        with self.lock:
            return [p for p, proc in self.procs.items() if proc.poll() is None]


def tail(text: str, n: int) -> list[str]:
    return text.splitlines()[-n:]


def drop_cache(path: Path, offset: int = 0, length: int = 0) -> None:
    """Drop a file's clean pages from the page cache (Linux; elsewhere nothing)."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def model_files(cfg: dict) -> list[Path]:
    """The files a worker reads: the GGUF shards next to "native", the pack's files."""
    out: list[Path] = []
    if cfg.get("native"):
        out += sorted(Path(cfg["native"]).parent.glob("*.gguf"))
    if cfg.get("pack") and Path(cfg["pack"]).is_dir():
        out += [p for p in Path(cfg["pack"]).rglob("*") if p.is_file()]
    return out


def merge_ranges(rs: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for off, n in sorted(rs):
        if out and off <= out[-1][0] + out[-1][1]:
            out[-1][1] = max(out[-1][1], off + n - out[-1][0])
        else:
            out.append([off, n])
    return out


class Store:
    """data_dir: the shipped files and, next to each, the byte ranges written so far (P.ranges.json)."""

    def __init__(self, cfg: dict):
        self.root = Path(cfg["data_dir"]).resolve() if cfg.get("data_dir") else None
        self.drop = bool(cfg.get("drop_cache", True))
        self.allow_bin = bool(cfg.get("allow_put_bin", False))
        self.lock = threading.Lock()

    def path(self, rel: str) -> Path:
        if self.root is None:
            raise ValueError("this node has no data_dir")
        p = (self.root / rel).resolve()
        if self.root not in p.parents:
            raise ValueError(f"{rel}: outside data_dir")
        return p

    @staticmethod
    def side(p: Path) -> Path:
        return p.with_name(p.name + ".ranges.json")

    def read_side(self, p: Path) -> dict:
        """{"fp": the source's fingerprint, "ranges": [[offset, length], ...]} (an old list: no fingerprint)."""
        s = self.side(p)
        if not s.exists():
            return {"fp": None, "ranges": []}
        d = json.loads(s.read_text())
        return d if isinstance(d, dict) else {"fp": None, "ranges": d}

    def ranges(self, rel: str) -> dict:
        p = self.path(rel)
        d = self.read_side(p)
        return {"ok": True, "path": rel, "size": p.stat().st_size if p.exists() else None, "ranges": d["ranges"],
                "fp": d["fp"]}

    def put(self, rel: str, offset: int, total: int, body, n: int, mode: str | None, fp: str | None) -> dict:
        p = self.path(rel)
        if not self.allow_bin and (mode or rel.split("/", 1)[0] == "bin"):
            return {"ok": False, "error": "this node does not take executables (node.json allow_put_bin)"}
        p.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            d = self.read_side(p) if p.exists() else {"fp": None, "ranges": []}
            # (a sidecar without a fingerprint, from before they existed: kept, and it takes this one)
            if not p.exists() or p.stat().st_size != total or (fp and d["fp"] is not None and d["fp"] != fp):
                # a new file, or a different source (size or fingerprint): what was written of it is void
                with open(p, "ab"):
                    pass
                os.truncate(p, 0)
                os.truncate(p, total)   # sparse: the bytes nobody writes stay holes
                self.side(p).write_text(json.dumps({"fp": fp, "ranges": []}))
        h = hashlib.sha256()
        with open(p, "r+b") as f:
            f.seek(offset)
            left = n
            while left > 0:
                chunk = body.read(min(left, 4 << 20))
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                left -= len(chunk)
            if self.drop:   # written through: not left in the page cache
                f.flush()
                os.fsync(f.fileno())
        if self.drop:
            drop_cache(p, offset, n)
        if left:
            return {"ok": False, "error": f"the body ended {left} bytes short"}
        if mode:
            os.chmod(p, int(mode, 8))
        with self.lock:
            d = self.read_side(p)
            self.side(p).write_text(json.dumps({"fp": d["fp"] or fp, "ranges": merge_ranges(d["ranges"] + [[offset, n]])}))
        return {"ok": True, "sha256": h.hexdigest(), "bytes": n}


def make_handler(cfg: dict, workers: Workers, store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # one line per request on stderr
            sys.stderr.write("stage_node: %s %s\n" % (self.address_string(), fmt % args))

        def reply(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self) -> bool:
            if hmac.compare_digest(self.headers.get("X-Stage-Token", "").encode(), cfg["_token"].encode()):
                return True
            self.reply(403, {"ok": False, "error": "wrong or missing X-Stage-Token"})
            return False

        def body_json(self) -> dict:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if not self.authorized():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/info":
                self.reply(200, {"ok": True, "host": platform.node(), "system": platform.platform(),
                                 "cpus": os.cpu_count(), "gpus": gpu_info(), "ram": ram_info(),
                                 "workers": workers.running(), "bind": cfg["bind"]})
            elif u.path == "/sha256":
                try:
                    p = store.path(q["path"][0])
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        for b in iter(lambda: f.read(8 << 20), b""):
                            h.update(b)
                    self.reply(200, {"ok": True, "sha256": h.hexdigest(), "size": p.stat().st_size})
                except (KeyError, ValueError, OSError) as e:
                    self.reply(200, {"ok": False, "error": str(e)})
            elif u.path == "/ranges":
                try:
                    self.reply(200, store.ranges(q["path"][0]))
                except (KeyError, ValueError, OSError) as e:
                    self.reply(400, {"ok": False, "error": str(e)})
            elif u.path == "/log":
                port = int(q.get("port", ["7841"])[0])
                lines = int(q.get("lines", ["200"])[0])
                p = workers.log_path(port)
                text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                self.reply(200, {"ok": True, "port": port, "log": tail(text, lines)})
            else:
                self.reply(404, {"ok": False, "error": "unknown path"})

        def do_POST(self):
            if not self.authorized():
                return
            u = urlparse(self.path)
            path = u.path
            if path == "/put":   # a shipped file's bytes (tools/stage_ship.py)
                q = parse_qs(u.query)
                try:
                    r = store.put(q["path"][0], int(q["offset"][0]), int(q["total"][0]), self.rfile,
                                  int(self.headers.get("Content-Length", "0")), q.get("mode", [None])[0],
                                  q.get("fp", [None])[0])
                except (KeyError, ValueError, OSError) as e:
                    r = {"ok": False, "error": str(e)}
                self.reply(200 if r.get("ok") else 400, r)
                return
            if path == "/sink":   # a link test: read the body as fast as it comes
                n = int(self.headers.get("Content-Length", "0"))
                t0 = time.perf_counter()
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    left -= len(chunk)
                self.reply(200, {"ok": True, "bytes": n - left, "seconds": time.perf_counter() - t0})
                return
            try:
                req = self.body_json()
                if not isinstance(req, dict):
                    raise ValueError("a JSON object expected")
            except (ValueError, json.JSONDecodeError) as e:
                self.reply(400, {"ok": False, "error": f"bad JSON: {e}"})
                return
            if path == "/start":
                if "begin" not in req:
                    self.reply(400, {"ok": False, "error": "start needs begin"})
                    return
                self.reply(200, workers.start(req))
            elif path == "/stop":
                self.reply(200, {"ok": True, "stopped": workers.stop(req.get("port"))})
            elif path == "/send":
                self.reply(200, send_test(req["url"], int(req.get("mib", 64)), cfg["_token"]))
            else:
                self.reply(404, {"ok": False, "error": "unknown path"})

    return Handler


def send_test(url: str, mib: int, token: str) -> dict:
    """POST `mib` MiB to another node's /sink: the link's throughput from here (the wall time of the request)."""
    data = b"\0" * (mib << 20)
    req = urllib.request.Request(url.rstrip("/") + "/sink", data=data, method="POST",
                                 headers={"X-Stage-Token": token, "Content-Type": "application/octet-stream"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        got = json.load(r)
    s = time.perf_counter() - t0
    return {"ok": bool(got.get("ok")), "bytes": len(data), "seconds": s, "mb_s": len(data) / s / 1e6}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    workers = Workers(cfg)
    srv = ThreadingHTTPServer((cfg["bind"], int(cfg.get("agent_port", 7840))), make_handler(cfg, workers, Store(cfg)))
    sys.stderr.write(f"stage_node: listening on {cfg['bind']}:{cfg.get('agent_port', 7840)}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        workers.stop(None)


if __name__ == "__main__":
    main()
