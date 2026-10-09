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
      "token_file": "/srv/stage/stage-token"   or STRATA_STAGE_TOKEN in the environment
    }

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
"""
from __future__ import annotations

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
        port = int(req.get("port", 7841))
        self.stop(port)
        c = self.cfg
        args = [c["exe"], "--serve", "--stage-worker", str(port), "--stage-begin", str(int(req["begin"]))]
        if req.get("end") is not None:
            args += ["--stage-end", str(int(req["end"])), "--stage-next", str(req["next"])]
        args += ["--pack", c["pack"], "--native", c["native"], "--ple-gguf", c["ple_gguf"],
                 "--expert-profile", c["expert_profile"], "--expert-cache", str(req.get("cache", "auto")),
                 "--prefill", str(int(req.get("prefill", 2048))), "--stage-bind", c["bind"]]
        args += [str(a) for a in c.get("args", [])] + [str(a) for a in req.get("extra", [])]
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


def make_handler(cfg: dict, workers: Workers):
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
            if self.headers.get("X-Stage-Token", "") == cfg["_token"]:
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
            path = urlparse(self.path).path
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
    srv = ThreadingHTTPServer((cfg["bind"], int(cfg.get("agent_port", 7840))), make_handler(cfg, workers))
    sys.stderr.write(f"stage_node: listening on {cfg['bind']}:{cfg.get('agent_port', 7840)}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        workers.stop(None)


if __name__ == "__main__":
    main()
