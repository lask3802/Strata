"""The Monitor tab's Workers card: a remote stage's workers (docs/REMOTE_STAGE.md), one entry per worker of the chain.

The main process's own log says what it knows of its link: the worker's address and layers once the hello is through
("strata serve: remote stage HOST:PORT: layers 26-47 there (...)"), the round trips with STRATA_REMOTE_TIMING=1, and
a lost link.  What only the worker PC knows - its GPU and VRAM, its engine's version, its expert cache, its own time
per window, a relay's next worker - comes from its node agent (tools/stage_node.py: GET /info and GET /log, with the
shared STRATA_STAGE_TOKEN in X-Stage-Token, which never leaves this server).  The agents are asked only when the run
config says so ("stage_agents": true, or a map of worker address to agent URL), on a thread of their own, at most
every AGENT_EVERY_S and only while a Monitor tab asks for them (GET /metrics?workers=1), so neither a slow agent nor
the agent's nvidia-smi is ever on a request's path.

The lines read are the engine's (a reconnect within a run: "waiting for the worker", "connected again", with
INFO remote_stage_wait= on stdout while a request waits; the hand-off rows' format, ", rows bf16"), and the older
remote stage's without them.  The run config is read when the server starts.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

AGENT_PORT = 7840          # tools/stage_node.py's default "agent_port"
AGENT_EVERY_S = 15.0       # an agent's /info runs nvidia-smi on the worker PC: not more often than this
AGENT_TIMEOUT_S = 4.0
LOG_TAIL = 300             # the worker log's lines read each time; the whole log once per worker start
LOG_OVERLAP = 5            # the last lines of a read that the next tail must hold to be read on from them
CHAIN_MAX = 8              # relays followed at most (a loop in a broken config ends here)
READ_MAX = 1 << 20         # at most the last MiB of the main log's new lines per read ...
HEAD_MAX = 256 << 10       # ... and the first 256 KiB of the engine's current run (its link's start-up lines)
WAIT_STALE_S = 25.0        # a request's wait for its worker says so every 10 s (INFO remote_stage_wait=)


def remote_args(args) -> tuple[str | None, int | None]:
    """The engine's --remote-stage HOST:PORT and its --layer-split K (the worker's first layer), or None."""
    args = list(args or [])
    addr = k = None
    for i, a in enumerate(args[:-1]):
        if a == "--remote-stage":
            addr = str(args[i + 1])
        elif a == "--layer-split":
            try:
                k = int(str(args[i + 1]).split(",")[0])
            except ValueError:
                pass
    return addr, k


def host_of(addr: str) -> str:
    """"10.0.0.2:7841" -> "10.0.0.2"; "[fe80::1]:7841" -> "fe80::1"."""
    h = addr.rsplit(":", 1)[0] if ":" in addr else addr
    return h[1:-1] if h.startswith("[") and h.endswith("]") else h


def port_of(addr: str) -> int | None:
    try:
        return int(addr.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


class MainLog:
    """The main process's link as its log tells it, read on from where it was last read (as ConvCacheLog does).
    Unlike the conversation cache, the link's history spans engine starts: each start that connects after the first
    is a reconnect.  The server's own "[strata] ... engine started:" line marks where a start begins."""
    STARTED = " engine started: "
    CONNECT = re.compile(r"strata serve: remote stage (\S+): layers (\d+)-(\d+) there \(prompt chunk (\d+), "
                         r"context (\d+)(?:, rows (\w+))?\)")
    WAITING = re.compile(r"strata serve: remote stage: waiting for the worker at (\S+) \((.*?)\)(?:, up to |$)")
    RELINK = re.compile(r"strata serve: remote stage: the link to (\S+) was lost \((.*)\): connected again")
    WINDOWS = re.compile(r"strata remote: (\d+) windows: mean ([\d.]+) ms a round trip = worker ([\d.]+) \+ "
                         r"link ([\d.]+)")
    GAP = re.compile(r"strata remote: main process between windows ([\d.]+) ms each")
    # a failed link as the start prints it ("strata serve: " + the link's error, src/net/stage_link.cpp): every
    # "strata serve: remote stage: ..." line but the ones that say how it goes
    ERROR = re.compile(r"strata serve: (remote stage(?: \(worker\))?: (?!checkpoint |continuing |waiting for the worker)"
                       r".*)")

    def __init__(self):
        self.key, self.pos = None, 0
        self.read_at = 0.0             # when the log was last read
        self.lock = threading.Lock()   # two pages (or a scraper) reading at once must not count a line twice
        self.state = {"connect": None, "connects": 0, "relinks": 0, "waiting": None, "error": None,
                      "windows": None, "gap_ms": None, "runs": 0, "run_connected": False}

    def poll(self, path, first, start) -> dict:
        """The state after the log's new lines.  `first`: where the engine's first run in this server began (read
        from there once), `start`: where its current run began."""
        if not path or start is None:
            return dict(self.state)
        with self.lock:
            if self.key != path:
                self.key, self.pos = path, start if first is None else first
            try:
                size = os.path.getsize(path)
                if size < self.pos:                     # the log was emptied or replaced
                    self.pos = 0
                with open(path, "rb") as f:
                    if size - self.pos > READ_MAX:      # too much to read: the current run's start, and the end
                        cut = size - READ_MAX
                        if self.pos <= start < cut:
                            f.seek(start)
                            head = f.read(min(HEAD_MAX, cut - start))
                            self.feed(head[:head.rfind(b"\n") + 1])
                        elif start >= cut:
                            cut = start
                        f.seek(cut)
                        data = f.read(size - cut)
                        if cut != start:                # it starts inside a line: from the next one
                            nl = data.find(b"\n") + 1
                            data, cut = data[nl:], cut + nl
                        self.pos = cut
                    else:
                        f.seek(self.pos)
                        data = f.read(size - self.pos)
            except OSError:
                return dict(self.state)
            end = data.rfind(b"\n") + 1                 # whole lines only: the rest is read next time
            self.pos += end
            self.feed(data[:end])
            self.read_at = time.time()
            return dict(self.state)

    def feed(self, data: bytes) -> None:
        now = time.time()
        st = self.state
        for line in data.decode("utf-8", "replace").splitlines():
            if line.startswith("[strata]") and self.STARTED in line:
                st["runs"] += 1
                st["run_connected"], st["waiting"] = False, None
                st["windows"] = st["gap_ms"] = None             # the timing of this run's link only
                continue
            if "remote" not in line:
                continue
            m = self.CONNECT.search(line)
            if m:
                st["connects"] += 1
                st["run_connected"], st["waiting"] = True, None
                st["connect"] = {"address": m[1], "layers": [int(m[2]), int(m[3])], "chunk": int(m[4]),
                                 "context": int(m[5]), "rows": m[6], "at": now, "relinked": False}
                continue
            m = self.RELINK.search(line)
            if m:
                st["relinks"] += 1
                st["run_connected"], st["waiting"] = True, None
                st["error"] = {"text": m[2], "at": now}
                if st["connect"]:
                    st["connect"] = {**st["connect"], "at": now, "relinked": True}
                continue
            m = self.WAITING.search(line)
            if m:
                # `after`: it was written after the last read (a lazy read sees it late)
                st["waiting"] = {"address": m[1], "why": m[2], "at": now, "after": self.read_at}
                continue
            m = self.WINDOWS.search(line)
            if m:
                st["windows"] = {"n": int(m[1]), "round_trip_ms": float(m[2]), "worker_ms": float(m[3]),
                                 "link_ms": float(m[4]), "at": now}
                continue
            m = self.GAP.search(line)
            if m:
                st["gap_ms"] = float(m[1])
                continue
            m = self.ERROR.search(line)
            if m:
                st["error"] = {"text": m[1], "at": now}
                st["run_connected"], st["waiting"] = False, None   # (a start that waited and gave up ends)


class WorkerLog:
    """A worker's facts from its log (the node agent's GET /log): its engine, expert cache, layers, the next worker
    of a relay, its own time per window, and the main process's (or the previous relay's) connection to it."""
    VERSION = re.compile(r"session is up \(engine ([^)\s]+)\)")
    CACHE = re.compile(r"strata generate: expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
    LAYERS = re.compile(r"strata serve: stage worker: layers (\d+)-(\d+), (?:relaying to ([^,\s]+)|no head)")
    LISTEN = re.compile(r"strata stage worker: listening on ")
    OWN = re.compile(r"strata stage worker: windows: own layers ([\d.]+) ms(?:, next worker ([\d.]+) ms)? each")
    MAIN = re.compile(r"strata stage worker: main process (\S+): layers")
    GONE = re.compile(r"strata stage worker: (\S+) disconnected(?:: (.*))?")
    RELAY = re.compile(r"strata serve: relay: layers (\d+)-(\d+) on (\S+)")
    RELAY_ERR = re.compile(r"strata serve: relay: (?!layers \d)(.*)")

    @classmethod
    def parse(cls, lines: list[str], facts: dict | None = None) -> dict:
        """`facts` updated by `lines` (a log's whole text, or its tail after it was read whole once)."""
        f = dict(facts or {})
        for line in lines:
            if cls.LISTEN.search(line):
                f["listening"] = True
            elif m := cls.VERSION.search(line):
                f["version"] = m[1]
            elif m := cls.CACHE.search(line):
                f["cache_slots"], f["cache_gib"] = int(m[1]), float(m[2])
            elif m := cls.LAYERS.search(line):
                f["layers"], f["next"] = [int(m[1]), int(m[2])], m[3]
            elif m := cls.OWN.search(line):
                f["own_ms"] = float(m[1])
                f["next_ms"] = float(m[2]) if m[2] else None
            elif m := cls.MAIN.search(line):
                f["peer"], f["peer_error"] = m[1], None
            elif m := cls.GONE.search(line):
                f["peer"], f["peer_error"] = None, m[2]
                # a relay whose next worker failed in a window: the error names it when it could not reach it
                if m[2] and f.get("next") and f["next"] in m[2]:
                    f["next_ok"], f["next_error"] = False, m[2]
            elif m := cls.RELAY.search(line):
                f["next_layers"], f["next_ok"], f["next_error"] = [int(m[1]), int(m[2])], True, None
            elif m := cls.RELAY_ERR.search(line):
                f["next_ok"], f["next_error"] = False, m[1]
        return f


class AgentSaid(Exception):
    """An agent's own error answer ({"ok": false, "error": ...})."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):   # the token goes to the agent named, never on to another address
        return None


_opener = urllib.request.build_opener(_NoRedirect)


class Agent:
    """One node agent: its last /info and the facts of one worker's log, refreshed on a thread."""

    def __init__(self, url: str, port: int | None):
        self.url, self.port = url.rstrip("/"), port
        self.info = None          # the agent's /info, trimmed to what the card shows
        self.facts: dict = {}
        self.read_whole = False   # the worker's log was read whole since its start (its start-up lines are in facts)
        self.last_lines: list[str] = []   # the end of the last read: the next tail is read on from them
        self.seen_running = False  # the agent ran this worker at some look (else its port's log may be an old one)
        self.ok = None            # None: not asked yet
        self.error = None
        self.at = None            # when it last answered (or failed)
        self.log_lines = 0

    def get(self, path: str, token: str):
        req = urllib.request.Request(self.url + path, headers={"X-Stage-Token": token})
        with _opener.open(req, timeout=AGENT_TIMEOUT_S) as r:
            out = json.loads(r.read().decode("utf-8", "replace"))
        if not isinstance(out, dict):
            raise AgentSaid("not an agent's answer")
        return out

    def log(self, token: str, whole: bool) -> list[str]:
        """The worker's log: its last LOG_TAIL lines, or all of them (once per worker start: its start-up lines)."""
        log = self.get(f"/log?port={self.port}&lines={10 ** 7 if whole else LOG_TAIL}", token)
        return [str(x) for x in log.get("log") or []]

    def new_lines(self, lines: list[str]) -> list[str] | None:
        """The lines of a tail after the last read's end, or None when it does not hold that end (a new worker's
        log, or more than a tail written since)."""
        k = len(self.last_lines)
        if not k:
            return None
        for i in range(len(lines) - k, -1, -1):
            if lines[i:i + k] == self.last_lines:
                return lines[i + k:]
        return None

    def scrub(self, text: str, token: str) -> str:
        """`text` without the token or the URL's user:password - an error goes to the browser."""
        user = urlsplit(self.url).netloc.rpartition("@")[0]
        for s in (token, user):
            if s:
                text = text.replace(s, "...")
        return text

    def refresh(self, token: str) -> None:
        try:
            info = self.get("/info", token)
            if not info.get("ok", True):
                raise AgentSaid(str(info.get("error") or "the agent said no"))
            gpus = info.get("gpus") if isinstance(info.get("gpus"), list) else []
            running = [int(p) for p in info.get("workers") or [] if str(p).lstrip("-").isdigit()]
            run = self.port in running if self.port is not None else None
            self.seen_running = self.seen_running or bool(run)
            self.info = {"host": info.get("host"), "gpus": [{k: g.get(k) for k in ("name", "vram_mib", "used_mib")}
                                                             for g in gpus if isinstance(g, dict)], "running": run}
            if self.port is not None:
                lines = self.log(token, whole=not self.read_whole)
                # the agent starts each worker's log anew (tools/stage_node.py): a tail that holds the last read's
                # end goes on from it, any other is a new worker's log (or more than a tail was written since)
                new = self.new_lines(lines) if self.read_whole and len(lines) >= LOG_TAIL else None
                if new is not None:
                    self.facts = WorkerLog.parse(new, self.facts)
                else:
                    # the log from its start: a tail shorter than LOG_TAIL is all of it; else read it whole again
                    if self.read_whole and len(lines) >= LOG_TAIL:
                        lines = self.log(token, whole=True)
                    self.facts = WorkerLog.parse(lines)
                    self.read_whole = bool(lines)
                self.last_lines = lines[-LOG_OVERLAP:]
                self.log_lines = len(lines)
            self.ok, self.error = True, None
        except urllib.error.HTTPError as e:
            self.ok = False
            self.error = "the agent refused the token (X-Stage-Token)" if e.code == 403 else f"HTTP {e.code}"
            e.close()
        except (AgentSaid, urllib.error.URLError, OSError) as e:   # the network's or the agent's own words
            self.ok = False
            reason = getattr(e, "reason", None) or e
            self.error = self.scrub(str(reason), token) or type(e).__name__
        except Exception as e:   # noqa: BLE001 - any failure is this agent's state, never the thread's end
            # its type only: a message may quote the request (http.client's invalid header value quotes the token)
            self.ok = False
            self.error = f"the agent's request failed ({type(e).__name__})"
        self.at = time.time()

    def view(self) -> dict:
        u = urlsplit(self.url)
        shown = u._replace(netloc=u.netloc.rsplit("@", 1)[-1]).geturl()   # no user:password of the URL
        return {"url": shown, "ok": self.ok, "error": self.error, "at": self.at}


class StageMonitor:
    """The Workers card's data (GET /metrics "stages"): None when the engine runs no remote stage."""

    def __init__(self):
        self.main_log = MainLog()
        # the run config's "stage_agents" (opt-in): true asks each worker's agent at HOST:7840, a map
        # {"HOST:PORT": "http://agent:7840", ...} names an agent of its own for some (false for one: not asked)
        self.agents_cfg = None
        self.agents: dict[str, Agent] = {}
        self.lock = threading.Lock()
        self.busy = False
        self.asked_at = 0.0

    def agents_on(self) -> bool:
        return self.agents_cfg is True or isinstance(self.agents_cfg, dict)

    def agent_for(self, addr: str) -> Agent | None:
        cfg = self.agents_cfg
        if not self.agents_on():
            return None
        url = cfg.get(addr) if isinstance(cfg, dict) else None
        if url is False or (url is not None and not isinstance(url, str)):
            return None
        if not url:
            h = host_of(addr)
            url = f"http://{'[' + h + ']' if ':' in h else h}:{AGENT_PORT}"
        if urlsplit(url).scheme not in ("http", "https"):
            return None
        with self.lock:
            a = self.agents.get(addr)
            if a is None or a.url != url.rstrip("/"):
                a = self.agents[addr] = Agent(url, port_of(addr))
            return a

    def ask_agents(self, chain: list[str], token: str) -> None:
        """Every agent of the chain, on a thread, at most every AGENT_EVERY_S; a call while one runs does nothing."""
        now = time.time()
        with self.lock:
            if self.busy or now - self.asked_at < AGENT_EVERY_S:
                return
            self.busy, self.asked_at = True, now

        def run():
            try:
                for a in [a for a in (self.agent_for(x) for x in chain) if a is not None]:
                    a.refresh(token)
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=run, daemon=True, name="stage-agents").start()

    def view(self, engine, ask: bool = False) -> dict | None:
        spawn = getattr(engine, "spawn", None)
        args = spawn[1] if isinstance(spawn, tuple) and len(spawn) > 1 else []
        first, k = remote_args(args)
        if not first:
            return None
        env = spawn[4] if len(spawn) > 4 and isinstance(spawn[4], dict) else None
        token = (env or os.environ).get("STRATA_STAGE_TOKEN", "")
        log = self.main_log.poll(getattr(engine, "log_path", None), getattr(engine, "first_log_start", None),
                                 getattr(engine, "log_start", None))
        # the chain: the main process's worker, then each relay's next worker as its log names it
        chain = [first]
        while len(chain) < CHAIN_MAX:
            a = self.agents.get(chain[-1])
            nxt = a.facts.get("next") if a is not None else None
            if not nxt or nxt in chain:
                break
            chain.append(nxt)
        on = self.agents_on()
        if ask and on:
            self.ask_agents(chain, token)
        workers = [self.worker_view(i, addr, chain, engine, log, on) for i, addr in enumerate(chain)]
        c = log["connect"]
        rows = c.get("rows") if c else None
        return {"local_layers": [0, k - 1] if k else None, "rows": rows, "chunk": c["chunk"] if c else None,
                "context": c["context"] if c else None, "gap_ms": log["gap_ms"], "agents": on, "workers": workers}

    def worker_view(self, i: int, addr: str, chain: list[str], engine, log: dict, on: bool) -> dict:
        a = self.agents.get(addr) if on else None
        f = a.facts if a is not None and a.ok else {}
        info = a.info if a is not None and a.ok else None
        gpu = (info.get("gpus") or [None])[0] if info else None
        w = {"address": addr, "host": info.get("host") if info else None, "agent": a.view() if a else None,
             "gpu": gpu, "version": f.get("version"), "cache_slots": f.get("cache_slots"),
             "cache_gib": f.get("cache_gib"), "own_ms": f.get("own_ms"), "relay_to": f.get("next"),
             "layers": f.get("layers"), "round_trip_ms": None, "worker_ms": None, "link_ms": None,
             "reconnects": None, "since": None, "error": None}
        if i == 0:
            state, since, err = self.main_state(engine, log)
            c = log["connect"]
            if c and not w["layers"]:
                w["layers"] = c["layers"]
            win = log["windows"]
            if win:
                w.update(round_trip_ms=win["round_trip_ms"], worker_ms=win["worker_ms"], link_ms=win["link_ms"])
            w["reconnects"] = max(0, log["connects"] - 1) + log["relinks"]
        else:
            prev = self.agents.get(chain[i - 1]) if on else None
            pf = prev.facts if prev is not None and prev.ok else {}
            since, err = None, None
            if not w["layers"] and pf.get("next_layers"):
                w["layers"] = pf["next_layers"]
            w["round_trip_ms"] = pf.get("next_ms")
            if pf.get("next_ok") is True:
                state = "connected"
            elif pf.get("next_ok") is False:
                state, err = "down", {"text": pf.get("next_error"), "at": prev.at}
            else:
                state = "unknown"
        # the worker's agent: a worker process it ran that ended is down, whatever the link last said
        if info is not None and info.get("running") is False and a.seen_running and a.log_lines:
            if state in ("connected", "unknown"):
                state = "down"
                why = f.get("peer_error")
                err = {"text": "the worker process ended" + (f" ({why})" if why else ""), "at": a.at}
        w.update(state=state, since=since, error=err)
        return w

    def main_state(self, engine, log: dict):
        """The main process's link: (state, since, error)."""
        connect = log["connect"]
        # errors from before the current link was made are history (an engine that later dies of something else
        # does not show an old link error as its cause)
        made = max(getattr(engine, "ready_at", None) or 0, connect["at"] if connect and connect["relinked"] else 0)
        errs = [e for e in (log["error"], getattr(engine, "stage_err", None) and
                            {"text": engine.stage_err[0], "at": engine.stage_err[1]}) if e]
        err = max(errs, key=lambda e: e["at"]) if errs else None
        if getattr(engine, "unloaded", False):
            return "unloaded", None, None
        alive = engine.alive() if hasattr(engine, "alive") else True
        proc = getattr(engine, "proc", None)
        starting = not alive and proc is not None and proc.poll() is None
        again = log["connects"] > 0 or log["relinks"] > 0
        wait, stage_err = log["waiting"], getattr(engine, "stage_err", None)
        if wait:
            # a request's wait for the worker (main-next) is over when its ERR came after the wait began and after
            # its last heartbeat, or when no heartbeat came for WAIT_STALE_S (the link stays closed; the next request
            # connects again).  A start's wait lasts while the process starts.
            beat = getattr(engine, "stage_wait_at", None) or 0
            ended = stage_err is not None and stage_err[1] >= (wait.get("after") or 0) and stage_err[1] >= beat
            if ended or (not starting and time.time() - max(wait["at"], beat) > WAIT_STALE_S):
                wait = None
        if alive and log["run_connected"] and not wait:
            since = connect["at"] if connect and connect["relinked"] else getattr(engine, "ready_at", None)
            if since is None and connect:
                since = connect["at"]
            # the engine said its link failed after it was made (and nothing made it again): the next request finds
            # it broken
            if stage_err and stage_err[1] > (since or 0):
                return "down", stage_err[1], {"text": stage_err[0], "at": stage_err[1]}
            return "connected", since, None
        if wait or starting:
            if wait:                                            # why it waits: the last try's error
                err = {"text": wait["why"], "at": wait["at"]}
            return ("reconnecting" if again else "connecting"), wait["at"] if wait else None, \
                err if again or wait else None
        if err and not starting and err["at"] <= made:
            err = None
        if alive:                                               # up, but its log has not said so (yet)
            return "unknown", None, err
        return "down", err["at"] if err else None, err
