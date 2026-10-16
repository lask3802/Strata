"""The Monitor tab's Workers card: a remote stage's workers (docs/REMOTE_STAGE.md), one entry per worker of the chain.

The main process's own log says what it knows of its link: the worker's address and layers once the hello is through
("strata serve: remote stage HOST:PORT: layers 26-47 there (...)"), the round trips with STRATA_REMOTE_TIMING=1, and
a lost link.  What only the worker PC knows - its GPU and VRAM, its engine's version, its expert cache, its own time
per window, a relay's next worker - comes from its node agent (tools/stage_node.py: GET /info and GET /log, with the
shared STRATA_STAGE_TOKEN in X-Stage-Token, which never leaves this server).  The agents are asked only when the run
config says so ("stage_agents": true, or a map of worker address to agent URL: those only), on a thread of their
own, at most every AGENT_EVERY_S and only while a Monitor tab asks for them (GET /metrics?workers=1), so neither a
slow agent nor the agent's nvidia-smi is ever on a request's path.

The lines read are the engine's (a reconnect within a run: "waiting for the worker", "connected again", with
INFO remote_stage_wait= on stdout while a request waits; the hand-off rows' format, ", rows bf16"), and the older
remote stage's without them.  The run config is read when the server starts.
"""
from __future__ import annotations

import http.client
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
SCAN_MAX = 256 << 20       # the lines before those: at most this much scanned per look (the rest at the next)
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
    is a reconnect.  The server's own "[strata] ... engine started:" line marks where a start begins.  When more than
    READ_MAX was written since the last read, the lines before the last READ_MAX are scanned, not read one by one:
    the starts, connects and relinks in them counted, and the last line of each kind that changes the state read, in
    the log's order (the state reading every line would leave), SCAN_MAX of them per look; "behind": what is left
    to scan (the counts are provisional until then).  The current run's first lines are read whole (its link)."""
    STARTED = " engine started: "
    CONNECT = re.compile(r"strata serve: remote stage (\S+): layers (\d+)-(\d+) there \(prompt chunk (\d+), "
                         r"context (\d+)(?:, rows (\w+))?\)")
    WAITING = re.compile(r"strata serve: remote stage: waiting for the worker at (\S+) \((.*?)\)(?:, up to |$)")
    RELINK = re.compile(r"strata serve: remote stage: the link to (\S+) was lost \((.*)\): connected again")
    WINDOWS = re.compile(r"strata remote: (\d+) windows: mean ([\d.]+) ms a round trip = worker ([\d.]+) \+ "
                         r"link ([\d.]+)")
    GAP = re.compile(r"strata remote: main process between windows ([\d.]+) ms each")
    # A failed link as the start prints it: "strata serve: " + the link's error.  Each of src/net/stage_link.cpp's
    # errors by its start, and the worker's own ("remote stage (worker): ..."); every other "strata serve: remote
    # stage: ..." line says how it goes (a checkpoint saved or restored, a session continued or dropped after
    # connecting again, a K/V check an earlier worker cannot take part in) and leaves the link as it is.
    ERROR = re.compile(r"strata serve: (remote stage \(worker\): .*|"
                       r"remote stage: (?:cannot connect to |cannot resolve |expected HOST:PORT|"
                       r"the sockets did not start|not connected|receive failed|send failed|the link failed earlier|"
                       r"the peer closed the connection|the two sides differ|"
                       r"the wait was cancelled|the worker refused|the worker was started without --vision|"
                       r"unexpected reply|bad message magic|a window's rows|an empty position table).*)")
    # a scanned part of the log: the lines counted, and the lines looked for (the last of each kind; an error: the
    # last "strata serve: remote stage" line that ERROR takes)
    COUNTED = {"runs": STARTED.encode(), "connects": b" there (prompt chunk ", "relinks": b"): connected again"}
    MARKS = {"start": STARTED.encode(), "connect": b" there (prompt chunk ", "relink": b"): connected again",
             "wait": b"strata serve: remote stage: waiting for the worker at ", "windows": b" windows: mean ",
             "gap": b"strata remote: main process between windows "}
    REMOTE = b"strata serve: remote stage"

    def __init__(self):
        self.key, self.pos = None, 0
        self.read_at = 0.0             # when the log was last read
        self.lock = threading.Lock()   # two pages (or a scraper) reading at once must not count a line twice
        # Each event carries where its line is in the log ("pos"), which orders it against the engine's stdout ERR
        # (the server notes the log's size when it reads one), and "at", when it was read (a lazy read sees it late:
        # shown, never compared).  "link": the current run's last connect or relink; "run_pos": where that run began;
        # "size": the log's size at this read; "rotated": the log was found emptied or cut short (a rotation) -
        # "by": the places read move that much before its start, "at": the read before (an ERR then is of the old log)
        self.state = {"connect": None, "connects": 0, "relinks": 0, "waiting": None, "error": None, "link": None,
                      "windows": None, "gap_ms": None, "runs": 0, "run_pos": None, "run_connected": False,
                      "behind": 0, "size": 0, "rotated": None}

    def poll(self, path, first, start) -> dict:
        """The state after the log's new lines.  `first`: where the engine's first run in this server began (read
        from there once), `start`: where its current run began."""
        if not path or start is None:
            return dict(self.state)
        with self.lock:
            st = self.state
            if self.key != path:
                self.key, self.pos = path, start if first is None else first
            try:
                size = os.path.getsize(path)
                if size < self.pos:     # emptied, replaced or cut short (a rotation): what was read keeps its order,
                    by = self.pos       # before the new log's first line
                    for k in ("link", "error", "waiting"):
                        if st[k]:
                            st[k] = {**st[k], "pos": st[k]["pos"] - by}
                    if st["run_pos"] is not None:
                        st["run_pos"] -= by
                    st["rotated"], self.pos = {"by": by, "at": self.read_at}, 0
                with open(path, "rb") as f:
                    budget = SCAN_MAX
                    while size - self.pos > READ_MAX and budget > 0:
                        if self.pos == start:           # the current run's first lines: read
                            f.seek(start)
                            head = f.read(min(HEAD_MAX, size - READ_MAX - start))
                            n = head.rfind(b"\n") + 1
                            if n:
                                self.feed(head[:n], start)
                                self.pos, budget = self.pos + n, budget - n
                                continue
                        stop = min(size - READ_MAX, self.pos + budget)
                        if self.pos < start < stop:
                            stop = start
                        n = self.scan(f, self.pos, stop)
                        if not n:                       # no whole line in it: the budget is spent, or a line
                            budget = 0 if stop < size - READ_MAX else budget   # runs into the end (read below)
                            break
                        self.pos, budget = self.pos + n, budget - n
                    behind = size - self.pos > READ_MAX and budget <= 0
                    st["behind"] = size - READ_MAX - self.pos if behind else 0
                    f.seek(self.pos)
                    data = b"" if behind else f.read(size - self.pos)
            except OSError:
                return dict(self.state)
            end = data.rfind(b"\n") + 1                 # whole lines only: the rest is read next time
            self.feed(data[:end], self.pos)
            self.pos += end
            self.read_at, st["size"] = time.time(), size
            return dict(st)

    @staticmethod
    def line_at(buf: bytes, i: int, base: int) -> tuple[int, str]:
        """The line of buf that holds byte i: where it begins in the log (buf begins at `base`), and its text."""
        a = buf.rfind(b"\n", 0, i) + 1
        b = buf.find(b"\n", i)
        return base + a, buf[a:b].decode("utf-8", "replace").rstrip("\r")

    def scan(self, f, a: int, b: int) -> int:
        """The whole lines from byte a to b, scanned (see the class): the bytes scanned, up to the last line end."""
        f.seek(a)
        pos, left, carry, last = a, b - a, b"", {}
        while left > 0:
            chunk = f.read(min(left, 8 << 20))
            if not chunk:
                break
            left -= len(chunk)
            buf = carry + chunk
            end = buf.rfind(b"\n") + 1
            carry = buf[end:]
            if not end:
                continue
            for k, pat in self.COUNTED.items():
                self.state[k] += buf.count(pat, 0, end)
            for kind, mark in self.MARKS.items():
                i = buf.rfind(mark, 0, end)
                if i >= 0:
                    last[kind] = self.line_at(buf, i, pos)
            hi = end
            while (i := buf.rfind(self.REMOTE, 0, hi)) >= 0:
                at, line = self.line_at(buf, i, pos)
                if self.ERROR.search(line):
                    last["error"] = (at, line)
                    break
                hi = at - pos
            pos += end
        now = time.time()
        for at, line in sorted(last.values()):         # in the log's order
            self.event(line, at, now, count=False)
        return pos - a

    def feed(self, data: bytes, base: int) -> None:
        """Whole lines that begin at byte `base` of the log."""
        now = time.time()
        pos = base
        for raw in data.split(b"\n")[:-1]:
            at, pos = pos, pos + len(raw) + 1
            self.event(raw.decode("utf-8", "replace").rstrip("\r"), at, now)

    def event(self, line: str, at: int, now: float, count: bool = True) -> None:
        """One line at byte `at` (`count`: a start, connect or relink is counted here, not by a scan)."""
        st = self.state
        if line.startswith("[strata]") and self.STARTED in line:
            st["runs"] += count
            st.update(run_pos=at, run_connected=False, waiting=None, link=None)
            st["windows"] = st["gap_ms"] = None             # the timing of this run's link only
            return
        if "remote" not in line:
            return
        m = self.CONNECT.search(line)
        if m:
            st["connects"] += count
            st["run_connected"], st["waiting"] = True, None
            st["link"] = {"pos": at, "at": now}
            st["connect"] = {"address": m[1], "layers": [int(m[2]), int(m[3])], "chunk": int(m[4]),
                             "context": int(m[5]), "rows": m[6], "at": now, "relinked": False}
            return
        m = self.RELINK.search(line)
        if m:
            st["relinks"] += count
            st["run_connected"], st["waiting"] = True, None
            st["link"] = {"pos": at, "at": now}
            st["error"] = {"text": m[2], "at": now, "pos": at}
            if st["connect"]:
                st["connect"] = {**st["connect"], "at": now, "relinked": True}
            return
        m = self.WAITING.search(line)
        if m:
            st["waiting"] = {"address": m[1], "why": m[2], "at": now, "pos": at}
            return
        m = self.WINDOWS.search(line)
        if m:
            st["windows"] = {"n": int(m[1]), "round_trip_ms": float(m[2]), "worker_ms": float(m[3]),
                             "link_ms": float(m[4]), "at": now}
            return
        m = self.GAP.search(line)
        if m:
            st["gap_ms"] = float(m[1])
            return
        m = self.ERROR.search(line)
        if m:
            st["error"] = {"text": m[1], "at": now, "pos": at}
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


def _opener():
    """Straight to the agent: no redirect, and no proxy (HTTP_PROXY, the system's) that would see the token."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


class Agent:
    """One node agent: its last /info and the facts of one worker's log, refreshed on a thread."""

    def __init__(self, url: str, port: int | None):
        self.url, self.port = url.rstrip("/"), port
        self.info = None          # the agent's /info, trimmed to what the card shows
        self.facts: dict = {}
        self.read_whole = False   # the worker's log was read since its start (whole, or its tail when that failed)
        self.last_lines: list[str] = []   # the end of the last read: the next tail is read on from them
        self.seen_running = False  # the agent ran this worker at some look (else its port's log may be an old one)
        self.ok = None            # None: not asked yet
        self.error = None
        self.at = None            # when it last answered (or failed)
        self.log_lines = 0

    def get(self, path: str, token: str):
        req = urllib.request.Request(self.url + path, headers={"X-Stage-Token": token})
        with _opener().open(req, timeout=AGENT_TIMEOUT_S) as r:
            out = json.loads(r.read().decode("utf-8", "replace"))
        if not isinstance(out, dict):
            raise AgentSaid("not an agent's answer")
        return out

    def log(self, token: str, whole: bool) -> list[str]:
        """The worker's log: its last LOG_TAIL lines, or all of them (once per worker start: its start-up lines)."""
        log = self.get(f"/log?port={self.port}&lines={10 ** 7 if whole else LOG_TAIL}", token)
        return [str(x) for x in log.get("log") or []]

    def whole_log(self, token: str, tail: list[str] | None) -> tuple[list[str], bool]:
        """All of the worker's log, and True; when that read fails (a log too long to send in time), its tail and
        False - read on from there next time, with the facts known so far, instead of failing at every look."""
        try:
            return self.log(token, whole=True), True
        except (AgentSaid, ValueError, OSError, http.client.HTTPException):
            return (tail if tail is not None else self.log(token, whole=False)), False

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
                # the agent starts each worker's log anew (tools/stage_node.py): a tail that holds the last read's
                # end goes on from it, any other is a new worker's log (or more than a tail was written since)
                lines = self.log(token, whole=False) if self.read_whole else None
                new = self.new_lines(lines) if lines is not None and len(lines) >= LOG_TAIL else None
                if new is not None:
                    self.facts = WorkerLog.parse(new, self.facts)
                elif lines is not None and len(lines) < LOG_TAIL:
                    self.facts = WorkerLog.parse(lines)             # a tail shorter than LOG_TAIL is all of it
                else:
                    lines, whole = self.whole_log(token, lines)
                    self.facts = WorkerLog.parse(lines, None if whole else self.facts)
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
        # {"HOST:PORT": "http://agent:7840", ...} asks the agents it names and no other
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
        # a map names every agent that is asked (the token goes to those only); true: each worker's own host,
        # which the stage token reaches through the link anyway
        url = cfg.get(addr) if isinstance(cfg, dict) else None
        if isinstance(cfg, dict) and not (isinstance(url, str) and url):
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
        # the round trips are in the log only with STRATA_REMOTE_TIMING=1 in the engine's environment (as it reads it)
        timing = str((env or os.environ).get("STRATA_REMOTE_TIMING", "")).startswith("1")
        return {"local_layers": [0, k - 1] if k else None, "rows": rows, "chunk": c["chunk"] if c else None,
                "context": c["context"] if c else None, "gap_ms": log["gap_ms"], "agents": on, "timing": timing,
                "provisional": log["behind"] > 0, "workers": workers}

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
        """The main process's link: (state, since, error).  Its events are ordered by where they are in the engine's
        log (an ERR on stdout by the log's size when the server read it), never by when the log was read."""
        if getattr(engine, "unloaded", False):
            return "unloaded", None, None
        alive = engine.alive() if hasattr(engine, "alive") else True
        proc = getattr(engine, "proc", None)
        starting = not alive and proc is not None and proc.poll() is None
        if not getattr(engine, "log_path", None):              # the run config has no "log": nothing says how it goes
            return ("nolog" if alive else "connecting" if starting else "down"), None, None
        # this run's errors only: the log's (a failed start), the ERR of a request (from this run's start on)
        run_pos, rot = log["run_pos"], log["rotated"]
        errs = [e for e in (log["error"],) if e and (run_pos is None or e["pos"] > run_pos)]
        se = getattr(engine, "stage_err", None)
        if se:
            # its place: unknown (the log's size could not be read) is before every line; one of the log before a
            # rotation (an ERR before the read that saw the old log, or past the new log's end) moves with it
            pos = se[2] if len(se) > 2 and se[2] is not None else float("-inf")
            if rot and (se[1] <= rot["at"] or pos > log["size"]):
                pos = min(pos - rot["by"], -1)
            if run_pos is None or pos > run_pos:      # (an ERR at run_pos came before the start)
                errs.append({"text": se[0], "at": se[1], "pos": pos})
        err = max(errs, key=lambda e: e["pos"]) if errs else None
        link, wait = log["link"], log["waiting"]
        closed = False         # the link was closed after it was made, the engine kept: the next request connects again
        if wait:
            # a request's wait for the worker (main-next) is over at its ERR, or when no heartbeat came for
            # WAIT_STALE_S (an ERR the server did not see).  A start's wait lasts while the process starts.
            beat = getattr(engine, "stage_wait_at", None) or 0
            stale = not starting and time.time() - max(wait["at"], beat) > WAIT_STALE_S
            if (err and err["pos"] > wait["pos"]) or stale:   # (an ERR at wait["pos"] came before it)
                wait, closed = None, True
        again = log["connects"] > 0 or log["relinks"] > 0
        if wait:                                                # why it waits: the last try's error
            return ("reconnecting" if again else "connecting"), wait["at"], {"text": wait["why"], "at": wait["at"]}
        if starting:
            return ("reconnecting" if again else "connecting"), None, err if again else None
        if err and link and err["pos"] <= link["pos"]:          # history: the link was made again after it
            err = None
        if not alive:
            return "down", err["at"] if err else None, err
        if link is None:                                        # up, but its log has not said so (yet)
            return "unknown", None, err
        # the worker's own failure of a step leaves the link as it is; any other error after the link closed it
        if closed or (err and not err["text"].startswith("remote stage (worker)")):
            return "idle", err["at"] if err else None, err
        c = log["connect"]
        since = link["at"] if c and c["relinked"] else getattr(engine, "ready_at", None) or link["at"]
        return "connected", since, err
