"""The Monitor's Workers card (serve/stages.py): the main process's log, a worker's log, and the node agent's answers
- with tools/stage_node.py's own HTTP handler in this process (no GPU, no engine, no network beyond 127.0.0.1).

    python -m unittest serve.test_stages
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from serve import stages
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import stage_node as sn  # noqa: E402

TOKEN = "the-shared-secret"
STARTED = "[strata] 2026-10-16 09:00:00 engine started: strata --serve --layer-split 26 ...\n"
MAIN_RUN = ("strata generate: remote stage: this process holds layers 0-25\n"
            "strata serve: layer split: layers 0-25 (CUDA0), 26-47 (10.10.10.2:7841), the head (CUDA0), one hand-off "
            "per window\n"
            "strata serve: remote stage 10.10.10.2:7841: layers 26-47 there (prompt chunk 5888, context 131072, rows fp32)\n")
MAIN_TIMING = ("strata remote: 64 windows: mean 23.04 ms a round trip = worker 21.87 + link 1.17 (last: T=4 at 900, "
               "22.10 = 21.00 + 1.10)\n"
               "strata remote: main process between windows 25.32 ms each (64 windows: head, drafter, layers 0-K)\n")
WORKER_LOG = """strata generate: session is up (engine 0.1.44.1)
strata generate: expert cache 2345 slots, 9.14 GiB of VRAM; policy is
strata serve: stage worker: layers 26-47, no head, prompt chunk 5888, port 7841
strata stage worker: listening on 10.10.10.2:7841 (a token is required)
strata stage worker: connection from 192.168.1.140
strata stage worker: main process 192.168.1.140: layers 26.. here, context 131072, chunk 5888, rows fp32
strata stage worker: windows: own layers 11.08 ms each (64 windows; wait for the GPU 10.42, pool + plan 0.41, host staging 0.00)
"""
RELAY_LOG = """strata generate: session is up (engine 0.1.44.1)
strata generate: expert cache 1800 slots, 7.02 GiB of VRAM; policy is
strata serve: relay: layers 40-47 on 10.10.10.3:7841 (rows fp32)
strata serve: stage worker: layers 26-39, relaying to 10.10.10.3:7841, prompt chunk 5888, port 7841
strata stage worker: listening on 10.10.10.2:7841 (a token is required)
strata stage worker: main process 192.168.1.140: layers 26.. here, context 131072, chunk 5888, rows fp32
strata stage worker: windows: own layers 7.50 ms, next worker 9.25 ms each (64 windows; wait for the GPU 7.00, pool + plan 0.30, host staging 0.00)
"""
LAST_LOG = """strata generate: session is up (engine 0.1.44.1)
strata generate: expert cache 1500 slots, 5.80 GiB of VRAM; policy is
strata serve: stage worker: layers 40-47, no head, prompt chunk 5888, port 7841
strata stage worker: listening on 10.10.10.3:7841 (a token is required)
strata stage worker: main process 10.10.10.2: layers 40.. here, context 131072, chunk 5888, rows fp32
strata stage worker: windows: own layers 8.40 ms each (64 windows; wait for the GPU 8.00, pool + plan 0.30, host staging 0.00)
"""
GPU = [{"name": "NVIDIA GeForce RTX 2080 Ti", "vram_mib": 11264, "free_mib": 1100, "used_mib": 10164, "pcie_gen": "3",
        "pcie_width": "16", "pcie_width_max": "16"}]
_gpus = threading.local()


class FakeAgent:
    """tools/stage_node.py's handler on 127.0.0.1 with a worker log of our own (`log`, None: no worker started)."""

    def __init__(self, log: str | None, running: bool = True, gpus=None):
        self.dir = tempfile.TemporaryDirectory()
        cfg = {"_token": TOKEN, "bind": "127.0.0.1", "log_dir": self.dir.name}
        self.workers = sn.Workers(cfg)
        self.running = running
        self.workers.running = lambda: [7841] if self.running else []
        self.workers.reap = lambda: None
        self.set_log(log)
        self.gpus = GPU if gpus is None else gpus
        # each agent its own GPUs: the module's gpu_info answers what the request's own handler set (one thread each)
        self.patch = mock.patch.object(sn, "gpu_info", lambda: _gpus.value)
        self.patch.start()
        agent = self

        class Handler(sn.make_handler(cfg, self.workers, sn.Store(cfg))):
            def log_message(self, *a):                              # no request lines on stderr
                pass

            def do_GET(self):
                _gpus.value = agent.gpus
                super().do_GET()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def set_log(self, text: str | None) -> None:
        p = self.workers.log_path(7841)
        if text is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(text, encoding="utf-8")

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.patch.stop()
        self.dir.cleanup()


class FakeProc:
    def __init__(self, rc=None):
        self.rc = rc

    def poll(self):
        return self.rc


class RemoteEngine(MockEngine):
    """A StrataEngine's face as serve/stages.py reads it: its spawn (args, env), its log, alive or not."""

    def __init__(self, tok, log_path: str, args=None, env=None):
        super().__init__(tok, "Hello.")
        args = args or ["--layer-split", "26", "--split-device", "0", "--remote-stage", "10.10.10.2:7841"]
        self.spawn = ("strata", args, None, log_path, {"STRATA_STAGE_TOKEN": TOKEN} if env is None else env, False,
                      None)
        self.log_path, self.log_start = log_path, 0
        self.running, self.unloaded, self.proc = True, False, FakeProc()
        self.ready_at = 1_000_000.0

    def alive(self):
        return self.running


def wait_agents(mon: stages.StageMonitor, timeout=10.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with mon.lock:
            if not mon.busy:
                return
        time.sleep(0.02)
    raise AssertionError("the agents' thread did not finish")


def view_now(mon: stages.StageMonitor, engine) -> dict:
    """A view after the agents were asked once (the first call asks, on a thread)."""
    mon.asked_at = 0.0
    mon.view(engine, ask=True)
    wait_agents(mon)
    return mon.view(engine)


class LogCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.dir.name, "strata.log")
        Path(self.log).write_text("", encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def append(self, text: str) -> None:
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(text)


class MainLogTest(LogCase):
    def test_connect_timing_and_restarts(self):
        ml = stages.MainLog()
        self.append("older lines of an earlier server: strata serve: remote stage 1.2.3.4:1: layers 9-9 there "
                    "(prompt chunk 1, context 1)\n")
        start = os.path.getsize(self.log)
        self.append(STARTED + MAIN_RUN)
        s = ml.poll(self.log, None, start)
        self.assertEqual(s["connects"], 1)                          # the line before this run's start is not read
        self.assertTrue(s["run_connected"])
        self.assertEqual(s["connect"]["address"], "10.10.10.2:7841")
        self.assertEqual(s["connect"]["layers"], [26, 47])
        self.assertEqual((s["connect"]["chunk"], s["connect"]["context"], s["connect"]["rows"]), (5888, 131072, "fp32"))
        self.assertIsNone(s["windows"])
        self.append(MAIN_TIMING + "strata serve: rem")               # half a line: read next time
        s = ml.poll(self.log, None, start)
        self.assertEqual((s["windows"]["round_trip_ms"], s["windows"]["worker_ms"], s["windows"]["link_ms"]),
                         (23.04, 21.87, 1.17))
        self.assertEqual(s["gap_ms"], 25.32)
        # the engine ended and was started again: a new run that connects is a reconnect
        self.append("ote stage: checkpoint 3 at 900 tokens saved on the worker\n")
        start2 = os.path.getsize(self.log)
        self.append(STARTED)
        s = ml.poll(self.log, None, start2)
        self.assertFalse(s["run_connected"])
        self.assertEqual(s["runs"], 2)
        self.assertIsNone(s["windows"])                             # the earlier link's round trips are gone
        self.append(MAIN_RUN.replace("rows fp32", "rows bf16"))
        s = ml.poll(self.log, None, start2)
        self.assertEqual(s["connects"], 2)
        self.assertEqual(s["connect"]["rows"], "bf16")
        self.assertTrue(s["run_connected"])
        # the remote stage before main-next: no rows in the line
        self.append(STARTED + MAIN_RUN.replace(", rows fp32", ""))
        s = ml.poll(self.log, None, start2)
        self.assertEqual((s["connects"], s["connect"]["rows"], s["connect"]["layers"]), (3, None, [26, 47]))

    def test_waiting_relink_and_errors(self):
        ml = stages.MainLog()
        self.append(STARTED + "strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (remote stage: "
                    "cannot connect to 10.10.10.2:7841: connection refused), up to 120 s (STRATA_REMOTE_CONNECT_S)\n")
        s = ml.poll(self.log, None, 0)
        self.assertEqual(s["waiting"]["address"], "10.10.10.2:7841")
        self.assertFalse(s["run_connected"])
        self.append(MAIN_RUN)
        s = ml.poll(self.log, None, 0)
        self.assertIsNone(s["waiting"])
        self.append("strata serve: remote stage: the link to 10.10.10.2:7841 was lost (remote stage: the peer closed "
                    "the connection): connected again\n")
        s = ml.poll(self.log, None, 0)
        self.assertEqual((s["connects"], s["relinks"]), (1, 1))
        self.assertTrue(s["connect"]["relinked"])
        self.assertEqual(s["error"]["text"], "remote stage: the peer closed the connection")
        self.append(STARTED + "strata serve: remote stage: cannot connect to 10.10.10.2:7841: Connection refused\n")
        s = ml.poll(self.log, None, 0)
        self.assertFalse(s["run_connected"])
        self.assertEqual(s["error"]["text"], "remote stage: cannot connect to 10.10.10.2:7841: Connection refused")
        # a start that waited and gave up: not waiting any more
        self.append(STARTED + "strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (x), up to 120 s "
                    "(STRATA_REMOTE_CONNECT_S)\nstrata serve: remote stage: cannot connect to 10.10.10.2:7841: timed "
                    "out\n")
        self.assertIsNone(ml.poll(self.log, None, 0)["waiting"])
        # the hello's check of the two sides, as the start prints it
        self.append("strata serve: remote stage: the two sides differ: kv type 1 here, 2 there\n")
        self.assertEqual(ml.poll(self.log, None, 0)["error"]["text"],
                         "remote stage: the two sides differ: kv type 1 here, 2 there")
        self.append("strata serve: remote stage: cannot connect to 10.10.10.2:7841: Connection refused\n")
        # informational remote-stage lines are not errors
        self.append("strata serve: remote stage: continuing the live session at 900 tokens\n"
                    "strata serve: remote stage: checkpoint 3 at 900 tokens saved on the worker\n")
        self.assertEqual(ml.poll(self.log, None, 0)["error"]["text"],
                         "remote stage: cannot connect to 10.10.10.2:7841: Connection refused")


    def test_which_lines_are_errors(self):
        """Each remote-stage line of the engine (src/program/generate.cpp, main-next 3db1c468) as it prints it: the
        ones that say how it goes leave a made link as it is; the link's errors (src/net/stage_link.cpp) end it."""
        info = ["strata serve: remote stage: checkpoint 3 at 900 tokens saved on the worker",
                "strata serve: remote stage: checkpoint 3 at 900 tokens restored on both sides",
                "strata serve: remote stage: continuing the live session at 900 tokens",
                "strata serve: remote stage: the worker's hello predates the K/V element check: --kv must match by "
                "hand",
                # an earlier build of the K/V check (f6426fe1)
                "strata serve: remote stage: the K/V type is not checked (the worker's hello does not carry it: an "
                "earlier build); --kv must match by hand",
                "strata serve: remote stage: the worker holds no session or checkpoint after connecting again: dropped "
                "the live session and 2 checkpoints, reading the prompt from token 0",
                "strata serve: remote stage: the worker holds no checkpoint after connecting again: reading the prompt "
                "from token 0"]
        for line in info:
            ml = stages.MainLog()
            Path(self.log).write_text(STARTED + MAIN_RUN + line + "\n", encoding="utf-8")
            s = ml.poll(self.log, None, 0)
            self.assertEqual((s["error"], s["run_connected"]), (None, True), line)
        errors = ["remote stage: cannot connect to 10.10.10.2:7841: Connection refused",
                  "remote stage: cannot resolve pc-b:7841: Name or service not known",
                  "remote stage: expected HOST:PORT, got \"pc-b\"",
                  "remote stage: the sockets did not start (WSAStartup)",
                  "remote stage: not connected",
                  "remote stage: receive failed: timed out (no reply to the hello)",
                  "remote stage: send failed: Broken pipe",
                  "remote stage: the link failed earlier",
                  "remote stage: the peer closed the connection",
                  "remote stage: the two sides differ (K/V type --kv int8/f16)",
                  "remote stage: the wait was cancelled",
                  "remote stage: the worker refused: a wrong token",
                  "remote stage: the worker was started without --vision (its layers need the image positions too)",
                  "remote stage: unexpected reply to hello (type 3, 12 bytes)",
                  "remote stage: bad message magic (not a Strata stage peer?)",
                  "remote stage: a window's rows are not T whole rows",
                  "remote stage: an empty position table",
                  "remote stage (worker): CUDA error: out of memory"]
        for text in errors:
            ml = stages.MainLog()
            Path(self.log).write_text(STARTED + "strata serve: " + text + "\n", encoding="utf-8")
            s = ml.poll(self.log, None, 0)
            self.assertEqual(s["error"] and s["error"]["text"], text)
        # every error text src/net/stage_link.cpp has, as its literal starts
        src = Path(__file__).resolve().parents[1] / "src" / "net" / "stage_link.cpp"
        if not src.exists():
            self.skipTest("no engine source here")
        literals = set(re.findall(r'"(remote stage(?: \(worker\))?: [^"\\]*)', src.read_text(encoding="utf-8")))
        self.assertGreater(len(literals), 15)
        for lit in literals:
            self.assertTrue(stages.MainLog.ERROR.search("strata serve: " + lit + "x"), lit)

    def test_from_the_first_start_and_a_long_log(self):
        """Restarts before the first look count (read from this server's first start), and a log too long to read
        on still gives the current run's link (its start's lines, then the end)."""
        ml = stages.MainLog()
        self.append(STARTED + MAIN_RUN)
        start2 = os.path.getsize(self.log)
        self.append(STARTED + MAIN_RUN)
        s = ml.poll(self.log, 0, start2)
        self.assertEqual((s["connects"], s["runs"]), (2, 2))
        filler = "strata serve: request done: some line of the engine's own\n" * 200
        start3 = os.path.getsize(self.log)
        self.append(filler + STARTED + MAIN_RUN.replace("26-47 there", "26-40 there") + filler * 5 + MAIN_TIMING)
        start3 += len(filler.encode()) + filler.count("\n") * (os.linesep == "\r\n")   # where the run's line begins
        with mock.patch.object(stages, "READ_MAX", 4096), mock.patch.object(stages, "HEAD_MAX", 2048):
            s = ml.poll(self.log, 0, start3)
        self.assertEqual((s["connects"], s["runs"], s["connect"]["layers"]), (3, 3, [26, 40]))
        self.assertTrue(s["run_connected"])
        self.assertEqual(s["windows"]["round_trip_ms"], 23.04)
        self.append("strata remote: main process between windows 30.00 ms each (64 windows: ...)\n")
        self.assertEqual(ml.poll(self.log, 0, start3)["gap_ms"], 30.0)   # read on as usual afterwards

    def test_a_long_log_counts_what_it_skips(self):
        """The parts of a log too long to read still count their starts, connects and relinks (Reconnects)."""
        ml = stages.MainLog()
        filler = "strata serve: request done: some line of the engine's own\n" * 200
        relink = ("strata serve: remote stage: the link to 10.10.10.2:7841 was lost (remote stage: the peer closed the "
                  "connection): connected again\n")
        self.append(STARTED + MAIN_RUN + filler + relink + filler)          # an earlier run, not read
        start = os.path.getsize(self.log)
        self.append(STARTED + MAIN_RUN + filler + relink + filler * 5 + MAIN_TIMING)   # its relink: after the head
        with mock.patch.object(stages, "READ_MAX", 4096), mock.patch.object(stages, "HEAD_MAX", 2048):
            s = ml.poll(self.log, 0, start)
        self.assertEqual((s["runs"], s["connects"], s["relinks"]), (2, 2, 2))
        self.assertEqual((s["run_pos"], s["link"]["pos"] > start, s["windows"]["n"]), (start, True, 64))

    def test_a_scan_in_parts(self):
        """At most SCAN_MAX scanned per look: the counts are provisional ("behind") until the scan is through."""
        ml = stages.MainLog()
        relink = ("strata serve: remote stage: the link to 10.10.10.2:7841 was lost (remote stage: the peer closed the "
                  "connection): connected again\n")
        filler = "strata serve: request done: some line of the engine's own\n" * 100
        self.append(STARTED + MAIN_RUN + (filler + relink) * 20 + filler * 3)
        looks = []
        small = {"READ_MAX": 4096, "SCAN_MAX": 32768, "HEAD_MAX": 2048}
        with mock.patch.multiple(stages, **small):
            for _ in range(20):
                s = ml.poll(self.log, None, 0)
                looks.append(s["behind"])
                if not s["behind"]:
                    break
        self.assertGreater(len(looks), 3)
        self.assertEqual(looks, sorted(looks, reverse=True))        # less behind at each look
        self.assertEqual((s["connects"], s["relinks"], s["link"]["pos"] > 0), (1, 20, True))
        self.assertEqual(s["error"]["text"], "remote stage: the peer closed the connection")
        mon = stages.StageMonitor()
        with mock.patch.multiple(stages, **small):
            v = mon.view(RemoteEngine(ByteTokenizer(), self.log))
            self.assertTrue(v["provisional"])
            while v["provisional"]:
                v = mon.view(RemoteEngine(ByteTokenizer(), self.log))
        self.assertEqual(v["workers"][0]["reconnects"], 20)

    def test_a_start_after_a_line_that_never_ended(self):
        """The engine before died mid-line: the next start's line begins there, after no line end.  Read from there,
        scanned up to it (no stop short of it at every look) or read line by line."""
        filler = b"strata serve: request done: some line of the engine's own\n" * 100
        with open(self.log, "ab") as f:
            f.write((STARTED + MAIN_RUN).encode() + filler + b"strata generate: loading 45%")
        start = os.path.getsize(self.log)
        with open(self.log, "ab") as f:
            f.write((STARTED + MAIN_RUN.replace("26-47 there", "26-40 there")).encode() + filler * 3)
        for small in ({"READ_MAX": 4096, "SCAN_MAX": 32768, "HEAD_MAX": 2048}, {"READ_MAX": stages.READ_MAX}):
            ml = stages.MainLog()
            with mock.patch.multiple(stages, **small):
                for _ in range(5):
                    s = ml.poll(self.log, 0, start)
            self.assertEqual((s["behind"], s["runs"], s["connects"], s["run_pos"], s["connect"]["layers"]),
                             (0, 2, 2, start, [26, 40]), small)
            self.assertEqual(ml.pos, os.path.getsize(self.log))

    def test_a_scanned_start_begins_its_line(self):
        """A scanned part takes a start where a line is read as one: a line that begins "[strata]"."""
        filler = "strata serve: request done: some line of the engine's own\n" * 100
        self.append(filler * 2 + STARTED + MAIN_RUN + filler + "strata serve: request done: a prompt that says "
                    "\"[strata] x engine started: y\"\n" + filler * 5)
        ml = stages.MainLog()
        with mock.patch.object(stages, "READ_MAX", 4096), mock.patch.object(stages, "HEAD_MAX", 2048):
            s = ml.poll(self.log, None, 0)
        self.assertEqual((s["runs"], s["run_pos"], s["run_connected"]),
                         (1, Path(self.log).read_bytes().index(b"[strata]"), True))

    def test_a_scanned_part_s_last_error(self):
        """A scanned part's error is its last line ERROR takes (not a later remote stage line that is not one)."""
        filler = "strata serve: request done: some line of the engine's own\n" * 100
        refused = "strata serve: remote stage: cannot connect to 10.10.10.2:7841: refused ({})\n"
        self.append(STARTED + filler * 2 + refused.format(1) + filler + refused.format(2) +
                    "strata serve: remote stage: checkpoint 3 at 900 tokens saved on the worker\n" + filler * 5)
        ml = stages.MainLog()
        with mock.patch.object(stages, "READ_MAX", 4096), mock.patch.object(stages, "HEAD_MAX", 2048):
            s = ml.poll(self.log, None, 0)
        self.assertEqual((s["error"]["text"], s["error"]["pos"]),
                         ("remote stage: cannot connect to 10.10.10.2:7841: refused (2)",
                          Path(self.log).read_bytes().index(b"strata serve: remote stage: cannot connect to "
                                                            b"10.10.10.2:7841: refused (2)")))


class WorkerLogTest(unittest.TestCase):
    def test_worker_and_relay(self):
        f = stages.WorkerLog.parse(WORKER_LOG.splitlines())
        self.assertEqual(f["version"], "0.1.44.1")
        self.assertEqual((f["cache_slots"], f["cache_gib"]), (2345, 9.14))
        self.assertEqual((f["layers"], f["next"]), ([26, 47], None))
        self.assertEqual((f["own_ms"], f["next_ms"]), (11.08, None))
        self.assertEqual(f["peer"], "192.168.1.140")
        f = stages.WorkerLog.parse(["strata stage worker: 192.168.1.140 disconnected: remote stage: the peer closed "
                                    "the connection"], f)
        self.assertIsNone(f["peer"])
        self.assertEqual(f["version"], "0.1.44.1")                   # kept from the earlier lines
        r = stages.WorkerLog.parse(RELAY_LOG.splitlines())
        self.assertEqual((r["layers"], r["next"], r["next_layers"], r["next_ok"]),
                         ([26, 39], "10.10.10.3:7841", [40, 47], True))
        self.assertEqual((r["own_ms"], r["next_ms"]), (7.5, 9.25))
        r = stages.WorkerLog.parse(["strata serve: relay: remote stage: cannot connect to 10.10.10.3:7841: refused"], r)
        self.assertEqual((r["next_ok"], r["next_error"]), (False, "remote stage: cannot connect to 10.10.10.3:7841: refused"))
        r = stages.WorkerLog.parse(["strata serve: relay: layers 40-47 on 10.10.10.3:7841"], r)
        self.assertTrue(r["next_ok"])                               # linked again at the next reset
        # the next worker failed in a window: the relay's main connection ends with the error naming it
        r = stages.WorkerLog.parse(["strata stage worker: 192.168.1.140 disconnected: remote stage: cannot connect to "
                                    "10.10.10.3:7841: timed out"], r)
        self.assertFalse(r["next_ok"])
        r = stages.WorkerLog.parse(["strata stage worker: 192.168.1.140 disconnected: remote stage: the peer closed "
                                    "the connection"], {**r, "next_ok": True})
        self.assertTrue(r["next_ok"])                               # the main process went: not the next worker

    def test_args(self):
        self.assertEqual(stages.remote_args(["--layer-split", "26", "--remote-stage", "h:1"]), ("h:1", 26))
        self.assertEqual(stages.remote_args(["--layer-split", "26"]), (None, 26))
        self.assertEqual(stages.host_of("[fe80::1]:7841"), "fe80::1")
        self.assertEqual(stages.port_of("10.0.0.2:7841"), 7841)


class MonitorTest(LogCase):
    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()
        self.agents = []

    def tearDown(self):
        for a in self.agents:
            a.close()
        super().tearDown()

    def agent(self, *a, **kw) -> FakeAgent:
        x = FakeAgent(*a, **kw)
        self.agents.append(x)
        return x

    def stage_err(self, e, text: str) -> None:
        """The engine's `ERR remote stage...` as the server's pump takes it: after the log's lines so far."""
        e.stage_err = (text, time.time(), os.path.getsize(self.log))

    def test_no_remote_stage(self):
        mon = stages.StageMonitor()
        self.assertIsNone(mon.view(MockEngine(self.tok, "x")))
        e = RemoteEngine(self.tok, self.log, args=["--layer-split", "26", "--split-device", "0"])
        self.assertIsNone(mon.view(e, ask=True))
        self.assertEqual(mon.agents, {})                            # no agent is asked either

    def test_one_worker(self):
        ag = self.agent(WORKER_LOG)
        self.append(STARTED + MAIN_RUN + MAIN_TIMING)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        e = RemoteEngine(self.tok, self.log)
        v = view_now(mon, e)
        self.assertEqual(v["local_layers"], [0, 25])
        self.assertEqual(v["gap_ms"], 25.32)
        (w,) = v["workers"]
        self.assertEqual((w["state"], w["since"], w["reconnects"], w["error"]), ("connected", 1_000_000.0, 0, None))
        self.assertEqual((w["address"], w["layers"], w["relay_to"]), ("10.10.10.2:7841", [26, 47], None))
        self.assertEqual((w["round_trip_ms"], w["worker_ms"], w["link_ms"], w["own_ms"]), (23.04, 21.87, 1.17, 11.08))
        self.assertEqual(w["gpu"], {"name": "NVIDIA GeForce RTX 2080 Ti", "vram_mib": 11264, "used_mib": 10164})
        self.assertEqual((w["version"], w["cache_slots"], w["cache_gib"]), ("0.1.44.1", 2345, 9.14))
        self.assertTrue(w["agent"]["ok"])
        self.assertNotIn(TOKEN, json.dumps(v))                      # the token never leaves the server
        # the worker process ended (the agent no longer runs it): down, though the main process has not noticed
        ag.running = False
        ag.set_log(WORKER_LOG + "strata stage worker: 192.168.1.140 disconnected: CUDA error: an illegal memory "
                                "access\n")
        w = view_now(mon, e)["workers"][0]
        self.assertEqual(w["state"], "down")
        self.assertIn("the worker process ended", w["error"]["text"])
        self.assertIn("illegal memory access", w["error"]["text"])

    def test_agent_tail_and_restart(self):
        ag = self.agent(WORKER_LOG)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        e = RemoteEngine(self.tok, self.log)
        view_now(mon, e)
        # many lines later the start-up lines are out of the tail: the facts read from them stay
        ag.set_log(WORKER_LOG + "strata stage worker: chunk T=5888 at 0: read in 2704 ms\n" * (stages.LOG_TAIL + 50) +
                   "strata stage worker: windows: own layers 12.50 ms each (64 windows; ...)\n")
        w = view_now(mon, e)["workers"][0]
        self.assertEqual((w["version"], w["cache_slots"], w["own_ms"]), ("0.1.44.1", 2345, 12.5))
        # the agent started a new worker (another engine, another cache): read whole again
        ag.set_log(WORKER_LOG.replace("0.1.44.1", "0.1.45").replace("2345 slots", "2000 slots"))
        w = view_now(mon, e)["workers"][0]
        self.assertEqual((w["version"], w["cache_slots"]), ("0.1.45", 2000))
        # ... also when it wrote more than a tail before the next look (its start-up lines are not in the tail)
        ag.set_log(WORKER_LOG.replace("0.1.44.1", "0.1.46").replace("2345 slots", "1900 slots").replace(
            "no head", "relaying to 10.10.10.3:7841") + "strata stage worker: chunk T=5888 at 0: read in 2704 ms\n" * 400)
        w = view_now(mon, e)["workers"][0]
        self.assertEqual((w["version"], w["cache_slots"], w["relay_to"]), ("0.1.46", 1900, "10.10.10.3:7841"))

    def test_relay_chain(self):
        a1, a2 = self.agent(RELAY_LOG), self.agent(LAST_LOG, gpus=[{**GPU[0], "name": "NVIDIA GeForce RTX 3070"}])
        self.append(STARTED + MAIN_RUN.replace("26-47 there", "26-39 there"))
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": a1.url, "10.10.10.3:7841": a2.url}
        e = RemoteEngine(self.tok, self.log)
        view_now(mon, e)                                            # the relay's log names its next worker ...
        v = view_now(mon, e)                                        # ... whose agent is asked from then on
        w1, w2 = v["workers"]
        self.assertEqual((w1["layers"], w1["relay_to"], w1["state"]), ([26, 39], "10.10.10.3:7841", "connected"))
        self.assertEqual((w2["address"], w2["layers"], w2["state"], w2["round_trip_ms"], w2["own_ms"]),
                         ("10.10.10.3:7841", [40, 47], "connected", 9.25, 8.4))
        self.assertEqual((w1["gpu"]["name"], w2["gpu"]["name"]), ("NVIDIA GeForce RTX 2080 Ti", "NVIDIA GeForce RTX 3070"))
        self.assertIsNone(w2["reconnects"])                         # only the main process's link is counted
        a1.set_log(RELAY_LOG + "strata serve: relay: remote stage: cannot connect to 10.10.10.3:7841: refused\n")
        w2 = view_now(mon, e)["workers"][1]
        self.assertEqual(w2["state"], "down")
        self.assertIn("cannot connect", w2["error"]["text"])

    def test_states(self):
        self.append(STARTED)
        mon = stages.StageMonitor()
        mon.agents_cfg = False                                      # no agent is asked
        e = RemoteEngine(self.tok, self.log)
        e.running = False                                           # starting: the process runs, no READY yet
        w = mon.view(e, ask=True)["workers"][0]
        self.assertEqual((w["state"], w["agent"], w["gpu"]), ("connecting", None, None))
        self.append(MAIN_RUN)
        e.running = True
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        # the link failed in a request: the engine's ERR ends it
        self.stage_err(e, "remote stage: the peer closed the connection")
        e.running, e.proc = False, FakeProc(rc=1)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]), ("down", "remote stage: the peer closed the connection"))
        # the next request starts it again: reconnecting, then connected with one reconnect
        e.proc = FakeProc()
        self.append(STARTED)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "reconnecting")
        self.append(MAIN_RUN)
        e.running, e.ready_at = True, time.time()
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["reconnects"], w["error"]), ("connected", 1, None))
        # it ends later of something else: the earlier link's error is not its cause
        e.running, e.proc = False, FakeProc(rc=-9)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]), ("down", None))
        e.running, e.proc = True, FakeProc()
        e.unloaded = True
        self.assertEqual(mon.view(e)["workers"][0]["state"], "unloaded")

    def test_waiting_for_the_worker(self):
        """The recovery branch's start waits for its worker and says why: connecting, the last try's error."""
        self.append(STARTED + "strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (remote stage: "
                    "cannot connect to 10.10.10.2:7841: timed out), up to 120 s (STRATA_REMOTE_CONNECT_S)\n")
        mon = stages.StageMonitor()
        mon.agents_cfg = False
        e = RemoteEngine(self.tok, self.log)
        e.running = False
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]),
                         ("connecting", "remote stage: cannot connect to 10.10.10.2:7841: timed out"))
        self.assertIsNotNone(w["since"])

    def test_wrong_token_and_no_agent(self):
        ag = self.agent(WORKER_LOG)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        e = RemoteEngine(self.tok, self.log, env={"STRATA_STAGE_TOKEN": "not it"})
        w = view_now(mon, e)["workers"][0]
        self.assertEqual((w["state"], w["agent"]["ok"], w["agent"]["error"]),
                         ("connected", False, "the agent refused the token (X-Stage-Token)"))
        self.assertIsNone(w["gpu"])
        mon.agents_cfg = {"10.10.10.2:7841": "http://127.0.0.1:9"}  # nothing listens there
        w = view_now(mon, RemoteEngine(self.tok, self.log))["workers"][0]
        self.assertFalse(w["agent"]["ok"])
        self.assertTrue(w["agent"]["error"])
        mon.agents_cfg = None                                       # the default: no agent is asked (opt-in)
        self.assertIsNone(mon.agent_for("10.10.10.2:7841"))
        mon.agents_cfg = True                                       # each worker's host, port 7840
        self.assertEqual(mon.agent_for("10.10.10.2:7841").url, "http://10.10.10.2:7840")
        self.assertEqual(mon.agent_for("[fe80::1]:7841").url, "http://[fe80::1]:7840")
        mon.agents_cfg = {"10.10.10.2:7841": 7840, "10.10.10.3:7841": False, "10.10.10.4:7841": "ftp://x"}
        self.assertEqual([mon.agent_for(f"10.10.10.{i}:7841") for i in (2, 3, 4)], [None, None, None])
        self.assertIsNone(mon.agent_for("10.10.10.5:7841"))         # a map names every agent asked: not this one
        self.assertEqual(stages.Agent("http://u:secret@h:7840", 7841).view()["url"], "http://h:7840")

    def test_token_never_in_an_error(self):
        """An error message that quotes the request (http.client's invalid header value quotes the token whole) is
        not shown; the network's words are, without the token or the URL's user:password."""
        ag = self.agent(WORKER_LOG)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        secret = "bad\r\nsecret-xyz"
        v = view_now(mon, RemoteEngine(self.tok, self.log, env={"STRATA_STAGE_TOKEN": secret}))
        self.assertEqual(v["workers"][0]["agent"]["error"], "the agent's request failed (ValueError)")
        self.assertNotIn("secret-xyz", json.dumps(v))
        mon.agents_cfg = {"10.10.10.2:7841": "http://user:pw-123@127.0.0.1:9"}
        with mock.patch.object(stages.Agent, "get",
                               side_effect=OSError(f"failed with {TOKEN} for http://user:pw-123@127.0.0.1:9")):
            v = view_now(mon, RemoteEngine(self.tok, self.log))
        err = v["workers"][0]["agent"]["error"]
        self.assertIn("failed with", err)
        for s in (TOKEN, "pw-123"):
            self.assertNotIn(s, json.dumps(v))

    def test_a_request_waits_for_the_worker(self):
        """main-next: a request that finds the link lost waits for the worker (INFO remote_stage_wait= every 10 s);
        connected again, or its ERR ends the wait with the engine kept and the link closed."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        wait = ("strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (remote stage: cannot connect to "
                "10.10.10.2:7841: Connection refused), up to 180 s (STRATA_REMOTE_CONNECT_S)\n")
        self.append(wait)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]),
                         ("reconnecting", "remote stage: cannot connect to 10.10.10.2:7841: Connection refused"))
        e.stage_wait_at = time.time()                               # a heartbeat
        self.assertEqual(mon.view(e)["workers"][0]["state"], "reconnecting")
        # its ERR: the wait is over, the engine stays with the link closed (idle: the next request connects again)
        self.stage_err(e, "remote stage: the peer closed the connection; connecting again: timed out")
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]),
                         ("idle", "remote stage: the peer closed the connection; connecting again: timed out"))
        # the next request: connected again at once
        self.append("strata serve: remote stage: the link to 10.10.10.2:7841 was lost (remote stage: the peer closed "
                    "the connection): connected again\n")
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["reconnects"], w["error"]), ("connected", 1, None))

    def test_a_wait_read_late(self):
        """The log read after the wait's ERR (no Monitor open meanwhile): the wait is over all the same."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        mon.view(e)
        self.append("strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (x), up to 180 s (y)\n")
        self.stage_err(e, "remote stage: lost; connecting again: timed out")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")

    def test_a_new_wait_after_an_older_error(self):
        """An earlier request's ERR, then a new wait read late: its heartbeat says it still waits."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        mon.view(e)
        self.stage_err(e, "remote stage: lost; connecting again: timed out")
        self.append("strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (x), up to 180 s (y)\n")
        e.stage_wait_at = time.time()
        self.assertEqual(mon.view(e)["workers"][0]["state"], "reconnecting")

    def test_a_silent_wait_is_over(self):
        """No heartbeat for WAIT_STALE_S (an ERR the server did not see): not waiting any more, the link closed."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        mon.view(e)
        self.append("strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (x), up to 180 s (y)\n")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "reconnecting")
        later = time.time() + stages.WAIT_STALE_S + 1
        with mock.patch.object(stages.time, "time", return_value=later):
            self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
            e.running, e.proc = False, FakeProc()                   # a start waits as long as it runs
            self.assertEqual(mon.view(e)["workers"][0]["state"], "reconnecting")

    def test_events_in_the_log_s_order(self):
        """A relink and an ERR both read at the next look (no Monitor open meanwhile): the one later in the log wins,
        whenever each was read."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        mon.view(e)
        relink = ("strata serve: remote stage: the link to 10.10.10.2:7841 was lost (remote stage: the peer closed the "
                  "connection): connected again\n")
        self.append(relink)
        self.stage_err(e, "remote stage: the peer closed the connection")   # a later request lost it again
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["reconnects"], w["error"]["text"]),
                         ("idle", 1, "remote stage: the peer closed the connection"))
        self.append(relink)                                         # the next request: connected again
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["reconnects"], w["error"]), ("connected", 2, None))

    def test_an_earlier_run_s_error(self):
        """The first start's link refused, the next start connected, the Monitor opened later, then the engine was
        killed (OOM): no link error of the first run is its cause."""
        self.append(STARTED + "strata serve: remote stage: cannot connect to 10.10.10.2:7841: Connection refused\n" +
                    STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        e.running, e.proc = False, FakeProc(rc=-9)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]), ("down", None))
        # a request's ERR ended the engine, the next start died before its link was made: not that start's cause
        self.stage_err(e, "remote stage: the peer closed the connection")
        self.append(STARTED)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]), ("down", None))
        # nor the log's error of an earlier start
        Path(self.log).write_text(STARTED + "strata serve: remote stage: cannot connect to 10.10.10.2:7841: refused\n" +
                                  STARTED, encoding="utf-8")
        e2 = RemoteEngine(self.tok, self.log)
        e2.running, e2.proc = False, FakeProc(rc=-9)
        self.assertEqual(stages.StageMonitor().view(e2)["workers"][0]["error"], None)

    def test_a_worker_s_own_failure(self):
        """The worker failed a step (`ERR remote stage (worker): ...`): the link stays, the error is shown."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        self.stage_err(e, "remote stage (worker): CUDA error: out of memory")
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]), ("connected", "remote stage (worker): CUDA error: out of memory"))

    def test_no_log_and_timing(self):
        """No "log" in the run config: the link's state is not known (said so); STRATA_REMOTE_TIMING as the engine
        reads it."""
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        e.log_path = None
        v = mon.view(e)
        self.assertEqual((v["workers"][0]["state"], v["timing"]), ("nolog", False))
        e.running, e.proc = False, FakeProc(rc=1)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "down")
        e = RemoteEngine(self.tok, self.log, env={"STRATA_STAGE_TOKEN": TOKEN, "STRATA_REMOTE_TIMING": "1"})
        self.assertTrue(mon.view(e)["timing"])

    def test_a_map_asks_only_the_agents_it_names(self):
        """"stage_agents" as a map: a relay's next worker not in it is shown (from the relay's log) but not asked."""
        a1, a2 = self.agent(RELAY_LOG), self.agent(LAST_LOG)
        self.append(STARTED + MAIN_RUN.replace("26-47 there", "26-39 there"))
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": a1.url}
        e = RemoteEngine(self.tok, self.log)
        view_now(mon, e)
        w1, w2 = view_now(mon, e)["workers"]
        self.assertTrue(w1["agent"]["ok"])
        self.assertEqual((w2["address"], w2["agent"], w2["state"], w2["gpu"]), ("10.10.10.3:7841", None, "connected", None))
        self.assertNotIn("10.10.10.3:7841", mon.agents)
        mon.agents_cfg = True                                       # true: the relay's next worker's host too
        self.assertEqual(mon.agent_for("10.10.10.3:7841").url, "http://10.10.10.3:7840")

    def test_no_redirect_and_no_proxy(self):
        """The token goes to the agent named only: not on to where a redirect points, not through a proxy."""
        hits = []

        class Record(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hits.append((self.server.server_address[1], self.path, self.headers.get("X-Stage-Token")))
                target = getattr(self.server, "target", None)
                self.send_response(302 if target else 502)
                if target:
                    self.send_header("Location", target + self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()
        servers = []
        for _ in range(2):
            h = ThreadingHTTPServer(("127.0.0.1", 0), Record)
            threading.Thread(target=h.serve_forever, daemon=True).start()
            servers.append(h)
            self.addCleanup(h.server_close)
            self.addCleanup(h.shutdown)
        moved, other = servers
        moved.target = f"http://127.0.0.1:{other.server_address[1]}"
        a = stages.Agent(f"http://127.0.0.1:{moved.server_address[1]}", 7841)
        a.refresh(TOKEN)
        self.assertEqual((a.ok, a.error), (False, "HTTP 302"))
        self.assertEqual([p for p, _, _ in hits], [moved.server_address[1]])   # the redirect was not followed
        # a proxy in the environment (HTTP_PROXY) is not used
        hits.clear()
        ag = self.agent(WORKER_LOG)
        proxy = f"http://127.0.0.1:{other.server_address[1]}"
        other.target = None
        with mock.patch.dict(os.environ, {"HTTP_PROXY": proxy, "http_proxy": proxy, "NO_PROXY": "", "no_proxy": ""}):
            a = stages.Agent(ag.url, 7841)
            a.refresh(TOKEN)
        self.assertEqual((a.ok, hits), (True, []))

    def test_a_whole_log_too_long_to_read(self):
        """The whole log's read fails (too long to send in time): the tail is read on from, not an error at every
        look."""
        filler = "".join(f"strata stage worker: chunk T=5888 at {i * 5888}: read in {2700 + i} ms\n"
                         for i in range(stages.LOG_TAIL + 20))
        ag = self.agent(WORKER_LOG + filler)
        a = stages.Agent(ag.url, 7841)
        a.refresh(TOKEN)
        self.assertEqual(a.facts["version"], "0.1.44.1")
        real = stages.Agent.log

        def no_whole(self, token, whole):
            if whole:
                raise TimeoutError("timed out")
            return real(self, token, whole)
        with mock.patch.object(stages.Agent, "log", no_whole):
            other = filler.replace("read in", "read again in")
            ag.set_log(WORKER_LOG.replace("0.1.44.1", "0.1.49") + other)   # a new worker, a long log
            a.refresh(TOKEN)
            self.assertEqual((a.ok, a.facts["version"]), (True, "0.1.44.1"))   # its tail, with what was known
            ag.set_log(WORKER_LOG.replace("0.1.44.1", "0.1.49") + other +
                       "strata stage worker: windows: own layers 12.50 ms each (64 windows; ...)\n")
            a.refresh(TOKEN)                                        # read on from that tail's end
            self.assertEqual((a.ok, a.facts["own_ms"]), (True, 12.5))
        # a first look whose whole read fails: the tail
        with mock.patch.object(stages.Agent, "log", no_whole):
            b = stages.Agent(ag.url, 7841)
            b.refresh(TOKEN)
        self.assertEqual((b.ok, b.facts["own_ms"], b.read_whole), (True, 12.5, True))

    def test_a_relink_in_a_scanned_part(self):
        """A wait ended (Idle), the Monitor closed; the next request linked again and more than READ_MAX was written:
        the scanned part's relink is the link, and a wait after it in that part is the state."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        mon.view(e)
        self.append("strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (x), up to 180 s (y)\n")
        self.stage_err(e, "remote stage: the wait was cancelled")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
        filler = "strata serve: request done: some line of the engine's own\n" * 200
        relink = ("strata serve: remote stage: the link to 10.10.10.2:7841 was lost (remote stage: the peer closed the "
                  "connection): connected again\n")
        # a later wait, then the relink, both in the scanned part: the later one is the state
        self.append(filler + "strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (w), up to 180 s "
                    "(y)\n" + filler + relink + filler)
        with mock.patch.object(stages, "READ_MAX", 4096):
            w = mon.view(e)["workers"][0]
            self.assertEqual((w["state"], w["reconnects"], w["error"]), ("connected", 1, None))
            self.append(filler + "strata serve: remote stage: waiting for the worker at 10.10.10.2:7841 (z), up to "
                        "180 s (y)\n" + filler)
            e.stage_wait_at = time.time()
            w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]), ("reconnecting", "z"))

    def test_a_rotated_log(self):
        """The log emptied or cut short (copytruncate): what was read keeps its order before the new log's lines."""
        self.append(STARTED + MAIN_RUN + "strata serve: request done: some line of the engine's own\n" * 50)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        self.stage_err(e, "remote stage: the peer closed the connection; connecting again: timed out")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
        time.sleep(0.05)                                            # (Windows' clock: the next ERR on a later tick)
        Path(self.log).write_text("strata serve: request done: x\n", encoding="utf-8")
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]),
                         ("idle", "remote stage: the peer closed the connection; connecting again: timed out"))
        relink = "strata serve: remote stage: the link to 10.10.10.2:7841 was lost (x): connected again\n"
        self.append(relink)                                         # before the old ERR's place: later all the same
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        time.sleep(0.05)
        self.stage_err(e, "remote stage: the peer closed the connection")   # a new ERR in the new log counts
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]), ("idle", "remote stage: the peer closed the connection"))
        # an ERR read before the rotation, whose place is inside the new (shorter) log: of the old log all the same
        Path(self.log).write_text(STARTED + MAIN_RUN, encoding="utf-8")
        mon, e = stages.StageMonitor(), RemoteEngine(self.tok, self.log)
        self.stage_err(e, "remote stage: the peer closed the connection")
        self.append("strata serve: request done: some line of the engine's own\n" * 50)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
        time.sleep(0.05)
        Path(self.log).write_text(relink + "strata serve: request done: x\n" * 20, encoding="utf-8")
        self.assertGreater(os.path.getsize(self.log), e.stage_err[2])
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        # an ERR after the last read, then the rotation: its place, past the new log's end, is of the old log
        Path(self.log).write_text(STARTED + MAIN_RUN + "strata serve: request done: x\n" * 20, encoding="utf-8")
        mon, e = stages.StageMonitor(), RemoteEngine(self.tok, self.log)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        time.sleep(0.05)
        self.stage_err(e, "remote stage: the peer closed the connection")
        Path(self.log).write_text(relink, encoding="utf-8")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")

    def test_two_rotations(self):
        """An ERR of a log rotated twice since moves once at each rotation: before a link made again in between."""
        filler = "strata serve: request done: some line of the engine's own\n"
        relink = "strata serve: remote stage: the link to 10.10.10.2:7841 was lost (x): connected again\n"
        self.append(STARTED + MAIN_RUN + filler * 100)
        mon, e = stages.StageMonitor(), RemoteEngine(self.tok, self.log)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        time.sleep(0.05)
        self.stage_err(e, "remote stage: the peer closed the connection")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
        time.sleep(0.05)
        Path(self.log).write_bytes(b"")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
        self.append(filler * 20 + relink + filler * 200)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        time.sleep(0.05)
        Path(self.log).write_bytes(b"")
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"], w["reconnects"]), ("connected", None, 1))

    def test_an_err_past_the_last_line_read(self):
        """An ERR whose place is past the last whole line read (half a line after it), then a rotation: of the old
        log all the same, before a relink at the new log's start."""
        self.append(STARTED + MAIN_RUN + "strata serve: request done: half a li")
        mon, e = stages.StageMonitor(), RemoteEngine(self.tok, self.log)
        self.stage_err(e, "remote stage: the peer closed the connection")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
        time.sleep(0.05)
        Path(self.log).write_bytes(b"")
        mon.view(e)
        self.append("strata serve: remote stage: the link to 10.10.10.2:7841 was lost (x): connected again\n")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")

    def test_a_rotated_log_grown_past_the_last_read(self):
        """Rotated, then grown past what was read before the next look: its first bytes say it was rotated."""
        relink = "strata serve: remote stage: the link to 10.10.10.2:7841 was lost (x): connected again\n"
        self.append(STARTED + MAIN_RUN)
        mon, e = stages.StageMonitor(), RemoteEngine(self.tok, self.log)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
        self.stage_err(e, "remote stage: the peer closed the connection")
        self.assertEqual(mon.view(e)["workers"][0]["state"], "idle")
        time.sleep(0.05)
        Path(self.log).write_text(relink + "strata serve: request done: x\n" * 50, encoding="utf-8")
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["reconnects"]), ("connected", 1))

    def test_a_start_of_a_rotated_log(self):
        """After a rotation the current run's start is a place in the old log, not in the new one: a line there
        is read whole (here a relink across it)."""
        filler = b"strata serve: request done: some line of the engine's own\n"
        relink = b"strata serve: remote stage: the link to 10.10.10.2:7841 was lost (x): connected again\n"
        with open(self.log, "ab") as f:
            f.write(filler * 300)
        e = RemoteEngine(self.tok, self.log)
        e.log_start = os.path.getsize(self.log)
        with open(self.log, "ab") as f:
            f.write((STARTED + MAIN_RUN).encode())
        mon = stages.StageMonitor()
        with mock.patch.multiple(stages, READ_MAX=4096, SCAN_MAX=32768, HEAD_MAX=2048):
            self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
            Path(self.log).write_bytes(b"x" * 7 + b"\n")
            mon.view(e)
            with open(self.log, "ab") as f:                         # the relink begins 10 bytes before that place
                f.write(b"y" * (e.log_start - 19) + b"\n" + relink + filler * 1000)
            for _ in range(10):
                v = mon.view(e)
                if not v["provisional"]:
                    break
        self.assertEqual((v["provisional"], v["workers"][0]["reconnects"]), (False, 1))

    def test_a_start_after_a_rotation(self):
        """Rotated, then the engine started again after a line the dead engine never ended, before the next look
        (and with a look in between, and at the old run's own place): the new run's start is at its place in the
        new log, read as without the rotation."""
        filler = "strata serve: request done: some line of the engine's own\n"
        for rotate, look, same in ((False, False, False), (True, False, False), (True, True, False),
                                   (True, True, True)):
            with self.subTest(rotate=rotate, look=look, same=same):
                Path(self.log).write_text(filler * 3, encoding="utf-8")
                mon, e = stages.StageMonitor(), RemoteEngine(self.tok, self.log)
                e.log_start, e.ready_at = os.path.getsize(self.log), None
                self.append(STARTED + MAIN_RUN + filler * 100)
                self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")
                if rotate:
                    Path(self.log).write_bytes(b"x" * (e.log_start - 12) + b"\n" if same else b"")
                self.append("loading 45%")
                if look:                                            # the rotation seen while the old run is current
                    mon.view(e)
                self.assertEqual(os.path.getsize(self.log) == e.log_start, same)
                e.log_start = os.path.getsize(self.log)
                self.append(STARTED + filler * 5)
                v, st = mon.view(e), mon.main_log.state
                self.assertEqual((v["workers"][0]["state"], st["runs"], st["run_pos"], st["link"]),
                                 ("unknown", 2, e.log_start, None))
                self.assertIsNone(mon.main_log.old_start)           # (a later run: the rotated one's place is gone)

    def test_an_err_without_its_place(self):
        """The log's size could not be read when the ERR came: it is older than every line, not newer."""
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        e.stage_err = ("remote stage: the peer closed the connection", time.time(), None)
        self.assertEqual(mon.view(e)["workers"][0]["state"], "connected")

    def test_agents_off_by_default(self):
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        e = RemoteEngine(self.tok, self.log)
        with mock.patch.object(stages.Agent, "refresh") as refresh:
            v = mon.view(e, ask=True)
            wait_agents(mon)
        refresh.assert_not_called()
        self.assertEqual((v["agents"], v["workers"][0]["agent"], v["workers"][0]["state"]), (False, None, "connected"))

    def test_agent_failures_are_its_state(self):
        """Any failure of one agent (a reply that is not an agent's) is that agent's error; the next is still asked."""
        good = self.agent(WORKER_LOG)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": good.url}
        e = RemoteEngine(self.tok, self.log)
        with mock.patch.object(stages.Agent, "get", side_effect=AttributeError("'list' object has no attribute")):
            w = view_now(mon, e)["workers"][0]
        self.assertEqual((w["agent"]["ok"], w["state"]), (False, "connected"))
        self.assertFalse(mon.busy)
        self.assertTrue(view_now(mon, e)["workers"][0]["agent"]["ok"])

    def test_worker_log_tail(self):
        """The tail is read on from the last read's end; a new start in it, or no end of the last read, reads the
        whole log again (a new worker: another engine, maybe no cache)."""
        filler = "".join(f"strata stage worker: chunk T=5888 at {i * 5888}: read in {2700 + i} ms\n"
                         for i in range(stages.LOG_TAIL + 20))
        ag = self.agent(WORKER_LOG + filler)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        e = RemoteEngine(self.tok, self.log)
        wholes = []
        real = stages.Agent.log
        with mock.patch.object(stages.Agent, "log", lambda self, t, whole: (wholes.append(whole), real(self, t, whole))[1]):
            view_now(mon, e)
            ag.set_log(WORKER_LOG + filler + "strata stage worker: windows: own layers 12.50 ms each (64 windows; ...)\n")
            w = view_now(mon, e)["workers"][0]
            self.assertEqual((w["version"], w["cache_slots"], w["own_ms"]), ("0.1.44.1", 2345, 12.5))
            self.assertEqual(wholes, [True, False])                # the second look read the tail only
            # the agent started a new worker (no expert cache this time) that wrote more than a tail since
            ag.set_log(WORKER_LOG.replace("0.1.44.1", "0.1.47").replace(
                "strata generate: expert cache 2345 slots, 9.14 GiB of VRAM; policy is\n", "") + filler)
            w = view_now(mon, e)["workers"][0]
            self.assertEqual(wholes, [True, False, False, True])   # its tail, then all of it
        self.assertEqual((w["version"], w["cache_slots"]), ("0.1.47", None))

    def test_restart_with_the_same_last_line(self):
        """A new worker whose short log ends as the last one did is still read from its start."""
        last = "strata stage worker: main process 192.168.1.140: layers 26.. here, context 131072, chunk 5888\n"
        ag = self.agent(WORKER_LOG + last)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        e = RemoteEngine(self.tok, self.log)
        view_now(mon, e)
        ag.set_log(WORKER_LOG.replace("0.1.44.1", "0.1.48") + last)
        self.assertEqual(view_now(mon, e)["workers"][0]["version"], "0.1.48")

    def test_a_port_the_agent_never_ran(self):
        """A worker started by hand: the agent does not run it, and its port's log there is an old one - not down."""
        ag = self.agent(WORKER_LOG, running=False)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        self.assertEqual(view_now(mon, RemoteEngine(self.tok, self.log))["workers"][0]["state"], "connected")

    def test_agents_asked_now_and_then(self):
        ag = self.agent(WORKER_LOG)
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": ag.url}
        e = RemoteEngine(self.tok, self.log)
        calls = []
        real = stages.Agent.refresh
        with mock.patch.object(stages.Agent, "refresh", lambda self, t: (calls.append(t), real(self, t))):
            mon.view(e)                                             # without the Monitor open: not asked
            self.assertEqual(calls, [])
            for _ in range(5):
                mon.view(e, ask=True)
                wait_agents(mon)
        self.assertEqual(calls, [TOKEN])                            # once in AGENT_EVERY_S

    def test_metrics_endpoint(self):
        ag = self.agent(WORKER_LOG)
        self.append(STARTED + MAIN_RUN)
        e = RemoteEngine(self.tok, self.log)
        svc = Service(e, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.stages.agents_cfg = {"10.10.10.2:7841": ag.url}
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            # without ?workers=1 (another tab, a scraper): no card, and neither the log nor an agent is read
            with mock.patch.object(stages.MainLog, "poll", side_effect=AssertionError("the log read")):
                for q in ("", "?workers=10&xworkers=1"):
                    with urllib.request.urlopen(base + "/metrics" + q, timeout=10) as r:
                        self.assertNotIn("stages", json.loads(r.read()))
            self.assertEqual(svc.stages.asked_at, 0.0)
            with urllib.request.urlopen(base + "/metrics?requests=all&workers=1", timeout=10) as r:
                r.read()
            wait_agents(svc.stages)
            with urllib.request.urlopen(base + "/metrics?workers=1", timeout=10) as r:
                m = json.loads(r.read())
            self.assertEqual(m["stages"]["workers"][0]["gpu"]["name"], "NVIDIA GeForce RTX 2080 Ti")
        finally:
            httpd.shutdown()
            httpd.server_close()
        svc = Service(MockEngine(self.tok, "x"), self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertNotIn("stages", svc.metrics(workers=True))      # the payload as before: no key


FAKE_MAIN = r'''
import sys
from pathlib import Path
state = Path(sys.argv[sys.argv.index("--state") + 1])
sys.stderr.write("strata serve: remote stage 10.10.10.2:7841: layers 26-47 there (prompt chunk 5888, context 131072, rows fp32)\n")
sys.stderr.flush()
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    if line.startswith(("GEN", "BGEN")):
        if not state.exists():   # the first request finds the link gone, and the engine ends
            state.write_text("lost")
            print("ERR remote stage: the peer closed the connection", flush=True)
            sys.exit(1)
        print("T 90\nDONE 1 1 0 1 stop", flush=True)
'''


class EngineWiring(unittest.TestCase):
    """serve/server.py's StrataEngine with a fake engine process: its log, READY time and ERR line reach the card."""

    def setUp(self):
        from serve.server import StrataEngine
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        py = root / "native.py"
        py.write_text(FAKE_MAIN)
        if os.name == "nt":   # no shebangs on Windows: a .cmd runs the same fake engine
            exe = root / "native.cmd"
            exe.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            exe = py
            py.write_text("#!" + sys.executable + "\n" + FAKE_MAIN)
            py.chmod(0o700)
        self.e = StrataEngine(str(exe), ["--state", str(root / "state"), "--layer-split", "26", "--split-device", "0",
                                         "--remote-stage", "10.10.10.2:7841"],
                              log=str(root / "engine.log"), env=dict(os.environ, STRATA_STAGE_TOKEN=TOKEN))
        self.addCleanup(lambda: self.svc.engine.close())
        self.svc = Service(self.e, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.stages.agents_cfg = False

    def request(self):
        return list(self.svc.run([1], False, [], 8, {}, threading.Event()))

    def lose_the_link(self):
        with self.assertRaises(Exception):
            self.request()
        end = time.monotonic() + 5
        while (self.e.alive() or self.e.proc.poll() is None) and time.monotonic() < end:   # (its exit can lag)
            time.sleep(0.01)

    def test_link_lost_in_a_request(self):
        w = self.svc.metrics(workers=True)["stages"]["workers"][0]
        self.assertEqual((w["state"], w["since"], w["layers"], w["reconnects"]),
                         ("connected", self.e.ready_at, [26, 47], 0))
        self.lose_the_link()
        w = self.svc.metrics(workers=True)["stages"]["workers"][0]
        self.assertEqual((w["state"], w["error"]["text"]), ("down", "remote stage: the peer closed the connection"))
        # the ERR's place among the log's lines: after the connect line the engine wrote before it
        pos, log = self.e.stage_err[2], self.svc.stages.main_log.state
        self.assertGreater(pos, log["link"]["pos"])

    def test_restart_before_the_first_look(self):
        """The Monitor opened after the engine was started again: that start's link is a reconnect all the same."""
        self.lose_the_link()
        self.assertIsNotNone(self.e.stage_err)
        self.request()                                              # the next request starts the engine again
        self.assertIsNone(self.e.stage_err)                         # the earlier process's ERR is not this one's
        w = self.svc.metrics(workers=True)["stages"]["workers"][0]
        self.assertEqual((w["state"], w["reconnects"], w["error"]), ("connected", 1, None))

    def test_an_err_of_the_old_process_while_starting_again(self):
        """The earlier process's pump can write its ERR until the new process is started: not this one's."""
        import serve.server as server
        real = server.popen

        def popen(*a, **kw):
            self.e.stage_err = ("remote stage: the peer closed the connection", time.time(), 1 << 30)   # (late)
            return real(*a, **kw)
        with mock.patch.object(server, "popen", popen):
            self.e.restart()
        self.assertIsNone(self.e.stage_err)

    def test_a_start_after_a_line_that_never_ended(self):
        """An engine that died mid-line: the next start line begins a line of its own all the same."""
        self.e.close()
        with open(self.e.log_path, "ab") as f:
            f.write(b"loading 45%")
        self.e.restart()
        log = Path(self.e.log_path).read_bytes()
        self.assertEqual(log[self.e.log_start - 1:self.e.log_start], b"\n")
        self.assertTrue(log[self.e.log_start:].startswith(b"[strata] "))
        self.assertIn(b"\nloading 45%\r\n" if os.name == "nt" else b"\nloading 45%\n", log)


if __name__ == "__main__":
    unittest.main()
