"""Tests for DeepSeek-V4.1's remote stage in the node agent (tools/stage_node.py: /start with "engine": "dsv41") and the
shipper (tools/stage_ship.py --engine dsv41): which tensors a worker reads, the files it is sent, the model_dir the agent
sets, the worker's command line, and that the Qwen start is the one it was.  No GPU, no network, no engine.

    python -m unittest tools.test_stage_dsv41
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_node as sn  # noqa: E402
import stage_ship as ss  # noqa: E402
import test_stage_node as tsn  # noqa: E402  (its ShipTest.agent: an agent in this process; not imported by name, or it would run here too)

# A worker that says it listens and leaves its command line and the pack's ds4.txt in its log (the log is the agent's).
FAKE_ENGINE = """\
import json, sys
print(json.dumps(sys.argv[1:]))
pack = sys.argv[sys.argv.index("--pack") + 1] if "--pack" in sys.argv else ""
try:
    print("ds4.txt:", open(pack + "/ds4.txt").read().splitlines()[-1])
except OSError:
    print("ds4.txt: none")
print("strata stage worker: listening on 127.0.0.1:7899 (a token is required)", flush=True)
import time
time.sleep(30)
"""

NL = 6   # the synthetic pack's trunk layers


def make_pack(root: Path) -> tuple[Path, Path]:
    """A pack of the real one's shape (tools/dsv41/pack_gguf.py): a 6-layer trunk in two shards, the Engram (layers 1
    and 3 here) and the drafter in files beside the model's folder.  Returns (pack folder, weights folder)."""
    models = root / "src" / "models"
    md = models / "ds-test-2bpw"
    shared = models / "ds-shared"
    md.mkdir(parents=True)
    shared.mkdir()
    sizes = {"s0.gguf": 1000, "s1.gguf": 1000}
    for fn, n in sizes.items():
        (md / fn).write_bytes(bytes((i * 7 + len(fn)) % 251 for i in range(n)))
    (shared / "engram.gguf").write_bytes(b"E" * 700)
    (shared / "drafter.gguf").write_bytes(b"D" * 300)
    pack = root / "src" / "packs" / "ds-test-2bpw"
    (pack / "tokenizer").mkdir(parents=True)
    (pack / "tokenizer" / "vocab.json").write_text("{}")
    (pack / "engram.bin").write_bytes(b"DSE1" + b"\0" * 8)
    (pack / "expert-profile.bin").write_bytes(b"prof")
    (pack / "ds4.txt").write_text(f"hidden_size 8\nnum_hidden_layers {NL}\nformat gguf\nmodel_dir {md}\n")
    lines = ["file 0 s0.gguf", "file 1 s1.gguf", "file 2 ../ds-shared/engram.gguf", "file 3 ../ds-shared/drafter.gguf"]
    # tensor: name, dtype, file, offset, bytes
    t = [("embed.weight", 0, 0, 100), ("head.weight", 0, 100, 100), ("norm.weight", 0, 200, 10),
         ("layers.0.attn.wq.weight", 0, 300, 50), ("layers.1.attn.wq.weight", 0, 400, 50),
         ("layers.2.attn.wq.weight", 0, 500, 50), ("layers.1.engram.embed.weight", 2, 0, 400),
         ("layers.3.engram.embed.weight", 2, 400, 300), ("layers.3.engram.q_weight", 1, 100, 20),
         ("layers.3.attn.wq.weight", 1, 0, 50), ("layers.3.ffn.experts.0.w1.weight", 1, 50, 200),
         ("layers.4.attn.wq.weight", 1, 300, 50), ("layers.4.ffn.experts.0.w1.weight", 1, 400, 100),
         ("layers.5.attn.wq.weight", 1, 600, 50), ("layers.5.ffn.experts.0.w1.weight", 1, 700, 100),
         ("mtp.0.attn.wq.weight", 3, 0, 300)]
    lines += [f"t {n} Q5_K {f} {o} {b} 4 4" for n, f, o, b in t]
    (pack / "tensors.txt").write_text("\n".join(lines) + "\n")
    return pack, md


class WantedTest(unittest.TestCase):
    def test_what_a_worker_reads(self):
        w = lambda n, lo=3, hi=6: ss.dsv41_wanted(n, lo, hi)   # noqa: E731
        self.assertFalse(w("layers.2.attn.wq.weight"))          # the main process's layer
        self.assertTrue(w("layers.3.attn.wq.weight"))
        self.assertTrue(w("layers.5.ffn.experts.383.w2.weight"))
        self.assertFalse(w("layers.6.attn.wq.weight"))          # past the range (the drafter's blocks are mtp.*)
        self.assertFalse(w("embed.weight"))
        self.assertFalse(w("head.weight"))
        self.assertFalse(w("mtp.0.attn.wq_a.weight"))
        self.assertTrue(w("norm.weight"))
        # the Engram's tensors are a layer's own: a worker from layer 15 on has none, one that holds layer 14 would
        self.assertFalse(w("layers.14.engram.embed.weight", 15, 40))
        self.assertTrue(w("layers.14.engram.embed.weight", 10, 40))


class ItemsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pack, self.md = make_pack(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def by_rel(self, items):
        return {it.rel: it for it in items}

    def test_files_and_ranges_of_a_worker_from_layer_3(self):
        items = self.by_rel(ss.dsv41_items(self.pack, None, 3))
        self.assertEqual(sorted(items), ["models/ds-shared/drafter.gguf", "models/ds-shared/engram.gguf",
                                         "models/ds-test-2bpw/s0.gguf", "models/ds-test-2bpw/s1.gguf"])
        s0 = items["models/ds-test-2bpw/s0.gguf"]
        self.assertEqual(s0.ranges, [[200, 10]])   # norm.weight only: layers 0-2, embed and head stay home
        self.assertEqual(s0.total, 1000)
        s1 = items["models/ds-test-2bpw/s1.gguf"]
        # layer 3's attention + experts + its Engram q (a layer in range), layers 4 and 5: one run at 0-250 + 100-120
        self.assertEqual(s1.ranges, [[0, 250], [300, 50], [400, 100], [600, 50], [700, 100]])
        self.assertFalse(s1.whole)
        # the Engram's table and the drafter: opened by the engine, read by a worker from layer 4 on only if its range holds them
        self.assertEqual(items["models/ds-shared/drafter.gguf"].ranges, [])
        self.assertEqual(items["models/ds-shared/engram.gguf"].ranges, [[400, 300]])   # (layer 3 is in range)
        self.assertEqual(self.by_rel(ss.dsv41_items(self.pack, None, 4))["models/ds-shared/engram.gguf"].ranges, [])

    def test_to_the_last_layer_or_an_end(self):
        full = self.by_rel(ss.dsv41_items(self.pack, None, 4))["models/ds-test-2bpw/s1.gguf"].ranges
        cut = self.by_rel(ss.dsv41_items(self.pack, None, 4, 5))["models/ds-test-2bpw/s1.gguf"].ranges
        self.assertEqual(full, [[300, 50], [400, 100], [600, 50], [700, 100]])
        self.assertEqual(cut, [[300, 50], [400, 100]])

    def test_bad_ranges_and_missing_weights(self):
        for lo, hi in ((0, None), (6, None), (3, 3), (3, 9)):
            with self.assertRaisesRegex(SystemExit, "a range of the trunk"):
                ss.dsv41_items(self.pack, None, lo, hi)
        with self.assertRaisesRegex(SystemExit, "weights' folder"):
            ss.dsv41_items(self.pack, Path(self.tmp.name, "nowhere"), 3)

    def test_a_file_outside_the_models_folder_is_refused(self):
        t = (self.pack / "tensors.txt").read_text().replace("../ds-shared/drafter.gguf", "../../drafter.gguf")
        (self.pack / "tensors.txt").write_text(t)
        with self.assertRaisesRegex(SystemExit, "outside the models folder"):
            ss.dsv41_items(self.pack, None, 3)

    def test_model_dir_option_over_ds4_txt(self):
        # a pack made on another PC: ds4.txt's model_dir does not exist here, --model-dir is where the files are
        txt = (self.pack / "ds4.txt").read_text().replace(str(self.md), "/not/this/pc/ds-test-2bpw")
        (self.pack / "ds4.txt").write_text(txt)
        with self.assertRaises(SystemExit):
            ss.dsv41_items(self.pack, None, 3)
        self.assertEqual(len(ss.dsv41_items(self.pack, self.md, 3)), 4)

    def test_dry_run_lists_without_an_agent(self):
        out = io.StringIO()
        argv = ["stage_ship.py", "--engine", "dsv41", "--pack-dir", str(self.pack), "--layers", "3-", "--dry-run"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            ss.main()
        text = out.getvalue()
        self.assertIn("models/ds-test-2bpw/s1.gguf", text)
        self.assertIn("packs/ds-test-2bpw/tensors.txt", text)
        self.assertRegex(text, r"total [0-9.]+ GiB wanted \(\d+ files\)")

    def test_qwen_options_still_required_for_qwen(self):
        argv = ["stage_ship.py", "--agent", "http://x", "--pack-dir", "p", "--layers", "1-2"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as cm:
                ss.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--model-dir, --profile", err.getvalue())


class ShipDsv41Test(unittest.TestCase):
    """The whole way: a pack shipped to an agent in this process, then the agent's view of it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.pack, self.md = make_pack(root)
        self.dst = root / "dst"
        self.servers = []
        self.piece = mock.patch.object(ss, "PIECE", 64)
        self.piece.start()

    def tearDown(self):
        for srv in self.servers:
            srv.shutdown()
            srv.server_close()
        self.piece.stop()
        self.tmp.cleanup()

    def test_ship_and_resume(self):
        agent, _ = tsn.ShipTest.agent(self)
        items = ss.plan_dsv41(self.pack, None, None, 3, None)
        sent = ss.ship(agent, items, say=lambda m: None, parallel=2)
        self.assertEqual(sent, sum(it.nbytes for it in items))
        for it in items:
            got = (self.dst / it.rel).read_bytes()
            want = it.local.read_bytes()
            self.assertEqual(len(got), len(want), it.rel)          # every file at its full size (the engine maps it whole)
            for off, n in it.ranges:
                self.assertEqual(got[off:off + n], want[off:off + n], it.rel)
        # the files a worker reads none of are there, and empty of data
        self.assertEqual((self.dst / "models/ds-shared/drafter.gguf").read_bytes(), b"\0" * 300)
        self.assertEqual((self.dst / "models/ds-test-2bpw/s0.gguf").read_bytes()[:200], b"\0" * 200)
        self.assertEqual((self.dst / "packs/ds-test-2bpw/expert-profile.bin").read_bytes(), b"prof")
        self.assertEqual(ss.ship(agent, items, say=lambda m: None), 0)   # all there: nothing again
        # a changed source (same size, other bytes) is sent again, the empty files too
        (self.md / "s1.gguf").write_bytes(b"\x01" * 1000)
        items = ss.plan_dsv41(self.pack, None, None, 3, None)
        self.assertGreater(ss.ship(agent, items, say=lambda m: None), 0)
        self.assertEqual((self.dst / "models/ds-test-2bpw/s1.gguf").read_bytes()[:250], b"\x01" * 250)

    def test_a_node_runs_what_was_shipped(self):
        agent, _ = tsn.ShipTest.agent(self)
        ss.ship(agent, ss.plan_dsv41(self.pack, None, None, 3, None), say=lambda m: None)
        w = sn.Workers({"dsv41_exe": "x", "bind": "x", "data_dir": str(self.dst)})
        m = w.model({"engine": "dsv41", "model": {"dir": "ds-test-2bpw", "pack": "ds-test-2bpw"}})
        self.assertEqual(Path(m["pack"]), self.dst / "packs" / "ds-test-2bpw")
        self.assertEqual(Path(m["model_dir"]), self.dst / "models" / "ds-test-2bpw")

    def test_node_paths(self):
        p = ss.node_paths_dsv41("/srv/stage/", "ds-test-2bpw", Path("packs/ds-pack"), 20)
        self.assertEqual(p["node.json"], {"dsv41_exe": "/srv/stage/bin/strata-dsv41", "lib_dirs": ["/srv/stage/bin"],
                                          "dsv41_pack": "/srv/stage/packs/ds-pack"})
        self.assertEqual(p["/start"], {"engine": "dsv41", "begin": 20, "model": {"dir": "ds-test-2bpw", "pack": "ds-pack"}})
        win = ss.node_paths_dsv41("D:\\stage", "m", Path("p"), 20)
        self.assertEqual(win["node.json"]["dsv41_exe"], "D:\\stage\\bin\\strata-dsv41.exe")


class PinModelDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_replaced_once(self):
        node = Path("/node/models/m")   # (written with this PC's separators, as the agent writes it)
        (self.d / "ds4.txt").write_bytes(b"a 1\nmodel_dir /ct580/models/m\nb 2\n")
        self.assertTrue(sn.pin_model_dir(self.d, node))
        self.assertEqual((self.d / "ds4.txt").read_bytes(), f"a 1\nmodel_dir {node}\nb 2\n".encode())
        self.assertFalse(sn.pin_model_dir(self.d, node))   # already: not written again
        self.assertFalse((self.d / "ds4.txt.tmp").exists())

    def test_crlf_and_a_pack_without_the_line(self):
        (self.d / "ds4.txt").write_bytes(b"a 1\r\nmodel_dir /x\r\n")
        y = Path("/y")
        sn.pin_model_dir(self.d, y)
        self.assertEqual((self.d / "ds4.txt").read_bytes(), f"a 1\r\nmodel_dir {y}\r\n".encode())
        (self.d / "ds4.txt").write_bytes(b"a 1\nb 2\n")
        sn.pin_model_dir(self.d, y)
        self.assertEqual((self.d / "ds4.txt").read_bytes(), f"a 1\nb 2\nmodel_dir {y}\n".encode())

    def test_only_the_model_dir_line_is_not_what_the_fingerprint_hashes(self):
        # the main process's hello and the worker's must still match: ds4_stage.hpp's pack_hash leaves this line out
        a = b"a 1\nmodel_dir /ct580/m\nb 2\n"
        (self.d / "ds4.txt").write_bytes(a)
        sn.pin_model_dir(self.d, Path("/node/m"))
        strip = lambda b: [x for x in b.splitlines() if not x.startswith(b"model_dir ")]   # noqa: E731
        self.assertEqual(strip(a), strip((self.d / "ds4.txt").read_bytes()))


class StartTest(unittest.TestCase):
    """/start against a fake engine: the command line, and the Qwen one beside it."""

    PORT = 7899

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.fake = self.d / "fake_engine.py"
        self.fake.write_text(FAKE_ENGINE)
        real = subprocess.Popen
        self.argv = []

        def popen(args, **kw):
            self.argv.append(list(args))
            return real([sys.executable, "-I", str(self.fake)] + list(args[1:]), **kw)

        self.patches = [mock.patch.object(sn.subprocess, "Popen", side_effect=popen),
                        mock.patch.object(sn, "LISTEN_POLL_S", 0.05)]
        for p in self.patches:
            p.start()
        pack, md = make_pack(self.d)
        self.data = self.d / "node"
        (self.data / "packs").mkdir(parents=True)
        (self.data / "models").mkdir()
        # a shipped copy: the pack as it came, ds4.txt still the first PC's
        import shutil
        shutil.copytree(pack, self.data / "packs" / "ds-pack")
        (self.data / "models" / "ds-model").mkdir()
        self.w = self.workers()

    def workers(self, **extra):
        cfg = {"exe": "strata", "dsv41_exe": "strata-dsv41", "bind": "127.0.0.1", "_token": "t", "log_dir": str(self.d),
               "drop_cache": True, "restart": False, "data_dir": str(self.data), "pack": "p", "native": "n",
               "expert_profile": "e", "dsv41_args": ["--max-context", "8192", "--vision", "off"], **extra}
        return sn.Workers(cfg)

    def tearDown(self):
        self.w.close()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def log(self) -> list[str]:
        return (self.d / f"stage-worker-{self.PORT}.log").read_text().splitlines()

    def test_the_workers_command_line(self):
        r = self.w.start({"engine": "dsv41", "begin": 20, "port": self.PORT, "prefill": 5888, "cache": "auto",
                          "model": {"dir": "ds-model", "pack": "ds-pack"}, "extra": ["--cpu-threads", "6"]})
        self.assertTrue(r["ok"], r)
        pack = str(self.data / "packs" / "ds-pack")
        self.assertEqual(self.argv[0], ["strata-dsv41", "--pack", pack, "--stage-worker", str(self.PORT), "--stage-begin",
                                        "20", "--stage-bind", "127.0.0.1", "--max-context", "8192", "--vision", "off",
                                        "--cpu-threads", "6"])
        # (none of the Qwen engine's: it stops on a flag it does not know)
        for f in ("--serve", "--native", "--ple-gguf", "--expert-cache", "--prefill", "--expert-profile", "--stage-end"):
            self.assertNotIn(f, self.argv[0])
        self.assertEqual(json.loads(self.log()[0]), self.argv[0][1:])

    def test_the_pack_is_pointed_at_the_nodes_weights_before_the_start(self):
        self.assertIn("model_dir " + str(self.d / "src" / "models"), (self.data / "packs/ds-pack/ds4.txt").read_text())
        r = self.w.start({"engine": "dsv41", "begin": 20, "port": self.PORT,
                          "model": {"dir": "ds-model", "pack": "ds-pack"}})
        self.assertTrue(r["ok"], r)
        # the engine saw the node's folder when it opened ds4.txt
        self.assertEqual(self.log()[1], "ds4.txt: model_dir " + str(self.data / "models" / "ds-model"))
        # a restart after a ship that wrote the first PC's line again: set again
        (self.data / "packs/ds-pack/ds4.txt").write_text("model_dir /ct580/m\n")
        self.assertTrue(self.w.start({"engine": "dsv41", "begin": 20, "port": self.PORT,
                                      "model": {"dir": "ds-model", "pack": "ds-pack"}})["ok"])
        self.assertEqual(self.log()[1], "ds4.txt: model_dir " + str(self.data / "models" / "ds-model"))

    def test_nodes_own_pack_is_left_alone(self):
        own = self.d / "own-pack"
        own.mkdir()
        (own / "ds4.txt").write_text("model_dir /somewhere\n")
        w = self.workers(dsv41_pack=str(own))
        try:
            self.assertTrue(w.start({"engine": "dsv41", "begin": 20, "port": self.PORT})["ok"])
            self.assertEqual((own / "ds4.txt").read_text(), "model_dir /somewhere\n")
            self.assertEqual(self.argv[0][2], str(own))
        finally:
            w.close()

    def test_refused_without_starting_anything(self):
        base = {"engine": "dsv41", "begin": 20, "port": self.PORT, "model": {"dir": "ds-model", "pack": "ds-pack"}}
        for req, why in (({"end": 40, "next": "h:1"}, "no end, no next"),
                         ({"next": "h:1"}, "no end, no next"),
                         ({"begin": 0}, "first layer is 1 or more"),
                         ({"extra": ["--kv", "int8"]}, "flags not allowed"),            # a Qwen flag: strata-dsv41 stops on it
                         ({"extra": ["--pack", "/etc"]}, "flags not allowed"),
                         ({"engine": "other"}, "engine: one of"),
                         ({"model": {"dir": "../x", "pack": "ds-pack"}}, "plain file or directory name"),
                         ({"model": {"dir": "ds-model", "pack": "missing"}}, "not shipped here"),
                         ({"model": None}, "names no dsv41_pack")):
            r = self.w.start({**base, **req})
            self.assertFalse(r["ok"], req)
            self.assertIn(why, r["error"], req)
        self.assertEqual(self.argv, [])
        self.assertIn("names no dsv41_exe", sn.Workers({"bind": "x", "data_dir": str(self.data)}).start(base)["error"])

    def test_extra_allow_widens_the_list(self):
        w = self.workers(extra_allow=["--dspark"])
        self.assertIsNone(w._refusal({"engine": "dsv41", "begin": 20, "extra": ["--dspark", "none"],
                                      "model": {"dir": "ds-model", "pack": "ds-pack"}}))

    def test_the_page_cache_is_kept_for_dsv41_and_dropped_for_qwen(self):
        with mock.patch.object(sn, "drop_cache") as drop:
            self.assertTrue(self.w.start({"engine": "dsv41", "begin": 20, "port": self.PORT,
                                          "model": {"dir": "ds-model", "pack": "ds-pack"}})["ok"])
            drop.assert_not_called()
            self.w.stop(self.PORT)
            # the Qwen engine: its worker's experts are pinned, the cache is a second copy (node.json's own model)
            (self.d / "p").mkdir()
            (self.d / "p" / "f.bin").write_bytes(b"x")
            w = self.workers(pack=str(self.d / "p"))
            try:
                self.assertTrue(w.start({"begin": 20, "port": self.PORT})["ok"])
                drop.assert_called_once_with(self.d / "p" / "f.bin")
            finally:
                w.close()

    def test_the_qwen_start_is_what_it_was(self):
        r = self.w.start({"begin": 24, "port": self.PORT, "prefill": 5888, "cache": "auto", "extra": ["--kv", "int8"]})
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.argv[0], ["strata", "--serve", "--stage-worker", str(self.PORT), "--stage-begin", "24",
                                        "--pack", "p", "--native", "n", "--expert-profile", "e", "--expert-cache", "auto",
                                        "--prefill", "5888", "--stage-bind", "127.0.0.1", "--kv", "int8"])
        # and the relay's flags where they were
        self.w.stop(self.PORT)
        self.assertTrue(self.w.start({"begin": 24, "end": 40, "next": "h:7842", "port": self.PORT})["ok"])
        self.assertEqual(self.argv[1][1:9], ["--serve", "--stage-worker", str(self.PORT), "--stage-begin", "24",
                                             "--stage-end", "40", "--stage-next"])


if __name__ == "__main__":
    unittest.main()
