"""Tests for the remote-stage node agent's file store (tools/stage_node.py) and the shipper's node paths
(tools/stage_ship.py): no GPU, no network, no engine.

    python -m unittest tools.test_stage_node
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_node as sn  # noqa: E402
import stage_ship as ss  # noqa: E402


def on_disk(p: Path) -> int:
    """The bytes a file occupies on the drive (a sparse file's holes take none)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        f = ctypes.windll.kernel32.GetCompressedFileSizeW
        f.restype = wintypes.DWORD
        hi = wintypes.DWORD()
        lo = f(str(p), ctypes.byref(hi))
        return (hi.value << 32) | lo
    return os.stat(p).st_blocks * 512


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.st = sn.Store({"data_dir": str(self.d), "drop_cache": False})

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, rel, off, total, data, fp, mode=None, st=None):
        return (st or self.st).put(rel, off, total, io.BytesIO(data), len(data), mode, fp)

    def test_ranges_merge_and_fingerprint(self):
        self.assertTrue(self.put("models/m/a.gguf", 0, 100, b"x" * 10, "fp1")["ok"])
        self.put("models/m/a.gguf", 50, 100, b"y" * 10, "fp1")
        rg = self.st.ranges("models/m/a.gguf")
        self.assertEqual(rg["ranges"], [[0, 10], [50, 10]])
        self.assertEqual(rg["fp"], "fp1")
        self.put("models/m/a.gguf", 10, 100, b"z" * 5, "fp1")
        self.assertEqual(self.st.ranges("models/m/a.gguf")["ranges"], [[0, 15], [50, 10]])

    def test_another_source_voids_what_was_written(self):
        self.put("models/m/a.gguf", 0, 100, b"x" * 10, "fp1")
        self.put("models/m/a.gguf", 20, 100, b"z" * 5, "fp2")   # same size, another fingerprint
        rg = self.st.ranges("models/m/a.gguf")
        self.assertEqual((rg["ranges"], rg["fp"]), ([[20, 5]], "fp2"))
        data = (self.d / "models/m/a.gguf").read_bytes()
        self.assertEqual(data[:10], b"\0" * 10)
        self.assertEqual(data[20:25], b"z" * 5)
        self.put("models/m/a.gguf", 0, 200, b"w" * 4, "fp2")    # another size
        self.assertEqual(self.st.ranges("models/m/a.gguf")["ranges"], [[0, 4]])
        self.assertEqual((self.d / "models/m/a.gguf").stat().st_size, 200)

    def test_legacy_sidecar_is_kept(self):
        (self.d / "models/m").mkdir(parents=True)
        (self.d / "models/m/b.gguf").write_bytes(b"\0" * 64)
        (self.d / "models/m/b.gguf.ranges.json").write_text(json.dumps([[0, 32]]))
        self.put("models/m/b.gguf", 32, 64, b"q" * 8, "fpb")
        rg = self.st.ranges("models/m/b.gguf")
        self.assertEqual((rg["ranges"], rg["fp"]), ([[0, 40]], "fpb"))

    def test_executables_need_allow_put_bin(self):
        self.assertFalse(self.put("bin/strata", 0, 4, b"abcd", None, mode="755")["ok"])
        self.assertFalse(self.put("bin/strata", 0, 4, b"abcd", None)["ok"])
        st2 = sn.Store({"data_dir": str(self.d), "drop_cache": False, "allow_put_bin": True})
        self.assertTrue(self.put("bin/strata", 0, 4, b"abcd", None, mode="755", st=st2)["ok"])

    def test_path_outside_data_dir(self):
        with self.assertRaises(ValueError):
            self.put("../evil", 0, 1, b"e", None)

    def test_new_file_is_sparse(self):
        total, n = 2 << 30, 1 << 20
        self.assertTrue(self.put("models/m/big.gguf", total - n, total, b"x" * n, "fp")["ok"])
        p = self.d / "models/m/big.gguf"
        self.assertEqual(p.stat().st_size, total)
        self.assertLess(on_disk(p), 64 << 20, "the gap before the written range was allocated")
        with open(p, "rb") as f:
            f.seek(total - n - 4)
            self.assertEqual(f.read(8), b"\0\0\0\0xxxx")


class StartCheckTest(unittest.TestCase):
    def test_refused_before_a_process_starts(self):
        w = sn.Workers({"exe": "false", "pack": "p", "native": "n", "ple_gguf": "q", "expert_profile": "e", "bind": "x"})
        self.assertIn("not allowed", w.start({"begin": 40, "extra": ["--expert-profile-save", "x"]})["error"])
        self.assertIn("both end and next", w.start({"begin": 24, "end": 40})["error"])


class NodePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = Path(self.tmp.name, "rvn")
        self.m.mkdir()
        for i in (1, 2):
            (self.m / f"M-0000{i}-of-00002.gguf").write_bytes(b"")

    def tearDown(self):
        self.tmp.cleanup()

    def test_posix_node(self):
        p = ss.node_paths("/srv/stage/", self.m, Path("packs/pk"), Path("data/prof.bin"))
        self.assertEqual(p["exe"], "/srv/stage/bin/strata")
        self.assertEqual(p["native"], "/srv/stage/models/rvn/M-00001-of-00002.gguf")
        self.assertEqual(p["ple_gguf"], "/srv/stage/models/rvn/M-00002-of-00002.gguf")

    def test_windows_node(self):
        p = ss.node_paths("R:\\stage", self.m, Path("packs/pk"), Path("data/prof.bin"))
        self.assertEqual(p["exe"], "R:\\stage\\bin\\strata.exe")
        self.assertEqual(p["pack"], "R:\\stage\\packs\\pk")
        self.assertEqual(p["expert_profile"], "R:\\stage\\data\\prof.bin")


if __name__ == "__main__":
    unittest.main()
