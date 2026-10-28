"""#3469 #3437: the 0.2 releases carry no flat engine zip, so a source checkout (UPDATE.bat, setup) fetches the engine package
for its card from the signed update feed and installs its files in engine/.

    python -m unittest tools.test_setup_feed_engine
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402


class FeedEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.eng = self.root / "engine"
        self.eng.mkdir()
        (self.eng / "BUILD.json").write_text(json.dumps({"version": "0.1.45", "source": "prebuilt", "archs": [89]}), encoding="utf-8")
        (self.eng / setup.EXE).write_bytes(b"old")
        self.package = self.root / "components" / "engine" / "0.1.46-test"
        self.package.mkdir(parents=True)
        (self.package / "BUILD.json").write_text(json.dumps({"version": "0.1.46", "archs": [89]}), encoding="utf-8")
        (self.package / setup.EXE).write_bytes(b"new")
        for name, value in (("home", lambda: self.root), ("say", lambda *a, **k: None), ("ok", lambda *a, **k: None),
                            ("warn", lambda *a, **k: None)):
            patch = mock.patch.object(setup, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(setup.acquisition, "require", lambda what: None)
        patch.start()
        self.addCleanup(patch.stop)

    def run_update(self, code=0):
        return mock.patch.object(setup.subprocess, "run", return_value=types.SimpleNamespace(returncode=code, stdout="", stderr="no feed"))

    def test_the_package_becomes_the_installed_engine(self):
        with self.run_update() as run, mock.patch.object(ec, "resolve", return_value=types.SimpleNamespace(root=self.package)):
            got = setup.engine_from_feed(self.eng, updating=True)
        self.assertEqual(got, self.eng)
        cmd = run.call_args.args[0]
        self.assertIn("update", cmd)
        self.assertIn("--no-control", cmd)
        self.assertNotIn("STRATA_NO_ACQUIRE", run.call_args.kwargs["env"])        # this one step may fetch
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"new")
        meta = json.loads((self.eng / "BUILD.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["version"], meta["source"]), ("0.1.46", "prebuilt"))
        self.assertEqual((self.eng / ".previous" / setup.EXE).read_bytes(), b"old")      # one generation to roll back to

    def test_no_feed_keeps_the_installed_engine(self):
        with self.run_update(code=6):
            self.assertIsNone(setup.engine_from_feed(self.eng, updating=True))
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"old")

    def test_a_release_without_the_zip_does_not_fall_back_to_the_feed(self):
        # Strata Link: upstream installs its feed package here; the fork's releases carry the zips, and upstream's
        # package would be an engine without Link's changes (the remote stage), so it compiles instead
        gpu = {"arch": 89, "archs": [89], "vram_gb": 8.0, "index": 0}
        (self.eng / "BUILD.json").write_text(json.dumps({"version": "0.1.45", "source": "prebuilt", "archs": [89]}), encoding="utf-8")
        with mock.patch.object(setup, "engine_dir", lambda toolkit=13: self.eng),                 mock.patch.object(setup, "_urlopen", side_effect=OSError("HTTP Error 404")),                 mock.patch.object(setup, "engine_from_feed", return_value=self.eng) as feed:
            for toolkit in (13, 12):
                self.assertIsNone(setup.get_prebuilt(setup.PREBUILT_URL, gpu, "none", updating=True, toolkit=toolkit))
            feed.assert_not_called()

    def test_arm64_linux_compiles_instead_of_taking_the_feed_package(self):
        # Strata Link: upstream 0.2.7.1 installs the linux-aarch64 feed package on a DGX Spark; Link compiles it there, as before
        gpu = {"arch": 121, "archs": [121], "vram_gb": 100.0, "index": 0, "uma": True}
        (self.eng / "BUILD.json").unlink()                                    # a first install
        with mock.patch.object(setup, "engine_dir", lambda toolkit=13: self.eng), mock.patch.object(setup, "is_arm64", return_value=True), \
                mock.patch.object(setup.sys, "platform", "linux"), mock.patch.object(setup, "_urlopen") as urlopen, \
                mock.patch.object(setup, "engine_from_feed", return_value=self.eng) as feed:
            for updating in (False, True):
                self.assertIsNone(setup.get_prebuilt(setup.PREBUILT_URL, gpu, "none", updating=updating))
            feed.assert_not_called()
            urlopen.assert_not_called()

    def test_the_minimum_is_the_0_2_engine(self):
        self.assertLess((0, 1, 45), setup.MIN_ENGINE)
        self.assertGreaterEqual((0, 1, 46), setup.WIN_HIP_MIN_ENGINE)


if __name__ == "__main__":
    unittest.main()
