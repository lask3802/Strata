"""Strata Link's own releases: setup asks the fork for the engine of this checkout's LINK_VERSION (tag
v0.1.44.1-link.N) first and its latest release second, and a ready-made engine that is not that Link release or newer
(upstream's, or an older Link one) is replaced like one below MIN_ENGINE, while a current one is left alone, with no
network.  Mocked network - nothing is downloaded.

    python -m unittest tools.test_setup_link_release
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

MIN = ".".join(map(str, setup.MIN_ENGINE))


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TheCheckoutsRelease(unittest.TestCase):
    def test_link_version_is_this_engine_version_plus_a_link_number(self):
        link = setup.link_version()
        m = re.fullmatch(r"(\d+(?:\.\d+)+)-link\.(\d+)", link)
        self.assertIsNotNone(m, f"LINK_VERSION is {link!r}")
        # an upstream merge that moves CMakeLists.txt's version needs a new LINK_VERSION: the old tag holds an engine
        # MIN_ENGINE refuses, and the latest release would never be asked
        self.assertEqual(m.group(1), setup.source_version())

    def test_the_urls_come_from_the_repository_name(self):
        self.assertEqual(setup.PREBUILT_URL, f"https://github.com/{setup.REPO}/releases/latest/download/")
        self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL),
                         [f"https://github.com/{setup.REPO}/releases/download/v{setup.link_version()}/",
                          setup.PREBUILT_URL])

    def test_the_link_tag_is_the_release_its_digest_comes_from(self):
        tag = setup.PREBUILT_TAG_URL.format(version="0.1.44.1-link.2")
        self.assertEqual(setup.release_of(tag),
                         (f"https://api.github.com/repos/{setup.REPO}/releases", "0.1.44.1-link.2"))

    def test_link_releases_compare_by_version_then_number(self):
        order = [None, "", "0.1.44.1", "0.1.44.1-link.1", "0.1.44.1-link.2", "0.1.44.1-link.10", "0.1.45-link.1"]
        keys = [setup.link_key(v) for v in order]
        self.assertEqual(keys[:3], [((), 0)] * 3)               # not a Link release: older than every one
        self.assertEqual(keys[3:], sorted(keys[3:]))
        self.assertLess(keys[2], keys[3])
        self.assertLess(setup.link_key("0.1.44.1-link.9"), setup.link_key("0.1.44.1-link.10"))

    def test_link_required_is_not_newer_than_link_version(self):
        self.assertLessEqual(setup.link_key(setup.LINK_REQUIRED), setup.link_key(setup.link_version()))
        self.assertIsNotNone(re.fullmatch(r"\d+(\.\d+)+-link\.\d+", setup.LINK_REQUIRED))

    def test_which_engines_are_outdated(self):
        with mock.patch.object(setup, "link_version", return_value="0.1.44.1-link.2"):
            self.assertTrue(setup.link_outdated({"version": MIN, "source": "release"}))     # upstream's
            self.assertTrue(setup.link_outdated({"version": MIN, "source": "release", "link": "0.1.44.1-link.1"}))
            self.assertFalse(setup.link_outdated({"version": MIN, "source": "release", "link": "0.1.44.1-link.2"}))
            self.assertFalse(setup.link_outdated({"version": MIN, "source": "release", "link": "0.1.44.1-link.3"}))
            self.assertFalse(setup.link_outdated({"version": MIN, "source": "local"}))      # compiled here
            self.assertFalse(setup.link_outdated({"version": MIN, "source": "prebuilt", "backend": "hip"}))


class GetPrebuilt(unittest.TestCase):
    LINK = "0.1.44.1-link.2"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "engine").mkdir()
        for p in (mock.patch.object(setup, "ROOT", self.root),
                  mock.patch.object(setup, "link_version", lambda: self.LINK),
                  mock.patch.object(setup, "source_version", lambda: self.LINK.split("-")[0]),
                  mock.patch.object(setup, "UNCHECKED_ENGINES", set()),
                  mock.patch.object(setup, "RELEASE_RECHECK", False)):
            p.start()
            self.addCleanup(p.stop)

    def installed(self, **meta):
        (self.root / "engine" / "BUILD.json").write_text(json.dumps({"version": MIN, "archs": [89], **meta}))
        (self.root / "engine" / setup.EXE).write_bytes(b"old")

    def run_get(self, published, zipped, updating=False):
        """get_prebuilt with `published` (URL prefixes that answer) and a download that writes BUILD.json `zipped`."""
        with self.network(published, zipped) as (heads, got):
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu", updating=updating)
        return eng, out, heads, got

    @contextlib.contextmanager
    def network(self, published, zipped):
        heads, got, wrote = [], [], []

        def urlopen(req, timeout=None):
            heads.append(req.full_url)
            if not any(req.full_url.startswith(p) for p in published):
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
            return Response()

        def download(url, dst, what=None):
            got.append(url)
            wrote.append(Path(dst))
            with zipfile.ZipFile(dst, "w") as z:
                z.writestr("BUILD.json", json.dumps(zipped))
                z.writestr(setup.EXE, b"new")

        def digest(asset, base):
            data = wrote[-1].read_bytes()
            return len(data), hashlib.sha256(data).hexdigest()

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup, "download", download), mock.patch.object(setup, "engine_digest", digest):
            yield heads, got

    def release(self, link):
        return {"version": MIN, "link": link, "source": "release", "archs": [89], "ptx": True, "cuda": "13.0"}

    def test_a_current_link_engine_is_kept_without_the_network(self):
        self.installed(source="release", link=self.LINK)
        eng, out, heads, got = self.run_get([], {})
        self.assertEqual((eng, heads, got), (self.root / "engine", [], []))
        self.assertIn("already installed", out)

    def test_upstreams_engine_is_replaced_by_the_link_release(self):
        self.installed(source="release")                     # Niko1221's 0.1.44.1: no remote stage, no "link"
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        eng, out, heads, got = self.run_get([tag], self.release(self.LINK))
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(got, [tag + setup.PREBUILT_ASSET])
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"new")
        self.assertIn(f"-> {self.LINK}", out)

    def test_an_older_link_engine_is_updated_from_the_latest_when_the_tag_is_not_out(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        eng, out, heads, got = self.run_get([setup.PREBUILT_URL], self.release("0.1.44.1-link.3"))
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(got, [setup.PREBUILT_URL + setup.PREBUILT_ASSET])
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"new")
        self.assertIn("No ready-made engine for v", out)

    def test_a_release_not_newer_than_the_installed_one_is_not_installed(self):
        """The latest release is still the installed Link one (this checkout's is uploading): the installed engine
        stays, BUILD.json and all, and the tag is asked again a day later."""
        self.installed(source="release", link="0.1.44.1-link.1")
        before = (self.root / "engine" / "BUILD.json").read_text()
        eng, out, heads, got = self.run_get([setup.PREBUILT_URL], self.release("0.1.44.1-link.1"), updating=True)
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"old")
        self.assertEqual((self.root / "engine" / "BUILD.json").read_text(), before)
        self.assertFalse((self.root / "engine" / "BUILD.json.prev").exists())
        self.assertFalse((self.root / "engine" / setup.PREBUILT_ASSET).exists())
        self.assertIn("keeping the installed engine 0.1.44.1-link.1", out)
        self.assertEqual(self.note()["why"], "unpublished")

    def test_the_amd_engine_still_comes_from_upstream(self):
        """Strata Link builds no HIP zip: the default AMD engine is upstream's release of this engine version, then
        upstream's latest - not the fork's releases, which have none (an AMD install would stop)."""
        heads = []

        def urlopen(req, timeout=None):
            heads.append(req.full_url)
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup, "source_version", lambda: "0.1.44.1"):
            eng, _ = quiet(setup.get_prebuilt_hip, setup.PREBUILT_URL, {"arch": "gfx1100"})
        self.assertIsNone(eng)
        self.assertEqual(heads, [f"https://github.com/Niko1221/Strata/releases/download/v0.1.44.1/{setup.WIN_HIP_ASSET}",
                                 f"https://github.com/Niko1221/Strata/releases/latest/download/{setup.WIN_HIP_ASSET}"])
        heads.clear()
        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):   # an explicit --prebuilt is still only that
            quiet(setup.get_prebuilt_hip, "https://mirror.example/x/", {"arch": "gfx1100"})
        self.assertEqual(heads, [f"https://mirror.example/x/{setup.WIN_HIP_ASSET}"])

    def test_update_installed_engine_leaves_a_current_link_engine_alone(self):
        self.installed(source="release", link=self.LINK)
        with mock.patch.object(setup, "get_prebuilt", side_effect=AssertionError("asked to update")):
            quiet(setup.update_installed_engine, setup.PREBUILT_URL, 13)

    def test_update_installed_engine_replaces_an_older_link_engine(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        calls = []
        with mock.patch.object(setup, "get_prebuilt", lambda *a, **kw: calls.append(kw) or None), \
                mock.patch.object(setup, "update_engine_gpu", lambda tk=13: {"arch": 89, "index": 0}):
            _, out = quiet(setup.update_installed_engine, setup.PREBUILT_URL, 13)
        self.assertEqual(calls, [{"updating": True, "toolkit": 13}])
        self.assertIn("could not update the engine", out)          # nothing published here: the installed one starts

    # ---- a newer Link release is wanted, not needed: the installed engine is kept when it cannot be had (a)

    def test_setup_keeps_an_older_link_engine_when_the_new_release_cannot_be_had(self):
        """--setup or a second model (not an update), offline: the installed engine of MIN_ENGINE or newer is used as
        main-next did ("already installed"), with one warning - never compiled because its link number is older."""
        self.installed(source="release", link="0.1.44.1-link.1")
        before = (self.root / "engine" / "BUILD.json").read_text()
        eng, out, heads, got = self.run_get([], {})
        self.assertEqual((eng, got), (self.root / "engine", []))
        self.assertEqual(len(heads), 2)                            # the tag, then the latest
        self.assertEqual((self.root / "engine" / "BUILD.json").read_text(), before)
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"old")
        self.assertIn("keeping the installed engine 0.1.44.1-link.1", out)
        self.assertNotIn("compiling instead", out)

    def test_upstreams_engine_is_kept_offline_and_said_so(self):
        self.installed(source="release")                           # Niko1221's 0.1.44.1, no "link"
        eng, out, _, _ = self.run_get([], {})
        self.assertEqual(eng, self.root / "engine")
        self.assertIn("Strata Link's own engine changes are off until it updates", out)

    def test_a_download_that_fails_keeps_the_installed_engine(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        with self.network([tag], {}), mock.patch.object(setup, "download", side_effect=OSError("connection reset")):
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        self.assertEqual(eng, self.root / "engine")
        self.assertIn("connection reset", out)
        self.assertEqual(json.loads((self.root / "engine" / "BUILD.json").read_text())["link"], "0.1.44.1-link.1")

    def test_a_download_that_stops_setup_keeps_the_installed_engine(self):
        """download() stops setup (SystemExit) when the server cannot be reached five times: the installed engine is
        kept, BUILD.json and all, and the next run offline keeps it again instead of compiling."""
        self.installed(source="release", link="0.1.44.1-link.1")
        before = (self.root / "engine" / "BUILD.json").read_text()
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        with self.network([tag], {}), mock.patch.object(setup, "download", side_effect=SystemExit(1)):
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual((self.root / "engine" / "BUILD.json").read_text(), before)
        self.assertFalse((self.root / "engine" / "BUILD.json.prev").exists())
        self.assertIn("keeping the installed engine 0.1.44.1-link.1", out)
        eng, out, _, _ = self.run_get([], {})
        self.assertEqual(eng, self.root / "engine")
        self.assertNotIn("compiling instead", out)

    def test_ctrl_c_during_the_download_puts_build_json_back(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        before = (self.root / "engine" / "BUILD.json").read_text()
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        with self.network([tag], {}), mock.patch.object(setup, "download", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        self.assertEqual((self.root / "engine" / "BUILD.json").read_text(), before)
        self.assertFalse((self.root / "engine" / "BUILD.json.prev").exists())

    def test_a_failure_once_the_new_engine_is_in_is_not_called_keeping_the_old_one(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        real_ok = setup.ok

        def ok(msg):                                                     # past install_unpacked, as a chmod would
            if msg.startswith("ready-made engine "):
                raise OSError("chmod")
            real_ok(msg)

        with self.network([tag], self.release(self.LINK)), mock.patch.object(setup, "ok", ok), \
                self.assertRaises(OSError):
            quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"new")
        self.assertEqual(json.loads((self.root / "engine" / "BUILD.json").read_text())["link"], self.LINK)

    def test_one_warning_when_the_new_release_cannot_be_had(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        for published, zipped in (([], {}), ([setup.PREBUILT_URL], self.release("0.1.44.1-link.1"))):
            with self.subTest(published=published):
                (self.root / "engine" / setup.RELEASE_NOTE).unlink(missing_ok=True)
                _, out, _, _ = self.run_get(published, zipped)
                self.assertEqual(out.count("[!]"), 1, out)

    def test_an_engine_below_min_engine_is_still_replaced_or_compiled(self):
        """MIN_ENGINE stays a must: an installed engine below it is not kept when nothing newer can be had."""
        (self.root / "engine" / "BUILD.json").write_text(json.dumps(
            {"version": "0.1.0", "archs": [89], "source": "release", "link": "0.1.0-link.1"}))
        (self.root / "engine" / setup.EXE).write_bytes(b"old")
        eng, out, _, _ = self.run_get([], {})
        self.assertIsNone(eng)
        self.assertIn("compiling instead", out)

    # ---- a downloaded Link release: LINK_REQUIRED or newer, an older one than LINK_VERSION too (b)

    def test_a_first_install_takes_the_older_latest_release_and_asks_for_the_tag_again(self):
        """No engine yet, LINK_VERSION's tag not out: the latest (link.1, at least LINK_REQUIRED) is installed instead
        of compiling, and noted, so the tag is asked again a day later."""
        eng, out, heads, got = self.run_get([setup.PREBUILT_URL], self.release("0.1.44.1-link.1"))
        self.assertEqual((eng, got), (self.root / "engine", [setup.PREBUILT_URL + setup.PREBUILT_ASSET]))
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"new")
        self.assertEqual((self.note()["why"], self.note()["found"]), ("unpublished", "0.1.44.1-link.1"))

    def test_a_release_below_link_required_is_not_installed(self):
        with mock.patch.object(setup, "LINK_REQUIRED", "0.1.44.1-link.2"):
            eng, out, _, got = self.run_get([setup.PREBUILT_URL], self.release("0.1.44.1-link.1"))
        self.assertIsNone(eng)
        self.assertEqual(len(got), 1)
        self.assertIn("needs Strata Link 0.1.44.1-link.2 or newer: compiling instead", out)
        self.assertFalse((self.root / "engine" / setup.EXE).exists())

    def test_upstreams_release_zip_is_not_installed(self):
        upstream = {"version": MIN, "source": "release", "archs": [89], "ptx": True, "cuda": "13.0"}
        eng, out, _, _ = self.run_get([setup.PREBUILT_URL], upstream)
        self.assertIsNone(eng)
        self.assertIn("compiling instead", out)

    # ---- between a LINK_VERSION bump and its release: one download, then a note instead of ~150 MB every start

    def note(self):
        return json.loads((self.root / "engine" / setup.RELEASE_NOTE).read_text(encoding="utf-8"))

    def test_an_unpublished_release_is_not_downloaded_again_the_same_day(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        older = self.release("0.1.44.1-link.1")                     # the tag is not out, the latest is the old one
        _, _, heads, got = self.run_get([setup.PREBUILT_URL], older, updating=True)
        self.assertEqual(len(got), 1)
        self.assertEqual((self.note()["why"], self.note()["found"]), ("unpublished", "0.1.44.1-link.1"))
        eng, out, heads, got = self.run_get([setup.PREBUILT_URL], older, updating=True)
        self.assertEqual((eng, heads, got), (self.root / "engine", [], []))   # no network at all, the engine kept
        self.assertIn(f"engine {self.LINK} was not published yet", out)

    def test_a_day_later_only_its_tag_is_asked(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        setup.note_release(self.root / "engine", "unpublished", "0.1.44.1-link.1")
        day = setup.RELEASE_RECHECK_S + 60
        then = self.note()["checked"] + day
        with mock.patch.object(setup.time, "time", lambda: then):
            eng, out, heads, got = self.run_get([setup.PREBUILT_URL], self.release("0.1.44.1-link.1"), updating=True)
        self.assertEqual((eng, heads, got), (self.root / "engine", [tag + setup.PREBUILT_ASSET], []))
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"old")
        checked = self.note()["checked"]
        self.assertEqual(checked, then)                                  # asked now: not again for a day
        with mock.patch.object(setup.time, "time", lambda: checked + day):            # published now
            eng, out, heads, got = self.run_get([tag], self.release(self.LINK), updating=True)
        self.assertEqual((eng, got), (self.root / "engine", [tag + setup.PREBUILT_ASSET]))
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"new")
        self.assertFalse((self.root / "engine" / setup.RELEASE_NOTE).exists())

    def test_update_asks_again_now(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        setup.note_release(self.root / "engine", "unpublished", "0.1.44.1-link.1")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        with mock.patch.object(setup, "RELEASE_RECHECK", True):
            eng, _, heads, got = self.run_get([tag], self.release(self.LINK), updating=True)
        self.assertEqual((eng, got), (self.root / "engine", [tag + setup.PREBUILT_ASSET]))

    def test_another_releases_note_a_damaged_note_or_an_explicit_url_is_not_held_back(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        setup.note_release(self.root / "engine", "unpublished", "0.1.44.1-link.1")
        with mock.patch.object(setup, "link_version", lambda: "0.1.44.1-link.3"):   # LINK_VERSION moved on
            self.assertIsNone(setup.release_note(self.root / "engine"))
        for checked in ("yesterday", None, [1]):
            (self.root / "engine" / setup.RELEASE_NOTE).write_text(json.dumps(
                {"link": self.LINK, "why": "unpublished", "checked": checked}))
            self.assertIsNone(setup.release_note(self.root / "engine"), checked)
        (self.root / "engine" / setup.RELEASE_NOTE).write_text("[1, 2]")
        self.assertIsNone(setup.release_note(self.root / "engine"))
        setup.note_release(self.root / "engine", "unpublished", "0.1.44.1-link.1")
        mirror = "https://mirror.example/x/"
        with self.network([mirror], self.release(self.LINK)) as (heads, got):
            eng, _ = quiet(setup.get_prebuilt, mirror, {"arch": 89}, "gpu", updating=True)
        self.assertEqual(got, [mirror + setup.PREBUILT_ASSET])

    # ---- the ready-made engine runs once it is installed (Linux: a glibc older than the build's)

    def prebuilt_runs(self, starts, ran=True, **kw):
        with mock.patch.object(setup, "engine_starts", lambda eng, tk=13: (starts, ran)):
            return quiet(setup.prebuilt_runs, self.root / "engine", 13, **kw)

    def test_an_engine_that_does_not_start_puts_the_replaced_one_back(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        self.run_get([tag], self.release(self.LINK), updating=True)
        eng = self.root / "engine"
        self.assertEqual((eng / setup.EXE).read_bytes(), b"new")
        ok, out = self.prebuilt_runs("/lib64/libc.so.6: version `GLIBC_2.34' not found", updating=True)
        self.assertTrue(ok)                                          # the one it replaced starts instead
        self.assertEqual((eng / setup.EXE).read_bytes(), b"old")
        self.assertEqual(json.loads((eng / "BUILD.json").read_text())["link"], "0.1.44.1-link.1")
        self.assertFalse((eng / setup.PREVIOUS_ENGINE).exists())
        self.assertIn("GLIBC_2.34", out)
        self.assertNotIn("compiling instead", out)
        self.assertEqual((self.note()["why"], self.note()["found"]), ("does not start", self.LINK))
        eng2, out, heads, got = self.run_get([tag], self.release(self.LINK))   # --setup: not downloaded again, kept
        self.assertEqual((eng2, heads, got), (eng, [], []))
        self.assertIn("did not start on this PC", out)
        self.assertEqual((eng / setup.EXE).read_bytes(), b"old")

    def test_an_older_release_that_did_not_start_does_not_hold_back_link_version(self):
        """LINK_VERSION link.3, its tag not out: the latest (link.2) did not start here.  Once link.3 is published it
        is asked for a day later (its tag alone), not held back until the next LINK_VERSION."""
        self.LINK = "0.1.44.1-link.3"
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        for installed in (True, False):                              # an older Link engine kept; a first install
            with self.subTest(installed=installed):
                for name in (setup.EXE, "BUILD.json", setup.RELEASE_NOTE):
                    (self.root / "engine" / name).unlink(missing_ok=True)
                if installed:
                    self.installed(source="release", link="0.1.44.1-link.1")
                setup.note_release(self.root / "engine", "does not start", "0.1.44.1-link.2", "GLIBC_2.34")
                first = self.note()["checked"]
                eng, out, heads, got = self.run_get([tag], self.release(self.LINK))     # the same day: not asked
                self.assertEqual((heads, got), ([], []))
                self.assertEqual(eng, self.root / "engine" if installed else None)
                then = first + setup.RELEASE_RECHECK_S + 60
                with mock.patch.object(setup.time, "time", lambda: then):               # a day later, not out yet
                    eng, out, heads, got = self.run_get([setup.PREBUILT_URL], self.release("0.1.44.1-link.2"))
                self.assertEqual((heads, got), ([tag + setup.PREBUILT_ASSET], []))
                self.assertEqual((self.note()["why"], self.note()["checked"]), ("does not start", then))
                with mock.patch.object(setup.time, "time", lambda: then + setup.RELEASE_RECHECK_S + 60):  # out
                    eng, out, heads, got = self.run_get([tag], self.release(self.LINK))
                self.assertEqual((eng, got), (self.root / "engine", [tag + setup.PREBUILT_ASSET]))
                self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"new")

    def test_an_engine_put_back_below_min_engine_is_compiled_over(self):
        (self.root / "engine" / "BUILD.json").write_text(json.dumps(
            {"version": "0.1.0", "archs": [89], "source": "release", "link": "0.1.0-link.1"}))
        (self.root / "engine" / setup.EXE).write_bytes(b"old")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        self.run_get([tag], self.release(self.LINK))
        ok, out = self.prebuilt_runs("GLIBC_2.34 not found")
        self.assertFalse(ok)
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"old")
        self.assertIn("compiling instead", out)

    def test_a_timeout_leaves_no_lasting_note(self):
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        self.run_get([tag], self.release(self.LINK))
        ok, out = self.prebuilt_runs("Command '--version' timed out after 120 seconds", ran=False)
        self.assertFalse(ok)
        self.assertFalse((self.root / "engine" / setup.RELEASE_NOTE).exists())

    def test_a_first_engine_that_does_not_start_is_removed(self):
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        self.run_get([tag], self.release(self.LINK))
        ok, out = self.prebuilt_runs("error while loading shared libraries: libcublas.so.13")
        self.assertFalse(ok)
        self.assertFalse((self.root / "engine" / setup.EXE).exists())
        self.assertFalse((self.root / "engine" / "BUILD.json").exists())
        self.assertIn("compiling instead", out)
        eng, out, heads, got = self.run_get([tag], self.release(self.LINK))  # the next run: compiled, not downloaded
        self.assertEqual((eng, heads, got), (None, [], []))

    def test_an_engine_that_starts_is_kept_and_run_once(self):
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        self.run_get([tag], self.release(self.LINK))
        self.assertTrue(self.prebuilt_runs(None)[0])
        self.assertTrue(self.prebuilt_runs("not run again")[0])             # checked once, in the run that installed it
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"new")

    def test_an_installed_engine_is_not_run(self):
        self.installed(source="release", link=self.LINK)
        self.assertTrue(self.prebuilt_runs("would fail")[0])

    def engine_step(self, eng, built=None):
        a = types.SimpleNamespace(build=False, prebuilt=setup.PREBUILT_URL, yes=True)
        with mock.patch.object(setup, "get_prebuilt", lambda *x, **kw: eng), \
                mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: None), \
                mock.patch.object(setup, "engine_starts", lambda e, tk=13: ("GLIBC_2.34 not found", True)), \
                mock.patch.object(setup, "get_llama_cpp", lambda: "llama"), \
                mock.patch.object(setup, "build_engine", lambda *x, **kw: built):
            return quiet(setup.engine_step, a, {"arch": 89}, False, None, 13, "none")

    def test_setup_compiles_the_engine_when_the_ready_made_one_does_not_start(self):
        eng = self.root / "engine"
        (eng / "BUILD.json").write_text(json.dumps(self.release(self.LINK)))
        setup.UNCHECKED_ENGINES.add(eng)
        built = self.root / "built"
        built.mkdir()
        (built / "BUILD.json").write_text(json.dumps({"version": MIN, "source": "local", "lib_dirs": ["/cuda"]}))
        (llama, got, meta, lib_dirs, _, _), out = self.engine_step(eng, built)
        self.assertEqual((got, meta["source"], lib_dirs), (built, "local", ["/cuda"]))
        self.assertIn("does not start on this PC (GLIBC_2.34 not found)", out)

    def test_setup_starts_the_engine_put_back_instead_of_compiling(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        eng = self.run_get([tag], self.release(self.LINK))[0]          # installed, not run yet
        (llama, got, meta, _, _, _), out = self.engine_step(eng, built="compiled")
        self.assertEqual((got, meta["link"]), (eng, "0.1.44.1-link.1"))
        self.assertEqual((eng / setup.EXE).read_bytes(), b"old")

    def test_the_cuda12_engine_that_does_not_start_is_compiled_instead(self):
        eng = setup.engine_dir(12)
        eng.mkdir()
        (eng / "BUILD.json").write_text(json.dumps(self.release(self.LINK)))
        setup.UNCHECKED_ENGINES.add(eng)
        with mock.patch.object(setup, "get_prebuilt", lambda *a, **kw: eng), \
                mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: None), \
                mock.patch.object(setup, "engine_starts", lambda e, tk=13: ("GLIBC_2.34 not found", True)), \
                mock.patch.object(setup, "get_llama_cpp", lambda: None), \
                mock.patch.object(setup, "build_engine", lambda *a, **kw: "compiled"):
            got, _ = quiet(setup.get_cuda12_engine, setup.PREBUILT_URL, {"arch": 70}, "none", True)
        self.assertEqual(got, "compiled")

    def test_an_update_that_does_not_start_keeps_the_installed_engine(self):
        self.installed(source="release", link="0.1.44.1-link.1")
        tag = setup.PREBUILT_TAG_URL.format(version=self.LINK)
        with self.network([tag], self.release(self.LINK)) as (_, got), \
                mock.patch.object(setup, "update_engine_gpu", lambda tk=13: {"arch": 89, "index": 0}), \
                mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: None), \
                mock.patch.object(setup, "engine_starts", lambda e, tk=13: ("GLIBC_2.34 not found", True)):
            _, out = quiet(setup.update_installed_engine, setup.PREBUILT_URL, 13)
        self.assertEqual(got, [tag + setup.PREBUILT_ASSET])
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"old")
        self.assertIn("the engine it replaced (0.1.44.1-link.1) is back", out)


class EngineStarts(unittest.TestCase):
    """`strata --version` with the CUDA wheels' folders first on LD_LIBRARY_PATH; the loader's first line when not;
    whether it ran at all (a timeout or an OSError says nothing lasting)."""

    def run_it(self, result):
        seen = {}

        def run(cmd, **kw):
            seen.update(cmd=cmd, env=kw["env"])
            if isinstance(result, BaseException):
                raise result
            return result

        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.subprocess, "run", run), \
                mock.patch.object(setup, "cuda_lib_dirs", lambda tk=13: ["/v/nvidia/cu13/lib"]), \
                mock.patch.dict(setup.os.environ, {"LD_LIBRARY_PATH": "/usr/local/lib"}):
            return setup.engine_starts(Path("/e"), 13), seen

    def test_it_starts(self):
        (why, ran), seen = self.run_it(subprocess.CompletedProcess([], 0, "strata 0.1.44.1\n", ""))
        self.assertEqual((why, ran), (None, True))
        self.assertEqual(seen["cmd"][1:], ["--version"])
        self.assertEqual(seen["env"]["LD_LIBRARY_PATH"], os.pathsep.join(["/v/nvidia/cu13/lib", "/usr/local/lib"]))

    def test_the_loader_says_why_not(self):
        err = ("/e/strata: /lib64/libc.so.6: version `GLIBC_2.34' not found (required by /e/strata)\n"
               "/e/strata: /lib64/libc.so.6: version `GLIBC_2.32' not found (required by /e/strata)\n")
        (why, ran), _ = self.run_it(subprocess.CompletedProcess([], 1, "", err))
        self.assertEqual(why, "/e/strata: /lib64/libc.so.6: version `GLIBC_2.34' not found (required by /e/strata)")
        self.assertTrue(ran)

    def test_a_timeout_or_an_oserror_did_not_run_it(self):
        for e in (subprocess.TimeoutExpired(["strata"], 120), PermissionError(13, "Permission denied")):
            with self.subTest(e=type(e).__name__):
                (why, ran), _ = self.run_it(e)
                self.assertTrue(why)
                self.assertFalse(ran)

    def test_not_run_on_windows(self):
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup.subprocess, "run", side_effect=AssertionError("run")):
            self.assertEqual(setup.engine_starts(Path("/e"), 13), (None, False))

    @unittest.skipIf(setup.WIN, "a shell script as the engine")
    def test_a_real_program(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / setup.EXE
            exe.write_text('#!/bin/sh\ncase "$LD_LIBRARY_PATH" in /v/cu13/lib*) echo "strata 0.1.44.1";; '
                           '*) echo "libcublas.so.13: cannot open shared object file" >&2; exit 127;; esac\n')
            exe.chmod(0o755)
            with mock.patch.object(setup, "cuda_lib_dirs", lambda tk=13: ["/v/cu13/lib"]):
                self.assertEqual(setup.engine_starts(Path(d), 13), (None, True))
            with mock.patch.object(setup, "cuda_lib_dirs", lambda tk=13: []):
                why, ran = setup.engine_starts(Path(d), 13)
            self.assertIn("libcublas.so.13", why)
            self.assertTrue(ran)


if __name__ == "__main__":
    unittest.main()
