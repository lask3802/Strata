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
import re
import sys
import tempfile
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
                  mock.patch.object(setup, "link_version", lambda: self.LINK)):
            p.start()
            self.addCleanup(p.stop)

    def installed(self, **meta):
        (self.root / "engine" / "BUILD.json").write_text(json.dumps({"version": MIN, "archs": [89], **meta}))
        (self.root / "engine" / setup.EXE).write_bytes(b"old")

    def run_get(self, published, zipped, updating=False):
        """get_prebuilt with `published` (URL prefixes that answer) and a download that writes BUILD.json `zipped`."""
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
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu", updating=updating)
        return eng, out, heads, got

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
        self.assertIn(f"No ready-made engine for v{self.LINK}", out)

    def test_a_release_older_than_the_checkout_is_not_installed(self):
        """The latest release is still the previous Link one (this checkout's is uploading): it is refused, as an
        engine below MIN_ENGINE is (#58), and the installed engine stays."""
        self.installed(source="release", link="0.1.44.1-link.1")
        eng, out, heads, got = self.run_get([setup.PREBUILT_URL], self.release("0.1.44.1-link.1"), updating=True)
        self.assertIsNone(eng)
        self.assertIn(f"engine {self.LINK} is not published yet", out)
        self.assertFalse((self.root / "engine" / setup.PREBUILT_ASSET).exists())

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


if __name__ == "__main__":
    unittest.main()
