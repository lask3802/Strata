"""tools/make_release.py on fake build folders: the zip's layout and BUILD.json are what setup.py reads, SHA256.txt is
upstream's format, and a build that is not a release (native, a stale binary, a LINK_VERSION of another version) is
refused.  Nothing is compiled.

    python -m unittest tools.test_make_release
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import make_release  # noqa: E402
import setup  # noqa: E402

VERSION = setup.source_version()
LINK = f"{VERSION}-link.3"


def build_folder(path: Path, cache: dict, cuda: str, files: dict) -> Path:
    path.mkdir(parents=True)
    (path / "CMakeCache.txt").write_text("".join(f"{k}:{t}={v}\n" for k, (t, v) in cache.items()), encoding="utf-8")
    cc = path / "CMakeFiles" / "4.4.3" / "CMakeCUDACompiler.cmake"
    cc.parent.mkdir(parents=True)
    cc.write_text(f'set(CMAKE_CUDA_COMPILER "/x/nvcc")\nset(CMAKE_CUDA_COMPILER_VERSION "{cuda}")\n', encoding="utf-8")
    for name, data in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(data)
    return path


def engine_build(path: Path, platform="windows", cuda="13.0.88", archs="75-real;86-real;89-real;120",
                 portable="ON", said=VERSION, kq=True) -> Path:
    exe, kql = ("strata.exe", "strata_kq.dll") if platform == "windows" else ("strata", "libstrata_kq.so")
    files = {exe: b"MZ engine=" + said.encode() + b"\n rest", **({kql: b"kq"} if kq else {})}
    return build_folder(path, {"CMAKE_PROJECT_VERSION": ("STATIC", VERSION), "CMAKE_BUILD_TYPE": ("STRING", "Release"),
                               "STRATA_ENABLE_CUDA": ("BOOL", "ON"), "STRATA_PORTABLE": ("BOOL", portable),
                               "CMAKE_CUDA_ARCHITECTURES": ("UNINITIALIZED", archs)}, cuda, files)


def vision_build(path: Path, platform="windows", cuda="13.0.88", archs="75-real;86-real;89-real;120") -> Path:
    vexe = "strata-vision.exe" if platform == "windows" else "strata-vision"
    return build_folder(path, {"CMAKE_BUILD_TYPE": ("STRING", "Release"), "STRATA_VISION_CUDA": ("BOOL", "ON"),
                               "STRATA_PORTABLE": ("BOOL", "ON"), "CMAKE_CUDA_ARCHITECTURES": ("UNINITIALIZED", archs)},
                        cuda, {f"bin/{vexe}": b"vision"})


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


class Pack(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.out = self.tmp / "dist"

    def test_the_cuda13_windows_zip(self):
        z = quiet(make_release.pack, engine_build(self.tmp / "e"), vision_build(self.tmp / "v"), self.out, "windows",
                  LINK, "abc123")
        self.assertEqual(z.name, "strata-windows-x64.zip")
        with zipfile.ZipFile(z) as f:
            self.assertEqual(f.namelist(), ["strata.exe", "strata_kq.dll", "strata-vision.exe", "BUILD.json"])
            meta = json.loads(f.read("BUILD.json"))
        self.assertEqual(meta, {"version": VERSION, "link": LINK, "source": "release", "archs": [75, 86, 89, 120],
                                "vision_archs": [75, 86, 89, 120], "ptx": True, "cuda": "13.0", "vision": "gpu",
                                "portable": True, "commit": "abc123"})
        # what setup.py does with it: the version passes MIN_ENGINE, the link this checkout's (same number) passes too
        ver = tuple(int(x) for x in meta["version"].split(".")[:4] if x.isdigit())
        self.assertGreaterEqual(ver, setup.MIN_ENGINE)
        with mock.patch.object(setup, "link_version", return_value=LINK):
            self.assertFalse(setup.link_outdated(meta))
        self.assertEqual(setup.prebuilt_vision(meta, {"arch": 89}, "gpu"), "gpu")

    def test_the_cuda12_zip_is_the_experimental_one(self):
        archs = "60-real;61-real;70-real;75-real;80-real;86-real;89"
        z = quiet(make_release.pack, engine_build(self.tmp / "e", cuda="12.9.86", archs=archs),
                  vision_build(self.tmp / "v", cuda="12.9.86", archs=archs), self.out, "windows", LINK, None)
        self.assertEqual(z.name, "strata-windows-x64-cuda12.zip")
        with zipfile.ZipFile(z) as f:
            meta = json.loads(f.read("BUILD.json"))
        self.assertEqual((meta["archs"], meta["ptx"], meta["cuda"], meta["toolkit"], meta["experimental"]),
                         ([60, 61, 70, 75, 80, 86, 89], True, "12.9", 12, True))
        self.assertNotIn("commit", meta)

    def test_the_linux_zip_keeps_the_programs_executable(self):
        z = quiet(make_release.pack, engine_build(self.tmp / "e", platform="linux"),
                  vision_build(self.tmp / "v", platform="linux"), self.out, "linux", LINK, None)
        self.assertEqual(z.name, "strata-linux-x64.zip")
        with zipfile.ZipFile(z) as f:
            modes = {i.filename: (i.external_attr >> 16) & 0o777 for i in f.infolist()}
        self.assertEqual(modes["strata"], 0o755)
        self.assertEqual(modes["strata-vision"], 0o755)
        self.assertEqual(modes["libstrata_kq.so"], 0o644)

    def test_the_names_are_setups(self):
        win = "windows" if setup.WIN else "linux"
        self.assertEqual(make_release.asset_name(win, 13), setup.PREBUILT_ASSET)
        self.assertEqual(make_release.asset_name(win, 12), setup.CUDA12_ASSET)

    def test_sha256_txt_lists_every_zip_like_upstream(self):
        quiet(make_release.pack, engine_build(self.tmp / "e"), vision_build(self.tmp / "v"), self.out, "windows",
              LINK, None)
        quiet(make_release.pack, engine_build(self.tmp / "e12", cuda="12.9.86", archs="75-real;89"),
              vision_build(self.tmp / "v12", cuda="12.9.86", archs="75-real;89"), self.out, "windows", LINK, None)
        sums = make_release.write_sums(self.out).read_bytes().decode("ascii")
        want = "".join(f"{hashlib.sha256((self.out / n).read_bytes()).hexdigest()} *{n}\n"
                       for n in ("strata-windows-x64-cuda12.zip", "strata-windows-x64.zip"))
        self.assertEqual(sums, want)

    def test_what_is_not_a_release_is_refused(self):
        cases = {
            "native": dict(e=dict(portable="OFF")),
            "stale binary": dict(e=dict(said="0.1.40")),
            "an unnamed arch list": dict(e=dict(archs="native")),
            "two toolkits": dict(v=dict(cuda="12.9.86")),
            "no strata_kq": dict(e=dict(kq=False)),
        }
        for why, kw in cases.items():
            with self.subTest(why), tempfile.TemporaryDirectory() as d:
                e = engine_build(Path(d) / "e", **kw.get("e", {}))
                v = vision_build(Path(d) / "v", **kw.get("v", {}))
                with self.assertRaises(SystemExit):
                    quiet(make_release.pack, e, v, Path(d) / "out", "windows", LINK, None)
        with self.assertRaises(SystemExit):              # a LINK_VERSION of another engine version
            quiet(make_release.pack, engine_build(self.tmp / "e"), vision_build(self.tmp / "v"), self.out, "windows",
                  "0.1.40-link.1", None)
        self.assertFalse(self.out.exists() and any(self.out.glob("*.zip")))

    def test_ptx_and_real_code(self):
        self.assertEqual(make_release.cuda_archs("75-real;86-real;89-real;120", ROOT), ([75, 86, 89, 120], True))
        self.assertEqual(make_release.cuda_archs("86-real;89-real", ROOT), ([86, 89], False))
        self.assertEqual(make_release.cuda_archs("86;120a-real;120-virtual", ROOT), ([86, 120], True))


if __name__ == "__main__":
    unittest.main()
