#!/usr/bin/env python3
"""Pack built engines as the ready-made zips setup.py downloads, with their BUILD.json and the release's SHA256.txt.

    python tools/make_release.py --engine build-rel --vision build-rel-vision --out dist [--platform windows|linux]
    python tools/make_release.py --sums dist         (SHA256.txt again, for every strata-*.zip in dist)

--engine is a CUDA build folder of the engine (cmake --build <it> --target strata, configured with
-DSTRATA_ENABLE_CUDA=ON -DSTRATA_PORTABLE=ON), --vision one of tools/vision (-DSTRATA_VISION_CUDA=ON, portable is its
default).  .github/workflows/release.yml runs it on the builds' files: each folder needs the programs, CMakeCache.txt and
CMakeFiles/<cmake version>/CMakeCUDACompiler.cmake, nothing else.  The zip (the same layout as upstream's releases):

    strata(.exe), strata_kq.dll / libstrata_kq.so, strata-vision(.exe), BUILD.json

BUILD.json is what setup.py reads: "version" (CMakeLists.txt's, the one every MIN_ENGINE comparison reads), "link"
(LINK_VERSION, Strata Link's release: an installed engine older than the checkout's is replaced), "source": "release",
"archs" (the real code, CMAKE_CUDA_ARCHITECTURES) and "ptx" (PTX too: newer cards JIT it), "cuda" (the toolkit, major.minor),
"vision": "gpu" with "vision_archs", "portable", and for a CUDA 12 build "toolkit": 12 and "experimental".  The asset's
name follows from the platform and the CUDA major version (setup.py's PREBUILT_ASSET / CUDA12_ASSET).

SHA256.txt has a "<sha256> *<asset>" line per zip in the output folder, sorted by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = "SHA256.txt"


def asset_name(platform: str, cuda_major: int) -> str:
    """setup.py's PREBUILT_ASSET (CUDA 13) / CUDA12_ASSET (CUDA 12) for this platform."""
    return f"strata-{platform}-x64" + ("-cuda12" if cuda_major == 12 else "") + ".zip"


def cache_values(build: Path) -> dict:
    """CMakeCache.txt's NAME:TYPE=value lines, by name."""
    out = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"([A-Za-z_][\w.\-]*):[A-Z]+=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def cuda_version(build: Path) -> str:
    """The nvcc version a build folder was configured with, as major.minor (CMakeFiles/<v>/CMakeCUDACompiler.cmake)."""
    for f in sorted(build.glob("CMakeFiles/*/CMakeCUDACompiler.cmake")):
        m = re.search(r'set\(CMAKE_CUDA_COMPILER_VERSION "(\d+)\.(\d+)', f.read_text(encoding="utf-8"))
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    sys.exit(f"{build}: no CMakeFiles/*/CMakeCUDACompiler.cmake - not a CUDA build folder")


def cuda_archs(value: str, where: Path) -> tuple[list[int], bool]:
    """CMAKE_CUDA_ARCHITECTURES -> (the archs with real code, whether PTX is in it).  "120" is both, "120-real" the
    code alone, "120-virtual" the PTX alone; "native" / "all" name no list a user's GPU can be checked against."""
    archs, ptx = set(), False
    for entry in [x for x in value.split(";") if x]:
        m = re.fullmatch(r"(\d+)[af]?(-real|-virtual)?", entry)
        if not m:
            sys.exit(f"{where}: CMAKE_CUDA_ARCHITECTURES={value}: a release names its archs ('{entry}' is not one)")
        if m.group(2) != "-virtual":
            archs.add(int(m.group(1)))
        if m.group(2) != "-real":
            ptx = True
    if not archs:
        sys.exit(f"{where}: CMAKE_CUDA_ARCHITECTURES={value} has no real code")
    return sorted(archs), ptx


def program(folder: Path, name: str) -> Path:
    """A build's program: in the folder itself (the engine) or in its bin/ (tools/vision's RUNTIME_OUTPUT_DIRECTORY)."""
    for p in (folder / name, folder / "bin" / name):
        if p.is_file():
            return p
    sys.exit(f"{folder}: no {name} (in it or in its bin/)")


def link_version() -> str:
    return (ROOT / "LINK_VERSION").read_text(encoding="utf-8").strip()


def git_commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def write_sums(out: Path) -> Path:
    """SHA256.txt for every strata-*.zip in `out`, in the format of upstream's releases (sha256sum -b)."""
    zips = sorted(out.glob("strata-*.zip"), key=lambda p: p.name)
    if not zips:
        sys.exit(f"{out}: no strata-*.zip to list")
    sums = out / SUMS
    sums.write_bytes("".join(f"{sha256(z)} *{z.name}\n" for z in zips).encode("ascii"))
    return sums


def pack(engine: Path, vision: Path, out: Path, platform: str, link: str, commit: str | None) -> Path:
    win = platform == "windows"
    exe, vexe = ("strata.exe", "strata-vision.exe") if win else ("strata", "strata-vision")
    kq = "strata_kq.dll" if win else "libstrata_kq.so"
    ec, vc = cache_values(engine), cache_values(vision)
    for build, cache, need in ((engine, ec, {"STRATA_ENABLE_CUDA": "ON", "STRATA_PORTABLE": "ON",
                                             "CMAKE_BUILD_TYPE": "Release"}),
                               (vision, vc, {"STRATA_VISION_CUDA": "ON", "STRATA_PORTABLE": "ON",
                                             "CMAKE_BUILD_TYPE": "Release"})):
        for k, v in need.items():
            if cache.get(k, "").upper() != v.upper():
                sys.exit(f"{build}: {k}={cache.get(k)!r}, a release needs {v}")
    version = ec.get("CMAKE_PROJECT_VERSION") or ec.get("strata_VERSION") or ""
    if not re.fullmatch(r"\d+(\.\d+)+", version):
        sys.exit(f"{engine}: no project version in CMakeCache.txt (CMAKE_PROJECT_VERSION={version!r})")
    m = re.fullmatch(r"(\d+(?:\.\d+)+)-link\.(\d+)", link)
    if not m or m.group(1) != version:
        sys.exit(f"LINK_VERSION is {link!r}: it must be <the engine's version>-link.<n>, the engine is {version}")
    cuda, vcuda = cuda_version(engine), cuda_version(vision)
    if cuda != vcuda:
        sys.exit(f"the engine is built with CUDA {cuda}, the image encoder with {vcuda}: one toolkit per zip")
    archs, ptx = cuda_archs(ec.get("CMAKE_CUDA_ARCHITECTURES", ""), engine)
    varchs, _ = cuda_archs(vc.get("CMAKE_CUDA_ARCHITECTURES", ""), vision)
    files = [program(engine, exe)]
    if ec.get("STRATA_MMQ_KQUANTS", "ON").upper() != "OFF":   # on by default: the K-quant prompt kernels' library
        if not (engine / kq).is_file():           # a zip without it would move the user's aside (install_unpacked)
            sys.exit(f"{engine}: no {kq}, and STRATA_MMQ_KQUANTS is on")
        files.append(engine / kq)
    files.append(program(vision, vexe))
    said = re.search(rb"engine=(\d+\.\d+\.\d+(?:\.\d+)?)\n", files[0].read_bytes())
    if not said or said.group(1).decode() != version:   # setup.py's stale_binary() reads the same line
        sys.exit(f"{files[0]} says engine={said.group(1).decode() if said else '?'}, the build folder is {version}: "
                 "a stale binary")
    major = int(cuda.split(".")[0])
    meta = {"version": version, "link": link, "source": "release", "archs": archs, "vision_archs": varchs,
            "ptx": ptx, "cuda": cuda, **({"toolkit": 12, "experimental": True} if major == 12 else {}),
            "vision": "gpu", "portable": True, **({"commit": commit} if commit else {})}
    out.mkdir(parents=True, exist_ok=True)
    asset = out / asset_name(platform, major)
    tmp = asset.with_name(asset.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            info = zipfile.ZipInfo.from_file(f, f.name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if not win:                          # unzip keeps the mode; setup.py chmods the two programs itself
                info.create_system = 3
                info.external_attr = (0o100644 if f.name == kq else 0o100755) << 16
            with open(f, "rb") as src, z.open(info, "w") as dst:
                while True:
                    b = src.read(16 << 20)
                    if not b:
                        break
                    dst.write(b)
        z.writestr("BUILD.json", json.dumps(meta))
    tmp.replace(asset)
    print(f"{asset.name}: {', '.join(f.name for f in files)}, BUILD.json {json.dumps(meta)}")
    return asset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", type=Path, help="the engine's CUDA build folder")
    ap.add_argument("--vision", type=Path, help="tools/vision's CUDA build folder")
    ap.add_argument("--out", type=Path, help="the folder the zip and SHA256.txt go to")
    ap.add_argument("--platform", choices=("windows", "linux"), default="windows" if os.name == "nt" else "linux",
                    help="whose programs these are (default: this machine's)")
    ap.add_argument("--link", help="Strata Link's release (default: the LINK_VERSION file)")
    ap.add_argument("--sums", type=Path, metavar="DIR", help="only write DIR/SHA256.txt for the zips in DIR")
    a = ap.parse_args()
    if a.sums:
        print(write_sums(a.sums).read_text(encoding="ascii"), end="")
        return 0
    if not (a.engine and a.vision and a.out):
        ap.error("--engine, --vision and --out are needed (or --sums DIR)")
    pack(a.engine, a.vision, a.out, a.platform, a.link or link_version(), git_commit())
    print(write_sums(a.out).read_text(encoding="ascii"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
