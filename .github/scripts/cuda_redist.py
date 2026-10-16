#!/usr/bin/env python3
"""The parts of a CUDA toolkit a build needs, from NVIDIA's redistributable archives, into one folder.

    python .github/scripts/cuda_redist.py --version 13.0.1 --dest D:/cuda-13.0.1 \\
        cuda_nvcc cuda_cudart libcublas cuda_crt libnvvm cuda_cccl

https://developer.download.nvidia.com/compute/cuda/redist/redistrib_<version>.json names each component's archive and
its SHA-256; every archive is checked against it before it is unpacked, and each one's top folder is dropped, so the
folder ends up laid out as an installed toolkit (bin, include, lib/x64, nvvm): nvcc's own nvcc.profile finds the rest
from there, and CMake takes it as -DCUDAToolkit_ROOT.  The release workflow uses it on Windows; Linux takes NVIDIA's
apt packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"


def fetch(url: str, dst: Path) -> None:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f, 16 << 20)
            return
        except OSError as e:
            if attempt == 3:
                raise
            print(f"  {url}: {e}, again", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def unpack(archive: Path, dest: Path) -> None:
    """The zip's files without its top folder (<component>-<platform>-<version>-archive/)."""
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            rel = m.filename.split("/", 1)[1] if "/" in m.filename else ""
            if not rel or m.is_dir():
                continue
            out = (dest / rel).resolve()
            if dest.resolve() not in out.parents:
                sys.exit(f"{archive.name}: {m.filename} is outside the archive's folder")
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="the toolkit release, e.g. 13.0.1 (redistrib_<it>.json)")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--platform", default="windows-x86_64", help="the manifest's platform (zip archives)")
    ap.add_argument("components", nargs="+")
    a = ap.parse_args()
    with urllib.request.urlopen(f"{BASE}redistrib_{a.version}.json", timeout=120) as r:
        manifest = json.loads(r.read())
    a.dest.mkdir(parents=True, exist_ok=True)
    dl = a.dest / "_download"
    dl.mkdir(exist_ok=True)
    for name in a.components:
        entry = (manifest.get(name) or {}).get(a.platform)
        if not entry:
            sys.exit(f"{name}: not in redistrib_{a.version}.json for {a.platform}")
        archive = dl / Path(entry["relative_path"]).name
        print(f"{name} {manifest[name]['version']}: {entry['relative_path']}", flush=True)
        fetch(BASE + entry["relative_path"], archive)
        got = sha256(archive)
        if got != entry["sha256"]:
            sys.exit(f"{archive.name}: SHA-256 {got}, NVIDIA's manifest says {entry['sha256']}")
        unpack(archive, a.dest)
        archive.unlink()
    shutil.rmtree(dl)
    (a.dest / "redist.json").write_text(json.dumps({"version": a.version, "components": {
        n: manifest[n]["version"] for n in a.components}}, indent=1), encoding="utf-8")
    nvcc = a.dest / "bin" / ("nvcc.exe" if a.platform.startswith("windows") else "nvcc")
    if not nvcc.is_file():
        sys.exit(f"no {nvcc} after unpacking")
    print(f"CUDA {a.version} in {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
