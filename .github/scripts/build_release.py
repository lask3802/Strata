#!/usr/bin/env python3
"""Configure and build one part of a ready-made engine zip: the engine (strata + strata_kq) or the image encoder
(tools/vision's strata-vision), portable (AVX2 baseline, the static MSVC runtime), for the CUDA archs given.

    python .github/scripts/build_release.py engine --build build-rel --cuda D:/cuda-13.0.1 --llama third_party/llama.cpp \\
        --archs "75-real;86-real;89-real;120" [-D NAME=VALUE ...]

It is setup.py's build_engine() with the release's options: the same CMake definitions (STRATA_GGML_DIR / LLAMA_DIR,
CMAKE_CUDA_COMPILER, CUDAToolkit_ROOT), STRATA_PORTABLE=ON instead of OFF, and a failed build tried once more (CUDA's
ptxas now and then fails to parse a PTX file it just wrote, issue #45).  On Windows the compiler's environment comes
from vcvars64.bat (vswhere's newest Visual Studio with the C++ tools, or STRATA_VCVARS).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIN = os.name == "nt"


def msvc_env() -> dict:
    """The environment vcvars64.bat makes, read back from `set`."""
    vcvars = os.environ.get("STRATA_VCVARS")
    if not vcvars:
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                       "Microsoft Visual Studio", "Installer", "vswhere.exe")
        where = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                               capture_output=True, text=True, check=True).stdout.strip()
        vcvars = str(Path(where, "VC", "Auxiliary", "Build", "vcvars64.bat"))
    print(f"MSVC: {vcvars}", flush=True)
    out = subprocess.run(f'cmd /d /c ""{vcvars}" >nul && set"', capture_output=True, text=True, shell=False, check=True)
    env = {}
    for line in out.stdout.splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            env[k] = v
    return env


def run(cmd: list, env: dict) -> int:
    print("+ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("engine", "vision"))
    ap.add_argument("--build", required=True, type=Path, help="the build folder")
    ap.add_argument("--cuda", required=True, type=Path, help="the CUDA toolkit's root (bin/nvcc in it)")
    ap.add_argument("--llama", required=True, type=Path, help="llama.cpp at the pinned commit (setup.get_llama_cpp())")
    ap.add_argument("--archs", required=True, help="CMAKE_CUDA_ARCHITECTURES")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--configure-only", action="store_true", help="stop after CMake's configure step")
    ap.add_argument("-D", dest="defs", action="append", default=[], metavar="NAME=VALUE", help="another definition")
    a = ap.parse_args()
    nvcc = a.cuda / "bin" / ("nvcc.exe" if WIN else "nvcc")
    if not nvcc.is_file():
        sys.exit(f"no {nvcc}")
    common = ["-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_PORTABLE=ON", f"-DCMAKE_CUDA_ARCHITECTURES={a.archs}",
              f"-DCMAKE_CUDA_COMPILER={nvcc.as_posix()}", f"-DCUDAToolkit_ROOT={a.cuda.as_posix()}"]
    if a.part == "engine":
        src, target = ROOT, "strata"
        defs = ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DSTRATA_GGML_DIR={a.llama.resolve().as_posix()}"]
    else:
        src, target = ROOT / "tools" / "vision", "strata-vision"
        defs = ["-DSTRATA_VISION_CUDA=ON", f"-DLLAMA_DIR={a.llama.resolve().as_posix()}"]
    # this PATH's cmake and ninja (the pinned pip ones), not the ones vcvars puts first (Visual Studio's own)
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if not cmake or not ninja:
        sys.exit("cmake and ninja are needed on PATH (pip install cmake ninja)")
    env = msvc_env() if WIN else dict(os.environ)
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", src, "-B", a.build, *common, *defs,
            *(f"-D{d}" for d in a.defs)]
    if run(conf, env) != 0:
        return 1
    if a.configure_only:
        return 0
    build = [cmake, "--build", a.build, "--target", target, "-j", str(a.jobs)]
    if run(build, env) != 0:
        print("(the build stopped - trying it once more)", flush=True)
        if run(build, env) != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
