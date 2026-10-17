"""The DLLs each Windows program imports, and a failure for one outside ALLOWED: the CUDA libraries setup.py installs
from pip (CUDA_WHEELS / CUDA12_WHEELS: cuBLAS and the CUDA runtime), the NVIDIA driver's nvcuda.dll, and Windows' own.
A new import (a shared MSVC runtime, another CUDA library) would make the zip fail to start on a user's PC.

    python .github/scripts/check_imports.py strata.exe strata_kq.dll strata-vision.exe    (needs pefile)
"""
import sys
from pathlib import Path

import pefile

ALLOWED = {
    # pip's nvidia-cublas / nvidia-cuda-runtime (13), nvidia-cublas-cu12 / nvidia-cuda-runtime-cu12 (12)
    "cublas64_13.dll", "cublaslt64_13.dll", "cudart64_13.dll",
    "cublas64_12.dll", "cublaslt64_12.dll", "cudart64_12.dll",
    "nvcuda.dll",                                      # the driver's
    # Windows'
    "kernel32.dll", "advapi32.dll", "ws2_32.dll", "user32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll",
    "bcrypt.dll", "ntdll.dll", "shlwapi.dll", "dbghelp.dll", "psapi.dll", "iphlpapi.dll", "winmm.dll", "version.dll",
}


def main() -> int:
    bad = 0
    found = [name for name in sys.argv[1:] if Path(name).is_file()]   # a build part has some of them (engine, vision)
    if not found:
        print(f"::error::none of {sys.argv[1:]} was built: nothing to check")
        return 1
    for name in found:
        pe = pefile.PE(name, fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        dlls = [e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])]
        print(name, dlls)
        for d in dlls:
            if d.lower() not in ALLOWED:
                print(f"::error::{name} imports {d}, which a user's PC may not have (not in check_imports.py's ALLOWED)")
                bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main())
