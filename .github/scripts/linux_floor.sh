#!/bin/bash
# The Linux zip's programs on old distributions (the docker images given): strata --version and --families with the
# CUDA libraries of setup.py's CUDA_WHEELS, and ldd -r on each program (every library and symbol version resolves;
# strata-vision's libcuda.so.1 is the NVIDIA driver's, which a container has not).
#
#   .github/scripts/linux_floor.sh <unzipped strata-linux-x64.zip> <the wheels' nvidia/cu13/lib> <image>...
set -euo pipefail
dir=$1 lib=$2
shift 2
fail=0
for img in "$@"; do
  echo "=== $img"
  # shellcheck disable=SC2016
  if ! docker run --rm -v "$dir:/x:ro" -v "$lib:/cuda:ro" -e LD_LIBRARY_PATH=/cuda "$img" sh -c '
      grep -m1 PRETTY_NAME /etc/os-release; ldd --version | head -1
      /x/strata --version && /x/strata --families || exit 1
      bad=0
      for f in /x/strata /x/libstrata_kq.so /x/strata-vision; do
        miss=$(ldd -r "$f" 2>&1 | grep -E "not found|undefined symbol" |
               grep -vE "libcuda\.so\.1 => not found|undefined symbol: cu[A-Z]" || true)
        if [ -z "$miss" ]; then echo "$f: every library and symbol version resolves"; else echo "$f:"; echo "$miss"; bad=1; fi
      done
      exit $bad'; then
    echo "::error::the Linux zip does not run on $img"
    fail=1
  fi
done
exit $fail
