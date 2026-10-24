#!/bin/bash
# tools/dsv41/remote_loopback.sh - strata-dsv41's remote stage on one Linux PC and one GPU: a worker
# (--stage-worker PORT --stage-begin K, layers K to the last) and its main process (--remote-stage 127.0.0.1:PORT
# --stage-end K, layers 0 to K - 1 and the head) over the loopback, then the smoke test's question through the main
# process (tools/dsv41/smoke.py: the reply must be words with "Paris" in it) and, with IDS, a prompt's logits through
# the split and through one process, compared.
#
#   tools/dsv41/remote_loopback.sh <strata-dsv41> <pack> <out dir> [K]
#     K: the worker's first layer (default 20)
#   env: GPU (default 0); PORT (7841); CTX (8192, the worker's and the comparison's context; the smoke run asks for
#        1024); VRAM_EXPERTS (0.3: each process's share of the VRAM free when it loads - the worker loads first, the
#        main process takes its share of what is left); CPU_THREADS (half the CPUs each); ENGRAM_MB (64, each
#        process's Engram row cache); IDS (a file of prompt ids, comma separated: also the logits check - 64 greedy
#        tokens after it through the split, then through one process with the same flags, compared); PYTHON (python3)
#
# Both processes read the experts in place (--mmap-experts on: the files' page cache, shared by both, no copies in
# RAM), with no SSD store, a fixed placement (--no-expert-cache --expert-profile none) and Engram row caches of
# ENGRAM_MB: what one PC's RAM holds twice over is the dense weights and the KV caches.  The worker listens on
# 127.0.0.1 only and the link carries a token made here (STRATA_STAGE_TOKEN, unless one is set).
#
# The split's logits are not expected to be the same bits as one process's: each process places its own experts
# (which ones VRAM holds follows the VRAM free at its load), and a window's sum of experts in another place is another
# order of additions.  The check prints the top-1 agreement and the KL of the split's distribution from the single
# process's, and whether the greedy tokens agree.
set -u
if [ $# -lt 3 ]; then
    sed -n '2,24p' "$0"
    exit 2
fi
exe=$1
pack=$2
out=$3
K=${4:-20}
here=$(cd "$(dirname "$0")" && pwd)
gpu=${GPU:-0}
port=${PORT:-7841}
ctx=${CTX:-8192}
vram=${VRAM_EXPERTS:-0.3}
thr=${CPU_THREADS:-$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))}
eg_mb=${ENGRAM_MB:-64}
py=${PYTHON:-python3}
mkdir -p "$out"
export STRATA_STAGE_TOKEN=${STRATA_STAGE_TOKEN:-loopback-$$-$RANDOM$RANDOM}
export CUDA_VISIBLE_DEVICES=$gpu
export DS4_ENGRAM_CACHE_MB=$eg_mb
common=(--pack "$pack" --vram-experts "$vram" --mmap-experts on --ssd-cache 0 --no-expert-cache --expert-profile none
        --cpu-threads "$thr" --vision off --verbose)

# the worker, in the background until this script ends
"$exe" "${common[@]}" --max-context "$ctx" --stage-worker "$port" --stage-bind 127.0.0.1 --stage-begin "$K" \
    > "$out/worker.out" 2> "$out/worker.err" &
wpid=$!
trap 'kill $wpid 2> /dev/null; wait $wpid 2> /dev/null' EXIT
echo "worker pid $wpid: layers $K to the last on GPU $gpu, port $port (log $out/worker.err)"
until grep -q 'remote stage worker:' "$out/worker.err"; do   # (the line comes once the worker has loaded)
    if ! kill -0 $wpid 2> /dev/null; then
        echo "the worker stopped while loading:"
        tail -5 "$out/worker.err"
        exit 1
    fi
    sleep 2
done
echo "worker ready: $(grep -h 'in VRAM' "$out/worker.err" | head -1)"

# the main process as smoke.py runs an engine: its own arguments, then the remote stage's
main="$out/main.sh"
{
    echo '#!/bin/bash'
    printf 'exec %q "$@"' "$exe"
    printf ' %q' "${common[@]}" --remote-stage "127.0.0.1:$port" --stage-end "$K"
    echo
} > "$main"
chmod +x "$main"

rc=0
echo "== smoke (tools/dsv41/smoke.py through the main process; its log: $out/smoke.txt)"
"$py" "$here/smoke.py" --exe "$main" --pack "$pack" > "$out/smoke.txt" 2>&1
s=$?
cat "$out/smoke.txt"
echo "smoke rc=$s"
[ $s -eq 0 ] || rc=1

if [ -n "${IDS:-}" ]; then
    ids=$(cat "$IDS")
    echo "== logits: the split, then one process"
    "$main" --max-context "$ctx" --dspark none --n 64 --ids "$ids" --dump-logits "$out/split.f32" \
        > "$out/split.out" 2> "$out/split.err"
    echo "split rc=$?"
    kill $wpid 2> /dev/null
    wait $wpid 2> /dev/null
    DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk 0 --dspark none --n 64 --ids "$ids" \
        --dump-logits "$out/single.f32" > "$out/single.out" 2> "$out/single.err"
    echo "single rc=$?"
    for b in split single; do
        echo "  $b: $(grep -h 'in VRAM' "$out/$b.err" | head -1 | sed 's/.*experts/experts/')"
    done
    echo "  logits: $(cmp -s "$out/split.f32" "$out/single.f32" && echo IDENTICAL || echo DIFFER);" \
        "tokens: $(cmp -s <(grep -h '^[0-9]' "$out/split.out") <(grep -h '^[0-9]' "$out/single.out") && echo IDENTICAL || echo DIFFER)"
    "$py" -c '
import sys, numpy as np
a, b = (np.fromfile(f, dtype=np.float32).reshape(-1, 129280).astype(np.float64) for f in sys.argv[1:3])
n = min(len(a), len(b))
kl, top = [], 0
for x, y in zip(b[:n], a[:n]):
    px = np.exp(x - x.max()); px /= px.sum()
    py = np.exp(y - y.max()); py /= py.sum()
    kl.append(float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30)))))
    top += int(x.argmax() == y.argmax())
print("  %d positions: top-1 agrees at %d, KL(single || split) mean %.3e max %.3e" % (n, top, np.mean(kl), np.max(kl)))
' "$out/split.f32" "$out/single.f32" || rc=1
fi
exit $rc
