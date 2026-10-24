#!/bin/bash
# tools/dsv41/remote_rows_compare.sh - strata-dsv41's remote stage with fp32 rows against bf16 rows (STRATA_REMOTE_ROWS=bf16:
# the streams and pre as bf16, the indexer's int32 picks exact), on one Linux PC and one GPU over the loopback, and both
# against one process: the same prompt's logits and 64 greedy tokens through each, then the top-1 agreement and the KL
# of each split's distribution from the single process's (and bf16's from fp32's: what the rows alone cost), the prompt
# and decode tok/s of the three, and the timing lines (STRATA_REMOTE_TIMING=1) of both sides.
#
#   tools/dsv41/remote_rows_compare.sh <strata-dsv41> <pack> <out dir> <ids file> [K]
#     ids file: the prompt's token ids, comma separated (the IDS file of remote_loopback.sh); K: the worker's first layer (20)
#   env: GPU (0); PORT (7841); CTX (8192); VRAM_EXPERTS (0.3); CPU_THREADS (half the CPUs each); ENGRAM_MB (64); PYTHON (python3)
#        (as remote_loopback.sh).  ARMS (default "fp32 bf16"): the split arms to run, in order.
#
# Each arm starts its own worker (STRATA_REMOTE_ROWS goes to the worker AND the main process: the hello refuses a pair that
# differs) and ends it; the single process runs last with the same flags.  What it proves, and what it does not:
#   - proves: the compact rows reach the head and give the same tokens (or not) at a KL the numbers show; the three arms'
#     tok/s on this PC (the loopback has no network: the rows' cost here is the pack and unpack, not the wire)
#   - does not prove: the speed over 10 GbE (the wire time is in the notes of the unit's report: 92192 bytes a token in
#     fp32, 51216 in bf16)
# The arms place their experts by the VRAM each finds free at its load (the "in VRAM" line each prints), so a split's
# logits differ from the single process's by placement too (remote_loopback.sh says why): the bf16-against-fp32 line,
# both split the same way, is the one that isolates the rows.
set -u
if [ $# -lt 4 ]; then
    sed -n '2,22p' "$0"
    exit 2
fi
exe=$1
pack=$2
out=$3
ids_file=$4
K=${5:-20}
gpu=${GPU:-0}
port=${PORT:-7841}
ctx=${CTX:-8192}
vram=${VRAM_EXPERTS:-0.3}
thr=${CPU_THREADS:-$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))}
eg_mb=${ENGRAM_MB:-64}
py=${PYTHON:-python3}
arms=${ARMS:-"fp32 bf16"}
mkdir -p "$out"
ids=$(cat "$ids_file")
export STRATA_STAGE_TOKEN=${STRATA_STAGE_TOKEN:-loopback-$$-$RANDOM$RANDOM}
export CUDA_VISIBLE_DEVICES=$gpu
export DS4_ENGRAM_CACHE_MB=$eg_mb
export STRATA_REMOTE_TIMING=1
common=(--pack "$pack" --vram-experts "$vram" --mmap-experts on --ssd-cache 0 --no-expert-cache --expert-profile none
        --cpu-threads "$thr" --vision off --verbose)
rc=0
wpid=""
trap '[ -n "$wpid" ] && kill $wpid 2> /dev/null; wait 2> /dev/null' EXIT

for arm in $arms; do
    echo "== split, $arm rows"
    STRATA_REMOTE_ROWS=$arm "$exe" "${common[@]}" --max-context "$ctx" --stage-worker "$port" --stage-bind 127.0.0.1 \
        --stage-begin "$K" > "$out/worker-$arm.out" 2> "$out/worker-$arm.err" &
    wpid=$!
    until grep -q 'remote stage worker:' "$out/worker-$arm.err"; do
        if ! kill -0 $wpid 2> /dev/null; then
            echo "the $arm worker stopped while loading:"
            tail -5 "$out/worker-$arm.err"
            exit 1
        fi
        sleep 2
    done
    STRATA_REMOTE_ROWS=$arm "$exe" "${common[@]}" --remote-stage "127.0.0.1:$port" --stage-end "$K" --max-context "$ctx" \
        --dspark none --n 64 --ids "$ids" --dump-logits "$out/split-$arm.f32" > "$out/split-$arm.out" 2> "$out/split-$arm.err"
    echo "  rc=$?"
    kill $wpid 2> /dev/null
    wait $wpid 2> /dev/null
    wpid=""
done

echo "== one process"
DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk 0 --dspark none --n 64 --ids "$ids" \
    --dump-logits "$out/single.f32" > "$out/single.out" 2> "$out/single.err"
echo "  rc=$?"

echo "== placement and speed"
for b in $(for a in $arms; do echo "split-$a"; done) single; do
    echo "  $b: $(grep -h 'in VRAM' "$out/$b.err" | head -1 | sed 's/.*experts/experts/')"
    echo "      $(grep -h '^prompt:' "$out/$b.err" | tail -1)"
    echo "      $(grep -h '^decode:' "$out/$b.err" | tail -1)"
done
echo "== rows and timing lines"
for a in $arms; do
    echo "  $a: $(grep -h 'remote stage 127.0.0.1' "$out/split-$a.err" | head -1)"
    grep -h 'strata remote:\|strata stage worker: windows' "$out/split-$a.err" "$out/worker-$a.err" | tail -6 | sed 's/^/      /'
done
echo "== tokens and logits"
for a in $arms; do
    echo "  split-$a tokens vs single: $(cmp -s <(grep -h '^[0-9]' "$out/split-$a.out") <(grep -h '^[0-9]' "$out/single.out") && echo IDENTICAL || echo DIFFER)"
done
"$py" - "$out" $arms << 'PYEOF' || rc=1
import sys, numpy as np
out, arms = sys.argv[1], sys.argv[2:]
V = 129280
def load(f):
    return np.fromfile(f, dtype=np.float32).reshape(-1, V).astype(np.float64)
def sm(x):
    p = np.exp(x - x.max()); return p / p.sum()
def cmp(name, ref, got):
    n = min(len(ref), len(got)); kl = []; top = 0
    for x, y in zip(ref[:n], got[:n]):
        px, py = sm(x), sm(y)
        kl.append(float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30)))))
        top += int(x.argmax() == y.argmax())
    print("  %-26s %d positions: top-1 agrees at %d, KL mean %.3e max %.3e" % (name, n, top, np.mean(kl), np.max(kl)))
single = load(out + "/single.f32")
logits = {a: load("%s/split-%s.f32" % (out, a)) for a in arms}
for a in arms:
    cmp("KL(single || split-%s)" % a, single, logits[a])
if "fp32" in logits and "bf16" in logits:
    cmp("KL(split-fp32 || split-bf16)", logits["fp32"], logits["bf16"])
PYEOF
exit $rc
