#!/bin/bash
# tools/dsv41/remote_rows_compare.sh - strata-dsv41's remote stage with fp32 rows against bf16 rows (STRATA_REMOTE_ROWS=bf16:
# the streams as bf16, pre and the indexer's int32 picks exact), on one Linux PC and one GPU over the loopback, and both
# against one process: the same prompt's logits and 64 greedy tokens through each, then the top-1 agreement and the KL
# of each split's distribution from the single process's, the KL of bf16's from fp32's and of a second fp32 run's from the
# first's (the noise floor: what two runs of the same rows differ by), the prompt and decode tok/s of all, and the
# timing lines (STRATA_REMOTE_TIMING=1) of both sides.
#
#   tools/dsv41/remote_rows_compare.sh <strata-dsv41> <pack> <out dir> <ids file> [K]
#     ids file: the prompt's token ids, comma separated (the IDS file of remote_loopback.sh); K: the worker's first layer (20)
#   env: GPU (0); PORT (7841); CTX (8192); VRAM_EXPERTS (0.3); CPU_THREADS (half the CPUs each); ENGRAM_MB (64); PYTHON (python3)
#        (as remote_loopback.sh).  ARMS (default "fp32 fp32b bf16"): the split arms to run, in order; fp32b is a second
#        fp32 arm with its own output names.
#
# Each arm starts its own worker (STRATA_REMOTE_ROWS goes to the worker AND the main process: the hello refuses a pair that
# differs) and ends it; the single process runs last with the same flags.  What it proves, and what it does not:
#   - proves: the compact rows reach the head and give the same tokens (or not) at a KL the numbers show, against the
#     noise floor of two identical runs; the arms' tok/s on this PC (the loopback has no network: the rows' cost here is
#     the pack and unpack, not the wire)
#   - does not prove: the speed over 10 GbE (the wire time is in the notes of the unit's report: 92192 bytes a token in
#     fp32, 51232 in bf16)
# The arms place their experts by the VRAM each finds free at its load (the "in VRAM" line each prints), so a split's
# logits differ from the single process's by placement too (remote_loopback.sh says why).  Every arm's worker and main
# process run --prefill-chunk 0 (an fp32 worker would allocate chunk buffers a bf16 one does not, and place fewer
# experts); VRAM_EXPERTS=0 leaves none in VRAM on any arm.  The script FAILS (exit 1) when the split arms' expert counts
# in VRAM ("(N of M in VRAM)", main and worker) differ - the KL between them would then be placement, not rows - and
# when the logits dumps differ in length.  KL(split-fp32 || split-fp32b) is what two runs of the same rows differ by;
# the bf16 line is read against it.
set -u
if [ $# -lt 4 ]; then
    sed -n '2,28p' "$0"
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
arms=${ARMS:-"fp32 fp32b bf16"}
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

# an arm's STRATA_REMOTE_ROWS: fp32b is a second fp32 run
rows_of() {
    case $1 in
        fp32b) echo fp32 ;;
        *) echo "$1" ;;
    esac
}

for arm in $arms; do
    rows=$(rows_of "$arm")
    echo "== split, $arm rows ($rows)"
    : > "$out/worker-$arm.err"   # (made first: the wait below never reads a file the background shell has not opened)
    STRATA_REMOTE_ROWS=$rows "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk 0 --stage-worker "$port" \
        --stage-bind 127.0.0.1 --stage-begin "$K" > "$out/worker-$arm.out" 2> "$out/worker-$arm.err" &
    wpid=$!
    until grep -q 'remote stage worker:' "$out/worker-$arm.err"; do
        if ! kill -0 $wpid 2> /dev/null; then
            echo "the $arm worker stopped while loading:"
            tail -5 "$out/worker-$arm.err"
            exit 1
        fi
        sleep 2
    done
    # (the prompt in windows on every arm, as the single process: fp32 rows could cross as chunks, bf16 ones cannot)
    STRATA_REMOTE_ROWS=$rows DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --remote-stage "127.0.0.1:$port" --stage-end "$K" \
        --max-context "$ctx" --prefill-chunk 0 --dspark none --n 64 --ids "$ids" --dump-logits "$out/split-$arm.f32" > "$out/split-$arm.out" 2> "$out/split-$arm.err"
    echo "  rc=$?"
    kill $wpid 2> /dev/null
    wait $wpid 2> /dev/null
    wpid=""
done

echo "== one process"
DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk 0 --dspark none --n 64 --ids "$ids" \
    --dump-logits "$out/single.f32" > "$out/single.out" 2> "$out/single.err"
echo "  rc=$?"

# an arm's placement: the "in VRAM" line of its main process and of its worker, from "experts" on
placement() {
    echo "main: $(grep -h 'in VRAM' "$out/split-$1.err" | head -1 | sed 's/.*experts/experts/')"
    echo "worker: $(grep -h 'in VRAM' "$out/worker-$1.err" | head -1 | sed 's/.*experts/experts/')"
}

# an arm's expert counts in VRAM, main and worker: what the arms must agree on (the lines' free MiB may differ by the
# rows' buffers)
counts() {
    echo "main $(grep -h -o '([0-9]* of [0-9]* in VRAM)' "$out/split-$1.err" | head -1)," \
        "worker $(grep -h -o '([0-9]* of [0-9]* in VRAM)' "$out/worker-$1.err" | head -1)"
}

echo "== placement and speed"
first=""
for a in $arms; do
    p=$(placement "$a")
    echo "  split-$a:"
    echo "$p" | sed 's/^/      /'
    if [ -z "$first" ]; then
        first=$a
        first_p=$(counts "$a")
    elif [ "$(counts "$a")" != "$first_p" ]; then
        echo "FAIL: the experts in VRAM of split-$a ($(counts "$a")) differ from split-$first's ($first_p): the KL between them would be placement, not rows"
        rc=1
    fi
done
for b in $(for a in $arms; do echo "split-$a"; done) single; do
    echo "  $b:"
    [ "$b" = single ] && echo "      $(grep -h 'in VRAM' "$out/single.err" | head -1 | sed 's/.*experts/experts/')"
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
    raw = np.fromfile(f, dtype=np.float32)
    if raw.size % V:
        print("FAIL: %s is %d floats, not a whole number of %d-logit rows" % (f, raw.size, V))
        sys.exit(1)
    return raw.reshape(-1, V).astype(np.float64)
def sm(x):
    p = np.exp(x - x.max()); return p / p.sum()
def cmp(name, ref, got):
    kl = []; top = 0
    for x, y in zip(ref, got):
        px, py = sm(x), sm(y)
        kl.append(float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30)))))
        top += int(x.argmax() == y.argmax())
    print("  %-30s %d positions: top-1 agrees at %d, KL mean %.3e max %.3e" % (name, len(ref), top, np.mean(kl), np.max(kl)))
single = load(out + "/single.f32")
logits = {a: load("%s/split-%s.f32" % (out, a)) for a in arms}
lens = {"single": len(single)}
lens.update({"split-" + a: len(l) for a, l in logits.items()})
if len(set(lens.values())) != 1:
    print("FAIL: the logits dumps differ in length (positions: %s)" % ", ".join("%s %d" % kv for kv in lens.items()))
    sys.exit(1)
for a in arms:
    cmp("KL(single || split-%s)" % a, single, logits[a])
if "fp32" in logits and "fp32b" in logits:
    cmp("KL(split-fp32 || split-fp32b)", logits["fp32"], logits["fp32b"])
if "fp32" in logits and "bf16" in logits:
    cmp("KL(split-fp32 || split-bf16)", logits["fp32"], logits["bf16"])
PYEOF
exit $rc
