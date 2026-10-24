#!/bin/bash
# tools/dsv41/remote_loopback.sh - strata-dsv41's remote stage on one Linux PC and one GPU: a worker
# (--stage-worker PORT --stage-begin K, layers K to the last) and its main process (--remote-stage 127.0.0.1:PORT
# --stage-end K, layers 0 to K - 1 and the head) over the loopback, then the smoke test's question through the main
# process (tools/dsv41/smoke.py: the reply must be words with "Paris" in it); with IDS, a prompt's logits through the
# split and through one process, compared; with SPEC_IDS, speculative decoding (--dspark on: the drafter on the main
# process, the draft checks through the worker) through the split and through one process, compared; with LONG (or
# LONG_N), a long prompt in chunks through the split and through one process, its every row's logits compared and
# both prompts' speeds.
#
#   tools/dsv41/remote_loopback.sh <strata-dsv41> <pack> <out dir> [K]
#     K: the worker's first layer (default 20)
#   env: GPU (default 0); PORT (7841); CTX (8192, the worker's and the comparison's context; the smoke run asks for
#        1024); VRAM_EXPERTS (0.3: each process's share of the VRAM free when it loads - the worker loads first, the
#        main process takes its share of what is left); CPU_THREADS (half the CPUs each); ENGRAM_MB (64, each
#        process's Engram row cache); CHUNK (1024: the prompt chunk's rows, --prefill-chunk, on both sides of the
#        split); IDS (a file of prompt ids, comma separated: also the logits check - 64 greedy tokens after it through
#        the split, then through one process in windows of 16, compared); KL_MAX (0.02) and TOP1_MIN (0.9): a logits
#        comparison fails (exit 1) when the mean KL is above KL_MAX or the top-1 tokens agree at fewer than TOP1_MIN of
#        the positions; SPEC_IDS (a file of prompt ids: SPEC_N (128) greedy tokens after it with --dspark on through
#        the split and through one process, each run's token ids, its drafts offered and accepted and its decode
#        tok/s; SPEC_CONF: --dspark-conf for both, 0 checks every draft - the most rejected rows to undo; SPEC_NONE=1:
#        the same SPEC_N tokens with --dspark none too, through each, for the drafts' speed against plain decoding);
#        LONG (a file of a long prompt's ids, comma separated - 2K to 4K tokens - or LONG_N=N: N made-up ids, a fixed
#        sequence: the chunks' check below); PYTHON (python3)
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
#
# The speculative check's token ids are expected IDENTICAL with VRAM_EXPERTS=0: then neither process puts an expert
# in VRAM (the drafter's included), both read every expert on the CPU lane with CPU_THREADS threads, the drafter runs
# on the main process in both, the prompt goes in windows of 16 in both (a main process with the drafter sends no
# prompt chunks: their replies carry no taps), and each window - a draft's check too - is the same rows on the same
# GPU kernels, the worker undoing a check's rejected rows as one process does.  With experts in VRAM the two
# placements differ (the worker loads first), so the drafts and the trunk's sums can differ by rounding; then a greedy
# run may part at a near tie, and the speculative check says where.
#
# The chunks' check (LONG): the prompt as one feed through the chunked prompt path on both sides, in chunks of CHUNK
# rows - the main process's layers on chunk c + 1 while the worker reads chunk c - and through one process with the
# same chunks (DS4_YOCO=0: no long prompt's skip, which a split cannot take; DS4_PATH_PICK=0: the chunks, never
# windows).  Twice each: with --dump-logits and --window <its length> (every row's logits, every row back over the
# link), compared as above, then plainly (the last row's logits only) for the prompt's tokens/s.  DS4_LAYER_MAJOR and
# the other engine variables pass through to both (DS4_LAYER_MAJOR=0: the main process chunk after chunk, overlapped
# with the worker; else it may take its layers one after another over all the chunks, which overlaps little).
# STRATA_REMOTE_ROWS=bf16 passes through as well, and turns the chunks off on both sides (windows of 16).
set -u
if [ $# -lt 3 ]; then
    sed -n '2,52p' "$0"
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
chunk=${CHUNK:-1024}
py=${PYTHON:-python3}
mkdir -p "$out"
export STRATA_STAGE_TOKEN=${STRATA_STAGE_TOKEN:-loopback-$$-$RANDOM$RANDOM}
export CUDA_VISIBLE_DEVICES=$gpu
export DS4_ENGRAM_CACHE_MB=$eg_mb
common=(--pack "$pack" --vram-experts "$vram" --mmap-experts on --ssd-cache 0 --no-expert-cache --expert-profile none
        --cpu-threads "$thr" --vision off --verbose)

# the long prompt's ids (LONG, or LONG_N made up: BOS, then a fixed walk over ordinary token ids)
long=
if [ -n "${LONG:-}" ]; then
    long=$LONG
elif [ -n "${LONG_N:-}" ]; then
    long="$out/long.ids"
    "$py" -c '
import sys
n = int(sys.argv[1])
print(",".join(str(x) for x in [0] + [1000 + (i * 7919) % 50000 for i in range(n - 1)]))
' "$LONG_N" > "$long" || exit 1
fi
nlong=0
if [ -n "$long" ]; then
    nlong=$(tr ',' '\n' < "$long" | grep -c '[0-9]')
    if [ "$nlong" -lt 2 ]; then
        echo "no long prompt's ids in $long"
        exit 2
    fi
fi

# the worker, in the background until this script ends (or the single-process runs need its GPU)
"$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk "$chunk" --stage-worker "$port" --stage-bind 127.0.0.1 \
    --stage-begin "$K" > "$out/worker.out" 2> "$out/worker.err" &
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
echo "  $(grep -h 'remote stage worker:' "$out/worker.err" | head -1)"

# the main process as smoke.py runs an engine: its own arguments, then the remote stage's
main="$out/main.sh"
{
    echo '#!/bin/bash'
    printf 'exec %q "$@"' "$exe"
    printf ' %q' "${common[@]}" --prefill-chunk "$chunk" --remote-stage "127.0.0.1:$port" --stage-end "$K"
    echo
} > "$main"
chmod +x "$main"

# the logits of two runs (files of 129280 floats a row): top-1 agreement and KL(the second || the first); fails (1)
# when the mean KL is above KL_MAX or the top-1 agreement below TOP1_MIN
compare() {
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
if n == 0:
    print("  FAIL: no logits to compare")
    sys.exit(1)
print("  %d positions: top-1 agrees at %d, KL(single || split) mean %.3e max %.3e" % (n, top, np.mean(kl), np.max(kl)))
kl_max, top_min = float(sys.argv[3]), float(sys.argv[4])
if np.mean(kl) > kl_max or top < top_min * n:
    print("  FAIL: the split is further from one process than KL_MAX %g / TOP1_MIN %g allow" % (kl_max, top_min))
    sys.exit(1)
' "$1" "$2" "${KL_MAX:-0.02}" "${TOP1_MIN:-0.9}"
}
# whether two runs' logits files and greedy tokens are the same
same() {
    echo "  logits: $(cmp -s "$out/$1.f32" "$out/$2.f32" && echo IDENTICAL || echo DIFFER);" \
        "tokens: $(cmp -s <(grep -h '^[0-9]' "$out/$1.out") <(grep -h '^[0-9]' "$out/$2.out") && echo IDENTICAL || echo DIFFER)"
}
# a run's prompt line ("prompt: N tokens in X ms (Y tok/s, ...)")
speed() { grep -h '^prompt: ' "$out/$1.err" | head -1; }

rc=0
echo "== smoke (tools/dsv41/smoke.py through the main process; its log: $out/smoke.txt)"
"$py" "$here/smoke.py" --exe "$main" --pack "$pack" > "$out/smoke.txt" 2>&1
s=$?
cat "$out/smoke.txt"
echo "smoke rc=$s"
[ $s -eq 0 ] || rc=1

# the split's runs while the worker is up
spec_n=${SPEC_N:-128}
spec=(--dspark on --n "$spec_n")
[ -n "${SPEC_CONF:-}" ] && spec+=(--dspark-conf "$SPEC_CONF")
if [ -n "${SPEC_IDS:-}" ]; then
    [ "$vram" = 0 ] || echo "(VRAM_EXPERTS=$vram: the two placements differ - token ids may part at a near tie)"
    echo "== speculative decoding through the split (${spec[*]})"
    "$main" --max-context "$ctx" "${spec[@]}" --ids "@$SPEC_IDS" > "$out/spec-split.out" 2> "$out/spec-split.err"
    ss=$?
    echo "spec split rc=$ss"
    if [ "${SPEC_NONE:-0}" = 1 ]; then
        "$main" --max-context "$ctx" --dspark none --n "$spec_n" --ids "@$SPEC_IDS" > "$out/plain-split.out" 2> "$out/plain-split.err"
        echo "plain split rc=$?"
    fi
fi
if [ -n "${IDS:-}" ]; then
    ids=$(cat "$IDS")
    echo "== logits: the split, then one process"
    "$main" --max-context "$ctx" --dspark none --n 64 --ids "$ids" --dump-logits "$out/split.f32" \
        > "$out/split.out" 2> "$out/split.err"
    s=$?; echo "split rc=$s"; [ $s -eq 0 ] || rc=1
fi
if [ -n "$long" ]; then
    echo "== a long prompt ($nlong tokens) in chunks of $chunk: the split, then one process"
    "$main" --max-context "$ctx" --dspark none --n 16 --ids "@$long" --window "$nlong" --dump-logits "$out/split-long.f32" \
        > "$out/split-long.out" 2> "$out/split-long.err"
    s=$?; echo "split (every row's logits) rc=$s"; [ $s -eq 0 ] || rc=1
    "$main" --max-context "$ctx" --dspark none --n 16 --ids "@$long" > "$out/split-speed.out" 2> "$out/split-speed.err"
    s=$?; echo "split (the prompt's speed) rc=$s"; [ $s -eq 0 ] || rc=1
    grep -h 'chunks of\|remote stage 127' "$out/split-speed.err" | head -2 | sed 's/^/  /'
fi
kill $wpid 2> /dev/null
wait $wpid 2> /dev/null

# one process, the whole trunk, on the same GPU
if [ -n "${SPEC_IDS:-}" ]; then
    echo "== speculative decoding through one process"
    DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk 0 "${spec[@]}" --ids "@$SPEC_IDS" \
        > "$out/spec-single.out" 2> "$out/spec-single.err"
    s1=$?
    echo "spec single rc=$s1"
    arms="spec-split spec-single"
    if [ "${SPEC_NONE:-0}" = 1 ]; then
        DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk 0 --dspark none --n "$spec_n" \
            --ids "@$SPEC_IDS" > "$out/plain-single.out" 2> "$out/plain-single.err"
        echo "plain single rc=$?"
        arms="$arms plain-split plain-single"
    fi
    for b in $arms; do
        echo "  $b: $(grep -h '^decode:' "$out/$b.err" | head -1 | cut -c1-60)"
        echo "    $(grep -h '^dspark:' "$out/$b.err" | head -1 | cut -c1-120)"
        echo "    $(grep -h '^ds4: layers \[' "$out/$b.err" | head -1 | cut -c1-160)"
    done
    a=$(grep -h '^[0-9]' "$out/spec-split.out" | head -1 | tr ',' '\n' | grep -c .)
    b=$(grep -h '^[0-9]' "$out/spec-single.out" | head -1 | tr ',' '\n' | grep -c .)
    if [ $ss -ne 0 ] || [ $s1 -ne 0 ]; then
        echo "  speculative token ids: not compared - a run failed (its .err: $out/spec-split.err, $out/spec-single.err)"
        rc=1
    elif [ "$a" -gt 0 ] && cmp -s <(grep -h '^[0-9]' "$out/spec-split.out") <(grep -h '^[0-9]' "$out/spec-single.out"); then
        echo "  speculative token ids: IDENTICAL ($a tokens)"
    else
        first=$(paste -d' ' <(grep -h '^[0-9]' "$out/spec-split.out" | head -1 | tr ',' '\n') \
                            <(grep -h '^[0-9]' "$out/spec-single.out" | head -1 | tr ',' '\n') |
                awk '$1 != $2 { print NR - 1; exit }')
        echo "  speculative token ids: DIFFER (split $a tokens, single $b; first different at token ${first:-?})"
        [ "$vram" = 0 ] && rc=1   # (experts in VRAM: the placements differ, a near tie may part them - said above)
    fi
fi
if [ -n "${IDS:-}" ]; then
    DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk 0 --dspark none --n 64 --ids "$ids" \
        --dump-logits "$out/single.f32" > "$out/single.out" 2> "$out/single.err"
    s=$?; echo "single rc=$s"; [ $s -eq 0 ] || rc=1
    for b in split single; do
        echo "  $b: $(grep -h 'in VRAM' "$out/$b.err" | head -1 | sed 's/.*experts/experts/')"
    done
    same split single
    compare "$out/split.f32" "$out/single.f32" || rc=1
fi
if [ -n "$long" ]; then
    DS4_YOCO=0 DS4_PATH_PICK=0 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk "$chunk" --dspark none --n 16 \
        --ids "@$long" --window "$nlong" --dump-logits "$out/single-long.f32" > "$out/single-long.out" 2> "$out/single-long.err"
    s=$?; echo "single (every row's logits) rc=$s"; [ $s -eq 0 ] || rc=1
    DS4_YOCO=0 DS4_PATH_PICK=0 "$exe" "${common[@]}" --max-context "$ctx" --prefill-chunk "$chunk" --dspark none --n 16 \
        --ids "@$long" > "$out/single-speed.out" 2> "$out/single-speed.err"
    s=$?; echo "single (the prompt's speed) rc=$s"; [ $s -eq 0 ] || rc=1
    same split-long single-long
    compare "$out/split-long.f32" "$out/single-long.f32" || rc=1
    echo "  tokens after the prompt (speed runs): $(cmp -s <(grep -h '^[0-9]' "$out/split-speed.out") \
        <(grep -h '^[0-9]' "$out/single-speed.out") && echo IDENTICAL || echo DIFFER)"
    echo "  split:  $(speed split-speed)"
    echo "  single: $(speed single-speed)"
fi
exit $rc
