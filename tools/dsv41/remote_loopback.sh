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
#        the split, then through one process, compared: with CHUNK 0 in windows of 16 on both, else in chunks on both -
#        DS4_CHUNK_MIN=2 for the two runs, so every window of 16 the dump feeds goes as a chunk, the last row as a
#        window, on either side); KL_MAX (0.02) and TOP1_MIN (0.9): a logits
#        comparison fails (exit 1) when the mean KL is above KL_MAX or the top-1 tokens agree at fewer than TOP1_MIN of
#        the positions; SPEC_IDS (a file of prompt ids: SPEC_N (128) greedy tokens after it with --dspark on through
#        the split and through one process, each run's token ids, its drafts offered and accepted and its decode
#        tok/s; SPEC_CONF: --dspark-conf for both, 0 checks every draft - the most rejected rows to undo; SPEC_NONE=1:
#        the same SPEC_N tokens with --dspark none too, through each, for the drafts' speed against plain decoding);
#        LONG (a file of a long prompt's ids, comma separated - 2K to 4K tokens - or LONG_N=N: N made-up ids, a fixed
#        sequence: the chunks' check below); FLOOR_VRAM (with LONG: the noise floor - one process once more, every
#        row's logits, the same chunks, with --vram-experts FLOOR_VRAM, another placement; the split's rule is then
#        the floor's: mean KL <= max(KL_MAX, FLOOR_X (1.5) x the floor's), top-1 >= the floor's less FLOOR_TOP (0.05)
#        of the positions, in place of TOP1_MIN); LONG_LOGITS (1; 0: with LONG, the prompts' speeds only - no every-row
#        logits runs, no floor, no comparison, and the single process's speed run with the engine's defaults, not
#        DS4_YOCO=0 DS4_PATH_PICK=0); PYTHON (python3)
#
# Both processes read the experts in place (--mmap-experts on: the files' page cache, shared by both, no copies in
# RAM), with no SSD store, a fixed placement (--no-expert-cache, and an expert profile file of each process's own in
# the out dir, removed before it starts: none is read, and none is shared - not the pack's expert-profile.bin, which a
# layer range writes its own rows into) and Engram row caches of ENGRAM_MB: what one PC's RAM holds twice over is the
# dense weights and the KV caches.  The worker listens on 127.0.0.1 only and the link carries a token made here
# (STRATA_STAGE_TOKEN, unless one is set).
#
# The split's logits are not expected to be the same bits as one process's: the rows cross the link as the engine's
# compact ones (its default: the streams bf16, the picks exact), and each process places its own experts (which ones
# VRAM holds follows the VRAM free at its load) - a window's sum of experts in another place is another order of
# additions.  The check prints the top-1 agreement and the KL of the split's distribution from the single process's,
# and whether the greedy tokens agree.
#
# The speculative check: with VRAM_EXPERTS=0 neither process puts an expert in VRAM (the drafter's included), both read
# every expert on the CPU lane with CPU_THREADS threads, the drafter runs on the main process in both, the prompt goes
# in windows of 16 in both (a main process with the drafter sends no prompt chunks: their replies carry no taps), and
# each window - a draft's check too - runs on the same GPU kernels, the worker undoing a check's rejected rows as one
# process does; the rows' bf16 streams are what still differs, so a greedy run may part from one process's at a near
# tie, and the check says where (a fail only with STRATA_REMOTE_ROWS=fp32 on both sides - debugging - where the ids are
# expected IDENTICAL).  With experts in VRAM the two placements differ as well (the worker loads first).
#
# The chunks' check (LONG): the prompt as one feed through the chunked prompt path on both sides, in chunks of CHUNK
# rows - the main process's layers on chunk c + 1 while the worker reads chunk c - and through one process with the
# same chunks (DS4_YOCO=0: no long prompt's skip, which a split cannot take; DS4_PATH_PICK=0: the chunks, never
# windows).  Twice each: with --dump-logits and --window <its length> (every row's logits, every row back over the
# link), compared as above over the prompt's rows (the dump's last write, after the tokens, is the prompt's rows again:
# a chunk's every-row logits are what the engine's logits() still holds after a decode step), then plainly (the last
# row's logits only) for the prompt's tokens/s; with FLOOR_VRAM a third single-process run, every row's logits only
# (the noise floor).  DS4_LAYER_MAJOR and
# the other engine variables pass through to all (the split's two sides go chunk after chunk, overlapped, unless
# DS4_LAYER_MAJOR=1: then the main process may take its layers one after another over all the chunks (the worker
# reads one chunk at a time either way), which overlaps little; the
# single process decides by its own gate, DS4_LAYER_MAJOR=0 never).
# The rows on the link are the engine's default, bf16, a window's and a chunk's alike; STRATA_REMOTE_ROWS=fp32 passes
# through to both split processes (debugging only - nothing here sets it).
set -u
if [ $# -lt 3 ]; then
    sed -n '2,68p' "$0"
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
long_logits=${LONG_LOGITS:-1}
py=${PYTHON:-python3}
mkdir -p "$out"
export STRATA_STAGE_TOKEN=${STRATA_STAGE_TOKEN:-loopback-$$-$RANDOM$RANDOM}
export CUDA_VISIBLE_DEVICES=$gpu
export DS4_ENGRAM_CACHE_MB=$eg_mb
common=(--pack "$pack" --vram-experts "$vram" --mmap-experts on --ssd-cache 0 --no-expert-cache
        --cpu-threads "$thr" --vision off --verbose)
# a process's own expert profile file, new (the main process's runs: main.sh names one by its PID)
prof() {
    rm -f "$out/$1.profile"
    echo "$out/$1.profile"
}

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

# the worker, in the background until this script ends (or the single-process runs need its GPU); its log made first,
# so the wait below never reads a file the background shell has not opened yet
: > "$out/worker.err"
"$exe" "${common[@]}" --expert-profile "$(prof worker)" --max-context "$ctx" --prefill-chunk "$chunk" \
    --stage-worker "$port" --stage-bind 127.0.0.1 --stage-begin "$K" > "$out/worker.out" 2> "$out/worker.err" &
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
    printf 'p=%q/main-$$.profile; rm -f "$p"   # (each run its own profile file)\n' "$out"
    printf 'exec %q "$@"' "$exe"
    printf ' %q' "${common[@]}" --prefill-chunk "$chunk" --remote-stage "127.0.0.1:$port" --stage-end "$K"
    printf ' --expert-profile "$p"'
    echo
} > "$main"
chmod +x "$main"

# the logits of two runs (files of 129280 floats a row; ROWS > 0: their first ROWS rows only): top-1 agreement and
# KL(the second || the first); fails (1) when the mean KL is above KL_MAX or the top-1 agreement below TOP1_MIN - or,
# with a third run's file (the noise floor: the second run's process at another placement), its numbers against the
# second's first, and the rule FLOOR_VRAM above
compare() {   # <split> <single> <rows> [<floor>]
    "$py" -c '
import sys, numpy as np
rows = int(sys.argv[3])
def load(f):
    x = np.memmap(f, dtype=np.float32, mode="r").reshape(-1, 129280)
    return x[:rows] if rows > 0 else x
def kl_top(ref, got):   # KL(ref || got) of each position, and the positions where the top-1 tokens agree
    n = min(len(ref), len(got))
    kl, top = [], 0
    for x, y in zip(ref[:n], got[:n]):
        x, y = x.astype(np.float64), y.astype(np.float64)
        px = np.exp(x - x.max()); px /= px.sum()
        py = np.exp(y - y.max()); py /= py.sum()
        kl.append(float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30)))))
        top += int(x.argmax() == y.argmax())
    return kl, top
a, b = load(sys.argv[1]), load(sys.argv[2])
kl, top = kl_top(b, a)
n = len(kl)
if n == 0 or (rows > 0 and n < rows):
    print("  FAIL: %d positions of logits to compare%s" % (n, " of %d" % rows if rows > 0 else ""))
    sys.exit(1)
print("  %d positions: top-1 agrees at %d, KL(single || split) mean %.3e max %.3e" % (n, top, np.mean(kl), np.max(kl)))
kl_max, top_min, fx, ft = (float(v) for v in sys.argv[4:8])
if len(sys.argv) > 8:
    fkl, ftop = kl_top(b, load(sys.argv[8]))
    if len(fkl) != n:
        print("  FAIL: the noise floor has %d positions, the split %d" % (len(fkl), n))
        sys.exit(1)
    print("  noise floor, %d positions: top-1 agrees at %d, KL(single || single at another placement) mean %.3e max %.3e"
          % (n, ftop, np.mean(fkl), np.max(fkl)))
    kl_lim, top_need = max(kl_max, fx * np.mean(fkl)), ftop / n - ft
    print("  the rule against the floor: KL mean <= %.3e (max(KL_MAX %g, %g x the floor)), top-1 >= %.1f%% (the floor %.1f%% less %g)"
          % (kl_lim, kl_max, fx, 100 * top_need, 100 * ftop / n, 100 * ft))
    if np.mean(kl) > kl_lim or top < ftop - ft * n - 1e-6:   # (top-1 counts, not shares: no rounding at the bound)
        print("  FAIL: the split is further from one process than the noise floor allows")
        sys.exit(1)
elif np.mean(kl) > kl_max or top < top_min * n:
    print("  FAIL: the split is further from one process than KL_MAX %g / TOP1_MIN %g allow" % (kl_max, top_min))
    sys.exit(1)
' "$1" "$2" "$3" "${KL_MAX:-0.02}" "${TOP1_MIN:-0.9}" "${FLOOR_X:-1.5}" "${FLOOR_TOP:-0.05}" ${4:+"$4"}
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
        s=$?; echo "plain split rc=$s"; [ $s -eq 0 ] || rc=1
    fi
fi
# the IDS runs' prompt path: windows on both sides with CHUNK 0, else chunks on both (a dump feeds windows of 16:
# DS4_CHUNK_MIN=2 sends each as a chunk on either side - the split main's own minimum is 2 only when all its experts
# are on the flow)
if [ "$chunk" -gt 0 ]; then
    ids_min=2
    ids_chunk=$chunk
else
    ids_min=100000
    ids_chunk=0
fi
if [ -n "${IDS:-}" ]; then
    ids=$(cat "$IDS")
    echo "== logits: the split, then one process (prompt chunks of $ids_chunk on both)"
    DS4_CHUNK_MIN=$ids_min "$main" --max-context "$ctx" --dspark none --n 64 --ids "$ids" --dump-logits "$out/split.f32" \
        > "$out/split.out" 2> "$out/split.err"
    s=$?; echo "split rc=$s"; [ $s -eq 0 ] || rc=1
fi
if [ -n "$long" ]; then
    echo "== a long prompt ($nlong tokens) in chunks of $chunk: the split, then one process"
    if [ "$long_logits" != 0 ]; then
        "$main" --max-context "$ctx" --dspark none --n 16 --ids "@$long" --window "$nlong" --dump-logits "$out/split-long.f32" \
            > "$out/split-long.out" 2> "$out/split-long.err"
        s=$?; echo "split (every row's logits) rc=$s"; [ $s -eq 0 ] || rc=1
    fi
    "$main" --max-context "$ctx" --dspark none --n 16 --ids "@$long" > "$out/split-speed.out" 2> "$out/split-speed.err"
    s=$?; echo "split (the prompt's speed) rc=$s"; [ $s -eq 0 ] || rc=1
    grep -h 'chunks of\|remote stage 127' "$out/split-speed.err" | head -2 | sed 's/^/  /'
fi
kill $wpid 2> /dev/null
wait $wpid 2> /dev/null

# one process, the whole trunk, on the same GPU
if [ -n "${SPEC_IDS:-}" ]; then
    echo "== speculative decoding through one process"
    DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --expert-profile "$(prof spec-single)" --max-context "$ctx" \
        --prefill-chunk 0 "${spec[@]}" --ids "@$SPEC_IDS" > "$out/spec-single.out" 2> "$out/spec-single.err"
    s1=$?
    echo "spec single rc=$s1"
    arms="spec-split spec-single"
    if [ "${SPEC_NONE:-0}" = 1 ]; then
        DS4_CHUNK_MIN=100000 "$exe" "${common[@]}" --expert-profile "$(prof plain-single)" --max-context "$ctx" \
            --prefill-chunk 0 --dspark none --n "$spec_n" --ids "@$SPEC_IDS" > "$out/plain-single.out" 2> "$out/plain-single.err"
        s=$?; echo "plain single rc=$s"; [ $s -eq 0 ] || rc=1
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
        # (bf16 rows, or experts in VRAM: a near tie may part them - said above)
        [ "$vram" = 0 ] && [ "${STRATA_REMOTE_ROWS:-bf16}" = fp32 ] && rc=1
    fi
fi
if [ -n "${IDS:-}" ]; then
    DS4_CHUNK_MIN=$ids_min "$exe" "${common[@]}" --expert-profile "$(prof single)" --max-context "$ctx" \
        --prefill-chunk "$ids_chunk" --dspark none --n 64 --ids "$ids" --dump-logits "$out/single.f32" \
        > "$out/single.out" 2> "$out/single.err"
    s=$?; echo "single rc=$s"; [ $s -eq 0 ] || rc=1
    for b in split single; do
        echo "  $b: $(grep -h 'in VRAM' "$out/$b.err" | head -1 | sed 's/.*experts/experts/')"
    done
    same split single
    compare "$out/split.f32" "$out/single.f32" 0 || rc=1
fi
if [ -n "$long" ] && [ "$long_logits" != 0 ]; then
    DS4_YOCO=0 DS4_PATH_PICK=0 "$exe" "${common[@]}" --expert-profile "$(prof single-long)" --max-context "$ctx" \
        --prefill-chunk "$chunk" --dspark none --n 16 --ids "@$long" --window "$nlong" --dump-logits "$out/single-long.f32" \
        > "$out/single-long.out" 2> "$out/single-long.err"
    s=$?; echo "single (every row's logits) rc=$s"; [ $s -eq 0 ] || rc=1
fi
if [ -n "$long" ]; then
    # (LONG_LOGITS=0, speed only: one process as the engine runs it by default - its skip and its own path pick)
    pin=(DS4_YOCO=0 DS4_PATH_PICK=0)
    [ "$long_logits" != 0 ] || pin=()
    env ${pin[@]+"${pin[@]}"} "$exe" "${common[@]}" --expert-profile "$(prof single-speed)" --max-context "$ctx" \
        --prefill-chunk "$chunk" --dspark none --n 16 --ids "@$long" > "$out/single-speed.out" 2> "$out/single-speed.err"
    s=$?; echo "single (the prompt's speed) rc=$s"; [ $s -eq 0 ] || rc=1
fi
if [ -n "$long" ] && [ "$long_logits" != 0 ]; then
    floor=
    if [ -n "${FLOOR_VRAM:-}" ]; then   # the noise floor: one process at another placement, the same chunks
        DS4_YOCO=0 DS4_PATH_PICK=0 "$exe" "${common[@]}" --vram-experts "$FLOOR_VRAM" --expert-profile "$(prof floor-long)" \
            --max-context "$ctx" --prefill-chunk "$chunk" --dspark none --n 16 --ids "@$long" --window "$nlong" \
            --dump-logits "$out/floor-long.f32" > "$out/floor-long.out" 2> "$out/floor-long.err"
        s=$?; echo "single at --vram-experts $FLOOR_VRAM, the noise floor (every row's logits) rc=$s"; [ $s -eq 0 ] || rc=1
        floor=$out/floor-long.f32
    fi
    for b in split-long single-long ${floor:+floor-long}; do
        echo "  $b: $(grep -h 'in VRAM' "$out/$b.err" | head -1 | sed 's/.*experts/experts/')"
    done
    same split-long single-long
    compare "$out/split-long.f32" "$out/single-long.f32" "$nlong" $floor || rc=1
fi
if [ -n "$long" ]; then
    echo "  tokens after the prompt (speed runs): $(cmp -s <(grep -h '^[0-9]' "$out/split-speed.out") \
        <(grep -h '^[0-9]' "$out/single-speed.out") && echo IDENTICAL || echo DIFFER)"
    echo "  split:  $(speed split-speed)"
    echo "  single: $(speed single-speed)"
fi
exit $rc
