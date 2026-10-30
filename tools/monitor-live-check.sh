#!/bin/bash
# monitor-live-check.sh MAIN WORKER - the Monitor's Workers card (serve/stages.py) against a running remote-stage
# split: MAIN is the server's HOST:PORT, WORKER the worker PC's node agent HOST:PORT (tools/stage_node.py).  Only HTTP:
# the server's /metrics and /v1/chat/completions, and over HTTPS with this PC's key folder (TLS_DIR) the agent's /stop
# and /start; it starts no server.  One line
# "PASS|FAIL|SKIP step N: ..." per step, and the number of FAILs as the exit code.
#
# The server must run with "stage_agents" in its run config (true, or the worker's address mapped to its agent) and
# STRATA_REMOTE_TIMING=1 in the engine's environment (the round trips).  Steps:
#   1  /metrics?workers=1 has "stages" (the engine runs a remote stage) and asks the agents
#   2  after a request: worker 1 connected, its layers (EXPECT_LAYERS), the rows' format, a round trip, and from the
#      agent its GPU (EXPECT_GPU), engine version and expert cache
#   3  the worker stopped through its agent: the card says down; a request then waits (reconnecting), the worker is
#      started again (WORKER_START: the agent /start body the split was started with) and the request ends;
#      connected again, reconnects + 1.  Without WORKER_START it is skipped and the worker is left alone; when the
#      step fails between /stop and /start, the EXIT trap starts it again
#   4  this PC's private key (TLS_DIR/key.pem) is nowhere in /metrics, /metrics?workers=1 or the page
#   5  decode tok/s (the engine's timings.predicted_per_second) with and without a Monitor polling
#      /metrics?workers=1 every POLL_S, ROUNDS x ABBA; FAIL when the polled median is more than TOL below
#   6  SINGLE (HOST:PORT of a server without a remote stage, optional): no "stages" in its /metrics?workers=1
# env: TLS_DIR (STRATA_STAGE_TLS: this PC's key folder, strata --stage-keygen, whose peers/ holds the worker PC's
#      cert.pem; the agent's peers/ must hold this PC's), WPORT (the worker's port, 7841), WORKER_START (a file), EXPECT_LAYERS
#      (26-47), EXPECT_GPU (a part of the GPU's name, e.g. "2080 Ti"; empty: any), API_KEY (the server's, if set),
#      SINGLE, ROUNDS (2), TOL (0.05), POLL_S (1: the Monitor tab's cadence), TOKENS (300 a speed request),
#      STEPS ("1 2 3 4 5 6")
set -u
MAIN=${1:?MAIN host:port}
WORKER=${2:?WORKER agent host:port}
TLS_DIR=${TLS_DIR:-${STRATA_STAGE_TLS:-}}
HERE=$(cd "$(dirname "$0")" && pwd)
WPORT=${WPORT:-7841}
WORKER_START=${WORKER_START:-}
EXPECT_LAYERS=${EXPECT_LAYERS:-26-47}
EXPECT_GPU=${EXPECT_GPU:-}
SINGLE=${SINGLE:-}
ROUNDS=${ROUNDS:-2}
TOL=${TOL:-0.05}
POLL_S=${POLL_S:-1}
TOKENS=${TOKENS:-300}
STEPS=${STEPS:-1 2 3 4 5 6}
[ -n "$TLS_DIR" ] && [ -f "$TLS_DIR/key.pem" ] || { echo "FAIL: no key folder (TLS_DIR or STRATA_STAGE_TLS, with key.pem)"; exit 1; }
AUTH=()
[ -n "${API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $API_KEY")
TMP=$(mktemp -d)
FAILS=0
STOPPED=0   # step 3 stopped the worker and has not started it again

say() { echo "$1 step $2: $3"; [ "$1" != FAIL ] || FAILS=$((FAILS + 1)); }
note() { echo "  $*"; }
want() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
metrics() { curl -s -m 20 "${AUTH[@]}" "http://${2:-$MAIN}/metrics${1:+?$1}"; }
agent() {   # agent PATH BODY [TIMEOUT]: a POST to the worker PC's agent (BODY @FILE: that file), over HTTPS with TLS_DIR
    python3 -I -c '
import json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import stage_node
body = sys.argv[4]
data = Path(body[1:]).read_bytes() if body.startswith("@") else body.encode()
opener = stage_node.https_opener(stage_node.client_context(Path(sys.argv[2]).resolve(), "monitor-live-check"))
try:
    with opener.open(urllib.request.Request(sys.argv[3], data=data,
                                            headers={"Content-Type": "application/json"}),
                     timeout=float(sys.argv[5])) as r:
        sys.stdout.write(r.read().decode("utf-8", "replace"))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))' "$HERE" "$TLS_DIR" "https://$WORKER$1" "$2" "${3:-60}"
}
chat() {   # chat MAX_TOKENS: the whole answer (JSON)
    curl -s -m 900 "${AUTH[@]}" -H "Content-Type: application/json" "http://$MAIN/v1/chat/completions" -d \
        "{\"messages\": [{\"role\": \"user\", \"content\": \"Write a short story about a lighthouse keeper who finds a map.\"}], \"max_tokens\": $1, \"temperature\": 0, \"stream\": false}"
}
js() {   # js PATH < JSON: a field by its dotted path (list indexes as numbers); "" when missing or null, JSON when not a string
    python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    for k in sys.argv[1].split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
except (ValueError, KeyError, IndexError, TypeError):
    d = None
print("" if d is None else d if isinstance(d, str) else json.dumps(d))' "$1"
}
w1() { metrics workers=1 > "$TMP/m.json"; js "stages.workers.0.$1" < "$TMP/m.json"; }
restart_worker() {   # the agent /start of WORKER_START; 0 when the worker listens
    local r
    r=$(agent /start "@$WORKER_START" 900)
    STOPPED=0
    [ "$(echo "$r" | js ok)" = true ] && return 0
    note "the worker's /start: $(echo "$r" | js error) $(echo "$r" | js exit)"
    return 1
}
finish() {
    [ "$STOPPED" = 1 ] && { echo "  (the EXIT trap: starting the worker again)"; restart_worker; }
    rm -rf "$TMP"
}
trap finish EXIT
trap 'exit 143' TERM INT

if want 1; then
    metrics workers=1 > "$TMP/m.json"
    if [ -z "$(js stages < "$TMP/m.json")" ]; then
        say FAIL 1 "no \"stages\" in /metrics?workers=1 (no remote stage, or a server without the Workers card)"
    elif [ "$(js stages.agents < "$TMP/m.json")" != true ]; then
        say FAIL 1 "a remote stage, but the agents are not asked: \"stage_agents\" is not in the run config"
    else
        say PASS 1 "a remote stage, $(js stages.workers < "$TMP/m.json" | python3 -c 'import json, sys; print(len(json.load(sys.stdin)))') worker(s), agents asked"
    fi
fi

if want 2; then
    chat 512 > "$TMP/warm.json"   # windows enough for STRATA_REMOTE_TIMING's round trips
    [ -z "$(js usage < "$TMP/warm.json")" ] && note "the warm-up request failed: $(head -c 300 "$TMP/warm.json")"
    exp_layers="[${EXPECT_LAYERS%-*}, ${EXPECT_LAYERS#*-}]"
    for i in $(seq 1 25); do   # an agent is asked every 15 s, on a thread of its own
        metrics workers=1 > "$TMP/m.json"
        miss=""
        [ "$(js stages.workers.0.state < "$TMP/m.json")" = connected ] || miss="$miss state=$(js stages.workers.0.state < "$TMP/m.json")"
        [ "$(js stages.workers.0.layers < "$TMP/m.json")" = "$exp_layers" ] || miss="$miss layers=$(js stages.workers.0.layers < "$TMP/m.json")"
        [ -n "$(js stages.rows < "$TMP/m.json")" ] || miss="$miss rows"
        [ -n "$(js stages.workers.0.round_trip_ms < "$TMP/m.json")" ] || miss="$miss round_trip_ms"
        [ "$(js stages.workers.0.agent.ok < "$TMP/m.json")" = true ] || miss="$miss agent($(js stages.workers.0.agent.error < "$TMP/m.json"))"
        gpu=$(js stages.workers.0.gpu.name < "$TMP/m.json")
        [ -n "$gpu" ] && [[ "$gpu" == *"$EXPECT_GPU"* ]] || miss="$miss gpu=$gpu"
        [ -n "$(js stages.workers.0.version < "$TMP/m.json")" ] || miss="$miss version"
        [ -n "$(js stages.workers.0.cache_slots < "$TMP/m.json")$(js stages.workers.0.cache_gib < "$TMP/m.json")" ] || miss="$miss cache"
        [ -z "$miss" ] && break
        sleep 2
    done
    note "worker 1: $(js stages.workers.0 < "$TMP/m.json" | head -c 700)"
    note "rows $(js stages.rows < "$TMP/m.json"), local layers $(js stages.local_layers < "$TMP/m.json")"
    if [ -z "$miss" ]; then say PASS 2 "connected, layers $EXPECT_LAYERS, $gpu, round trip $(js stages.workers.0.round_trip_ms < "$TMP/m.json") ms"
    else say FAIL 2 "missing or wrong after 50 s:$miss"; fi
fi

if want 3; then
    if [ -z "$WORKER_START" ] || [ ! -f "$WORKER_START" ]; then
        say SKIP 3 "no WORKER_START file (the agent /start body): the worker is not stopped"
    else
        r0=$(w1 reconnects)
        state=$(w1 state)
        if [ "$state" != connected ]; then
            say FAIL 3 "worker 1 is $state before the stop, not connected: not stopped"
        else
            STOPPED=1
            t0=$(date +%s)
            note "the worker's /stop: $(agent /stop "{\"port\": $WPORT}" 120)"
            for i in $(seq 1 30); do state=$(w1 state); [ "$state" = down ] && break; sleep 2; done
            t_down=$(( $(date +%s) - t0 ))
            note "after the stop: $state in $t_down s ($(w1 error.text))"
            chat 64 > "$TMP/wait.json" &
            cp=$!
            seen=""
            for i in $(seq 1 20); do   # INFO remote_stage_wait= every 10 s while the request waits
                s=$(w1 state)
                [ "$s" = reconnecting ] && { seen=reconnecting; break; }
                sleep 2
            done
            note "while the request waits: ${seen:-not reconnecting (last: $s)}"
            restart_worker; started=$?
            note "the worker started again: $([ $started = 0 ] && echo yes || echo NO) after $(( $(date +%s) - t0 )) s"
            wait $cp
            [ -z "$(js usage < "$TMP/wait.json")" ] && note "the waiting request: $(head -c 300 "$TMP/wait.json")"
            for i in $(seq 1 15); do state=$(w1 state); [ "$state" = connected ] && break; sleep 2; done
            r1=$(w1 reconnects)
            note "after the request: $state, reconnects $r0 -> $r1"
            if [ "$t_down" -le 60 ] && [ "$(w1 state)" = connected ] && [ -n "$seen" ] && [ "$started" = 0 ] && \
               [ -n "$(js usage < "$TMP/wait.json")" ] && [ "$r1" = "$((r0 + 1))" ]; then
                say PASS 3 "down in $t_down s, reconnecting while the request waited, connected again, reconnects $r0 -> $r1"
            else
                say FAIL 3 "down in $t_down s ($([ "$t_down" -le 60 ] && echo ok || echo slow)), reconnecting seen: ${seen:-no}, worker started: $started, request ok: $([ -n "$(js usage < "$TMP/wait.json")" ] && echo yes || echo no), state $state, reconnects $r0 -> $r1"
            fi
        fi
    fi
fi

if want 4; then
    K=$(sed -n 2p "$TLS_DIR/key.pem")   # the key's first base64 line (never printed)
    if [ ${#K} -lt 16 ]; then
        say FAIL 4 "no base64 line in $TLS_DIR/key.pem: no check"
    else
        n=0
        for q in "" workers=1 "requests=all&workers=1"; do metrics "$q" | grep -qF -- "$K" && { n=$((n + 1)); note "the key in /metrics?$q"; }; done
        curl -s -m 20 "http://$MAIN/" | grep -qF -- "$K" && { n=$((n + 1)); note "the key in the page"; }
        [ $n = 0 ] && say PASS 4 "the private key is in no answer" || say FAIL 4 "the private key in $n answer(s)"
    fi
fi

if want 5; then
    poll_loop() { while :; do metrics workers=1 > /dev/null; sleep "$POLL_S"; done; }
    speed() {   # speed on|off: one request's predicted_per_second
        local pp=""
        [ "$1" = on ] && { poll_loop & pp=$!; }
        chat "$TOKENS" > "$TMP/s.json"
        [ -n "$pp" ] && { kill $pp; wait $pp 2> /dev/null; }
        js timings.predicted_per_second < "$TMP/s.json"
    }
    chat "$TOKENS" > /dev/null   # the prompt in the cache: every timed request reads the same
    off=""; on=""
    for r in $(seq 1 "$ROUNDS"); do
        for m in off on on off; do
            v=$(speed $m)
            note "round $r, Monitor $m: ${v:-no timings} tok/s"
            [ "$m" = on ] && on="$on $v" || off="$off $v"
        done
    done
    res=$(python3 -c '
import statistics, sys
off, on, tol = [float(x) for x in sys.argv[1].split()], [float(x) for x in sys.argv[2].split()], float(sys.argv[3])
if len(off) < 2 or len(on) < 2:
    print("FAIL too few timed requests"); sys.exit()
a, b = statistics.median(off), statistics.median(on)
print(("PASS" if b >= a * (1 - tol) else "FAIL") + f" median {a:.1f} tok/s without the Monitor, {b:.1f} with ({(b - a) / a * 100:+.1f}%, tolerance {tol * 100:.0f}%)")' "$off" "$on" "$TOL")
    [ -z "$res" ] && res="FAIL the comparison did not run"
    say "${res%% *}" 5 "${res#* }"
fi

if want 6; then
    if [ -z "$SINGLE" ]; then
        say SKIP 6 "no SINGLE server given"
    else
        metrics workers=1 "$SINGLE" > "$TMP/s1.json"
        if [ -z "$(js engine < "$TMP/s1.json")" ]; then say FAIL 6 "no /metrics from $SINGLE"
        elif [ -n "$(js stages < "$TMP/s1.json")" ]; then say FAIL 6 "\"stages\" in a single-PC server's /metrics"
        else say PASS 6 "no \"stages\": the card stays hidden"; fi
    fi
fi
echo "monitor-live-check: $FAILS FAIL(s)"
exit $FAILS
