#!/usr/bin/env bash
#
# bench-kimik3.sh — measure Kimi K3 serving performance (tokens/sec, TTFT, ITL)
# with SGLang's bench_serving client, run inside the same .sif against the live
# endpoint. Use it to compare configs (DSpark, radix cache, tuning knobs) with
# real numbers instead of guessing. See the README "Performance tuning".
#
# Usage:
#   ./bench-kimik3.sh                 sweep concurrency 2/8/32 (default)
#   ./bench-kimik3.sh latency         single-stream latency only (concurrency 1)
#   ./bench-kimik3.sh throughput      saturate at BENCH_MAX_CONCURRENCY
#   ./bench-kimik3.sh longcontext     100k in / 512 out, concurrency 1
#
# Tunables (env or kimik3.env): BENCH_INPUT_LEN, BENCH_OUTPUT_LEN, BENCH_NUM_PROMPTS,
#   BENCH_CONCURRENCY (space-separated list for sweep), BENCH_MAX_CONCURRENCY,
#   BENCH_EXTRA_ARGS (appended verbatim to sglang.bench_serving).
#
# 'longcontext' exists because every other shape here is 1024/512, and the
# agentic clients this repo is built for resend 100k+ every turn. Those are
# different regimes, not the same regime scaled: measured 9 Aug 2026, DSpark
# turned from a 3.1x win at 1024 into a ~3x loss at 106k. Cause established
# 12 Aug 2026 — the draft checkpoint pinned at the time was trained at 4,096
# tokens, so 106k was 25x outside its window. The draft default has moved; the
# pairing this mode measures has not been re-run yet.
# It only changes DEFAULTS — an explicit BENCH_* still wins. See the README
# "DSpark collapsed at long context".
#
# Upstream MI355 TP8 reference (sgl-project/sglang issue #32548):
#   baseline  c=2: 820 tok/s, TPOT 20.86 ms | c=8: 2356 | c=32: 4898
#   DSpark    c=2: 1659 tok/s, TPOT  8.94 ms | c=8: 2927 | c=32: 3715
# A large shortfall against these means something is mis-set — investigate
# rather than accept it.
#
# Runs as a pure HTTP client — no GPU needed — but Apptainer only exists on the
# compute node, so run this from a shell on the serving node (e.g. via
# `srun --overlap --jobid <jobid> --pty /bin/bash -l`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="sweep"
for arg in "$@"; do
    case "$arg" in
        latency|throughput|sweep|longcontext) MODE="$arg" ;;
        *) printf 'Unknown argument: %s (use latency | throughput | sweep | longcontext)\n' "$arg" >&2; exit 1 ;;
    esac
done

# Capture the shape the CALLER asked for, before kimik3.env is sourced.
# kimik3-env.example ships 'export BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"',
# so every real config file sets all of these unconditionally. Once it has been
# sourced, "${BENCH_INPUT_LEN:-100000}" can no longer tell a deliberate 1024
# from a config-file default — which silently ran 'longcontext' at 1024/512
# c=2/8/32 on 9 Aug 2026. A named shape mode has to outrank the config file.
for _v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_CONCURRENCY; do
    declare "CLI_$_v=${!_v:-}"
done

log()  { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bench WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bench ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config (shared with serve-kimik3.sh) ───────────────────────────────

ENV_FILE="${KIMIK3_ENV:-$SCRIPT_DIR/kimik3.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# The pre-source capture above is necessary but NOT sufficient, and that gap cost
# a 100k run on 12 Aug 2026. This README tells you to 'source kimik3.env' for
# MODEL_CACHE_DIR and friends — and once you have, the whole config file is in the
# environment, so the capture records the sweep shape as though the caller had
# typed it. All four values arrived together, which is the signature.
#
# By value alone the two cases are indistinguishable. So ask the config file what
# IT sets, in a subshell with these four unset, and treat an exact match as "this
# came from the config file". A caller passing precisely the configured value
# while also naming a shape mode is contradicting themselves; any other value
# still wins, so 'BENCH_INPUT_LEN=32768 ./bench-kimik3.sh longcontext' is intact.
if [[ -f "$ENV_FILE" ]]; then
    cfg_shape="$(env -u BENCH_INPUT_LEN -u BENCH_OUTPUT_LEN \
                     -u BENCH_NUM_PROMPTS -u BENCH_CONCURRENCY \
        bash -c 'source "$1" >/dev/null 2>&1
                 printf "%s\n%s\n%s\n%s\n" "${BENCH_INPUT_LEN:-}" \
                     "${BENCH_OUTPUT_LEN:-}" "${BENCH_NUM_PROMPTS:-}" \
                     "${BENCH_CONCURRENCY:-}"' _ "$ENV_FILE" 2>/dev/null || true)"
    { read -r CFG_BENCH_INPUT_LEN
      read -r CFG_BENCH_OUTPUT_LEN
      read -r CFG_BENCH_NUM_PROMPTS
      read -r CFG_BENCH_CONCURRENCY
    } <<<"$cfg_shape"
    for _v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_CONCURRENCY; do
        _cli="CLI_$_v"; _cfg="CFG_$_v"
        [[ -n "${!_cli:-}" && "${!_cli}" == "${!_cfg:-}" ]] && declare "$_cli="
    done
fi

PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k3}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
SIF_PATH="${SIF_PATH:-}"

[[ -n "$MODEL_CACHE_DIR" ]] || die "MODEL_CACHE_DIR is not set (source kimik3.env or export it)."
SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/kimik3-mi355x.sif}"
if [[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]]; then
    SIF_PATH="${SIF_PATH%/}/kimik3-mi355x.sif"
fi
[[ -f "$SIF_PATH" ]] || die "No .sif at $SIF_PATH (run './serve-kimik3.sh pull' first)."
command -v apptainer >/dev/null 2>&1 \
    || die "apptainer not found — run this on the serving compute node, not the login node."

# Benchmark parameters (overridable). Defaults are mode-dependent; anything
# already set in the environment or kimik3.env wins in either branch, so
# 'BENCH_INPUT_LEN=32768 ./bench-kimik3.sh longcontext' does what it looks like.
if [[ "$MODE" == "longcontext" ]]; then
    # n=4, not 200: at 100k a single prefill is measured in tens of seconds, so
    # 200 prompts is an overnight job that nobody will run twice. Four is enough
    # to see TPOT, and TTFT here is a headline number rather than a warmup cost.
    #
    # CLI_* — the pre-source capture — not the post-source values, so that
    # 'BENCH_INPUT_LEN=32768 ./bench-kimik3.sh longcontext' still works while a
    # kimik3.env carrying the sweep shape does not silently override the mode.
    BENCH_INPUT_LEN="${CLI_BENCH_INPUT_LEN:-100000}"
    BENCH_OUTPUT_LEN="${CLI_BENCH_OUTPUT_LEN:-512}"
    BENCH_NUM_PROMPTS="${CLI_BENCH_NUM_PROMPTS:-4}"
    BENCH_CONCURRENCY="${CLI_BENCH_CONCURRENCY:-1}"
else
    BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
    BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
    BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
    BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-2 8 32}"
fi
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

for v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a positive integer, got '${!v}'."
done

# A request longer than the server's context window is refused, and at 100k that
# is an easy way to waste an allocation: CONTEXT_LEN=262144 is this repo's own
# documented fix for KV allocation failures, and it silently caps what can be
# measured here. Catch it before the first prefill rather than after.
if [[ "${CONTEXT_LEN:-}" =~ ^[0-9]+$ ]] \
   && (( BENCH_INPUT_LEN + BENCH_OUTPUT_LEN > CONTEXT_LEN )); then
    die "This shape needs $((BENCH_INPUT_LEN + BENCH_OUTPUT_LEN)) tokens of context but the
     server was started with CONTEXT_LEN=$CONTEXT_LEN. Either lower BENCH_INPUT_LEN,
     or raise CONTEXT_LEN in kimik3.env and restart the server."
fi

# sglang's random dataset samples each length uniformly from
# [BENCH_*_LEN * ratio, BENCH_*_LEN]. Its own default is 0.0, which means every
# request is on average HALF its nominal size — measured 29 Jul 2026: a nominal
# 1024/512 sweep actually sent 507 in / 262 out per request. Aggregate tok/s at
# fixed concurrency scales with tokens per request, so that quietly halves the
# headline number while leaving TPOT alone, and makes the result incomparable to
# any published figure. Default to 1.0 (fixed sizes, exactly as configured).
BENCH_RANGE_RATIO="${BENCH_RANGE_RATIO:-1.0}"

# How many times to repeat each concurrency. A single run gives you a number
# with no error bar, which is not enough to compare two configurations — or two
# nodes. Measured 29 Jul 2026: within a run, TPOT P90 sits 15-23% above the
# median, so a 5% gap between two single runs is not a result. Set 3-5 when you
# actually need to tell two things apart; each repeat costs a full sweep.
# Repeats are interleaved per concurrency so slow drift hits every repeat alike.
BENCH_REPEATS="${BENCH_REPEATS:-1}"

# ── Resolve API key ─────────────────────────────────────────────────────────

if [[ -z "${KIMIK3_API_KEY:-}" && -r "$MODEL_CACHE_DIR/kimik3-api-key" ]]; then
    KIMIK3_API_KEY="$(<"$MODEL_CACHE_DIR/kimik3-api-key")"
fi
[[ -n "${KIMIK3_API_KEY:-}" ]] || warn "No API key resolved — bench will 401 (set KIMIK3_API_KEY)."

# ── Server must be healthy ──────────────────────────────────────────────────

curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
    || die "No healthy server on http://127.0.0.1:${PORT}. Start it with './serve-kimik3.sh serve --detach' first."
log "Server on port $PORT is healthy."

# ── Tuned-MoE detector ──────────────────────────────────────────────────────
# The FP4 MoE is the dominant perf lever: on the heuristic FlyDSL fallback it
# runs far below the tuned kernel. Surface which path is active.

LOG_FILE="${LOG_FILE:-$MODEL_CACHE_DIR/kimik3-server.log}"
if [[ -r "$LOG_FILE" ]]; then
    if grep -qiE 'no tuned FlyDSL config|heuristic FlyDSL fallback|falling back to.*heuristic' "$LOG_FILE"; then
        warn "MoE is on the SLOW heuristic FlyDSL fallback (no tuned FP4 config for these shapes)."
        warn "  Confirm ENABLE_AITER=1 so SGLANG_AITER_K3_OPT / AITER_FLYDSL_FORCE /"
        warn "  AITER_SITUV2_A8W4 are set — those are what select the tuned K3 path."
    else
        log "No FlyDSL-fallback warning in the server log — tuned MoE path looks active. OK"
    fi
else
    warn "Server log $LOG_FILE not readable; skipping the tuned-MoE check."
fi

# ── Results file ────────────────────────────────────────────────────────────

BENCH_DIR="$MODEL_CACHE_DIR/bench"
mkdir -p "$BENCH_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="$BENCH_DIR/${STAMP}-${MODE}.txt"
{
    echo "# kimik3 benchmark  $(date)"
    # Record the node. Measured 29 Jul 2026, bun161 vs bun159 differed by 5-10%
    # on the same image and config, so a result without a hostname cannot be
    # compared against one taken somewhere else.
    echo "# host=$(hostname -s 2>/dev/null || echo '?')  rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?')"
    echo "# mode=$MODE model=$SERVED_MODEL_NAME port=$PORT"
    echo "# MODEL_ID=${MODEL_ID:-?}  IMAGE=${SGLANG_IMAGE:-?}"
    echo "# in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN num_prompts=$BENCH_NUM_PROMPTS"
    echo "# TP_SIZE=${TP_SIZE:-?} DP_SIZE=${DP_SIZE:-?} MEM_FRACTION=${MEM_FRACTION:-?} CONTEXT_LEN=${CONTEXT_LEN:-modelmax}"
    echo "# SPECULATIVE=${SPECULATIVE:-off} DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-?} ENABLE_AITER=${ENABLE_AITER:-?}"
    echo "# upstream MI355 TP8 ref: baseline 820/2356/4898 tok/s, DSpark 1659/2927/3715 @ c=2/8/32"
    echo
} > "$OUT_FILE"
log "Saving results to $OUT_FILE"

if [[ "$MODE" == "longcontext" ]]; then
    log "Long-context mode: $BENCH_INPUT_LEN in / $BENCH_OUTPUT_LEN out, concurrency $BENCH_CONCURRENCY, n=$BENCH_NUM_PROMPTS."
    log "  Expect a long wait before the first token. Prefill at this size is the"
    log "  slow path on ATTENTION_BACKEND=triton, and that TTFT is itself a result."
    if [[ "${SPECULATIVE:-}" == "dspark" ]]; then
        warn "  SPECULATIVE=dspark. Measured 12 Aug 2026 with the long-context draft:"
        warn "  DSpark WINS below ~55k (17.20 ms TPOT at 32k vs ~28 ms off) and LOSES"
        warn "  above it (51.13 ms at 100k vs 29.83 ms off). Gamma is not a lever at"
        warn "  100k — block size 3 and 7 land within 2% of each other. Run this mode"
        warn "  again with SPECULATIVE=\"\" and compare; that pair is the point of the"
        warn "  mode. Watch 'accept len' in $LOG_FILE — below ~2 at any"
        warn "  context means the draft is out of its depth, which is a DIFFERENT"
        warn "  failure from the step simply costing too much."
        warn "  NOTE those numbers were taken on the retired day-0 image. The 20 Aug"
        warn "  default moves BOTH sides — sglang #33981 rewrote the DSpark verify"
        warn "  attention, and MLA_DECODE_TUNE=1 (#34580) speeds decode either way —"
        warn "  so treat the ~55-60k crossover as provisional until you re-measure it."
    fi
fi

# ── Which bench module does this image have? ────────────────────────────────
# Upstream moved the implementation to sglang.benchmark.serving and left
# sglang.bench_serving as a shim. PREFER THE OLD PATH WHEREVER IT EXISTS.
#
# Measured 9 Aug 2026, and it cost a run: the day-0 20260727 image carries
# BOTH, and the new module imports disaggregation.utils -> quantization ->
# aiter at module scope, which the old one never touches. This benchmark is a
# pure HTTP client that needs no GPU and no kernels, so the lighter import
# chain is not just the safer choice, it is the correct one. Presence is not
# usability; only switch when there is nothing else to run.
BENCH_MODULE="${BENCH_MODULE:-}"
if [[ -z "$BENCH_MODULE" ]]; then
    BENCH_MODULE="sglang.bench_serving"
    # find_spec, not import: importing drags in torch and prints banners.
    if ! apptainer exec "$SIF_PATH" python3 -c \
         'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("sglang.bench_serving") else 1)' \
         2>/dev/null; then
        BENCH_MODULE="sglang.benchmark.serving"
        warn "This image has no sglang.bench_serving; falling back to $BENCH_MODULE."
        warn "  That module imports aiter at module scope. If it dies in aiter's"
        warn "  jit/core.py, AITER_JIT_DIR is the culprit — see below."
    fi
fi
log "Bench module: $BENCH_MODULE"
echo "# bench_module=$BENCH_MODULE" >> "$OUT_FILE"

# aiter's get_user_jit_dir() branches on `"AITER_JIT_DIR" in os.environ`, NOT on
# whether it has a value, then calls os.makedirs("") on an empty one:
#
#     FileNotFoundError: [Errno 2] No such file or directory: ''
#
# kimik3-env.example ships 'export AITER_JIT_DIR="${AITER_JIT_DIR:-}"', and
# apptainer passes the host environment straight through, so sourcing the config
# is enough to poison any import of aiter inside the container. serve-kimik3.sh
# resolves this to a real path and never sees it; this script did not. Resolve it
# the same way, and bind it so the container can actually write there.
AITER_JIT_DIR="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit}"
bench_binds=()
[[ -d "$AITER_JIT_DIR" ]] && bench_binds+=(--bind "$AITER_JIT_DIR")

# ── One benchmark run ───────────────────────────────────────────────────────
# $1 = max concurrency, $2 = request rate ("inf" to saturate)

run_one() {
    local conc="$1" rate="$2"
    log "Run: concurrency=$conc request-rate=$rate${rep:+  repeat $rep/$BENCH_REPEATS}  (in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN n=$BENCH_NUM_PROMPTS)"
    echo "=== concurrency=$conc request-rate=$rate${rep:+ repeat=$rep} ===" >> "$OUT_FILE"

    # shellcheck disable=SC2086  # intentional splitting of BENCH_EXTRA_ARGS
    apptainer exec \
        --env "OPENAI_API_KEY=${KIMIK3_API_KEY:-}" \
        --env "AITER_JIT_DIR=$AITER_JIT_DIR" \
        ${bench_binds[@]+"${bench_binds[@]}"} \
        "$SIF_PATH" \
        python3 -m "$BENCH_MODULE" \
            --backend sglang-oai \
            --base-url "http://127.0.0.1:${PORT}" \
            --model "$SERVED_MODEL_NAME" \
            --dataset-name random \
            --random-input-len "$BENCH_INPUT_LEN" \
            --random-output-len "$BENCH_OUTPUT_LEN" \
            --random-range-ratio "$BENCH_RANGE_RATIO" \
            --num-prompts "$BENCH_NUM_PROMPTS" \
            --max-concurrency "$conc" \
            --request-rate "$rate" \
            $BENCH_EXTRA_ARGS \
        2>&1 | tee -a "$OUT_FILE" \
        | grep -iE 'throughput|TTFT|TPOT|ITL|latency|Successful|concurrency|accept' || true
    echo >> "$OUT_FILE"
}

# ── Modes ───────────────────────────────────────────────────────────────────

case "$MODE" in
    latency)    run_one 1 inf ;;
    throughput) run_one "$BENCH_MAX_CONCURRENCY" inf ;;
    sweep|longcontext)
                for c in $BENCH_CONCURRENCY; do
                    for rep in $(seq 1 "$BENCH_REPEATS"); do run_one "$c" inf; done
                done ;;
esac

log "Done. Full output: $OUT_FILE"
if [[ "$BENCH_REPEATS" -le 1 ]]; then
    warn "Single run per point — no error bar. Set BENCH_REPEATS=3 before"
    warn "  concluding that two configs, or two nodes, actually differ."
fi
log "Compare runs with:  ls -t $BENCH_DIR"
