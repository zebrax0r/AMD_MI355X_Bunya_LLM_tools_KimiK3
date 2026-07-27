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
#
# Tunables (env or kimik3.env): BENCH_INPUT_LEN, BENCH_OUTPUT_LEN, BENCH_NUM_PROMPTS,
#   BENCH_CONCURRENCY (space-separated list for sweep), BENCH_MAX_CONCURRENCY,
#   BENCH_EXTRA_ARGS (appended verbatim to sglang.bench_serving).
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
        latency|throughput|sweep) MODE="$arg" ;;
        *) printf 'Unknown argument: %s (use latency | throughput | sweep)\n' "$arg" >&2; exit 1 ;;
    esac
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

# Benchmark parameters (overridable).
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-2 8 32}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

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
    echo "# mode=$MODE model=$SERVED_MODEL_NAME port=$PORT"
    echo "# MODEL_ID=${MODEL_ID:-?}  IMAGE=${SGLANG_IMAGE:-?}"
    echo "# in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN num_prompts=$BENCH_NUM_PROMPTS"
    echo "# TP_SIZE=${TP_SIZE:-?} DP_SIZE=${DP_SIZE:-?} MEM_FRACTION=${MEM_FRACTION:-?} CONTEXT_LEN=${CONTEXT_LEN:-modelmax}"
    echo "# SPECULATIVE=${SPECULATIVE:-off} DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-?} ENABLE_AITER=${ENABLE_AITER:-?}"
    echo "# upstream MI355 TP8 ref: baseline 820/2356/4898 tok/s, DSpark 1659/2927/3715 @ c=2/8/32"
    echo
} > "$OUT_FILE"
log "Saving results to $OUT_FILE"

# ── One benchmark run ───────────────────────────────────────────────────────
# $1 = max concurrency, $2 = request rate ("inf" to saturate)

run_one() {
    local conc="$1" rate="$2"
    log "Run: concurrency=$conc request-rate=$rate  (in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN n=$BENCH_NUM_PROMPTS)"
    echo "=== concurrency=$conc request-rate=$rate ===" >> "$OUT_FILE"

    # shellcheck disable=SC2086  # intentional splitting of BENCH_EXTRA_ARGS
    apptainer exec \
        --env "OPENAI_API_KEY=${KIMIK3_API_KEY:-}" \
        "$SIF_PATH" \
        python3 -m sglang.bench_serving \
            --backend sglang-oai \
            --base-url "http://127.0.0.1:${PORT}" \
            --model "$SERVED_MODEL_NAME" \
            --dataset-name random \
            --random-input-len "$BENCH_INPUT_LEN" \
            --random-output-len "$BENCH_OUTPUT_LEN" \
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
    sweep)      for c in $BENCH_CONCURRENCY; do run_one "$c" inf; done ;;
esac

log "Done. Full output: $OUT_FILE"
log "Compare runs with:  ls -t $BENCH_DIR"
