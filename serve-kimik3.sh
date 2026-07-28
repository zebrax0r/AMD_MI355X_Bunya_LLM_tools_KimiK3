#!/usr/bin/env bash
#
# serve-kimik3.sh — one-click Kimi K3 serving on Bunya's AMD MI355X nodes
# (bun159/160/161, 8x gfx950) via SGLang inside an Apptainer container.
#
# Reproduces SGLang's validated day-0 8x MI355X TP8 recipe
# (sgl-project/sglang issue #32548), with three deliberate deviations for our
# environment — see the README "Our deviations from upstream":
#   1. binds 0.0.0.0 (not 127.0.0.1) so SSH tunnels and off-node opencode work
#   2. adds --api-key and --served-model-name
#   3. forces SGLANG_SET_CPU_AFFINITY=0 (SLURM cgroup; upstream has none)
#
# Usage:
#   ./serve-kimik3.sh [serve]    start the server (default; runs until killed)
#   ./serve-kimik3.sh serve --detach
#                                start the server, wait until healthy, then
#                                return the shell (server keeps running in the
#                                background for the life of the SLURM job) —
#                                use this to run opencode on the GPU node itself
#   ./serve-kimik3.sh pull       build the .sif from the container image (once)
#   ./serve-kimik3.sh check      can this image load this model? (arch vs registry)
#   ./serve-kimik3.sh parsers    list tool-call/reasoning parsers this image has
#   ./serve-kimik3.sh download   prefetch model weights only (no GPU needed)
#   ./serve-kimik3.sh stop       stop a running server
#   ./serve-kimik3.sh status     show server state + health endpoint
#
# Run 'check' BEFORE 'download' — it is the cheap gate on a 1.5 TB commitment.
#
# Apptainer lives ONLY on Bunya compute nodes, never the login nodes — so every
# mode except a bare 'stop'/'status' must run inside a salloc/sbatch allocation.
#
# Configuration comes from kimik3.env next to this script (or $KIMIK3_ENV),
# see kimik3-env.example. Environment variables you export beforehand win.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="serve"
DETACH=0
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *)           MODE="$arg" ;;
    esac
done

log()  { printf '\033[1;34m[kimik3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[kimik3 WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[kimik3 ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config ─────────────────────────────────────────────────────────────

ENV_FILE="${KIMIK3_ENV:-$SCRIPT_DIR/kimik3.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    log "Loaded config from $ENV_FILE"
else
    warn "No config file at $ENV_FILE (copy kimik3-env.example to kimik3.env); using environment only."
fi

MODEL_ID="${MODEL_ID:-moonshotai/Kimi-K3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k3}"
SGLANG_IMAGE="${SGLANG_IMAGE:-docker://lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-256}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
TOOL_PARSER="${TOOL_PARSER:-kimi_k3}"
REASONING_PARSER="${REASONING_PARSER:-kimi_k3}"
SPECULATIVE="${SPECULATIVE:-}"
DSPARK_MODEL="${DSPARK_MODEL:-RadixArk/Kimi-K3-DSpark}"
ENABLE_AITER="${ENABLE_AITER:-1}"
FLYDSL_FORCE="${FLYDSL_FORCE:-1}"
SET_CPU_AFFINITY="${SET_CPU_AFFINITY:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-14400}"
LAUNCH_CMD="${LAUNCH_CMD:-sglang serve}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-}"
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
SIF_PATH="${SIF_PATH:-}"

case "$SPECULATIVE" in
    ""|none|dspark) ;;
    *) die "SPECULATIVE must be empty or 'dspark', got '$SPECULATIVE'." ;;
esac

# Runtime state (PID + log) lives under MODEL_CACHE_DIR so it survives detach
# and is reachable by 'stop'/'status' from any shell in the allocation.
if [[ -n "$MODEL_CACHE_DIR" ]]; then
    PID_FILE="${PID_FILE:-$MODEL_CACHE_DIR/kimik3-server.pid}"
    LOG_FILE="${LOG_FILE:-$MODEL_CACHE_DIR/kimik3-server.log}"
else
    PID_FILE="${PID_FILE:-$SCRIPT_DIR/kimik3-server.pid}"
    LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/kimik3-server.log}"
fi

server_running() { [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; }

stop_server() {
    if server_running; then
        local pid; pid="$(<"$PID_FILE")"
        log "Stopping server (pid $pid) ..."
        kill "$pid" 2>/dev/null || true
        # Give SGLang's worker processes a moment, then make sure they're gone.
        for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    else
        warn "No running server recorded in $PID_FILE."
    fi
    # Belt-and-braces: this node is a single-user 8-GPU grab, so clean up strays
    # from either invocation form.
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    pkill -f 'sglang serve'         2>/dev/null || true
    rm -f "$PID_FILE"
}

# ── Simple modes first ──────────────────────────────────────────────────────

case "$MODE" in
    stop)
        stop_server
        log "Stopped."
        exit 0
        ;;
    status)
        if server_running; then
            log "Server process alive (pid $(<"$PID_FILE"))."
        else
            log "No server process recorded (pidfile $PID_FILE)."
        fi
        if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            log "Health check on port $PORT: OK"
            key=""
            [[ -r "${MODEL_CACHE_DIR:-}/kimik3-api-key" ]] && key="$(<"$MODEL_CACHE_DIR/kimik3-api-key")"
            curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" \
                 -H "Authorization: Bearer ${KIMIK3_API_KEY:-$key}" 2>/dev/null | head -c 400 || true
            echo
        else
            warn "http://127.0.0.1:${PORT}/health not responding (not started, or still loading)."
            [[ -f "$LOG_FILE" ]] && log "Follow progress with: tail -f $LOG_FILE"
        fi
        exit 0
        ;;
    serve|pull|download|check|parsers) ;;
    *)
        die "Unknown mode '$MODE'. Use: serve | pull | download | check | parsers | stop | status"
        ;;
esac

# ── Preflight (shared) ──────────────────────────────────────────────────────

command -v apptainer >/dev/null 2>&1 \
    || die "apptainer not found — run this INSIDE a compute-node allocation (salloc/sbatch). Apptainer is not installed on Bunya login nodes."
command -v curl >/dev/null 2>&1 || die "curl not found on PATH."

[[ -n "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR is not set. Point it at scratch (e.g. /scratch/user/\$USER/kimik3/hf-cache). See kimik3-env.example."
mkdir -p "$MODEL_CACHE_DIR" 2>/dev/null || true
[[ -d "$MODEL_CACHE_DIR" && -w "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR '$MODEL_CACHE_DIR' does not exist or is not writable."

SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/kimik3-mi355x.sif}"
# SIF_PATH must name a .sif FILE, not a directory. If it points at a directory
# (or ends with '/'), treat it as a folder and drop the default filename in —
# 'apptainer pull' otherwise refuses ("Image file already exists").
if [[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]]; then
    SIF_PATH="${SIF_PATH%/}/kimik3-mi355x.sif"
    log "SIF_PATH was a directory — using $SIF_PATH"
fi

# Keep Apptainer's cache and scratch off /home (which has a tight quota) — point
# them at scratch. Set both APPTAINER_* and the SINGULARITY_* aliases.
: "${APPTAINER_CACHEDIR:=$MODEL_CACHE_DIR/../apptainer-cache}"
: "${APPTAINER_TMPDIR:=$MODEL_CACHE_DIR/../apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" 2>/dev/null || true
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

# ── Pull mode: build the .sif from the container image ──────────────────────

if [[ "$MODE" == "pull" ]]; then
    if [[ -f "$SIF_PATH" ]]; then
        log "Image already present: $SIF_PATH (delete it to re-pull)."
        exit 0
    fi
    log "Pulling $SGLANG_IMAGE -> $SIF_PATH (multi-GB; one-time) ..."
    apptainer pull "$SIF_PATH" "$SGLANG_IMAGE" \
        || die "apptainer pull failed. Does this node have outbound internet? Check APPTAINER_CACHEDIR space ($APPTAINER_CACHEDIR).
     Note the default image is built from SGLang's unmerged 'kimi-k3' branch and
     is pinned to a dated tag — if it has been removed, see the README
     'When the branch merges' for how to pick a current one."
    log "Image ready: $SIF_PATH"
    exit 0
fi

# For every remaining mode we need the .sif. Auto-build it if missing.
if [[ ! -f "$SIF_PATH" ]]; then
    log "No .sif at $SIF_PATH yet — pulling it now (one-time)."
    apptainer pull "$SIF_PATH" "$SGLANG_IMAGE" \
        || die "apptainer pull failed. Run './serve-kimik3.sh pull' explicitly to debug."
    log "Image ready: $SIF_PATH"
fi

# Resolve HF token: env var, then token file.
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
    [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi

# ── parsers mode: what can this image actually parse? ───────────────────────

if [[ "$MODE" == "parsers" ]]; then
    log "Parsers available in $SIF_PATH:"
    apptainer exec "$SIF_PATH" \
        bash -c "sglang serve --help 2>&1 || python3 -m sglang.launch_server --help 2>&1" \
        | grep -A 6 -iE '\-\-(tool-call|reasoning)-parser' || \
        warn "Could not read parser choices from --help."
    echo
    log "K3 ships its own parsers (kimi_k3), NOT kimi_k2."
    log "Set TOOL_PARSER / REASONING_PARSER in kimik3.env from this list ('none' to omit)."
    exit 0
fi

# ── check mode: can this image load this model? ─────────────────────────────
# The cheap gate before committing to a ~1.5 TB download.

if [[ "$MODE" == "check" ]]; then
    log "Image:   $SIF_PATH"
    log "Model:   $MODEL_ID"
    echo
    # Don't let 'set -e' swallow the exit code: 1 = arch unsupported,
    # 2 = config unreadable. Both are meaningful answers, not script failures.
    rc=0
    apptainer exec \
        --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env MODEL_ID="$MODEL_ID" \
        "$SIF_PATH" python3 - <<'PY' || rc=$?
import os, sys

model_id = os.environ["MODEL_ID"]

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    print(f"  config.json architectures : {archs}")
    text = getattr(cfg, "text_config", None)
    src = text if text is not None else cfg
    for attr in ("num_hidden_layers", "num_experts", "num_experts_per_tok",
                 "max_position_embeddings", "quantization_config"):
        if hasattr(src, attr):
            v = getattr(src, attr)
            if attr == "quantization_config" and isinstance(v, dict):
                grp = (v.get("config_groups") or {}).get("group_0", {})
                v = grp.get("format", v.get("quant_method", "?"))
            print(f"  {attr:26}: {v}")
except Exception as e:
    print(f"  !! could not load config for {model_id}: {e}")
    print("     (gated, or no network from this node?)")
    sys.exit(2)

print()
from sglang.srt.models.registry import ModelRegistry  # type: ignore
known = set(getattr(ModelRegistry, "models", {}) or {})

missing = []
for a in archs:
    if a in known:
        print(f"  OK   SGLang registry knows '{a}'")
    else:
        print(f"  MISS SGLang registry does NOT know '{a}'")
        missing.append(a)

print(f"\n  ({len(known)} architectures registered in this image)")
kimi = sorted(a for a in known if "kimi" in a.lower())
print(f"  Kimi-family architectures present: {kimi or '<none>'}")

if missing:
    print("\n  => This image CANNOT serve this model. It is probably built from "
          "main rather\n     than the 'kimi-k3' branch — see the README "
          "'When the branch merges'.")
    sys.exit(1)
print("\n  => This image can load this model. Safe to download the weights.")
PY
    case "$rc" in
        0) log "check passed." ;;
        1) warn "check FAILED: this image cannot serve $MODEL_ID." ;;
        2) warn "check INCONCLUSIVE: could not read the model config." ;;
        *) warn "check exited with status $rc." ;;
    esac
    exit "$rc"
fi

# ── Weights cache accounting ────────────────────────────────────────────────

# HF hub layout: models--org--name
weights_dir="$MODEL_CACHE_DIR/hub/models--${MODEL_ID//\//--}"
weights_cached=0
[[ -d "$weights_dir/snapshots" ]] && weights_cached=1

# Measured from moonshotai/Kimi-K3's safetensors index.
EST_GB=1561
NEED_GB=1700

if [[ "$weights_cached" -eq 0 ]]; then
    [[ -n "${HF_TOKEN:-}" ]] \
        || die "No HF_TOKEN / HF_TOKEN_FILE set and weights for $MODEL_ID are not cached yet in $MODEL_CACHE_DIR."
    free_gb="$(df -Pk "$MODEL_CACHE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')"
    if [[ "${free_gb:-0}" -lt "$NEED_GB" ]]; then
        warn "Only ${free_gb} GB free in $MODEL_CACHE_DIR; $MODEL_ID is ~${EST_GB} GB (${NEED_GB} GB recommended with the .sif). Download will likely fail."
        warn "  Check your quota with 'rquota' before starting a multi-hour transfer."
    fi
    log "Weights not cached — first start downloads ~${EST_GB} GB. Run './serve-kimik3.sh download' first."
else
    log "Found cached weights for $MODEL_ID."
fi

# ── Download mode (no GPU required) ─────────────────────────────────────────

if [[ "$MODE" == "download" ]]; then
    log "Prefetching $MODEL_ID into $MODEL_CACHE_DIR (~${EST_GB} GB, no GPU required) ..."
    apptainer exec \
        --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env HF_HUB_ENABLE_HF_TRANSFER=1 \
        "$SIF_PATH" \
        bash -c "hf download '$MODEL_ID' || huggingface-cli download '$MODEL_ID'" \
        || die "Weights download failed. Re-run to resume."

    if [[ "$SPECULATIVE" == "dspark" ]]; then
        log "Prefetching the DSpark draft model $DSPARK_MODEL ..."
        apptainer exec \
            --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
            --env HF_HOME="$MODEL_CACHE_DIR" \
            --env HF_TOKEN="${HF_TOKEN:-}" \
            --env HF_HUB_ENABLE_HF_TRANSFER=1 \
            "$SIF_PATH" \
            bash -c "hf download '$DSPARK_MODEL' || huggingface-cli download '$DSPARK_MODEL'" \
            || die "DSpark draft download failed."
    fi
    log "Download complete."
    exit 0
fi

# ── Serve-mode preflight (GPU node checks) ──────────────────────────────────

[[ -e /dev/kfd ]] || die "/dev/kfd not found — is this a ROCm GPU node? (Apptainer + --rocm needs it.)"
[[ -e /dev/dri ]] || die "/dev/dri not found — is this a ROCm GPU node?"

if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]*' || true)"
    case "$gfx" in
        gfx950) log "Detected gfx950 (MI350X/MI355X) — matches the configured MXFP4 image/model." ;;
        gfx942) warn "Detected gfx942 (MI300X/MI325X). K3 needs ~1.5 TB and gfx950's native MXFP4 —
        this recipe will not work here. See the README 'Running on other hardware'." ;;
        "")     warn "Could not detect GPU arch from rocminfo." ;;
        *)      warn "Detected $gfx — this recipe is validated for gfx950 (MI355X)." ;;
    esac
fi

# --tp IS the total GPU count. --dp (with dp-attention) only subdivides those
# GPUs for attention — it does NOT multiply the count — so DP must divide TP.
GPUS_USED="$TP_SIZE"
if [[ "${DP_SIZE:-1}" -gt 1 && $(( TP_SIZE % DP_SIZE )) -ne 0 ]]; then
    die "DP_SIZE=$DP_SIZE must divide TP_SIZE=$TP_SIZE (dp-attention splits the $TP_SIZE GPUs into $DP_SIZE groups)."
fi

GPU_VIS="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
alloc_count=""
if [[ -n "$GPU_VIS" ]]; then
    alloc_count="$(awk -F, '{print NF}' <<<"$GPU_VIS")"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    alloc_count="$SLURM_GPUS_ON_NODE"
fi
[[ -n "$GPU_VIS" ]] && log "Allocated GPUs: [$GPU_VIS]"
if [[ -n "$alloc_count" && "$alloc_count" -gt 0 && "$GPUS_USED" -ne "$alloc_count" ]]; then
    warn "TP_SIZE=$TP_SIZE uses $GPUS_USED GPU(s) but $alloc_count are allocated. K3 needs all 8 (~1.5 TB of weights); set TP_SIZE=$alloc_count."
fi

# Port free?
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    exec 3>&- 3<&- || true
    die "Port $PORT is already in use on this node (another server running? try './serve-kimik3.sh status')."
fi

if server_running; then
    die "A server is already recorded running (pid $(<"$PID_FILE")). Use './serve-kimik3.sh status' or 'stop' first."
fi

# ── API key ─────────────────────────────────────────────────────────────────

API_KEY_FILE="$MODEL_CACHE_DIR/kimik3-api-key"
if [[ -z "${KIMIK3_API_KEY:-}" ]]; then
    if [[ -r "$API_KEY_FILE" ]]; then
        KIMIK3_API_KEY="$(<"$API_KEY_FILE")"
        log "Using API key from $API_KEY_FILE"
    else
        KIMIK3_API_KEY="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        (umask 077 && printf '%s' "$KIMIK3_API_KEY" > "$API_KEY_FILE")
        log "Generated new API key and saved it to $API_KEY_FILE"
    fi
fi

# ── aiter JIT directory bind ────────────────────────────────────────────────
# The MXFP4 MoE JIT-compiles FlyDSL kernels at CUDA-graph capture time and
# writes them INSIDE the image, under aiter/jit/. An Apptainer .sif is
# read-only, so that write dies with:
#
#   OSError: [Errno 30] Read-only file system:
#     '/sgl-workspace/aiter/aiter/jit/flydsl_cache/launch_hgemm_kernel_*/*.lock'
#
# ...and it dies *after* the ~1.5 TB weight load, which is an expensive way to
# find out. Fix: give the container a writable aiter/jit.
#
# We bind the whole jit/ directory, not just jit/flydsl_cache, because the
# image ships prebuilt modules there (module_rmsnorm_quant.so and friends) and
# aiter writes build artefacts beside the FlyDSL cache. Binding an empty dir
# over it would hide the prebuilt kernels, so seed the scratch copy from the
# image once, then bind it back at the SAME path (compiled artefacts can embed
# absolute paths) and reuse it on every later run.

AITER_JIT_DIR="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit}"

# Pre-fix configs set FLYDSL_CACHE_TARGET to the .../jit/flydsl_cache path.
# Honour it rather than silently ignoring it, but bind one level up.
if [[ -z "${AITER_JIT_TARGET:-}" && -n "${FLYDSL_CACHE_TARGET:-}" ]]; then
    AITER_JIT_TARGET="${FLYDSL_CACHE_TARGET%/flydsl_cache}"
    warn "FLYDSL_CACHE_TARGET is superseded by AITER_JIT_TARGET; using $AITER_JIT_TARGET"
fi

# Ask the image where aiter lives. Use find_spec, NOT 'import aiter': importing
# it needs a GPU (this exec has no --rocm) and prints '[aiter] import [...]'
# banners to stdout that would corrupt the captured path.
if [[ -z "${AITER_JIT_TARGET:-}" ]]; then
    AITER_JIT_TARGET="$(apptainer exec "$SIF_PATH" python3 - <<'PY' 2>/dev/null | tail -n 1
import importlib.util, os
spec = importlib.util.find_spec("aiter")
origin = getattr(spec, "origin", None) if spec is not None else None
print(os.path.join(os.path.dirname(origin), "jit") if origin else "")
PY
)"
fi

cache_bind=()
if [[ "${AITER_JIT_TARGET:-}" != /* ]]; then
    warn "Could not locate aiter's jit/ directory in the image (got: '${AITER_JIT_TARGET:-}')."
    warn "  Startup will likely die at CUDA-graph capture with"
    warn "  \"Read-only file system: .../aiter/jit/flydsl_cache/...\"."
    warn "  Set AITER_JIT_TARGET in kimik3.env to the jit/ directory from that path."
else
    mkdir -p "$AITER_JIT_DIR" || die "Cannot create $AITER_JIT_DIR"

    if [[ ! -f "$AITER_JIT_DIR/.seeded" ]]; then
        log "Seeding writable aiter JIT dir from the image (one-off copy) ..."
        apptainer exec --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" "$SIF_PATH" \
            cp -a "$AITER_JIT_TARGET/." "$AITER_JIT_DIR/" \
            || die "Failed to copy $AITER_JIT_TARGET out of the image into $AITER_JIT_DIR"
        touch "$AITER_JIT_DIR/.seeded"
        log "Seeded $AITER_JIT_DIR ($(du -sh "$AITER_JIT_DIR" 2>/dev/null | cut -f1 || echo '?'))"
    fi

    cache_bind=(--bind "$AITER_JIT_DIR":"$AITER_JIT_TARGET")
    log "aiter JIT dir: $AITER_JIT_DIR -> $AITER_JIT_TARGET"
fi
# The writability preflight lives just before the launch, so it can run against
# the exact argv the server gets — see "Preflight" below.

# ── Build the launch command ────────────────────────────────────────────────

# shellcheck disable=SC2206  # intentional word splitting of the configured launcher
launcher=($LAUNCH_CMD)
# shellcheck disable=SC2206  # intentional word splitting of user-provided extra args
extra_args=($EXTRA_ENGINE_ARGS)

dp_flag=()
[[ "${DP_SIZE:-1}" -gt 1 ]] && dp_flag=(--dp "$DP_SIZE" --enable-dp-attention)

ctx_flag=()
[[ -n "$CONTEXT_LEN" ]] && ctx_flag=(--context-length "$CONTEXT_LEN")

kv_flag=()
[[ -n "$KV_CACHE_DTYPE" ]] && kv_flag=(--kv-cache-dtype "$KV_CACHE_DTYPE")

radix_flag=()
[[ "$DISABLE_RADIX_CACHE" == "1" ]] && radix_flag=(--disable-radix-cache)

parser_flags=()
[[ "$TOOL_PARSER" != "none" && -n "$TOOL_PARSER" ]] \
    && parser_flags+=(--tool-call-parser "$TOOL_PARSER")
[[ "$REASONING_PARSER" != "none" && -n "$REASONING_PARSER" ]] \
    && parser_flags+=(--reasoning-parser "$REASONING_PARSER")

spec_flags=()
if [[ "$SPECULATIVE" == "dspark" ]]; then
    spec_flags=(--speculative-draft-model-path "$DSPARK_MODEL"
                --speculative-algorithm DSPARK)
    log "DSpark speculative decoding ENABLED (draft: $DSPARK_MODEL)."
    warn "  DSpark has an open crash report upstream (sglang issue #32569)."
    warn "  If startup fails with \"TypeError: 'NoneType' object is not callable\","
    warn "  unset SPECULATIVE in kimik3.env and retry the baseline config."
fi

perf_flags=()
[[ -n "$CHUNKED_PREFILL_SIZE" ]] && perf_flags+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
[[ -n "$MAX_RUNNING_REQUESTS" ]] && perf_flags+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
if [[ -n "$SCHEDULE_POLICY" ]]; then
    perf_flags+=(--schedule-policy "$SCHEDULE_POLICY")
    [[ "$DISABLE_RADIX_CACHE" == "1" ]] \
        && warn "SCHEDULE_POLICY=$SCHEDULE_POLICY has little effect while DISABLE_RADIX_CACHE=1 (prefix reuse is off)."
fi

# Upstream's four K3/AITER variables — this is how the K3 fused FP4 path is
# turned on. Note there is no --enable-aiter-allreduce-fusion in this recipe.
aiter_env=()
if [[ "$ENABLE_AITER" == "1" ]]; then
    aiter_env=(--env SGLANG_USE_AITER=1
               --env SGLANG_AITER_K3_OPT=1
               --env AITER_SITUV2_A8W4=1)
    # AITER_FLYDSL_FORCE is what routes gemms through the FlyDSL JIT compiler,
    # which writes into aiter/jit/flydsl_cache inside the read-only .sif. Set
    # FLYDSL_FORCE=0 to fall back to aiter's prebuilt gemm path: slower, but it
    # compiles nothing and so cannot hit the read-only failure.
    if [[ "$FLYDSL_FORCE" == "1" ]]; then
        aiter_env+=(--env AITER_FLYDSL_FORCE=1)
    else
        warn "FLYDSL_FORCE=0 — using aiter's prebuilt gemm path, below upstream's numbers."
    fi
fi

# Forward the SLURM GPU-visibility vars into the container (Apptainer inherits
# host env by default, but be explicit) so ROCm sees exactly the allocated GPUs.
gpu_env=()
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && gpu_env+=(--env "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES")
[[ -n "${HIP_VISIBLE_DEVICES:-}"  ]] && gpu_env+=(--env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && gpu_env+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")

cmd=("${launcher[@]}"
     --model-path "$MODEL_ID"
     --served-model-name "$SERVED_MODEL_NAME"
     --trust-remote-code
     --tp "$TP_SIZE"
     ${dp_flag[@]+"${dp_flag[@]}"}
     --attention-backend "$ATTENTION_BACKEND"
     --dtype "$MODEL_DTYPE"
     --mem-fraction-static "$MEM_FRACTION"
     --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
     --host 0.0.0.0 --port "$PORT"
     --api-key "$KIMIK3_API_KEY"
     ${radix_flag[@]+"${radix_flag[@]}"}
     ${ctx_flag[@]+"${ctx_flag[@]}"}
     ${kv_flag[@]+"${kv_flag[@]}"}
     ${parser_flags[@]+"${parser_flags[@]}"}
     ${spec_flags[@]+"${spec_flags[@]}"}
     ${perf_flags[@]+"${perf_flags[@]}"}
     ${extra_args[@]+"${extra_args[@]}"})

# ── Launch ──────────────────────────────────────────────────────────────────

log "Starting SGLang: $MODEL_ID  (TP=$TP_SIZE -> $GPUS_USED GPUs, ctx=${CONTEXT_LEN:-model max (1M)}, port=$PORT)"
log "Image: $SIF_PATH"
log "Logs:  $LOG_FILE"

# Assemble the container argv ONCE. Everything below — the preflight and the
# launch itself — uses this same array, so the configuration we test is by
# construction the configuration we run. (An earlier version tested a
# separately-built command; it passed while the real launch was missing a
# bind, which is a very expensive way to be wrong.)
apptainer_args=(exec --rocm
    --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR"
    ${cache_bind[@]+"${cache_bind[@]}"}
    --env HF_HOME="$MODEL_CACHE_DIR"
    --env HF_TOKEN="${HF_TOKEN:-}"
    --env HF_HUB_ENABLE_HF_TRANSFER=1
    --env SGLANG_SET_CPU_AFFINITY="$SET_CPU_AFFINITY"
    ${aiter_env[@]+"${aiter_env[@]}"}
    ${gpu_env[@]+"${gpu_env[@]}"})

# ── Preflight: aiter's JIT dir must be writable *as the server will see it* ──
# The FP4 MoE compiles FlyDSL kernels at CUDA-graph capture, which happens
# after the ~1.5 TB weight load. Catch a read-only path in seconds instead.
if [[ "${AITER_JIT_TARGET:-}" == /* ]]; then
    probe="$AITER_JIT_TARGET/flydsl_cache/.write-test.$$"
    if ! probe_err="$(apptainer "${apptainer_args[@]}" "$SIF_PATH" \
            sh -c "mkdir -p '$AITER_JIT_TARGET/flydsl_cache' \
                   && touch '$probe' && rm -f '$probe'" 2>&1)"; then
        die "aiter's JIT dir is NOT writable inside the container.
  Bind attempted : ${cache_bind[*]:-<none — detection failed>}
  Path tested    : $probe
  Container said : ${probe_err:-<no output>}
  Startup would die at CUDA-graph capture after the full weight load.
  Fix the bind, or set AITER_JIT_TARGET in kimik3.env to the jit/ directory
  shown in the 'Read-only file system' traceback."
    fi
    log "Preflight OK: $AITER_JIT_TARGET is writable in the container."
else
    warn "No aiter JIT bind — startup will likely die at CUDA-graph capture."
fi

: > "$LOG_FILE"
{
    printf '### serve-kimik3.sh  %s\n' "$(date)"
    printf '### image=%s\n' "$SGLANG_IMAGE"
    printf '### apptainer: apptainer %s %s\n' "${apptainer_args[*]}" "$SIF_PATH"
    printf '### command: %s\n\n' "${cmd[*]}"
} >> "$LOG_FILE"

apptainer "${apptainer_args[@]}" \
    "$SIF_PATH" \
    "${cmd[@]}" \
    >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

cleanup() {
    log "Shutting down server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    pkill -f 'sglang serve'         2>/dev/null || true
    rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Wait for readiness ──────────────────────────────────────────────────────

log "Waiting for the server to become healthy (timeout ${READY_TIMEOUT}s)."
log "A cold start reads ~1.5 TB off scratch and JIT-compiles FP4 kernels — be patient."
log "Follow detailed progress in another shell with: tail -f $LOG_FILE"

start_ts="$(date +%s)"
while true; do
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        tail -n 60 "$LOG_FILE" 2>&1 || true
        rm -f "$PID_FILE"
        die "Server process exited during startup. Last 60 log lines above ($LOG_FILE).
     Unknown architecture?   ./serve-kimik3.sh check
     Unknown parser name?    ./serve-kimik3.sh parsers
     KV cache OOM?           set CONTEXT_LEN=262144 in kimik3.env and retry
     DSpark TypeError?       unset SPECULATIVE (sglang issue #32569)"
    fi
    if (( $(date +%s) - start_ts > READY_TIMEOUT )); then
        die "Server did not become healthy within ${READY_TIMEOUT}s. Check: tail -f $LOG_FILE"
    fi
    sleep 10
done

# ── Connection banner ───────────────────────────────────────────────────────

NODE_HOST="$(hostname -s 2>/dev/null || hostname)"
JOBID="${SLURM_JOB_ID:-<jobid>}"
cat <<EOF

============================================================================
  $SERVED_MODEL_NAME is up and serving on Bunya's MI355X.

  Model:       $MODEL_ID
  Layout:      TP=$TP_SIZE, ctx=${CONTEXT_LEN:-model max (1M)}, spec=${SPECULATIVE:-off}
  Node:        $NODE_HOST     (job $JOBID)
  Endpoint:    http://$NODE_HOST:$PORT/v1   (OpenAI-compatible)
  Model name:  $SERVED_MODEL_NAME
  API key:     $API_KEY_FILE
               export KIMIK3_API_KEY="\$(cat $API_KEY_FILE)"

  Smoke test (from this node) — READ THE REPLY, coherence is the real test:
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \\
      -H "Authorization: Bearer \$KIMIK3_API_KEY" \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"$SERVED_MODEL_NAME","messages":[{"role":"user","content":"Say hello in one sentence."}]}'

  Second shell into this job (to run opencode alongside the server):
    srun --overlap --jobid $JOBID --pty /bin/bash -l

  From your laptop (tunnel through the login node, then use localhost):
    ssh -N -L $PORT:$NODE_HOST:$PORT \${USER}@bunya1.rcc.uq.edu.au

  opencode: run ./opencode-setup.sh --host $NODE_HOST --port $PORT
            (or --host localhost when tunnelling), then pick
            '$SERVED_MODEL_NAME' via /models inside opencode.

  Benchmark: ./bench-kimik3.sh sweep
             (upstream MI355 TP8 reference: 820 / 2356 / 4898 tok/s @ c=2/8/32)

  Stop with Ctrl-C, 'scancel $JOBID', or './serve-kimik3.sh stop'.
============================================================================

EOF

if [[ "$DETACH" -eq 1 ]]; then
    # Disarm the cleanup traps: the server keeps running in the background
    # (until './serve-kimik3.sh stop' or the SLURM job/allocation ends).
    trap - EXIT INT TERM
    disown "$SERVER_PID" 2>/dev/null || true
    log "Detached. You have your shell back — the server keeps running on this node."
    log "  Logs:  tail -f $LOG_FILE"
    log "  Stop:  ./serve-kimik3.sh stop"
    exit 0
fi

# Stay attached: keeps the SLURM job alive and tears the server down on
# Ctrl-C / scancel via the traps above.
log "Attached. Ctrl-C (or scancel) stops the server. Streaming logs:"
tail -f "$LOG_FILE" &
TAIL_PID=$!
# Wait on the server; when it exits, stop tailing.
wait "$SERVER_PID"
kill "$TAIL_PID" 2>/dev/null || true
