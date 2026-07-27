#!/usr/bin/env bash
#
# serve-kimik3.sh — one-click Kimi K3 serving on Bunya's AMD MI355X nodes
# (bun159/160/161, 8x gfx950) via vLLM or SGLang inside an Apptainer container.
#
# Usage:
#   ./serve-kimik3.sh [serve]    start the server (default; runs until killed)
#   ./serve-kimik3.sh serve --detach
#                                start the server, wait until healthy, then
#                                return the shell (server keeps running in the
#                                background for the life of the SLURM job) —
#                                use this to run opencode on the GPU node itself
#   ./serve-kimik3.sh pull       build the .sif from the container image (once)
#   ./serve-kimik3.sh download   prefetch model weights only (no GPU needed)
#   ./serve-kimik3.sh check      can this image load this model? (arch vs registry)
#   ./serve-kimik3.sh parsers    list tool-call/reasoning parsers this image has
#   ./serve-kimik3.sh stop       stop a running server
#   ./serve-kimik3.sh status     show server state + health endpoint
#
# Apptainer lives ONLY on Bunya compute nodes, never the login nodes — so every
# mode except a bare 'stop'/'status' must run inside a salloc/sbatch allocation.
#
# Configuration comes from kimik3.env next to this script (or $KIMIK3_ENV),
# see kimik3-env.example. Environment variables you export beforehand win.
#
# ENGINE=vllm|sglang selects the engine. The two differ only in the launch
# command and flag spellings; everything else (Apptainer, SLURM, caches, health,
# API key, banner) is shared.

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

ENGINE="${ENGINE:-vllm}"
case "$ENGINE" in
    vllm|sglang) ;;
    *) die "ENGINE must be 'vllm' or 'sglang', got '$ENGINE'." ;;
esac

MODEL_ID="${MODEL_ID:-amd/Kimi-K2.6-MXFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k3}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0}"
SGLANG_IMAGE="${SGLANG_IMAGE:-docker://lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260726}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
ENABLE_EP="${ENABLE_EP:-0}"
CONTEXT_LEN="${CONTEXT_LEN:-131072}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-kimi_k2}"
REASONING_PARSER="${REASONING_PARSER:-kimi_k2}"
ENABLE_AITER="${ENABLE_AITER:-1}"
SET_CPU_AFFINITY="${SET_CPU_AFFINITY:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-10800}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
SIF_PATH="${SIF_PATH:-}"

# Pick the image for the selected engine.
if [[ "$ENGINE" == "vllm" ]]; then
    ENGINE_IMAGE="$VLLM_IMAGE"
    SERVER_PATTERN='vllm serve'
else
    ENGINE_IMAGE="$SGLANG_IMAGE"
    SERVER_PATTERN='sglang.launch_server'
fi

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
        # Give the engine's worker processes a moment, then make sure they're gone.
        for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    else
        warn "No running server recorded in $PID_FILE."
    fi
    # Belt-and-braces: this node is a single-user 8-GPU grab, so clean up strays
    # from BOTH engines (you may have switched ENGINE since starting).
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    pkill -f 'vllm serve'           2>/dev/null || true
    pkill -f 'VLLM::EngineCore'     2>/dev/null || true
    pkill -f 'vllm.entrypoints'     2>/dev/null || true
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
            curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" \
                 -H "Authorization: Bearer ${KIMIK3_API_KEY:-$( [[ -r "${MODEL_CACHE_DIR:-}/kimik3-api-key" ]] && cat "$MODEL_CACHE_DIR/kimik3-api-key" || echo)}" \
                 2>/dev/null | head -c 400 || true
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

# Default the .sif next to the HF cache, named per engine so both can coexist
# and switching ENGINE doesn't force a re-pull.
SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/kimik3-${ENGINE}-mi355x.sif}"
# SIF_PATH must name a .sif FILE, not a directory. If it points at a directory
# (or ends with '/'), treat it as a folder and drop the default filename in —
# 'apptainer pull' otherwise refuses ("Image file already exists").
if [[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]]; then
    SIF_PATH="${SIF_PATH%/}/kimik3-${ENGINE}-mi355x.sif"
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
    log "Pulling $ENGINE_IMAGE -> $SIF_PATH (multi-GB; one-time) ..."
    apptainer pull "$SIF_PATH" "$ENGINE_IMAGE" \
        || die "apptainer pull failed. Does this node have outbound internet? Check APPTAINER_CACHEDIR space ($APPTAINER_CACHEDIR)."
    log "Image ready: $SIF_PATH"
    exit 0
fi

# For every remaining mode we need the .sif. Auto-build it if missing.
if [[ ! -f "$SIF_PATH" ]]; then
    log "No .sif at $SIF_PATH yet — pulling it now (one-time)."
    apptainer pull "$SIF_PATH" "$ENGINE_IMAGE" \
        || die "apptainer pull failed. Run './serve-kimik3.sh pull' explicitly to debug."
    log "Image ready: $SIF_PATH"
fi

# Resolve HF token: env var, then token file.
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
    [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi

# ── parsers mode: what can this image actually parse? ───────────────────────
# K3 will very likely register its own tool-call/reasoning parser names. Rather
# than guessing and getting an unhelpful argparse error at launch, ask the image.

if [[ "$MODE" == "parsers" ]]; then
    log "Parsers available in $SIF_PATH (ENGINE=$ENGINE):"
    if [[ "$ENGINE" == "vllm" ]]; then
        apptainer exec "$SIF_PATH" python3 - <<'PY' || warn "Could not introspect vLLM parser registries; falling back to --help below."
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    print("  tool-call parsers:", ", ".join(sorted(ToolParserManager.tool_parsers)))
except Exception as e:
    print("  tool-call parsers: <introspection failed:", e, ">")
try:
    from vllm.reasoning import ReasoningParserManager
    print("  reasoning parsers:", ", ".join(sorted(ReasoningParserManager.reasoning_parsers)))
except Exception as e:
    print("  reasoning parsers: <introspection failed:", e, ">")
PY
    else
        apptainer exec "$SIF_PATH" \
            python3 -m sglang.launch_server --help 2>&1 \
            | grep -A 6 -iE '\-\-(tool-call|reasoning)-parser' || true
    fi
    echo
    log "Set TOOL_PARSER / REASONING_PARSER in kimik3.env from this list ('none' to omit)."
    exit 0
fi

# ── check mode: can this image load this model? ─────────────────────────────
# The single most useful question on K3 flip day. Reads the model's
# 'architectures' from config.json and asks the engine's registry if it knows it.

if [[ "$MODE" == "check" ]]; then
    log "Engine:  $ENGINE"
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
        --env ENGINE="$ENGINE" \
        "$SIF_PATH" python3 - <<'PY' || rc=$?
import os, sys

model_id = os.environ["MODEL_ID"]
engine = os.environ["ENGINE"]

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    print(f"  config.json architectures : {archs}")
    for attr in ("num_hidden_layers", "num_experts", "num_experts_per_tok",
                 "max_position_embeddings", "quantization_config"):
        if hasattr(cfg, attr):
            v = getattr(cfg, attr)
            if attr == "quantization_config" and isinstance(v, dict):
                v = {k: v[k] for k in list(v)[:6]}
            print(f"  {attr:26}: {v}")
except Exception as e:
    print(f"  !! could not load config for {model_id}: {e}")
    print("     (not released yet, gated, or no network from this node?)")
    sys.exit(2)

print()
missing = []
if engine == "vllm":
    from vllm.model_executor.models.registry import ModelRegistry
    known = set(ModelRegistry.get_supported_archs())
else:
    from sglang.srt.models.registry import ModelRegistry  # type: ignore
    known = set(getattr(ModelRegistry, "models", {}) or {})
    if not known:
        from sglang.srt.models import registry as _r
        known = set(getattr(_r, "ModelRegistry").models)

for a in archs:
    if a in known:
        print(f"  OK   {engine} registry knows '{a}'")
    else:
        print(f"  MISS {engine} registry does NOT know '{a}'")
        missing.append(a)

print(f"\n  ({len(known)} architectures registered in this image)")
kimi = sorted(a for a in known if "kimi" in a.lower() or "moonshot" in a.lower())
print(f"  Kimi-family architectures present: {kimi or '<none>'}")

if missing:
    print("\n  => This image CANNOT serve this model. Use a newer image "
          "(./check-k3-readiness.sh prints current candidate tags).")
    sys.exit(1)
print("\n  => This image can load this model's architecture.")
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

if [[ -z "${HF_TOKEN:-}" && "$weights_cached" -eq 0 ]]; then
    die "No HF_TOKEN / HF_TOKEN_FILE set and weights for $MODEL_ID are not cached yet in $MODEL_CACHE_DIR."
fi

# Rough download size, used only for the free-space warning.
# (Measured: amd/Kimi-K2.6-MXFP4 is 559 GB — only experts/shared_experts are
# FP4, attention and embeddings stay higher precision, so it is roughly double a
# naive params x 4.25 bits estimate. The K3 figure scales from that ratio.)
case "$MODEL_ID" in
    *Kimi-K3*|*Kimi-K3-MXFP4*) est_gb=1600; need_gb=1700 ;;
    *K2.6*|*K2.5*)             est_gb=560;  need_gb=650  ;;
    *)                         est_gb=0;    need_gb=0    ;;
esac

if [[ "$weights_cached" -eq 0 ]]; then
    if [[ "$need_gb" -gt 0 ]]; then
        free_gb="$(df -Pk "$MODEL_CACHE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')"
        if [[ "${free_gb:-0}" -lt "$need_gb" ]]; then
            warn "Only ${free_gb} GB free in $MODEL_CACHE_DIR; $MODEL_ID needs ~${est_gb} GB (${need_gb} GB recommended). Download may fail."
        fi
        log "Weights not cached — first start downloads ~${est_gb} GB. Consider './serve-kimik3.sh download' first."
    else
        log "Weights for $MODEL_ID not cached yet — they'll download on first start."
    fi
else
    log "Found cached weights for $MODEL_ID."
fi

# ── Download mode (no GPU required) ─────────────────────────────────────────

if [[ "$MODE" == "download" ]]; then
    log "Prefetching $MODEL_ID into $MODEL_CACHE_DIR (no GPU required) ..."
    apptainer exec \
        --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env HF_HUB_ENABLE_HF_TRANSFER=1 \
        "$SIF_PATH" \
        bash -c "hf download '$MODEL_ID' || huggingface-cli download '$MODEL_ID'" \
        || die "Weights download failed. Re-run to resume."
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
        gfx942) warn "Detected gfx942 (MI300X/MI325X). This repo targets MI355X/MXFP4, which gfx942 cannot run.
        Switch to an FP8 model and an mi30x image — see the README 'Running on MI300X/MI325X' section." ;;
        "")     warn "Could not detect GPU arch from rocminfo." ;;
        *)      warn "Detected $gfx — this recipe is tuned for gfx950 (MI355X)." ;;
    esac
fi

# ── Parallelism layout ──────────────────────────────────────────────────────
# The two engines mean different things by DP, so validate per engine.
#
#   SGLang: --tp IS the total GPU count. --dp (with --enable-dp-attention) only
#           SUBDIVIDES those GPUs for attention, so DP must divide TP.
#   vLLM:   --data-parallel-size MULTIPLIES: total GPUs = DP * TP.

if [[ "$ENGINE" == "vllm" ]]; then
    GPUS_USED=$(( TP_SIZE * DP_SIZE ))
else
    GPUS_USED="$TP_SIZE"
    if [[ "${DP_SIZE:-1}" -gt 1 && $(( TP_SIZE % DP_SIZE )) -ne 0 ]]; then
        die "SGLang: DP_SIZE=$DP_SIZE must divide TP_SIZE=$TP_SIZE (dp-attention splits the $TP_SIZE GPUs into $DP_SIZE groups)."
    fi
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
    if [[ "$ENGINE" == "vllm" ]]; then
        die "vLLM needs DP_SIZE * TP_SIZE to equal the allocation: ${DP_SIZE} * ${TP_SIZE} = ${GPUS_USED}, but ${alloc_count} GPUs are allocated. Use TP_SIZE=${alloc_count} DP_SIZE=1 (or e.g. TP_SIZE=4 DP_SIZE=2 on 8 GPUs)."
    fi
    warn "TP_SIZE=$TP_SIZE uses $GPUS_USED GPU(s) but $alloc_count are allocated — you'd leave $((alloc_count - GPUS_USED)) idle (or over-subscribe). Set TP_SIZE=$alloc_count to use them all. (An MI355X node = 8 GPUs -> TP_SIZE=8.)"
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

# ── FlyDSL JIT cache bind ───────────────────────────────────────────────────
# The MXFP4 MoE JIT-compiles FlyDSL kernels and caches them INSIDE the image —
# but an Apptainer .sif is read-only, so that write fails ("Read-only file
# system"). Bind a writable scratch dir over that path (and keep the compiled
# kernels across runs). aiter lives in a different place in each image, so ask
# the image where it is rather than hardcoding a path.

FLYDSL_CACHE_DIR="${FLYDSL_CACHE_DIR:-$MODEL_CACHE_DIR/flydsl-cache}"
mkdir -p "$FLYDSL_CACHE_DIR" 2>/dev/null || true

if [[ -z "${FLYDSL_CACHE_TARGET:-}" ]]; then
    FLYDSL_CACHE_TARGET="$(apptainer exec "$SIF_PATH" python3 -c \
        'import aiter, os; print(os.path.join(os.path.dirname(aiter.__file__), "jit", "flydsl_cache"))' \
        2>/dev/null || true)"
fi

cache_bind=()
if [[ -n "$FLYDSL_CACHE_TARGET" ]]; then
    cache_bind=(--bind "$FLYDSL_CACHE_DIR":"$FLYDSL_CACHE_TARGET")
    log "FlyDSL JIT cache: $FLYDSL_CACHE_DIR -> $FLYDSL_CACHE_TARGET"
else
    warn "Could not locate aiter in the image, so the FlyDSL JIT cache is NOT bound."
    warn "  If startup dies with \"Read-only file system: .../flydsl_cache/...\", set"
    warn "  FLYDSL_CACHE_TARGET in kimik3.env to the path from that error message."
fi

# ── Build the engine command ────────────────────────────────────────────────

# shellcheck disable=SC2206  # intentional word splitting of user-provided extra args
extra_args=($EXTRA_ENGINE_ARGS)

engine_env=()
cmd=()

if [[ "$ENGINE" == "vllm" ]]; then
    kv_flag=()
    case "$KV_CACHE_DTYPE" in
        ""|auto) ;;
        fp8|fp8_e4m3) kv_flag=(--kv-cache-dtype fp8) ;;
        *)            kv_flag=(--kv-cache-dtype "$KV_CACHE_DTYPE") ;;
    esac

    par_flags=(--tensor-parallel-size "$TP_SIZE")
    [[ "${DP_SIZE:-1}" -gt 1 ]] && par_flags+=(--data-parallel-size "$DP_SIZE")
    [[ "$ENABLE_EP" == "1" ]]   && par_flags+=(--enable-expert-parallel)

    parser_flags=()
    [[ "$TOOL_PARSER" != "none" && -n "$TOOL_PARSER" ]] \
        && parser_flags+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
    [[ "$REASONING_PARSER" != "none" && -n "$REASONING_PARSER" ]] \
        && parser_flags+=(--reasoning-parser "$REASONING_PARSER")

    perf_flags=()
    [[ -n "$CHUNKED_PREFILL_SIZE" ]] && perf_flags+=(--max-num-batched-tokens "$CHUNKED_PREFILL_SIZE")
    [[ -n "$MAX_RUNNING_REQUESTS" ]] && perf_flags+=(--max-num-seqs "$MAX_RUNNING_REQUESTS")
    [[ -n "$CUDA_GRAPH_MAX_BS"    ]] && perf_flags+=(--cuda-graph-sizes "$CUDA_GRAPH_MAX_BS")
    [[ -n "$SCHEDULE_POLICY"      ]] && warn "SCHEDULE_POLICY is SGLang-only and is ignored with ENGINE=vllm (vLLM enables prefix caching by default)."

    if [[ "$ENABLE_AITER" == "1" ]]; then
        engine_env+=(--env VLLM_ROCM_USE_AITER=1 --env VLLM_ROCM_USE_AITER_MOE=1)
    else
        engine_env+=(--env VLLM_ROCM_USE_AITER=0)
    fi
    # vLLM's own CPU pinning is off unless asked for; leave VLLM_CPU_OMP_THREADS_BIND
    # unset so we don't repeat SGLang's SLURM-cgroup affinity crash.

    cmd=(vllm serve "$MODEL_ID"
         --served-model-name "$SERVED_MODEL_NAME"
         "${par_flags[@]}"
         --host 0.0.0.0 --port "$PORT"
         --max-model-len "$CONTEXT_LEN"
         --gpu-memory-utilization "$MEM_FRACTION"
         ${kv_flag[@]+"${kv_flag[@]}"}
         --api-key "$KIMIK3_API_KEY"
         --trust-remote-code
         ${parser_flags[@]+"${parser_flags[@]}"}
         ${perf_flags[@]+"${perf_flags[@]}"}
         ${extra_args[@]+"${extra_args[@]}"})
else
    kv_flag=()
    case "$KV_CACHE_DTYPE" in
        ""|auto) ;;
        fp8|fp8_e4m3) kv_flag=(--kv-cache-dtype fp8_e4m3) ;;
        *)            kv_flag=(--kv-cache-dtype "$KV_CACHE_DTYPE") ;;
    esac

    par_flags=(--tp "$TP_SIZE")
    [[ "${DP_SIZE:-1}" -gt 1 ]] && par_flags+=(--dp "$DP_SIZE" --enable-dp-attention)
    [[ "$ENABLE_EP" == "1" ]]   && par_flags+=(--ep-size "$TP_SIZE")

    parser_flags=()
    [[ "$TOOL_PARSER" != "none" && -n "$TOOL_PARSER" ]] \
        && parser_flags+=(--tool-call-parser "$TOOL_PARSER")
    [[ "$REASONING_PARSER" != "none" && -n "$REASONING_PARSER" ]] \
        && parser_flags+=(--reasoning-parser "$REASONING_PARSER")

    perf_flags=()
    [[ -n "$SCHEDULE_POLICY"      ]] && perf_flags+=(--schedule-policy "$SCHEDULE_POLICY")
    [[ -n "$CHUNKED_PREFILL_SIZE" ]] && perf_flags+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
    [[ -n "$MAX_RUNNING_REQUESTS" ]] && perf_flags+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
    [[ -n "$CUDA_GRAPH_MAX_BS"    ]] && perf_flags+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
    [[ "$ENABLE_AITER" == "1"     ]] && perf_flags+=(--enable-aiter-allreduce-fusion)

    engine_env+=(--env "SGLANG_SET_CPU_AFFINITY=$SET_CPU_AFFINITY")

    cmd=(python3 -m sglang.launch_server
         --model-path "$MODEL_ID"
         --served-model-name "$SERVED_MODEL_NAME"
         "${par_flags[@]}"
         --host 0.0.0.0 --port "$PORT"
         --context-length "$CONTEXT_LEN"
         --mem-fraction-static "$MEM_FRACTION"
         ${kv_flag[@]+"${kv_flag[@]}"}
         --api-key "$KIMIK3_API_KEY"
         --trust-remote-code
         ${parser_flags[@]+"${parser_flags[@]}"}
         ${perf_flags[@]+"${perf_flags[@]}"}
         ${extra_args[@]+"${extra_args[@]}"})
fi

# Forward the SLURM GPU-visibility vars into the container (Apptainer inherits
# host env by default, but be explicit) so ROCm sees exactly the allocated GPUs.
gpu_env=()
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && gpu_env+=(--env "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES")
[[ -n "${HIP_VISIBLE_DEVICES:-}"  ]] && gpu_env+=(--env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && gpu_env+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")

# ── Launch ──────────────────────────────────────────────────────────────────

log "Starting $ENGINE: $MODEL_ID  (TP=$TP_SIZE DP=$DP_SIZE -> $GPUS_USED GPUs, ctx=$CONTEXT_LEN, port=$PORT)"
log "Image: $SIF_PATH"
log "Logs:  $LOG_FILE"

: > "$LOG_FILE"
{
    printf '### serve-kimik3.sh  %s\n' "$(date)"
    printf '### engine=%s image=%s\n' "$ENGINE" "$ENGINE_IMAGE"
    printf '### command: %s\n\n' "${cmd[*]}"
} >> "$LOG_FILE"

apptainer exec --rocm \
    --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
    ${cache_bind[@]+"${cache_bind[@]}"} \
    --env HF_HOME="$MODEL_CACHE_DIR" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    ${engine_env[@]+"${engine_env[@]}"} \
    ${gpu_env[@]+"${gpu_env[@]}"} \
    "$SIF_PATH" \
    "${cmd[@]}" \
    >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

cleanup() {
    log "Shutting down server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -f "$SERVER_PATTERN" 2>/dev/null || true
    [[ "$ENGINE" == "vllm" ]] && pkill -f 'VLLM::EngineCore' 2>/dev/null || true
    rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Wait for readiness ──────────────────────────────────────────────────────

log "Waiting for the server to become healthy (timeout ${READY_TIMEOUT}s; model load takes several minutes, first-run download much longer) ..."
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
     If it failed on an unknown architecture, run: ./serve-kimik3.sh check
     If it failed on an unknown parser name, run:  ./serve-kimik3.sh parsers"
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
  Engine:      $ENGINE   (TP=$TP_SIZE DP=$DP_SIZE, ctx=$CONTEXT_LEN)
  Node:        $NODE_HOST     (job $JOBID)
  Endpoint:    http://$NODE_HOST:$PORT/v1   (OpenAI-compatible)
  Model name:  $SERVED_MODEL_NAME
  API key:     $API_KEY_FILE
               export KIMIK3_API_KEY="\$(cat $API_KEY_FILE)"

  Smoke test (from this node):
    curl -s http://127.0.0.1:$PORT/v1/models \\
         -H "Authorization: Bearer \$KIMIK3_API_KEY"

  Second shell into this job (to run opencode alongside the server):
    srun --overlap --jobid $JOBID --pty /bin/bash -l

  From your laptop (tunnel through the login node, then use localhost):
    ssh -N -L $PORT:$NODE_HOST:$PORT \${USER}@bunya1.rcc.uq.edu.au

  opencode: run ./opencode-setup.sh --host $NODE_HOST --port $PORT
            (or --host localhost when tunnelling), then pick
            '$SERVED_MODEL_NAME' via /models inside opencode.

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
