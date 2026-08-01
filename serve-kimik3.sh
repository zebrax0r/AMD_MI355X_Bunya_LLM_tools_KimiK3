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
#   ./serve-kimik3.sh gpucheck   can this image reach this node's GPUs? (~1 min)
#   ./serve-kimik3.sh toolcheck  two-turn tool-call round trip vs a running server
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
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
TOOL_PARSER="${TOOL_PARSER:-kimi_k3}"
REASONING_PARSER="${REASONING_PARSER:-kimi_k3}"
SPECULATIVE="${SPECULATIVE:-}"
DSPARK_MODEL="${DSPARK_MODEL:-RadixArk/Kimi-K3-DSpark}"
ENABLE_AITER="${ENABLE_AITER:-1}"
FLYDSL_FORCE="${FLYDSL_FORCE:-1}"
ROCM_MODE="${ROCM_MODE:-auto}"
AITER_GPU_ARCHS="${AITER_GPU_ARCHS:-}"
ROCMINFO_SHIM="${ROCMINFO_SHIM:-auto}"
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

case "$ROCM_MODE" in
    auto|rocm|devices) ;;
    *) die "ROCM_MODE must be auto, rocm or devices, got '$ROCM_MODE'." ;;
esac

case "$ROCMINFO_SHIM" in
    auto|off|force) ;;
    *) die "ROCMINFO_SHIM must be auto, off or force, got '$ROCMINFO_SHIM'." ;;
esac

# GPU_ARCHS is a build-time hint that aiter also consults at runtime, and a
# LIST makes it pick the first entry regardless of the actual device
# (ROCm/aiter#3807). One arch or nothing.
if [[ "$AITER_GPU_ARCHS" == *";"* || "$AITER_GPU_ARCHS" == *","* ]]; then
    die "AITER_GPU_ARCHS must name exactly one architecture (e.g. gfx950), got '$AITER_GPU_ARCHS'.
  A list makes aiter select the first entry at runtime even on other hardware
  — see https://github.com/ROCm/aiter/issues/3807."
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
    toolcheck)
        # Two-turn tool-call round trip against a RUNNING server. No .sif and no
        # GPU needed — it is pure HTTP, so it also works through a tunnel.
        #
        # Half the value is in turn 2. A model emitting a plausible tool_calls
        # object proves the parser serialises; it does not prove the loop
        # closes. So the tool returns a value the model cannot guess, and we
        # check that value appears in the final answer. opencode needs both.
        key=""
        [[ -r "${MODEL_CACHE_DIR:-}/kimik3-api-key" ]] && key="$(<"$MODEL_CACHE_DIR/kimik3-api-key")"
        BASE="${TOOLCHECK_URL:-http://127.0.0.1:$PORT}" \
        KEY="${KIMIK3_API_KEY:-$key}" \
        MODEL="$SERVED_MODEL_NAME" \
        python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error

BASE, KEY, MODEL = os.environ["BASE"], os.environ["KEY"], os.environ["MODEL"]
SECRET = 61.4          # unguessable: only the "tool" knows it
NODE   = "bun161"

TOOLS = [{"type": "function", "function": {
    "name": "get_gpu_temperature",
    "description": "Return the current GPU temperature in Celsius for a named Bunya compute node.",
    "parameters": {"type": "object",
                   "properties": {"node": {"type": "string",
                                           "description": "Node hostname, e.g. bun161"}},
                   "required": ["node"]}}}]

def post(payload):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:400]}")
        sys.exit(1)
    except Exception as e:
        print(f"  request failed: {e}\n  Is the server up? ./serve-kimik3.sh status")
        sys.exit(1)

fails = []
def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(label)

msgs = [{"role": "user",
         "content": f"What is the current GPU temperature on {NODE}? "
                    "Use the tool, then state the number."}]

print("Turn 1 — does the model emit a tool call?")
r1 = post({"model": MODEL, "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
           "max_tokens": 2048, "temperature": 0})
m1 = r1["choices"][0]["message"]
fin = r1["choices"][0]["finish_reason"]
calls = m1.get("tool_calls") or []

check(bool(calls), "model returned tool_calls",
      f"finish_reason={fin}" + ("" if calls else f", content={(m1.get('content') or '')[:120]!r}"))
if not calls:
    # A thinking model that ran out of budget mid-thought looks like a parser
    # failure but is not. Say which one it is.
    if fin == "length":
        print("\n  finish_reason=length: it never finished thinking. Raise max_tokens.")
    print("\n  If content contains a raw tool call as text, the tool parser is not")
    print("  matching this model. Check './serve-kimik3.sh parsers' and TOOL_PARSER.")
    sys.exit(1)

fn = calls[0].get("function", {})
check(fn.get("name") == "get_gpu_temperature", "correct function name", repr(fn.get("name")))

raw = fn.get("arguments")
try:
    args = json.loads(raw) if isinstance(raw, str) else raw
    ok_json = isinstance(args, dict)
except Exception as e:
    args, ok_json = None, False
    print(f"        arguments did not parse: {e}")
check(ok_json, "arguments are valid JSON", repr(raw)[:160])
check(bool(args) and NODE in str(args.get("node", "")), "argument value carried through",
      repr(args.get("node") if args else None))
check(bool(calls[0].get("id")), "tool_call has an id", repr(calls[0].get("id")))

print("\nTurn 2 — does the model use the tool result?")
msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls})
msgs.append({"role": "tool", "tool_call_id": calls[0].get("id"),
             "name": "get_gpu_temperature",
             "content": json.dumps({"node": NODE, "celsius": SECRET})})

r2 = post({"model": MODEL, "messages": msgs, "tools": TOOLS,
           "max_tokens": 2048, "temperature": 0})
m2 = r2["choices"][0]["message"]
final = (m2.get("content") or "").strip()
check(bool(final), "final answer has content",
      f"finish_reason={r2['choices'][0]['finish_reason']}")
check(str(SECRET) in final, f"final answer contains the tool's value ({SECRET})")
print(f"\n  final answer: {final[:300]}")

print()
if fails:
    print(f"TOOL-CALL ROUND TRIP FAILED: {len(fails)} check(s) — {', '.join(fails)}")
    sys.exit(1)
print("TOOL-CALL ROUND TRIP OK — opencode's agentic loop should work.")
PY
        exit $?
        ;;
    serve|pull|download|check|gpucheck|parsers) ;;
    *)
        die "Unknown mode '$MODE'. Use: serve | pull | download | check | gpucheck | toolcheck | parsers | stop | status"
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

# ── GPU passthrough: how does the container reach the GPUs? ─────────────────
#
# The image ships its own ROCm. The HOST's ROCm reaches into it through exactly
# three channels, and all three have broken a run here:
#
#   1. '--rocm' binds the host's ROCm libraries into /.singularity.d/libs and
#      PREPENDS that to LD_LIBRARY_PATH, so the container's binaries run against
#      the host's libhsa/libamdhip. Apptainer's docs are explicit that this
#      requires the two ROCm versions to be compatible. When RCC moved a node to
#      ROCm 7.14 under our 7.2 image, the container's OWN rocminfo started
#      exiting 1 and aiter died on import:
#        RuntimeError: Get GPU arch from rocminfo failed:
#          Command '['/opt/rocm-7.2.0/bin/rocminfo']' returned non-zero exit status 1
#   2. The kernel driver, via /dev/kfd. A KFD ioctl ABI break is not fixable
#      from here — it needs an image built for the node's ROCm.
#   3. Inherited environment: a *_VISIBLE_DEVICES value the container's ROCr
#      cannot parse (e.g. UUID form) takes down agent enumeration entirely.
#
# So don't assume: probe. 'devices' mode drops --rocm and passes the device
# nodes only, leaving the container's ROCm userspace intact end to end.

ROCM_MODES=(rocm devices)

# SLURM on a newer ROCm can hand out UUID-form device lists ("GPU-a1b2..."). An
# older ROCr in the container cannot parse those, and an unparseable value takes
# down agent enumeration for the WHOLE container — which looks exactly like a
# broken driver. Only forward index-form lists.
visible_devices_ok() { [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; }

# Identity of the .sif, cheap. Used both to key the cached passthrough mode and
# to detect that the seeded aiter jit/ dir came from a different image.
sif_stamp() {
    stat -c '%s:%Y' "$SIF_PATH" 2>/dev/null \
        || stat -f '%z:%m' "$SIF_PATH" 2>/dev/null \
        || echo '?'
}

# Probing costs ~a minute, so remember the answer. Key it to the node AND the
# image: either one changing invalidates the result.
ROCM_MODE_CACHE="$MODEL_CACHE_DIR/.rocm-mode"
rocm_cache_key() { printf '%s|%s' "$(hostname -s 2>/dev/null || echo node)" "$(sif_stamp)"; }

# ── rocminfo shim ───────────────────────────────────────────────────────────
#
# On bun161 (node ROCm 7.14, image ROCm 7.2) the container's rocminfo loads,
# reads the driver version, then fails:
#
#   ROCk module version 6.19.14.31400000 is loaded
#   hsa api call failure at: .../rocminfo.cc:357
#   Call returned HSA_STATUS_ERROR_INVALID_ARGUMENT
#
# ...while torch sees all 8 GPUs, because PyTorch's ROCm wheel bundles its own
# HIP runtime. So the image's ROCm *tools* are broken on this node and its
# *runtime* is fine. aiter only shells out to rocminfo to read the architecture
# out of its text — and GPU_ARCHS is not honoured by this aiter build — so give
# it text that is correct: a snapshot of the HOST's working rocminfo, replayed
# by a one-line script bound over the container's binary.
#
# This is exact rather than synthesised: it is real output for this node, so
# whatever aiter's parser expects, it gets. Nothing links against it and no
# host libraries are involved.

make_rocminfo_shim() {   # -> sets global shim_args
    shim_args=()
    [[ "$ROCMINFO_SHIM" == "off" ]] && return 0

    local target snap shim
    target="$(apptainer exec "$SIF_PATH" \
        sh -c 'ls -d /opt/rocm*/bin/rocminfo 2>/dev/null | head -n1' 2>/dev/null || true)"
    if [[ -z "$target" ]]; then
        [[ "$ROCMINFO_SHIM" == "force" ]] \
            && warn "ROCMINFO_SHIM=force but no rocminfo found in the image — skipping."
        return 0
    fi

    # Only step in when the container's own rocminfo is actually broken.
    if [[ "$ROCMINFO_SHIM" != "force" ]] \
       && apptainer exec --rocm "$SIF_PATH" "$target" >/dev/null 2>&1; then
        return 0
    fi

    if ! command -v rocminfo >/dev/null 2>&1; then
        warn "The container's rocminfo fails and the host has none to copy — aiter will not
  be able to detect the GPU architecture."
        return 0
    fi

    snap="$MODEL_CACHE_DIR/rocminfo-host.txt"
    if ! rocminfo > "$snap" 2>/dev/null || ! grep -q 'gfx' "$snap"; then
        rm -f "$snap"
        warn "The host's rocminfo did not produce usable output — cannot shim."
        return 0
    fi

    # Embed the snapshot in the script so the bind is self-contained.
    shim="$MODEL_CACHE_DIR/rocminfo-shim.sh"
    {
        printf '#!/bin/sh\ncat <<'\''KIMIK3_ROCMINFO_EOF'\''\n'
        cat "$snap"
        printf 'KIMIK3_ROCMINFO_EOF\n'
    } > "$shim"
    chmod +x "$shim"

    shim_args=(--bind "$shim":"$target")
    log "rocminfo shim: the image's rocminfo fails on this node; binding the host's output over $target"
}

set_rocm_mode_args() {   # $1 = mode -> sets global rocm_args
    case "$1" in
        rocm)    rocm_args=(--rocm) ;;
        devices) rocm_args=(--bind /dev/kfd:/dev/kfd --bind /dev/dri:/dev/dri) ;;
        *)       die "internal: unknown ROCm mode '$1'" ;;
    esac
}

# ROCm version string, best effort. /opt/rocm/.info/version is the canonical
# file; fall back to the versioned directory name.
ROCM_VER_CMD='cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm-* 2>/dev/null | sed -n "1s|.*/opt/rocm-||p"'
sif_rocm_ver()  { apptainer exec "$SIF_PATH" sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1; }
host_rocm_ver() {
    local v
    v="$(sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1 || true)"
    # Bunya installs ROCm from a module tree, not /opt, so fall back to the
    # version embedded in rocminfo's own path (/…/rocm/7.14.0/bin/rocminfo).
    [[ -z "$v" ]] && v="$(command -v rocminfo 2>/dev/null \
        | grep -o '[0-9][0-9]*\.[0-9][0-9]*\(\.[0-9][0-9]*\)\?' | tail -n1 || true)"
    [[ -z "$v" && -n "${ROCM_PATH:-}" ]] && v="${ROCM_PATH##*/}"
    printf '%s' "$v"
}

# Ask the container, under a given mode, the two questions that matter: can
# torch see the GPUs, and can aiter name the architecture? The second is the
# exact call that crashed — testing anything less is testing the wrong thing.
#
# aiter prints '[aiter] ...' banners on import, so answers go out as tagged
# lines and are grepped back rather than read positionally.
GPU_PROBE_PY='
n, gfx, err = -1, "", ""
try:
    import torch
    n = torch.cuda.device_count()
except BaseException as e:
    err = "torch: %s: %s" % (type(e).__name__, e)
if not err:
    try:
        from aiter.jit.utils.chip_info import get_gfx
        gfx = str(get_gfx() or "")
    except BaseException as e:
        err = "aiter: %s: %s" % (type(e).__name__, e)
print("KIMIK3_DEVICES %d" % n)
print("KIMIK3_GFX %s" % (gfx or "-"))
print("KIMIK3_ERR %s" % (" ".join(err.split()) or "-"))
'

# gpu_probe <mode> [extra apptainer args ...]
# Sets PROBE_DEVICES / PROBE_GFX / PROBE_ERR. Returns 0 only if the container
# saw at least one GPU and aiter named the architecture.
gpu_probe() {
    local mode="$1"; shift
    local out
    set_rocm_mode_args "$mode"
    out="$(apptainer exec "${rocm_args[@]}" "$@" "$SIF_PATH" \
              python3 -c "$GPU_PROBE_PY" 2>&1 || true)"

    PROBE_DEVICES="$(sed -n 's/^KIMIK3_DEVICES //p' <<<"$out" | tail -n1)"
    PROBE_GFX="$(sed -n 's/^KIMIK3_GFX //p' <<<"$out" | tail -n1)"
    PROBE_ERR="$(sed -n 's/^KIMIK3_ERR //p' <<<"$out" | tail -n1)"

    # No tagged output at all means python never ran (bad bind, missing device,
    # apptainer refused). Surface whatever it did say.
    if [[ -z "$PROBE_DEVICES" ]]; then
        PROBE_DEVICES="-1"
        PROBE_GFX="-"
        PROBE_ERR="$(tail -n 3 <<<"$out" | tr '\n' ' ')"
        PROBE_ERR="${PROBE_ERR:-container produced no output}"
    fi

    [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt 0 \
       && "$PROBE_GFX" == gfx* ]]
}

# ── gpucheck mode: can this image reach this node's GPUs? ───────────────────
# The diagnostic AND the permanent preflight — same code path, so what we debug
# with is what we run. Costs about a minute; the failure it replaces costs a
# crash 30 frames deep in aiter, or worse, a 1.5 TB weight load.

if [[ "$MODE" == "gpucheck" ]]; then
    log "Image:  $SIF_PATH"
    echo
    printf '  host ROCm       : %s\n' "$(host_rocm_ver || true)"
    printf '  container ROCm  : %s\n' "$(sif_rocm_ver || true)"
    host_gfx=""
    if command -v rocminfo >/dev/null 2>&1; then
        # grep -m1 would exit early and SIGPIPE rocminfo, which under
        # 'set -o pipefail' reads as a failed pipeline. Take the head instead.
        host_gfx="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -n1 || true)"
        printf '  host rocminfo       : %s\n' "${host_gfx:-ran, but printed no gfx line}"
    else
        printf '  host rocminfo       : not on PATH\n'
    fi
    for v in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do
        if [[ -n "${!v:-}" ]]; then
            note=""
            visible_devices_ok "${!v}" \
                || note="   <- NOT index-form; ROCr may fail to enumerate"
            printf '  %-20s: %s%s\n' "$v" "${!v}" "$note"
        fi
    done
    echo

    # Probe with the same aiter jit/ bind the server gets, when we already know
    # it — the .seeded marker records the in-image target. Testing a container
    # configured differently from the one we launch is how the read-only-.sif
    # bug survived three attempts.
    probe_extra=()
    jit_dir="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit}"
    if [[ -f "$jit_dir/.seeded" ]]; then
        jit_target="$(cut -d'|' -f2 "$jit_dir/.seeded" 2>/dev/null || true)"
        if [[ "$jit_target" == /* ]]; then
            probe_extra=(--bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR"
                         --bind "$jit_dir":"$jit_target")
            log "Probing with the aiter JIT bind: $jit_dir -> $jit_target"
        fi
    fi
    # Probe with the same GPU_ARCHS the server would get, so this mode can
    # actually verify the workaround it recommends.
    if [[ -n "$AITER_GPU_ARCHS" ]]; then
        probe_extra+=(--env "GPU_ARCHS=$AITER_GPU_ARCHS")
        log "Probing with GPU_ARCHS=$AITER_GPU_ARCHS"
    fi
    make_rocminfo_shim
    probe_extra+=(${shim_args[@]+"${shim_args[@]}"})
    echo

    # What does the image's rocminfo actually say? aiter only reports the exit
    # status, which hides the reason.
    printf '  in-container rocminfo:\n'
    apptainer exec --rocm "$SIF_PATH" sh -c \
        'rocminfo 2>&1 | head -n 12 || true' 2>&1 | sed 's/^/    /' || true
    echo

    gpucheck_ok=""
    best_mode=""; best_devices=0
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" ${probe_extra[@]+"${probe_extra[@]}"}; then
            printf '  %-8s : OK    devices=%s gfx=%s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX"
            [[ -z "$gpucheck_ok" ]] && gpucheck_ok="$m"
        else
            printf '  %-8s : FAIL  devices=%s gfx=%s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX"
            printf '             %s\n' "$PROBE_ERR"
        fi
        # Track the best result separately: "torch saw the GPUs but aiter could
        # not name the architecture" is a completely different problem from
        # "the container cannot reach the GPUs", and only the second one needs
        # a new image. Reporting both as one failure sends you down a
        # multi-hour rebuild for a broken rocminfo.
        if [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt "$best_devices" ]]; then
            best_devices="$PROBE_DEVICES"; best_mode="$m"
        fi
    done
    echo

    if [[ -n "$gpucheck_ok" ]]; then
        log "Verdict: use ROCM_MODE=$gpucheck_ok on this node."
        [[ "$gpucheck_ok" != "rocm" ]] && log \
            "  '--rocm' injects the host's ROCm libraries; '$gpucheck_ok' does not, which is
  what a host/container ROCm mismatch needs."
        [[ "$PROBE_DEVICES" != "$TP_SIZE" ]] && warn \
            "Container sees $PROBE_DEVICES GPU(s) but TP_SIZE=$TP_SIZE. Fix the allocation or TP_SIZE."
        # Record it so 'serve' does not pay for the probe again.
        printf '%s %s\n' "$(rocm_cache_key)" "$gpucheck_ok" > "$ROCM_MODE_CACHE" 2>/dev/null || true
        exit 0
    fi

    rm -f "$ROCM_MODE_CACHE" 2>/dev/null || true

    if [[ "$best_devices" -gt 0 ]]; then
        warn "Verdict: the GPUs ARE reachable ($best_devices via '$best_mode') — only aiter's
  architecture detection is broken. It shells out to the image's rocminfo, and
  that binary fails on this node even though HIP is fine. This does NOT need a
  new image."
        if [[ ${#shim_args[@]} -gt 0 ]]; then
            warn "  The rocminfo shim was already applied and aiter still could not read the
  architecture, so it is not parsing rocminfo the way we assumed. Send the
  output of:
      apptainer exec $SIF_PATH \\
          sed -n '1,90p' /sgl-workspace/aiter/aiter/jit/utils/chip_info.py"
        else
            warn "  No shim was applied. Set ROCMINFO_SHIM=force in $ENV_FILE to replay the
  host's rocminfo output inside the container, then re-run gpucheck."
        fi
        exit 1
    fi

    warn "Verdict: the container cannot reach the GPUs in any mode (torch saw none)."
    warn "  The node's kernel driver is newer than the container's ROCm and no bind
  fixes it — you need an image built for this node's ROCm. See the README
  'When the node's ROCm changes'.
    host ROCm $(host_rocm_ver || echo '?')  vs  container ROCm $(sif_rocm_ver || echo '?')"
    exit 1
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

# The image ships its own ROCm and the node has its own. They do not have to
# match, but a MAJOR difference is the thing most likely to break passthrough —
# say so up front rather than letting it surface as an aiter import error.
host_rocm="$(host_rocm_ver || true)"
sif_rocm="$(sif_rocm_ver || true)"
if [[ -n "$host_rocm" && -n "$sif_rocm" ]]; then
    log "ROCm: node $host_rocm / container $sif_rocm"
    if [[ "$(cut -d. -f1,2 <<<"$host_rocm")" != "$(cut -d. -f1,2 <<<"$sif_rocm")" ]]; then
        warn "Node and container ROCm versions differ ($host_rocm vs $sif_rocm).
  '--rocm' injects the NODE's ROCm libraries into the container, which is what
  breaks first when they diverge. ROCM_MODE=$ROCM_MODE will sort it out; run
  './serve-kimik3.sh gpucheck' if startup fails in aiter."
    fi
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

# The old FLYDSL_CACHE_* names are gone. An earlier version quietly remapped
# FLYDSL_CACHE_TARGET onto the new variable, which was a mistake: it is a HOST
# path in existing configs, so it overrode detection, bound scratch over a path
# nothing reads, and still passed the writability probe. Refuse to guess.
if [[ -n "${FLYDSL_CACHE_TARGET:-}" || -n "${FLYDSL_CACHE_DIR:-}" ]]; then
    obsolete_src="$ENV_FILE"
    grep -qE '^[[:space:]]*(export[[:space:]]+)?FLYDSL_CACHE_(DIR|TARGET)=' "$ENV_FILE" 2>/dev/null \
        || obsolete_src="your shell environment, NOT $ENV_FILE
          (exported vars override the file) — fix with:
              unset FLYDSL_CACHE_DIR FLYDSL_CACHE_TARGET"
    die "FLYDSL_CACHE_DIR / FLYDSL_CACHE_TARGET are obsolete and are NOT read any more.
  Set in: $obsolete_src
  The writable scratch dir is AITER_JIT_DIR (default \$MODEL_CACHE_DIR/aiter-jit)
  and the in-image path is auto-detected. Leaving them set previously produced
  a bind to the wrong destination."
fi

# Ask the image where aiter lives. Use find_spec, NOT 'import aiter': importing
# it needs a GPU (this exec has no --rocm) and prints '[aiter] import [...]'
# banners to stdout that would corrupt the captured path.
detected_jit="$(apptainer exec "$SIF_PATH" python3 - <<'PY' 2>/dev/null | tail -n 1
import importlib.util, os
spec = importlib.util.find_spec("aiter")
origin = getattr(spec, "origin", None) if spec is not None else None
print(os.path.join(os.path.dirname(origin), "jit") if origin else "")
PY
)"

if [[ -n "${AITER_JIT_TARGET:-}" ]]; then
    # A manual override must name a directory INSIDE the image. This is the
    # check whose absence caused three failed 1.5 TB loads: a host path here
    # binds somewhere harmless and the real path stays read-only.
    apptainer exec "$SIF_PATH" test -d "$AITER_JIT_TARGET" 2>/dev/null \
        || die "AITER_JIT_TARGET='$AITER_JIT_TARGET' does not exist inside the image.
  It must be a path in the CONTAINER (e.g. /sgl-workspace/aiter/aiter/jit),
  not a host directory. Detected value: ${detected_jit:-<detection failed>}
  Unset it in $ENV_FILE to use auto-detection."
    [[ -n "$detected_jit" && "$AITER_JIT_TARGET" != "$detected_jit" ]] \
        && warn "AITER_JIT_TARGET='$AITER_JIT_TARGET' overrides detected '$detected_jit'."
else
    AITER_JIT_TARGET="$detected_jit"
fi

cache_bind=()
if [[ "${AITER_JIT_TARGET:-}" != /* ]]; then
    warn "Could not locate aiter's jit/ directory in the image (got: '${AITER_JIT_TARGET:-}')."
    warn "  Startup will likely die at CUDA-graph capture with"
    warn "  \"Read-only file system: .../aiter/jit/flydsl_cache/...\"."
    warn "  Set AITER_JIT_TARGET in kimik3.env to the jit/ directory from that path."
else
    mkdir -p "$AITER_JIT_DIR" || die "Cannot create $AITER_JIT_DIR"

    # The seed marker records WHICH image it came from. A jit/ dir seeded from
    # one image and bound over another hides that image's prebuilt kernels
    # behind stale ones — the 'module_*.so: undefined symbol' crash. The README
    # has always said to delete this dir after changing SGLANG_IMAGE; nothing
    # enforced it, and moving between ROCm 7.2 and 7.14 images makes that a
    # certainty rather than a risk. The seed is derived data, so re-deriving it
    # is always safe.
    seed_stamp="$SGLANG_IMAGE|$AITER_JIT_TARGET|$(sif_stamp)"
    if [[ "$(cat "$AITER_JIT_DIR/.seeded" 2>/dev/null || true)" != "$seed_stamp" ]]; then
        if [[ -e "$AITER_JIT_DIR/.seeded" ]]; then
            log "Image changed since $AITER_JIT_DIR was seeded — re-seeding."
            # Guard the rm: this must never be able to point at scratch itself.
            case "$AITER_JIT_DIR" in
                ""|/|"$HOME"|"$MODEL_CACHE_DIR")
                    die "Refusing to clear AITER_JIT_DIR='$AITER_JIT_DIR' — set it to a directory of its own." ;;
            esac
            rm -rf "${AITER_JIT_DIR:?}"/* "${AITER_JIT_DIR:?}"/.[!.]* 2>/dev/null || true
        else
            log "Seeding writable aiter JIT dir from the image (one-off copy) ..."
        fi
        apptainer exec --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" "$SIF_PATH" \
            cp -a "$AITER_JIT_TARGET/." "$AITER_JIT_DIR/" \
            || die "Failed to copy $AITER_JIT_TARGET out of the image into $AITER_JIT_DIR"
        printf '%s\n' "$seed_stamp" > "$AITER_JIT_DIR/.seeded"
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

# GPU_ARCHS lets aiter skip its rocminfo shell-out when that call is the only
# broken part of the stack. It is NOT a fix for a GPU the container cannot
# reach: if torch sees no devices this changes nothing. Deliberately outside the
# ENABLE_AITER gate — sglang imports aiter during module import regardless, so
# ENABLE_AITER=0 does not avoid the detection path.
if [[ -n "$AITER_GPU_ARCHS" ]]; then
    aiter_env+=(--env "GPU_ARCHS=$AITER_GPU_ARCHS")
    log "AITER_GPU_ARCHS=$AITER_GPU_ARCHS — aiter will skip rocminfo architecture detection."
fi

if [[ "$ENABLE_AITER" == "1" ]]; then
    aiter_env+=(--env SGLANG_USE_AITER=1
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
#
# ...but only if the container's ROCr can parse them. A newer host stack can
# hand out UUID-form lists ("GPU-a1b2..."), and an older ROCr given one of those
# fails to enumerate ANY agent — which presents as a dead driver rather than as
# a bad variable. SLURM's cgroup already limits which devices are visible, so
# dropping an unparseable value is safe.
gpu_env=()
for v in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do
    val="${!v:-}"
    [[ -z "$val" ]] && continue
    if visible_devices_ok "$val"; then
        gpu_env+=(--env "$v=$val")
    else
        warn "$v='$val' is not an index list — NOT forwarding it into the container.
  The container's ROCm may not parse this form, and an unparseable value stops
  it enumerating any GPU at all. The SLURM allocation still constrains what is
  visible. Export a plain index list (e.g. 0,1,2,3,4,5,6,7) to override."
    fi
done

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
make_rocminfo_shim

base_args=(--bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR"
    ${cache_bind[@]+"${cache_bind[@]}"}
    ${shim_args[@]+"${shim_args[@]}"}
    --env HF_HOME="$MODEL_CACHE_DIR"
    --env HF_TOKEN="${HF_TOKEN:-}"
    --env HF_HUB_ENABLE_HF_TRANSFER=1
    --env SGLANG_SET_CPU_AFFINITY="$SET_CPU_AFFINITY"
    ${aiter_env[@]+"${aiter_env[@]}"}
    ${gpu_env[@]+"${gpu_env[@]}"})

# ── Pick the GPU passthrough mode ───────────────────────────────────────────
# Probed against base_args, so the configuration we test is the one we launch.

chosen_mode=""
cache_key="$(rocm_cache_key)"

if [[ "$ROCM_MODE" != "auto" ]]; then
    log "ROCM_MODE=$ROCM_MODE (explicit) — verifying it reaches the GPUs ..."
    if gpu_probe "$ROCM_MODE" "${base_args[@]}"; then
        chosen_mode="$ROCM_MODE"
    else
        die "ROCM_MODE=$ROCM_MODE cannot reach this node's GPUs.
  devices seen : $PROBE_DEVICES        gfx: $PROBE_GFX
  container    : $PROBE_ERR
  Run './serve-kimik3.sh gpucheck' — it tries every mode and names the cause.
  Or set ROCM_MODE=auto to let this script choose."
    fi
elif [[ "$(cut -d' ' -f1 "$ROCM_MODE_CACHE" 2>/dev/null || true)" == "$cache_key" ]]; then
    chosen_mode="$(cut -d' ' -f2 "$ROCM_MODE_CACHE" 2>/dev/null || true)"
    log "GPU passthrough: $chosen_mode (remembered for this node+image; delete $ROCM_MODE_CACHE to re-probe)"
else
    log "Probing GPU passthrough modes (~1 min; the answer is cached per node+image) ..."
    probe_best_devices=0
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" "${base_args[@]}"; then
            log "  $m: OK — $PROBE_DEVICES GPU(s), $PROBE_GFX"
            chosen_mode="$m"
            break
        fi
        warn "  $m: devices=$PROBE_DEVICES gfx=$PROBE_GFX — $PROBE_ERR"
        [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt "$probe_best_devices" ]] \
            && probe_best_devices="$PROBE_DEVICES"
    done
    [[ -n "$chosen_mode" ]] \
        && printf '%s %s\n' "$cache_key" "$chosen_mode" > "$ROCM_MODE_CACHE" 2>/dev/null || true
fi

if [[ -z "$chosen_mode" ]]; then
    # Torch seeing the GPUs while aiter cannot name the architecture is a
    # broken rocminfo, not a broken container — and needs a one-line fix
    # rather than a new image. Do not conflate the two.
    if [[ "${probe_best_devices:-0}" -gt 0 ]]; then
        die "The GPUs are reachable ($probe_best_devices seen) but aiter cannot determine the
  architecture — the image's rocminfo fails on this node even though HIP works.
  This does NOT need a new image.
  rocminfo shim applied: ${shim_args[*]:-<none>}
  Run './serve-kimik3.sh gpucheck' for the full report."
    fi
    die "No GPU passthrough mode works on this node — the container cannot reach the GPUs.
  Tried: ${ROCM_MODES[*]}
  node ROCm $(host_rocm_ver || echo '?')  vs  container ROCm $(sif_rocm_ver || echo '?')
  torch saw 0 devices in every mode, so the node's kernel driver is newer than
  the container's ROCm and no bind fixes it — you need an image built for this
  node's ROCm. Run './serve-kimik3.sh gpucheck' for the full report, and see
  the README 'When the node's ROCm changes'."
fi

if [[ "${PROBE_DEVICES:-}" =~ ^[0-9]+$ && "$PROBE_DEVICES" -lt "$GPUS_USED" ]]; then
    warn "Container sees $PROBE_DEVICES GPU(s) but TP_SIZE=$TP_SIZE needs $GPUS_USED. Startup will fail."
fi

set_rocm_mode_args "$chosen_mode"
log "GPU passthrough: $chosen_mode (${rocm_args[*]})"

apptainer_args=(exec "${rocm_args[@]}" "${base_args[@]}")

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

# The log is world-readable on scratch and gets pasted into bug reports, so
# never write the HF token or the API key into it.
redact_args() {
    local out=() prev="" a
    for a in "$@"; do
        if [[ "$prev" == "--api-key" ]]; then out+=("<redacted>")
        elif [[ "$a" == HF_TOKEN=* ]]; then   out+=("HF_TOKEN=<redacted>")
        else out+=("$a"); fi
        prev="$a"
    done
    printf '%s' "${out[*]}"
}

: > "$LOG_FILE"
{
    printf '### serve-kimik3.sh  %s\n' "$(date)"
    printf '### image=%s\n' "$SGLANG_IMAGE"
    printf '### apptainer: apptainer %s %s\n' "$(redact_args "${apptainer_args[@]}")" "$SIF_PATH"
    printf '### command: %s\n\n' "$(redact_args "${cmd[@]}")"
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

  Second shell into this job (to run a client alongside the server):
    srun --overlap --jobid $JOBID --pty /bin/bash -l

  From your laptop (tunnel through the login node, then use localhost):
    ssh -N -L $PORT:$NODE_HOST:$PORT \${USER}@bunya1.rcc.uq.edu.au

  opencode: run ./opencode-setup.sh --host $NODE_HOST --port $PORT
            (or --host localhost when tunnelling), then pick
            '$SERVED_MODEL_NAME' via /models inside opencode.

  kimicode: run ./kimicode-setup.sh --host $NODE_HOST --port $PORT
            (or --host localhost when tunnelling), then start it with
            kimi -m kimik3-bunya/$SERVED_MODEL_NAME

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
