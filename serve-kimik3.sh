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
#   ./serve-kimik3.sh loadstat   why the last cold start was slow (reads the log)
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
# Pinned, not tracking main: the draft repo is rewritten under you — three times
# in the four days to 1 Aug 2026, twice breaking the launch outright.
#
# THE PIN MOVED ON 12 AUG 2026, from eb03982e (27 Jul) to this one. eb03982e's own
# model card says "Trained context: sequence length 4096", with unscaled RoPE
# ("rope_type": "default") to match. Serving it at 100k runs it 25x past its
# trained window: accept length 1.39, against 7.25 at 1024 — which is INSIDE it.
# That is what made DSpark a 2.93x NET LOSS at 100k on 9 Aug. It was the draft,
# not DSpark. 9c4b2577 (31 Jul) replaced the weights with a long-context retrain —
# 65,536 trained, YaRN factor 16 to 1M, RULER V2 acc_len 4.26 at 1M input — and
# 56ce616a is that same checkpoint with a fuller card. See the README,
# 'DSpark collapsed at long context'.
#
# eb03982e stays the ANCHOR for every DSpark number measured before 12 Aug 2026 —
# set it explicitly to reproduce those. "main" follows the branch and its drift.
DSPARK_REVISION="${DSPARK_REVISION:-56ce616ad7486f0e96cbb51ef23ed5a1bce1d92d}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-}"
REPLAYSSM_SPEC="${REPLAYSSM_SPEC:-0}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-}"
MAMBA_RADIX_STRATEGY="${MAMBA_RADIX_STRATEGY:-}"
MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK:-0}"
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
WEIGHT_LOAD_THREADS="${WEIGHT_LOAD_THREADS:-8}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
PRESHARDED_PATH="${PRESHARDED_PATH:-}"
PREFETCH_BLOCK_SIZE_MB="${PREFETCH_BLOCK_SIZE_MB:-}"

# Measured from the safetensors index; used for the effective-GB/s readout and
# the presharded space preflight. Kept next to EST_GB below, which is the same
# number for the same reason.
WEIGHTS_GB=1561

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

[[ "$WEIGHT_LOAD_THREADS" =~ ^[0-9]+$ ]] \
    || die "WEIGHT_LOAD_THREADS must be a non-negative integer, got '$WEIGHT_LOAD_THREADS' (0 disables the flag)."
[[ -z "$PREFETCH_BLOCK_SIZE_MB" || "$PREFETCH_BLOCK_SIZE_MB" =~ ^[0-9]+$ ]] \
    || die "PREFETCH_BLOCK_SIZE_MB must be a positive integer or empty, got '$PREFETCH_BLOCK_SIZE_MB'."

# ── KDA state pool + DSpark shaping ─────────────────────────────────────────
#
# K3 is a hybrid: 69 of its 93 layers are KDA (linear attention) holding a
# recurrent STATE per request, and 24 are MLA holding a normal KV cache. These
# knobs size and shape that state pool. All default to upstream behaviour — the
# argv is unchanged until one is set.
case "$MAMBA_RADIX_STRATEGY" in
    ""|auto|extra_buffer|extra_buffer_lazy|no_buffer) ;;
    *) die "MAMBA_RADIX_STRATEGY must be empty, auto, extra_buffer, extra_buffer_lazy or no_buffer, got '$MAMBA_RADIX_STRATEGY'." ;;
esac

case "$MAMBA_SSM_DTYPE" in
    ""|bfloat16|float16|float32) ;;
    *) die "MAMBA_SSM_DTYPE must be empty, bfloat16, float16 or float32, got '$MAMBA_SSM_DTYPE'." ;;
esac

[[ -z "$MAMBA_FULL_MEMORY_RATIO" || "$MAMBA_FULL_MEMORY_RATIO" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
    || die "MAMBA_FULL_MEMORY_RATIO must be a positive number or empty, got '$MAMBA_FULL_MEMORY_RATIO'."

[[ -z "$DSPARK_BLOCK_SIZE" || "$DSPARK_BLOCK_SIZE" =~ ^[0-9]+$ ]] \
    || die "DSPARK_BLOCK_SIZE must be a positive integer or empty, got '$DSPARK_BLOCK_SIZE' (empty = infer from the draft checkpoint)."

# DSPARK-only flags. Fail here rather than after a 1.5 TB load: SGLang rejects
# both combinations at argument-parse time, but only once the container is up.
if [[ -n "$DSPARK_BLOCK_SIZE" && "$SPECULATIVE" != "dspark" ]]; then
    warn "DSPARK_BLOCK_SIZE=$DSPARK_BLOCK_SIZE is ignored while SPECULATIVE is '${SPECULATIVE:-off}'."
fi
if [[ "$REPLAYSSM_SPEC" == "1" ]]; then
    [[ "$SPECULATIVE" == "dspark" ]] \
        || die "REPLAYSSM_SPEC=1 needs SPECULATIVE=dspark — the ReplaySSM ring is spec-verify-only
  scratch, and a server that never runs verify rejects the flag at startup."
    # Upstream: "--enable-gdn-replayssm-spec is not validated with
    # --mamba-radix-cache-strategy extra_buffer_lazy yet; use extra_buffer."
    if [[ "$DISABLE_RADIX_CACHE" != "1" && "$MAMBA_RADIX_STRATEGY" == "extra_buffer_lazy" ]]; then
        die "REPLAYSSM_SPEC=1 with the radix cache on is not validated against
  MAMBA_RADIX_STRATEGY=extra_buffer_lazy. Use extra_buffer (or leave it empty for auto)."
    fi
fi

# The buffered multi-thread loader holds ~(num_threads + 2) shards in host RAM at
# once. K3's shards are ~5 GB, so 8 threads is ~50 GB against --mem=1800G. Well
# past that and you are trading a weight-load win for a host OOM.
if (( WEIGHT_LOAD_THREADS > 32 )); then
    warn "WEIGHT_LOAD_THREADS=$WEIGHT_LOAD_THREADS holds roughly $(( (WEIGHT_LOAD_THREADS + 2) * 5 )) GB of shards in host RAM.
  Check that against your --mem. Past ~16 the GPFS client, not the thread count, is usually the limit."
fi

if [[ -n "$PRESHARDED_PATH" && "$LOAD_FORMAT" != "presharded" ]]; then
    warn "PRESHARDED_PATH is set but LOAD_FORMAT is '${LOAD_FORMAT:-auto}' — the path will be ignored.
  Set LOAD_FORMAT=presharded to use it."
fi

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
    LOADTIMES_FILE="${LOADTIMES_FILE:-$MODEL_CACHE_DIR/kimik3-loadtimes.log}"
else
    PID_FILE="${PID_FILE:-$SCRIPT_DIR/kimik3-server.pid}"
    LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/kimik3-server.log}"
    LOADTIMES_FILE="${LOADTIMES_FILE:-$SCRIPT_DIR/kimik3-loadtimes.log}"
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
    serve|pull|download|check|gpucheck|parsers|loadstat) ;;
    *)
        die "Unknown mode '$MODE'. Use: serve | pull | download | check | gpucheck | toolcheck | parsers | loadstat | stop | status"
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

# ── Building the .sif ───────────────────────────────────────────────────────

# Apptainer 1.5.0 (6 May 2026) started wrapping mksquashfs in a bundled `proot`
# so that an unprivileged build preserves the original owners and groups of files
# coming out of an OCI registry. proot works by ptrace, so on a host that refuses
# ptrace(PTRACE_TRACEME) — yama ptrace_scope=3, a seccomp filter, some SELinux
# policies — the pull dies with:
#
#   proot error: ptrace(TRACEME): Operation not permitted
#   ... mksquashfs command failed: exit status 1
#
# Apptainer 1.5.3 (21 Jul 2026) turned that into an INFO message and carries on,
# so this only bites the 1.5.0–1.5.2 window. The escape hatch is the hidden
# `--ignore-proot` build flag, which drops back to exactly the pre-1.5.0
# behaviour: ownership inside the SIF is not preserved. That costs us nothing —
# we mount the image read-only and run as ourselves — and it is how every
# unprivileged Apptainer build worked before May 2026.
#
# Note --ignore-proot is registered on `build`, NOT on `pull`, so the retry has
# to switch subcommand. `build <sif> docker://…` is otherwise equivalent.
pull_image() {
    local logf rc
    logf="$(mktemp "${APPTAINER_TMPDIR:-/tmp}/kimik3-pull.XXXXXX.log")"
    log "Pulling $SGLANG_IMAGE -> $SIF_PATH (multi-GB; one-time) ..."

    # tee, not command substitution: this downloads ~29 GB and the progress bar
    # has to stay on screen. PIPESTATUS[0] to get apptainer's status rather than
    # tee's — and read via if/else, because a trailing `|| true` would run a new
    # command and reset PIPESTATUS to 0, i.e. silently swallow every failure.
    if apptainer pull "$SIF_PATH" "$SGLANG_IMAGE" 2>&1 | tee "$logf"; then rc=0; else rc=${PIPESTATUS[0]}; fi
    if (( rc == 0 )); then
        rm -f "$logf"
        log "Image ready: $SIF_PATH"
        return 0
    fi

    # A failed pull can leave a truncated .sif behind, and every later check in
    # this script is just [[ -f ]]. Clear it on any failure so the next run
    # retries instead of reporting "already present" and dying hours later.
    rm -f "$SIF_PATH"

    if ! grep -qi 'proot\|ptrace' "$logf"; then
        rm -f "$logf"
        return "$rc"
    fi

    warn "Apptainer could not use proot on this host (ptrace is not permitted here)."
    warn "Retrying with --ignore-proot; the image is identical apart from file"
    warn "ownership inside it, which we do not rely on."
    if PROOT_NO_SECCOMP=1 apptainer build --ignore-proot "$SIF_PATH" "$SGLANG_IMAGE" 2>&1 | tee "$logf"
    then rc=0; else rc=${PIPESTATUS[0]}; fi
    if (( rc == 0 )); then
        rm -f "$logf"
        log "Image ready: $SIF_PATH (built with --ignore-proot)."
        return 0
    fi
    rm -f "$SIF_PATH"

    if grep -qi 'unknown flag' "$logf"; then
        rm -f "$logf"
        die "This Apptainer predates --ignore-proot (added in 1.5.0) yet failed inside
  proot, which should not happen. Report it with the output above.
  Workaround: build the .sif on a host that allows ptrace, then copy it to
  $SIF_PATH and re-run."
    fi
    rm -f "$logf"
    return "$rc"
}

# ── Pull mode: build the .sif from the container image ──────────────────────

if [[ "$MODE" == "pull" ]]; then
    if [[ -f "$SIF_PATH" ]]; then
        log "Image already present: $SIF_PATH (delete it to re-pull)."
        exit 0
    fi
    pull_image \
        || die "apptainer pull failed. Does this node have outbound internet? Check APPTAINER_CACHEDIR space ($APPTAINER_CACHEDIR).
     Note the default image comes from SGLang's 'kimi-k3' branch, which merged on
     4 Aug 2026 and was deleted; that tag stream ended at ...k3-20260803 and is no
     longer rebuilt. If the pinned tag has been removed, see the README
     'Moving to a mainline image' for how to pick a current one."
    exit 0
fi

# For every remaining mode we need the .sif. Auto-build it if missing.
if [[ ! -f "$SIF_PATH" ]]; then
    log "No .sif at $SIF_PATH yet — pulling it now (one-time)."
    pull_image \
        || die "apptainer pull failed. Run './serve-kimik3.sh pull' explicitly to debug."
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

if [[ "$MODE" == "loadstat" ]]; then
    # Answers one question: was the last cold start single-threaded, and how
    # fast was it really? The repo used to have no load baseline at all, so
    # "slow" was never comparable between runs.
    log "Weight loading report from $LOG_FILE"
    echo

    if [[ ! -r "$LOG_FILE" ]]; then
        warn "No server log at $LOG_FILE — start the server once, then re-run this."
    else
        if grep -qi "falling back to single-threaded" "$LOG_FILE"; then
            warn "SINGLE-THREADED weight loading was used. This is the sawtooth:"
            grep -i -m1 "falling back to single-threaded" "$LOG_FILE" | sed 's/^/    /'
            echo
            log "Fix: set WEIGHT_LOAD_THREADS=8 in kimik3.env and restart."
        else
            log "No single-threaded fallback warning found."
        fi

        echo
        log "Loader flags on the last launch (from the recorded argv):"
        grep -m1 -oE '\-\-load-format [^ ]+|\-\-model-loader-extra-config [^ ]+' "$LOG_FILE" \
            | sed 's/^/    /' || echo "    (none — upstream defaults)"
    fi

    # The server log is truncated on every launch; this history is not, so it is
    # the only place an A/B between runs survives.
    echo
    log "Time-to-ready history ($LOADTIMES_FILE):"
    if [[ -r "$LOADTIMES_FILE" ]]; then
        tail -10 "$LOADTIMES_FILE" | sed 's/^/    /'
        echo
        log "Compare COLD runs only. Host RAM is 1800 GB and the weights are 1561 GB,"
        log "so a restart on the same node is served largely from page cache and will"
        log "look fast whatever the thread count is set to."
    else
        echo "    (nothing recorded yet — it is written when the ready-wait sees /health)"
    fi

    echo
    log "Load formats this image supports:"
    apptainer exec "$SIF_PATH" \
        bash -c "sglang serve --help 2>&1 || python3 -m sglang.launch_server --help 2>&1" 2>/dev/null \
        | grep -A 12 -- '--load-format' | head -20 | sed 's/^/    /' \
        || warn "Could not read --load-format choices from the image."

    echo
    log "Bunya is GPFS, not Lustre — there is no 'lfs setstripe' here, and \$TMPDIR is"
    log "the same filesystem, so staging there buys nothing. The lever is thread count."
    exit 0
fi

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
    print("\n  => This image CANNOT serve this model. K3 merged to main on "
          "4 Aug 2026\n     (sglang #32541), so any mainline image from 0.5.17 on "
          "should carry it —\n     see the README, 'Moving to a mainline image'.")
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

# The draft is a SEPARATE checkpoint, and nothing above covers it. Without this
# check a missing draft surfaces as
#   RuntimeError: Cannot find any model weights with `RadixArk/Kimi-K3-DSpark`
# raised from the draft worker — which the scheduler only builds AFTER the main
# model has finished loading. That is the whole 1.5 TB thrown away to learn a
# second repo was never fetched. Hit for real on 1 Aug 2026.
if [[ "$SPECULATIVE" == "dspark" && "$MODE" == "serve" ]]; then
    draft_dir="$MODEL_CACHE_DIR/hub/models--${DSPARK_MODEL//\//--}"
    if [[ ! -d "$draft_dir/snapshots" ]]; then
        die "SPECULATIVE=dspark but the draft model '$DSPARK_MODEL' is not cached in $MODEL_CACHE_DIR.
  The draft loads only after the main model finishes, so starting now would waste that whole load.
  Fetch it first:   ./serve-kimik3.sh download
  Or serve without speculative decoding:   unset SPECULATIVE"
    fi

    # Resolve the snapshot the SAME WAY the hub does — refs/<branch> -> sha —
    # not "newest directory by mtime". Those differ, and an earlier version of
    # this check compared the wrong one: it validated a good snapshot while
    # SGLang read the one refs/main pointed at, so the preflight passed and the
    # launch still died. A cache can hold several snapshots at once; only one of
    # them is what a bare repo id resolves to.
    # A 40-hex revision names a snapshot directly; anything else is a branch or
    # tag, which the cache stores as refs/<name> -> sha. DSPARK_REVISION defaults
    # to a sha, but "main" has to keep working for anyone who wants the drift.
    draft_pin="pinned"
    if [[ "$DSPARK_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
        draft_snapshot="$draft_dir/snapshots/$DSPARK_REVISION"
        [[ -d "$draft_snapshot" ]] \
            || die "DSPARK_REVISION=$DSPARK_REVISION is not in the cache.
  Fetch it first:   DSPARK_REVISION=$DSPARK_REVISION ./serve-kimik3.sh download
  Or track the branch instead:   DSPARK_REVISION=main"
    else
        draft_pin="tracking ${DSPARK_REVISION:-main}"
        draft_ref="$draft_dir/refs/${DSPARK_REVISION:-main}"
        if [[ -r "$draft_ref" ]]; then
            draft_snapshot="$draft_dir/snapshots/$(<"$draft_ref")"
        else
            draft_snapshot="$(ls -1dt "$draft_dir"/snapshots/*/ 2>/dev/null | head -1)"
            draft_snapshot="${draft_snapshot%/}"
        fi
    fi

    # A present directory is not a usable checkpoint. SGLang resolves the draft's
    # config during ARGUMENT PARSING, before anything loads, and a config without
    # model_type surfaces as the unhelpful
    #   ValueError: Unrecognized model in RadixArk/Kimi-K3-DSpark.
    #               Should have a `model_type` key in its config.json
    # Check it here instead, where we can say what to do about it. Hit for real
    # on 1 Aug 2026 after the upstream repo replaced its snapshot.
    if [[ -z "$draft_snapshot" || ! -d "$draft_snapshot" ]]; then
        die "Could not resolve a draft snapshot under $draft_dir/snapshots.
  Re-fetch it:   rm -rf $draft_dir && ./serve-kimik3.sh download"
    fi
    # -e not -r: these are symlinks into blobs/, and a dangling one is exactly
    # what a half-finished download leaves behind.
    if [[ ! -e "$draft_snapshot/config.json" ]]; then
        die "The draft snapshot $draft_snapshot has no config.json (or it is a dangling symlink).
  The download is incomplete. Re-fetch:   rm -rf $draft_dir && ./serve-kimik3.sh download
  Or serve without speculative decoding:  unset SPECULATIVE"
    fi
    # Three fields out of one parse: model_type is the thing SGLang dies without,
    # and the rope type plus the draft's own trained context are what decide
    # whether this checkpoint can be trusted at the CONTEXT_LEN being served.
    # Read rope_parameters (transformers 5.x) or rope_scaling (4.x) — the draft
    # repo uses the former, and it is the field that changed on 31 Jul 2026.
    draft_cfg="$(python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
rp = c.get('rope_parameters') or c.get('rope_scaling') or {}
print(c.get('model_type') or '')
print(rp.get('rope_type') or rp.get('type') or '')
print(rp.get('original_max_position_embeddings') or '')
" "$draft_snapshot/config.json" 2>/dev/null || true)"
    { read -r draft_model_type
      read -r DRAFT_ROPE_TYPE
      read -r DRAFT_TRAINED_CTX
    } <<<"$draft_cfg"

    if [[ -z "${draft_model_type:-}" ]]; then
        die "The draft config at $draft_snapshot/config.json is unparseable or has no 'model_type',
  so SGLang cannot resolve the speculative algorithm and dies during argument parsing.
  Either the download is truncated, or upstream changed the repo under you — it has
  been rewritten as recently as 31 Jul 2026 ('Sync Kimi-K3-DSpark-0731 snapshot').

  Re-fetch it:                 rm -rf $draft_dir && ./serve-kimik3.sh download
  Pin the current default:     DSPARK_REVISION=56ce616ad7486f0e96cbb51ef23ed5a1bce1d92d ./serve-kimik3.sh download
  Or serve without it:         unset SPECULATIVE"
    fi

    # Hand SGLang the resolved PATH, never the repo id. The repo id makes it
    # re-resolve through the hub, which is how it ended up reading a different
    # snapshot than the one checked here.
    log "Draft: $DSPARK_MODEL -> $(basename "$draft_snapshot") ($draft_pin)"
    DSPARK_MODEL="$draft_snapshot"
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
        # Upstream rewrites this repo (a full snapshot swap landed 31 Jul 2026),
        # so allow pinning the exact commit that was measured rather than
        # whatever 'main' happens to be today.
        rev_arg=""
        [[ -n "$DSPARK_REVISION" ]] && rev_arg=" --revision '$DSPARK_REVISION'"
        log "Prefetching the DSpark draft model $DSPARK_MODEL${DSPARK_REVISION:+ @ $DSPARK_REVISION} ..."
        apptainer exec \
            --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
            --env HF_HOME="$MODEL_CACHE_DIR" \
            --env HF_TOKEN="${HF_TOKEN:-}" \
            --env HF_HUB_ENABLE_HF_TRANSFER=1 \
            "$SIF_PATH" \
            bash -c "hf download '$DSPARK_MODEL'$rev_arg || huggingface-cli download '$DSPARK_MODEL'$rev_arg" \
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

# Ask the image what flags it actually has. The pinned image is a day-0 BRANCH
# build, not mainline, so any given flag may simply not exist in it — and finding
# that out the other way costs a full 1.5 TB load before the server dies on an
# unknown argument.
#
# Memoised: several callers below ask about different flags, and shelling into
# the image once per question is a second each for the same answer. Empty means
# the probe itself failed, which is not the same as "the flag is absent" — every
# caller has to tell those two apart.
IMAGE_HELP=""
IMAGE_HELP_PROBED=0
image_help() {
    if (( ! IMAGE_HELP_PROBED )); then
        IMAGE_HELP_PROBED=1
        IMAGE_HELP="$(apptainer exec "$SIF_PATH" \
            bash -c "sglang serve --help 2>&1 || python3 -m sglang.launch_server --help 2>&1" 2>/dev/null || true)"
    fi
    printf '%s' "$IMAGE_HELP"
}

# Which HIP renorm bindings does this image have? DSpark's verify step reaches
# for top_k_renorm_prob / top_p_renorm_prob, and on ROCm those are bound in
# dflash_utils.py by an is_hip() branch that arrived in two halves:
#
#   sglang #32621 (28 Jul 07:37 UTC)  aliased top_P  -> a Triton kernel
#   sglang #32641 (31 Jul 06:21 UTC)  added  top_K   -> a Triton kernel
#
# Our pinned rocm720-mi35x-k3-20260727 was pushed 28 Jul 05:48 UTC — about two
# hours before the first of those. It has NEITHER, so both names are None, and
# calling one is `TypeError: 'NoneType' object is not callable` inside the
# scheduler's event loop. That kills the SERVER, not the request (sglang #32569).
#
# Grep for the alias names rather than infer from the image tag: the tag is a
# build date, the aliases are the thing that actually has to be there. Textual,
# not an import — importing sglang.srt.speculative drags in torch, which is slow
# and unhappy on a login node. Empty output means "could not tell", which is not
# the same as "absent"; the caller distinguishes them.
#
# The same probe answers a second, worse question — see the `v`/`t` markers below
# and sglang #33694. Markers: k, p (renorm kernels), v (the HIP branch claims
# non-greedy verify), t (#33694 applied), probed (the file was read at all).
RENORM_BINDINGS=""
RENORM_PROBED=0
renorm_bindings() {
    if (( ! RENORM_PROBED )); then
        RENORM_PROBED=1
        RENORM_BINDINGS="$(apptainer exec "$SIF_PATH" bash -c '
            f=/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py
            # Search only image-owned roots. A bare `find /` here would walk the
            # bound MODEL_CACHE_DIR — 1.5 TB of weights — to find a 30 KB file.
            [[ -r "$f" ]] || f=$(find /sgl-workspace /opt /usr/lib /usr/local \
                -name dflash_utils.py -path "*sglang/srt/speculative*" 2>/dev/null | head -1)
            [[ -n "$f" && -r "$f" ]] || exit 0
            grep -q top_k_renorm_probs_triton "$f" && echo k
            grep -q top_p_renorm_probs_triton "$f" && echo p
            # The SECOND landmine, and a much commoner trigger — sglang #33694.
            # #32541 opened an `elif is_hip():` branch that set
            # _DFLASH_SAMPLING_VERIFY_AVAILABLE = True without ever binding
            # tree_speculative_sampling_target_only, and that kernel does not
            # exist on ROCm at all. Any request with temperature > 0 then took
            # the non-greedy verify path and died on the unbound name.
            # Grep the BRANCH, not the file: the call site is present either
            # way, so a bare grep for the symbol proves nothing. #33694 fixed it
            # by binding the name to None inside the same branch.
            hip_branch=$(awk "/^elif is_hip\(\):/ {h=1; next} h && /^[^ \t]/ {h=0} h" "$f")
            if [[ -n "$hip_branch" ]]; then
                grep -q "_DFLASH_SAMPLING_VERIFY_AVAILABLE[[:space:]]*=[[:space:]]*True" \
                    <<<"$hip_branch" && echo v
                grep -q "tree_speculative_sampling_target_only" <<<"$hip_branch" && echo t
            fi
            echo probed
        ' 2>/dev/null || true)"
    fi
    printf '%s' "$RENORM_BINDINGS"
}

# Markers are one per line and matched WHOLE-LINE on purpose. Substring tests
# here are a trap: `probed` contains a `p`, so a *p* glob reports "top_p is fine"
# for the image where nothing is fine.
renorm_has() { grep -qxF "$1" <<<"$(renorm_bindings)"; }


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

    # gamma, the number of proposed draft tokens. Omitted, SGLang infers it from
    # the draft checkpoint's block_size — which is 7 today, so setting 7 changes
    # nothing. That is the point: it is an anchor against the draft repo being
    # rewritten under us again, not a speedup.
    [[ -n "$DSPARK_BLOCK_SIZE" ]] \
        && spec_flags+=(--speculative-dspark-block-size "$DSPARK_BLOCK_SIZE")

    # NOTE THE FLAG NAME. Upstream's own K3 cookbook prescribes
    # --enable-linear-replayssm-spec for every DSPARK recipe; that flag DOES NOT
    # EXIST (no such attribute anywhere in the tree, checked 2 Aug 2026). The
    # real one is --enable-gdn-replayssm-spec.
    #
    # Do not "fix" this to the similarly named --enable-linear-replayssm. That is
    # a different, mutually exclusive flag whose own help says KDA decode is
    # SLOWER than the packed baseline — so the plausible-looking name is a silent
    # regression rather than an error.
    if [[ "$REPLAYSSM_SPEC" == "1" ]]; then
        if grep -q -- '--enable-gdn-replayssm-spec' <<<"$(image_help)"; then
            spec_flags+=(--enable-gdn-replayssm-spec)
            log "ReplaySSM spec-verify ENABLED (--enable-gdn-replayssm-spec)."
        elif [[ -z "$(image_help)" ]]; then
            warn "Could not read --help from the image; passing --enable-gdn-replayssm-spec unverified."
            spec_flags+=(--enable-gdn-replayssm-spec)
        else
            die "This image has no --enable-gdn-replayssm-spec.
  Coexistence with the extra_buffer radix strategy landed upstream on 31 Jul 2026
  (sglang #32692), AFTER the pinned rocm720-mi35x-k3-20260727 build. Pull a
  mainline v0.5.17-rocm720-mi35x-* image into a SECOND SIF_PATH and test it there
  — see the README, 'Moving to a mainline image' — or set REPLAYSSM_SPEC=0."
        fi
    fi

    log "DSpark speculative decoding ENABLED (draft: $DSPARK_MODEL)."

    # This does NOT fail at startup, which is what makes it worth a warning here.
    # The server comes up healthy and serves correctly for as long as every
    # request leaves top_p at 1.0 and top_k unset — which is exactly what our
    # bench and toolcheck do, and why the README's numbers exist at all. The
    # first request that sets either one reaches the renorm call in SGLang's
    # verify step and takes the scheduler down with it, mid-session. See the
    # README, 'The sampling landmine'.
    if ! renorm_has probed; then
        warn "  Could not read the image's sampling bindings; cannot tell whether"
        warn "  top_p/top_k requests are safe with DSpark on (sglang #32569)."
    elif renorm_has k && renorm_has p; then
        log "  Sampling bindings OK: this image has the HIP top_k and top_p renorm kernels."
    else
        warn "  This image is missing the HIP renorm kernels DSpark's verify step needs."
        if renorm_has p; then
            warn "  top_p works here; a request setting TOP_K raises NameError and KILLS"
            warn "  THE SERVER — not just that request. Safe while no client sets top_k."
        else
            warn "  A request setting TOP_P (<1.0) or TOP_K raises"
            warn "  \"TypeError: 'NoneType' object is not callable\" and KILLS THE SERVER —"
            warn "  not just that request. One such request poisons its whole batch."
            warn "  Safe only while every client leaves top_p at 1.0 and top_k unset."
        fi
        warn "  Real fix: a mainline v0.5.17-rocm720-mi35x-* image (sglang #32621 +"
        warn "  #32641), tested in a SECOND SIF_PATH first. Zero-risk: unset SPECULATIVE."
    fi

    # The second landmine, and the one a real client is far likelier to step on:
    # top_p/top_k are optional, temperature > 0 is what every chat client sends.
    if renorm_has v && ! renorm_has t; then
        warn "  This image's is_hip() branch claims non-greedy verify is available but"
        warn "  never binds tree_speculative_sampling_target_only — the kernel does not"
        warn "  exist on ROCm at all. A request with TEMPERATURE > 0 then raises"
        warn "  \"NameError: tree_speculative_sampling_target_only\" and KILLS THE SERVER"
        warn "  (sglang #33694, fixed 6 Aug 2026 — mainline images only). Safe only"
        warn "  while every client sends temperature 0, which toolcheck and the bench do."
    fi

    # A draft has its own trained context, and past it the accept rate does not
    # degrade — it collapses. Measured on bun160, 9 Aug 2026: the 27 Jul draft
    # (trained at 4096, unscaled RoPE) accepted 1.39 tokens per step at 100k
    # against 7.25 at 1024, making DSpark a 2.93x net LOSS. Its own model card
    # said so from day one. That was the checkpoint, not DSpark.
    #
    # The condition is the ROPE TYPE, not the revision sha: it is the property
    # that actually decides this, and it keeps holding when upstream rewrites the
    # repo again. "default" means no context extension of any kind.
    if [[ "${DRAFT_ROPE_TYPE:-}" == "default" ]]; then
        warn "  This draft has UNSCALED RoPE (\"rope_type\": \"default\") — no context"
        warn "  extension at all${DRAFT_TRAINED_CTX:+, trained at $DRAFT_TRAINED_CTX tokens} — while you are serving ctx=${CONTEXT_LEN:-model max (1M)}."
        warn "  Past the draft's trained window the accept rate does not degrade, it"
        warn "  COLLAPSES, and DSpark turns into a net loss: 2.93x slower at 100k,"
        warn "  measured 9 Aug 2026. Fine for short prompts, wrong for agentic sessions."
        warn "  The long-context retrain is the default revision now:"
        warn "    DSPARK_REVISION=56ce616ad7486f0e96cbb51ef23ed5a1bce1d92d ./serve-kimik3.sh download"
    fi
fi

# K3's KDA layers hold a recurrent state per request, sized against the MLA KV
# pool. Every one of these is empty by default, so the argv is unchanged until
# you set one — measure with ./bench-kimik3.sh, one variable at a time.
mamba_flags=()
[[ -n "$MAMBA_FULL_MEMORY_RATIO" ]] && mamba_flags+=(--mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO")
[[ -n "$MAMBA_SSM_DTYPE" ]]        && mamba_flags+=(--mamba-ssm-dtype "$MAMBA_SSM_DTYPE")
[[ -n "$MAMBA_RADIX_STRATEGY" ]]   && mamba_flags+=(--mamba-radix-cache-strategy "$MAMBA_RADIX_STRATEGY")

if (( ${#mamba_flags[@]} )); then
    unknown_mamba=()
    for f in "${mamba_flags[@]}"; do
        [[ "$f" == --* ]] || continue
        grep -q -- "$f" <<<"$(image_help)" || unknown_mamba+=("$f")
    done
    if [[ -n "$(image_help)" ]] && (( ${#unknown_mamba[@]} )); then
        die "This image does not offer: ${unknown_mamba[*]}
  These are K3 hybrid-attention flags; an older or non-K3 image will not have them.
  Unset the matching MAMBA_* variables in kimik3.env, or use a newer K3 image."
    fi
    log "KDA state pool: ${mamba_flags[*]}"
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

# Skips the decode-time mamba lock, freeing one resident KDA state slot per
# request (extra_buffer 5->4, extra_buffer_lazy 4->3; no_buffer is unaffected).
# Upstream labels it experimental and ships it off. An env var, not a flag, so it
# cannot be capability-probed from --help — an image that does not know it simply
# ignores it, which is the safe direction.
if [[ "$MAMBA_SKIP_DECODE_LOCK" == "1" ]]; then
    aiter_env+=(--env SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1)
    log "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 — one fewer KDA state slot per request (experimental)."
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

# ── Weight loading ──────────────────────────────────────────────────────────
#
# SGLang silently falls back to SINGLE-THREADED weight loading whenever
# checkpoint prefetch is on and the user has not asked for threads explicitly:
#
#   "--weight-loader-prefetch-checkpoints is enabled; falling back to
#    single-threaded weight loading to avoid I/O oversubscription with the
#    prefetch threads. Set enable_multithread_load=true in
#    --model-loader-extra-config to keep multi-threaded loading."
#
# One sequential reader against GPFS is what makes a cold start sawtooth: a
# burst while a shard streams, a dip while it is converted and copied to HBM,
# then the next shard. Naming either key in the JSON suppresses that fallback,
# which is the whole reason this flag is set by default.
#
# Bunya is GPFS, not Lustre, so there is no 'lfs setstripe' to reach for and
# $TMPDIR is the same filesystem — the fix has to be client-side parallelism.
load_flags=()
loader_cfg=""

if (( WEIGHT_LOAD_THREADS > 0 )); then
    loader_cfg="\"enable_multithread_load\":true,\"num_threads\":$WEIGHT_LOAD_THREADS"
fi

if [[ "$LOAD_FORMAT" == "presharded" && -n "$PRESHARDED_PATH" ]]; then
    # presharded takes its target root in the SAME extra-config object, so both
    # settings have to be merged into one flag rather than passed twice.
    [[ -n "$loader_cfg" ]] && loader_cfg+=","
    loader_cfg+="\"presharded_path\":\"$PRESHARDED_PATH\""
    # The speculative draft is a separate model and gets its own root. Without
    # it the draft dumps to <draft_model_path>/presharded — inside the HF cache,
    # which is exactly the read-only mount upstream warns against.
    if [[ "$SPECULATIVE" == "dspark" ]]; then
        loader_cfg+=",\"draft_presharded_path\":\"$PRESHARDED_PATH/dspark\""
    fi
fi

if [[ -n "$loader_cfg" || -n "$LOAD_FORMAT" ]]; then
    help_text="$(image_help)"

    if [[ -z "$help_text" ]]; then
        warn "Could not read --help from the image; passing the weight-loading flags unverified."
    else
        if [[ -n "$loader_cfg" ]] && ! grep -q -- '--model-loader-extra-config' <<<"$help_text"; then
            warn "This image has no --model-loader-extra-config, so multi-threaded weight
  loading cannot be requested and the cold start will stay single-threaded.
  Set WEIGHT_LOAD_THREADS=0 in kimik3.env to silence this."
            loader_cfg=""
        fi
        if [[ -n "$LOAD_FORMAT" ]] && ! grep -q -- "$LOAD_FORMAT" <<<"$help_text"; then
            die "This image's --load-format does not offer '$LOAD_FORMAT'.
  Run './serve-kimik3.sh loadstat' to see what it does support, then set
  LOAD_FORMAT in kimik3.env accordingly (empty = the image's default)."
        fi
    fi
fi

[[ -n "$LOAD_FORMAT" ]] && load_flags+=(--load-format "$LOAD_FORMAT")
[[ -n "$loader_cfg" ]] && load_flags+=(--model-loader-extra-config "{$loader_cfg}")

if [[ -n "$loader_cfg" ]]; then
    log "Weight loading: ${WEIGHT_LOAD_THREADS} threads${LOAD_FORMAT:+, format=$LOAD_FORMAT}"
fi

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
     ${mamba_flags[@]+"${mamba_flags[@]}"}
     ${perf_flags[@]+"${perf_flags[@]}"}
     ${load_flags[@]+"${load_flags[@]}"}
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

# presharded writes a second, per-rank copy of the checkpoint. Refuse to start a
# dump that cannot finish — the same reasoning as the weights-cache accounting
# above, and the same failure mode if skipped: hours lost, then no space.
presharded_bind=()
if [[ "$LOAD_FORMAT" == "presharded" && -n "$PRESHARDED_PATH" ]]; then
    mkdir -p "$PRESHARDED_PATH" 2>/dev/null || true
    [[ "$SPECULATIVE" == "dspark" ]] && mkdir -p "$PRESHARDED_PATH/dspark" 2>/dev/null || true
    [[ -d "$PRESHARDED_PATH" && -w "$PRESHARDED_PATH" ]] \
        || die "PRESHARDED_PATH '$PRESHARDED_PATH' does not exist or is not writable."
    ps_free_gb="$(df -Pk "$PRESHARDED_PATH" | awk 'NR==2 {print int($4/1024/1024)}')"
    if [[ "${ps_free_gb:-0}" -lt "$WEIGHTS_GB" ]]; then
        warn "Only ${ps_free_gb} GB free at $PRESHARDED_PATH; the presharded dump is up to ${WEIGHTS_GB} GB
  (less with dedup, but do not count on it). The dump happens AFTER a full load."
    fi
    # Only needs its own bind when it sits outside the cache dir already bound.
    [[ "$PRESHARDED_PATH" != "$MODEL_CACHE_DIR"/* ]] \
        && presharded_bind=(--bind "$PRESHARDED_PATH":"$PRESHARDED_PATH")
fi

prefetch_env=()
[[ -n "$PREFETCH_BLOCK_SIZE_MB" ]] \
    && prefetch_env=(--env SGLANG_PREFETCH_BLOCK_SIZE_MB="$PREFETCH_BLOCK_SIZE_MB")

base_args=(--bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR"
    ${cache_bind[@]+"${cache_bind[@]}"}
    ${shim_args[@]+"${shim_args[@]}"}
    ${presharded_bind[@]+"${presharded_bind[@]}"}
    --env HF_HOME="$MODEL_CACHE_DIR"
    --env HF_TOKEN="${HF_TOKEN:-}"
    --env HF_HUB_ENABLE_HF_TRANSFER=1
    --env SGLANG_SET_CPU_AFFINITY="$SET_CPU_AFFINITY"
    ${prefetch_env[@]+"${prefetch_env[@]}"}
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
        ready_secs=$(( $(date +%s) - start_ts ))
        # Effective GB/s over the whole startup, not just the read: it includes
        # conversion, H2D and graph capture. It is a comparable number between
        # runs, which is the point — the repo had no load baseline at all.
        ready_gbs="$(awk -v g="$WEIGHTS_GB" -v s="$ready_secs" \
            'BEGIN {printf "%.2f", (s > 0 ? g / s : 0)}')"
        log "Ready in ${ready_secs}s (${ready_gbs} GB/s effective over ${WEIGHTS_GB} GB of weights)"

        # $LOG_FILE is truncated on every launch, so the comparison you actually
        # want — this run against the last one — would be gone. Keep a small
        # append-only history instead. This is what makes an A/B possible.
        printf '%s\t%s\tthreads=%s\tformat=%s\tspec=%s\t%ss\t%s GB/s\n' \
            "$(date -Is)" "$(hostname -s 2>/dev/null || hostname)" \
            "$WEIGHT_LOAD_THREADS" "${LOAD_FORMAT:-auto}" "${SPECULATIVE:-none}" \
            "$ready_secs" "$ready_gbs" >> "$LOADTIMES_FILE" 2>/dev/null || true

        if grep -qi "falling back to single-threaded" "$LOG_FILE" 2>/dev/null; then
            warn "SGLang fell back to SINGLE-THREADED weight loading — that is the sawtooth.
  Set WEIGHT_LOAD_THREADS=8 in kimik3.env. See './serve-kimik3.sh loadstat'."
        fi
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
