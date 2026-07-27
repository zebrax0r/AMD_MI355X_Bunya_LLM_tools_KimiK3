#!/usr/bin/env bash
#
# share-kimik3.sh — expose the local Kimi K3 endpoint to the public internet
# over HTTPS using a Cloudflare quick tunnel, so someone without SSH access to
# Bunya can point their opencode at it.
#
# The tunnel is OUTBOUND-ONLY (the node dials out to Cloudflare), so it works
# on locked-down HPC nodes with no inbound connectivity and needs no root and
# no Cloudflare account. The endpoint stays gated by your API key.
#
# NOTE (Bunya): this needs the compute node to have outbound internet — the
# same path the HF weights download uses. If that's blocked, use the SSH
# tunnel instead:  ssh -N -L 30000:<gpu-node>:30000 $USER@bunya1.rcc.uq.edu.au
#
# Usage:
#   ./share-kimik3.sh                 start a tunnel, stay attached (Ctrl-C stops)
#   ./share-kimik3.sh share --detach  start a tunnel, return the shell
#   ./share-kimik3.sh stop            stop a detached tunnel
#   ./share-kimik3.sh status          show tunnel state + current public URL
#
# SECURITY: the public URL + API key together grant anyone full use of your
# model (and Bunya GPU-hours on your a_rcc account). Share the key privately,
# and remember exposing university HPC compute externally may breach RCC's
# acceptable-use policy — check with rcc-support@uq.edu.au.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="share"
DETACH=0
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *)           MODE="$arg" ;;
    esac
done

log()  { printf '\033[1;34m[share]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[share WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[share ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config (shared with serve-kimik3.sh) ───────────────────────────────

ENV_FILE="${KIMIK3_ENV:-$SCRIPT_DIR/kimik3.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k3}"
CONTEXT_LEN="${CONTEXT_LEN:-131072}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"

[[ -n "$MODEL_CACHE_DIR" ]] || die "MODEL_CACHE_DIR is not set (source kimik3.env or export it). Needed to cache cloudflared and store tunnel state."

# Where we keep the cloudflared binary and tunnel state (persist across jobs).
CLOUDFLARED_DIR="${CLOUDFLARED_DIR:-$MODEL_CACHE_DIR/cloudflared}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$CLOUDFLARED_DIR/cloudflared}"
PID_FILE="$CLOUDFLARED_DIR/tunnel.pid"
LOG_FILE="$CLOUDFLARED_DIR/tunnel.log"
mkdir -p "$CLOUDFLARED_DIR"

# ── Helpers ─────────────────────────────────────────────────────────────────

current_url() {
    grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_FILE" 2>/dev/null | tail -1
}

tunnel_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null
}

resolve_api_key() {
    if [[ -n "${KIMIK3_API_KEY:-}" ]]; then
        printf '%s' "$KIMIK3_API_KEY"
    elif [[ -r "$MODEL_CACHE_DIR/kimik3-api-key" ]]; then
        printf '%s' "$(<"$MODEL_CACHE_DIR/kimik3-api-key")"
    fi
}

print_friend_banner() {
    local url="$1" key="$2"
    cat <<EOF

============================================================================
  Kimi K3 is now reachable from anywhere over HTTPS.

  Public endpoint:  ${url}/v1
  Model name:       $SERVED_MODEL_NAME
  API key:          ${key:-<none set!>}

  Send your colleague (over a PRIVATE channel) the URL + API key, and this
  opencode provider block for their ~/.config/opencode/opencode.json:

  {
    "\$schema": "https://opencode.ai/config.json",
    "provider": {
      "kimik3-remote": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Kimi K3 (shared)",
        "options": {
          "baseURL": "${url}/v1",
          "apiKey": "${key:-PASTE_API_KEY_HERE}"
        },
        "models": {
          "$SERVED_MODEL_NAME": {
            "name": "Kimi K3",
            "limit": { "context": $CONTEXT_LEN, "output": 32768 },
            "tool_call": true,
            "reasoning": true
          }
        }
      }
    }
  }

  They then restart opencode and pick 'Kimi K3 (shared)' via /models.

  Quick check they can run:
    curl -s ${url}/v1/models -H "Authorization: Bearer ${key:-KEY}"

  NOTE: this is a Cloudflare quick tunnel — the URL is random and changes
  every time you restart it. Stop with Ctrl-C (attached) or
  './share-kimik3.sh stop' (detached).
============================================================================

EOF
}

# ── Simple modes ────────────────────────────────────────────────────────────

case "$MODE" in
    stop)
        if tunnel_running; then
            kill "$(<"$PID_FILE")" 2>/dev/null || true
            rm -f "$PID_FILE"
            log "Tunnel stopped."
        else
            warn "No running tunnel (pidfile $PID_FILE)."
            rm -f "$PID_FILE"
        fi
        exit 0
        ;;
    status)
        if tunnel_running; then
            log "Tunnel running (pid $(<"$PID_FILE"))."
            url="$(current_url)"
            [[ -n "$url" ]] && log "Public URL: ${url}/v1" || warn "URL not found in log yet."
        else
            log "No tunnel running."
        fi
        exit 0
        ;;
    share) ;;
    *) die "Unknown mode '$MODE'. Use: share | stop | status" ;;
esac

# ── Preflight ───────────────────────────────────────────────────────────────

command -v curl >/dev/null 2>&1 || die "curl not found on PATH."

if tunnel_running; then
    die "A tunnel is already running (pid $(<"$PID_FILE")). Use './share-kimik3.sh status' or 'stop' first."
fi

# The local server must be healthy before we expose it.
if ! curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    die "No healthy server on http://127.0.0.1:${PORT}. Start it first with './serve-kimik3.sh serve --detach'."
fi
log "Local server on port $PORT is healthy."

API_KEY="$(resolve_api_key)"
[[ -n "$API_KEY" ]] || warn "No API key found — your endpoint will be UNAUTHENTICATED and open to anyone with the URL. Set KIMIK3_API_KEY or restart serve-kimik3.sh to generate one."

# ── Ensure cloudflared binary ───────────────────────────────────────────────

if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
    case "$(uname -m)" in
        x86_64|amd64)   cf_arch="amd64" ;;
        aarch64|arm64)  cf_arch="arm64" ;;
        *)              die "Unsupported CPU arch $(uname -m) for cloudflared auto-download; install it manually and set CLOUDFLARED_BIN." ;;
    esac
    cf_url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${cf_arch}"
    log "Downloading cloudflared ($cf_arch) to $CLOUDFLARED_BIN ..."
    curl -fL --retry 3 -o "$CLOUDFLARED_BIN" "$cf_url" \
        || die "Failed to download cloudflared from $cf_url (does the node have outbound internet?)."
    chmod +x "$CLOUDFLARED_BIN"
fi
log "Using cloudflared: $CLOUDFLARED_BIN"

# ── Start the tunnel ────────────────────────────────────────────────────────

: > "$LOG_FILE"
"$CLOUDFLARED_BIN" tunnel --no-autoupdate --url "http://localhost:${PORT}" \
    >>"$LOG_FILE" 2>&1 &
CF_PID=$!
echo "$CF_PID" > "$PID_FILE"

cleanup() {
    log "Stopping tunnel ..."
    kill "$CF_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
}
if [[ "$DETACH" -eq 0 ]]; then
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
fi

# Wait for the public URL to appear in the log.
log "Establishing tunnel (waiting for public URL) ..."
url=""
for _ in $(seq 1 30); do
    if ! kill -0 "$CF_PID" 2>/dev/null; then
        echo; tail -20 "$LOG_FILE" 2>&1 || true
        die "cloudflared exited during startup (see $LOG_FILE)."
    fi
    url="$(current_url)"
    [[ -n "$url" ]] && break
    sleep 1
done
[[ -n "$url" ]] || die "Timed out waiting for a trycloudflare URL. See $LOG_FILE."

print_friend_banner "$url" "$API_KEY"

if [[ "$DETACH" -eq 1 ]]; then
    disown "$CF_PID" 2>/dev/null || true
    log "Detached. Tunnel keeps running in the background (pid $CF_PID)."
    log "  URL again: ./share-kimik3.sh status"
    log "  Stop:      ./share-kimik3.sh stop"
    exit 0
fi

log "Tunnel is up. Press Ctrl-C to stop and take the endpoint offline."
wait "$CF_PID"
