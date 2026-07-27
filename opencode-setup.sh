#!/usr/bin/env bash
#
# opencode-setup.sh — point opencode at the Kimi K3 endpoint on Bunya.
#
# Run this wherever opencode runs: the GPU node itself (bun159/160/161), the
# login node, or your laptop (with an SSH tunnel to the node).
#
# Usage:
#   ./opencode-setup.sh                          # endpoint on localhost:30000
#   ./opencode-setup.sh --host bun159 --port 30000
#   ./opencode-setup.sh --host localhost --api-key sk-...   # laptop w/ tunnel
#
# The generated provider references {env:KIMIK3_API_KEY}, so opencode reads
# the key from your environment at runtime (nothing secret in the config).
# Pass --embed-key to write the key literally instead (single-user machines).
#
# The advertised context limit and model name are taken from CONTEXT_LEN and
# SERVED_MODEL_NAME in kimik3.env, so they stay in step with what the server is
# actually configured to accept. Override with --context / --model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/opencode.kimik3.json"

# Pick up defaults from kimik3.env if it's here (cluster-side).
if [[ -f "$SCRIPT_DIR/kimik3.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/kimik3.env"
fi

HOST="localhost"
PORT="${PORT:-30000}"
MODEL="${SERVED_MODEL_NAME:-kimi-k3}"
# CONTEXT_LEN is empty by default (server uses the model max), so advertise
# K3's full 1M window to opencode unless it has been explicitly capped.
CONTEXT="${CONTEXT_LEN:-}"
CONTEXT="${CONTEXT:-1048576}"
API_KEY="${KIMIK3_API_KEY:-}"
EMBED_KEY=0
CONFIG_PATH="${OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --context)   CONTEXT="$2"; shift 2 ;;
        --api-key)   API_KEY="$2"; shift 2 ;;
        --embed-key) EMBED_KEY=1; shift ;;
        --config)    CONFIG_PATH="$2"; shift 2 ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -f "$TEMPLATE" ]] || die "Template not found: $TEMPLATE"

# Resolve the API key if not given: try the persisted key file on scratch.
if [[ -z "$API_KEY" && -n "${MODEL_CACHE_DIR:-}" && -r "$MODEL_CACHE_DIR/kimik3-api-key" ]]; then
    API_KEY="$(<"$MODEL_CACHE_DIR/kimik3-api-key")"
fi
if [[ -z "$API_KEY" ]]; then
    die "No API key found. Export KIMIK3_API_KEY, pass --api-key, or run on the cluster where \$MODEL_CACHE_DIR/kimik3-api-key exists.
On the cluster:  cat \$MODEL_CACHE_DIR/kimik3-api-key"
fi

# Build the provider JSON from the template.
provider_json="$(sed -e "s/__HOST__/$HOST/" \
                     -e "s/__PORT__/$PORT/" \
                     -e "s/__MODEL__/$MODEL/" \
                     -e "s/__CONTEXT__/$CONTEXT/" "$TEMPLATE")"
if [[ "$EMBED_KEY" -eq 1 ]]; then
    provider_json="${provider_json//\{env:KIMIK3_API_KEY\}/$API_KEY}"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    mkdir -p "$(dirname "$CONFIG_PATH")"
    printf '%s\n' "$provider_json" > "$CONFIG_PATH"
    echo "Wrote new opencode config: $CONFIG_PATH"
elif command -v jq >/dev/null 2>&1; then
    # Merge just our provider entry into the existing config, preserving
    # everything else (including other providers).
    tmp="$(mktemp)"
    jq --argjson new "$provider_json" \
       '.provider = ((.provider // {}) + $new.provider)' \
       "$CONFIG_PATH" > "$tmp"
    cp "$CONFIG_PATH" "$CONFIG_PATH.bak"
    mv "$tmp" "$CONFIG_PATH"
    echo "Merged provider 'kimik3-bunya' into $CONFIG_PATH (backup at $CONFIG_PATH.bak)"
else
    echo "Existing config found at $CONFIG_PATH and 'jq' is not available for a safe merge."
    echo "Add this block under the top-level \"provider\" object yourself:"
    echo
    printf '%s\n' "$provider_json"
    exit 0
fi

echo
echo "Model '$MODEL' advertised with a ${CONTEXT}-token context limit."
if [[ "$EMBED_KEY" -eq 1 ]]; then
    echo "API key embedded in the config."
else
    echo "Before starting opencode, export the key in that shell:"
    echo "  export KIMIK3_API_KEY=\"$API_KEY\""
fi
if [[ "$HOST" == "localhost" || "$HOST" == "127.0.0.1" ]]; then
    echo
    echo "If opencode runs on a different machine than the server, keep a tunnel open:"
    echo "  ssh -N -L $PORT:<gpu-node>:$PORT \${USER}@bunya1.rcc.uq.edu.au   # <gpu-node> = bun159/160/161"
fi
echo
echo "Then restart opencode and select the model via /models -> 'Kimi K3 (Bunya MI355X)'."
