#!/usr/bin/env bash
#
# kimicode-setup.sh — point kimicode (Kimi Code CLI) at the Kimi K3 endpoint on Bunya.
#
# Run this wherever kimicode runs: the GPU node itself (bun159/160/161), the
# login node, or your laptop (with an SSH tunnel to the node).
#
# Usage:
#   ./kimicode-setup.sh                          # endpoint on localhost:30000
#   ./kimicode-setup.sh --host bun159 --port 30000
#   ./kimicode-setup.sh --host localhost --api-key sk-...   # laptop w/ tunnel
#
# This targets Kimi Code CLI (the Node.js rewrite, ~/.kimi-code/config.toml), NOT
# the legacy Python kimi-cli (~/.kimi/config.toml, provider type "openai_legacy").
# Both install a binary called `kimi`. See README Step 6b.
#
# Unlike opencode, Kimi Code's TOML has no documented {env:VAR} interpolation, so
# the key is written literally and the config is created with umask 077. If you
# would rather keep no key on disk, use --no-key and the KIMI_MODEL_* environment
# variables instead (README Step 6b).
#
# The advertised context limit and model name are taken from CONTEXT_LEN and
# SERVED_MODEL_NAME in kimik3.env, so they stay in step with what the server is
# actually configured to accept. Override with --context / --model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/kimicode.kimik3.toml"

# Pick up defaults from kimik3.env if it's here (cluster-side).
if [[ -f "$SCRIPT_DIR/kimik3.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/kimik3.env"
fi

HOST="localhost"
PORT="${PORT:-30000}"
MODEL="${SERVED_MODEL_NAME:-kimi-k3}"
# CONTEXT_LEN is empty by default (server uses the model max), so advertise
# K3's full 1M window to kimicode unless it has been explicitly capped.
CONTEXT="${CONTEXT_LEN:-}"
CONTEXT="${CONTEXT:-1048576}"
API_KEY="${KIMIK3_API_KEY:-}"
NO_KEY=0
CONFIG_PATH="${KIMI_CODE_HOME:-$HOME/.kimi-code}/config.toml"

# Sentinels must match the template, and must start at column 1.
BEGIN_MARK='# >>> kimik3-bunya'
END_MARK='# <<< kimik3-bunya'

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --context)   CONTEXT="$2"; shift 2 ;;
        --api-key)   API_KEY="$2"; shift 2 ;;
        --no-key)    NO_KEY=1; shift ;;
        --config)    CONFIG_PATH="$2"; shift 2 ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -f "$TEMPLATE" ]] || die "Template not found: $TEMPLATE"

# Resolve the API key if not given: try the persisted key file on scratch.
if [[ "$NO_KEY" -eq 0 ]]; then
    if [[ -z "$API_KEY" && -n "${MODEL_CACHE_DIR:-}" && -r "$MODEL_CACHE_DIR/kimik3-api-key" ]]; then
        API_KEY="$(<"$MODEL_CACHE_DIR/kimik3-api-key")"
    fi
    if [[ -z "$API_KEY" ]]; then
        die "No API key found. Export KIMIK3_API_KEY, pass --api-key, use --no-key, or run on the cluster where \$MODEL_CACHE_DIR/kimik3-api-key exists.
On the cluster:  cat \$MODEL_CACHE_DIR/kimik3-api-key"
    fi
else
    API_KEY=""
fi

# Build the provider TOML from the template. Bash substitution rather than sed:
# the key is arbitrary text and must not be re-interpreted.
provider_toml="$(<"$TEMPLATE")"
provider_toml="${provider_toml//__HOST__/$HOST}"
provider_toml="${provider_toml//__PORT__/$PORT}"
provider_toml="${provider_toml//__MODEL__/$MODEL}"
provider_toml="${provider_toml//__CONTEXT__/$CONTEXT}"
provider_toml="${provider_toml//__APIKEY__/$API_KEY}"

ALIAS="kimik3-bunya/$MODEL"

umask 077
mkdir -p "$(dirname "$CONFIG_PATH")"

if [[ ! -f "$CONFIG_PATH" ]]; then
    printf '%s\n' "$provider_toml" > "$CONFIG_PATH"
    echo "Wrote new kimicode config: $CONFIG_PATH"
else
    cp "$CONFIG_PATH" "$CONFIG_PATH.bak"
    tmp="$(mktemp)"
    # Strip any previously managed block, keeping everything else untouched,
    # then re-append. Appending is TOML-safe because the block is all tables.
    awk -v s="$BEGIN_MARK" -v e="$END_MARK" '
        index($0, s) == 1 { skip = 1 }
        !skip             { print }
        index($0, e) == 1 { skip = 0 }
    ' "$CONFIG_PATH" \
    | awk 'NF { last = NR } { line[NR] = $0 } END { for (i = 1; i <= last; i++) print line[i] }' \
    > "$tmp"
    # Exactly one blank line before the block, however many times this is re-run.
    if [[ -s "$tmp" ]]; then
        printf '\n' >> "$tmp"
    fi
    printf '%s\n' "$provider_toml" >> "$tmp"
    mv "$tmp" "$CONFIG_PATH"
    if grep -qF "$BEGIN_MARK" "$CONFIG_PATH.bak"; then
        echo "Replaced provider 'kimik3-bunya' in $CONFIG_PATH (backup at $CONFIG_PATH.bak)"
    else
        echo "Added provider 'kimik3-bunya' to $CONFIG_PATH (backup at $CONFIG_PATH.bak)"
    fi
fi

echo
echo "Model alias '$ALIAS' advertised with a ${CONTEXT}-token context limit."
if [[ "$NO_KEY" -eq 1 ]]; then
    echo "No key written. Start kimicode with the environment route instead:"
    echo "  export KIMI_MODEL_NAME=\"$MODEL\""
    echo "  export KIMI_MODEL_PROVIDER_TYPE=openai"
    echo "  export KIMI_MODEL_BASE_URL=\"http://$HOST:$PORT/v1\""
    echo "  export KIMI_MODEL_API_KEY=\"<key>\""
    echo "  export KIMI_MODEL_MAX_CONTEXT_SIZE=\"$CONTEXT\""
else
    echo "API key written into the config (mode 600)."
fi
if [[ "$HOST" == "localhost" || "$HOST" == "127.0.0.1" ]]; then
    echo
    echo "If kimicode runs on a different machine than the server, keep a tunnel open:"
    echo "  ssh -N -L $PORT:<gpu-node>:$PORT \${USER}@bunya1.rcc.uq.edu.au   # <gpu-node> = bun159/160/161"
fi
echo
echo "Then start kimicode against it:"
echo "  kimi -m $ALIAS"
