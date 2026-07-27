#!/usr/bin/env bash
#
# check-k3-readiness.sh — is Kimi K3 actually servable yet?
#
# As of 27 Jul 2026 the answer was no, on two independent counts: Moonshot had
# not published the weights, and neither vLLM nor SGLang had K3 model code. This
# script re-checks both, plus the quantized variants and the current container
# tags, so you know the moment the picture changes.
#
# Needs only curl + python3 and outbound internet. No GPU, no Apptainer — run it
# from your laptop or a Bunya login node.
#
# Usage:
#   ./check-k3-readiness.sh            full report
#   ./check-k3-readiness.sh --quiet    one-line verdict (for cron / polling)
#
# Exit status:
#   0  weights published AND an engine registers the K3 architecture -> go
#   1  not ready yet
#   2  the check itself failed (no network, HF/GitHub unreachable)

set -uo pipefail

QUIET=0
[[ "${1:-}" == "--quiet" || "${1:-}" == "-q" ]] && QUIET=1

MODEL_REPO="${K3_REPO:-moonshotai/Kimi-K3}"

say()  { [[ "$QUIET" -eq 1 ]] || printf '%s\n' "$*"; }
head_() { [[ "$QUIET" -eq 1 ]] || printf '\n\033[1;36m── %s \033[0m%s\n' "$*" "$(printf '─%.0s' {1..40})"; }
ok()   { [[ "$QUIET" -eq 1 ]] || printf '  \033[1;32mOK\033[0m   %s\n' "$*"; }
no()   { [[ "$QUIET" -eq 1 ]] || printf '  \033[1;31mNO\033[0m   %s\n' "$*"; }
info() { [[ "$QUIET" -eq 1 ]] || printf '       %s\n' "$*"; }

command -v curl    >/dev/null 2>&1 || { echo "curl not found on PATH" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found on PATH" >&2; exit 2; }

HF="https://huggingface.co"
GH_RAW="https://raw.githubusercontent.com"
# -L matters: HF answers /resolve/main/<file> with a 307 to its CDN. Without
# following it we'd report "not published" for a model that is in fact released.
# (curl drops the Authorization header on the cross-host hop, which is correct —
# the CDN URL is already signed.)
UA=(-sSL -m 25 -H "User-Agent: kimik3-bunya-readiness/1")
[[ -n "${HF_TOKEN:-}" ]] && UA+=(-H "Authorization: Bearer $HF_TOKEN")

WEIGHTS_READY=0
ENGINE_READY=0
K3_ARCHS=""

if [[ "$QUIET" -eq 0 ]]; then
cat <<EOF
============================================================================
  Kimi K3 readiness check          $(date '+%Y-%m-%d %H:%M:%S %Z')
  Target: $MODEL_REPO  on Bunya MI355X (8x gfx950, 288 GB each)
============================================================================
EOF
fi

# ── 1. Are the weights published? ───────────────────────────────────────────

head_ "1. Weights on Hugging Face"

cfg="$(curl "${UA[@]}" "$HF/$MODEL_REPO/resolve/main/config.json" 2>/dev/null)"
if printf '%s' "$cfg" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
    WEIGHTS_READY=1
    ok "$MODEL_REPO has a config.json — weights are published."
    K3_ARCHS="$(printf '%s' "$cfg" | python3 -c '
import json, sys
c = json.load(sys.stdin)
archs = c.get("architectures") or []
print(",".join(archs))
' 2>/dev/null)"
    if [[ "$QUIET" -eq 0 ]]; then
        printf '%s' "$cfg" | python3 -c '
import json, sys
c = json.load(sys.stdin)
for k in ("architectures", "model_type", "num_hidden_layers", "hidden_size",
          "num_experts", "n_routed_experts", "num_experts_per_tok",
          "max_position_embeddings", "torch_dtype"):
    if k in c:
        print(f"       {k:26}: {c[k]}")
q = c.get("quantization_config")
if isinstance(q, dict):
    q = q.get("quant_method", q)
if q:
    label = "quantization"
    print(f"       {label:26}: {q}")
' 2>/dev/null
    fi

    # Total weight size from the safetensors index.
    idx="$(curl "${UA[@]}" "$HF/$MODEL_REPO/resolve/main/model.safetensors.index.json" 2>/dev/null)"
    sz="$(printf '%s' "$idx" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    n = d.get("metadata", {}).get("total_size")
    print(f"{n/1e9:.0f} GB" if n else "")
except Exception:
    print("")
' 2>/dev/null)"
    [[ -n "$sz" ]] && info "total weight size          : $sz  (node HBM = 2304 GB)"
else
    no "$MODEL_REPO has no readable config.json — weights not published (or gated to you)."
    info "If you believe it is released, export HF_TOKEN and accept any licence at"
    info "  $HF/$MODEL_REPO"
fi

# ── 2. Quantized variants (what we actually want to serve) ──────────────────

head_ "2. Quantized K3 variants"

vars_json="$(curl "${UA[@]}" "$HF/api/models?search=Kimi-K3&limit=100&sort=downloads" 2>/dev/null)"
found="$(printf '%s' "$vars_json" | python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)
hits = []
for m in rows:
    mid = m.get("modelId") or m.get("id") or ""
    low = mid.lower()
    if "k3" not in low:
        continue
    if any(t in low for t in ("mxfp4", "fp8", "int4", "awq", "gptq", "nvfp4")) \
       or low.startswith(("amd/", "moonshotai/")):
        hits.append(f"{mid}   (updated {(m.get('lastModified') or '')[:10]})")
for h in sorted(set(hits))[:20]:
    print(h)
' 2>/dev/null)"

if [[ -n "$found" ]]; then
    while IFS= read -r line; do ok "$line"; done <<<"$found"
else
    no "No K3 quantizations found (looking for amd/*, MXFP4, FP8, INT4, AWQ)."
fi
info "MXFP4 is what we want: ~1.5 TB fits one node; FP8 (~2.8 TB) does not."

# ── 3. Engine support (does the model code exist?) ──────────────────────────

head_ "3. Engine support"

# Grep an engine's architecture list for the K3 architecture name. If we don't
# know the real name yet (weights unpublished), fall back to listing Kimi entries
# so you can see new ones appear.
#
# vLLM keeps an explicit registry.py mapping arch name -> module. SGLang instead
# auto-discovers models from its models/ directory and reads EntryClass from each
# file, so for SGLang we concatenate its kimi*.py sources and look there.
check_registry() {
    local name="$1" reg="$2"
    if [[ -z "$reg" ]]; then
        no "$name: could not fetch its model list."
        return 1
    fi

    local kimi
    kimi="$(printf '%s' "$reg" | grep -oiE '"[A-Za-z0-9_]*(Kimi|Moonshot)[A-Za-z0-9_]*"' \
            | tr -d '"' | sort -u | paste -sd' ' -)"
    info "$name Kimi-family architectures: ${kimi:-<none>}"

    if [[ -n "$K3_ARCHS" ]]; then
        local hit=0
        IFS=',' read -ra arr <<<"$K3_ARCHS"
        for a in "${arr[@]}"; do
            [[ -z "$a" ]] && continue
            if printf '%s' "$reg" | grep -q "\"$a\""; then
                ok "$name registers '$a'"
                hit=1
            else
                no "$name does NOT register '$a'"
            fi
        done
        return $(( hit ? 0 : 1 ))
    fi

    # No published arch name to match. Heuristic: anything Kimi + "K3".
    if printf '%s' "$reg" | grep -qiE '"[A-Za-z0-9_]*Kimi[A-Za-z0-9_]*K3[A-Za-z0-9_]*"'; then
        ok "$name has a Kimi*K3* architecture registered."
        return 0
    fi
    no "$name has no K3 architecture registered."
    return 1
}

# List the kimi*.py model files each engine has on main. Also the raw material
# for SGLang's architecture check below.
kimi_files() {
    curl -sS -m 25 "https://api.github.com/repos/$1/contents/$2?ref=main" 2>/dev/null \
    | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
if isinstance(d, list):
    print(" ".join(f["name"] for f in d if "kimi" in f["name"].lower()))
' 2>/dev/null
}

VLLM_MODELS_PATH="vllm/model_executor/models"
SGLANG_MODELS_PATH="python/sglang/srt/models"

vl_files="$(kimi_files vllm-project/vllm      "$VLLM_MODELS_PATH")"
sg_files="$(kimi_files sgl-project/sglang     "$SGLANG_MODELS_PATH")"

# vLLM: registry.py is authoritative.
vllm_reg="$(curl -sS -m 25 "$GH_RAW/vllm-project/vllm/main/$VLLM_MODELS_PATH/registry.py" 2>/dev/null)"
if check_registry "vLLM  " "$vllm_reg"; then
    ENGINE_READY=1
    READY_ENGINE="vllm"
fi
[[ -n "$vl_files" ]] && info "vLLM   models/kimi*: $vl_files"

# SGLang: concatenate its kimi*.py sources and look for architecture classes.
sglang_src=""
for f in $sg_files; do
    sglang_src+="$(curl -sS -m 25 "$GH_RAW/sgl-project/sglang/main/$SGLANG_MODELS_PATH/$f" 2>/dev/null)"$'\n'
done
# Quote the class names so the same '"Arch"' grep pattern works on both engines.
sglang_reg="$(printf '%s' "$sglang_src" \
    | grep -oE '^class [A-Za-z0-9_]+' | awk '{print "\""$2"\""}' | sort -u)"
if check_registry "SGLang" "$sglang_reg"; then
    ENGINE_READY=1
    READY_ENGINE="${READY_ENGINE:-sglang}"
fi
[[ -n "$sg_files" ]] && info "SGLang models/kimi*: $sg_files"

# ── 4. Container images to try ──────────────────────────────────────────────

head_ "4. Current container tags (candidates for kimik3.env)"

list_tags() {
    local repo="$1" filter="$2" n="$3"
    curl -sS -m 25 "https://hub.docker.com/v2/repositories/$repo/tags?page_size=100&ordering=last_updated" 2>/dev/null \
    | python3 -c "
import json, sys, re
flt = re.compile(r'''$filter''') if '''$filter''' else None
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
out = []
for t in d.get('results', []):
    n = t['name']
    if flt and not flt.search(n):
        continue
    out.append(f\"       {n}   ({t['last_updated'][:10]})\")
print('\n'.join(out[:$n]) or '       <none matched>')
"
}

if [[ "$QUIET" -eq 0 ]]; then
    say "  rocm/vllm (release):"
    list_tags "rocm/vllm" "cdna|gfx950" 4
    say "  rocm/vllm-dev (nightly base builds):"
    list_tags "rocm/vllm-dev" "^(base|nightly)" 4
    say "  lmsysorg/sglang-rocm (mi35x = gfx950):"
    list_tags "lmsysorg/sglang-rocm" "mi35x" 4

    say ""
    info "Once weights + an engine exist, confirm YOUR image can load the model with:"
    info "  ./serve-kimik3.sh check          (runs inside the .sif, on the compute node)"
fi

# ── Verdict ─────────────────────────────────────────────────────────────────

head_ "Verdict"

if [[ "$WEIGHTS_READY" -eq 1 && "$ENGINE_READY" -eq 1 ]]; then
    if [[ "$QUIET" -eq 1 ]]; then
        echo "READY: $MODEL_REPO published and ${READY_ENGINE:-an engine} registers ${K3_ARCHS:-a K3 arch}"
    else
        printf '  \033[1;32mGO.\033[0m Weights are published and %s registers %s.\n' \
               "${READY_ENGINE:-an engine}" "${K3_ARCHS:-a K3 architecture}"
        cat <<'EOF'

  Next steps:
    1. Set MODEL_ID (prefer an MXFP4 build) and ENGINE in kimik3.env.
    2. Pick a container image from section 4 that is NEWER than the commit
       that added K3 support, then: ./serve-kimik3.sh pull
    3. ./serve-kimik3.sh check       -- confirm the image knows the architecture
    4. ./serve-kimik3.sh parsers     -- confirm the tool/reasoning parser names
    5. ./serve-kimik3.sh download    -- ~1.5 TB, no GPU needed
    6. ./serve-kimik3.sh serve --detach
EOF
    fi
    exit 0
fi

if [[ "$QUIET" -eq 1 ]]; then
    echo "NOT READY: weights=$WEIGHTS_READY engine=$ENGINE_READY ($MODEL_REPO)"
else
    printf '  \033[1;33mNot ready yet.\033[0m  weights=%s   engine=%s\n' \
           "$( ((WEIGHTS_READY)) && echo yes || echo no )" \
           "$( ((ENGINE_READY))  && echo yes || echo no )"
    cat <<'EOF'

  Both must be yes. In the meantime the stand-in config in kimik3-env.example
  (amd/Kimi-K2.6-MXFP4) exercises the identical gfx950 FP4 MoE path, so you can
  validate the whole stack today and flip MODEL_ID on the day.

  Poll this from a login node with:
    watch -n 1800 ./check-k3-readiness.sh --quiet
EOF
fi
exit 1
