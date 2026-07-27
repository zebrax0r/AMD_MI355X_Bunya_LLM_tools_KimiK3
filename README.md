# Kimi K3 on Bunya's AMD MI355X — vLLM/SGLang serving for opencode

One-click serving of Moonshot's **Kimi K3** (2.8T-parameter MoE, Kimi Delta
Attention, 1M context) on a **Bunya MI355X node** — 8× gfx950, 288 GB HBM each,
2,304 GB total — inside an Apptainer container, exposed as an OpenAI-compatible
endpoint and wired into [opencode](https://opencode.ai).

Same shape as the [GLM-5.2 sibling repo](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_GLM5.2),
with one structural difference: **the engine is selectable** (`ENGINE=vllm` or
`ENGINE=sglang`), because K3 is new enough that which engine gets there first is
not yet settled.

---

## ⚠️ K3 readiness status — checked 27 Jul 2026

**Kimi K3 cannot be served by anything yet.** Two independent blockers:

| Thing | Status |
|---|---|
| `moonshotai/Kimi-K3` weights | ❌ **Not published.** The HF page is an "Upcoming release" placeholder; the org's newest published model is `Kimi-K2.7-Code`. Moonshot promised open weights "by 27 Jul 2026". |
| `amd/Kimi-K3-MXFP4` | ❌ **Does not exist.** AMD has shipped MXFP4 builds of K2, K2.5 and K2.6 — not K3. |
| vLLM K3 support | ❌ **Not in code.** `v0.26.0` (released 27 Jul 2026) ships `kimi_k25.py`, `kimi_linear.py`, `kimi_vl.py`, `kimi_audio.py` — no `kimi_k3.py`, no registry entry. The only K3 signal is a CI auto-labeling PR (#49895) merged 26 Jul. |
| SGLang K3 support | ❌ **No activity at all.** Zero PRs or issues matching "Kimi K3". |

Refresh this picture any time — no GPU, no container, works from your laptop:

```bash
./check-k3-readiness.sh          # full report; exit 0 only when it's genuinely go
./check-k3-readiness.sh --quiet  # one line, for cron/polling
```

**So what is this repo for right now?** Everything except the model itself is
ready and testable. The default config serves **`amd/Kimi-K2.6-MXFP4`** (550B,
MXFP4 experts, built for MI350/MI355 + ROCm 7.2, ungated) as a stand-in. It is
the same Moonshot lineage (559 GB, comfortably one node) and drives the
*identical* gfx950 FP4 MoE kernel path
K3 will, so you can validate the entire stack — Apptainer, SLURM, FP4 kernels,
tool calling, opencode, benchmarking — today, and change one line on the day.
See [Switching to K3](#switching-to-k3).

**Why vLLM is the default.** The [22 Jul vLLM blog](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
previews production-scale K3 support: FlashKDA and Triton KDA prefill paths, a
fused decode kernel, MXFP4 TRTLLM-Gen/DeepGEMM paths, and — the part that
matters here — **AMD contributing initial ROCm support via FlyDSL's MLIR kernel
stack with A16W4/A8W4 fused operators**. Both engines already carry a Kimi Delta
Attention implementation inherited from Kimi Linear (vLLM's
`KimiGatedDeltaNetAttention`, SGLang's `kimi_linear.py`), so the hard kernel
groundwork exists on both sides — but the K3-specific work (Attention Residuals,
896-expert Stable LatentMoE, native vision) is landing in vLLM. If SGLang gets
there first, flip `ENGINE=sglang` and nothing else changes.

---

## What gets served

| | |
|---|---|
| Model | `$MODEL_ID` — default `amd/Kimi-K2.6-MXFP4` (stand-in), served under the name `kimi-k3` |
| Engine | vLLM (default) or SGLang, pinned ROCm image, selected with `ENGINE` |
| Parallelism | **TP8** across the node's 8 GPUs (optional DP / expert-parallel — see below) |
| Context | 131,072 tokens by default (`CONTEXT_LEN`; K3 supports up to 1M — see the memory budget) |
| KV cache | FP8 (`KV_CACHE_DTYPE=fp8`, mapped to each engine's spelling) |
| Tool calling / thinking | `TOOL_PARSER` / `REASONING_PARSER`, default `kimi_k2` — required for opencode's agentic loop |
| Auth | Bearer API key, auto-generated and persisted to `$MODEL_CACHE_DIR/kimik3-api-key` |

### Memory budget — why 131072 and not 1M

```
node HBM            8 × 288 GB   = 2304 GB
K3 @ MXFP4          estimated    ≈ 1600 GB   (~200 GB/GPU at TP8)
headroom                         ≈  700 GB   (~ 88 GB/GPU)
                                 -> KV cache + activations + graph capture
```

**Where the 1.6 TB comes from, and why it isn't 1.5 TB.** A naive
`2.8T × 4.25 bits` gives ~1.49 TB, but that is wrong for these builds: only the
`experts`/`shared_experts` are quantized to FP4, while attention, embeddings and
`lm_head` stay at higher precision. Measured against the real thing —
`amd/Kimi-K2.6-MXFP4` reports **559 GB** in its safetensors index — the true
ratio is close to double the naive figure for the non-expert remainder. Scaling
that to 2.8T lands around 1.6 TB. Treat it as an estimate: the moment K3's
weights are published, `./check-k3-readiness.sh` prints the *actual* total from
its index, and that number should replace this one.

It fits on one node, with real but not generous headroom. **FP8 (~2.8 TB) and
BF16 (~5.6 TB) do not fit at all**, so MXFP4 is mandatory here — the same
conclusion as GLM-5.2, but with far less slack. Helpfully, Moonshot state that
K3 was quantization-aware-trained with MXFP4 weights and MXFP8 activations, so
the official release may already be FP4-native and no `amd/` requant may be
needed. KDA's fixed-size linear-attention state also makes long context much
cheaper than full attention would, so there is likely real room above 131072 —
find it by measurement, not by hoping.

**If K3 turns out materially larger than 1.6 TB, single-node serving is off** and
you'll need a 2-node (16-GPU) layout — which this repo does not currently ship.
The environment variables are structured so that adding it is additive rather
than a rewrite, but budget for that possibility.

The stand-in is more comfortable: 559 GB is ~70 GB/GPU, leaving most of the node
for KV cache.

---

## Bunya specifics you need to know

- **Apptainer, not podman**, and it lives **only on compute nodes** — never the
  login nodes. So `pull`, `download`, `check`, `parsers` and `serve` all run
  inside a `salloc`/`sbatch` allocation. (`check-k3-readiness.sh`, `stop` and
  `status` work anywhere.)
- **The MI355X nodes are `bun159`, `bun160`, `bun161`**, and they currently sit
  in the **`admin_test`** partition (not `gpu_rocm`). Submit with
  `--partition=admin_test --account=a_rcc --qos=sdf --gres=gpu:mi355x:8` — the
  scheduler lands you on whichever is free; the serve banner prints the actual
  hostname to use for tunnels.
- **Run from `/scratch`** (`/scratch/user/$USER/...`), not `/home` (tight quota)
  and not `/QRISdata` (RDM — jobs can't be submitted from there). Check quota
  with `rquota`.
- **GPU passthrough is `--rocm`** — Apptainer binds `/dev/kfd`, `/dev/dri`, and
  the ROCm libraries automatically. No `--device`/`--group-add`/`--shm-size`
  needed; the container shares the host network and IPC (so the port and
  `/dev/shm` just work).

Confirm the live SLURM recipe any time with:

```bash
sinfo -o "%P %.10l %G %f" | grep mi355x       # partition + gres label
sacctmgr -np show assoc user=$USER format=account,qos   # your account + QoS
scontrol show node bun161 | grep -iE 'Gres|Partitions|State'
```

---

## Prerequisites

- **A Bunya account** with the `a_rcc` scheduling account and access to the
  `admin_test` partition / `sdf` QoS that reaches the MI355X nodes (a special
  allocation — confirm with `rcc-support@uq.edu.au` if `salloc` is rejected).
- **Scratch space** on `/scratch/user/$USER`, plus the container `.sif` (~30 GB):
  - **~650 GB free** for the `Kimi-K2.6-MXFP4` stand-in (559 GB of weights, measured)
  - **~1.7 TB free** for Kimi K3 (~1.6 TB of MXFP4 weights, estimated)

  Check with `rquota`. If your scratch allocation can't hold 1.7 TB, sort that
  out with RCC *before* K3 lands — it is the longest-lead item here.
- **A HuggingFace account + access token** (read scope), from
  <https://huggingface.co/settings/tokens>.
- **`opencode`** wherever you want to *use* the model (the node, the login node,
  or your laptop) — see <https://opencode.ai>.
- Optional, for a nicer opencode config merge: **`jq`** on the machine running
  `opencode-setup.sh`.

---

## Repository contents

| File | Purpose |
|---|---|
| `serve-kimik3.sh` | The core script: preflight, pull, download, check, parsers, serve, stop, status |
| `serve-kimik3.sbatch` | SLURM batch wrapper around `serve-kimik3.sh serve` (MI355X recipe) |
| `kimik3-env.example` | Config template — copy to `kimik3.env` and edit |
| `check-k3-readiness.sh` | Are the weights out? Does an engine support K3? Which images to try? |
| `opencode-setup.sh` | Writes/merges the opencode provider config on any machine |
| `opencode.kimik3.json` | The provider template `opencode-setup.sh` fills in |
| `share-kimik3.sh` | Optional: public HTTPS tunnel via Cloudflare for users without SSH |
| `bench-kimik3.sh` | Benchmark tok/s / TTFT / ITL to compare configs — see [Performance tuning](#performance-tuning) |
| `README.md` | This file |

Secrets never live in the repo: `kimik3.env` (your HF token), the generated API
key, and the `.sif` are gitignored / stored under `$MODEL_CACHE_DIR` on scratch.

---

## Walkthrough

### Step 0 — Get the code onto Bunya

```bash
cd /scratch/user/$USER
git clone <this-repo-url> kimik3-bunya
cd kimik3-bunya
```

### Step 1 — Check whether K3 is out yet

```bash
./check-k3-readiness.sh
```

Works on the login node. If it says **not ready**, carry on with the stand-in —
everything below is identical either way.

### Step 2 — Configure

```bash
cp kimik3-env.example kimik3.env
$EDITOR kimik3.env
```

At minimum set two values:

- `MODEL_CACHE_DIR` — an absolute scratch path, e.g.
  `/scratch/user/$USER/kimik3/hf-cache`. This holds the weights, the `.sif`, the
  API key, and the server log/PID. It must be writable from the compute node.
- `HF_TOKEN` — your HuggingFace token. (Or leave it blank and set
  `HF_TOKEN_FILE` to a path containing just the token.)

Everything else defaults to the correct MI355X recipe (see
[Configuration reference](#configuration-reference)).

### Step 3 — Allocate an MI355X node

```bash
salloc --partition=admin_test --account=a_rcc --qos=sdf \
       --nodes=1 --gres=gpu:mi355x:8 --ntasks-per-node=1 \
       --cpus-per-task=192 --mem=1800G --time=8:00:00
```

Apptainer only exists here, not on the login node.

### Step 4 — Build the image + fetch the weights (first time)

```bash
./serve-kimik3.sh pull        # builds the .sif (one-time, multi-GB)
./serve-kimik3.sh check       # can this image load this model?
./serve-kimik3.sh parsers     # what tool/reasoning parsers does it have?
./serve-kimik3.sh download    # prefetch weights (no GPU needed)
```

`check` reads the model's `architectures` from its `config.json` and asks the
engine's own registry inside the container whether it knows them. On K3 flip day
this is the question that matters, and it answers it in about ten seconds
instead of after a two-hour download.

### Step 5 — Serve

Interactive:

```bash
./serve-kimik3.sh serve --detach     # returns your shell once healthy
```

Or as a batch job:

```bash
mkdir -p logs && sbatch serve-kimik3.sbatch
# once RUNNING, read node + endpoint + key from the job log:
grep -A 24 'is up and serving' logs/kimik3-<jobid>.out
```

### Step 6 — Verify it's serving

```bash
export KIMIK3_API_KEY="$(cat $MODEL_CACHE_DIR/kimik3-api-key)"
curl -s http://127.0.0.1:30000/v1/models -H "Authorization: Bearer $KIMIK3_API_KEY"

curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Authorization: Bearer $KIMIK3_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"In one sentence: what is Bunya?"}]}'
```

Read the reply. **Coherence is a pass/fail criterion, not a nicety** — see the
gibberish note in [Troubleshooting](#notes--troubleshooting).

### Step 7 — Connect opencode

On the GPU node (attach a second shell to the job first):

```bash
srun --overlap --jobid <jobid> --pty /bin/bash -l
cd /scratch/user/$USER/kimik3-bunya
./opencode-setup.sh --host localhost --port 30000
export KIMIK3_API_KEY="$(cat $MODEL_CACHE_DIR/kimik3-api-key)"
opencode
```

From your laptop, keep a tunnel open and point at localhost:

```bash
# terminal 1 (<node> = the hostname the serve banner printed):
ssh -N -L 30000:<node>:30000 $USER@bunya1.rcc.uq.edu.au
# terminal 2:
./opencode-setup.sh --host localhost --port 30000 --api-key <key>
```

Restart opencode, then pick **Kimi K3 (Bunya MI355X)** via `/models`.

### Step 8 — Shut down

```bash
./serve-kimik3.sh stop     # or Ctrl-C if attached, or scancel <jobid>
```

---

## Switching to K3

When `./check-k3-readiness.sh` exits 0, the change is small:

```diff
# kimik3.env
-export MODEL_ID="${MODEL_ID:-amd/Kimi-K2.6-MXFP4}"
+export MODEL_ID="${MODEL_ID:-moonshotai/Kimi-K3}"      # or amd/Kimi-K3-MXFP4
-export VLLM_IMAGE="${VLLM_IMAGE:-docker://rocm/vllm:rocm7.14.0_cdna_...}"
+export VLLM_IMAGE="${VLLM_IMAGE:-docker://rocm/vllm-dev:<nightly with K3>}"
-export MEM_FRACTION="${MEM_FRACTION:-0.85}"
+export MEM_FRACTION="${MEM_FRACTION:-0.92}"
```

Then work through this, in order — each step fails fast and cheaply:

1. **`./serve-kimik3.sh pull`** into a *new* `SIF_PATH` so the working stand-in
   image is preserved and you can revert instantly.
2. **`./serve-kimik3.sh check`** — does the image's registry know K3's
   architecture? If not, the image is too old; `check-k3-readiness.sh` section 4
   lists newer candidate tags. Don't download 1.6 TB until this passes.
3. **`./serve-kimik3.sh parsers`** — K3 will very likely register its own
   tool-call/reasoning parser rather than reusing `kimi_k2`. Set `TOOL_PARSER`
   and `REASONING_PARSER` from what the image actually reports, cross-checked
   against the model card.
4. **Confirm scratch has ~1.7 TB free** (`rquota`, `df -h $MODEL_CACHE_DIR`) —
   and check the real size that `./check-k3-readiness.sh` now reports — before
   `./serve-kimik3.sh download`.
5. **Serve, and expect to tune.** With weights at ~65% of HBM there is much less
   slack than the stand-in. If KV-cache allocation fails, back `CONTEXT_LEN`
   down (65536) or `MEM_FRACTION` up in small steps — not both at once.
6. **Raise `CONTEXT_LEN`** from measurement once it's stable. KDA should make
   long context cheap; find the real ceiling rather than assuming 1M works.
7. **Update `opencode.kimik3.json`'s advertised context** — or just re-run
   `./opencode-setup.sh`, which reads `CONTEXT_LEN` from `kimik3.env`.

---

## Sharing with someone who has no SSH access (optional)

`share-kimik3.sh` exposes the running endpoint over public HTTPS via a
**Cloudflare quick tunnel** — outbound-only, no root, no Cloudflare account. Run
it on the GPU node after the server is up:

```bash
./share-kimik3.sh share --detach     # prints https://<random>.trycloudflare.com
```

On first use it downloads `cloudflared` into `$MODEL_CACHE_DIR/cloudflared/`,
checks the local server is healthy, opens the tunnel, and prints a ready-to-paste
opencode provider block (with the public URL and API key) to hand over. Manage
with `./share-kimik3.sh status` / `stop`.

> ⚠️ Needs the compute node to have **outbound internet** (same path the weights
> download uses). If that's blocked, use the SSH tunnel in Step 7 instead.

> ⚠️ **The public URL + API key together grant full use of your model and your
> Bunya GPU-hours (billed to `a_rcc`).** Share the key over a private channel
> only, rotate it if it leaks (delete `$MODEL_CACHE_DIR/kimik3-api-key` and
> restart the server), and **check RCC's acceptable-use policy before exposing
> HPC compute externally** — the API key is the only gate.

---

## Script reference

```
./check-k3-readiness.sh          weights + engine + image status (no GPU, no container)
./check-k3-readiness.sh --quiet  one-line verdict for cron/polling

./serve-kimik3.sh [serve]        start serving (default), stays attached
./serve-kimik3.sh serve --detach start serving, wait until healthy, return the shell
./serve-kimik3.sh pull           build the .sif from the container image (one-time)
./serve-kimik3.sh download       prefetch weights only (no GPU)
./serve-kimik3.sh check          can this image load this model? (arch vs engine registry)
./serve-kimik3.sh parsers        list tool-call/reasoning parsers this image supports
./serve-kimik3.sh stop           stop the server (kills strays from both engines)
./serve-kimik3.sh status         server state + health check + /v1/models

./opencode-setup.sh [--host H] [--port P] [--model M] [--context N]
                    [--api-key K] [--embed-key] [--config PATH]

./share-kimik3.sh [share]        open a public HTTPS Cloudflare tunnel (add --detach)
./share-kimik3.sh stop | status

./bench-kimik3.sh [sweep]        measure tok/s: latency (c=1) + concurrency sweep
./bench-kimik3.sh latency        single-stream latency only
./bench-kimik3.sh throughput     saturate at BENCH_MAX_CONCURRENCY
```

Handy extras:

```bash
tail -f $MODEL_CACHE_DIR/kimik3-server.log          # follow startup / requests
head -3 $MODEL_CACHE_DIR/kimik3-server.log          # the exact launch command used
srun --overlap --jobid <jobid> --pty /bin/bash -l   # second shell on the serving node
```

---

## Configuration reference

All knobs live in `kimik3.env` (copied from `kimik3-env.example`). Anything you
`export` before running a script takes precedence over the file.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CACHE_DIR` | *(required)* | Scratch path for HF cache + `.sif` + API key + server log |
| `HF_TOKEN` / `HF_TOKEN_FILE` | *(required first run)* | HuggingFace token, inline or from a file |
| `ENGINE` | `vllm` | `vllm` or `sglang` |
| `MODEL_ID` | `amd/Kimi-K2.6-MXFP4` | Model repo to serve (the stand-in until K3 lands) |
| `SERVED_MODEL_NAME` | `kimi-k3` | Name clients use in the `model` field |
| `VLLM_IMAGE` | `rocm/vllm:rocm7.14.0_cdna_...vllm_0.23.0` | Container image when `ENGINE=vllm` |
| `SGLANG_IMAGE` | `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260726` | Container image when `ENGINE=sglang` |
| `SIF_PATH` | `$MODEL_CACHE_DIR/kimik3-<engine>-mi355x.sif` | Where the Apptainer image is stored (per-engine, so both coexist) |
| `APPTAINER_CACHEDIR` / `_TMPDIR` | *(near `$MODEL_CACHE_DIR`)* | Apptainer cache/scratch, kept off `/home` |
| `FLYDSL_CACHE_DIR` | `$MODEL_CACHE_DIR/flydsl-cache` | Writable dir bound over aiter's in-image FP4 JIT cache |
| `FLYDSL_CACHE_TARGET` | *(auto-detected)* | In-image path to bind over; only set if auto-detection is wrong |
| `KIMIK3_API_KEY` | *(auto-generated)* | Endpoint bearer key; saved to `$MODEL_CACHE_DIR/kimik3-api-key` |
| `PORT` | `30000` | Endpoint port on the node |
| `TP_SIZE` | `8` | Tensor-parallel degree |
| `DP_SIZE` | `1` | Data parallel. **vLLM: `DP*TP` = total GPUs.** SGLang: dp-attention groups, must divide `TP_SIZE` |
| `ENABLE_EP` | `0` | `1` adds vLLM `--enable-expert-parallel` / SGLang `--ep-size $TP_SIZE` |
| `CONTEXT_LEN` | `131072` | Max context (vLLM `--max-model-len` / SGLang `--context-length`) |
| `MEM_FRACTION` | `0.85` | vLLM `--gpu-memory-utilization` / SGLang `--mem-fraction-static` |
| `KV_CACHE_DTYPE` | `fp8` | Mapped per engine (`fp8` on vLLM, `fp8_e4m3` on SGLang); `auto` for the default |
| `TOOL_PARSER` | `kimi_k2` | Tool-call parser; `none` to omit. List options with `./serve-kimik3.sh parsers` |
| `REASONING_PARSER` | `kimi_k2` | Thinking parser; `none` to omit |
| `ENABLE_AITER` | `1` | AMD fused kernels: vLLM `VLLM_ROCM_USE_AITER*` env / SGLang `--enable-aiter-allreduce-fusion` |
| `SET_CPU_AFFINITY` | `0` | `SGLANG_SET_CPU_AFFINITY`; keep `0` on a SLURM cgroup (see troubleshooting) |
| `READY_TIMEOUT` | `10800` | Seconds to wait for health before giving up |
| `CHUNKED_PREFILL_SIZE` | — | vLLM `--max-num-batched-tokens` / SGLang `--chunked-prefill-size` |
| `MAX_RUNNING_REQUESTS` | — | vLLM `--max-num-seqs` / SGLang `--max-running-requests` |
| `SCHEDULE_POLICY` | — | SGLang `--schedule-policy` only (`lpm` for prefix reuse); ignored on vLLM |
| `CUDA_GRAPH_MAX_BS` | — | vLLM `--cuda-graph-sizes` / SGLang `--cuda-graph-max-bs` |
| `EXTRA_ENGINE_ARGS` | — | Extra flags appended verbatim to the engine's launch command |

**Mind the DP difference.** SGLang's `--tp` *is* the total GPU count and `--dp`
subdivides it; vLLM's `--data-parallel-size` *multiplies* it. On 8 GPUs that
means SGLang `TP_SIZE=8 DP_SIZE=2` and vLLM `TP_SIZE=4 DP_SIZE=2` describe
comparable layouts. `serve-kimik3.sh` validates this per engine and refuses a
layout that doesn't match your allocation.

---

## Performance tuning

Everything here is **measurement-driven** — change one thing, re-run
`./bench-kimik3.sh`, keep it only if the numbers improve. Run the benchmark from
a shell on the serving node while the server is up:

```bash
./bench-kimik3.sh sweep     # latency (c=1) + concurrency sweep -> $MODEL_CACHE_DIR/bench/
```

**How to read the output:** *Output token throughput* (tok/s — the headline for
aggregate throughput), *Mean/Median TTFT* (time-to-first-token — interactivity),
*Median ITL/TPOT* (inter-token latency — single-stream speed).

In order of expected payoff:

### 1. Get onto the tuned FP4 MoE kernel

`bench-kimik3.sh` warns if the server log mentions a *heuristic FlyDSL fallback*
— that means the MoE is JIT-falling-back to a slow generic kernel rather than
one tuned for these FP4 shapes. On GLM-5.2 this was worth roughly **+80%**
(~1461 → ~2626 tok/s/node), so it is the first thing to chase. The fix is
usually a newer engine image. Do it reversibly:

```bash
# in kimik3.env, point at a fresh sif so the working one is preserved:
export VLLM_IMAGE="docker://rocm/vllm-dev:<newer-tag>"
export SIF_PATH="$MODEL_CACHE_DIR/kimik3-vllm-new.sif"
./serve-kimik3.sh pull && ./serve-kimik3.sh stop && ./serve-kimik3.sh serve --detach
./bench-kimik3.sh sweep     # did the fallback warning go? did tok/s jump?
```

Revert those two variables if it regresses.

### 2. Parallelism layout — bench each, pick from numbers

Same 8 GPUs, different split. Restart between each:

| Config | `kimik3.env` | Best for |
|---|---|---|
| Pure TP8 | `TP_SIZE=8 DP_SIZE=1` | single-session latency (default) |
| Expert-parallel | `ENABLE_EP=1` | MoE aggregate throughput — **likely the big one for K3**, which is 896 experts with only 16 active |
| Data-parallel | vLLM `TP_SIZE=4 DP_SIZE=2`, SGLang `TP_SIZE=8 DP_SIZE=2` | high-concurrency throughput |

K3's sparsity is extreme compared to GLM-5.2 (16/896 active vs 8/256), so expert
parallelism deserves the first serious sweep rather than the last.

### 3. Serving knobs (cheap, safe to sweep)

- `MEM_FRACTION` — more KV cache. Raise in small steps and watch free memory in
  the log; with K3 the weights alone are ~65% of HBM.
- `SCHEDULE_POLICY=lpm` (SGLang) — prefix reuse; big for agentic/opencode where
  context repeats. vLLM does prefix caching by default.
- `MAX_RUNNING_REQUESTS`, `CHUNKED_PREFILL_SIZE`, `CUDA_GRAPH_MAX_BS` — raise for
  concurrency; lower the first two if you OOM.

### 4. Not worth it yet

MTP/EAGLE **speculative decoding** and **context-parallel** are not validated on
ROCm for this model family — skip them until AMD says otherwise.

---

## Running on MI300X/MI325X instead

You can't. K3 at MXFP4 is ~1.6 TB; a gfx942 node gives you 8× 192 GB (MI300X) =
1,536 GB or 8× 256 GB (MI325X) = 2,048 GB, and gfx942 has no native MXFP4
support in the first place. K3 on gfx942 needs multiple nodes and a different
quantization — out of scope for this repo.

The **stand-in** does run on gfx942 with an FP8 variant and an `mi30x` image
(`lmsysorg/sglang-rocm:*-mi30x-*`), if you want to exercise the plumbing on a
non-MI355X node. `serve-kimik3.sh` detects gfx942 at runtime and warns.

---

## Notes & troubleshooting

- **"apptainer not found"**: you're on a login node. Apptainer is only on compute
  nodes — start an allocation first.
- **Server exits immediately with an unknown-architecture error**: the image is
  older than the model. Run `./serve-kimik3.sh check` — it tells you exactly
  which architectures the image's registry knows.
- **Server exits on an argparse error about the parser name**: run
  `./serve-kimik3.sh parsers` and set `TOOL_PARSER`/`REASONING_PARSER` from the
  list, or set them to `none`.
- **Coherent-looking server, gibberish output**: this is a real, observed failure
  mode on this hardware — vLLM issue
  [#36337](https://github.com/vllm-project/vllm/issues/36337) reports
  `Kimi-K2.5-MXFP4` producing gibberish on gfx950 with ROCm 7.2. Always read an
  actual completion before trusting a run. If output is garbage, try a different
  image (both a newer *and* an older one), and try `ENABLE_AITER=0` to rule out
  the fused FP4 kernels.
- **Startup time**: with cached weights, several minutes for the stand-in and
  considerably longer for K3's ~1.6 TB. Watch
  `tail -f $MODEL_CACHE_DIR/kimik3-server.log`. The script health-polls and
  prints the banner only when `/health` returns 200.
- **Crash: "CPU number N is not eligible; choose between [...]"** (SGLang, in
  `set_gpu_proc_affinity`): the image sets `SGLANG_SET_CPU_AFFINITY=1`, but
  SGLang then pins workers to CPUs from the *full* node topology, which fail
  under a SLURM cgroup owning only a subset of cores. The script forces
  `SGLANG_SET_CPU_AFFINITY=0` by default. Set `SET_CPU_AFFINITY=1` only if you
  allocate the whole node's CPUs (`--cpus-per-task=384` / `--exclusive`).
- **Crash: `[Errno 30] Read-only file system: '.../flydsl_cache/...'`**: the FP4
  MoE JIT-compiles FlyDSL kernels and caches them *inside* the image, but an
  Apptainer `.sif` is read-only (unlike a Docker image's writable layer). The
  script locates aiter inside the image and binds `FLYDSL_CACHE_DIR` over its
  `jit/flydsl_cache`. If auto-detection fails it warns — set
  `FLYDSL_CACHE_TARGET` in `kimik3.env` to the path from the error message.
- **Crash during graph capture: `.aiter/jit/module_*.so: undefined symbol`**:
  aiter JIT-compiles some kernels at runtime and caches the `.so` files in
  `$HOME/.aiter/jit` (your home is bind-mounted into the container). A startup
  killed *mid-compile* leaves a truncated module that every later run reloads.
  Fix:
  ```bash
  ./serve-kimik3.sh stop
  rm -rf ~/.aiter        # (and ~/.triton if a triton kernel is the culprit)
  ./serve-kimik3.sh serve --detach
  ```
  It recompiles cleanly (adds ~a minute). This cache also eats your home
  file-quota — see `rquota`.
- **OOM allocating the KV cache (K3)**: expected if `MEM_FRACTION`/`CONTEXT_LEN`
  are too ambitious. Lower `CONTEXT_LEN` to 65536 first, then raise
  `MEM_FRACTION` — one at a time so you learn which mattered.
- **`salloc` rejected**: verify the recipe is still current (`sinfo`/`sacctmgr`
  above). `admin_test` is a restricted partition; if access was revoked, ask
  `rcc-support@uq.edu.au`.
- **Out of space during pull/download**: `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR`
  default to scratch near `MODEL_CACHE_DIR` — make sure that has room and isn't
  on `/home`. Check with `rquota`.
- **opencode doesn't see the model**: it only reads config at startup — restart
  it after `opencode-setup.sh`, and make sure `KIMIK3_API_KEY` is exported in
  that shell (or you used `--embed-key`).

---

## Sources

- [Kimi K3 tech blog](https://www.kimi.com/blog/kimi-k3) · [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — architecture (KDA, Attention Residuals, 16/896 experts), MXFP4 QAT, open-weights date
- [vLLM: A Preview of Production-Scale Kimi K3 Support](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) — FlashKDA/Triton prefill, fused decode kernel, MXFP4 paths, AMD FlyDSL A16W4/A8W4 ROCm support
- [amd/Kimi-K2.6-MXFP4](https://huggingface.co/amd/Kimi-K2.6-MXFP4) — the stand-in: 550B, OCP MXFP4 experts, MI350/MI355 + ROCm 7.2
- [ROCm Blogs: Serve Kimi-K2.5-MXFP4 on MI355X with ATOM](https://rocm.blogs.amd.com/software-tools-optimization/kimi-k25-mxfp4-atom/README.html) — gfx950 block-scaled FP4 kernels, AITER f4gemm, FlyDSL fused MoE
- [vLLM issue #36337](https://github.com/vllm-project/vllm/issues/36337) — Kimi MXFP4 gibberish on gfx950 / ROCm 7.2
- [UQ-RCC Bunya docs](https://github.com/UQ-RCC/hpc-docs) — Apptainer, SLURM, GPU partitions, filesystems
- [ROCm/aiter](https://github.com/ROCm/aiter) — MoE/GEMM kernels + tuned FP4 configs
- [GLM-5.2 sibling repo](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_GLM5.2) — this repo's parent; the Bunya-specific knowledge here came from it
- [opencode custom providers](https://opencode.ai/docs/providers/)

---

## License

The scripts in this repo are provided under the MIT License (see `LICENSE` —
fill in your name before publishing). The container images
(`rocm/vllm`, `lmsysorg/sglang-rocm`), the vLLM and SGLang engines, and the
model weights are covered by their own separate licences.
