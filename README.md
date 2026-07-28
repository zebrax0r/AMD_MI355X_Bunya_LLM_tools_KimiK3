# Kimi K3 on Bunya's AMD MI355X — SGLang serving for opencode

One-click serving of Moonshot's **Kimi K3** (2.8T-parameter hybrid MoE, Kimi
Delta Attention, 1M context, native vision) on a **single Bunya MI355X node** —
8× gfx950, 288 GB HBM each, 2,304 GB total — inside an Apptainer container,
exposed as an OpenAI-compatible endpoint and wired into
[opencode](https://opencode.ai).

Same shape as the [GLM-5.2 sibling repo](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_GLM5.2),
from which all the Bunya-specific knowledge here is inherited.

---

## What works today — verified 28 Jul 2026

K3 open weights landed on 27 Jul, and **SGLang shipped day-0 AMD support with a
validated 8× MI355X recipe**. This repo reproduces that recipe.

| | |
|---|---|
| Model | **`moonshotai/Kimi-K3`** — arch `KimiK3ForConditionalGeneration`, **1561 GB**, `mxfp4-pack-quantized`. The official release is already **MXFP4-native**, so there is no AMD requant to wait for. |
| Shape | 93 layers, 896 experts (16 active), 1,048,576 context, `hidden_size` 7168, vision tower. KDA on every layer except every 4th, which is full MLA (`kv_lora_rank` 512). |
| Engine | SGLang, image **`lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727`** |
| Parallelism | **TP8** — the whole node |
| Throughput | 820 / 2356 / 4898 tok/s at concurrency 2 / 8 / 32 (upstream, MI355 TP8) |
| Tool calling / thinking | `--tool-call-parser kimi_k3 --reasoning-parser kimi_k3` |
| Auth | Bearer API key, auto-generated into `$MODEL_CACHE_DIR/kimik3-api-key` |

**Why SGLang and not vLLM.** vLLM has K3 support in flight (PR #50000), but its
day-0 image is `vllm/vllm-openai:kimi-k3` — **NVIDIA only, no ROCm build** — and
that branch carries private dependencies. SGLang published an AMD image, a
validated MI355X recipe, and measured numbers. For this hardware it is not close.

> **On third-party "how to self-host K3" blog posts.** At least one widely-shared
> article recommends vLLM `v0.7.2` with a repo `moonshotai/Kimi-K3-MXFP4` and
> "~594 GB" of weights. All three are wrong: 0.7.2 is a 2025 release (current is
> 0.26.0), that repo does not exist (HTTP 401), and the real download is
> **1561 GB**. Prefer the upstream sources linked at the bottom of this file.

### Provenance

Everything above comes from primary sources, not inference:
[sglang#32541](https://github.com/sgl-project/sglang/pull/32541) (day-0 support +
image tags), [sglang#32548](https://github.com/sgl-project/sglang/issues/32548)
(the AMD MI355X recipe and perf tables), and `config.json` on
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3).

---

## The recipe

Upstream's validated MI355X launch (issue #32548) is what `serve-kimik3.sh`
runs:

```bash
SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 \
AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1 \
sglang serve \
  --model-path moonshotai/Kimi-K3 --trust-remote-code \
  --tp 8 --attention-backend triton --dtype bfloat16 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 \
  --host 127.0.0.1 --port 30000 --disable-radix-cache \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

Those four environment variables are not decoration — they select the tuned
gfx950 FP4 MoE path. Note also `--cuda-graph-max-bs-decode`: the old
`--cuda-graph-max-bs` **no longer exists** on this branch.

### Our deviations from upstream

Three, each deliberate:

1. **`--host 0.0.0.0`** instead of `127.0.0.1`, so SSH tunnels and off-node
   opencode work. Upstream can rely on loopback for isolation; we cannot.
2. **`--api-key` and `--served-model-name` added.** The first compensates for
   (1) — the key is now the only thing gating the port. The second gives
   opencode a stable model name.
3. **`SGLANG_SET_CPU_AFFINITY=0` forced.** The image enables CPU pinning, but
   SGLang then pins to CPUs from the *full* node topology, which fails under a
   SLURM cgroup owning a subset of cores. Upstream has no cgroup to trip over;
   we do. This one is ours.
4. **Radix cache left ON** (`DISABLE_RADIX_CACHE=0`); upstream disables it.
   Added 29 Jul 2026 from measurement, not preference — see
   [Performance tuning](#performance-tuning). Better throughput *and* per-token
   latency at every concurrency tested, and the benchmark understates it,
   because random prompts never exercise the prefix reuse opencode depends on.

### Memory budget

```
node HBM     8 × 288 GB = 2304 GB
K3 weights   1561 GB (measured)   -> ~195 GB/GPU at TP8   (68% of HBM)
mem-frac     0.85                 -> ~1958 GB budget
KV + graphs  ~397 GB              -> ~50 GB/GPU
```

Context is left **unset** by default, exactly as upstream does, so the server
uses the model's full 1,048,576. That is only viable because KDA is cheap: just
24 of 93 layers do full attention, so KV growth is a fraction of a dense 1M
model's. If KV allocation fails at startup, set `CONTEXT_LEN=262144` **before**
touching `MEM_FRACTION`, so you learn which knob mattered.

FP8 (~2.8 TB) and BF16 (~5.6 TB) would not fit on one node — MXFP4 is what makes
this a single-node job at all.

---

## Bunya specifics you need to know

- **Apptainer, not podman**, and it lives **only on compute nodes** — never the
  login nodes. So `pull`, `check`, `parsers`, `download` and `serve` all run
  inside a `salloc`/`sbatch` allocation. (`stop` and `status` work anywhere.)
- **The MI355X nodes are `bun159`, `bun160`, `bun161`**, in the **`admin_test`**
  partition (not `gpu_rocm`). Submit with
  `--partition=admin_test --account=a_rcc --qos=sdf --gres=gpu:mi355x:8` — the
  scheduler lands you on whichever is free; the serve banner prints the actual
  hostname to use for tunnels.
- **Run from `/scratch`** (`/scratch/user/$USER/...`), not `/home` (tight quota)
  and not `/QRISdata` (RDM — jobs can't be submitted from there). Check with
  `rquota`.
- **GPU passthrough is `--rocm`** — Apptainer binds `/dev/kfd`, `/dev/dri` and
  the ROCm libraries automatically. No `--device`/`--group-add`/`--shm-size`
  needed; the container shares host network and IPC.

Confirm the live SLURM recipe any time with:

```bash
sinfo -o "%P %.10l %G %f" | grep mi355x       # partition + gres label
sacctmgr -np show assoc user=$USER format=account,qos   # your account + QoS
scontrol show node bun161 | grep -iE 'Gres|Partitions|State'
```

---

## Prerequisites

- **A Bunya account** with the `a_rcc` scheduling account and access to the
  `admin_test` partition / `sdf` QoS (a special allocation — confirm with
  `rcc-support@uq.edu.au` if `salloc` is rejected).
- **~1.7 TB free scratch** on `/scratch/user/$USER`: 1561 GB of weights, ~30 GB
  for the `.sif`, plus the DSpark draft if you enable it. **Check this with
  `rquota` before starting a multi-hour download** — it is the longest-lead item
  here, and running out at 90% wastes the whole transfer.
- **A HuggingFace account + access token** (read scope), from
  <https://huggingface.co/settings/tokens>.
- **`opencode`** wherever you want to *use* the model — <https://opencode.ai>.
- Optional, for a nicer opencode config merge: **`jq`**.

---

## Repository contents

| File | Purpose |
|---|---|
| `serve-kimik3.sh` | The core script: pull, check, parsers, download, serve, stop, status |
| `serve-kimik3.sbatch` | SLURM batch wrapper (MI355X recipe) |
| `kimik3-env.example` | Config template — copy to `kimik3.env` and edit |
| `opencode-setup.sh` | Writes/merges the opencode provider config on any machine |
| `opencode.kimik3.json` | The provider template `opencode-setup.sh` fills in |
| `share-kimik3.sh` | Optional: public HTTPS tunnel via Cloudflare for users without SSH |
| `bench-kimik3.sh` | Benchmark tok/s / TTFT / ITL against upstream's numbers |

Secrets never live in the repo: `kimik3.env` (your HF token), the generated API
key and the `.sif` are gitignored / stored under `$MODEL_CACHE_DIR` on scratch.

---

## Walkthrough

### Step 0 — Get the code onto Bunya

```bash
cd /scratch/user/$USER
git clone <this-repo-url> kimik3-bunya
cd kimik3-bunya
```

### Step 1 — Configure

```bash
cp kimik3-env.example kimik3.env
$EDITOR kimik3.env
```

At minimum set `MODEL_CACHE_DIR` (absolute scratch path, ~1.7 TB free) and
`HF_TOKEN`. Everything else defaults to the validated MI355X recipe.

### Step 2 — Allocate an MI355X node

```bash
salloc --partition=admin_test --account=a_rcc --qos=sdf \
       --nodes=1 --gres=gpu:mi355x:8 --ntasks-per-node=1 \
       --cpus-per-task=192 --mem=1800G --time=12:00:00
```

### Step 3 — Build the image, then **check before you download**

```bash
./serve-kimik3.sh pull        # builds the .sif (one-time, multi-GB)
./serve-kimik3.sh check       # <- the cheap gate on a 1.5 TB commitment
./serve-kimik3.sh parsers     # confirm kimi_k3 is available
df -h $MODEL_CACHE_DIR        # confirm ~1.7 TB free
./serve-kimik3.sh download    # ~1561 GB
```

`check` reads `architectures` from the model's `config.json` and asks the
image's own SGLang registry whether it knows `KimiK3ForConditionalGeneration`.
It answers in seconds. **Do not skip it** — an image built from `main` rather
than the `kimi-k3` branch will fail *after* you've moved 1.5 TB.

### Step 4 — Serve

```bash
./serve-kimik3.sh serve --detach     # returns your shell once healthy
```

Or as a batch job:

```bash
mkdir -p logs && sbatch serve-kimik3.sbatch
grep -A 24 'is up and serving' logs/kimik3-<jobid>.out
```

A cold start reads ~1.5 TB off scratch and JIT-compiles FP4 kernels, so
`READY_TIMEOUT` defaults to 4 hours. Watch it with
`tail -f $MODEL_CACHE_DIR/kimik3-server.log`.

### Step 5 — Verify, and actually read the output

```bash
export KIMIK3_API_KEY="$(cat $MODEL_CACHE_DIR/kimik3-api-key)"
curl -s http://127.0.0.1:30000/v1/models -H "Authorization: Bearer $KIMIK3_API_KEY"

curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Authorization: Bearer $KIMIK3_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"In one sentence: what is Bunya?"}]}'
```

**Coherence is a pass/fail criterion, not a nicety.** vLLM issue
[#36337](https://github.com/vllm-project/vllm/issues/36337) documents Kimi MXFP4
weights producing fluent-looking *gibberish* on gfx950 with ROCm 7.2. A healthy
`/health` and a 200 response prove nothing about numerics. Read the sentence.

**Result on Bunya, 29 Jul 2026: PASS.** Asked why MXFP4 fits 2.8T parameters in
2304 GB, the model derived 2.8T x 2 bytes = 5.6 TB for BF16 and 4.25 effective
bits for MXFP4 (4-bit values + an 8-bit E8M0 scale per 32-element block) —
correct, and independently the same arithmetic as the memory budget above. The
#36337 failure mode does not appear on this SGLang build.

**K3 is a thinking model, so the text arrives in `reasoning_content`, not
`content`.** With the `kimi_k3` reasoning parser, `content` stays empty until
thinking finishes — a small `max_tokens` returns `finish_reason: length` and an
empty `content`, which looks like a broken server and is not. Give it room.

### Step 5b — Prove tool calling round-trips

opencode's whole agentic loop rests on this, and `kimi_k3` is a new parser on an
unmerged branch, so verify it rather than assume:

```bash
./serve-kimik3.sh toolcheck        # exit 0 = pass
```

**Result on bun161, 29 Jul 2026: PASS**, both turns — `tool_calls` well-formed
with parseable arguments, and the final answer carried the tool's value back
(*"The current GPU temperature on bun161 is **61.4°C**"*). The `kimi_k3` parser
round-trips correctly on this image.

One cosmetic note: tool-call ids come back as `get_gpu_temperature:0` rather than
OpenAI's `call_<random>` form. Unique per call and accepted on the way back in,
so it round-trips — but a client that assumes the `call_` prefix would be
surprised.

It runs a **two-turn** exchange, which is the point. Turn 1 checks the model
emits a well-formed `tool_calls` object with parseable JSON arguments and an id.
Turn 2 feeds a result back and checks the model *uses* it — the tool returns a
temperature the model cannot possibly guess, so a correct final answer proves it
consumed the result instead of inventing one. A parser that serialises tool
calls but breaks on the way back passes turn 1 and fails turn 2.

It is plain HTTP: no `.sif`, no GPU, no allocation. Point it anywhere, including
through a tunnel:

```bash
TOOLCHECK_URL=http://localhost:30000 ./serve-kimik3.sh toolcheck
```

What the failures mean:

| Symptom | Cause |
|---|---|
| `finish_reason=length`, no tool call | K3 is a thinking model and ran out of budget mid-thought. Not a parser fault. |
| Tool call appears as raw text in `content` | `TOOL_PARSER` is not matching this model. Check `./serve-kimik3.sh parsers`. |
| Turn 1 passes, turn 2 misses the value | The result is not reaching the model — the round trip is broken where opencode needs it most. |

### Step 6 — Connect opencode

On the GPU node (attach a second shell to the job first):

```bash
srun --overlap --jobid <jobid> --pty /bin/bash -l
cd /scratch/user/$USER/kimik3-bunya
./opencode-setup.sh --host localhost --port 30000
export KIMIK3_API_KEY="$(cat $MODEL_CACHE_DIR/kimik3-api-key)"
opencode
```

From your laptop, tunnel and point at localhost:

```bash
# terminal 1 (<node> = the hostname the serve banner printed):
ssh -N -L 30000:<node>:30000 $USER@bunya1.rcc.uq.edu.au
# terminal 2:
./opencode-setup.sh --host localhost --port 30000 --api-key <key>
```

Restart opencode, then pick **Kimi K3 (Bunya MI355X)** via `/models`.

### Step 7 — Shut down

```bash
./serve-kimik3.sh stop     # or Ctrl-C if attached, or scancel <jobid>
```

---

## Speculative decoding with DSpark

Upstream's numbers make this the single biggest interactive win available:

| | baseline | DSpark |
|---|---|---|
| c=2 median TPOT | 20.86 ms | **8.94 ms** |
| c=2 throughput | 820 tok/s | **1659 tok/s** |
| c=2 median TTFT | 1121 ms | **706 ms** |
| accept length | — | 5.29–5.93 |

Roughly half the per-token latency for a single opencode session. Enable it:

```bash
# kimik3.env
export SPECULATIVE="dspark"
./serve-kimik3.sh download     # fetches RadixArk/Kimi-K3-DSpark too
./serve-kimik3.sh stop && ./serve-kimik3.sh serve --detach
```

**Measured on bun161 (ROCm 7.14), 29 Jul 2026 — it works, and it is the single
biggest win available.** Same node, same sweep, DSpark the only variable:

| c | | total tok/s | median TPOT | accept len |
|---|---|---|---|---|
| 2 | baseline | 214 | 26.49 ms | — |
| 2 | **DSpark** | **673** | **5.93 ms** | **4.92** |
| 8 | baseline | 655 | 34.06 ms | — |
| 8 | **DSpark** | **1502** | **10.84 ms** | **5.26** |
| 32 | baseline | 1572 | 54.87 ms | — |
| 32 | **DSpark** | **2510** | **25.88 ms** | **5.41** |

3.1x / 2.3x / 1.6x on throughput, and accept length lands inside upstream's
5.29-5.93 — so the draft model is behaving. **Our c=2 TPOT of 5.93 ms beats
upstream's own DSpark figure of 8.94 ms.**

SGLang issue [#32569](https://github.com/sgl-project/sglang/issues/32569)
reports DSPARK crashing with `TypeError: 'NoneType' object is not callable`. We
have not hit it on this image. It stays off by default anyway, so that a first
launch has one thing to go wrong instead of two — turn it on once baseline
serving is proven.

**One caveat, unexplained.** A long-context opencode session (217k tokens) showed
`accept len: 1.23, accept rate: 0.03` — speculation collapsing to nothing, where
this 1k-token sweep gets ~5. Whether that is context length, workload content, or
a warmup artefact of the first decode batch after prefill is not yet established.
If DSpark seems to hurt in a long agentic session, measure before concluding: one
decode line is not a benchmark. That mistake was made here once already.

Note the throughput crossover: DSpark *loses* above ~concurrency 16 (3715 vs
4898 tok/s at c=32). It is a latency optimisation for interactive use, not a
throughput one.

---

## Performance tuning

Measurement-driven only — change one thing, re-run `./bench-kimik3.sh`, keep it
if the numbers improve. Run from a shell on the serving node:

```bash
./bench-kimik3.sh sweep     # c=2/8/32 -> $MODEL_CACHE_DIR/bench/
```

### Measured on bun161, 29 Jul 2026 (DSpark, 1024/512, TP8)

| c | total tok/s | output tok/s | median TPOT | median TTFT | accept len |
|---|---|---|---|---|---|
| 2 | 918 | 306 | **5.61 ms** | 253 ms | 7.29 |
| 8 | 2074 | 691 | 10.13 ms | 299 ms | 7.26 |
| 32 | 3793 | 1264 | 21.85 ms | 663 ms | 7.25 |

**Do not chase upstream's `820 / 2356 / 4898`.** Those figures do not survive
arithmetic against a 1024/512 workload: their c=2 row pairs 820 tok/s with a
20.86 ms TPOT and 1121 ms TTFT, which at concurrency 2 implies an 11.8 s E2E,
0.17 req/s and ~260 tok/s — not 820. Their *ratios* are internally consistent
(20.86/8.94 = 2.33 vs 1659/820 = 2.02), so the measurements are real; they are
simply not this workload, and the config was never published.

Compare on metrics that are immune to request sizing instead:

| | upstream (DSpark) | here |
|---|---|---|
| median TPOT @ c=2 | 8.94 ms | **5.61 ms** |
| accept length | 5.29-5.93 | **7.25-7.29** |
| total tok/s @ c=32 | 3715 | **3793** |

Use the table above as the local baseline for tuning: change one thing, re-run,
keep it if the numbers improve.

### Radix cache — measured, and now the default

Upstream passes `--disable-radix-cache`. We do not. Same node, same sweep,
DSpark on, prefix caching the only variable:

| c | total tok/s | median TPOT | median TTFT |
|---|---|---|---|
| 2 | 918 -> **964** | 5.61 -> **5.35 ms** | 253 -> 255 ms |
| 8 | 2074 -> **2299** | 10.13 -> **9.10 ms** | 299 -> **288 ms** |
| 32 | 3793 -> **4608** | 21.85 -> **14.04 ms** | 663 -> 1877 ms |

**The benchmark understates this.** `--dataset-name random` generates prompts
that share no prefixes, so what is being measured is better KV reuse and
scheduling — *not* prefix caching. The gain that matters for opencode, reusing a
200k-token context across turns, is not exercised at all. Treat +5% at c=2 as a
floor.

The one cost is TTFT at c=32: median 663 -> 1877 ms, P99 2383 -> 7998 ms. The
scheduler admits requests later and batches harder, so end-to-end latency still
*improves* (11.8 -> 9.3 s) and the sweep finishes faster (81 -> 67 s). Set
`DISABLE_RADIX_CACHE=1` if you are serving several interactive users at once and
first-token latency matters more than total throughput. For a single opencode
session at c=1-2 there is no TTFT penalty at all.

**Read the token totals before believing a throughput number.** sglang's random
dataset samples each length uniformly from `[len * ratio, len]`, and its default
ratio is `0.0` — so a nominal 1024/512 sweep really sends about 507 in / 262 out
per request (measured, 29 Jul 2026). Aggregate tok/s at fixed concurrency scales
with tokens per request, so that halves the headline figure while leaving TPOT
untouched, and makes the result incomparable to any published number. This repo
defaults `BENCH_RANGE_RATIO=1.0` so sizes are exactly what you configured. Check
`Total input tokens` / `Total generated tokens` in the output against
`num-prompts x len` whenever a result surprises you.

**Prefer median TPOT for cross-system comparison.** It is per-token and immune to
request sizing, so it survives differences in benchmark configuration that
aggregate throughput does not.

In order of expected payoff:

1. **Confirm the tuned FP4 MoE path is active.** `bench-kimik3.sh` warns if the
   server log mentions a heuristic FlyDSL fallback. If it does, check
   `ENABLE_AITER=1` — the four `SGLANG_AITER_K3_OPT` / `AITER_FLYDSL_FORCE` /
   `AITER_SITUV2_A8W4` variables are what select the tuned kernels.
2. **DSpark** for interactive latency (above).
3. **Prefix caching.** Upstream passes `--disable-radix-cache`, which this repo
   follows. But opencode resends large similar contexts every turn, which is
   exactly what prefix reuse rewards — so set `DISABLE_RADIX_CACHE=0` and bench
   it, optionally with `SCHEDULE_POLICY=lpm`. Upstream presumably disabled it
   deliberately (possibly a KDA recurrent-state interaction), so treat a win
   here as something to verify for correctness, not just speed.
4. **`CUDA_GRAPH_MAX_BS_DECODE`** — raise for more decode concurrency, costs
   memory. **`MAX_RUNNING_REQUESTS` / `CHUNKED_PREFILL_SIZE`** — lower if you
   OOM in decode / prefill respectively.
5. **Expert parallelism.** Not in upstream's recipe, but K3 is extremely sparse
   (16 of 896 active). Worth a try via
   `EXTRA_ENGINE_ARGS="--ep-size 8"` — bench before believing it.

---

## When the branch merges

The default image is built from SGLang's **unmerged `kimi-k3` branch**
(PR #32541) and pinned to a dated tag. That is fine for now but is not a
long-term footing: pre-merge tags can be rebuilt or removed.

Once #32541 merges to `main`, switch to a mainline image and re-verify:

```bash
# check what's current:
curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags?page_size=100&ordering=last_updated" \
  | python3 -c "import json,sys;[print(t['name'],t['last_updated'][:10]) for t in json.load(sys.stdin)['results'] if 'mi35x' in t['name']]"

# then, in kimik3.env, point at a fresh sif so the working one is preserved:
export SGLANG_IMAGE="docker://lmsysorg/sglang-rocm:<newer-mi35x-tag>"
export SIF_PATH="$MODEL_CACHE_DIR/kimik3-mi355x-new.sif"
./serve-kimik3.sh pull && ./serve-kimik3.sh check     # must still know KimiK3ForConditionalGeneration
```

Revert those two variables if anything regresses. Same procedure for trying any
newer image.

---

## When the node's ROCm changes

The image ships its own ROCm (7.2 in the pinned tag) and the node has its own.
They do not have to match — but on **28 Jul 2026** RCC moved a test node to
**ROCm 7.14** and the unchanged toolkit died on import:

```
RuntimeError: Get GPU arch from rocminfo failed:
  Command '['/opt/rocm-7.2.0/bin/rocminfo']' returned non-zero exit status 1.
```

Note the path: that is the **container's own** 7.2 `rocminfo`, not the node's.
The image did not change, so the node reached in through one of exactly three
channels:

1. **`--rocm` library injection.** Apptainer binds the *node's* ROCm libraries
   into `/.singularity.d/libs` and prepends that to `LD_LIBRARY_PATH`, so the
   container's binaries run against them. Apptainer's own docs require the two
   ROCm versions to be compatible. Fixable: `ROCM_MODE=devices`.
2. **The kernel driver, via `/dev/kfd`.** A KFD ioctl ABI break is *not*
   fixable by any bind — it needs an image built for the node's ROCm.
3. **Inherited `*_VISIBLE_DEVICES`.** A UUID-form list the container's older
   ROCr cannot parse stops it enumerating any GPU at all, which looks exactly
   like a dead driver. The script now refuses to forward non-index-form values.

…and a fourth possibility, which is what bun161 turned out to be:

4. **Nothing is wrong with the passthrough at all.** `torch.cuda.device_count()`
   returned **8** in *both* modes — the GPUs were reachable the whole time. Only
   the `rocminfo` *binary* was failing. PyTorch's ROCm wheel carries its own HIP
   runtime, so HIP kept working while the image's standalone ROCm 7.2 tools did
   not. aiter shells out to `rocminfo` purely to name the architecture, and
   reports only its exit status — which makes a cosmetic failure look fatal.

   What the image's `rocminfo` actually prints on such a node:

   ```
   ROCk module version 6.19.14.31400000 is loaded
   hsa api call failure at: .../rocminfo.cc:357
   Call returned HSA_STATUS_ERROR_INVALID_ARGUMENT
   ```

   It loads, reads the driver version, then fails an HSA call against the newer
   KFD. **Fix: `ROCMINFO_SHIM` (default `auto`, so this is automatic.)** The
   script snapshots the *host's* working `rocminfo` output and binds a one-line
   script replaying it over the container's binary. That is real output for
   this node, so whatever aiter's parser expects it gets, and nothing links
   against it — no host libraries are involved.

   `AITER_GPU_ARCHS=gfx950` looks like the obvious fix and **does not work**:
   the `rocm720-mi35x-k3-20260727` build of aiter ignores `GPU_ARCHS` at
   runtime (tested on bun161, 28 Jul 2026).

`gpucheck` tells you which one you have, in about a minute:

```bash
./serve-kimik3.sh gpucheck
```

It prints both ROCm versions, dumps what the image's `rocminfo` actually says,
tries every passthrough mode, and gives a verdict — and it distinguishes case 4
from case 2, because "torch saw 8 GPUs but aiter could not name the arch" and
"the container cannot reach the GPUs" need completely different responses. Set
`AITER_GPU_ARCHS` and re-run it to confirm the workaround before committing to
a long startup.

`ROCM_MODE=auto` (the default) does the same probing automatically at launch and
caches the answer per node+image, so a working node pays for it once.

If **torch sees zero devices in every mode**, you are in case 2 and need a
different image. In order of cost:

```bash
# 1. Has K3 support landed in a mainline image on a newer ROCm?
export SGLANG_IMAGE="docker://rocm/sgl-dev:v0.5.13.post1-ubuntu24.04-py3.14-rocm7.14"
export SIF_PATH="$MODEL_CACHE_DIR/kimik3-rocm714.sif"
./serve-kimik3.sh pull && ./serve-kimik3.sh check   # needs KimiK3ForConditionalGeneration

# 2. Has upstream rebuilt the K3 image on a newer ROCm? (see the tag query above,
#    and also check rocm/sgl-dev)
```

Only if both fail is building one worth it: `sglang/docker/rocm.Dockerfile`
takes `SGL_BRANCH`, `GPU_ARCH=gfx950` and a `BASE_IMAGE_950_*` override, and
prebuilds aiter's kernels — 45–90 minutes. The Dockerfile is the easy part;
**Bunya has no Docker**, so it needs an Apptainer def file with fakeroot (check
with RCC first) or a machine that does. Serving on the ROCm 7.2 nodes while
waiting for an upstream 7.14 K3 image is a legitimate answer, not a failure.

---

## Running on other hardware

**MI300X / MI325X (gfx942): no.** K3 needs ~1.5 TB, and a gfx942 node gives
1,536 GB (MI300X) or 2,048 GB (MI325X) — with no native MXFP4 support in the
first place. K3 there would need multiple nodes and a different quantization,
which is out of scope for this repo. `serve-kimik3.sh` detects gfx942 and warns.

---

## Script reference

```
./serve-kimik3.sh [serve]        start serving (default), stays attached
./serve-kimik3.sh serve --detach start serving, wait until healthy, return the shell
./serve-kimik3.sh pull           build the .sif from the container image (one-time)
./serve-kimik3.sh check          can this image load this model? (run BEFORE download)
./serve-kimik3.sh gpucheck       can this image reach this node's GPUs? (~1 min)
./serve-kimik3.sh toolcheck      two-turn tool-call round trip vs a running server
./serve-kimik3.sh parsers        list tool-call/reasoning parsers this image supports
./serve-kimik3.sh download       prefetch weights (+ DSpark draft if enabled)
./serve-kimik3.sh stop           stop the server
./serve-kimik3.sh status         server state + health check + /v1/models

./opencode-setup.sh [--host H] [--port P] [--model M] [--context N]
                    [--api-key K] [--embed-key] [--config PATH]

./share-kimik3.sh [share]        open a public HTTPS Cloudflare tunnel (add --detach)
./share-kimik3.sh stop | status

./bench-kimik3.sh [sweep]        tok/s at concurrency 2/8/32
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
`export` beforehand takes precedence.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CACHE_DIR` | *(required)* | Scratch path for HF cache + `.sif` + API key + log (~1.7 TB) |
| `HF_TOKEN` / `HF_TOKEN_FILE` | *(required first run)* | HuggingFace token, inline or from a file |
| `MODEL_ID` | `moonshotai/Kimi-K3` | Model repo (already MXFP4-native) |
| `SERVED_MODEL_NAME` | `kimi-k3` | Name clients use in the `model` field |
| `SGLANG_IMAGE` | `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` | Day-0 AMD image (branch build — see above) |
| `SIF_PATH` | `$MODEL_CACHE_DIR/kimik3-mi355x.sif` | Where the Apptainer image is stored |
| `APPTAINER_CACHEDIR` / `_TMPDIR` | *(near `$MODEL_CACHE_DIR`)* | Apptainer cache/scratch, kept off `/home` |
| `AITER_JIT_DIR` | `$MODEL_CACHE_DIR/aiter-jit` | Writable copy of aiter's `jit/`, bound over the in-image one |
| `AITER_JIT_TARGET` | *(auto-detected)* | In-image `jit/` path to bind over; set only if detection is wrong |
| `KIMIK3_API_KEY` | *(auto-generated)* | Bearer key; saved to `$MODEL_CACHE_DIR/kimik3-api-key` |
| `PORT` | `30000` | Endpoint port on the node |
| `TP_SIZE` | `8` | Tensor parallel = **total GPU count** |
| `DP_SIZE` | `1` | dp-attention groups; must divide `TP_SIZE` |
| `CONTEXT_LEN` | *(empty)* | Empty = model max (1M), as upstream. Set `262144` if KV won't allocate |
| `MEM_FRACTION` | `0.85` | `--mem-fraction-static` (upstream's validated value) |
| `ATTENTION_BACKEND` | `triton` | `--attention-backend` |
| `MODEL_DTYPE` | `bfloat16` | `--dtype` (compute dtype; weights are MXFP4) |
| `CUDA_GRAPH_MAX_BS_DECODE` | `256` | `--cuda-graph-max-bs-decode` (note: not `--cuda-graph-max-bs`) |
| `DISABLE_RADIX_CACHE` | `0` | Prefix caching ON — our deviation, measured. `1` restores upstream's setting |
| `KV_CACHE_DTYPE` | *(empty)* | Upstream passes none |
| `TOOL_PARSER` | `kimi_k3` | Tool-call parser; `none` to omit |
| `REASONING_PARSER` | `kimi_k3` | Thinking parser; `none` to omit |
| `SPECULATIVE` | *(empty)* | `dspark` enables DSpark speculative decoding |
| `DSPARK_MODEL` | `RadixArk/Kimi-K3-DSpark` | Draft model for DSpark |
| `ENABLE_AITER` | `1` | Exports the four `SGLANG_*`/`AITER_*` K3 variables |
| `ROCM_MODE` | `auto` | GPU passthrough: `auto` probes, `rocm` uses `--rocm`, `devices` binds `/dev/kfd` only |
| `AITER_GPU_ARCHS` | *(empty)* | Sets `GPU_ARCHS`. Ignored at runtime by the pinned aiter build — prefer `ROCMINFO_SHIM` |
| `ROCMINFO_SHIM` | `auto` | Replay the host's `rocminfo` output inside the container when the image's own fails |
| `SET_CPU_AFFINITY` | `0` | Keep `0` under a SLURM cgroup (see troubleshooting) |
| `READY_TIMEOUT` | `14400` | Seconds to wait for health (cold load is ~1.5 TB) |
| `LAUNCH_CMD` | `sglang serve` | Escape hatch: `python3 -m sglang.launch_server` |
| `CHUNKED_PREFILL_SIZE` / `MAX_RUNNING_REQUESTS` / `SCHEDULE_POLICY` | — | Optional tuning |
| `BENCH_RANGE_RATIO` | `1.0` | Fixed-size bench requests. sglang's own default `0.0` halves them (see Performance tuning) |
| `EXTRA_ENGINE_ARGS` | — | Flags appended verbatim (e.g. `--ep-size 8`) |

---

## Notes & troubleshooting

- **"apptainer not found"**: you're on a login node. Start an allocation first.
- **Server exits with an unknown-architecture error**: the image predates K3
  support. Run `./serve-kimik3.sh check` — it names exactly which architectures
  the image knows. See [When the branch merges](#when-the-branch-merges).
- **Server exits on an argparse error about a parser name**: run
  `./serve-kimik3.sh parsers`. K3 uses `kimi_k3`, **not** `kimi_k2`.
- **Unknown-flag error on `--cuda-graph-max-bs`**: that flag was removed; the
  branch uses `--cuda-graph-max-bs-decode`. Nothing in this repo should emit the
  old name — if you added it via `EXTRA_ENGINE_ARGS`, drop it.
- **Fluent gibberish**: see Step 5. Try `ENABLE_AITER=0` to rule out the fused
  FP4 kernels, and try a different image tag. Do not ship a config that produces
  incoherent output because the throughput looked good.
- **`RuntimeError: Get GPU arch from rocminfo failed`** at startup — *hit for
  real on 28 Jul 2026 when a node moved to ROCm 7.14.* The container cannot
  reach the GPUs. Run `./serve-kimik3.sh gpucheck`; see
  [When the node's ROCm changes](#when-the-nodes-rocm-changes) for the three
  causes and which are fixable. Short version: if `gpucheck` shows torch seeing
  the GPUs, only `rocminfo` is broken and `ROCMINFO_SHIM=auto` (the default)
  handles it. `AITER_GPU_ARCHS=gfx950` looks right but is ignored at runtime.
- **KV cache OOM at startup**: set `CONTEXT_LEN=262144` first, and only then
  consider `MEM_FRACTION`. One knob at a time.
- **`TypeError: 'NoneType' object is not callable`** with DSpark: known upstream
  bug ([#32569](https://github.com/sgl-project/sglang/issues/32569)). Unset
  `SPECULATIVE`.
- **Crash: "CPU number N is not eligible; choose between [...]"** in
  `set_gpu_proc_affinity`: the image sets `SGLANG_SET_CPU_AFFINITY=1`, but
  SGLang pins to CPUs from the *full* node topology, which fails under a SLURM
  cgroup owning a subset of cores. The script forces `0` by default. Set
  `SET_CPU_AFFINITY=1` only with `--cpus-per-task=384` / `--exclusive`.
- **Crash at graph capture: `[Errno 30] Read-only file system:
  '.../aiter/jit/flydsl_cache/...'`** — *hit for real on 28 Jul 2026.* The FP4
  MoE JIT-compiles FlyDSL kernels at CUDA-graph capture and writes them into
  aiter's `jit/` directory *inside* the image, but a `.sif` is read-only. Note
  where this lands: capture happens **after** the ~1.5 TB weight load, so the
  failure costs a full load before it appears.

  The script seeds `AITER_JIT_DIR` from the image's `jit/` (once — this keeps
  the prebuilt `module_*.so`, which a bare empty bind would hide) and binds it
  back over the same path, then write-tests the bind up front so a broken one
  fails in seconds instead of after the load. If it cannot find aiter it warns
  loudly — set `AITER_JIT_TARGET` to the `jit/` directory from the error path
  (e.g. `/sgl-workspace/aiter/aiter/jit`).

  The seed marker records which image it came from, so changing `SGLANG_IMAGE`
  now re-seeds automatically — a copy left over from an older image causes
  undefined-symbol crashes, and moving between ROCm 7.2 and 7.14 images makes
  that a certainty rather than a risk.
- **Crash during graph capture: `.aiter/jit/module_*.so: undefined symbol`**:
  aiter JIT-compiles kernels into `$HOME/.aiter/jit` (your home is bind-mounted
  into the container). A startup killed *mid-compile* leaves a truncated module
  that every later run reloads. Fix:
  ```bash
  ./serve-kimik3.sh stop
  rm -rf ~/.aiter        # (and ~/.triton if a triton kernel is the culprit)
  ./serve-kimik3.sh serve --detach
  ```
  It recompiles cleanly (adds ~a minute). This cache also eats your home
  file-quota — see `rquota`.
- **Out of space during download**: `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR`
  default to scratch near `MODEL_CACHE_DIR` — make sure that has room and isn't
  on `/home`.
- **`salloc` rejected**: verify the recipe with the `sinfo`/`sacctmgr` commands
  above. `admin_test` is restricted; ask `rcc-support@uq.edu.au`.
- **opencode doesn't see the model**: it only reads config at startup — restart
  it after `opencode-setup.sh`, and make sure `KIMIK3_API_KEY` is exported in
  that shell (or you used `--embed-key`).

---

## Sharing with someone who has no SSH access (optional)

`share-kimik3.sh` exposes the running endpoint over public HTTPS via a
**Cloudflare quick tunnel** — outbound-only, no root, no Cloudflare account:

```bash
./share-kimik3.sh share --detach     # prints https://<random>.trycloudflare.com
```

It downloads `cloudflared` into `$MODEL_CACHE_DIR/cloudflared/`, checks the
server is healthy, opens the tunnel, and prints a ready-to-paste opencode
provider block. Manage with `./share-kimik3.sh status` / `stop`.

> ⚠️ Needs the compute node to have **outbound internet**. If that's blocked,
> use the SSH tunnel in Step 6 instead.

> ⚠️ **The public URL + API key together grant full use of your model and your
> Bunya GPU-hours (billed to `a_rcc`).** Share the key over a private channel
> only, rotate it if it leaks (delete `$MODEL_CACHE_DIR/kimik3-api-key` and
> restart), and **check RCC's acceptable-use policy before exposing HPC compute
> externally** — the API key is the only gate.

---

## Sources

- [sglang#32541 — day-0 Kimi K3 support](https://github.com/sgl-project/sglang/pull/32541) — branch, NVIDIA and AMD image tags
- [sglang#32548 — [Kimi-K3][AMD] Day 0 and Performance Tracking](https://github.com/sgl-project/sglang/issues/32548) — **the MI355X recipe and every perf number in this README**
- [sglang#32569](https://github.com/sgl-project/sglang/issues/32569) — the open DSPARK crash
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) · [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark)
- [LMSYS: Kimi K3 day-0 support](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support) · [Kimi K3 tech blog](https://www.kimi.com/blog/kimi-k3) — KDA, Attention Residuals, 16/896 sparsity
- [vLLM K3 preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) · [vllm#50000](https://github.com/vllm-project/vllm/pull/50000) — the NVIDIA-only alternative, for when a ROCm build appears
- [vllm#36337](https://github.com/vllm-project/vllm/issues/36337) — Kimi MXFP4 gibberish on gfx950/ROCm 7.2
- [UQ-RCC Bunya docs](https://github.com/UQ-RCC/hpc-docs) · [ROCm/aiter](https://github.com/ROCm/aiter) · [opencode providers](https://opencode.ai/docs/providers/)
- [GLM-5.2 sibling repo](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_GLM5.2) — where the Bunya-specific knowledge here came from

---

## License

The scripts in this repo are provided under the MIT License (see `LICENSE`).
The container image
(`lmsysorg/sglang-rocm`), the SGLang engine, and the model weights
(`moonshotai/Kimi-K3`) are covered by their own separate licences.
