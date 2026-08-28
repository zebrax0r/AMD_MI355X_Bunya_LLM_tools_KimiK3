# Kimi K3 on Bunya's AMD MI355X — SGLang serving for opencode and kimicode

One-click serving of Moonshot's **Kimi K3** (2.8T-parameter hybrid MoE, Kimi
Delta Attention, 1M context, native vision) on a **single Bunya MI355X node** —
8× gfx950, 288 GB HBM each, 2,304 GB total — inside an Apptainer container,
exposed as an OpenAI-compatible endpoint and wired into
[opencode](https://opencode.ai) or
[kimicode](https://www.kimi.com/code/docs/en/).

Same shape as the [GLM-5.2 sibling repo](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_GLM5.2),
from which all the Bunya-specific knowledge here is inherited.

---

## What works today — verified 28 Jul, measured through 12 Aug, image current 20 Aug 2026

K3 open weights landed on 27 Jul, and **SGLang shipped day-0 AMD support with a
validated 8× MI355X recipe**. This repo reproduces that recipe.

| | |
|---|---|
| Model | **`moonshotai/Kimi-K3`** — arch `KimiK3ForConditionalGeneration`, **1561 GB**, `mxfp4-pack-quantized`. The official release is already **MXFP4-native**, so there is no AMD requant to wait for. |
| Shape | 93 layers, 896 experts (16 active), 1,048,576 context, `hidden_size` 7168, vision tower. KDA on every layer except every 4th, which is full MLA (`kv_lora_rank` 512). |
| Engine | SGLang, image **`lmsysorg/sglang-rocm:v0.5.17-rocm724-mi35x-20260820`** — mainline since 20 Aug; the day-0 `rocm720-mi35x-k3-20260727` tag every measurement below was taken on is kept as the [anchor](#moving-to-a-mainline-image) |
| Parallelism | **TP8** — the whole node |
| Throughput | **918 / 2074 / 3793 tok/s** at concurrency 2 / 8 / 32, measured here at 1024/512 with DSpark. Upstream's `820 / 2356 / 4898` is a different, unpublished workload — [why](#measured-on-bun161-29-jul-2026-dspark-1024512-tp8) |
| Tool calling / thinking | `--tool-call-parser kimi_k3 --reasoning-parser kimi_k3` |
| Auth | Bearer API key, auto-generated into `$MODEL_CACHE_DIR/kimik3-api-key` |
| Clients | **opencode and kimicode both verified end to end** — tool calls, edits, long sessions |

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
| `kimicode-setup.sh` | Writes/merges the kimicode provider config on any machine |
| `kimicode.kimik3.toml` | The provider template `kimicode-setup.sh` fills in |
| `bench-kimik3.sh` | Benchmark tok/s / TTFT / ITL at 1024/512, or `longcontext` at 100k |

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
It answers in seconds. **Do not skip it** — a mainline image built before K3
merged on 4 Aug 2026 will fail *after* you've moved 1.5 TB.

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

opencode's whole agentic loop rests on this, and `kimi_k3` is a new parser that
was still being revised after our pin, so verify it rather than assume:

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

### Step 6b — Connect kimicode (optional)

Moonshot's own CLI agent talks to any OpenAI-compatible endpoint, so it will
drive this server too — Kimi's coding agent on a self-hosted Kimi model. Nothing
changes on the server side: `--host 0.0.0.0`, `--api-key` and the `kimi_k3`
parsers already do everything it needs.

**Verified on Bunya, 12 Aug 2026** — end-to-end agentic loop, against a live
server on bun161. Written 1 Aug from Moonshot's published config schema and
confirmed by use since.

**First, get the right `kimi`.** Moonshot ships two products whose binary is both
called `kimi`, and following a stale blog post lands you in the wrong config file
with a provider type that does not exist:

| | Legacy | **What this targets** |
|---|---|---|
| Name | Kimi CLI | Kimi Code CLI |
| Runtime | Python / uv / PyPI `kimi-cli` | Node.js |
| Config | `~/.kimi/config.toml` | `~/.kimi-code/config.toml` |
| Provider type | `openai_legacy` | `openai` |
| Status | "will no longer be maintained" | current |

```bash
curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
```

On the GPU node (attach a second shell to the job first):

```bash
srun --overlap --jobid <jobid> --pty /bin/bash -l
cd /scratch/user/$USER/kimik3-bunya
./kimicode-setup.sh --host localhost --port 30000
kimi -m kimik3-bunya/kimi-k3
```

From your laptop, tunnel and point at localhost:

```bash
# terminal 1 (<node> = the hostname the serve banner printed):
ssh -N -L 30000:<node>:30000 $USER@bunya1.rcc.uq.edu.au
# terminal 2:
./kimicode-setup.sh --host localhost --port 30000 --api-key <key>
kimi -m kimik3-bunya/kimi-k3
```

The script writes a sentinel-delimited block into `~/.kimi-code/config.toml`,
leaving the rest of the file — including `default_model` and any managed
Kimi provider — untouched. Re-running it replaces that block rather than
appending a second one, so it is safe to run after every reallocation when the
node changes. A backup lands at `config.toml.bak`.

It deliberately does **not** set `default_model`: that is a top-level TOML key, and
appending one after the tables would silently bind it to the wrong table. Select
the model with `-m` as above, or `/model` inside the TUI.

**One real difference from opencode.** Kimi Code's TOML has no `{env:VAR}`
interpolation, so unlike `opencode-setup.sh` the key is written *literally* into
the config (created mode 600). If you would rather keep no key on disk, use
`--no-key` and the environment route instead, which needs no config file at all:

```bash
export KIMI_MODEL_NAME=kimi-k3
export KIMI_MODEL_PROVIDER_TYPE=openai
export KIMI_MODEL_BASE_URL=http://localhost:30000/v1
export KIMI_MODEL_API_KEY="$(cat $MODEL_CACHE_DIR/kimik3-api-key)"
export KIMI_MODEL_MAX_CONTEXT_SIZE=1048576
kimi
```

If you are setting it up fresh, or after an image change, verify in this order —
cheapest first:

```bash
./serve-kimik3.sh toolcheck                       # 1. server-side tool loop
curl -s http://localhost:30000/v1/models \
  -H "Authorization: Bearer $KIMIK3_API_KEY"      # 2. reachable from this machine
grep -A12 'kimik3-bunya' ~/.kimi-code/config.toml # 3. the block landed
kimi -m kimik3-bunya/kimi-k3                      # 4. starts, model listed in /model
```

Then **5. ask it to read a file and make a one-line edit.** That is the only step
that exercises tool calls end to end, and the one most likely to fail — see the
tool-call id note in [Step 5b](#step-5b--prove-tool-calling-round-trips). If step 1
fails, kimicode cannot work; fix that before touching the client.

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

### The sampling landmine — read this before enabling DSpark

SGLang issue [#32569](https://github.com/sgl-project/sglang/issues/32569)
reports DSPARK crashing with `TypeError: 'NoneType' object is not callable`, on
this exact image tag with these exact env vars. **The bug is live in our build.
Every number above was measured over it** — we have never sent the request that
fires it, which is not the same as being safe.

DSpark's verify step calls `top_k_renorm_prob` / `top_p_renorm_prob`. On ROCm
those are bound in `dflash_utils.py` by an `is_hip()` branch that landed in two
halves, both *after* our pin:

| Image | `top_k` | `top_p` | Symptom |
|---|---|---|---|
| **`…k3-20260727` — ours**, pushed 28 Jul 05:48 UTC | `None` | `None` | `TypeError: 'NoneType' object is not callable` |
| `…k3-20260728`–`0730` — [#32621](https://github.com/sgl-project/sglang/pull/32621), 28 Jul 07:37 UTC | unbound | Triton | `NameError: top_k_renorm_prob` |
| `…k3-20260731`+ — [#32641](https://github.com/sgl-project/sglang/pull/32641), 31 Jul 06:21 UTC | Triton | Triton | fixed |

Our image was built about two hours before the first of those merged.

**Why it has never fired.** `dspark_accept.py::_accept_sampling_core` skips the
whole renorm path when `not need_top_k_sampling and not need_top_p_sampling`,
and those come from `any(top_p != 1.0)` / `any(top_k != TOP_K_ALL)` across the
batch. Nothing we send sets either: `moonshotai/Kimi-K3`'s `generation_config.json`
carries only `max_length` and `eos_token_id`, so SGLang's own defaults (`top_p=1.0`,
`top_k=-1`) apply; `bench-kimik3.sh` sets neither; and `toolcheck`'s
`temperature: 0` becomes `top_k=1`, which takes the *sparse* branch and calls
nothing.

**What fires it.** One request with `top_p < 1.0` — a single curl, or one client
that sets it. Two things make it worse than a bad request:

- `need_top_*` is `any()` **over the batch**, so one such request poisons every
  request running beside it.
- the traceback bottoms out in `run_scheduler_process` → `event_loop_overlap`,
  so it kills the **server**, not the request. #32569's reporter saw it "work
  for about 5 minutes, then crash".

#### The second one, and it fires on `temperature > 0`

Found in the 12 Aug upstream review:
[#33694](https://github.com/sgl-project/sglang/pull/33694), merged 6 Aug. #32541's
`elif is_hip():` branch set `_DFLASH_SAMPLING_VERIFY_AVAILABLE = True` but never
bound `tree_speculative_sampling_target_only` — and that kernel **does not exist
on ROCm at all**: `speculative_sampling.cu` is listed only in the CUDA
`CMakeLists.txt`, and `common_extension_rocm.cc` registers just
`verify_tree_greedy`. With the flag unconditionally true, any request with
`temperature > 0` took the non-greedy verify path and died on the unbound name,
in the scheduler. Upstream's own CI shows the shape exactly: the `temperature=0`
tests passed, and the one test using `temperature=0.1` turned every later request
into `Connection refused`.

**`top_p` is optional; `temperature` is not**, so this is the likelier of the two
to find a real client. Whether a given image carries the offending branch cannot
be settled from its tag — it needs the probe. (The day-0 image, probed on-node
12 Aug 2026, turned out **not** to have it: `temperature > 0` was safe there. The
default image since 20 Aug carries the fix outright.)

`serve-kimik3.sh` greps the image at launch for both the two Triton aliases *and*
that binding, and warns when either is missing, so you cannot enable DSpark and
meet these unknowingly. It reads the `elif is_hip():` block specifically, because
the call site for the unbound symbol is present either way and a whole-file grep
proves nothing. **The default image since 20 Aug 2026 has both fixes**, so seeing
either warning means `SGLANG_IMAGE` points at a pre-August build. Options,
cheapest first: move forward; keep clients at `top_p: 1` and `temperature: 0`; or
leave `SPECULATIVE` unset.

It stays off by default anyway, so that a first launch has one thing to go wrong
instead of two — turn it on once baseline serving is proven.

### The draft repo is a moving target — pin it

`RadixArk/Kimi-K3-DSpark` is **rewritten under you**. The numbers above were
measured against commit `eb03982e` (27 Jul 2026). Since then:

| Commit | Date | What |
|---|---|---|
| `eb03982e` | 27 Jul 2026 | **the revision every DSpark number above was measured on.** Trained context 4,096, unscaled RoPE — see [why that matters](#the-cause-the-pinned-draft-was-trained-at-4096-tokens) |
| `9c4b2577` | 31 Jul 2026 | *"Sync Kimi-K3-DSpark-0731 snapshot"* — weights replaced, YaRN ×16 added, trained context 65,536 |
| `56ce616a` | 1 Aug 2026 | **the default since 12 Aug 2026.** Same weights and `config.json` as `9c4b2577`, fuller card |

Three rewrites in four days, two of which broke the launch outright. So the
script pins a sha rather than tracking `main`, and following the branch is
something you opt into. **The pin moved on 12 Aug 2026** from `eb03982e` to
`56ce616a` — not drift-chasing, but because `eb03982e` is a 4k draft and this
deployment serves a 1M window. Set `eb03982e` explicitly to reproduce the tables
above; anything else, take the default.

*Hit for real on 1 Aug 2026*: a re-download picked up a newer revision and the
server then died during **argument parsing**, before loading anything, with
`ValueError: Unrecognized model in RadixArk/Kimi-K3-DSpark. Should have a
model_type key in its config.json`. SGLang resolves the draft's config to pick
the speculative algorithm, so a bad or truncated draft config kills the launch
early — which at least is cheap.

The pin is the default, so all you need is to have fetched it:

```bash
./serve-kimik3.sh download     # fetches the pinned DSPARK_REVISION
```

To follow the branch instead, set it to a ref rather than a sha:

```bash
# kimik3.env
export DSPARK_REVISION="main"      # accept the drift
```

A 40-hex value names a snapshot directly; anything else is resolved through
`refs/<name>` the way the hub does. Either way the script hands SGLang the
**resolved snapshot path**, never the repo id — a repo id makes it re-resolve
through the hub, which is how an earlier version of this preflight validated one
snapshot while the server read another. The preflight also checks the cached
draft has a parseable `config.json` with a `model_type`, and says what to do
rather than letting upstream raise the confusing error.

Note the throughput crossover: DSpark *loses* above ~concurrency 16 (3715 vs
4898 tok/s at c=32). It is a latency optimisation for interactive use, not a
throughput one.

### DSpark collapsed at long context — because the draft was a 4k model

**Every DSpark number in this README was measured at 1024 tokens of context.**
By 100k the sign had flipped and DSpark cost several times more than it returned.
The cause was found on 12 Aug 2026 and it is **the draft checkpoint, not DSpark**
— [skip to it](#the-cause-the-pinned-draft-was-trained-at-4096-tokens) if you want
the answer before the evidence. Two sessions, both `#running-req: 1`:

| Session | context | accept len | gen throughput |
|---|---|---|---|
| opencode | 217k | 1.23 | — |
| kimicode, 9 Aug 2026 | 106k | 1.02–1.45 | **8.0–11.4 tok/s** |

Divide throughput by accept length on every decode line of the second one and it
is the same number to within 2%:

```
8.88/1.15=7.72   9.27/1.20=7.73   9.83/1.27=7.74   10.22/1.32=7.74
8.68/1.12=7.75   8.49/1.10=7.72   8.03/1.02=7.87   11.35/1.45=7.83
```

**7.8 verify steps/s — 128 ms per step, flat.** The token rate is doing nothing
but tracking accept length. Against this file's own c=2 bench (40.9 ms/step,
accept 7.29) that is 3.1× the step cost for 6.3× fewer tokens: 19.7× total,
predicting 9.1 tok/s where 8.0–11.4 was observed. The decomposition is exact,
which is what makes this a diagnosis rather than a guess.

`(accept_len − 1) / accept_rate` recovers the draft block: `(1.45−1)/0.06 = 7.5`,
`(1.20−1)/0.03 = 6.7`. **The draft proposes ~7 tokens per step and ~0.15 survive.**
So each 128 ms buys seven draft forwards plus an eight-token verify through the
full model, and returns 1.15 tokens. The draft also attends over the whole 106k
context on all seven of those passes, so its cost grows with context where the
target's barely does — that is the 3.1× step inflation (40.9 → 128 ms), and it
is the half of the problem that **survives any fix to the accept rate**.

It is not memory pressure. Both sessions ran at `full token usage: 0.16` and
`mamba usage: 0.03` — nothing is thrashing or being evicted.

```bash
# kimik3.env — for opencode / kimicode / anything that resends a large context
export SPECULATIVE=""
```

**Measured, bun160, 9 Aug 2026: turning DSpark off is worth 2.93× at 100k** —
median TPOT 87.38 → 29.83 ms, decode 11.4 → 33.5 tok/s, with TTFT unchanged.
Full numbers, the ITL distribution, and what they say about where the 121.5 ms
actually goes are in
[Long context](#long-context--the-agentic-shape-measured).

`/compact` appears to fix it, and does help for two real reasons — a smaller
attention working set, and putting the draft back inside the context window it
was trained on — but it is treating the symptom. At 106k you are using 11% of a
1M window and should not be at 8 tok/s.

#### The cause: the pinned draft was trained at 4096 tokens

Checked 12 Aug 2026. The revision this repo pinned until then, `eb03982e`
(27 Jul), **says so on its own model card**:

> **Trained context:** sequence length 4096.
>
> This DSpark checkpoint was trained with a maximum context length of 4,096
> tokens, which **may lead to reduced acceptance lengths in extreme long-context
> and agentic use cases**. Training for these scenarios is ongoing.

Its `config.json` matches — `"rope_type": "default"`, no extension of any kind,
under a `max_position_embeddings` of 1,048,576 that the weights cannot honour.
**Our 1024-token bench ran inside that window. 106k is 25× outside it.** Accept
7.25 at 1k and 1.39 at 100k is not a mystery, it is a draft being asked for
positions it never saw.

Upstream replaced the checkpoint four days later. `9c4b2577` (31 Jul, *"Sync
Kimi-K3-DSpark-0731 snapshot"*) swapped the weights — same 4.50 GB, different LFS
oid `29df0e8e…` → `ecd74645…` — and rewrote the rope block:

| | `eb03982e` (27 Jul) | `56ce616a` (1 Aug, **the default now**) |
|---|---|---|
| `rope_type` | `default` | **`yarn`** |
| `factor` | — | **16.0** |
| `original_max_position_embeddings` | — | **65536** |
| card | "A DSpark speculator for Kimi K3" | **"A long-context DSpark speculator… up to 1 million tokens"** |
| stated trained context | 4,096 | 65,536 (×16 → 1M) |

The new card's own numbers: **RULER V2 at 1M input — actual prompts 1,000,432 to
1,047,925 tokens — acc_len 4.2553**, SWE-Rebench 4.66, and a 32K+ output bucket
at 4.92. Corroborated in the field: sglang
[#32855](https://github.com/sgl-project/sglang/issues/32855) (3 Aug) — *"re-pull
the draft model, which has been re-trained for long contexts… stable 110–400 tps
single request within 200k, acc len almost firmly 1.5×–3.8×."*

**`DSPARK_REVISION` now defaults to `56ce616a…`**, and `serve-kimik3.sh` warns if
the draft you resolved has unscaled RoPE. It probes the checkpoint's *rope type*,
not its sha, so it keeps working the next time upstream rewrites the repo.

#### The fix works, and it moves the problem rather than removing it

Measured the same day, bun161. Accept length at 100k went **1.39 → 6.90** — past
the 3–4.5 forecast, and level with what the draft achieves at 1024 tokens. But
TPOT only improved 87.38 → 52.04 ms, still **1.7× worse than turning DSpark off**,
because the step cost tripled to 359 ms. Gamma does not help: γ3 gives 51.13 ms,
1.8% away. Full numbers, and one figure that does not reconcile, in
[the long-context table](#measured-at-100k--bun160-9-aug-and-bun161-12-aug-2026).

What changed is *where* the line falls. DSpark at **32k** is 17.20 ms TPOT against
an interpolated ~28 ms baseline — a 1.6× win — so the sign flips somewhere around
**55–60k**, not below 1024. See
[the crossover](#the-crossover-is-around-5560k--and-that-is-the-operating-policy).

```bash
# kimik3.env — sessions that stay under ~50k
export SPECULATIVE="dspark"      # 4.7x at 1k, ~1.6x at 32k
# kimik3.env — sessions that run to 100k+
export SPECULATIVE=""            # 1.7x better there, at any gamma
```

**On this image the first line is unsafe with a real client**, because DSpark's
verify reaches renorm kernels the 27 Jul build does not bind — one `top_p < 1.0`
request kills the server. That is the single strongest reason to move to a
mainline image, ahead of any of the performance PRs. See
[The sampling landmine](#the-sampling-landmine--read-this-before-enabling-dspark)
and [Moving to a mainline image](#moving-to-a-mainline-image).

---

## Performance tuning

Measurement-driven only — change one thing, re-run `./bench-kimik3.sh`, keep it
if the numbers improve. Run from a shell on the serving node:

```bash
./bench-kimik3.sh sweep          # 1024/512 at c=2/8/32 -> $MODEL_CACHE_DIR/bench/
./bench-kimik3.sh longcontext    # 100k/512 at c=1 — the agentic shape
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

### Long context — the agentic shape, measured

Every table above is **1024 tokens in**. opencode and kimicode resend 100k+ every
turn, and those are not the same regime scaled up — DSpark is a 4.7× win at 1024
and a 1.7× loss at 100k. This section is where that was measured.

`./bench-kimik3.sh longcontext` measures the agentic shape: 100k in / 512 out at
concurrency 1, `n=4`, `--random-range-ratio 1.0`. It refuses to start if the shape
exceeds a set `CONTEXT_LEN`, and warns when `SPECULATIVE=dspark`, because the
DSpark/no-DSpark **pair** is the measurement — one run alone says nothing.

`BENCH_INPUT_LEN=32768 ./bench-kimik3.sh longcontext` moves the shape for a
mid-point. A `kimik3.env` carrying the sweep shape does not, **even if you have
sourced it into your own shell** — the mode compares what is in the environment
against what the config file sets and ignores an exact match. Both halves of that
were learned by silently running `longcontext` at 1024/512 c=2/8/32, twice.

```bash
./bench-kimik3.sh longcontext                    # A: as configured
# then set SPECULATIVE="" in kimik3.env
./serve-kimik3.sh stop && ./serve-kimik3.sh serve --detach
./bench-kimik3.sh longcontext                    # B: same shape, DSpark off
```

Expect a long wait before the first token — prefill at 100k is the slow path on
`ATTENTION_BACKEND=triton`, and that TTFT is a headline result, not a warmup
cost. It is also the number
[AITER MLA prefill](#the-biggest-lever-we-still-cannot-pull--aiter-mla-prefill)
would move most.

#### Measured at 100k — bun160 9 Aug and bun161 12 Aug 2026

Every row is one variable against the previous one. **Read TPOT, not output
tok/s** — at this shape output throughput is end-to-end and a 15 s prefill
dominates it, so it says more about TTFT than about decode.

| Config | context | median TPOT | decode tok/s | median TTFT | accept len |
|---|---|---|---|---|---|
| DSpark γ7, `eb03982e` draft *(4k, unscaled RoPE)* | 100k | 87.38 ms | 11.4 | 15.5 s | 1.39 |
| **`SPECULATIVE=""`** | 100k | **29.83 ms** | **33.5** | 15.4 s | — |
| DSpark γ7, `56ce616a` draft *(long-context retrain)* | 100k | 52.04 ms | 19.2 | 15.4 s | **6.90** |
| DSpark γ3, `56ce616a` | 100k | 51.13 ms | 19.6 | 15.5 s | 2.42 |
| **DSpark γ3, `56ce616a`** | **32k** | **17.20 ms** | **58.1** | **1.86 s** | 2.86 |
| DSpark γ7, `56ce616a`, c=2 sweep shape | 1024 | — | 309–327 *(gen)* | — | 6.1–7.7 |

**Every row in this table is the day-0 `rocm720-mi35x-k3-20260727` image**, which
stopped being the default on 20 Aug 2026. Two merged changes in the new default
move these numbers — #33981 rewrote the DSpark verify attention, and #34580's
`MLA_DECODE_TUNE` retunes MLA decode with DSpark on *or* off — so treat rows 3–5
as the pre-migration baseline and
[the runbook](#the-runbook-in-order--as-of-20-aug-2026) as what replaces them.

Rows 1–2 are bun160, 9 Aug 2026. Rows 3–6 are bun161, 12 Aug 2026, after the draft
pin moved. Row 6 was an accidental run — a sourced `kimik3.env` leaked the sweep
shape past a named mode — and turned out to be the control that matters: at 1024
tokens the retrained draft matches the old one exactly (29 Jul gave 306 gen tok/s
at c=2, accept 7.25–7.29), so **nothing the retrain changed costs anything at
short context.**

**The prediction failed, in the useful direction.** Row 3 was predicted to land
near row 2 at ~33 tok/s, on the reasoning that a better draft cannot change the
128 ms step cost. Accept recovered far past the forecast — 1.39 → **6.90**, where
the card's 1M RULER figure of 4.26 suggested 3–4.5 — and TPOT still only reached
52.04 ms, because the step cost *tripled* to `52.04 × 6.90` = 359 ms. The draft
was the whole of the accept-length problem and none of the throughput problem.

**Gamma is not a lever at 100k.** Rows 3 and 4 differ by 1.8% on TPOT while accept
length differs by 2.85×; step time and accept fell together and the token rate did
not move. Gamma was confirmed live from the log — `(accept_len − 1) / accept_rate`
returned 3.00, 3.00, 2.98, 3.00, 3.01 on five consecutive decode lines — and the
step rate was flat at 8.56 steps/s (117 ms), matching `51.13 × 2.42` = 124 ms.

*One thing does not reconcile.* Row 4 is internally consistent: a 124 ms step
against a max ITL of 118.73 ms and P90 of 115.63 — the ITL tail **is** one step.
Row 3 is not: a 359 ms step should put ~74 gaps of ~359 ms into a 512-token
stream, and its max ITL is 124.12 ms. Either the reported accept length is not
tokens-per-verify-step in the sense used here, or ITL is not recording the burst
boundary. Both rows behave as though the real step is ~120 ms yielding ~2.4
tokens, which is what γ3 reports and not what γ7 does. **Unresolved** — the test
is to count `#full token` deltas across decode lines in the server log, which is
independent of both figures. (It works for γ3: 379 tokens in 19 s = 19.9 tok/s,
matching its TPOT.)

#### The crossover is around 55–60k — and that is the operating policy

DSpark is a 4.7× win at 1024 and a 1.7× loss at 100k. Interpolating the *step*
cost — 49 ms at 32k, 124 ms at 100k — against a decode baseline that is nearly
flat with context puts the sign change at roughly **55–60k**.

| context | DSpark | `SPECULATIVE=""` | |
|---|---|---|---|
| 1k | **5.61 ms** (γ7) | 26.49 ms | DSpark **4.7× win** |
| 32k | **17.20 ms** (γ3) | ~28 ms *(interpolated)* | DSpark **~1.6× win** |
| 100k | 51.13 ms (γ3) | **29.83 ms** | DSpark **1.7× loss** |

The `~28 ms` is not measured. It is the safest interpolation in this file:
[decode is nearly context-independent](#k3s-decode-is-nearly-context-independent--the-earlier-claim-was-wrong),
26.49 ms at 1k against 29.83 ms at 100k. Confirming it costs a full reload and the
conclusion does not hinge on it.

TTFT matters as much as TPOT here: **1.86 s at 32k against 15.5 s at 100k**, 8.3×
for 3× the context. Prefill is superlinear where decode is flat — that is the
[AITER MLA prefill](#the-biggest-lever-we-still-cannot-pull--aiter-mla-prefill)
lever, still stalled upstream.

So, for agentic sessions, in one line:

> **Compact at 40–50k and leave DSpark on: ~58 tok/s and 1.9 s to first token.
> Let it drift to 100k and the best available is DSpark off at 33.5 tok/s and
> 15.5 s.** Roughly 3× the throughput and 8× the latency, decided by session
> hygiene rather than by any setting.

**That policy is measured on the retired image and it is the thing most likely to
change next.** The 20 Aug default carries #33981, which attacks the DSpark step
directly, and #34580's `MLA_DECODE_TUNE`, which speeds MLA decode on both sides of
the comparison. Both push the crossover **up**, never down — the question is by
how much, and whether it clears 100k. Until steps 5–7 of
[the runbook](#the-runbook-in-order--as-of-20-aug-2026) have been run, treat
55–60k as the last number anyone measured rather than as the current answer.

That closes the question this repo opened with on 9 Aug — *"is the answer to
compact aggressively, or is something wrong?"* Both. A 4,096-token draft was
serving a 1M window, **and** compaction is genuinely correct, at a threshold that
is now measured rather than guessed.

**Untested and worth one reload:** γ7 at 32k. γ3 gave accept 2.86 there; at 1024
γ7 gives 7.29. If γ7 holds a higher accept at 32k against a similar step, 17.20 ms
improves further.

**The step-rate model survives contact with a controlled run.**
`87.38 ms × 1.39 = 121.5 ms` per verify step, against the **128 ms** derived from
a kimicode session's decode lines — 5% apart, different session, different
content.

**And it isolates the cause.** This run used *random-token* prompts and still
landed at accept 1.39, inside the 1.02–1.45 a real kimicode session showed at
106k. The same random generator at 1024 tokens gives 7.25–7.29. Same dataset,
same config, only context length differs: **the collapse tracks context length,
not workload content.** That closes the question this section was opened with.

TTFT of 15.5 s is a *cold* 100k prefill — four unrelated random prompts share no
prefix. A real agentic turn hits the radix cache (`#cached-token: 105344` against
`#new-token: 18`) and pays almost none of it. Do not quote this number as what
opencode users experience; quote it as what
[AITER MLA prefill](#the-biggest-lever-we-still-cannot-pull--aiter-mla-prefill)
would attack.

**Turning DSpark off is worth 2.93× at 100k.** TTFT moved 1% (15359 vs 15526 ms)
— prefill never touches the draft, which is the control that makes the TPOT
comparison trustworthy.

The inter-token latency distribution confirms the mechanism outright:

| | median ITL | P90 ITL | max ITL |
|---|---|---|---|
| DSpark on | 61.19 ms | 121.95 ms | 126.23 ms |
| `SPECULATIVE=""` | 29.84 ms | 30.05 ms | 31.28 ms |

DSpark's ITL is **bimodal at 121.5 ÷ 1 and 121.5 ÷ 2** — steps that accepted one
token and steps that accepted two, against the 121.5 ms step time computed
independently from run A. Turning it off also removes a 2× latency jitter that
the means do not show, which for interactive use matters on its own.

#### K3's decode is nearly context-independent — the earlier claim was wrong

This file previously attributed the long-context decode step to the triton MLA
kernel. **That is false.** The 1024-token baseline is 26.49 ms TPOT; at 100× the
context it is 29.83 ms — 13% slower. That fits the architecture: KDA on every
layer but the 4th, so only a quarter of the layers carry a growing KV read.

So run A's 121.5 ms step was almost entirely *the draft*: the verify pass is
~30 ms (a target forward; 8 tokens versus 1 barely matters when decode is
memory-bound), leaving ~91 ms for seven draft forwards at ~13 ms each. A draft
model costing 43% of a full K3 forward is the whole story, and no attention
backend fixes it.

Two consequences:

- **The aiter decode backend has little headroom.** At most ~11% separates 100k
  decode from 1k decode, so row 3 is now a small experiment, not a lever.
- **The remaining long-context weakness is prefill, not decode** — the 15.5 s
  cold TTFT, which is
  [AITER MLA prefill](#the-biggest-lever-we-still-cannot-pull--aiter-mla-prefill)
  territory and unavailable. In normal agentic use the radix cache means you
  rarely pay it.

`BENCH_REPEATS=1` here, so there are no error bars. That is acceptable only
because the effect being tested is multiples, not percent — tighten it before
claiming any difference under ~20%. The 2.93× is far outside anything n=4 noise
could produce; the 13% context penalty above is **not**, and should be re-run
before it is relied on.

### Nodes are all ROCm 7.14 now — what that settled, and what it did not

Until Aug 2026 the MI355X nodes ran a mix: bun161 on 7.14, bun159 on the same
7.2 the container ships. A sweep on each showed bun161 4.6–9.7% ahead on total
throughput, and **that did not establish anything.** n=1 per node, so no variance
estimate; driver version confounded with which physical machine you got; and the
signs disagreed — TPOT +11.4% at one concurrency and −1.7% at another, which is
what noise looks like. Within a single run, TPOT P90 sits 15–23% above the
median, so a 5% gap between two single runs is not a result.

The one thing worth keeping from it is the **null**: accept length differed by
~1.4% across the two stacks. If the older ROCm were perturbing numerics,
draft/target agreement is the sensitive detector, and it did not move. Both were
computing the same thing.

The question is now moot — every node is 7.14 — and it can no longer be answered,
because separating the driver from the machine needed the *same* node measured
before and after its upgrade, and that upgrade has already happened. Recorded
here as the shape of a comparison worth not repeating.

For anything you *do* want to compare, get the error bar first:

```bash
BENCH_REPEATS=5 ./bench-kimik3.sh sweep
```

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

### K3's KDA state pool — the knobs we did not pass

K3 is a **hybrid**: of its 93 layers, 69 are KDA (Kimi Delta Attention, linear)
and 24 are MLA. The MLA layers hold an ordinary KV cache; the KDA layers hold a
**recurrent state per request**. `--mem-fraction-static` buys one pool of memory
and these knobs decide how it is split — and how many slots each request costs.
At high concurrency what limits admission is usually state slots, not KV.

Until 2 Aug 2026 this repo passed none of them, so everything below ran at
upstream defaults. They are now exposed, still defaulting to those same values:

| Variable | Flag | Default | What it trades |
|---|---|---|---|
| `MAMBA_FULL_MEMORY_RATIO` | `--mamba-full-memory-ratio` | 0.9 | KDA state pool vs MLA KV pool |
| `MAMBA_SSM_DTYPE` | `--mamba-ssm-dtype` | fp32 | `bfloat16` halves state memory |
| `MAMBA_RADIX_STRATEGY` | `--mamba-radix-cache-strategy` | auto → `extra_buffer` | slots/request: 5, `extra_buffer_lazy` 4, `no_buffer` 3 |
| `MAMBA_SKIP_DECODE_LOCK` | `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` | off | one more slot/request, experimental |
| `REPLAYSSM_SPEC` | `--enable-gdn-replayssm-spec` | off | DSPARK only — frees slots for concurrency |
| `DSPARK_BLOCK_SIZE` | `--speculative-dspark-block-size` | inferred (7) | pins gamma against draft drift |

The one to look at first is **`MAMBA_FULL_MEMORY_RATIO`**. Its default of 0.9 is
a default, not a fit for any particular workload: the right split follows your
**mean request length**, since state cost is per-request while KV cost is
per-token. Upstream publishes a calculator on the [Kimi-K3 cookbook
page](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) — use it
rather than guessing.

When testing `MAMBA_SSM_DTYPE=bfloat16`, **watch accept length, not just
throughput**. Draft/target agreement is the sensitive detector for a numerics
change — the same reasoning as the ROCm-version section above, where the *null*
result on accept length was the informative part. If it drops, revert.

#### The flag upstream documents does not exist

The K3 cookbook prescribes `--enable-linear-replayssm-spec` for **every** DSPARK
recipe. That flag does not exist — there is no such argument anywhere in the tree
(checked against the `kimi-k3` branch, 2 Aug 2026). The real one is
**`--enable-gdn-replayssm-spec`**. Copying the cookbook verbatim gets a startup
rejection after the container is up.

Worse is the plausible correction. There *is* an `--enable-linear-replayssm`, and
it is a different, mutually exclusive flag — its own help says **KDA decode is
slower** with it than the packed baseline. So the obvious repair to the cookbook
name is a silent regression rather than an error.

`REPLAYSSM_SPEC=1` selects the right one, and **the reason it is off changed on
20 Aug 2026**. It used to be unavailable: the day-0 `rocm720-mi35x-k3-20260727`
image predates sglang #32692 (31 Jul), which is what lets ReplaySSM coexist with
the `extra_buffer` radix strategy we run, and the script probes `--help` and
refuses up front rather than letting you find out after a 1.5 TB load. The
[current default image](#moving-to-a-mainline-image) has the flag. What is missing
now is a measurement — nobody here has benched it — so it stays off until someone
does.

#### Measured

Nothing yet. Each row is one variable against the 29 Jul DSpark baseline
(1024/512, TP8) — and a knob that does not help is a result worth writing down,
same as the rest of this file.

| Variable | Value | Node / date | c=2 | c=8 | c=32 | accept len | Verdict |
|---|---|---|---|---|---|---|---|
| — | baseline | | | | | | |
| `DSPARK_BLOCK_SIZE` | 7 | | | | | | |
| `MAMBA_SSM_DTYPE` | bfloat16 | | | | | | |
| `MAMBA_FULL_MEMORY_RATIO` | | | | | | | |
| `MAMBA_RADIX_STRATEGY` | extra_buffer_lazy | | | | | | |
| `MAMBA_SKIP_DECODE_LOCK` | 1 | | | | | | |
| `REPLAYSSM_SPEC` | 1 | | | | | | |

**Not recommended: `--moe-runner-backend`.** It appears in every NVIDIA cell as
`marlin`, and `aiter` turns up in the AMD *Inkling* recipe — but the K3 MI35x
cell deliberately omits it and drives the MoE through the `SGLANG_USE_AITER` /
`SGLANG_AITER_K3_OPT` / `AITER_FLYDSL_FORCE` / `AITER_SITUV2_A8W4` variables this
repo already sets. Adding it would be borrowing a setting from a different model.

---

## Weight loading — why a cold start sawtooths

A cold start moves 1561 GB off scratch, and it does not move it smoothly: bursts,
dips, and long slow stretches on storage and a network capable of far more. The
cause is one line of upstream behaviour.

**SGLang silently falls back to single-threaded weight loading** whenever its
checkpoint prefetch is active and you have not asked for threads explicitly
(`model_loader/loader.py`):

```
--weight-loader-prefetch-checkpoints is enabled; falling back to single-threaded
weight loading to avoid I/O oversubscription with the prefetch threads. Set
enable_multithread_load=true in --model-loader-extra-config to keep
multi-threaded loading.
```

One sequential reader against a parallel filesystem gives exactly the observed
shape: a burst while a shard streams, a dip while it is converted and copied to
HBM, then the next shard. Naming either `enable_multithread_load` or `num_threads`
in the JSON suppresses the fallback — that is what `WEIGHT_LOAD_THREADS` (default
**8**) does.

Check which one you got:

```bash
./serve-kimik3.sh loadstat
```

### Bunya is GPFS, not Lustre

Worth stating because it invalidates the reflex. The [Bunya User
Guide](https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md)
is explicit — "the GPFS filesystem that underpins your `/scratch` and `/home`, as
well as `$TMPDIR`" — so there is **no `lfs setstripe`** to reach for, and `$TMPDIR`
is the same filesystem, so staging weights there buys nothing. The lever is
client-side parallelism, not storage layout.

### What is actually achievable

Not "hundreds of gigabits". 400 Gb/s would put 1561 GB in ~31 s; the floor is set
by H2D copy, MXFP4 handling, and how much aggregate GPFS bandwidth a *single*
client can absorb. Expect single-digit to low-tens of GB/s — minutes, not seconds.
The win available is a large constant factor and *consistency*, not two orders of
magnitude.

### Measured

> **Not yet measured on Bunya — the knobs landed 1 Aug 2026, the A/B has not been
> run.** `WEIGHT_LOAD_THREADS=8` is upstream's own recommendation, not a local
> measurement. Fill this table before treating any of it as fact.

| Config | Cold? | Time to ready | Effective GB/s |
|---|---|---|---|
| `WEIGHT_LOAD_THREADS=0` (baseline) | | | |
| `WEIGHT_LOAD_THREADS=8` (default) | | | |
| `WEIGHT_LOAD_THREADS=16` | | | |
| `LOAD_FORMAT=presharded`, dump run | | | |
| `LOAD_FORMAT=presharded`, restart | | | |

Every start appends a row to `$MODEL_CACHE_DIR/kimik3-loadtimes.log`, which
survives the server log being truncated on each launch. `loadstat` prints it.

> ⚠️ **Compare cold runs only.** Host RAM is 1800 GB and the weights are 1561 GB,
> so after any successful load most of the model is in page cache and the *next*
> start is fast no matter what the thread count is. A/B-ing back to back on one
> node flatters whichever ran second and proves nothing. Use a fresh allocation
> or a different node for each timed run, and record which were cold.

### Presharded, for allocations you restart in

`LOAD_FORMAT=presharded` loads normally once, then dumps a per-rank,
already-quantised checkpoint to `PRESHARDED_PATH`. Every later start reads only
its own 1/8 and skips re-quantisation — the structural fix rather than a faster
version of the same work. It costs one slow load and up to another 1561 GB of
scratch, so it pays off across a 48-hour allocation and not in a single run. The
script preflights the free space and refuses a dump it cannot finish.

With `SPECULATIVE=dspark` the draft is a second model and needs its own dump root,
or it writes into the read-only HF cache mount. The script sets
`draft_presharded_path` to `$PRESHARDED_PATH/dspark` for you.

### If more threads do not help

Watch `rocm-smi` and the log during a start. If HBM fills in bursts while the CPUs
idle, the bottleneck is read-side and threads are the right knob. If reads are
smooth but HBM lags, it is conversion and H2D — more reader threads will not help,
and `presharded` (which removes the re-quantisation work entirely) is the lever.

---

## Upstream drift — checked 20 Aug 2026

Everything here moves fast enough that a snapshot goes stale in days, so this
section records **what was found and how to re-run the check**, not just the
answer.

### Containers

| | |
|---|---|
| **Default here since 20 Aug** | **`lmsysorg/sglang-rocm:v0.5.17-rocm724-mi35x-20260820`** (25.8 GB) |
| Anchor for pre-20-Aug numbers | `rocm720-mi35x-k3-20260727` (29.2 GB, pushed 28 Jul) — day-0 branch build |
| Newest K3-branch tag | `rocm720-mi35x-k3-20260803` — **the last one. That stream is retired** |
| Nightly flavours | `gfx942`/`gfx950` × `rocm700`/`rocm720`/`rocm724`, all dated daily |
| Why `rocm724` | upstream made 7.2.4 its **primary AMD PR gate** on 20 Aug (#35602), runs both nightly (#35603) |
| Not yet in `rocm724` | torch 2.11 + triton 3.7 (#30984, merged 20 Aug) — first appears in `20260822` or later |
| aiter inside every image | `AITER_COMMIT=d9e5ef7c` — **still 29 Jul**, unchanged in three weeks |
| ROCm 7.14 | `rocm/pytorch` has twelve `rocm7.14_*` bases (16 Jul), **sglang has no 7.14 stage** |

**The ROCm 7.14 row is not the blocker it looks like.** The base images exist —
`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.1x.0` and eleven
siblings, all pushed 16 Jul. What does not exist is an **sglang** 7.14 stage:
`docker/rocm.Dockerfile` builds `rocm700`, `rocm720` and `rocm724`, and nothing
else. Building one needs an Apptainer def and fakeroot, because Bunya has no
Docker, and it would be an unsupported flavour serving a model whose AMD path
changes weekly. **`ROCMINFO_SHIM` stays necessary**, and waiting is still the
right posture — see [the ROCm section](#the-node-is-rocm-714-the-image-is-724--and-that-is-fine).

**The `20260810` recommendation is retired — take the newest.** The 12 Aug review
capped the migration target at `20260810` because the 11 Aug build lost a 5.38 GB
layer (40 → 39 layers, 29.60 → 23.65 GB) and this repo's aiter procedure depends
on the image shipping prebuilt `module_*.so` to seed `AITER_JIT_DIR` from. That
layer has now been identified: commit `6f3fe13` (11 Aug), *"[AMD] Install AITER's
pinned Triton wheel in the ROCm 7.2 image"*, deleted a clone-and-build-Triton-from-
source block and replaced it with aiter's own `install_triton.sh`. The diff leaves
`PREBUILD_KERNELS=1` and aiter's `build_ext --inplace` untouched. **The missing
layer was a build tree, not the kernels.** `check` still verifies it in a minute,
and should.

**The images' aiter is now three weeks behind aiter's own K3 work.** Every
published image, through `20260820`, still pins `AITER_COMMIT=d9e5ef7c` (29 Jul);
that default in `docker/rocm.Dockerfile` has not moved in eight days. Since it,
`ROCm/aiter` `main` has landed `ca68b4f3` *tune Kimi-K3 prefill GEMMs for gfx950*
(10 Aug), `05ec7fd9` *runtime-keyed Kimi-K3 A8W4 fmoe config* (10 Aug), `868ac1f7`
*correct + faster a16w4 SiTUv2* (8 Aug), `243bebb` *tune chunked_pa_prefill params
for gfx950* (19 Aug), `0159273` *[FlyDSL] MoE GEMM workgroup-cluster multicast*
(20 Aug) and `9279f97` *[gfx950] retune small-M tiles in the A16W16 fallback*
(20 Aug). **None of it ships in any image.** So: **check the image's
`AITER_COMMIT`, not its tag date** — `./serve-kimik3.sh check` now prints it.

### The branch merged; mainline is the only live stream

**PR #32541 merged to `main` on 4 Aug 2026 and the `kimi-k3` branch was deleted**
(404). `main` carries `srt/models/kimi_k3.py`, the whole `srt/speculative/`
dflash/dspark stack, `kernels/ops/sampling/renorm_triton.py`, and
`_mla_decode_fwd_with_head_pad`. The `rocm720-mi35x-k3-*` tags stopped the day
before, at **`20260803`**, and have no successor — that stream is retired, not
paused.

The mainline nightlies are **`main` itself**, not a release branch:
`.github/workflows/release-docker-amd-rocm720-nightly.yml` builds the
`gfx942`/`gfx950` × `rocm720`/`rocm724` matrix from a plain default-branch
checkout on a `0 12 * * *` cron.

**A tag date is about a day later than the code in it.** `DATE=$(date +%Y%m%d)`
is evaluated at *push* time, not at checkout, and these builds are long:
`v0.5.17-rocm720-mi35x-20260820` was pushed at **09:51 UTC on 20 Aug**, which is
before that day's cron even started. It is the run that began at 12:00 UTC on the
**19th**. The 12 Aug version of this section said `…-20260810` was `main` as of
10 Aug; it is closer to `main` as of the 9th. Working rule: **a tag contains a PR
only if the tag is dated at least two days after that PR merged.**

Mainline also settles the sampling landmine at the source — `main`'s
`dflash_utils.py` binds both kernels under `is_hip()`, and #33694 binds the
non-greedy verify symbol alongside them. On this repo's own measurements
([the crossover](#the-crossover-is-around-5560k--and-that-is-the-operating-policy))
that is what turns DSpark from bench-only into something a real client can use,
which makes it the strongest single reason to migrate — ahead of any performance
PR. Procedure in [Moving to a mainline image](#moving-to-a-mainline-image), always
into a second `SIF_PATH`.

**Upstream's cookbook has moved too, and it went the same way.** On 12 Aug its
MI355X cell still prescribed `rocm720-mi35x-k3-20260727` — the image this repo
had pinned. `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` on `main` now says
`v0.5.17-rocm720-mi35x-20260817`, still marked `verified: false`,
`verificationStatus: 'in-progress'`. Note that their pin predates
[#34580](#the-second-lever--34580-retuned-the-gfx950-mla-decode-geometry) by a
day, so `20260820` puts this deployment slightly ahead of the documented recipe
on the one change that matters most here.

Their flags are otherwise ours: the same four aiter env vars,
`--attention-backend triton`, `--tp-size 8`, `--dtype bfloat16`,
`--mem-fraction-static 0.85`, both `kimi_k3` parsers. Two differences remain, and
both are deliberate on this side. They set `--kv-cache-dtype fp8_e4m3` where this
repo leaves it empty — do not adopt it on faith, sglang
[#32938](https://github.com/sgl-project/sglang/issues/32938) reports fp8 KV
*slowing* DSpark. And they still write `--cuda-graph-max-bs 256`, a spelling
mainline renamed; `--cuda-graph-max-bs-decode` is the current one.

**AMD now runs K3 on MI35x in sglang's own nightly CI**, which is new since the
last check: [#32568](https://github.com/sgl-project/sglang/pull/32568) (accuracy,
17 Aug) and [#34985](https://github.com/sgl-project/sglang/pull/34985) (perf,
18 Aug). GSM8K **0.953** on 8×MI355X against a 92% threshold, and a sweep at 4096
input, batch [1, 1, 8, 16, 64], peaking at **1403.86 tok/s** output at batch 64.
Their harness sets this repo's four env vars, `--tp 8`,
`--attention-backend triton`, `--cuda-graph-max-bs-decode`,
`--max-running-requests` — and **no `--speculative-*` flags at all**. Upstream's
MI35x reference config is non-speculative, which is a mild independent vote for
the policy [below](#the-crossover-is-around-5560k--and-that-is-the-operating-policy).
It is also the first AMD figure published with a stated shape, unlike the
820/2356/4898 numbers this README takes apart earlier.

Branch commits between the pinned build and the merge that still matter:
**#32621** (28 Jul) and **#32641** (31 Jul), the two halves of the HIP renorm
binding; **#32692** (31 Jul), letting ReplaySSM coexist with the `extra_buffer`
radix strategy ([the KDA section](#k3s-kda-state-pool--the-knobs-we-did-not-pass));
and *"[Kimi K3] Add reasoning, tool-call, and OpenAI serving"* (1 Aug), which is
why `./serve-kimik3.sh parsers` is worth re-running against any newer image.

**Merged into `main` since, i.e. present in the `20260820` default:**

- **#33981** (8 Aug) — the DSpark verify kernel. Its own section below.
- **#33599** (5 Aug) — fuses K3's attn-residual aggregation on ROCm: `_score_kernel`,
  `_combine_kernel`, the out_norm RMSNorm, the pending residual add and the bank
  snapshot into one Triton kernel, **7 launches down to 2 per layer**. Also
  cherry-picked to `release/v0.5.17`.
- **#33764** (8 Aug) — fixes router GEMM inaccuracy when K3 uses `_front_w`. A
  correctness fix, not a perf one.
- **#33694** (6 Aug) — binds `tree_speculative_sampling_target_only` on HIP. See
  [the sampling landmine](#the-sampling-landmine--read-this-before-enabling-dspark);
  this is the `temperature > 0` half of it.
- **#33447** (4 Aug) — K3's SiTU kernels could not JIT-compile on ROCm at all
  (`situ_and_mul.cuh` included `<cuda_fp8.h>` unconditionally). Relevant because
  this repo sets `AITER_SITUV2_A8W4=1`.
- **#34234** (10 Aug) — sizes the DFLASH draft KV pool from the draft's own
  attention geometry instead of scaling the target's by the layer ratio.
- **#34580** (18 Aug) — the gfx950 MLA decode geometry. Its own section below.
- **#34881** (18 Aug) — four Kimi-K3 tool-call defects. Its own section below.
- **#32593** (15 Aug) — a Helion backend for Kimi Delta Attention, 1.1× decode and
  1.6× prefill on the KDA kernels. **NVIDIA only** — tested on GB200, no ROCm
  path — so `--linear-attn-backend helion` is not ours to try. Noted because KDA
  is 69 of K3's 93 layers and it will look relevant in a changelog.
- **#35077** (19 Aug) — mixed NVFP4/FP8 K3 checkpoints. For the CUDA NVFP4 build,
  not the `mxfp4-pack-quantized` weights this repo serves.

**Open, and worth watching — the MI355X decode headroom is queued, not landed:**

- **#33303** — a FlyDSL fused KDA decode kernel for gfx950. K3's KDA path launches
  `causal_conv1d_update → kda_packed_decode → rms_norm_gated` per layer across 69
  layers: **138 kernel launches per decode step**, fused into one dispatch.
- **#34198** — fuses the ROCm KDA decode boundary, deferring `f_b` into the same
  FlyDSL kernel.
- **#33735** — a hand-written gfx950 AMDGCN AttnResidual score kernel (DPP
  lane-shuffle + `ds_read_b128`), **+84% on the kernel and 8% end-to-end decode**;
  `_score_kernel` runs ~110× per decode step.
- **#33916 / #33838 / #33746** — AMD MoE and attn-res work.

That list is the independent corroboration of
[this repo's own correction](#k3s-decode-is-nearly-context-independent--the-earlier-claim-was-wrong):
the people optimising K3 decode on this hardware are removing **kernel launches**,
not attention work.

### The lever that did land — #33981 rewrote the DSpark verify attention

PR [#33981](https://github.com/sgl-project/sglang/pull/33981), *"[AMD] Add K3
verified mla kernel for DSpark on triton backend"*, **merged 8 Aug 2026** to
`main`. Its opening line is this repo's own finding, arrived at independently:

> Kimi-K3 DSpark throughput is slow, at high conc it is even slower than
> non-DSpark setting.

The cause is not the draft. Target-verify attention ran through
`verify_splitkv`, a kernel written for standard MHA: it parallelises **one program
per query head**. K3's MLA has `h_kv = 1`, so the single shared latent was
re-read once per head — **~12× redundant at TP8, on ~24 MLA layers per step**.
The PR adds `kernels/ops/attention/verify_mla.py` (one program per `BLOCK_H`
heads, so the latent tile is loaded once and reused; the 576-wide QK dot split
into nope/pe halves that are both powers of two) and wires it into
`triton_backend.py`.

Measured by AMD, 8×MI355X TP8, 8k in / 1k out, `--attention-backend triton` with
**the same four aiter env vars this repo sets**:

| concurrency | 2 | 8 | 32 |
|---|---|---|---|
| total tok/s vs baseline | 1.37× | 1.56× | **1.77×** |
| median ITL vs baseline | 1.46× | 1.84× | **2.42×** |
| accept len | 7.21 | 7.11 | 7.02 |

GSM8K 0.951. Note their accept length holds above 7 at 8k — consistent with the
long-context draft, and with the collapse here being the 4k checkpoint.

**This is one half of the long-context story.** The retrained draft fixes the
accept rate; #33981 attacks the ~117 ms step. It is no longer queued — it is in
the default image as of 20 Aug 2026, and re-measuring the crossover with it is
[the runbook](#the-runbook-in-order--as-of-20-aug-2026).

### The second lever — #34580 retuned the gfx950 MLA decode geometry

PR [#34580](https://github.com/sgl-project/sglang/pull/34580), *"[AMD] Optimize
KIMI-K3 with Triton MLA decode kernel by tuning the stage-1 geometry for gfx950"*,
**merged 18 Aug 2026**. Where #33981 fixes the DSpark verify path specifically,
this one speeds MLA decode itself — **with DSpark on or off**.

The author's framing is that the stage-1 constants were tuned on CDNA3 and fit
badly at the batch sizes long-context serving actually runs at:

- `BLOCK_N` 16 → **32** on the HIP path, so the MFMA tiles are used
- the KV split budget becomes a **hard ceiling** instead of a rounding target, so
  a split can no longer overshoot by a full wave
- split counts are passed to **both** stages, instead of stage 2 overwriting what
  the caller chose
- per-sequence geometry counts are preserved when deterministic inference is on

Measured by AMD: **46–73% better ITL at long context**, 2–12% ITL and 2–8% TTT at
short context, GSM8K unchanged (0.953 vs 0.955).

It is **opt-in**, and the gate is worth reading, because every clause of it is
satisfied here:

```python
def _mla_tuning_applies(has_mla: bool, head_dim: int) -> bool:
    return (_is_hip and has_mla and head_dim == 576
            and is_gfx95_supported() and envs.SGLANG_MLA_DECODE_TUNE.get())
```

HIP, MLA, `head_dim == 576`, gfx950 — and the file it patches,
`kernels/ops/attention/decode_attention.py`, is the path
`ATTENTION_BACKEND=triton` takes. Set `MLA_DECODE_TUNE=1` in `kimik3.env` to turn
it on.

**Why it is not on by default here.** Upstream ships it off because it *reorders
the fp32 accumulation* — output is not bit-identical to the untuned path. That is
a reasonable thing to accept, and an unreasonable thing to accept silently, so it
is an A/B in the runbook rather than a default. `./serve-kimik3.sh check` reports
whether the image has it at all, and `serve` warns if you set it where it would
be inert.

**This is the first upstream change aimed squarely at the number this deployment
is actually stuck on.** Not the accept rate — that was the draft, and it is fixed
— but the cost of a decode step at 100k, which is what makes `SPECULATIVE=""`
win up there at 29.83 ms TPOT.

### #34881 — Kimi K3 was losing tool calls, and two of the ways were silent

PR [#34881](https://github.com/sgl-project/sglang/pull/34881), *"Stop losing
Kimi-K3 tool calls to reasoning, constraint conflicts, and truncation"*, **merged
18 Aug 2026**, cherry-picked to `release/v0.5.18` as #35399. Upstream puts the
rate at **~190 parsing errors a day**, and says two of the four defects failed
silently — so that is a floor, not a measurement. The four:

1. native XTML/prose output was routed through the JSON-array decoder
2. `response_format` together with `tool_choice=required` silently dropped the
   tool constraint. **Behaviour change:** that combination is now a **400**
   (`auto` still warns and continues)
3. **tool calls emitted before `<|close|>think<|sep|>` were consumed as reasoning
   content.** Now the parser splits on whichever marker appears first
4. an incomplete tool section at stream end produced no output *and no log line*.
   Now it reports the truncation and releases the held-back text

It touches `function_call/kimik3_detector.py`, `parser/reasoning_parser.py`,
`serving_chat.py`, `serving_responses.py` and `protocol.py`.

**Why this matters here specifically.** Defects 3 and 4 are streaming-path
failures that present as *"the agent occasionally does nothing"* — no error, no
log, just a turn that produced no tool call. The
[Step 5b round trip](#step-5b--prove-tool-calling-round-trips) cannot see either
one, and neither can `toolcheck`: they are non-streaming, two-turn checks. The
opencode and kimicode verifications recorded here on 12 Aug are honest about what
they tested, and this is the class of bug they could not have caught. Re-run a
real session against the new image, not just `toolcheck`.

Same-day, and probably not a coincidence: `moonshotai/Kimi-K3` itself took its
first commit in three weeks on **20 Aug** — `a590ce09`, *"Update encoding_k3.py"*,
which rewrites `normalize_tool_arguments()` to return `(key, XTML type, text)`
triples and stops dropping whitespace-only reasoning. That file is Moonshot's
reference encoder; it is **not** in `tokenizer_config.json`'s `auto_map` (which
names `tokenization_kimi.TikTokenTokenizer`), so SGLang never loads it and
reimplements the same logic in `kimik3_detector.py`. Harmless in itself — but it
is why `MODEL_REVISION` now exists.

### The biggest lever we still cannot pull — AITER MLA prefill

PR [#33341](https://github.com/sgl-project/sglang/pull/33341), *"[AMD] Enable
aiter MLA for 12-head models via 12→16 zero-pad (Kimi-K3)"*, opened 3 Aug by AMD.
**Still open at the 12 Aug check, still `dirty`, and still untouched since the day
it opened** — nine days of conflicts nobody has resolved. It targets `main`,
which is where K3 lives now, so it is no longer aimed at a side branch; it is
simply stalled. It is in no image today, and `ATTENTION_BACKEND=triton` stays the
right setting.

K3 at TP8 gives each rank 12 attention heads, and AITER's MLA kernels want 16.
Both the old branch and `main` already pad for **decode**
(`_mla_decode_fwd_with_head_pad`); the gap this PR closes is the absorbed
**prefill** path, plus a non-power-of-2 head mask in `cache_ops` and a NoPE guard
for K3's `skip_rope`. That makes it a **TTFT lever, not an aggregate-throughput
one** — which is exactly the axis `bench-kimik3.sh throughput` does not measure.

Its numbers, 8×MI355X TP8, `--attention-backend aiter` vs `triton`:

| | 1024 in / 512 out, c=1 | 10240 in / 2048 out, c=16 |
|---|---|---|
| Mean TTFT | 306 → 218 ms (1.4×) | **6633 → 696 ms (9.5×)** |
| Total throughput | 250 → 294 tok/s (+17.7%) | **2613 → 6455 tok/s (2.47×)** |
| Mean TPOT | 7.36 → 6.48 ms | 25.9 → 6.97 ms |
| GSM8K | — | 0.976 |

Note how little it does at short context and how much at long: this matters for
agentic sessions that resend a large context every turn, and barely at all for
the 1024/512 shape every table in this README uses. If it merges and reaches an
image, it is worth a `check` + `parsers` + full re-bench in a second `SIF_PATH` —
and worth re-benching at 10k input, not just 1024.

#### The half of it we might already have — aiter for decode only

The decode head-pad is present in the image's own `aiter_backend.py`, so
the prefill gap need not block the decode path: SGLang takes
`--prefill-attention-backend` / `--decode-attention-backend`, and they override
the global `--attention-backend`. That is reachable today with no code change,
because `EXTRA_ENGINE_ARGS` is passed through verbatim:

```bash
# kimik3.env — UNTESTED, use a throwaway allocation
export EXTRA_ENGINE_ARGS="--decode-attention-backend aiter --prefill-attention-backend triton"
```

Probe first; if either command comes back empty, the image cannot do it:

```bash
apptainer exec "$SIF_PATH" grep -c _mla_decode_fwd_with_head_pad \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py
apptainer exec "$SIF_PATH" python -m sglang.launch_server --help 2>&1 \
  | grep -- --decode-attention-backend
```

**Expect little from it, on this repo's own measurements.** An earlier version of
this section attributed the long-context decode step to this kernel. The 9 Aug
run refuted that: decode TPOT is 26.49 ms at 1k and 29.83 ms at 100k, so at most
~11% is on the table for decode no matter which kernel serves it. #33341's
25.9 → 6.97 ms TPOT figure is measured at c=16, where contention with prefill
dominates — it is not a c=1 decode result and should not be read as one.

Worth running as a cheap experiment, not as a lever. **It has not been run.** A
wrong attention kernel produces fluent gibberish rather than a crash, so
`parsers` and `toolcheck` are mandatory before believing any number it produces.

### Weights

- **`moonshotai/Kimi-K3`'s weights have not moved since 27 Jul.** Still one MXFP4
  checkpoint, no quantization variants, no K3.1. **The repo did move on 20 Aug**,
  for the first time in three weeks: `a590ce09`, *"Update encoding_k3.py"* — the
  reference chat encoder, which SGLang does not load. No re-download needed, but
  it is why `MODEL_REVISION` exists now.
- `RadixArk/Kimi-K3-DSpark`'s weights have not moved since the 31 Jul retrain. The
  tip is `3c5bac30` (16 Aug), a README link edit, so the `56ce616a` default is
  still the right pin — see
  [the draft pinning section](#the-draft-repo-is-a-moving-target--pin-it).

### Re-running the check

```bash
# newest mi35x image tags, all flavours (the k3-* stream is retired)
curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags?page_size=100&ordering=last_updated" \
  | python3 -c "import json,sys;[print(t['name'],t['last_updated'][:10],round(t['full_size']/1e9,1)) for t in json.load(sys.stdin)['results'] if 'mi35x' in t['name']]"

# is the aiter pin still 29 Jul? (this is the one that does NOT move with the tag)
curl -s "https://raw.githubusercontent.com/sgl-project/sglang/main/docker/rocm.Dockerfile" \
  | grep -m1 AITER_COMMIT_DEFAULT

# have the weights changed under you?
for m in moonshotai/Kimi-K3 RadixArk/Kimi-K3-DSpark; do
  curl -s "https://huggingface.co/api/models/$m" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('$m', d['lastModified'][:10], d['sha'][:8])"
done

# have the queued decode PRs landed? (#33341 prefill, #33303/#34198 KDA, #33735 ASM)
for pr in 33341 33303 34198 33735; do
  curl -s "https://api.github.com/repos/sgl-project/sglang/pulls/$pr" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['number'], d['state'], d['merged_at'], '->', d['base']['ref'])"
done

# did the k3 image stream ever restart? (404 on the branch = still retired)
curl -s -o /dev/null -w '%{http_code}\n' \
  "https://api.github.com/repos/sgl-project/sglang/branches/kimi-k3"

# does a candidate image actually carry the HIP renorm kernels?
# (empty output = it does not; this is the DSpark sampling landmine)
apptainer exec "$SIF_PATH" grep -o 'top_[kp]_renorm_probs_triton' \
  /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py | sort -u
```

---

## Moving to a mainline image

**The default moved on 20 Aug 2026.** `SGLANG_IMAGE` now points at
`v0.5.17-rocm724-mi35x-20260820`; the day-0 `rocm720-mi35x-k3-20260727` tag is
kept as the **anchor** for every number in this file dated before then. Setting
it back, with the old `SIF_PATH`, is a complete rollback — nothing in the weight
cache changes.

What the new default carries that the old tag cannot:
[#33981](#the-lever-that-did-land--33981-rewrote-the-dspark-verify-attention)
(DSpark verify over MLA),
[#34580](#the-second-lever--34580-retuned-the-gfx950-mla-decode-geometry) (gfx950
MLA decode geometry, opt-in),
[#34881](#34881--kimi-k3-was-losing-tool-calls-and-two-of-the-ways-were-silent)
(tool-call losses), #33694 and both renorm bindings — which together retire
[the sampling landmine](#the-sampling-landmine--read-this-before-enabling-dspark).

**Still never migrate in place.** Build into a second `.sif` so the working one
survives, whether you are taking this default for the first time or trying
something newer:

```bash
# what's current, across all mi35x flavours:
curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags?page_size=100&ordering=last_updated" \
  | python3 -c "import json,sys;[print(t['name'],t['last_updated'][:10]) for t in json.load(sys.stdin)['results'] if 'mi35x' in t['name']]"

# in kimik3.env, keep the working sif and build a new one alongside it:
export SGLANG_IMAGE="docker://lmsysorg/sglang-rocm:v0.5.17-rocm724-mi35x-20260820"
export SIF_PATH="$MODEL_CACHE_DIR/sglang-rocm724-20260820.sif"
./serve-kimik3.sh pull && ./serve-kimik3.sh check
```

`check` now ends with a **lever report** — which of #33981, #34580 and #34881 the
image actually carries, and the `AITER_COMMIT` it was built with. That is the
answer to "is this tag new enough", and it does not depend on reading the date
right.

Then `parsers`, then `toolcheck`, then a full bench. A mainline image changes the
attention, sampling, speculative *and* tool-parsing paths at once, so nothing in
this file's tables carries over by assumption.

### The runbook, in order — as of 20 Aug 2026

Ordered so every number attributes to one change. **Clear `DSPARK_BLOCK_SIZE`
from `kimik3.env` first** if it is still `3` from the gamma test.

The first attempt at step 5, on 28 Aug, never reached the weight load: aiter's
tuned-GEMM configs are written to a `/tmp` path shared with every other user on
the node. Fixed in the script — `git pull` before you start, and see
[the troubleshooting entry](#notes--troubleshooting) for what it was.

**0 — login node.** `git pull`, then:

```bash
./serve-kimik3.sh download     # MODEL_REVISION and DSPARK_REVISION both pinned
```

A no-op against an existing cache unless a pin is overridden. If `MODEL_REVISION`
is new to your cache it fetches a snapshot, not a checkpoint: same-content files
are already in `blobs/` and get symlinked, so only what changed transfers.

**1 — build the new SIF, keeping the old one.**

```bash
SGLANG_IMAGE=docker://lmsysorg/sglang-rocm:v0.5.17-rocm724-mi35x-20260820 \
SIF_PATH=$MODEL_CACHE_DIR/sglang-rocm724-20260820.sif \
  ./serve-kimik3.sh pull
```

**2–4 — the gates, cheapest first. Stop at the first failure.**

1. `check` — the registry is non-empty and knows `KimiK3ForConditionalGeneration`;
   the image still ships prebuilt aiter `module_*.so`; and the lever report shows
   `verify_mla`, `mla_tune` and `aiter=d9e5ef7c`.
2. `parsers` — both `kimi_k3` parsers still registered. #34881 rewrote
   `kimik3_detector.py` and `reasoning_parser.py`, so this is no longer a
   formality.
3. `--help` against the old image — `--enable-gdn-replayssm-spec` is now
   reachable (`REPLAYSSM_SPEC=1` no longer dies), and
   `--cuda-graph-max-bs-decode` should still be the spelling.

**5 — the new baseline.** `SPECULATIVE=""`, `MLA_DECODE_TUNE` unset. Sweep at
1024, `longcontext` at 100k, and `BENCH_INPUT_LEN=32768 longcontext`. This is the
control for the whole image jump — three weeks of `main`, a new ROCm, a new
Python, a new torch. Compare against **26.49 ms** TPOT at 1k and **29.83 ms** at
100k.

**6 — one variable: `MLA_DECODE_TUNE=1`.** Still `SPECULATIVE=""`, restart, same
three shapes. The claim under test is 46–73% ITL at long context, and 29.83 ms is
the number to beat.

**7 — the run the migration is for.** `SPECULATIVE=dspark` with whichever
`MLA_DECODE_TUNE` setting won step 6, at 1024, 32k and 100k. #33981 moves the
DSpark side and #34580 moves both, so **the ~55–60k crossover has to be
re-located, not assumed** — it can only have moved up. If it lands past 100k, the
compaction policy [above](#the-crossover-is-around-5560k--and-that-is-the-operating-policy)
changes.

**8 — clients.** `toolcheck`, then a *real* kimicode or opencode session.
#34881's third and fourth defects only show on the streaming path; a two-turn
round trip cannot see them.

**9 — optional.** `KV_CACHE_DTYPE=fp8_e4m3` at 100k with DSpark off. Upstream's
cookbook sets it and #32938 says it hurts DSpark; one run settles it for this
shape.

**Rollback, at any point:**

```bash
export SGLANG_IMAGE="docker://lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727"
export SIF_PATH="$MODEL_CACHE_DIR/kimik3-mi355x.sif"
```

Then re-read the `SPECULATIVE` guidance in `kimik3-env.example` against step 7
rather than assuming it still holds.

---

## The node is ROCm 7.14, the image is 7.2.4 — and that is fine

Every MI355X node runs **ROCm 7.14** on bare metal; the default image ships
**7.2.4** (it shipped 7.2.0 until 20 Aug 2026, and the gap this section is about
is the same either way). They do not have to match, and since **28 Jul 2026** this
repo has handled the mismatch automatically. What follows is why it works,
because the failure it prevents looks fatal and is not.

The symptom, on an unpatched setup:

```
RuntimeError: Get GPU arch from rocminfo failed:
  Command '['/opt/rocm-7.2.0/bin/rocminfo']' returned non-zero exit status 1.
```

Note the path — that is the **container's own** 7.2 `rocminfo`, not the node's.
And it is cosmetic. `torch.cuda.device_count()` returns **8** throughout: PyTorch's
ROCm wheel carries its own HIP runtime, so HIP keeps working while the image's
standalone ROCm *tools* do not. What the container's `rocminfo` actually prints:

```
ROCk module version 6.19.14.31400000 is loaded
hsa api call failure at: .../rocminfo.cc:357
Call returned HSA_STATUS_ERROR_INVALID_ARGUMENT
```

It loads, reads the driver version, then fails an HSA call against the newer KFD.
aiter shells out to `rocminfo` purely to *name the architecture* and reports only
its exit status, which is how a cosmetic failure becomes a fatal one.

**`ROCMINFO_SHIM` (default `auto`) fixes it**, and will keep being necessary for
as long as the image ships anything older than the node. The script snapshots the *host's* working
`rocminfo` output and binds a one-line script replaying it over the container's
binary — real output for this node, nothing links against it, no host libraries
involved. `AITER_GPU_ARCHS=gfx950` looks like the obvious fix instead and **does
not work**: this build of aiter ignores `GPU_ARCHS` at runtime (tested on bun161,
28 Jul 2026).

The node reaches into the container through exactly three channels, worth knowing
if a future upgrade breaks something the shim does not cover:

1. **`--rocm` library injection** — Apptainer binds the node's ROCm libraries into
   `/.singularity.d/libs` and prepends them to `LD_LIBRARY_PATH`. Fixable with
   `ROCM_MODE=devices`.
2. **The kernel driver, via `/dev/kfd`** — a KFD ioctl ABI break is not fixable by
   any bind; it needs an image built for the node's ROCm.
3. **Inherited `*_VISIBLE_DEVICES`** — a UUID-form list that the container's older
   ROCr cannot parse stops it enumerating any GPU, which looks exactly like a dead
   driver. The script refuses to forward non-index-form values.

`gpucheck` tells you which case you are in, in about a minute:

```bash
./serve-kimik3.sh gpucheck
```

It prints both ROCm versions, dumps what the image's `rocminfo` says, tries every
passthrough mode, and separates "torch saw 8 GPUs but aiter could not name the
arch" from "the container cannot reach the GPUs" — those need opposite responses.
`ROCM_MODE=auto` does the same probing at launch and caches the answer per
node+image, so a working node pays for it once.

**If torch sees zero devices in every mode**, you need an image built on a newer
ROCm — and as of 20 Aug 2026 there still is not one. ROCm 7.2.4 is the newest
flavour sglang builds; `rocm/pytorch` publishes twelve `rocm7.14_*` base images
(16 Jul), but `docker/rocm.Dockerfile` has stages for `rocm700`, `rocm720` and
`rocm724` and nothing else, so there is no 7.14 sglang image to pull. Building one
means adding a stage and running that Dockerfile with `SGL_BRANCH`,
`GPU_ARCH=gfx950` and a `BASE_IMAGE_950_*` override, 45–90 minutes — and **Bunya
has no Docker**, so it needs an Apptainer def with fakeroot (check with RCC
first) or a machine that does.

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
./serve-kimik3.sh loadstat       why the last cold start was slow + time-to-ready history
./serve-kimik3.sh download       prefetch weights (+ DSpark draft if enabled)
./serve-kimik3.sh stop           stop the server
./serve-kimik3.sh status         server state + health check + /v1/models

./opencode-setup.sh [--host H] [--port P] [--model M] [--context N]
                    [--api-key K] [--embed-key] [--config PATH]

./kimicode-setup.sh [--host H] [--port P] [--model M] [--context N]
                    [--api-key K] [--no-key] [--config PATH]

./bench-kimik3.sh [sweep]        1024/512 tok/s at concurrency 2/8/32
./bench-kimik3.sh latency        single-stream latency only
./bench-kimik3.sh throughput     saturate at BENCH_MAX_CONCURRENCY
./bench-kimik3.sh longcontext    100k/512 at c=1 — the agentic shape
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
| `MODEL_REVISION` | `9f62e4e9…` | **Pinned by default** to the 27 Jul tip — the state every table here was measured against. A 40-hex value is a snapshot, `main` or empty tracks the branch. Resolves to a snapshot *path*, so SGLang cannot read a different one |
| `SERVED_MODEL_NAME` | `kimi-k3` | Name clients use in the `model` field |
| `SGLANG_IMAGE` | `lmsysorg/sglang-rocm:v0.5.17-rocm724-mi35x-20260820` | Mainline nightly, ROCm 7.2.4. Moved here 20 Aug 2026; `rocm720-mi35x-k3-20260727` is the anchor for older numbers ([why](#moving-to-a-mainline-image)) |
| `SIF_PATH` | `$MODEL_CACHE_DIR/kimik3-mi355x.sif` | Where the Apptainer image is stored |
| `APPTAINER_CACHEDIR` / `_TMPDIR` | *(near `$MODEL_CACHE_DIR`)* | Apptainer cache/scratch, kept off `/home` |
| `AITER_JIT_DIR` | `$MODEL_CACHE_DIR/aiter-jit` | Writable copy of aiter's `jit/`, bound over the in-image one |
| `AITER_JIT_TARGET` | *(auto-detected)* | In-image `jit/` path to bind over; set only if detection is wrong |
| `AITER_CONFIG_DIR` | `$MODEL_CACHE_DIR/aiter-configs` | Bound over aiter's hardcoded `/tmp/aiter_configs`, which is **shared between users** on the node ([why](#notes--troubleshooting)) |
| `KIMIK3_API_KEY` | *(auto-generated)* | Bearer key; saved to `$MODEL_CACHE_DIR/kimik3-api-key` |
| `PORT` | `30000` | Endpoint port on the node |
| `TP_SIZE` | `8` | Tensor parallel = **total GPU count** |
| `DP_SIZE` | `1` | dp-attention groups; must divide `TP_SIZE` |
| `CONTEXT_LEN` | *(empty)* | Empty = model max (1M), as upstream. Set `262144` if KV won't allocate |
| `MEM_FRACTION` | `0.85` | `--mem-fraction-static` (upstream's validated value) |
| `ATTENTION_BACKEND` | `triton` | `--attention-backend` |
| `MLA_DECODE_TUNE` | *(empty)* | `1` exports `SGLANG_MLA_DECODE_TUNE=1` — gfx950 MLA decode geometry, **46–73% ITL at long context** ([#34580](#the-second-lever--34580-retuned-the-gfx950-mla-decode-geometry)). Off by default because it reorders the fp32 accumulation; A/B it |
| `MODEL_DTYPE` | `bfloat16` | `--dtype` (compute dtype; weights are MXFP4) |
| `CUDA_GRAPH_MAX_BS_DECODE` | `256` | `--cuda-graph-max-bs-decode` (note: not `--cuda-graph-max-bs`) |
| `DISABLE_RADIX_CACHE` | `0` | Prefix caching ON — our deviation, measured. `1` restores upstream's setting |
| `KV_CACHE_DTYPE` | *(empty)* | Empty on purpose: upstream's cookbook now sets `fp8_e4m3`, but #32938 reports fp8 KV slowing DSpark |
| `TOOL_PARSER` | `kimi_k3` | Tool-call parser; `none` to omit |
| `REASONING_PARSER` | `kimi_k3` | Thinking parser; `none` to omit |
| `SPECULATIVE` | *(empty)* | `dspark` enables DSpark speculative decoding |
| `DSPARK_MODEL` | `RadixArk/Kimi-K3-DSpark` | Draft model for DSpark |
| `DSPARK_BLOCK_SIZE` | *(empty)* | `--speculative-dspark-block-size`. Empty infers gamma from the draft (7 today); setting it anchors against draft drift |
| `REPLAYSSM_SPEC` | `0` | `--enable-gdn-replayssm-spec` — **not** the cookbook's non-existent `--enable-linear-replayssm-spec`. Reachable since the 20 Aug image; still off because nobody has benched it ([why](#k3s-kda-state-pool--the-knobs-we-did-not-pass)) |
| `MAMBA_FULL_MEMORY_RATIO` | *(empty → 0.9)* | KDA state pool vs MLA KV pool. Follows mean request length, not a universal default |
| `MAMBA_SSM_DTYPE` | *(empty → fp32)* | `bfloat16` halves KDA state memory. Watch accept length |
| `MAMBA_RADIX_STRATEGY` | *(empty → `extra_buffer`)* | State slots per request: 5 / `extra_buffer_lazy` 4 / `no_buffer` 3 |
| `MAMBA_SKIP_DECODE_LOCK` | `0` | `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` — one fewer state slot per request (experimental) |
| `ENABLE_AITER` | `1` | Exports the four `SGLANG_*`/`AITER_*` K3 variables |
| `ROCM_MODE` | `auto` | GPU passthrough: `auto` probes, `rocm` uses `--rocm`, `devices` binds `/dev/kfd` only |
| `AITER_GPU_ARCHS` | *(empty)* | Sets `GPU_ARCHS`. Ignored at runtime by the aiter build every image ships — prefer `ROCMINFO_SHIM` |
| `ROCMINFO_SHIM` | `auto` | Replay the host's `rocminfo` output inside the container when the image's own fails |
| `SET_CPU_AFFINITY` | `0` | Keep `0` under a SLURM cgroup (see troubleshooting) |
| `READY_TIMEOUT` | `14400` | Seconds to wait for health (cold load is ~1.5 TB) |
| `DSPARK_REVISION` | `56ce616a…` | **Pinned by default**, moved 12 Aug 2026 to the long-context retrain. A 40-hex value is a snapshot, anything else a ref — `main` tracks the branch. `eb03982e…` is the 4k draft the pre-12-Aug tables were measured on; see [Speculative decoding](#the-draft-repo-is-a-moving-target--pin-it) |
| `WEIGHT_LOAD_THREADS` | `8` | Loader threads. Naming it is what stops SGLang silently going single-threaded (see [Weight loading](#weight-loading--why-a-cold-start-sawtooths)). `0` = image default |
| `LOAD_FORMAT` | *(empty)* | `--load-format`. `presharded` dumps a per-rank checkpoint so later starts skip re-quantisation |
| `PRESHARDED_PATH` | *(empty)* | Where `presharded` writes. Needs up to another 1561 GB |
| `PREFETCH_BLOCK_SIZE_MB` | *(empty)* | `SGLANG_PREFETCH_BLOCK_SIZE_MB`. Tune after the thread count, not before |
| `LAUNCH_CMD` | `sglang serve` | Escape hatch: `python3 -m sglang.launch_server` |
| `CHUNKED_PREFILL_SIZE` / `MAX_RUNNING_REQUESTS` / `SCHEDULE_POLICY` | — | Optional tuning |
| `BENCH_RANGE_RATIO` | `1.0` | Fixed-size bench requests. sglang's own default `0.0` halves them (see Performance tuning) |
| `BENCH_REPEATS` | `1` | Repeats per concurrency in a sweep. Use 3–5 before claiming two configs differ |
| `EXTRA_ENGINE_ARGS` | — | Flags appended verbatim (e.g. `--ep-size 8`) |

---

## Notes & troubleshooting

- **"apptainer not found"**: you're on a login node. Start an allocation first.
- **`apptainer pull` dies in `mksquashfs` with `proot error: ptrace(TRACEME):
  Operation not permitted`** — nothing to do with the image, the network, or
  disk space. Apptainer **1.5.0** (6 May 2026) started wrapping `mksquashfs` in a
  bundled `proot` so an unprivileged build preserves file ownership out of the
  OCI registry; proot works by `ptrace`, so it fails outright on a host that
  refuses `ptrace(PTRACE_TRACEME)` (yama `ptrace_scope=3`, a seccomp filter, some
  SELinux policies). Apptainer **1.5.3** (21 Jul 2026) downgraded this to an INFO
  message and carries on, so only the 1.5.0–1.5.2 window is affected — check with
  `apptainer --version`.

  `serve-kimik3.sh` now detects it and retries as
  `apptainer build --ignore-proot`, so a plain `./serve-kimik3.sh pull` should
  just work. `--ignore-proot` is a hidden flag that restores the pre-1.5.0
  behaviour: ownership inside the SIF is not preserved, which costs nothing here
  because the image is mounted read-only and the processes run as you. To do it
  by hand:

  ```bash
  apptainer build --ignore-proot "$SIF_PATH" "$SGLANG_IMAGE"
  ```

  A failed pull can leave a **truncated `.sif`** behind, and the "is the image
  present?" test is only `[[ -f ]]` — so if you hit this before updating, delete
  `$SIF_PATH` before retrying or you will get "Image already present" followed by
  a confusing failure much later.
- **Server exits with an unknown-architecture error**: the image predates K3
  support. Run `./serve-kimik3.sh check` — it names exactly which architectures
  the image knows. See [Moving to a mainline image](#moving-to-a-mainline-image).
- **Server exits on an argparse error about a parser name**: run
  `./serve-kimik3.sh parsers`. K3 uses `kimi_k3`, **not** `kimi_k2`.
- **Unknown-flag error on `--cuda-graph-max-bs`**: that flag was removed; the
  branch uses `--cuda-graph-max-bs-decode`. Nothing in this repo should emit the
  old name — if you added it via `EXTRA_ENGINE_ARGS`, drop it.
- **Fluent gibberish**: see Step 5. Try `ENABLE_AITER=0` to rule out the fused
  FP4 kernels, and try a different image tag. Do not ship a config that produces
  incoherent output because the throughput looked good.
- **`RuntimeError: Get GPU arch from rocminfo failed`** at startup — the image's
  7.2 `rocminfo` failing against the node's 7.14 KFD. Usually cosmetic: if
  `./serve-kimik3.sh gpucheck` shows torch seeing 8 GPUs, only `rocminfo` is
  broken and `ROCMINFO_SHIM=auto` (the default) handles it. `AITER_GPU_ARCHS=gfx950`
  looks right and is ignored at runtime. See
  [The node is ROCm 7.14, the image is 7.2.4](#the-node-is-rocm-714-the-image-is-724--and-that-is-fine).
- **KV cache OOM at startup**: set `CONTEXT_LEN=262144` first, and only then
  consider `MEM_FRACTION`. One knob at a time.
- **Cold start crawls, with bursts and dips**: SGLang went single-threaded. Run
  `./serve-kimik3.sh loadstat` — it looks for the `falling back to
  single-threaded weight loading` line. Set `WEIGHT_LOAD_THREADS=8`. See
  [Weight loading](#weight-loading--why-a-cold-start-sawtooths).
- **`unrecognized arguments: --model-loader-extra-config`**: the day-0 branch
  image predates the flag. The script probes `--help` and drops it with a
  warning rather than dying after a 1.5 TB load, so if you see this it came from
  `EXTRA_ENGINE_ARGS`. Set `WEIGHT_LOAD_THREADS=0`.
- **Host OOM during weight load**: `WEIGHT_LOAD_THREADS` is too high. The buffered
  loader holds ~`(threads + 2)` shards at ~5 GB each; check that against `--mem`.
- **kimicode rejects tool calls, or complains about a tool-call id**: this server
  returns ids like `get_gpu_temperature:0`, not OpenAI's `call_<random>` — see
  [Step 5b](#step-5b--prove-tool-calling-round-trips). The round trip is valid and
  `toolcheck` passes, but a client that validates the prefix would trip on it.
  This is the most likely way kimicode fails while every HTTP-level check passes.
- **kimicode shows the model's thinking inline, or drops it**: SGLang returns
  reasoning in `reasoning_content`, not `content` (Step 5). Kimi Code can be told
  where to look with `reasoning_key` on the `[models.*]` table.
- **`kimi` insists on `/login` even with a custom provider configured**: the OAuth
  flow is for Moonshot's managed service. Check you edited
  `~/.kimi-code/config.toml` and not `~/.kimi/config.toml` — that is the legacy
  Python CLI, and its provider type is `openai_legacy`, not `openai`. See the
  table in [Step 6b](#step-6b--connect-kimicode-optional).
- **`TypeError: 'NoneType' object is not callable`** (or `NameError:
  top_k_renorm_prob`) with DSpark, *mid-session, killing the server* — the image
  has no HIP renorm kernels ([#32569](https://github.com/sgl-project/sglang/issues/32569)).
  Triggered by a request with `top_p < 1.0` or `top_k` set; a healthy server that
  dies the moment a new client connects is this. **The default image since 20 Aug
  2026 has both kernels**, so this means `SGLANG_IMAGE` points at a pre-August
  build. Send `"top_p": 1`, move forward, or unset `SPECULATIVE`. Full mechanism in
  [The sampling landmine](#the-sampling-landmine--read-this-before-enabling-dspark).
- **`NameError: tree_speculative_sampling_target_only`** with DSpark, *killing the
  server*, on a request with `temperature > 0` — the non-greedy verify kernel does
  not exist on ROCm and an `is_hip()` branch claimed it did
  ([#33694](https://github.com/sgl-project/sglang/pull/33694), fixed 6 Aug 2026).
  Same family as the bullet above and a commoner trigger. `serve-kimik3.sh` warns
  at launch if the image has this shape.
- **Mid-response `[PAD]` storms** — the model emits literal `[PAD]` (id 163839)
  forever until `max_tokens`, usually inside the think channel, in bursts, at
  90k–237k context ([#32968](https://github.com/sgl-project/sglang/issues/32968);
  worst captured stream was 17,256 of them). Traced to NaN-contaminated logits;
  the write-side fix (#32477) is in `main` and in no K3 image.
  `SGLANG_SANITIZE_NAN_LOGITS=1` is the reported mitigation. Reported on NVIDIA —
  if you see it here, that is worth telling upstream.
- **`check` says "0 architectures registered" / "This image CANNOT serve this
  model", with a wall of `Ignore import error when loading sglang.srt.models.*:
  [Errno 2] No such file or directory: ''`** — the probe never ran; it is not a
  negative answer. That trailing `: ''` is the aiter `AITER_JIT_DIR` trap below.
  `serve` escapes it by resolving the variable to a real path, but it does that
  *after* `check` and `parsers` have already run. Fixed 12 Aug 2026 by passing a
  writable `/tmp` path to every probe, and `check` now exits 3 (inconclusive)
  rather than 1 (unsupported) when the registry comes back empty. Workaround on
  an older copy: `AITER_JIT_DIR=/tmp/aiter-jit ./serve-kimik3.sh check`.
  **Hit for real** migrating to a mainline image: the pinned build imported
  nothing that touched aiter at module scope, mainline's registry does it for
  every model, so all ~200 modules failed and the image was wrongly declared
  incapable of serving K3.
- **DSpark accept length collapses at long context** (`accept len` ≈ 1.0–1.5 on
  the decode lines) — the draft is past its trained window. The 27 Jul draft was
  trained at 4,096 tokens. `./serve-kimik3.sh download` with the current default
  `DSPARK_REVISION`; see
  [the cause](#the-cause-the-pinned-draft-was-trained-at-4096-tokens).
- **`ValueError: Unrecognized model in RadixArk/Kimi-K3-DSpark. Should have a
  'model_type' key in its config.json`** — *hit for real on 1 Aug 2026.* The
  cached draft config is truncated, or upstream rewrote the repo (it does — see
  [The draft repo is a moving target](#the-draft-repo-is-a-moving-target--pin-it)).
  This fires during argument parsing, before any load. Re-fetch with
  `rm -rf $MODEL_CACHE_DIR/hub/models--RadixArk--Kimi-K3-DSpark && ./serve-kimik3.sh download`.
  `DSPARK_REVISION` is pinned by default now, so this should only bite you if you
  set it to `main`.
- **`DSPARK_REVISION=… is not in the cache`** — the pin names a revision you have
  never fetched. `./serve-kimik3.sh download` gets it, or set `DSPARK_REVISION=main`
  to use whatever you already have. **Expected once, after 12 Aug 2026**: the
  default moved to the long-context draft `56ce616a…`, which is a different 4.5 GB
  checkpoint, so the first `git pull` needs one `download`.
- **`MODEL_REVISION=… is not in the cache`** — the weights pin names a snapshot you
  have never fetched. `./serve-kimik3.sh download` gets it, or set
  `MODEL_REVISION=main` to use whatever you already have. **This is not another
  1561 GB**: the HF cache stores one blob per content hash, so a second revision
  of the same repo is symlinks plus whatever genuinely changed — for the 20 Aug
  commit, one Python file. Expected once, after 20 Aug 2026, when `MODEL_REVISION`
  first appeared.
- **The agent occasionally does nothing** — a turn that produces no tool call, no
  error and no log line. Two of the four defects fixed by
  [#34881](#34881--kimi-k3-was-losing-tool-calls-and-two-of-the-ways-were-silent)
  look exactly like this: a tool call emitted before the reasoning marker gets
  consumed as reasoning content, and a tool section truncated at stream end is
  dropped in silence. Both are streaming-path only, which is why `toolcheck`
  passes throughout. Fixed in images from `20260820`; there is no client-side
  workaround.
- **HTTP 400 on a request that used to work**, mentioning `tool_choice` and
  `response_format` — also #34881, and deliberate. `tool_choice=required` combined
  with `response_format`/`regex`/`ebnf` is an unsatisfiable pair; it used to be
  dropped with a warning and is now rejected. `tool_choice=auto` still warns and
  continues. Drop one of the two constraints.
- **`MLA_DECODE_TUNE=1 but this image has no SGLANG_MLA_DECODE_TUNE`** — the image
  predates [#34580](#the-second-lever--34580-retuned-the-gfx950-mla-decode-geometry)
  (18 Aug 2026) and the variable is inert, so anything you measure is the untuned
  path. Not fatal, and deliberately only a warning: use a `20260820` or later
  image. `./serve-kimik3.sh check` lists which levers an image has before you
  spend an allocation finding out.
- **`This image has no --enable-gdn-replayssm-spec`** — `REPLAYSSM_SPEC=1` against
  a pre-August image, which predates sglang #32692. See
  [KDA state pool](#k3s-kda-state-pool--the-knobs-we-did-not-pass). The default
  image has the flag; on an older one, set `REPLAYSSM_SPEC=0`.
- **The server rejects `--enable-linear-replayssm-spec`** — because it does not
  exist, anywhere. It is an error in upstream's cookbook. Use `REPLAYSSM_SPEC=1`,
  which passes `--enable-gdn-replayssm-spec`. Do **not** substitute
  `--enable-linear-replayssm`: that is a different flag and is *slower* on KDA.
- **`RuntimeError: Cannot find any model weights with 'RadixArk/Kimi-K3-DSpark'`**
  — *hit for real on 1 Aug 2026.* The draft is a **separate** checkpoint, and
  `download` only fetches it when `SPECULATIVE=dspark` was set at the time. Worse,
  the scheduler builds the draft worker *after* the main model has finished
  loading, so this used to appear at the end of a full 1.5 TB load. The script now
  preflights the draft cache and refuses to start without it. Fix with
  `./serve-kimik3.sh download`, or `unset SPECULATIVE` to serve without it.
- **Crash: "CPU number N is not eligible; choose between [...]"** in
  `set_gpu_proc_affinity`: the image sets `SGLANG_SET_CPU_AFFINITY=1`, but
  SGLang pins to CPUs from the *full* node topology, which fails under a SLURM
  cgroup owning a subset of cores. The script forces `0` by default. Set
  `SET_CPU_AFFINITY=1` only with `--cpus-per-task=384` / `--exclusive`.
- **Benchmark dies in `aiter/jit/core.py` with `FileNotFoundError: [Errno 2] No
  such file or directory: ''`** — *hit for real on 9 Aug 2026.* Nothing is
  missing; an **empty but set** variable is. aiter's `get_user_jit_dir()`
  branches on `"AITER_JIT_DIR" in os.environ` rather than on whether it has a
  value, then calls `os.makedirs("")`. `kimik3-env.example` ships
  `export AITER_JIT_DIR="${AITER_JIT_DIR:-}"`, and Apptainer passes the host
  environment straight through, so merely sourcing the config poisons any import
  of aiter inside the container. `serve-kimik3.sh` resolves it to a real path and
  never sees this; `bench-kimik3.sh` now does the same and binds it.

  It only surfaces on the code path that imports aiter at all.
  `sglang.bench_serving` never does; `sglang.benchmark.serving` — the module
  upstream moved to — imports `disaggregation.utils` → quantization → aiter at
  module scope. The bench is a pure HTTP client needing no GPU and no kernels, so
  it **prefers the old path wherever it exists** and only falls back when the
  image has nothing else. Set `BENCH_MODULE` to force one. The resolved name is
  recorded in every results file.
- **Every mode dies at import with `PermissionError: [Errno 13] Permission
  denied: '/tmp/aiter_configs/bf16_tuned_gemm.csv.lock'`** — *hit for real on
  28 Aug 2026, the first serve on the mainline image.* aiter's
  `get_config_file()` globs `aiter/configs/model_configs/` for per-op tuned GEMM
  CSVs and, when it finds any, merges them with the base config and writes the
  result to a **hardcoded** `/tmp/aiter_configs/<op>.csv`, taking a `FileBaton`
  lock beside it first. There is no environment variable for that directory — it
  is a literal in `aiter/jit/core.py`. Apptainer bind-mounts the **host's**
  `/tmp` into the container, so on a shared node the first person to run a K3
  container owns `/tmp/aiter_configs` at mode 0755, and everyone after them gets
  `EACCES` on the lock file's `O_CREAT|O_EXCL`.

  Nothing is wrong with the image, and aiter did not change: `AITER_COMMIT` is
  the same `d9e5ef7c` (29 Jul) in the 27 Jul image and in `…-20260820`. The
  *import graph* changed. sglang's `moe/moe_runner/deep_gemm.py` now imports
  `sglang.kernels.ops.attention.dsv4`, whose `gemm.py` does
  `from aiter.tuned_gemm import tgemm`, and `tuned_gemm` reads
  `AITER_CONFIG_GEMM_BF16_FILE` at module scope. Three weeks of sglang reached a
  trap that had been sitting there the whole time.

  The script now binds `AITER_CONFIG_DIR` (default
  `$MODEL_CACHE_DIR/aiter-configs`) over `/tmp/aiter_configs` for every mode that
  imports sglang — `serve`, `check`, `parsers`, `loadstat`, `gpucheck` — after
  write-testing the bind, so a bind that cannot be made warns and degrades
  instead of killing the container at startup. The contents are regenerated on
  every process start, so deleting that directory is always safe.

  This is the same class as the `AITER_JIT_DIR` trap above, **and it fails the
  same misleading way in `check`**: the import error empties SGLang's model
  registry, and an empty registry reads as "this image cannot serve K3" if you
  are not looking for it. `check` now names both traps when the registry comes
  back empty.

  Two manual escapes if you are running an older copy of the script:
  `APPTAINER_BIND=$MODEL_CACHE_DIR/aiter-configs:/tmp/aiter_configs` does the
  same job from the environment; or point the per-op variable named in the
  traceback at a **single** path —
  `AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv`
  — which makes `update_config_files()` return early without touching `/tmp` at
  all. The second one costs you the `model_configs/` merge, which is where the
  per-model tuning lives, and there are ten such variables, so it is a way to
  get a run out, not a fix.
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

## Sources

- [sglang#32541 — day-0 Kimi K3 support](https://github.com/sgl-project/sglang/pull/32541) — branch, NVIDIA and AMD image tags
- [sglang#32548 — [Kimi-K3][AMD] Day 0 and Performance Tracking](https://github.com/sgl-project/sglang/issues/32548) — **the MI355X recipe and every perf number in this README**
- [sglang#32569](https://github.com/sgl-project/sglang/issues/32569) — the open DSPARK crash · [#33694](https://github.com/sgl-project/sglang/pull/33694) — its `temperature > 0` sibling
- [sglang#33981](https://github.com/sgl-project/sglang/pull/33981) — **the DSpark verify MLA kernel**, merged 8 Aug 2026 · [#34580](https://github.com/sgl-project/sglang/pull/34580) — **the gfx950 MLA decode geometry**, merged 18 Aug 2026, `SGLANG_MLA_DECODE_TUNE`
- [sglang#34881](https://github.com/sgl-project/sglang/pull/34881) — **four Kimi-K3 tool-call defects**, merged 18 Aug 2026 · [#32568](https://github.com/sgl-project/sglang/pull/32568) / [#34985](https://github.com/sgl-project/sglang/pull/34985) — AMD's own K3 MI35x nightly accuracy and perf jobs
- [sglang#32968](https://github.com/sgl-project/sglang/issues/32968) — long-context `[PAD]` storms and NaN logits · [#32855](https://github.com/sgl-project/sglang/issues/32855) — where upstream says to re-pull the retrained draft
- [SGLang Kimi-K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) — the MI35x cell, the KDA knob surface and the `--mamba-full-memory-ratio` calculator. **Its `--enable-linear-replayssm-spec` does not exist** — see [KDA state pool](#k3s-kda-state-pool--the-knobs-we-did-not-pass)
- [sglang#32692](https://github.com/sgl-project/sglang/pull/32692) — ReplaySSM with `extra_buffer`, landed 31 Jul 2026, after the day-0 image
- [sglang `docker/rocm.Dockerfile`](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile) — the `AITER_COMMIT_DEFAULT` pin and the three ROCm stages (7.0, 7.2, 7.2.4 — **no 7.14**)
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) · [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark)
- [LMSYS: Kimi K3 day-0 support](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support) · [Kimi K3 tech blog](https://www.kimi.com/blog/kimi-k3) — KDA, Attention Residuals, 16/896 sparsity
- [vLLM K3 preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) · [vllm#50000](https://github.com/vllm-project/vllm/pull/50000) — the NVIDIA-only alternative, for when a ROCm build appears
- [vllm#36337](https://github.com/vllm-project/vllm/issues/36337) — Kimi MXFP4 gibberish on gfx950/ROCm 7.2
- [UQ-RCC Bunya docs](https://github.com/UQ-RCC/hpc-docs) · [ROCm/aiter](https://github.com/ROCm/aiter) · [opencode providers](https://opencode.ai/docs/providers/)
- [Kimi Code CLI: providers and models](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/providers.html) · [config files](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/config-files.html) · [environment variables](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/env-vars.html) — **the provider schema Step 6b is built from**
- [GLM-5.2 sibling repo](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_GLM5.2) — where the Bunya-specific knowledge here came from

---

## License

The scripts in this repo are provided under the MIT License (see `LICENSE`).
The container image
(`lmsysorg/sglang-rocm`), the SGLang engine, and the model weights
(`moonshotai/Kimi-K3`) are covered by their own separate licences.
