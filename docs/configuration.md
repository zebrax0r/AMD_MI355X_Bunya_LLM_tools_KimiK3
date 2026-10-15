# Configuration reference

[Documentation home](../README.md) | [Compatibility](compatibility.md)

## Existing users

Keep your current `kimik3.env`. The shortened example has the same 63 export
statements, in the same order, as the baseline; only comments and blank lines
were edited. It is a new-install template, not a config migration tool.

Normally only `MODEL_CACHE_DIR` and `HF_TOKEN` or `HF_TOKEN_FILE` need local
values. Keep the file private and outside version control. A configured token
file must also have appropriate permissions. Do not print credentials for review.

## Resolution rules retained by this cleanup

The launcher and benchmark source `KIMIK3_ENV` when set, otherwise `kimik3.env`
next to the script. Their internal fallbacks then resolve unset/empty values.
The client helpers instead look for `kimik3.env` beside themselves and expose
explicit command-line overrides. No common config loader has been introduced.

The example uses `${VAR:-default}` for most assignments: both unset and empty
values select that default. In particular, empty `MODEL_REVISION` and
`DSPARK_REVISION` select their pins; use literal `main` only for a deliberate
unpinning experiment. The older prose saying empty tracks main was inaccurate.

`SPECULATIVE`, `PREFILL_ATTENTION_BACKEND` and `DECODE_ATTENTION_BACKEND` use
`${VAR-}`; an explicit empty value is retained. Non-empty exported values
survive the example's assignments. A locally edited unconditional assignment
can override an environment value; there is no universal precedence guarantee
for arbitrary shell code in a custom config.

Sourcing the file loads values into the calling shell; changing the file does
not reconfigure a running server. Restart-level engine changes need explicit
validation. The benchmark's named modes also have existing shape-resolution
logic: confirm their printed lengths and concurrency before interpreting results.

## Template defaults

These are the literal template defaults, not a report of a running server's
configuration and not always the final resolved filesystem paths. Empty optional
settings may omit an engine flag or trigger a launcher-derived path.

| Variable | Template default | Explicit empty value |
|---|---|---|
| `MODEL_CACHE_DIR` | (empty) | Preserved |
| `HF_TOKEN` | (empty) | Preserved |
| `HF_TOKEN_FILE` | (empty) | Preserved |
| `MODEL_ID` | `moonshotai/Kimi-K3` | Uses default |
| `MODEL_REVISION` | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | Uses default |
| `SERVED_MODEL_NAME` | `kimi-k3` | Uses default |
| `SGLANG_IMAGE` | `docker://lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261011` | Uses default |
| `SIF_PATH` | (empty) | Preserved |
| `APPTAINER_CACHEDIR` | (empty) | Preserved |
| `APPTAINER_TMPDIR` | (empty) | Preserved |
| `SGLANG_CACHE_DIR` | (empty) | Preserved |
| `AITER_JIT_DIR` | (empty) | Preserved |
| `AITER_JIT_TARGET` | (empty) | Preserved |
| `AITER_CONFIG_DIR` | (empty) | Preserved |
| `KIMIK3_API_KEY` | (empty) | Preserved |
| `PORT` | `30000` | Uses default |
| `TP_SIZE` | `8` | Uses default |
| `DP_SIZE` | `1` | Uses default |
| `CONTEXT_LEN` | (empty) | Preserved |
| `MEM_FRACTION` | `0.85` | Uses default |
| `ATTENTION_BACKEND` | `triton` | Uses default |
| `PREFILL_ATTENTION_BACKEND` | (empty) | Preserved |
| `DECODE_ATTENTION_BACKEND` | (empty) | Preserved |
| `MLA_DECODE_TUNE` | `1` | Uses default |
| `MODEL_DTYPE` | `bfloat16` | Uses default |
| `CUDA_GRAPH_MAX_BS_DECODE` | `256` | Uses default |
| `DISABLE_RADIX_CACHE` | `0` | Uses default |
| `KV_CACHE_DTYPE` | (empty) | Preserved |
| `TOOL_PARSER` | `kimi_k3` | Uses default |
| `REASONING_PARSER` | `kimi_k3` | Uses default |
| `SPECULATIVE` | (empty) | Preserved |
| `DSPARK_MODEL` | `RadixArk/Kimi-K3-DSpark` | Uses default |
| `DSPARK_BLOCK_SIZE` | (empty) | Preserved |
| `REPLAYSSM_SPEC` | `0` | Uses default |
| `MAMBA_FULL_MEMORY_RATIO` | (empty) | Preserved |
| `MAMBA_SSM_DTYPE` | (empty) | Preserved |
| `MAMBA_RADIX_STRATEGY` | (empty) | Preserved |
| `MAMBA_SKIP_DECODE_LOCK` | `0` | Uses default |
| `HSA_NO_SCRATCH_RECLAIM` | (empty) | Preserved |
| `KDA_FUSED_DECODE` | `0` | Uses default |
| `ROCM_MODE` | `auto` | Uses default |
| `AITER_GPU_ARCHS` | (empty) | Preserved |
| `ROCMINFO_SHIM` | `auto` | Uses default |
| `ENABLE_AITER` | `1` | Uses default |
| `FLYDSL_FORCE` | `1` | Uses default |
| `SET_CPU_AFFINITY` | `0` | Uses default |
| `READY_TIMEOUT` | `14400` | Uses default |
| `WEIGHT_LOAD_THREADS` | `8` | Uses default |
| `LOAD_FORMAT` | (empty) | Preserved |
| `PRESHARDED_PATH` | (empty) | Preserved |
| `PREFETCH_BLOCK_SIZE_MB` | (empty) | Preserved |
| `DSPARK_REVISION` | `56ce616ad7486f0e96cbb51ef23ed5a1bce1d92d` | Uses default |
| `LAUNCH_CMD` | `sglang serve` | Uses default |
| `CHUNKED_PREFILL_SIZE` | (empty) | Preserved |
| `MAX_RUNNING_REQUESTS` | (empty) | Preserved |
| `SCHEDULE_POLICY` | (empty) | Preserved |
| `EXTRA_ENGINE_ARGS` | (empty) | Preserved |
| `BENCH_INPUT_LEN` | `1024` | Uses default |
| `BENCH_OUTPUT_LEN` | `512` | Uses default |
| `BENCH_NUM_PROMPTS` | `200` | Uses default |
| `BENCH_CONCURRENCY` | `2 8 32` | Uses default |
| `BENCH_MAX_CONCURRENCY` | `32` | Uses default |
| `BENCH_EXTRA_ARGS` | (empty) | Preserved |

## Derived paths and additional controls

The existing launcher derives the SIF, key, log and PID paths under
`MODEL_CACHE_DIR` when their overrides are empty. It also resolves writable
Apptainer, SGLang and aiter cache paths. Keep those paths unchanged for an
existing deployment; an image label in a config is not proof that an existing
SIF was rebuilt from that image.

`PID_FILE`, `LOG_FILE` and `LOADTIMES_FILE` are additional existing runtime
path overrides. `BENCH_RANGE_RATIO` (default `1.0`), `BENCH_REPEATS` (default
`1`) and `BENCH_MODULE` are additional existing benchmark controls not exported
by the original template. No new settings are added in this cleanup.

The benchmark has no authoritative query of the running server's complete
launch configuration. Its environment-derived labels must be checked against
the server's actual launch record before comparing runs.

## Restrictions to read before tuning

Keep Triton attention, empty per-phase overrides and DSpark off for the normal
long-session baseline. Do not re-enable aiter prefill from an old local config
or through `EXTRA_ENGINE_ARGS`. The latter remains an escape hatch, not a
validated configuration schema. Preserve `SET_CPU_AFFINITY=0` for the existing
Slurm-cgroup recipe.

The repository's recorded first diagnostic for KV allocation failure is a
smaller `CONTEXT_LEN`, without simultaneously retuning memory fraction. Keep
`CHUNKED_PREFILL_SIZE` at the existing image default unless investigating a
specific issue; the historical October warning advised staying below roughly
32k. Numerical/cache/speculative changes require a fresh quality check, not
just a throughput result.

See [compatibility](compatibility.md) and [validation](validation.md). The
[original annotated template](../tests/fixtures/agentstress-406abdb/kimik3-env.example)
and [original reference](history/README-2026-10-13.md#configuration-reference)
preserve the detailed rationale and historical image-specific caveats.
