# Compatibility and validation status

[Documentation home](../README.md) | [Validation procedure](validation.md)

This page describes the repository's selected baseline at commit `406abdb`
(13 October 2026). The cleanup preserves that selection; it does not claim a
new benchmark, certify later upstream releases, or upgrade a local config.

## Recommended baseline

| Component | Selected value |
|---|---|
| Model | `moonshotai/Kimi-K3` |
| Model revision | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| Image | `docker://lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261011` |
| Parallelism | `TP_SIZE=8`, `DP_SIZE=1` |
| Attention | `ATTENTION_BACKEND=triton`; empty phase overrides |
| Decode tuning | `MLA_DECODE_TUNE=1` |
| Prefix cache | `DISABLE_RADIX_CACHE=0` |
| Speculative decoding | Empty/off |
| Memory fraction and compute dtype | `0.85`, `bfloat16` |
| Context override | Empty: no new launcher cap |
| CPU affinity | `SET_CPU_AFFINITY=0` |

## Limits and experimental configurations

**aiter prefill: do not use with this baseline.** The repository records
cached-prefix corruption and GPU faults on 13 October. Cold random prompts
had not exposed the problem. Keep both phase overrides empty for the normal
Triton path. Evidence: [13 October regression](history/README-2026-10-13.md#13-oct-what-the-benches-missed).

**DSpark: optional experiment, not the long-session default.** The repository's
real-text measurements favoured it only at shorter contexts. Preserve the draft
pin and remeasure the intended workload before enabling it. No universal
context crossover or performance guarantee is asserted here.

**DCP8/fp8 KV: not a qualified alternative runbook.** The recorded experiment
used aiter prefill and needs rechecking after the regression. A capacity result
does not establish correctness of cached-prefix agent traffic.

**Context limits are different from tested coverage.** The model advertises a
1,048,576-token window and the empty launcher cap leaves the model/engine limit
in place. This is not proof of a successful 1M-token agent session. The archived
regression report discusses smaller, approximate-context runs, including a
clean Triton run to approximately 240k. Do not relabel that as full-window
validation; the stress tool estimates context rather than tokenizing it exactly.

**Older images remain historical reproduction targets.** Their measurements,
sampling problems and compatibility workarounds are preserved in the archive,
but their instructions are not the recommended baseline. No existing pin,
image path, cache or workaround has been removed from the launcher.

## Source of truth

Executable defaults remain in the existing scripts and template assignments.
An existing `kimik3.env` or exported override can select something else. The
[configuration reference](configuration.md) explains this distinction.

A changed image, draft, backend or numerical setting needs the
[hardware qualification procedure](validation.md#bunya-qualification).
