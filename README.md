# Kimi K3 on Bunya MI355X

Serve Kimi K3 with SGLang in Apptainer on one Bunya MI355X node, and connect
opencode or Kimi Code to its OpenAI-compatible endpoint.

**Existing deployment? Keep your `kimik3.env`, image, weights and caches.**
This documentation cleanup does not require a restart, new download, config
migration or different launch command. See [the change boundary](docs/maintenance.md).

## Start here

| Task | Guide |
|---|---|
| Install, allocate, launch, verify and stop | [Operating runbook](docs/runbook.md) |
| Connect opencode or Kimi Code | [Client setup](docs/clients.md) |
| Understand a setting or override | [Configuration reference](docs/configuration.md) |
| Check the recommended configuration and its limits | [Compatibility and validation status](docs/compatibility.md) |
| Diagnose a failed launch or request | [Troubleshooting](docs/troubleshooting.md) |
| Run CPU checks, benchmarks or a soak test | [Validation](docs/validation.md) |
| Read measurements, incident analysis and upstream history | [Historical evidence](docs/history/README.md) |

## Recommended baseline

These are the repository's selected settings at commit `406abdb` (13 October
2026), preserved by this cleanup. They are not a new hardware qualification.

| Setting | Baseline |
|---|---|
| Image | `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261011` |
| Parallelism | TP8, DP1 |
| Attention | Triton; no per-phase overrides |
| MLA decode tuning | `MLA_DECODE_TUNE=1` |
| Prefix cache | Enabled (`DISABLE_RADIX_CACHE=0`) |
| Speculative decoding | Off (`SPECULATIVE=`) |
| Model and draft revisions | Existing pinned revisions, unchanged |

**Do not enable aiter prefill in this baseline.** The recorded 13 October
cached-prefix regression supersedes the previous day's recommendation.
DSpark and DCP8 are not the normal long-session operating path. Read
[compatibility](docs/compatibility.md) before experimenting.

## Before starting

The existing Bunya recipe requires access to `gpu_rocm`, the appropriate
account/QoS, eight MI355X GPUs, Apptainer on a compute node, and approximately
1.7 TB of available scratch for the baseline download and image. Verify your
actual quota and allocation permissions; do not infer them from this README.

For a new installation, follow [the runbook](docs/runbook.md). For an existing
installation, continue using your current top-level commands:

```bash
./serve-kimik3.sh serve --detach
./serve-kimik3.sh status
./serve-kimik3.sh toolcheck
./bench-kimik3.sh sweep
```

Run these against the intended allocation on its serving node. `stop` and
`status` are local operations, not remote Slurm job management.

## Repository layout

The launch, batch, benchmark and client scripts remain at their original paths.
`docs/` separates current instructions from historical evidence. `tests/`
contains CPU-only regression checks and the frozen stress-test corpus.

```bash
python3 -m unittest discover -s tests -v
```

The CPU checks do not load a model, contact an inference endpoint or submit a
Slurm job. Hardware qualification remains a separate step in
[validation](docs/validation.md).
