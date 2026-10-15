# Troubleshooting

[Documentation home](../README.md) | [Runbook](runbook.md)

Start with the serving hostname, Slurm job ID, exact SIF path and launch log.
Do not change several engine settings at once. Keep secrets out of shared
reports, including any key printed by a client helper.

| Symptom | First check/action |
|---|---|
| `apptainer` not found | Confirm the shell is on the allocated compute node. |
| Job stays queued | Inspect the requested partition/account/QoS and available GPUs; the recorded recipe uses `gpu_rocm`. |
| Image/model compatibility unclear | Run `./serve-kimik3.sh check` before downloading or loading weights. An inconclusive import probe is not a negative architecture result. |
| Container sees no usable GPUs | Run `./serve-kimik3.sh gpucheck`; retain the existing `ROCM_MODE=auto` and shim behaviour unless diagnosing a specific failure. |
| Startup stalls or fails around compilation | Check scratch/home quota, writable JIT/cache paths and the launcher preflight output. |
| KV allocation fails | The recorded first diagnostic is `CONTEXT_LEN=262144`, changing no other setting at the same time. Restart and validate the result. |
| Unknown parser/flag | Check the actual image and `parsers` output; do not copy flags from an unrelated historical image. |
| Health succeeds but output is empty or incoherent | Inspect real replies and both content/reasoning channels. Health is not a correctness test. |
| `[PAD]` bursts or cached-prefix GPU faults | Check for an aiter-prefill override, including `EXTRA_ENGINE_ARGS` and old config files. Return to the documented Triton baseline after stopping the affected job. |
| Tools fail only in a client | Run `toolcheck`, then test that client's streaming read/edit cycle; verify client configuration path and advertised model/context. |
| Bench prints `Done` but no useful result | Inspect its full output and successful-request count. Its current output pipeline can mask failures. |
| `status` contradicts the running job | Confirm you are on the serving node; the command checks localhost and local PIDs. |

## Batch script cannot find the launcher

The existing batch wrapper derives its directory from `BASH_SOURCE[0]`. A Slurm
spooled copy can make that directory differ from the submission directory.
The wrapper is deliberately unchanged in the behaviour-preserving cleanup.

Use the already working interactive allocation route while diagnosing this.
Do not move the launcher into a Slurm spool directory. A future targeted fix
should use an explicit, validated repository path and be tested on Bunya before
replacing the batch workflow. Keep the original working submission route until
that check is complete.

## Stopping and shared paths

Use `scancel <jobid>` for a specific remote job. The local `stop` path uses a
PID file plus broad SGLang process-name matching, and `status` uses localhost.
Neither interprets a shared PID file as a remote host/job identity. Avoid
multiple concurrent servers sharing the same state files and avoid local
`stop` when another matching server of yours is running on that host.

No process-lifecycle or path migration is hidden in this documentation change.
These require separate tests and are tracked in [maintenance](maintenance.md).

## Detailed historical investigations

The original explanations of proot, JIT mounts, GPU visibility, sampling,
loading and client incompatibilities remain in
[the archived troubleshooting section](history/README-2026-10-13.md#notes--troubleshooting).
Treat image/date-specific diagnoses there as historical evidence, not universal
current remedies. The [compatibility page](compatibility.md) governs the baseline.
