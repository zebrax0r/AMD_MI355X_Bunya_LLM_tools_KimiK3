# Maintenance and change boundary

[Documentation home](../README.md) | [Validation](validation.md)

## First cleanup: preserve the working deployment

Baseline: `406abdb123d0015c388bc90a57cf4d460d6de2ee`, 13 October 2026.

This change replaces the oversized entry-point README with task-focused guides,
archives the complete original README without editing it, and shortens the
configuration example without changing its executable assignment statements.
It freezes the original stress-test inputs and adds CPU-only regression checks.

The following existing files remain byte-for-byte unchanged, including their
Git executable modes:

- `serve-kimik3.sh`, `serve-kimik3.sbatch` and `bench-kimik3.sh`.
- Both client setup scripts and both client templates.
- `.gitignore` and `LICENSE`.

The only existing Python change redirects the stress-test corpus to immutable
fixtures. Its request construction, random seed, CLI arguments, output and exit
semantics are otherwise retained. Tests compare it against the saved original.

No launch flags, image/model/draft pins, GPU workarounds, ports, cache paths,
Slurm resource requests, API-key handling, or process-lifecycle code are changed.
The configuration example keeps all original exports, including benchmark defaults.
Existing user configuration files are never rewritten by this change.

## Adoption and rollback

Review the pull request and its CPU checks before merging. Keep the current
working checkout and use a separate checkout to inspect the cleanup. Do not
switch branches under a running job, overwrite the existing `kimik3.env`, or
rebuild/download anything solely because the docs changed.

When ready, use the same trusted local configuration and the same launch command
at a normal maintenance boundary. Keep the existing deployment available. The
cleanup introduces no on-disk cache/config schema migration, so rollback is to
the prior checkout with its original local configuration and image.

The CPU tests establish source/config/workload preservation, not successful GPU
execution. The [Bunya checks](validation.md#bunya-qualification) remain manual.

## Follow-up changes, not smuggled into this cleanup

| Area | Required targeted work |
|---|---|
| Benchmark failures | Preserve the benchmark process exit code independently of `grep`, and verify request success counts. |
| Strict stress validation | Explicitly handle empty/truncated streams, missing needles, actual token counts and transport/quality failures; retain a labelled legacy comparison mode. |
| Process ownership | Record hostname/job/process identity; stop only the intended process group/job rather than matching broad names. |
| Batch path | Validate `SLURM_SUBMIT_DIR` or an explicit repository path on Bunya before changing the wrapper. |
| Config consistency | Share resolution across commands; clarify empty/unset/disabled states and `KIMIK3_ENV` handling without breaking existing local files. |
| Reproducibility | Tie benchmark metadata to the running server's sanitised manifest, not only the benchmark shell's environment. |
| Launcher modularity | Extract independently tested configuration, argument building and probes while keeping public entry points stable. |

Each of these changes can alter behaviour. Implement them as small, separately
reviewable changes with targeted tests and a Bunya qualification where relevant.
Do not combine a serving refactor, a new image and a performance retune.
