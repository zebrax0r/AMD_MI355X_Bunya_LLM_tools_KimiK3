# Validation

[Documentation home](../README.md) | [Maintenance boundary](maintenance.md)

## CPU-only cleanup guardrails

From the repository root, with Python 3 and Bash available:

```bash
python3 -m unittest discover -s tests -v
```

These checks require no third-party Python packages, GPU, model download,
Slurm allocation or inference endpoint. The GitHub workflow runs the same
command with read-only repository permissions.

The tests check preserved deployment-file blob hashes and executable bits,
all configuration assignment statements and resolved environments, frozen
corpus hashes, and the stress test's single allowed source edit. Mocked SSE
responses compare request URLs, headers and JSON bodies against the original
stress tool, including its default eight-turn run. Syntax and current-doc links
are checked too.

These are deliberately conservative characterization tests for the first cleanup.
They preserve existing behaviour; they do not certify it as bug-free. Do not
update a baseline hash merely to make a behaviour-changing refactor pass.
Replace a guardrail only alongside targeted tests and a documented validation.

## Frozen stress-test workload

`agentstress.py` retains its existing command and request logic. Its only code
change is reading the original four corpus files from
`tests/fixtures/agentstress-406abdb/` instead of the mutable repository root.
Those files are exact Git blobs from `406abdb`. Their order, whitespace,
paragraph boundaries and the original random seed are retained. The original
stress script is stored alongside them as a test reference, not as a second
supported command.

Keep the fixture directory with the checkout. Do not rewrite it when changing
documentation. Identical input construction does not imply deterministic model
replies: sampling and the running engine remain outside the CPU test.

## Bunya qualification

This is a manual check against the intended allocation; the CPU workflow never
runs it. A documentation cleanup does not require cancelling a healthy job.
Use an appropriate maintenance/test allocation, never an unreviewed CI action.

Record the repository commit, hostname, Slurm job ID, actual SIF path/checksum,
model and draft revisions, effective non-secret overrides and client versions.
Retain the working checkout/configuration and do not rebuild the image or clear
caches to test this cleanup. For a future image experiment use a separate SIF
and isolated writable caches so rollback does not inherit experimental state.

Run the existing `check`, `gpucheck` and `parsers` probes, launch via the intended
interactive or batch route, and verify health, coherent chat and `toolcheck`.
Then perform a real streaming client read/edit cycle and a growing conversation.

For the existing soak test, on a machine that can reach the endpoint:

```bash
source ./kimik3.env
export KIMIK3_API_KEY="$(cat "$MODEL_CACHE_DIR/kimik3-api-key")"
./agentstress.py "http://127.0.0.1:${PORT:-30000}" "$KIMIK3_API_KEY" \
  "${SERVED_MODEL_NAME:-kimi-k3}" 160000 20000
```

Use localhost only on the serving node or through the intended tunnel. The API
key is a positional argument in the retained CLI; avoid exposing process output
or recorded commands. A safer key-input interface is a separate change.

**Inspect every turn.** The legacy tool exits zero on a missed needle or some
empty responses if its corruption heuristic does not fire. It also estimates
context and does not test tool execution. Do not treat exit zero alone as a
quality gate. This cleanup explicitly preserves, rather than silently changes,
those outcomes. Future strict validation should have separately reviewed semantics.

## Performance comparisons

The existing commands remain:

```bash
./bench-kimik3.sh sweep
./bench-kimik3.sh longcontext
```

Confirm the printed input/output lengths and concurrency, including after
sourcing a config. Inspect full saved output and successful-request counts:
the current benchmark's display pipeline can conceal an execution failure.
Use repetitions for comparisons and distinguish cold-start storage state from
warm-cache results. Record output-token throughput separately from total-token
throughput. Synthetic random prompts are not evidence of cached-prefix or
agentic correctness.

No new GPU performance result is produced by the cleanup or its CPU workflow.
