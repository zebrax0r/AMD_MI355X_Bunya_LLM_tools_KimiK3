# Historical evidence - not an operating runbook

[Documentation home](../../README.md) | [Current compatibility](../compatibility.md)

The [complete original README](README-2026-10-13.md) is preserved byte-for-byte
from commit `406abdb123d0015c388bc90a57cf4d460d6de2ee`. It contains all original
measurements, upstream links, incident investigations and superseded advice.
Nothing was selectively deleted or retrospectively rewritten.

**Statements such as "current", "not yet done" and "this image" in that file
belong to their original dated sections.** Do not follow a historical command
as a current recommendation. In particular, the 13 October cached-prefix failure
supersedes the 12 October aiter-prefill recommendation. Use the current
[runbook](../runbook.md) and [compatibility page](../compatibility.md).

| Evidence | Original section |
|---|---|
| 13 October agentic regression and correction | [What the benches missed](README-2026-10-13.md#13-oct-what-the-benches-missed) |
| 12 October sweep and its limitations | [The sweep](README-2026-10-13.md#12-oct-2026-the-sweep) |
| Earlier speculative-decoding investigations | [DSpark](README-2026-10-13.md#speculative-decoding-with-dspark) |
| Earlier performance comparisons | [Performance tuning](README-2026-10-13.md#performance-tuning) |
| Earlier operational failure diagnoses | [Troubleshooting](README-2026-10-13.md#notes--troubleshooting) |
| Original long configuration reference | [Configuration](README-2026-10-13.md#configuration-reference) |

The original annotated configuration example and other stress corpus inputs are
preserved as data under `tests/fixtures/agentstress-406abdb/`. They are not new
installation templates. The fixture filenames deliberately match the original
input list; do not move or edit them as part of a documentation tidy-up.

A later archival reorganisation can split investigations into dated records,
but must preserve provenance and supersession notices. The immutable fixture
copies must remain untouched to keep the benchmark workload stable.
