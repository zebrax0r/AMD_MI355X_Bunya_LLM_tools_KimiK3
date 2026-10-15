"""CPU-only characterization checks for the documentation-first cleanup.

No model, endpoint, Slurm job or third-party Python package is required.
The retained scripts have known limitations: preserving a behaviour does not
mean endorsing it. See docs/maintenance.md before changing these guardrails.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "agentstress-406abdb"
BASELINE = json.loads((ROOT / "tests" / "baseline.json").read_text())
ARCHIVE = ROOT / "docs" / "history" / "README-2026-10-13.md"


def blob_sha(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def shell_statements(path):
    # This template deliberately consists of single-line exports and comments.
    result = []
    for line in path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words:
            result.append(words)
    return result


def resolved_environment(path, overrides, home):
    env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": home, "USER": "cleanup-test"}
    env.update(overrides)
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", 'set -eu; source "$1"; env -0', "_", str(path)],
        cwd=ROOT, env=env, check=True, capture_output=True, timeout=10,
    )
    return dict(item.split(b"=", 1) for item in result.stdout.split(b"\0") if item)


def run_stress(path, outcome="clean", args=None):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, sorted(request.header_items()), request.data, timeout))
        if outcome == "transport":
            raise OSError("synthetic connection failure")
        body = json.loads(request.data)
        text = body["messages"][-1]["content"]
        codes = re.findall(r"NOTE: the deployment ticket for this section is ([0-9]{6}-[a-z]+)\.", text)
        if not codes:
            raise AssertionError("test fixture did not produce a needle")
        reply = codes[-1] + "; this section configures the deployment."
        reply = {"pad": "[PAD]", "missing": "No ticket found.", "empty": ""}.get(outcome, reply)
        events = [
            {"choices": [{"delta": {"reasoning_content": "Checking the supplied ticket."}}]},
            {"choices": [{"delta": {"content": reply}}]},
            {"choices": [], "usage": {"completion_tokens": 24}},
        ]
        data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        return io.BytesIO(data + b"data: [DONE]\n\n")

    argv = [str(path), "http://unused.invalid", "test-api-key", "kimi-k3"]
    argv += ["60000", "20000"] if args is None else args
    output = io.StringIO()
    with patch.object(sys, "argv", argv), patch("urllib.request.urlopen", fake_urlopen):
        with contextlib.redirect_stdout(output):
            try:
                runpy.run_path(str(path), run_name="__main__")
            except SystemExit as exc:
                return exc.code, requests, output.getvalue()
    raise AssertionError("stress tool did not exit")


def heading_ids(path):
    # GitHub-style slugs for the headings linked by these docs, including duplicates.
    seen, result, fenced = {}, set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```") or line.startswith("~~~"):
            fenced = not fenced
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if fenced or not match:
            continue
        text = match.group(1).lower()
        text = re.sub(r"<[^>]*>", "", text)
        slug = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
        number = seen.get(slug, 0)
        seen[slug] = number + 1
        result.add(slug + ("-" + str(number) if number else ""))
    return result


class PreservationTests(unittest.TestCase):
    def test_deployment_files_are_byte_identical(self):
        for name, (sha, _) in BASELINE["preserved"].items():
            with self.subTest(path=name):
                self.assertEqual(blob_sha(ROOT / name), sha)

    def test_deployment_executable_bits_are_preserved(self):
        for name, (_, mode) in BASELINE["preserved"].items():
            with self.subTest(path=name):
                executable = bool((ROOT / name).stat().st_mode & stat.S_IXUSR)
                self.assertEqual(executable, mode == "100755")
        self.assertTrue((ROOT / "agentstress.py").stat().st_mode & stat.S_IXUSR)

    def test_frozen_corpus_and_original_stress_tool(self):
        for name, sha in BASELINE["fixtures"].items():
            with self.subTest(path=name):
                self.assertEqual(blob_sha(FIXTURE / name), sha)

    def test_historical_readme_is_complete_and_unchanged(self):
        self.assertEqual(blob_sha(ARCHIVE), BASELINE["fixtures"]["README.md"])
        self.assertEqual(ARCHIVE.read_bytes(), (FIXTURE / "README.md").read_bytes())

    def test_config_assignment_statements_and_order_unchanged(self):
        original = shell_statements(FIXTURE / "kimik3-env.example")
        current = shell_statements(ROOT / "kimik3-env.example")
        self.assertEqual(current, original)
        self.assertEqual(len(current), 63)
        self.assertTrue(all(len(words) == 2 and words[0] == "export" for words in current))

    @unittest.skipUnless(shutil.which("bash"), "Bash is required for config checks")
    def test_config_resolves_identically_for_defaults_empty_and_overrides(self):
        statements = shell_statements(ROOT / "kimik3-env.example")
        names = [words[1].split("=", 1)[0] for words in statements]
        cases = [
            {},
            dict.fromkeys(names, ""),
            {name: "explicit-" + name for name in names},
            {"MODEL_CACHE_DIR": "/scratch/path with spaces & symbols", "SPECULATIVE": "",
             "MODEL_REVISION": "main", "DSPARK_REVISION": "main", "HF_TOKEN": "fake-test-token",
             "PREFILL_ATTENTION_BACKEND": "", "BENCH_INPUT_LEN": "32768"},
        ]
        with tempfile.TemporaryDirectory() as home:
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    original = resolved_environment(FIXTURE / "kimik3-env.example", overrides, home)
                    current = resolved_environment(ROOT / "kimik3-env.example", overrides, home)
                    self.assertEqual(current, original)

    def test_stress_only_redirects_the_corpus_path(self):
        old = (FIXTURE / "agentstress.py").read_bytes()
        self.assertEqual(old.count(b"repo = here\n"), 1)
        expected = old.replace(
            b"repo = here\n",
            b'repo = os.path.join(here, "tests", "fixtures", "agentstress-406abdb")\n',
        )
        self.assertEqual((ROOT / "agentstress.py").read_bytes(), expected)

    def test_stress_requests_are_identical_over_growing_conversation(self):
        old = run_stress(FIXTURE / "agentstress.py")
        new = run_stress(ROOT / "agentstress.py")
        self.assertEqual(old[:2], new[:2])
        self.assertEqual(new[0], 0)
        self.assertEqual(len(new[1]), 3)

    def test_stress_default_cli_still_produces_eight_identical_turns(self):
        old = run_stress(FIXTURE / "agentstress.py", args=[])
        new = run_stress(ROOT / "agentstress.py", args=[])
        self.assertEqual(old[:2], new[:2])
        self.assertEqual(len(new[1]), 8)

    def test_legacy_outcomes_are_unchanged_not_a_quality_endorsement(self):
        for outcome, status in (("pad", 1), ("transport", 2), ("missing", 0), ("empty", 0)):
            with self.subTest(outcome=outcome):
                old = run_stress(FIXTURE / "agentstress.py", outcome)
                new = run_stress(ROOT / "agentstress.py", outcome)
                self.assertEqual(old[:2], new[:2])
                self.assertEqual(new[0], status)

    @unittest.skipUnless(shutil.which("bash"), "Bash is required for shell syntax checks")
    def test_shell_syntax(self):
        names = list(ROOT.glob("*.sh")) + list(ROOT.glob("*.sbatch")) + [ROOT / "kimik3-env.example"]
        for path in names:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, timeout=10)

    def test_python_syntax_without_executing_probes(self):
        for path in (ROOT / "agentstress.py", FIXTURE / "agentstress.py", Path(__file__)):
            with self.subTest(path=str(path)):
                ast.parse(path.read_text(), filename=str(path))

    def test_entry_points_remain_short(self):
        self.assertLessEqual(len((ROOT / "README.md").read_text().splitlines()), 100)
        self.assertLessEqual(len((ROOT / "kimik3-env.example").read_text().splitlines()), 120)

    def test_current_documentation_relative_links_and_anchors(self):
        docs = [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))
        for path in docs:
            if path == ARCHIVE:
                continue  # Historical text is intentionally immutable.
            for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^\s)]+)\)", path.read_text()):
                with self.subTest(source=str(path.relative_to(ROOT)), target=target):
                    parts = urlsplit(target)
                    if parts.scheme or parts.netloc:
                        continue
                    linked = (path.parent / unquote(parts.path)).resolve() if parts.path else path
                    self.assertTrue(linked.is_relative_to(ROOT), "link escapes checkout")
                    self.assertTrue(linked.exists(), "missing link target")
                    if parts.fragment and linked.suffix == ".md":
                        self.assertIn(unquote(parts.fragment), heading_ids(linked))


if __name__ == "__main__":
    unittest.main()
