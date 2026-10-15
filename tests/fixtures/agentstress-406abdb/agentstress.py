#!/usr/bin/env python3
"""Agentic-shape stress test: one conversation that grows like a kimicode
session, with real text, streaming, temperature > 0, no cache-friendly
randomness. Each turn appends ~TURN_TOKENS of new material containing a fresh
needle, asks for it back, and checks the reply for corruption.

Usage: ./agentstress.py <base_url> <api_key> <model> [max_ctx_tokens] [turn_tokens]
Exit 0 = all turns clean, 1 = corruption seen, 2 = request failed (server down?).

Run it after ANY engine change, on top of the benches: bench-kimik3.sh uses
random prompts with no shared prefix, so it never exercises prefix-cached
prefill (where the 13 Oct aiter fault lived), and its random tokens make DSpark
look far better than it is on real text.
"""
import json, os, random, sys, time, urllib.request

base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]
max_ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 150000
turn_tok = int(sys.argv[5]) if len(sys.argv) > 5 else 20000

here = os.path.dirname(os.path.abspath(__file__))
repo = here
corpus = []
for f in ("README.md", "serve-kimik3.sh", "kimik3-env.example", "bench-kimik3.sh"):
    with open(os.path.join(repo, f), errors="replace") as fh:
        corpus += [p for p in fh.read().split("\n\n") if p.strip()]
rng = random.Random(11)


def chunk(n_tokens, code):
    # ~3.5 chars/token for this mix of prose and shell
    out, size = [], 0
    while size < n_tokens * 3.5:
        p = rng.choice(corpus)
        out.append(p)
        size += len(p)
    out.insert(len(out) // 3, f"NOTE: the deployment ticket for this section is {code}.")
    return "\n\n".join(out)


def stream(messages):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 1500,
                       "temperature": 0.6, "top_p": 0.95, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    content, reasoning, t0, ttft, ntok = [], [], time.time(), None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                ntok = d["usage"].get("completion_tokens") or ntok
            if not d.get("choices"):
                continue
            delta = d["choices"][0].get("delta", {})
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.time() - t0
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
    return "".join(content), "".join(reasoning), ttft or 0, time.time() - t0, ntok


def corrupt(text):
    if "[PAD]" in text or "!!!!!!!!!!" in text:
        return "pad/bang run"
    if len(text) > 200:
        tail = text[-200:]
        if len(set(tail)) < 6:
            return "degenerate repetition"
    return None


messages = [{"role": "system", "content": "You are a coding assistant reviewing a deployment repository."}]
ctx, turn, ok, rates = 0, 0, True, []
while ctx < max_ctx:
    turn += 1
    code = f"{rng.randint(100000, 999999)}-{rng.choice(['amber', 'cobalt', 'violet', 'saffron'])}"
    messages.append({"role": "user", "content": chunk(turn_tok, code) +
                     "\n\nBriefly: what is the deployment ticket for this section, and what is one "
                     "thing this section configures? Answer in under 80 words."})
    ctx += turn_tok
    try:
        content, reasoning, ttft, total, ntok = stream(messages)
        rates.append((ctx, ntok / max(total - ttft, 1e-3)))
    except Exception as e:
        print(f"turn {turn:2d} ~{ctx//1000:3d}k  REQUEST FAILED: {e}")
        sys.exit(2)
    bad = corrupt(reasoning) or corrupt(content)
    found = code in content
    status = "CORRUPT: " + bad if bad else ("ok" if found else "needle missed")
    ok &= not bad
    print(f"turn {turn:2d} ~{ctx//1000:3d}k  ttft {ttft:5.1f}s total {total:5.1f}s  {ntok:4d} tok {rates[-1][1]:5.1f} tok/s  {status}  "
          f"reply={content.strip()[:40]!r}", flush=True)
    if bad:
        print("  reasoning tail:", repr(reasoning[-160:]))
        print("  content tail:  ", repr(content[-160:]))
        sys.exit(1)
    messages.append({"role": "assistant", "content": content})
for lo, hi in ((0, 60000), (60000, 120000), (120000, 10**9)):
    r = [v for c, v in rates if lo < c <= hi]
    if r:
        print(f"decode tok/s, context {lo//1000}-{min(hi, ctx)//1000}k: mean {sum(r)/len(r):.1f}")
sys.exit(0 if ok else 1)
