"""Throughput / latency benchmark for an OpenAI-compatible LLM endpoint (e.g. vLLM).

The serving-side companion to the retrieval performance benchmark: it sweeps the
request concurrency and reports requests/s, tokens/s and latency p50/p95/p99.
Backend-agnostic — it talks plain HTTP to any `/chat/completions` endpoint
(vLLM, Ollama's OpenAI shim, OpenAI itself).

    python -m eval.serving_bench --base-url http://localhost:8000/v1 --model <name>
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONCURRENCY = (1, 2, 4, 8, 16)


def build_prompts(n: int) -> list[str]:
    """Return n distinct prompts so the server cannot trivially cache responses."""
    return [f"In two sentences, explain idea number {i} and why it matters." for i in range(n)]


def complete(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int) -> tuple[float, int]:
    """Send one chat completion. Return (latency_seconds, completion_tokens)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read())
    latency = time.perf_counter() - start
    return latency, int(body.get("usage", {}).get("completion_tokens", 0))


def run_at_concurrency(call, prompts: list[str], workers: int) -> dict:
    """Run every prompt through `call` with `workers` threads; summarise the level."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(call, prompts))
    wall = time.perf_counter() - start

    latencies_ms = np.array([lat for lat, _ in results]) * 1000
    tokens = sum(tok for _, tok in results)
    return {
        "workers": workers,
        "req_per_s": round(len(prompts) / wall, 1),
        "tokens_per_s": round(tokens / wall, 1),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 1),
    }


def run(base_url: str, api_key: str, model: str, n_prompts: int, max_tokens: int, output: Path) -> dict:
    prompts = build_prompts(n_prompts)

    def call(prompt: str) -> tuple[float, int]:
        return complete(base_url, api_key, model, prompt, max_tokens)

    call(prompts[0])  # warm up (model load, connection)

    sweep = [run_at_concurrency(call, prompts, w) for w in CONCURRENCY]

    payload = {"config": {"base_url": base_url, "model": model,
                          "n_prompts": n_prompts, "max_tokens": max_tokens},
               "sweep": sweep}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nServing benchmark — {model} @ {base_url}\n")
    print(f"  {'threads':>7} {'req/s':>8} {'tok/s':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    for row in sweep:
        print(f"  {row['workers']:7d} {row['req_per_s']:8.1f} {row['tokens_per_s']:8.1f} "
              f"{row['p50_ms']:7.0f}ms {row['p95_ms']:7.0f} {row['p99_ms']:7.0f}")
    print(f"\n✅ Wrote {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Throughput/latency of an OpenAI-compatible LLM endpoint.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "serving_results.json")
    args = ap.parse_args()
    run(args.base_url, args.api_key, args.model, args.n_prompts, args.max_tokens, args.output)


if __name__ == "__main__":
    main()
