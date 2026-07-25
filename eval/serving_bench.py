"""Throughput / latency benchmark for an OpenAI-compatible LLM endpoint (e.g. vLLM).

The serving-side companion to the retrieval performance benchmark: it sweeps the
request concurrency and reports requests/s, tokens/s, total latency, and — via
streaming — TTFT (time-to-first-token) and TPOT (time-per-output-token).
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
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)


def build_prompts(n: int) -> list[str]:
    """Return n distinct prompts so the server cannot trivially cache responses."""
    return [f"In two sentences, explain idea number {i} and why it matters." for i in range(n)]


def complete(base_url: str, api_key: str, model: str, prompt: str,
             max_tokens: int) -> tuple[float, int, float, float]:
    """Stream one chat completion. Return (total_s, completion_tokens, ttft_s, tpot_s).

    Streaming lets us time tokens as they arrive:
      - TTFT (time-to-first-token): request sent -> first token; dominated by prefill.
      - TPOT (time-per-output-token): mean gap between subsequent tokens; the decode speed.
    A blocking request could only see the total; streaming separates the two.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,                              # ask the server to send tokens as they come
        "stream_options": {"include_usage": True},   # + a final chunk with the exact token count
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    start = time.perf_counter()
    ttft = 0.0
    prev = start
    gaps: list[float] = []
    tokens = 0
    with urllib.request.urlopen(request) as response:
        for raw in response:  # server-sent events: one "data: {json}" line per token chunk
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):                    # final chunk: exact completion_tokens
                tokens = chunk["usage"].get("completion_tokens", tokens)
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                if ttft == 0.0:
                    ttft = now - start                # first token has arrived
                else:
                    gaps.append(now - prev)           # gap since the previous token
                prev = now
    total = time.perf_counter() - start
    tpot = sum(gaps) / len(gaps) if gaps else 0.0
    if tokens == 0:                                   # fallback if the server omitted usage
        tokens = len(gaps) + (1 if ttft else 0)
    return total, tokens, ttft, tpot


def run_at_concurrency(call, prompts: list[str], workers: int) -> dict:
    """Run every prompt through `call` with `workers` threads; summarise the level."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(call, prompts))
    wall = time.perf_counter() - start

    latencies_ms = np.array([r[0] for r in results]) * 1000
    tokens = sum(r[1] for r in results)
    ttft_ms = np.array([r[2] for r in results]) * 1000
    tpot_ms = np.array([r[3] for r in results if r[3] > 0]) * 1000
    return {
        "workers": workers,
        "req_per_s": round(len(prompts) / wall, 1),
        "tokens_per_s": round(tokens / wall, 1),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 1),
        "ttft_p50_ms": round(float(np.percentile(ttft_ms, 50)), 1),
        "ttft_p95_ms": round(float(np.percentile(ttft_ms, 95)), 1),
        "tpot_ms": round(float(tpot_ms.mean()), 1) if len(tpot_ms) else 0.0,
    }


def run(base_url: str, api_key: str, model: str, n_prompts: int, max_tokens: int, output: Path) -> dict:
    prompts = build_prompts(n_prompts)

    def call(prompt: str) -> tuple[float, int, float, float]:
        return complete(base_url, api_key, model, prompt, max_tokens)

    call(prompts[0])  # warm up (model load, connection)

    sweep = [run_at_concurrency(call, prompts, w) for w in CONCURRENCY]

    payload = {"config": {"base_url": base_url, "model": model,
                          "n_prompts": n_prompts, "max_tokens": max_tokens},
               "sweep": sweep}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nServing benchmark — {model} @ {base_url}\n")
    print(f"  {'threads':>7} {'req/s':>7} {'tok/s':>7} {'lat p50':>8} {'lat p95':>8} "
          f"{'TTFT p50':>9} {'TTFT p95':>9} {'TPOT':>7}")
    for row in sweep:
        print(f"  {row['workers']:7d} {row['req_per_s']:7.1f} {row['tokens_per_s']:7.1f} "
              f"{row['p50_ms']:7.0f}m {row['p95_ms']:7.0f}m "
              f"{row['ttft_p50_ms']:8.0f}m {row['ttft_p95_ms']:8.0f}m {row['tpot_ms']:6.1f}m")
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
