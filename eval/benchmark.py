"""Benchmark of the three RAG architectures (CLI script).

Builds the pipelines, runs each architecture on the question set, computes the
RAGAS metrics + average latencies, and writes the result as JSON
(`eval/results.json` by default). The Streamlit application then reads this file
(or reruns this benchmark via its "Benchmark" tab).

The RAGAS judge's key comes from `OPENAI_API_KEY` and its endpoint from
`OPENAI_BASE_URL` (default: OpenAI); both are read here and handed to the judge
explicitly, like its model names (`--judge-model`, `--judge-embeddings`).

Examples:
    python -m eval.benchmark
    python -m eval.benchmark --articles 100 --k 5 --questions 10
    # Ollama as the judge, through its OpenAI-compatible endpoint (a smoke test):
    OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 python -m eval.benchmark \
        --questions 2 --judge-model qwen2.5:1.5b --judge-embeddings nomic-embed-text
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")  # loads OPENAI_API_KEY (RAGAS judge) and others from .env

from eval.provenance import finish_metadata, run_metadata  # noqa: E402
from pipeline import build_stacks  # noqa: E402
from shared.llm import active_config  # noqa: E402


def _ground_truths(data: list[dict]) -> list[str]:
    return [d.get("ground_truth") or d.get("answer") or "" for d in data]


def judge_from_env(model: str, embedding_model: str):
    """The RAGAS judge configured by the environment (key, optional endpoint)."""
    from shared.evaluator import JudgeConfig

    return JudgeConfig(api_key=os.getenv("OPENAI_API_KEY", ""), model=model,
                       embedding_model=embedding_model,
                       base_url=os.getenv("OPENAI_BASE_URL") or None)


def run(n_articles: int, k: int, max_questions: int | None, output: Path,
        judge=None) -> dict:
    """Builds the stacks, evaluates them (RAGAS judged by `judge`, a `JudgeConfig`), and
    writes the results to `output`."""
    from shared.evaluator import evaluate_stacks

    provenance = run_metadata()  # the code that runs is the code at the start
    data = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    if max_questions:
        data = data[:max_questions]
    questions = [d["question"] for d in data]
    ground_truths = _ground_truths(data)
    types = [d.get("type", "?") for d in data]

    print(f"Building the stacks ({n_articles} articles)...")
    stacks = build_stacks(n_articles=n_articles)
    print(f"Generation + evaluation on {len(questions)} questions...")
    results = evaluate_stacks(stacks, questions, ground_truths, k=k, types=types, judge=judge)

    payload = {
        "config": {
            "n_articles": n_articles,
            "k": k,
            "n_questions": len(questions),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "llm": active_config(),
            # which judge scored the answers (never the endpoint or the key)
            "judge": ({"model": judge.model, "embeddings": judge.embedding_model}
                      if judge else None),
            "provenance": finish_metadata(provenance),
        },
        "stacks": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Results written to {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG benchmark (RAGAS + latencies).")
    parser.add_argument("--articles", type=int, default=100, help="number of articles in the corpus")
    parser.add_argument("--k", type=int, default=5, help="number of retrieved chunks")
    parser.add_argument("--questions", type=int, default=0, help="limit the number of questions (0 = all)")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.json")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="RAGAS judge chat model")
    parser.add_argument("--judge-embeddings", default="text-embedding-3-small",
                        help="RAGAS judge embedding model (answer relevancy)")
    args = parser.parse_args()
    judge = judge_from_env(args.judge_model, args.judge_embeddings)
    run(args.articles, args.k, args.questions or None, args.output, judge=judge)


if __name__ == "__main__":
    main()
