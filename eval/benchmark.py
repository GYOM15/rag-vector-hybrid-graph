"""Benchmark of the three RAG architectures (CLI script).

Builds the pipelines, runs each architecture on the question set, computes the
RAGAS metrics + average latencies, and writes the result as JSON
(`eval/results.json` by default). The Streamlit application then reads this file
(or reruns this benchmark via its "Benchmark" tab).

Examples:
    python -m eval.benchmark
    python -m eval.benchmark --articles 100 --k 5 --questions 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")  # loads OPENAI_API_KEY (RAGAS judge) and others from .env

from pipeline import build_stacks  # noqa: E402


def _ground_truths(data: list[dict]) -> list[str]:
    return [d.get("ground_truth") or d.get("answer") or "" for d in data]


def run(n_articles: int, k: int, max_questions: int | None, output: Path) -> dict:
    """Builds the stacks, evaluates them, and writes the results to `output`."""
    from shared.evaluator import evaluate_stacks

    data = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    if max_questions:
        data = data[:max_questions]
    questions = [d["question"] for d in data]
    ground_truths = _ground_truths(data)
    types = [d.get("type", "?") for d in data]

    print(f"Building the stacks ({n_articles} articles)...")
    stacks = build_stacks(n_articles=n_articles)
    print(f"Generation + evaluation on {len(questions)} questions...")
    results = evaluate_stacks(stacks, questions, ground_truths, k=k, types=types)

    payload = {
        "config": {
            "n_articles": n_articles,
            "k": k,
            "n_questions": len(questions),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
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
    args = parser.parse_args()
    run(args.articles, args.k, args.questions or None, args.output)


if __name__ == "__main__":
    main()
