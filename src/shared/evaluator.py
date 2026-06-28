"""RAGAS evaluation and the stack benchmark loop.

The heavy imports (ragas, datasets) are done **on demand**: importing this
module stays lightweight even without the `[eval]` extra installed. It is only
required when RAGAS is actually called.
"""


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall + per_question).

    OpenAI judge by default -> requires OPENAI_API_KEY. Raises ValueError if the lists differ in length.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    n = len(questions)
    if not (n == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError(
            f"All input lists must have the same length. Got "
            f"questions={len(questions)}, answers={len(answers)}, "
            f"contexts={len(contexts)}, ground_truths={len(ground_truths)}."
        )

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(eval_dataset, metrics=metrics)

    summary = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }
    if hasattr(result, "to_pandas"):
        summary["per_question"] = result.to_pandas().to_dict(orient="records")
    else:
        summary["per_question"] = []
    return summary


_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _mean_quality(per_question: list[dict], indices: list[int]) -> dict:
    """Average of the RAGAS metrics over a subset of questions (by index)."""
    summary = {}
    for metric in _RAGAS_METRICS:
        values = [
            float(per_question[i][metric])
            for i in indices
            if isinstance(per_question[i].get(metric), (int, float))
            and per_question[i][metric] == per_question[i][metric]  # discards NaN
        ]
        if values:
            summary[metric] = round(sum(values) / len(values), 4)
    return summary


def evaluate_stacks(
    stacks: dict,
    questions: list[str],
    ground_truths: list[str],
    k: int = 5,
    types: list[str] | None = None,
) -> dict[str, dict]:
    """Evaluate each stack (generation + latencies + RAGAS), overall and by `types` if provided.

    Returns {stack_name: metrics}; if RAGAS fails (missing key...), only the latencies.
    """
    n = len(questions)
    results: dict[str, dict] = {}

    for name, rag in stacks.items():
        answers, contexts, latencies = [], [], []
        retrieval = generation = total = 0.0
        for question in questions:
            r = rag.query(question, k=k)
            answers.append(r["answer"])
            contexts.append([c["text"] for c in r["contexts"]])
            latencies.append(r["latency_ms"])
            retrieval += r["retrieval_ms"]
            generation += r["generation_ms"]
            total += r["latency_ms"]

        metrics: dict = {}
        per_question: list[dict] = []
        try:
            full = evaluate_rag(questions, answers, contexts, ground_truths)
            per_question = full.pop("per_question", []) or []
            metrics = full
        except Exception:
            pass  # RAGAS optional: without a judge/key, we keep just the latencies

        metrics["avg_retrieval_ms"] = round(retrieval / n, 2)
        metrics["avg_generation_ms"] = round(generation / n, 2)
        metrics["avg_latency_ms"] = round(total / n, 2)

        if types:
            valid = per_question if len(per_question) == n else []
            by_type: dict[str, dict] = {}
            for t in dict.fromkeys(types):  # unique categories, order preserved
                idx = [i for i, tt in enumerate(types) if tt == t]
                entry = {
                    "n": len(idx),
                    "avg_latency_ms": round(sum(latencies[i] for i in idx) / len(idx), 2),
                }
                if valid:
                    entry.update(_mean_quality(valid, idx))
                by_type[t] = entry
            metrics["by_type"] = by_type

        results[name] = metrics

    return results
