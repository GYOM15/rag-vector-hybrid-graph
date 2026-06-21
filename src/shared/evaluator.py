"""Évaluation RAGAS et boucle de benchmark des stacks.

Les imports lourds (ragas, datasets) sont faits **à la demande** : importer ce
module reste léger même sans l'extra `[eval]` installé. C'est seulement quand on
appelle réellement RAGAS qu'il est requis.
"""


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Métriques RAGAS (faithfulness, answer_relevancy, context_precision, context_recall + per_question).

    Juge OpenAI par défaut → nécessite OPENAI_API_KEY. Lève ValueError si les listes diffèrent en longueur.
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
    """Moyenne des métriques RAGAS sur un sous-ensemble de questions (par index)."""
    summary = {}
    for metric in _RAGAS_METRICS:
        values = [
            float(per_question[i][metric])
            for i in indices
            if isinstance(per_question[i].get(metric), (int, float))
            and per_question[i][metric] == per_question[i][metric]  # écarte les NaN
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
    """Évalue chaque stack (génération + latences + RAGAS), global et par `types` si fourni.

    Renvoie {nom_stack: metrics} ; si RAGAS échoue (clé absente…), seules les latences.
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
            pass  # RAGAS optionnel : sans juge/clé, on garde juste les latences

        metrics["avg_retrieval_ms"] = round(retrieval / n, 2)
        metrics["avg_generation_ms"] = round(generation / n, 2)
        metrics["avg_latency_ms"] = round(total / n, 2)

        if types:
            valid = per_question if len(per_question) == n else []
            by_type: dict[str, dict] = {}
            for t in dict.fromkeys(types):  # catégories uniques, ordre conservé
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
