"""Pure retrieval (IR) metrics: recall@k, nDCG@k, MRR.

Dependency-free (stdlib only) -> testable in isolation. `ranked` is the list
of document ids sorted by descending score; `relevant` is a dict
{doc_id: gain} (the relevance judgments / qrels, gain > 0).
"""

import math


def recall_at_k(ranked: list[str], relevant: dict, k: int) -> float:
    """Fraction of relevant documents present in the top-k."""
    if not relevant:
        return 0.0
    topk = set(ranked[:k])
    return sum(1 for doc in relevant if doc in topk) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: dict) -> float:
    """1 / rank of the 1st relevant document (0 if none)."""
    for i, doc in enumerate(ranked, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict, k: int) -> float:
    """nDCG@k with graded gains (DCG of the ranking / ideal DCG)."""
    dcg = sum(relevant.get(doc, 0.0) / math.log2(i + 1)
              for i, doc in enumerate(ranked[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0
