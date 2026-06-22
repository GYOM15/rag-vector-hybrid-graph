"""Métriques de récupération (IR) pures : recall@k, nDCG@k, MRR.

Sans dépendance (stdlib seule) → testables en isolation. `ranked` est la liste
des ids de documents triés par score décroissant ; `relevant` est un dict
{doc_id: gain} (les jugements de pertinence / qrels, gain > 0).
"""

import math


def recall_at_k(ranked: list[str], relevant: dict, k: int) -> float:
    """Fraction des documents pertinents présents dans le top-k."""
    if not relevant:
        return 0.0
    topk = set(ranked[:k])
    return sum(1 for doc in relevant if doc in topk) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: dict) -> float:
    """1 / rang du 1er document pertinent (0 si aucun)."""
    for i, doc in enumerate(ranked, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict, k: int) -> float:
    """nDCG@k avec gains gradués (DCG du classement / DCG idéal)."""
    dcg = sum(relevant.get(doc, 0.0) / math.log2(i + 1)
              for i, doc in enumerate(ranked[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0
