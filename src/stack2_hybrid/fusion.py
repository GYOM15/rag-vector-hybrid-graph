"""Reciprocal Rank Fusion (RRF) — fusion de classements (pur Python).

Isolé dans son propre module (aucune dépendance lourde) pour rester testable
sans charger FAISS ni rank_bm25.
"""


def reciprocal_rank_fusion(ranked_lists: list[list[int]], rrf_k: int = 60) -> dict[int, float]:
    """Fusionne plusieurs classements (listes d'ids ordonnés) en scores RRF.

    score(id) = somme sur les classements de 1 / (rrf_k + rang), le rang
    commençant à 0. Un id bien classé dans plusieurs listes obtient un score
    plus élevé. `rrf_k` (60 par défaut) amortit le poids des premiers rangs.
    """
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return scores
