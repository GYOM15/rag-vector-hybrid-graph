"""Reciprocal Rank Fusion (RRF) — ranking fusion (pure Python).

Isolated in its own module (no heavy dependencies) so it stays testable
without loading FAISS or rank_bm25.
"""


def reciprocal_rank_fusion(ranked_lists: list[list[int]], rrf_k: int = 60) -> dict[int, float]:
    """Fuses several rankings (lists of ordered ids) into RRF scores.

    score(id) = sum over the rankings of 1 / (rrf_k + rank), with the rank
    starting at 0. An id ranked high in several lists gets a higher score.
    `rrf_k` (60 by default) dampens the weight of the top ranks.
    """
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return scores
