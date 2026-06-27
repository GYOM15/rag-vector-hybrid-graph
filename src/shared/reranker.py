"""Reranking par cross-encoder : étape 2 d'une récupération en deux temps.

Le récupérateur (vecteur / hybride / graphe) sort un **top-N large** ; le
cross-encoder, qui lit la paire (requête, document) **ensemble**, les re-note
finement et on garde le **top-k**. Plus précis qu'un bi-encodeur (qui encode la
requête et le document *séparément*), mais trop lent pour tout le corpus — d'où
les deux étapes : on ne le passe que sur les candidats déjà filtrés.
"""

from functools import lru_cache


@lru_cache(maxsize=2)
def _model(name: str):
    """Charge (et met en cache) le cross-encoder ; import paresseux."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


class CrossEncoderReranker:
    """Re-note des candidats avec un cross-encoder et renvoie le top-k réordonné."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def rerank(
        self, query: str, candidates: list[dict], top_k: int,
        mode: str = "replace", rrf_k: int = 60,
    ) -> list[dict]:
        """Réordonne `candidates` (dicts avec une clé "text") et renvoie le top-k.

        - `replace` : on suit le cross-encoder seul (il **remplace** le classement de base).
        - `fusion`  : RRF entre le rang de base (position d'entrée) et celui du
          cross-encoder — chaque classement « vote », ce qui **protège** un
          récupérateur déjà fort de se faire tirer vers le bas.
        """
        if not candidates:
            return []
        scores = _model(self.model_name).predict([(query, c["text"]) for c in candidates])
        ce_order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        if mode == "fusion":
            ce_rank = [0] * len(candidates)
            for rank, i in enumerate(ce_order):
                ce_rank[i] = rank
            order = sorted(
                range(len(candidates)),
                key=lambda i: 1.0 / (rrf_k + i) + 1.0 / (rrf_k + ce_rank[i]),
                reverse=True,
            )
        else:
            order = ce_order
        return [candidates[i] for i in order[:top_k]]
