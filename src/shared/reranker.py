"""Cross-encoder reranking: step 2 of a two-stage retrieval.

The retriever (vector / hybrid / graph) outputs a **large top-N**; the
cross-encoder, which reads the (query, document) pair **together**, re-scores
them finely and we keep the **top-k**. More accurate than a bi-encoder (which encodes the
query and the document *separately*), but too slow for the whole corpus -- hence
the two stages: we only run it on the already-filtered candidates.
"""

from functools import lru_cache


@lru_cache(maxsize=2)
def _model(name: str):
    """Load (and cache) the cross-encoder; lazy import."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


class CrossEncoderReranker:
    """Re-score candidates with a cross-encoder and return the reordered top-k."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def rerank(
        self, query: str, candidates: list[dict], top_k: int,
        mode: str = "replace", rrf_k: int = 60,
    ) -> list[dict]:
        """Reorder `candidates` (dicts with a "text" key) and return the top-k.

        - `replace`: we follow the cross-encoder alone (it **replaces** the base ranking).
        - `fusion` : RRF between the base rank (input position) and the
          cross-encoder rank -- each ranking "votes", which **protects** an
          already-strong retriever from being dragged down.
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


class RerankedRetriever:
    """Decorator: add a reranking stage to any retriever.

    Exposes the **same** `search(query, k)`: retrieves an enlarged top-N via the
    inner retriever, then the cross-encoder reorders and we return the top-k.
    Transparent for the RAG and the application -- we wrap the retriever, nothing
    else changes (single responsibility). Disabled by default in the pipeline.
    """

    def __init__(self, inner, reranker: CrossEncoderReranker,
                 mode: str = "replace", candidates: int = 30):
        self.inner = inner
        self.reranker = reranker
        self.mode = mode
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[dict]:
        results = self.inner.search(query, k=max(self.candidates, k))
        return self.reranker.rerank(query, results, top_k=k, mode=self.mode)
