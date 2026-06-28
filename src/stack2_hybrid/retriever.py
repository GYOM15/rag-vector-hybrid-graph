"""In-memory hybrid retrieval: vector (FAISS) + lexical (BM25).

The two rankings are fused by Reciprocal Rank Fusion (RRF). Reuses the FAISS
index from stack 1 for the vector part and builds an in-memory BM25 index
over the same chunks — no external server.
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from shared.embeddings import EmbeddingModel
from shared.vector_index import FaissIndexer

from .fusion import reciprocal_rank_fusion
from .tokenizer import tokenize


class HybridRetriever:
    """Dense (FAISS) + lexical (BM25) search, fused by RRF."""

    def __init__(self, indexer: FaissIndexer, embedding_model: EmbeddingModel, candidates: int = 20):
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.candidates = candidates
        self._bm25 = BM25Okapi([tokenize(text) for text in indexer.chunks])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Returns the k most relevant chunks (score = combined RRF score)."""
        if self.indexer.size == 0:
            return []
        n = min(self.candidates, self.indexer.size)
        rankings = [self._vector_ranking(query, n), self._bm25_ranking(query, n)]
        fused = reciprocal_rank_fusion(rankings)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self._build_result(idx, score) for idx, score in top]

    def _vector_ranking(self, query: str, n: int) -> list[int]:
        """Ids of the n nearest chunks by vector similarity."""
        query_emb = np.array([self.embedding_model.encode_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)
        _, indices = self.indexer.index.search(query_emb, n)
        return [int(i) for i in indices[0] if i != -1]

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        """Ids of the n top-scoring chunks by BM25 (lexical search)."""
        scores = self._bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(scores)[::-1][:n]]

    def _build_result(self, idx: int, score: float) -> dict:
        return {
            "text": self.indexer.chunks[idx],
            "metadata": self.indexer.metadata[idx],
            "score": float(score),
        }
