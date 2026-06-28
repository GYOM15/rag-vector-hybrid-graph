"""Dense vector search with FAISS.

Encodes the query with the same embedding model used at indexing time, then
queries the FAISS index for the nearest neighbors by cosine similarity.
"""

import numpy as np
import faiss

from shared.embeddings import EmbeddingModel
from shared.vector_index import FaissIndexer


class VectorRetriever:
    """Dense retriever backed by a FAISS index."""

    def __init__(self, indexer: FaissIndexer, embedding_model: EmbeddingModel):
        self.indexer = indexer
        self.embedding_model = embedding_model

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Returns the k nearest chunks: list of {text, metadata, score} (cosine)."""
        if self.indexer.size == 0:
            return []

        k = min(k, self.indexer.size)
        scores, indices = self._search_index(query, k)
        return self._build_results(scores[0], indices[0])

    def _search_index(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Encodes and normalizes the query, then queries the FAISS index."""
        query_emb = np.array([self.embedding_model.encode_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)
        return self.indexer.index.search(query_emb, k)

    def _build_results(self, scores: np.ndarray, indices: np.ndarray) -> list[dict]:
        """Assembles the results, ignoring empty slots (FAISS -1)."""
        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            results.append(
                {
                    "text": self.indexer.chunks[idx],
                    "metadata": self.indexer.metadata[idx],
                    "score": float(score),
                }
            )
        return results
