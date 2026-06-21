"""Recherche dense par vecteurs avec FAISS.

Encode la requête avec le même modèle d'embeddings qu'à l'indexation, puis
interroge l'index FAISS pour les plus proches voisins par similarité cosinus.
"""

import numpy as np
import faiss

from shared.embeddings import EmbeddingModel
from shared.vector_index import FaissIndexer


class VectorRetriever:
    """Récupérateur dense adossé à un index FAISS."""

    def __init__(self, indexer: FaissIndexer, embedding_model: EmbeddingModel):
        self.indexer = indexer
        self.embedding_model = embedding_model

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Renvoie les k chunks les plus proches : liste de {text, metadata, score} (cosinus)."""
        if self.indexer.size == 0:
            return []

        k = min(k, self.indexer.size)
        scores, indices = self._search_index(query, k)
        return self._build_results(scores[0], indices[0])

    def _search_index(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Encode et normalise la requête, puis interroge l'index FAISS."""
        query_emb = np.array([self.embedding_model.encode_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)
        return self.indexer.index.search(query_emb, k)

    def _build_results(self, scores: np.ndarray, indices: np.ndarray) -> list[dict]:
        """Assemble les résultats, en ignorant les emplacements vides (-1 FAISS)."""
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
