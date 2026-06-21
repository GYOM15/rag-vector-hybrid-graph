"""Récupération hybride en mémoire : vectoriel (FAISS) + lexical (BM25).

Les deux classements sont fusionnés par Reciprocal Rank Fusion (RRF). Réutilise
l'index FAISS du stack 1 pour la partie vectorielle et construit un index BM25
en mémoire sur les mêmes chunks — aucun serveur externe.
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from shared.embeddings import EmbeddingModel
from shared.vector_index import FaissIndexer

from .fusion import reciprocal_rank_fusion


def _tokenize(text: str) -> list[str]:
    """Tokenisation minimale pour BM25 : minuscules + découpage sur les espaces."""
    return text.lower().split()


class HybridRetriever:
    """Recherche dense (FAISS) + lexicale (BM25), fusionnées par RRF."""

    def __init__(self, indexer: FaissIndexer, embedding_model: EmbeddingModel, candidates: int = 20):
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.candidates = candidates
        self._bm25 = BM25Okapi([_tokenize(text) for text in indexer.chunks])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Renvoie les k chunks les plus pertinents (score = score RRF combiné)."""
        if self.indexer.size == 0:
            return []
        n = min(self.candidates, self.indexer.size)
        rankings = [self._vector_ranking(query, n), self._bm25_ranking(query, n)]
        fused = reciprocal_rank_fusion(rankings)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self._build_result(idx, score) for idx, score in top]

    def _vector_ranking(self, query: str, n: int) -> list[int]:
        """Ids des n chunks les plus proches par similarité vectorielle."""
        query_emb = np.array([self.embedding_model.encode_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)
        _, indices = self.indexer.index.search(query_emb, n)
        return [int(i) for i in indices[0] if i != -1]

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        """Ids des n chunks les mieux notés par BM25 (recherche lexicale)."""
        scores = self._bm25.get_scores(_tokenize(query))
        return [int(i) for i in np.argsort(scores)[::-1][:n]]

    def _build_result(self, idx: int, score: float) -> dict:
        return {
            "text": self.indexer.chunks[idx],
            "metadata": self.indexer.metadata[idx],
            "score": float(score),
        }
