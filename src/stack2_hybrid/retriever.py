"""In-memory hybrid retrieval: vector (FAISS) + lexical (BM25).

The two rankings are fused by Reciprocal Rank Fusion (RRF). Reuses the FAISS
index from stack 1 for the vector part and builds an in-memory BM25 index
over the same chunks — no external server.

Only chunks containing a query term enter the BM25 list: argsort would otherwise
pad it with zero-score chunks in arbitrary order, and RRF would credit them like
real lexical hits. Both lists are fused at a fixed depth (`candidates`), so the
top-m does not depend on k; when that pool holds fewer than k chunks, the next
vector hits complete it, so the hybrid still returns k results and hands a
reranker the same pool size as Vector/Graph.
"""

from itertools import islice

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
        # BM25Okapi divides by the corpus size and the vocabulary size: it cannot be
        # built on an empty index, nor on chunks without any token (punctuation only).
        corpus_tokens = [tokenize(text) for text in indexer.chunks]
        self._bm25 = BM25Okapi(corpus_tokens) if any(corpus_tokens) else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Returns the k most relevant chunks (score = combined RRF score).

        The fusion depth does not depend on k (a deeper fusion would reorder the
        top-10), so `search(q, k)[:m] == search(q, m)` for every m <= k.
        """
        if self.indexer.size == 0:
            return []
        depth = min(self.candidates, self.indexer.size)
        vector = self._vector_ranking(query, min(max(depth, k), self.indexer.size))
        fused = reciprocal_rank_fusion([vector[:depth], self._bm25_ranking(query, depth)])
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        if len(top) < k:
            # Pool smaller than k: complete it with the next vector hits, in vector
            # order, scored by their vector rank alone (below every fused score).
            vector_scores = reciprocal_rank_fusion([vector])
            top += [(idx, vector_scores[idx]) for idx in vector if idx not in fused][:k - len(top)]
        return [self._build_result(idx, score) for idx, score in top]

    def _vector_ranking(self, query: str, n: int) -> list[int]:
        """Ids of the n nearest chunks by vector similarity."""
        query_emb = np.array([self.embedding_model.encode_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)
        _, indices = self.indexer.index.search(query_emb, n)
        return [int(i) for i in indices[0] if i != -1]

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        """Ids of the (at most) n top-scoring chunks by BM25 (lexical search).

        Chunks without any query term are not lexical hits: they are dropped. The
        filter tests term presence, not `score > 0`: in BM25Okapi a term found in
        exactly half the chunks has an IDF of 0, and a more common one can score < 0.
        """
        if self._bm25 is None:
            return []
        terms = tokenize(query)
        scores = self._bm25.get_scores(terms)
        hits = (int(i) for i in np.argsort(scores)[::-1]
                if any(term in self._bm25.doc_freqs[i] for term in terms))
        return list(islice(hits, n))

    def _build_result(self, idx: int, score: float) -> dict:
        return {
            "text": self.indexer.chunks[idx],
            "metadata": self.indexer.metadata[idx],
            "score": float(score),
        }
