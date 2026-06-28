"""Hybrid RAG pipeline: HybridRetriever (vector + BM25 + RRF) + shared skeleton.

Adds nothing to the generic pipeline; only the retrieval (hybrid) changes.
"""

from shared.rag import BaseRAG


class HybridRAG(BaseRAG):
    """Hybrid RAG on the common RAG skeleton. See :class:`shared.rag.BaseRAG`."""
