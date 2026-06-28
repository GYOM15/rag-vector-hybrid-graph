"""Stack 2: in-memory hybrid search (vector + BM25 + RRF)."""

from .fusion import reciprocal_rank_fusion
from .retriever import HybridRetriever
from .rag import HybridRAG

__all__ = ["reciprocal_rank_fusion", "HybridRetriever", "HybridRAG"]
