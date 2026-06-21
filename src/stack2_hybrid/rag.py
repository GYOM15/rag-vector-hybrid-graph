"""Pipeline RAG hybride : HybridRetriever (vectoriel + BM25 + RRF) + squelette partagé.

N'ajoute rien au pipeline générique ; seule la récupération (hybride) change.
"""

from shared.rag import BaseRAG


class HybridRAG(BaseRAG):
    """RAG hybride sur le squelette RAG commun. Voir :class:`shared.rag.BaseRAG`."""
