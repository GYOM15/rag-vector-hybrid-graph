"""Pipeline RAG sensible au graphe : GraphRetriever + squelette partagé.

N'ajoute rien au pipeline générique ; seule la récupération (graphe) change.
"""

from shared.rag import BaseRAG


class GraphRAG(BaseRAG):
    """RAG sensible au graphe sur le squelette RAG commun. Voir :class:`shared.rag.BaseRAG`."""
