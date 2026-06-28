"""Graph-aware RAG pipeline: GraphRetriever + shared skeleton.

Adds nothing to the generic pipeline; only the retrieval (graph) changes.
"""

from shared.rag import BaseRAG


class GraphRAG(BaseRAG):
    """Graph-aware RAG on the common RAG skeleton. See :class:`shared.rag.BaseRAG`."""
