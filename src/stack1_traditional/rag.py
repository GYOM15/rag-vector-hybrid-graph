"""Traditional RAG pipeline: FAISS vector search + LLM generation.

Adds nothing to the shared RAG skeleton: only the retrieval strategy
(dense vector, via VectorRetriever) is specific to this stack.
"""

from shared.rag import BaseRAG


class TraditionalRAG(BaseRAG):
    """Vector RAG: VectorRetriever (FAISS) on the common RAG skeleton.

    Instantiate with ``TraditionalRAG(retriever, llm_fn)``. See
    :class:`shared.rag.BaseRAG` for the pipeline and the result format.
    """
