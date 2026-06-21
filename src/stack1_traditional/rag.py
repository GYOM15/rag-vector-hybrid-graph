"""Pipeline RAG traditionnel : recherche vectorielle FAISS + génération LLM.

N'ajoute rien au squelette RAG partagé : seule la stratégie de récupération
(vectorielle dense, via VectorRetriever) est propre à ce stack.
"""

from shared.rag import BaseRAG


class TraditionalRAG(BaseRAG):
    """RAG vectoriel : VectorRetriever (FAISS) sur le squelette RAG commun.

    S'instancie avec ``TraditionalRAG(retriever, llm_fn)``. Voir
    :class:`shared.rag.BaseRAG` pour le pipeline et le format du résultat.
    """
