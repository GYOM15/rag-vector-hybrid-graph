"""Construction des trois pipelines RAG sur un même corpus.

Charge le corpus, le découpe (chunking partagé), l'encode, construit l'index
FAISS et le graphe d'entités, puis renvoie les trois architectures prêtes à
interroger. **Source unique** utilisée par l'application Streamlit ET le script
de benchmark — aucune logique dupliquée.
"""

import os

from shared.chunker import recursive_chunk
from shared.embeddings import EmbeddingModel
from shared.llm import call_llm
from shared.vector_index import FaissIndexer
from stack1_traditional import TraditionalRAG, VectorRetriever
from stack2_hybrid import HybridRAG, HybridRetriever
from stack3_graphrag import GraphRAG, GraphRetriever, build_graph

# Noms d'affichage des trois architectures (l'ordre = l'ordre d'affichage).
STACK_NAMES = {
    "vector": "Stack 1 — Vectoriel (FAISS)",
    "hybrid": "Stack 2 — Hybride (BM25 + RRF)",
    "graph": "Stack 3 — Graphe (networkx)",
}

_DATASET = "wikimedia/wikipedia"
_CONFIG = "20231101.simple"


def load_chunks(n_articles: int = 500, max_size: int = 500, overlap: int = 50):
    
    """Charge le corpus Wikipédia et le découpe en chunks (liste de `Chunk`)."""
    
    from datasets import load_dataset

    ds = load_dataset(_DATASET, _CONFIG, split=f"train[:{n_articles}]")
    chunks = []
    for doc in ds:
        chunks.extend(recursive_chunk(
            doc["text"], max_size=max_size, overlap=overlap,
            metadata={"title": doc["title"], "url": doc["url"]},
        ))
    return chunks


def assemble_stacks(texts, metadata, embedder: str = "all-MiniLM-L6-v2", llm_fn=call_llm,
                    rerank_mode: str | None = None, rerank_candidates: int = 30) -> dict:
    """Construit les 3 RAG à partir d'unités prêtes (chunks ou documents) + métadonnées.

    Index FAISS et graphe partagés ; seule la récupération diffère. Permet de
    brancher n'importe quel corpus (Wikipédia, BEIR…). Renvoie {nom affiché: RAG}.

    Si `rerank_mode` ∈ {"replace", "fusion"}, chaque récupérateur est enveloppé d'un
    étage de reranking cross-encoder. **Désactivé par défaut** : l'éval reranking
    montre que son intérêt dépend des données (cf. README).
    """
    embeddings = EmbeddingModel(embedder)
    indexer = FaissIndexer(dimension=embeddings.dimension)
    indexer.add(embeddings.encode(texts), texts, metadata)
    graph = build_graph(texts, metadata)

    reranker = None
    if rerank_mode in ("replace", "fusion"):
        from shared.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    def _staged(retriever):
        if reranker is None:
            return retriever
        from shared.reranker import RerankedRetriever
        return RerankedRetriever(retriever, reranker, mode=rerank_mode, candidates=rerank_candidates)

    return {
        STACK_NAMES["vector"]: TraditionalRAG(_staged(VectorRetriever(indexer, embeddings)), llm_fn),
        STACK_NAMES["hybrid"]: HybridRAG(_staged(HybridRetriever(indexer, embeddings)), llm_fn),
        STACK_NAMES["graph"]: GraphRAG(_staged(GraphRetriever(indexer, embeddings, graph)), llm_fn),
    }


def build_stacks(
    n_articles: int = 500,
    max_size: int = 500,
    overlap: int = 50,
    llm_fn=call_llm,
    embedder: str = "all-MiniLM-L6-v2",
    rerank_mode: str | None = None,
    rerank_candidates: int = 30,
) -> dict:

    """Construit les 3 RAG sur le corpus (mêmes chunking/index/prompt ; seule la
    récupération diffère). Renvoie {nom affiché: RAG}.

    `rerank_mode` (ou la variable d'environnement `RERANK_MODE`, off par défaut)
    active l'étage de reranking optionnel — voir `assemble_stacks`."""

    if rerank_mode is None:
        env = os.getenv("RERANK_MODE", "").strip().lower()
        rerank_mode = env if env in ("replace", "fusion") else None
    chunks = load_chunks(n_articles, max_size, overlap)
    return assemble_stacks(
        [c.text for c in chunks], [c.metadata for c in chunks], embedder, llm_fn,
        rerank_mode=rerank_mode, rerank_candidates=rerank_candidates,
    )
