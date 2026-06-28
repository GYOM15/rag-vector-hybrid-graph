"""Building the three RAG pipelines on a single corpus.

Loads the corpus, splits it (shared chunking), encodes it, builds the FAISS index
and the entity graph, then returns the three architectures ready to query. **Single
source** used by both the Streamlit app AND the benchmark script — no duplicated
logic.
"""

import os

from shared.chunker import recursive_chunk
from shared.embeddings import EmbeddingModel
from shared.llm import call_llm
from shared.vector_index import FaissIndexer
from stack1_traditional import TraditionalRAG, VectorRetriever
from stack2_hybrid import HybridRAG, HybridRetriever
from stack3_graphrag import GraphRAG, GraphRetriever, build_graph

# Display names of the three architectures (order = display order).
STACK_NAMES = {
    "vector": "Stack 1 — Vector (FAISS)",
    "hybrid": "Stack 2 — Hybrid (BM25 + RRF)",
    "graph": "Stack 3 — Graph (networkx)",
}

_DATASET = "wikimedia/wikipedia"
_CONFIG = "20231101.simple"


def load_chunks(n_articles: int = 500, max_size: int = 500, overlap: int = 50):

    """Loads the Wikipedia corpus and splits it into chunks (list of `Chunk`)."""

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
    """Builds the 3 RAGs from ready-made units (chunks or documents) + metadata.

    Shared FAISS index and graph; only retrieval differs. Lets you plug in any
    corpus (Wikipedia, BEIR…). Returns {display name: RAG}.

    If `rerank_mode` ∈ {"replace", "fusion"}, each retriever is wrapped in a
    cross-encoder reranking stage. **Off by default**: the reranking eval shows its
    benefit is data-dependent (see README).
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

    """Builds the 3 RAGs on the corpus (same chunking/index/prompt; only retrieval
    differs). Returns {display name: RAG}.

    `rerank_mode` (or the `RERANK_MODE` environment variable, off by default)
    enables the optional reranking stage — see `assemble_stacks`."""

    if rerank_mode is None:
        env = os.getenv("RERANK_MODE", "").strip().lower()
        rerank_mode = env if env in ("replace", "fusion") else None
    chunks = load_chunks(n_articles, max_size, overlap)
    return assemble_stacks(
        [c.text for c in chunks], [c.metadata for c in chunks], embedder, llm_fn,
        rerank_mode=rerank_mode, rerank_candidates=rerank_candidates,
    )
