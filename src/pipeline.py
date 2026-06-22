"""Construction des trois pipelines RAG sur un même corpus.

Charge le corpus, le découpe (chunking partagé), l'encode, construit l'index
FAISS et le graphe d'entités, puis renvoie les trois architectures prêtes à
interroger. **Source unique** utilisée par l'application Streamlit ET le script
de benchmark — aucune logique dupliquée.
"""

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


def load_chunks(n_articles: int = 100, max_size: int = 500, overlap: int = 50):
    
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


def build_stacks(
    n_articles: int = 100,
    max_size: int = 500,
    overlap: int = 50,
    llm_fn=call_llm,
    embedder: str = "all-MiniLM-L6-v2",
) -> dict:
    
    """Construit les 3 RAG sur le corpus (mêmes chunking/index/prompt ; seule la
    récupération diffère). Renvoie {nom affiché: RAG}."""
    
    chunks = load_chunks(n_articles, max_size, overlap)
    texts = [c.text for c in chunks]
    metadata = [c.metadata for c in chunks]

    embeddings = EmbeddingModel(embedder)
    indexer = FaissIndexer(dimension=embeddings.dimension)
    indexer.add(embeddings.encode(texts), texts, metadata)
    graph = build_graph(texts, metadata)

    return {
        STACK_NAMES["vector"]: TraditionalRAG(VectorRetriever(indexer, embeddings), llm_fn),
        STACK_NAMES["hybrid"]: HybridRAG(HybridRetriever(indexer, embeddings), llm_fn),
        STACK_NAMES["graph"]: GraphRAG(GraphRetriever(indexer, embeddings, graph), llm_fn),
    }
