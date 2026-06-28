"""Graph retrieval via local search over entities.

1. Vector seeds (FAISS) — semantic base and fallback.
2. Entities of the *query* (spaCy) -> matching graph nodes.
3. Linked chunks: mentioning those entities (MENTIONS) or neighboring entities
   (RELATED_TO, 1 hop, weighted more lightly).
Score = vector similarity + entity overlap weighted by **IDF**
(rare entities count more -> neutralizes "Plant", "Role", etc.),
**normalized by the chunk's entity richness**: without it, a "hub" document
(e.g. the "June" page citing dozens of countries) accumulates a huge additive
boost and displaces the real chunks as the corpus grows. The
sqrt(num entities) normalization — analogous to BM25 length normalization — favors
*focused* chunks and lets vector similarity decide.
Falls back to pure vector if the query has no known entity.
"""

import math

import faiss
import numpy as np

from shared.embeddings import EmbeddingModel
from shared.vector_index import FaissIndexer

from .entity_extractor import extract_entities

_VEC_SEEDS = 20          # vector seeds (semantic base + fallback)
_RELATED_DISCOUNT = 0.5  # weight of entities reached via RELATED_TO (1 hop)
_GRAPH_WEIGHT = 0.3      # weight of the graph signal vs vector similarity

# Normalization of the entity boost by chunk richness: we divide by f(num entities).
# "none" = naive version ("hub" documents rich in entities grab the boost and
# displace focused chunks). sqrt (default) = analogous to BM25 length normalization.
# Single knob, swept on a *held-out* split (cf. eval/sweep_entity_norm.py) — not
# tuned on the test set. All forms are >= 1 for n >= 1 (never any amplification).
_ENTITY_NORMS = {
    "none": lambda n: 1.0,
    "log": lambda n: 1.0 + math.log(n),
    "p25": lambda n: n ** 0.25,
    "sqrt": lambda n: math.sqrt(n),
    "p75": lambda n: n ** 0.75,
    "linear": lambda n: float(n),
}
_DEFAULT_ENTITY_NORM = "p75"  # chosen by held-out sweep (eval/sweep_entity_norm.py)


class GraphRetriever:
    """Entity local search: vector seeds + entity-linked chunks,
    scored by vector similarity + IDF-weighted entity overlap."""

    def __init__(self, indexer: FaissIndexer, embedding_model: EmbeddingModel, graph,
                 entity_norm: str = _DEFAULT_ENTITY_NORM):
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.graph = graph
        self._idf = self._compute_idf()
        self._chunk_n_entities = self._count_chunk_entities()
        self._norm = _ENTITY_NORMS.get(entity_norm, _ENTITY_NORMS[_DEFAULT_ENTITY_NORM])

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.indexer.size == 0:
            return []

        scored = dict(self._vector_seeds(query, max(_VEC_SEEDS, k)))  # vector base
        shared: dict[int, set] = {}

        for chunk_idx, entities, boost in self._entity_candidates(query):
            scored[chunk_idx] = scored.get(chunk_idx, 0.0) + _GRAPH_WEIGHT * boost
            shared.setdefault(chunk_idx, set()).update(entities)

        top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self._build_result(idx, score, shared.get(idx)) for idx, score in top]

    def _compute_idf(self) -> dict[str, float]:
        """Per-entity normalized IDF: log(1 + N/df) / log(1 + N), in (0, 1]."""
        n_chunks = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "chunk") or 1
        denom = math.log(1 + n_chunks)
        idf: dict[str, float] = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "entity":
                continue
            df = sum(1 for nb in self.graph.neighbors(node)
                     if self.graph.nodes[nb].get("type") == "chunk")
            if df:
                idf[node] = math.log(1 + n_chunks / df) / denom
        return idf

    def _count_chunk_entities(self) -> dict[int, int]:
        """Number of distinct entities per chunk (used to normalize the boost: penalizes hubs)."""
        counts: dict[int, int] = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "chunk":
                continue
            counts[data["index"]] = sum(
                1 for nb in self.graph.neighbors(node)
                if self.graph.nodes[nb].get("type") == "entity"
            )
        return counts

    def _entity_candidates(self, query: str) -> list[tuple[int, set, float]]:
        """Chunks linked to the query entities (direct + RELATED_TO), with IDF boost.

        Returns (chunk_index, {entities}, boost), boost = sum of the IDFs of the
        mentioned query entities (weight 1) and their RELATED_TO neighbors (reduced weight).
        """
        query_ids = [f"entity:{name.lower()}" for name in extract_entities(query)]
        query_ids = [eid for eid in query_ids if eid in self.graph]
        if not query_ids:
            return []

        # Weight per reached entity: 1.0 for the query entities, reduced for their neighbors.
        weight: dict[str, float] = {}
        for eid in query_ids:
            weight[eid] = 1.0
            for nb in self.graph.neighbors(eid):
                if self.graph.nodes[nb].get("type") == "entity":
                    weight[nb] = max(weight.get(nb, 0.0), _RELATED_DISCOUNT)

        contributions: dict[int, list] = {}
        for eid, w in weight.items():
            idf = self._idf.get(eid, 0.0)
            if not idf:
                continue
            name = self.graph.nodes[eid].get("name", eid)
            for nb in self.graph.neighbors(eid):
                node = self.graph.nodes[nb]
                if node.get("type") != "chunk":
                    continue
                entry = contributions.setdefault(node["index"], [0.0, set()])
                entry[0] += w * idf
                entry[1].add(name)
        # Normalization by entity richness: without it, a "hub" chunk (many
        # entities) accumulates an outsized boost and displaces focused chunks.
        return [(idx, ents, boost / self._norm(self._chunk_n_entities.get(idx, 1) or 1))
                for idx, (boost, ents) in contributions.items()]

    def _vector_seeds(self, query: str, n: int) -> list[tuple[int, float]]:
        """(idx, cosine similarity) of the n nearest chunks."""
        query_emb = np.array([self.embedding_model.encode_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)
        n = min(n, self.indexer.size)
        scores, indices = self.indexer.index.search(query_emb, n)
        return [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i != -1]

    def _build_result(self, idx: int, score: float, shared_entities: set | None) -> dict:
        metadata = dict(self.indexer.metadata[idx])
        if shared_entities:
            metadata["shared_entities"] = sorted(shared_entities)
        return {"text": self.indexer.chunks[idx], "metadata": metadata, "score": float(score)}
