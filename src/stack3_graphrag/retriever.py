"""Récupération sensible au graphe : amorçage vectoriel + expansion par le graphe.

1. Recherche vectorielle (FAISS) pour trouver les chunks-graines.
2. Expansion : pour chaque graine, on suit les arêtes MENTIONS afin de trouver
   les chunks voisins partageant des entités. Le score combine la similarité
   vectorielle et un bonus proportionnel au nombre d'entités partagées.
"""

import numpy as np
import faiss

from shared.embeddings import EmbeddingModel
from shared.vector_index import FaissIndexer

_GRAPH_BONUS = 0.1  # bonus de score par entité partagée avec une graine


class GraphRetriever:
    """Récupérateur combinant index vectoriel FAISS et graphe networkx (graines + voisins)."""

    def __init__(self, indexer: FaissIndexer, embedding_model: EmbeddingModel, graph):
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.graph = graph

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Renvoie les k chunks les plus pertinents (vectoriel + voisins de graphe)."""
        if self.indexer.size == 0:
            return []

        seeds = self._vector_seeds(query, k)
        scored = {idx: score for idx, score in seeds}
        shared: dict[int, set] = {}

        for idx, _ in seeds:
            for neighbor_idx, entities in self._graph_neighbors(idx):
                scored[neighbor_idx] = scored.get(neighbor_idx, 0.0) + _GRAPH_BONUS * len(entities)
                shared.setdefault(neighbor_idx, set()).update(entities)

        top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self._build_result(idx, score, shared.get(idx)) for idx, score in top]

    def _vector_seeds(self, query: str, k: int) -> list[tuple[int, float]]:
        """(idx, score) des k chunks les plus proches par similarité vectorielle."""
        query_emb = np.array([self.embedding_model.encode_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)
        scores, indices = self.indexer.index.search(query_emb, k)
        return [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i != -1]

    def _graph_neighbors(self, chunk_index: int) -> list[tuple[int, set]]:
        """Chunks voisins partageant des entités avec le chunk `chunk_index`.

        Suit le motif (chunk)-[:MENTIONS]->(entité)<-[:MENTIONS]-(voisin) et
        renvoie des tuples (index_voisin, {entités partagées}).
        """
        chunk_id = f"chunk:{chunk_index}"
        if chunk_id not in self.graph:
            return []

        neighbors: dict[int, set] = {}
        for entity_id in self.graph.neighbors(chunk_id):
            if self.graph.nodes[entity_id].get("type") != "entity":
                continue
            entity_name = self.graph.nodes[entity_id]["name"]
            for other_id in self.graph.neighbors(entity_id):
                node = self.graph.nodes[other_id]
                if node.get("type") != "chunk" or other_id == chunk_id:
                    continue
                neighbors.setdefault(node["index"], set()).add(entity_name)
        return list(neighbors.items())

    def _build_result(self, idx: int, score: float, shared_entities: set | None) -> dict:
        metadata = dict(self.indexer.metadata[idx])
        if shared_entities:
            metadata["shared_entities"] = sorted(shared_entities)
        return {"text": self.indexer.chunks[idx], "metadata": metadata, "score": float(score)}
