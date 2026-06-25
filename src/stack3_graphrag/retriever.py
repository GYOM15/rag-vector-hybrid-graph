"""Récupération graphe par local-search sur les entités.

1. Graines vectorielles (FAISS) — base sémantique et repli.
2. Entités de la *requête* (spaCy) → nœuds du graphe correspondants.
3. Chunks reliés : mentionnant ces entités (MENTIONS) ou des entités voisines
   (RELATED_TO, 1 hop, pondéré plus faiblement).
Score = similarité vectorielle + recouvrement d'entités pondéré par **IDF**
(les entités rares comptent plus → neutralise « Plant », « Role », etc.),
**normalisé par la richesse en entités du chunk** : sans cela, un document « hub »
(p. ex. la page « June » qui cite des dizaines de pays) accumule un boost additif
énorme et déloge les vrais chunks à mesure que le corpus grandit. La normalisation
√(nb d'entités) — analogue à la normalisation par longueur de BM25 — favorise les
chunks *focalisés* et laisse la similarité vectorielle trancher.
Repli sur le pur vectoriel si la requête n'a aucune entité connue.
"""

import math

import faiss
import numpy as np

from shared.embeddings import EmbeddingModel
from shared.vector_index import FaissIndexer

from .entity_extractor import extract_entities

_VEC_SEEDS = 20          # graines vectorielles (base sémantique + repli)
_RELATED_DISCOUNT = 0.5  # poids des entités atteintes via RELATED_TO (1 hop)
_GRAPH_WEIGHT = 0.3      # poids du signal graphe vs similarité vectorielle


class GraphRetriever:
    """Local-search par entités : graines vectorielles + chunks liés par entités,
    scorés par similarité vectorielle + recouvrement d'entités pondéré IDF."""

    def __init__(self, indexer: FaissIndexer, embedding_model: EmbeddingModel, graph):
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.graph = graph
        self._idf = self._compute_idf()
        self._chunk_n_entities = self._count_chunk_entities()

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.indexer.size == 0:
            return []

        scored = dict(self._vector_seeds(query, max(_VEC_SEEDS, k)))  # base vectorielle
        shared: dict[int, set] = {}

        for chunk_idx, entities, boost in self._entity_candidates(query):
            scored[chunk_idx] = scored.get(chunk_idx, 0.0) + _GRAPH_WEIGHT * boost
            shared.setdefault(chunk_idx, set()).update(entities)

        top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self._build_result(idx, score, shared.get(idx)) for idx, score in top]

    def _compute_idf(self) -> dict[str, float]:
        """IDF normalisé par entité : log(1 + N/df) / log(1 + N), dans (0, 1]."""
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
        """Nb d'entités distinctes par chunk (sert à normaliser le boost : pénalise les hubs)."""
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
        """Chunks reliés aux entités de la requête (direct + RELATED_TO), avec boost IDF.

        Renvoie (index_chunk, {entités}, boost), boost = somme des IDF des entités
        de requête mentionnées (poids 1) et de leurs voisines RELATED_TO (poids réduit).
        """
        query_ids = [f"entity:{name.lower()}" for name in extract_entities(query)]
        query_ids = [eid for eid in query_ids if eid in self.graph]
        if not query_ids:
            return []

        # Poids par entité atteinte : 1.0 pour les entités de la requête, réduit pour leurs voisines.
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
        # Normalisation par √(richesse en entités) : sans elle, un chunk « hub »
        # (beaucoup d'entités) accumule un boost démesuré et déloge les chunks focalisés.
        return [(idx, ents, boost / math.sqrt(self._chunk_n_entities.get(idx, 1) or 1))
                for idx, (boost, ents) in contributions.items()]

    def _vector_seeds(self, query: str, n: int) -> list[tuple[int, float]]:
        """(idx, similarité cosinus) des n chunks les plus proches."""
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
