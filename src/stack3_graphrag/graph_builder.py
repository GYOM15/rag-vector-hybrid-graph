"""Construction d'un graphe de connaissances en mémoire avec networkx.

Chaque chunk et chaque entité devient un nœud. Une arête `MENTIONS` relie un
chunk aux entités qu'il contient. Deux entités co-occurrentes (dans le même
chunk) sont reliées par une arête `RELATED_TO` pondérée par leur nombre de
co-occurrences.
"""

import networkx as nx

from .entity_extractor import extract_entities


def build_graph(chunks: list[str], metadata: list[dict]) -> "nx.Graph":
    """Construit le graphe : nœuds ``chunk:{i}`` et ``entity:{nom}``, arêtes MENTIONS
    (chunk→entité) et RELATED_TO (entité↔entité, pondérée par co-occurrence)."""
    graph = nx.Graph()

    for i, (text, meta) in enumerate(zip(chunks, metadata)):
        chunk_id = f"chunk:{i}"
        graph.add_node(chunk_id, type="chunk", text=text, metadata=meta, index=i)

        entities = extract_entities(text)
        for entity in entities:
            entity_id = f"entity:{entity.lower()}"
            if entity_id not in graph:
                graph.add_node(entity_id, type="entity", name=entity)
            graph.add_edge(chunk_id, entity_id, kind="MENTIONS")

        _link_cooccurrences(graph, entities)

    return graph


def _link_cooccurrences(graph: "nx.Graph", entities: list[str]) -> None:
    """Relie (ou renforce) les entités apparaissant ensemble dans un même chunk."""
    for a_pos in range(len(entities)):
        for b_pos in range(a_pos + 1, len(entities)):
            a = f"entity:{entities[a_pos].lower()}"
            b = f"entity:{entities[b_pos].lower()}"
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, kind="RELATED_TO", weight=1)
