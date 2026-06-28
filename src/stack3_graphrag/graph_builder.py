"""Building an in-memory knowledge graph with networkx.

Each chunk and each entity becomes a node. A `MENTIONS` edge links a chunk
to the entities it contains. Two co-occurring entities (in the same chunk)
are linked by a `RELATED_TO` edge weighted by their number of
co-occurrences.
"""

import networkx as nx

from .entity_extractor import extract_entities


def build_graph(chunks: list[str], metadata: list[dict]) -> "nx.Graph":
    """Builds the graph: ``chunk:{i}`` and ``entity:{name}`` nodes, MENTIONS edges
    (chunk->entity) and RELATED_TO edges (entity<->entity, weighted by co-occurrence)."""
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
    """Links (or strengthens) entities appearing together in the same chunk."""
    for a_pos in range(len(entities)):
        for b_pos in range(a_pos + 1, len(entities)):
            a = f"entity:{entities[a_pos].lower()}"
            b = f"entity:{entities[b_pos].lower()}"
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, kind="RELATED_TO", weight=1)
