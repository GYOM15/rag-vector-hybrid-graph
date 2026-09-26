"""Retriever tests on the real libraries (faiss, numpy, networkx, rank_bm25).

No sentence-transformers: a fake embedding model returns a fixed vector per query
and the FAISS index is filled with synthetic unit vectors whose cosine with the
query is known exactly, so every expected ranking can be written down. These pin
the fixes that biased the Vector/Hybrid/Graph comparison (zero-score BM25 hits,
hybrid depth < k, graph-only chunks scored without their cosine...). Skipped, not
failed, in an environment without the retrieval libraries.
"""

import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")
nx = pytest.importorskip("networkx")

from shared.embeddings import _infer_prefixes  # noqa: E402
from shared.vector_index import FaissIndexer  # noqa: E402
from stack1_traditional.retriever import VectorRetriever  # noqa: E402
from stack2_hybrid.retriever import HybridRetriever  # noqa: E402
from stack3_graphrag import retriever as graph_module  # noqa: E402
from stack3_graphrag.retriever import GraphRetriever  # noqa: E402

_DIM = 96  # > number of chunks: axis 0 = query direction, axis 1+i = private to chunk i
_QUERY_VEC = np.eye(_DIM, dtype=np.float32)[0]


class _FakeEmbeddings:
    """Stands in for EmbeddingModel: every query maps to the same unit vector."""

    dimension = _DIM

    def encode_query(self, query: str) -> np.ndarray:
        return _QUERY_VEC


def _indexer(cosines: list[float], texts: list[str] | None = None) -> FaissIndexer:
    """Index of unit vectors whose cosine with the query is exactly `cosines[i]`."""
    vectors = np.zeros((len(cosines), _DIM), dtype=np.float32)
    for i, cos in enumerate(cosines):
        vectors[i, 0] = cos
        vectors[i, 1 + i] = np.sqrt(1.0 - cos * cos)
    texts = texts or [f"chunk {i}" for i in range(len(cosines))]
    indexer = FaissIndexer(dimension=_DIM)
    indexer.add(vectors, texts, [{"doc_id": f"d{i}"} for i in range(len(cosines))])
    return indexer


def _descending(n: int) -> list[float]:
    """Distinct cosines 0.9, 0.89, ... -> vector order = chunk order."""
    return [0.9 - 0.01 * i for i in range(n)]


def _texts(results: list[dict]) -> list[str]:
    return [r["text"] for r in results]


# ---------------------------------------------------------------------------
# Hybrid (vector + BM25 + RRF)
# ---------------------------------------------------------------------------

def test_hybrid_oov_query_returns_pure_vector_order():
    """No query term in the corpus: every BM25 score is 0. Those chunks are not
    lexical hits — RRF used to credit them (in arbitrary order) and reshuffle
    the vector ranking."""
    indexer = _indexer(_descending(40))
    hybrid = HybridRetriever(indexer, _FakeEmbeddings(), candidates=20)
    vector = VectorRetriever(indexer, _FakeEmbeddings())

    assert _texts(hybrid.search("xyzzy quux", k=10)) == _texts(vector.search("xyzzy quux", k=10))


def test_hybrid_real_lexical_hit_still_counts():
    """A chunk sharing a rare query term is lifted by BM25 despite a low cosine."""
    texts = [f"common filler text {i}" for i in range(40)]
    texts[39] = "the zebra migration"  # last by cosine
    hybrid = HybridRetriever(_indexer(_descending(40), texts), _FakeEmbeddings(), candidates=20)

    assert "the zebra migration" in _texts(hybrid.search("zebra", k=5))


@pytest.mark.parametrize("k", [30, 50])
def test_hybrid_returns_k_results_beyond_candidates(k):
    """Each list is min(max(candidates, k), size) deep: with candidates=20, k=30
    used to return fewer than 30 results and k > 40 at most 40."""
    texts = [f"filler {i}" for i in range(60)]
    texts[:3] = ["rare term alpha", "rare term beta", "rare term gamma"]
    hybrid = HybridRetriever(_indexer(_descending(60), texts), _FakeEmbeddings(), candidates=20)

    assert len(hybrid.search("rare", k=k)) == k


def test_hybrid_empty_index_returns_empty():
    hybrid = HybridRetriever(FaissIndexer(dimension=_DIM), _FakeEmbeddings())  # no BM25 crash
    assert hybrid.search("anything", k=5) == []


# ---------------------------------------------------------------------------
# Graph (vector seeds + entity boost)
# ---------------------------------------------------------------------------

def _graph(n_chunks: int, mentions: dict[int, list[str]]) -> "nx.Graph":
    """Same schema as graph_builder.build_graph, built directly (no spaCy)."""
    graph = nx.Graph()
    for i in range(n_chunks):
        graph.add_node(f"chunk:{i}", type="chunk", index=i)
    for i, names in mentions.items():
        for name in names:
            eid = f"entity:{name.lower()}"
            graph.add_node(eid, type="entity", name=name)
            graph.add_edge(f"chunk:{i}", eid, kind="MENTIONS")
    return graph


@pytest.fixture
def query_entities(monkeypatch):
    """The query mentions 'Atlantis' iff its text does (replaces spaCy NER)."""
    monkeypatch.setattr(
        graph_module, "extract_entities", lambda text: ["Atlantis"] if "Atlantis" in text else []
    )


def _graph_retriever(cosines: list[float], mentions: dict[int, list[str]], **kw):
    graph = _graph(len(cosines), mentions)
    return GraphRetriever(_indexer(cosines), _FakeEmbeddings(), graph, **kw)


def test_graph_entity_only_chunk_gets_cosine_plus_boost(query_entities):
    """Chunk 35 is outside the 20/30 vector seeds, reached only through the graph.
    It must be scored like the others: its real cosine + _GRAPH_WEIGHT * boost
    (it used to get 0 + boost, i.e. <= 0.3, and never make the top-k)."""
    cosines = _descending(40)
    cosines[35] = 0.555
    retriever = _graph_retriever(cosines, {35: ["Atlantis"]})

    results = retriever.search("Where is Atlantis?", k=10)

    # Only chunk mentioning a unique entity -> IDF 1, 1 entity -> normalized boost 1.
    expected = 0.555 + graph_module._GRAPH_WEIGHT * 1.0
    assert results[5]["text"] == "chunk 35"  # between cosines 0.86 and 0.85
    assert results[5]["score"] == pytest.approx(expected, abs=1e-5)
    assert results[5]["metadata"]["shared_entities"] == ["Atlantis"]


def test_graph_top10_does_not_depend_on_k(query_entities):
    """Chunk 25 (cosine rank 26) is a seed at k=30 but not at k=10: it used to be
    absent from the top-10 at k=10 and ranked first at k=30."""
    retriever = _graph_retriever(_descending(40), {25: ["Atlantis"]})

    top_k10 = _texts(retriever.search("Atlantis", k=10))
    top_k30 = _texts(retriever.search("Atlantis", k=30))

    assert top_k10 == top_k30[:10]
    assert top_k10[0] == "chunk 25"


def test_graph_without_query_entity_is_pure_vector(query_entities):
    retriever = _graph_retriever(_descending(40), {25: ["Atlantis"]})
    assert _texts(retriever.search("no known entity", k=5)) == [f"chunk {i}" for i in range(5)]


def test_graph_unknown_entity_norm_raises():
    with pytest.raises(ValueError, match="entity_norm"):
        _graph_retriever(_descending(5), {}, entity_norm="cubic")


@pytest.mark.parametrize("norm", list(graph_module._ENTITY_NORMS))
def test_graph_known_entity_norms_accepted(norm):
    _graph_retriever(_descending(5), {}, entity_norm=norm)


# ---------------------------------------------------------------------------
# FaissIndexer persistence
# ---------------------------------------------------------------------------

def test_indexer_roundtrip_with_dotted_prefix(tmp_path):
    """with_suffix() truncated "index.v1" and "index.v2" to the same "index.faiss"."""
    v1, v2 = _indexer(_descending(3)), _indexer(_descending(5))
    v1.save(str(tmp_path / "index.v1"))
    v2.save(str(tmp_path / "index.v2"))
    assert (tmp_path / "index.v2.faiss").exists()
    assert (tmp_path / "index.v2.chunks.json").exists()
    assert (tmp_path / "index.v2.meta.json").exists()

    loaded = FaissIndexer()
    loaded.load(str(tmp_path / "index.v1"))
    assert (loaded.size, loaded.dimension) == (3, _DIM)
    assert loaded.chunks == v1.chunks and loaded.metadata == v1.metadata
    assert _texts(VectorRetriever(loaded, _FakeEmbeddings()).search("q", k=3)) == v1.chunks


def test_indexer_load_rejects_misaligned_files(tmp_path):
    prefix = tmp_path / "index"
    _indexer(_descending(4)).save(str(prefix))
    (tmp_path / "index.chunks.json").write_text('["only", "three", "chunks"]', encoding="utf-8")

    fresh = FaissIndexer(dimension=_DIM)
    with pytest.raises(ValueError, match="4 vectors, 3 chunks, 4 metadata"):
        fresh.load(str(prefix))
    assert fresh.size == 0 and fresh.chunks == []  # state left untouched


# ---------------------------------------------------------------------------
# Embedding prefixes (model-family inference)
# ---------------------------------------------------------------------------

_E5 = ("query: ", "passage: ")
_BGE = ("Represent this sentence for searching relevant passages: ", "")


@pytest.mark.parametrize("name, expected", [
    ("intfloat/e5-base-v2", _E5),
    ("intfloat/multilingual-e5-large", _E5),
    ("BAAI/bge-small-en-v1.5", _BGE),
    ("all-MiniLM-L6-v2", ("", "")),
    ("acme/tree5-embedder", ("", "")),                    # "e5" inside a word
    ("/home/joe5/models/all-MiniLM-L6-v2", ("", "")),     # "e5" in a parent directory
    ("acme/hubgear-mini", ("", "")),                      # "bge" inside a word
])
def test_infer_prefixes_matches_whole_tokens(name, expected):
    assert _infer_prefixes(name) == expected


def test_embeddings_import_does_not_load_sentence_transformers():
    """Every retriever imports shared.embeddings for type hints: it must stay light."""
    src = Path(__file__).resolve().parent.parent / "src"
    code = (f"import sys; sys.path.insert(0, {str(src)!r}); import shared.embeddings; "
            "sys.exit('sentence_transformers' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
