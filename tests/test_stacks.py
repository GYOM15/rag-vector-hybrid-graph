"""Tests des briques pures des stacks 2 et 3 (sans dépendances lourdes).

On charge les modules par chemin de fichier pour éviter de déclencher les
imports lourds des packages (faiss, rank_bm25, networkx, sentence-transformers).
"""

import importlib.util
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _SRC / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reciprocal_rank_fusion = _load("stack2_hybrid/fusion.py", "fusion").reciprocal_rank_fusion
extract_entities = _load("stack3_graphrag/entity_extractor.py", "entity_extractor").extract_entities
tokenize = _load("stack2_hybrid/tokenizer.py", "hybrid_tokenizer").tokenize
_ir = _load("shared/ir_metrics.py", "ir_metrics")
recall_at_k, ndcg_at_k, reciprocal_rank = _ir.recall_at_k, _ir.ndcg_at_k, _ir.reciprocal_rank


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def test_rrf_rewards_higher_ranks():
    fused = reciprocal_rank_fusion([[10, 20, 30]])
    assert fused[10] > fused[20] > fused[30]


def test_rrf_combines_across_lists():
    # 5 est premier dans les deux classements -> meilleur score combiné.
    fused = reciprocal_rank_fusion([[5, 1, 2], [5, 3, 4]])
    assert max(fused, key=fused.get) == 5
    assert fused[5] == 2 / 60  # 1/(60+0) compté deux fois


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == {}


def test_rrf_k_parameter():
    assert reciprocal_rank_fusion([[7]], rrf_k=9)[7] == 1 / 9


# ---------------------------------------------------------------------------
# Extraction d'entités (spaCy NER)
# ---------------------------------------------------------------------------

def test_extract_finds_person_and_place():
    ents = extract_entities("Albert Einstein was born in Germany.")
    assert any("Einstein" in e for e in ents)
    assert "Germany" in ents


def test_extract_dedups_case_insensitive():
    ents = extract_entities("Paris is nice. Paris is big.")
    assert sum(e.lower() == "paris" for e in ents) == 1


def test_extract_excludes_dates_and_numbers():
    ents = extract_entities("There were 500 people in April 1912.")
    assert not any(any(c.isdigit() for c in e) for e in ents)
    assert "April" not in ents


# ---------------------------------------------------------------------------
# Tokenisation BM25 (hybride)
# ---------------------------------------------------------------------------

def test_tokenize_strips_punctuation_and_case():
    assert tokenize("April 15, 1912.") == tokenize("april 15 1912")


def test_tokenize_removes_stopwords():
    toks = tokenize("the plant and the fungus")
    assert "the" not in toks and "and" not in toks
    assert len(toks) == 2


def test_tokenize_stems_plurals():
    assert tokenize("plants") == tokenize("plant")
    assert tokenize("diseases") == tokenize("disease")


# ---------------------------------------------------------------------------
# Métriques IR (recall@k, nDCG@k, MRR)
# ---------------------------------------------------------------------------

def test_recall_at_k():
    rel = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x", "b"], rel, 1) == 0.5
    assert recall_at_k(["a", "x", "b"], rel, 3) == 1.0
    assert recall_at_k(["x", "y"], rel, 2) == 0.0


def test_reciprocal_rank():
    rel = {"b": 1}
    assert reciprocal_rank(["a", "b", "c"], rel) == 0.5
    assert reciprocal_rank(["x"], rel) == 0.0


def test_ndcg_perfect_and_zero():
    rel = {"a": 1}
    assert ndcg_at_k(["a", "b"], rel, 10) == 1.0
    assert ndcg_at_k(["b", "c"], rel, 10) == 0.0


def test_ndcg_rewards_higher_rank():
    rel = {"a": 1}
    assert ndcg_at_k(["x", "a"], rel, 10) < ndcg_at_k(["a", "x"], rel, 10)
