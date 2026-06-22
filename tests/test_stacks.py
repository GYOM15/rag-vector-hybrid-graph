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
# Extraction d'entités heuristique
# ---------------------------------------------------------------------------

def test_extract_finds_proper_nouns():
    ents = extract_entities("The Titanic sank near Newfoundland in April.")
    assert "Titanic" in ents
    assert "Newfoundland" in ents


def test_extract_strips_leading_stopword():
    # « The » en tête est retiré -> on garde « Titanic », pas « The Titanic ».
    assert "Titanic" in extract_entities("The Titanic was a ship.")
    assert "The Titanic" not in extract_entities("The Titanic was a ship.")


def test_extract_dedups_case_insensitive():
    ents = extract_entities("Paris is great. Paris again, Paris once more.")
    assert ents.count("Paris") == 1


def test_extract_skips_pure_stopwords():
    ents = extract_entities("The dog ran. This is fine.")
    assert "The" not in ents
    assert "This" not in ents


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
