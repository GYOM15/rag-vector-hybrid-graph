"""Tests for the retrieval-eval hit rule (source article + whole-word gold) — no index."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.retrieval_eval import _aggregate, _first_hit_rank, _is_hit, unfindable_golds  # noqa: E402


def _ctx(text: str, title: str = "December") -> dict:
    return {"text": text, "metadata": {"title": title}}


_WRIGHT = {"question": "When did the Wright brothers fly?", "gold": "Wright", "title": "December"}


def test_hit_needs_the_items_source_article():
    text = "December 17, 1903 - The Wright brothers make their first flight."
    assert _is_hit(_ctx(text), _WRIGHT)
    assert not _is_hit(_ctx(text, title="Aviation"), _WRIGHT)  # right words, wrong article


def test_item_without_title_matches_any_article():
    item = {"question": "?", "gold": "Kabul"}
    assert _is_hit(_ctx("Kabul is the capital.", title="Anything"), item)


def test_gold_must_be_whole_words():
    assert not _is_hit(_ctx("A famous playwright was born."), _WRIGHT)
    assert not _is_hit(_ctx("Wrights and Wrightson."), _WRIGHT)
    assert _is_hit(_ctx("the wright brothers"), _WRIGHT)  # case-insensitive
    assert _is_hit(_ctx("(Wright)"), _WRIGHT)


def test_multi_word_gold_tolerates_any_whitespace():
    item = {"question": "?", "gold": "Sierra Leone becomes a republic", "title": "April"}
    assert _is_hit(_ctx("1971 - Sierra Leone\nbecomes a  republic.", title="April"), item)
    assert not _is_hit(_ctx("Sierra Leone becomes independent.", title="April"), item)


def test_gold_with_non_word_characters():
    fungi = {"question": "?", "gold": "Mycology—Fungi", "title": "Botany"}
    assert _is_hit(_ctx("Mycology—Fungi\nPhycology—Algae", title="Botany"), fungi)
    assert not _is_hit(_ctx("Mycology—Fungis", title="Botany"), fungi)
    cpp = {"question": "?", "gold": "C++"}  # \b would never match after the final "+"
    assert _is_hit(_ctx("Written in C++."), cpp)
    assert not _is_hit(_ctx("Written in C++x."), cpp)
    dotted = {"question": "?", "gold": "a.b"}  # regex metacharacters are literal
    assert not _is_hit(_ctx("axb"), dotted)


def test_unfindable_golds_lists_items_no_chunk_can_hit():
    chunks = [_ctx("December 17, 1903 - The Wright brothers fly.")]
    lost = {"question": "Who?", "gold": "Naismith", "title": "December"}
    assert unfindable_golds(chunks, [_WRIGHT, lost]) == [lost]


class _FakeRetriever:
    def __init__(self, results: list[dict]):
        self.results = results

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self.results[:k]


def test_first_hit_rank_skips_false_hits():
    retriever = _FakeRetriever([
        _ctx("A playwright."),                                 # substring only
        _ctx("The Wright brothers.", title="Aviation"),        # wrong article
        _ctx("December 17, 1903 - The Wright brothers fly."),  # the answer chunk
    ])
    assert _first_hit_rank(retriever, _WRIGHT, k_max=10) == 3
    assert _first_hit_rank(retriever, _WRIGHT, k_max=2) is None


def test_aggregate_reports_n_and_an_mrr_interval():
    out = _aggregate([1, 2, None, 1])
    assert out["n"] == 4
    assert out["mrr"] == pytest.approx((1 + 0.5 + 0 + 1) / 4, abs=1e-3)
    lo, hi = out["mrr_ci95"]
    assert lo <= out["mrr"] <= hi
