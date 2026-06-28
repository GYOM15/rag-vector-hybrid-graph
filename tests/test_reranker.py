"""Unit tests for the reranker — we fake the cross-encoder to isolate the sorting.

The real model is never loaded: we replace ``_model`` with a fake that returns
fixed scores. The tests stay fast and free of heavy dependencies, and verify
the logic of both modes (``replace`` and ``fusion`` RRF).
"""

import reranker
from reranker import CrossEncoderReranker


class _FakeModel:
    """Returns fixed scores, regardless of the query — to test the sorting alone."""

    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def _patch(monkeypatch, scores):
    monkeypatch.setattr(reranker, "_model", lambda name: _FakeModel(scores))


# Candidates in retriever order (base rank = position): a, b, c, d
_CANDS = [{"text": t} for t in ("a", "b", "c", "d")]
# Cross-encoder scores: a > c > d > b  → cross-encoder order = a, c, d, b
_SCORES = [0.9, 0.1, 0.8, 0.7]


def test_replace_follows_cross_encoder(monkeypatch):
    _patch(monkeypatch, _SCORES)
    out = CrossEncoderReranker().rerank("q", _CANDS, top_k=4, mode="replace")
    assert [c["text"] for c in out] == ["a", "c", "d", "b"]


def test_default_mode_is_replace(monkeypatch):
    _patch(monkeypatch, _SCORES)
    out = CrossEncoderReranker().rerank("q", _CANDS, top_k=4)
    assert [c["text"] for c in out] == ["a", "c", "d", "b"]


def test_fusion_blends_base_rank(monkeypatch):
    _patch(monkeypatch, _SCORES)
    out = CrossEncoderReranker().rerank("q", _CANDS, top_k=4, mode="fusion", rrf_k=1)
    # the base rank lifts 'b' ahead of 'd' (which the cross-encoder alone placed above)
    assert [c["text"] for c in out] == ["a", "c", "b", "d"]


def test_fusion_differs_from_replace(monkeypatch):
    _patch(monkeypatch, _SCORES)
    r = CrossEncoderReranker()
    replaced = [c["text"] for c in r.rerank("q", _CANDS, 4, mode="replace")]
    fused = [c["text"] for c in r.rerank("q", _CANDS, 4, mode="fusion", rrf_k=1)]
    assert replaced != fused  # fusion reinjects the base ranking


def test_top_k_truncates(monkeypatch):
    _patch(monkeypatch, _SCORES)
    out = CrossEncoderReranker().rerank("q", _CANDS, top_k=2, mode="replace")
    assert [c["text"] for c in out] == ["a", "c"]


def test_empty_candidates_returns_empty():
    assert CrossEncoderReranker().rerank("q", [], top_k=5) == []
