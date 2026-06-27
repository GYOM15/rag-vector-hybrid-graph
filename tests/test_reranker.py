"""Tests unitaires du reranker — on simule le cross-encoder pour isoler le tri.

Le vrai modèle n'est jamais chargé : on remplace ``_model`` par un faux qui renvoie
des scores fixés. Les tests restent rapides et sans dépendance lourde, et vérifient
la logique des deux modes (``replace`` et ``fusion`` RRF).
"""

import reranker
from reranker import CrossEncoderReranker


class _FakeModel:
    """Renvoie des scores fixés, quelle que soit la requête — pour tester le tri seul."""

    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def _patch(monkeypatch, scores):
    monkeypatch.setattr(reranker, "_model", lambda name: _FakeModel(scores))


# Candidats dans l'ordre du récupérateur (rang de base = position) : a, b, c, d
_CANDS = [{"text": t} for t in ("a", "b", "c", "d")]
# Scores cross-encoder : a > c > d > b  → ordre cross-encoder = a, c, d, b
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
    # le rang de base remonte 'b' devant 'd' (que le cross-encoder seul plaçait au-dessus)
    assert [c["text"] for c in out] == ["a", "c", "b", "d"]


def test_fusion_differs_from_replace(monkeypatch):
    _patch(monkeypatch, _SCORES)
    r = CrossEncoderReranker()
    replaced = [c["text"] for c in r.rerank("q", _CANDS, 4, mode="replace")]
    fused = [c["text"] for c in r.rerank("q", _CANDS, 4, mode="fusion", rrf_k=1)]
    assert replaced != fused  # la fusion réinjecte le classement de base


def test_top_k_truncates(monkeypatch):
    _patch(monkeypatch, _SCORES)
    out = CrossEncoderReranker().rerank("q", _CANDS, top_k=2, mode="replace")
    assert [c["text"] for c in out] == ["a", "c"]


def test_empty_candidates_returns_empty():
    assert CrossEncoderReranker().rerank("q", [], top_k=5) == []
