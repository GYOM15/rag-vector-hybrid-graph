"""Test du décorateur `RerankedRetriever` — récupère large puis rerank.

On simule le récupérateur interne et le reranker (le vrai cross-encoder n'est jamais
chargé) pour vérifier le câblage : élargissement du top-N, propagation de `mode`,
réduction au top-k.
"""

from reranker import RerankedRetriever


class _FakeInner:
    def __init__(self):
        self.last_k = None

    def search(self, query, k):
        self.last_k = k
        return [{"text": f"d{i}"} for i in range(k)]


class _FakeReranker:
    def __init__(self):
        self.last = None

    def rerank(self, query, candidates, top_k, mode="replace"):
        self.last = (top_k, mode)
        return list(reversed(candidates))[:top_k]


def test_widens_to_candidates_then_reranks_to_k():
    inner, rr = _FakeInner(), _FakeReranker()
    out = RerankedRetriever(inner, rr, mode="fusion", candidates=30).search("q", k=5)
    assert inner.last_k == 30          # a élargi le top-N à `candidates`
    assert rr.last == (5, "fusion")    # rerank vers top_k=k, mode propagé
    # inner renvoie 30 candidats (d0..d29) ; le faux reranker inverse → top-5 = d29..d25
    assert [c["text"] for c in out] == ["d29", "d28", "d27", "d26", "d25"]


def test_widens_to_k_when_k_exceeds_candidates():
    inner, rr = _FakeInner(), _FakeReranker()
    RerankedRetriever(inner, rr, candidates=10).search("q", k=25)
    assert inner.last_k == 25          # max(candidates=10, k=25)


def test_default_mode_is_replace():
    inner, rr = _FakeInner(), _FakeReranker()
    RerankedRetriever(inner, rr).search("q", k=3)
    assert rr.last == (3, "replace")
