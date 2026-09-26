"""Tests for the anti-regression guardrail logic (`check`) — no heavy build.

We only test the baseline ↔ scores comparisons (`check`, `check_queries`) and the
self-test verdict (the heavy imports in `check_regression` are lazy, so importing it
stays lightweight).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from check_regression import check, check_queries  # noqa: E402

_BASE = {"metric": "ndcg@5", "tolerance": 0.05,
         "scores": {"Vector": 1.0, "Hybrid": 0.9, "Graph": 0.8}}


def test_no_regression_when_scores_hold():
    assert check({"Vector": 1.0, "Hybrid": 0.9, "Graph": 0.8}, _BASE) == []


def test_small_dip_within_tolerance_passes():
    # −0.04 stays below the 0.05 tolerance → tolerated
    assert check({"Vector": 0.96, "Hybrid": 0.9, "Graph": 0.8}, _BASE) == []


def test_drop_beyond_tolerance_is_flagged():
    assert check({"Vector": 1.0, "Hybrid": 0.9, "Graph": 0.50}, _BASE) == [("Graph", 0.8, 0.50)]


def test_boundary_exactly_at_tolerance_passes():
    # want − tol = 0.75 ; got = 0.75 → OK (strict < comparison)
    assert check({"Vector": 1.0, "Hybrid": 0.9, "Graph": 0.75}, _BASE) == []


def test_missing_architecture_counts_as_zero():
    assert check({"Vector": 1.0, "Hybrid": 0.9}, _BASE) == [("Graph", 0.8, 0.0)]


def test_multiple_regressions_all_flagged():
    failures = check({"Vector": 0.5, "Hybrid": 0.9, "Graph": 0.5}, _BASE)
    assert {f[0] for f in failures} == {"Vector", "Graph"}


# --- per-query check: the partial breakages a mean over 18 queries absorbs --------

_QBASE = _BASE | {"query_tolerance": 0.05,
                  "per_query": {"Vector": {"q0": 1.0, "lq1": 0.5}, "Hybrid": {"q0": 1.0, "lq1": 1.0}}}


def test_check_queries_flags_one_query_slipping_from_rank_1_to_2():
    # nDCG@5 1.0 -> 0.631: invisible in a mean over 18 queries (-0.02), caught per query.
    per_query = {"Vector": {"q0": 0.6309, "lq1": 0.5}, "Hybrid": {"q0": 1.0, "lq1": 1.0}}
    assert check_queries(per_query, _QBASE) == [("Vector", "q0", 1.0, 0.6309)]


def test_check_queries_tolerates_small_moves_and_gains():
    per_query = {"Vector": {"q0": 0.96, "lq1": 1.0}, "Hybrid": {"q0": 1.0, "lq1": 1.0}}
    assert check_queries(per_query, _QBASE) == []


def test_check_queries_counts_a_missing_query_as_zero():
    assert check_queries({"Vector": {"q0": 1.0}}, _QBASE) == [
        ("Vector", "lq1", 0.5, 0.0), ("Hybrid", "q0", 1.0, 0.0), ("Hybrid", "lq1", 1.0, 0.0)]


def test_check_queries_is_a_no_op_without_per_query_baselines():
    assert check_queries({}, _BASE) == []


# --- --self-test verdict (build/scoring stubbed: no model, no index) --------------

def _stub_self_test(monkeypatch, drop: dict[str, float], baseline: dict = _BASE):
    """self_test() where ablating arch X lowers X's q0 score by drop[X] (0 = unnoticed).

    Every arch has a single query "q0" scoring its baseline mean, so a drop moves the
    mean and q0 alike."""
    import contextlib

    import check_regression as cr

    active: list[str] = []

    @contextlib.contextmanager
    def ablate(arch):
        active.append(arch)
        yield
        active.pop()

    monkeypatch.setattr(cr, "build", lambda embedder=None: ({}, {}))
    monkeypatch.setattr(cr, "ablations", lambda golden: [
        (arch, f"{arch} off", lambda arch=arch: ablate(arch)) for arch in baseline["scores"]])
    monkeypatch.setattr(cr, "per_query_ndcg", lambda retrievers, golden: {
        arch: {"q0": want - (drop.get(arch, 0.0) if arch in active else 0.0)}
        for arch, want in baseline["scores"].items()})
    return cr.self_test(baseline)


def test_self_test_passes_when_every_ablation_is_caught(monkeypatch):
    assert _stub_self_test(monkeypatch, {"Vector": 0.5, "Hybrid": 0.2, "Graph": 0.1}) is True


def test_self_test_fails_when_an_ablation_goes_unnoticed(monkeypatch):
    # Disabling Graph's component costs only 0.03 < tolerance 0.05: the guard is blind to it.
    assert _stub_self_test(monkeypatch, {"Vector": 0.5, "Hybrid": 0.2, "Graph": 0.03}) is False


def test_self_test_counts_a_per_query_catch(monkeypatch):
    # Mean tolerance 0.2 misses Graph's 0.1 drop; the per-query tolerance 0.05 catches it.
    baseline = _BASE | {"tolerance": 0.2, "query_tolerance": 0.05,
                        "per_query": {a: {"q0": w} for a, w in _BASE["scores"].items()}}
    drops = {"Vector": 0.5, "Hybrid": 0.3, "Graph": 0.1}
    assert _stub_self_test(monkeypatch, drops, baseline) is True
    assert _stub_self_test(monkeypatch, drops, baseline | {"per_query": {}}) is False
