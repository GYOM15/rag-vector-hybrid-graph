"""Tests for the anti-regression guardrail logic (`check`) — no heavy build.

We only test the baseline ↔ scores comparison (the heavy imports in
`check_regression.measure` are lazy, so importing `check` stays lightweight).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from check_regression import check  # noqa: E402

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


# --- --self-test verdict (build/score stubbed: no model, no index) ---------------

def _stub_self_test(monkeypatch, drop: dict[str, float]):
    """self_test() where ablating arch X lowers X's score by drop[X] (0 = unnoticed)."""
    import contextlib

    import check_regression as cr

    active: list[str] = []

    @contextlib.contextmanager
    def ablate(arch):
        active.append(arch)
        yield
        active.pop()

    monkeypatch.setattr(cr, "build", lambda embedder=None: ({}, {}))
    monkeypatch.setattr(cr, "ablations", lambda: {
        arch: (f"{arch} off", lambda arch=arch: ablate(arch)) for arch in _BASE["scores"]})
    monkeypatch.setattr(cr, "score", lambda retrievers, golden: {
        arch: want - (drop.get(arch, 0.0) if arch in active else 0.0)
        for arch, want in _BASE["scores"].items()})
    return cr.self_test(_BASE)


def test_self_test_passes_when_every_ablation_is_caught(monkeypatch):
    assert _stub_self_test(monkeypatch, {"Vector": 0.5, "Hybrid": 0.2, "Graph": 0.1}) is True


def test_self_test_fails_when_an_ablation_goes_unnoticed(monkeypatch):
    # Disabling Graph's component costs only 0.03 < tolerance 0.05: the guard is blind to it.
    assert _stub_self_test(monkeypatch, {"Vector": 0.5, "Hybrid": 0.2, "Graph": 0.03}) is False
