"""Tests de la logique du garde-fou anti-régression (`check`) — sans build lourd.

On teste uniquement la comparaison baseline ↔ scores (les imports lourds de
`check_regression.measure` sont paresseux, donc importer `check` reste léger).
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
    # −0.04 reste sous la tolérance de 0.05 → toléré
    assert check({"Vector": 0.96, "Hybrid": 0.9, "Graph": 0.8}, _BASE) == []


def test_drop_beyond_tolerance_is_flagged():
    assert check({"Vector": 1.0, "Hybrid": 0.9, "Graph": 0.50}, _BASE) == [("Graph", 0.8, 0.50)]


def test_boundary_exactly_at_tolerance_passes():
    # want − tol = 0.75 ; got = 0.75 → OK (comparaison stricte <)
    assert check({"Vector": 1.0, "Hybrid": 0.9, "Graph": 0.75}, _BASE) == []


def test_missing_architecture_counts_as_zero():
    assert check({"Vector": 1.0, "Hybrid": 0.9}, _BASE) == [("Graph", 0.8, 0.0)]


def test_multiple_regressions_all_flagged():
    failures = check({"Vector": 0.5, "Hybrid": 0.9, "Graph": 0.5}, _BASE)
    assert {f[0] for f in failures} == {"Vector", "Graph"}
