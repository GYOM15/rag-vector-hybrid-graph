"""Tests for the eval statistics (bootstrap CIs, paired sign-flip test) — numpy only."""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.stats import bootstrap_ci, paired_bootstrap, pairwise  # noqa: E402


def test_constant_values_collapse_the_interval():
    assert bootstrap_ci([0.7] * 20) == pytest.approx((0.7, 0.7, 0.7))
    assert bootstrap_ci([0.3]) == pytest.approx((0.3, 0.3, 0.3))


def test_mean_is_the_sample_mean_and_ci_contains_it():
    values = np.random.default_rng(42).random(200)
    mean, lo, hi = bootstrap_ci(values)
    assert mean == pytest.approx(values.mean())
    assert lo < mean < hi


def test_bernoulli_ci_matches_normal_approximation():
    # p = 0.5, n = 100: standard error 0.05 -> 95% CI ≈ 0.5 ± 0.098.
    mean, lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert mean == 0.5
    assert lo == pytest.approx(0.402, abs=0.02)
    assert hi == pytest.approx(0.598, abs=0.02)


def test_wider_confidence_gives_wider_interval():
    values = np.random.default_rng(0).random(50)
    _, lo95, hi95 = bootstrap_ci(values, alpha=0.05)
    _, lo80, hi80 = bootstrap_ci(values, alpha=0.20)
    assert lo95 < lo80 and hi80 < hi95


def test_bootstrap_is_deterministic_for_a_seed():
    values = np.random.default_rng(7).random(80)
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) != bootstrap_ci(values, seed=4)


def test_paired_identical_inputs_give_zero_diff_and_p_one():
    a = [0.1, 0.4, 0.9, 0.0, 1.0]
    res = paired_bootstrap(a, list(a))
    assert res == {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0, "n": 5}


def test_paired_symmetric_differences_are_not_significant():
    # Differences +1/-1 in equal numbers: observed mean 0 -> every flip is as extreme.
    a = [1.0, 0.0] * 20
    b = [0.0, 1.0] * 20
    assert paired_bootstrap(a, b)["p_value"] == 1.0


def test_paired_consistent_gain_is_significant():
    # b beats a by exactly 0.1 on all 30 queries: only the 2 all-same-sign flips out of
    # 2**30 are as extreme, so p hits its Monte Carlo floor 1 / (1 + n_boot).
    a = np.random.default_rng(1).random(30)
    res = paired_bootstrap(a + 0.1, a, n_boot=2_000)
    assert res["mean_diff"] == pytest.approx(0.1)
    assert res["lo"] == pytest.approx(0.1) and res["hi"] == pytest.approx(0.1)
    assert res["p_value"] == pytest.approx(1 / 2_001)


def test_paired_is_antisymmetric_and_deterministic():
    rng = np.random.default_rng(5)
    a, b = rng.random(60), rng.random(60)
    ab, ba = paired_bootstrap(a, b), paired_bootstrap(b, a)
    assert ab["mean_diff"] == pytest.approx(-ba["mean_diff"])
    assert ab["p_value"] == ba["p_value"]  # same seed -> same flips; |mean| is symmetric
    assert ab == paired_bootstrap(a, b)


def test_paired_rejects_mismatched_or_empty_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_pairwise_keys_follow_stack_order_and_sign():
    per_stack = {"Vector": [0.2, 0.4, 0.6], "Hybrid": [0.3, 0.5, 0.7], "Graph": [0.2, 0.4, 0.6]}
    pairs = pairwise(per_stack)
    assert list(pairs) == ["Hybrid-Vector", "Graph-Vector", "Graph-Hybrid"]
    assert pairs["Hybrid-Vector"]["mean_diff"] == pytest.approx(0.1)  # diff = Hybrid - Vector
    assert pairs["Graph-Hybrid"]["mean_diff"] == pytest.approx(-0.1)
    assert pairs["Graph-Vector"]["p_value"] == 1.0
