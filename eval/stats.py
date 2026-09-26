"""Uncertainty for per-query evaluation scores: bootstrap CIs and paired tests.

The stacks are compared on a few hundred queries at most, and the gaps between them
are often 0.001-0.02 nDCG — well inside the query-sampling noise. A mean alone
cannot tell a real gap from luck, so every eval keeps its per-query scores and
reports, from them:

- `bootstrap_ci`: the mean and its percentile-bootstrap confidence interval
  (resampling the queries with replacement);
- `paired_bootstrap`: the mean per-query difference between two stacks, its
  bootstrap CI and a two-sided p-value from a paired **sign-flip permutation test**.
  Pairing matters: both stacks answer the *same* queries, so query difficulty cancels
  out in the differences and the test is far more sensitive than comparing two
  independent CIs.

numpy only, and **deterministic**: every function takes a `seed` (a fixed default),
so re-running an eval reproduces its intervals bit for bit.
"""

from collections.abc import Sequence
from itertools import combinations

import numpy as np

N_BOOT = 10_000
_BLOCK = 1_000  # resamples drawn per batch: bounds memory to _BLOCK × n floats


def _resampled_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means of `n_boot` resamples (with replacement) of `x`, drawn in blocks.

    One (n_boot × n) index matrix would reach gigabytes for n in the thousands;
    drawing it block by block keeps memory flat and the output identical for a seed.
    """
    means = np.empty(n_boot)
    for start in range(0, n_boot, _BLOCK):
        stop = min(start + _BLOCK, n_boot)
        idx = rng.integers(0, x.size, size=(stop - start, x.size))
        means[start:stop] = x[idx].mean(axis=1)
    return means


def _as_array(values: Sequence[float], name: str = "values") -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence of numbers")
    return x


def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOT, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi): the mean and its percentile-bootstrap (1 - alpha) CI.

    Resamples the queries with replacement `n_boot` times; the bounds are the
    alpha/2 and 1 - alpha/2 quantiles of the resampled means. With a single value
    (or constant values) the interval collapses onto the mean.
    """
    x = _as_array(values)
    means = _resampled_means(x, n_boot, np.random.default_rng(seed))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired comparison of two stacks scored on the same queries (a[i] ↔ b[i]).

    Returns {mean_diff, lo, hi, p_value, n}, where diff = a - b:
    - `lo`, `hi`: percentile-bootstrap CI of the mean per-query difference;
    - `p_value`: two-sided **paired sign-flip permutation test**. Under H0 (the two
      stacks are exchangeable on each query) the sign of each difference is a coin
      flip; p = (1 + #{|mean of flipped diffs| >= |observed mean|}) / (1 + n_boot).
      The +1 keeps p > 0 (a Monte Carlo p-value can never be exactly 0). Identical
      inputs give diff 0 and p = 1.
    """
    x, y = _as_array(a, "a"), _as_array(b, "b")
    if x.shape != y.shape:
        raise ValueError(f"paired samples differ in length: {x.size} vs {y.size}")
    d = x - y
    rng = np.random.default_rng(seed)
    means = _resampled_means(d, n_boot, rng)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])

    observed = abs(float(d.mean()))
    extreme = 0
    for start in range(0, n_boot, _BLOCK):
        signs = rng.choice((-1.0, 1.0), size=(min(_BLOCK, n_boot - start), d.size))
        # Tolerance: float sums of ± the same values can differ in the last bit.
        extreme += int(np.sum(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))
    p_value = (1 + extreme) / (1 + n_boot)
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_value": float(p_value), "n": int(d.size)}


def pairwise(per_stack: dict[str, Sequence[float]], seed: int = 0,
             digits: int = 4) -> dict[str, dict]:
    """`paired_bootstrap` for every pair of stacks, keyed "B-A" (diff = B - A).

    Pairs follow the dict order and each later stack is compared to the earlier ones
    ({Vector, Hybrid, Graph} -> "Hybrid-Vector", "Graph-Vector", "Graph-Hybrid"), so
    the first stack serves as the reference. Values are rounded for the JSON snapshot.
    """
    out = {}
    for first, second in combinations(per_stack, 2):
        res = paired_bootstrap(per_stack[second], per_stack[first], seed=seed)
        out[f"{second}-{first}"] = {k: (round(v, digits) if isinstance(v, float) else v)
                                    for k, v in res.items()}
    return out

