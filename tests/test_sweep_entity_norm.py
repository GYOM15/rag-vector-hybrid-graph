"""Tests for the entity-norm sweep's held-out selection and JSON summary (no corpus)."""

import json
import sys
from pathlib import Path

import pytest

for module in ("numpy", "faiss", "rank_bm25", "networkx"):  # imported by the pipeline
    pytest.importorskip(module)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import sweep_entity_norm as sweep  # noqa: E402


def _grid(val: dict[str, float], test: dict[str, float]) -> dict:
    """results[label][norm] where every norm scores 0.5 unless overridden."""
    base = dict.fromkeys(sweep.NORMS, 0.5)
    return {"A-val": base | val, "B-val": base | val, "A-test": base | test}


def test_selection_uses_validation_only():
    # 'linear' wins on val, 'log' would win on test: the test split must not leak.
    results = _grid(val={"linear": 0.7}, test={"log": 0.9})
    roles = {"A-val": "val", "B-val": "val", "A-test": "test"}
    summary = sweep.summarize(results, roles)
    assert summary["chosen"] == "linear"
    assert summary["val_labels"] == ["A-val", "B-val"]
    assert summary["test_labels"] == ["A-test"]
    assert summary["mean_val_ndcg@10"]["linear"] == 0.7


def test_summary_reports_the_real_default_and_is_json_ready():
    summary = sweep.summarize(_grid(val={}, test={}), {"A-val": "val", "B-val": "val",
                                                       "A-test": "test"})
    assert summary["default"] == sweep._DEFAULT_ENTITY_NORM
    assert summary["chosen_is_default"] == (summary["chosen"] == sweep._DEFAULT_ENTITY_NORM)
    assert set(summary["ndcg@10"]["A-test"]) == set(sweep.NORMS)
    json.dumps(summary)  # serializable as is
