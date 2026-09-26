"""Sweep over the entity-boost normalization (graph), validated *held-out*.

We pick the normalization form on the **validation** splits
(SciFact-train, NFCorpus-dev), then report on the untouched **test** splits
(SciFact-test, HotpotQA-val, NFCorpus-test). No leakage: the test set is never
used for selection. The index/graph is built **only once** per corpus;
we merely swap the normalization function (scoring happens at query time).

The whole grid (nDCG@10 of every norm on every split), the val/test roles, the chosen
norm and the run's provenance go to a JSON snapshot (--output), so the choice of
`_DEFAULT_ENTITY_NORM` can be audited and re-checked after a retriever change.

    python -m eval.sweep_entity_norm
    python -m eval.sweep_entity_norm --output eval/reference/sweep_entity_norm.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shared.ir_metrics import ndcg_at_k  # noqa: E402
from pipeline import STACK_NAMES, assemble_stacks  # noqa: E402
from stack3_graphrag.retriever import _DEFAULT_ENTITY_NORM, _ENTITY_NORMS  # noqa: E402

from eval.beir_eval import _ranked_doc_ids, K_MAX, load_beir, load_hotpot_distractor  # noqa: E402
from eval.provenance import run_metadata  # noqa: E402

NORMS = list(_ENTITY_NORMS)  # order: none, log, p25, sqrt, p75, linear


def graph_ndcg(retr, norm_fn, queries_eval, qrels) -> float:
    """Average nDCG@10 of the graph for a given normalization (hot swap)."""
    retr._norm = norm_fn
    total = 0.0
    for qid, qtext in queries_eval:
        ranked = _ranked_doc_ids(retr.search(qtext, k=K_MAX))
        total += ndcg_at_k(ranked, qrels[qid], 10)
    return total / max(1, len(queries_eval))


def graph_retriever(texts, metadata):
    """Builds the stacks (once) and returns the graph retriever."""
    return assemble_stacks(texts, metadata)[STACK_NAMES["graph"]].retriever


def summarize(results: dict[str, dict[str, float]], roles: dict[str, str]) -> dict:
    """Held-out selection from the grid results[label][norm] (label roles: val / test).

    The norm is chosen on the mean nDCG@10 of the *val* splits only; the test splits
    are merely reported. Returns the JSON-ready part of the snapshot.
    """
    val_labels = [lbl for lbl in results if roles[lbl] == "val"]
    test_labels = [lbl for lbl in results if roles[lbl] == "test"]
    mean_val = {n: sum(results[lbl][n] for lbl in val_labels) / len(val_labels) for n in NORMS}
    best = max(NORMS, key=lambda n: mean_val[n])
    return {
        "val_labels": val_labels,
        "test_labels": test_labels,
        "ndcg@10": {lbl: {n: round(v, 4) for n, v in per_norm.items()}
                    for lbl, per_norm in results.items()},
        "mean_val_ndcg@10": {n: round(v, 4) for n, v in mean_val.items()},
        "chosen": best,
        "default": _DEFAULT_ENTITY_NORM,
        "chosen_is_default": best == _DEFAULT_ENTITY_NORM,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out sweep of the graph's entity-boost norm.")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "sweep_results.json")
    args = ap.parse_args()

    # (label, role, retriever, queries, qrels) — role ∈ {val, test}
    evals = []

    print("Loading + indexing SciFact…", flush=True)
    texts, meta, q_test, qr_test = load_beir("scifact", "test")
    _, _, q_train, qr_train = load_beir("scifact", "train")
    retr = graph_retriever(texts, meta)
    evals.append(("SciFact-train", "val", retr, q_train, qr_train))
    evals.append(("SciFact-test", "test", retr, q_test, qr_test))

    print("Loading + indexing NFCorpus…", flush=True)
    texts, meta, q_test, qr_test = load_beir("nfcorpus", "test")
    _, _, q_val, qr_val = load_beir("nfcorpus", "validation")
    retr = graph_retriever(texts, meta)
    evals.append(("NFCorpus-val", "val", retr, q_val, qr_val))
    evals.append(("NFCorpus-test", "test", retr, q_test, qr_test))

    print("Loading + indexing HotpotQA…", flush=True)
    texts, meta, q_hp, qr_hp = load_hotpot_distractor(500)
    retr = graph_retriever(texts, meta)
    evals.append(("HotpotQA-val", "test", retr, q_hp, qr_hp))

    # nDCG[label][norm]
    results: dict[str, dict[str, float]] = {}
    for label, role, retr, queries, qrels in evals:
        print(f"\n== {label} ({role}, {len(queries)} queries) ==", flush=True)
        results[label] = {}
        for norm in NORMS:
            val = graph_ndcg(retr, _ENTITY_NORMS[norm], queries, qrels)
            results[label][norm] = val
            print(f"   {norm:7s} nDCG@10 = {val:.4f}", flush=True)

    # Selection on validation only
    summary = summarize(results, {label: role for label, role, *_ in evals})
    val_labels, test_labels = summary["val_labels"], summary["test_labels"]
    mean_val, best = summary["mean_val_ndcg@10"], summary["chosen"]
    payload = {"config": {"norms": NORMS, "k": K_MAX, "graph": STACK_NAMES["graph"],
                          "n_queries": {label: len(queries) for label, _, _, queries, _ in evals},
                          "provenance": run_metadata()},
               **summary}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 64)
    print("VALIDATION (held-out) — mean nDCG@10 over", val_labels)
    for n in sorted(NORMS, key=lambda n: mean_val[n], reverse=True):
        star = "  <== chosen" if n == best else ""
        print(f"   {n:7s} {mean_val[n]:.4f}{star}")

    print(f"\nChosen norm: '{best}'  →  report on TEST (untouched):")
    header = "   " + "".join(f"{lbl:16s}" for lbl in test_labels)
    print(header)
    print("   " + "".join(f"{results[lbl][best]:<16.4f}" for lbl in test_labels))
    print(f"\n   (comparison) 'none' (no normalization) vs '{_DEFAULT_ENTITY_NORM}' "
          "(current default) vs chosen, on test:")
    for n in dict.fromkeys(("none", _DEFAULT_ENTITY_NORM, best)):
        print(f"   {n:7s} " + "".join(f"{results[lbl][n]:<16.4f}" for lbl in test_labels))
    if best != _DEFAULT_ENTITY_NORM:
        print(f"\n⚠️  the held-out choice '{best}' differs from the default "
              f"'{_DEFAULT_ENTITY_NORM}' (stack3_graphrag/retriever.py)")
    print(f"\n✅ Snapshot written to {args.output}")


if __name__ == "__main__":
    main()
