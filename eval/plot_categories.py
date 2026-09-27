"""Plot MRR by question type × architecture (tagged toy corpus).

Reads the retrieval_eval snapshot eval/reference/retrieval_results.json (--input
overrides it) and writes docs/per-category.svg: how each architecture fares on
semantic (factoid) vs exact-token (keyword) questions. When the snapshot has 95%
bootstrap CIs ("mrr_ci95"), they are drawn as error bars — with ~11-16 questions per
type they are wide. Requires the [notebooks] extra (matplotlib).

    python -m eval.retrieval_eval --embedders all-MiniLM-L6-v2 \\
        --output eval/reference/retrieval_results.json
    python -m eval.plot_categories
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"vector": "#3b82f6", "hybrid": "#22c55e", "graph": "#a855f7"}
SHORT = {"vector": "Vector", "hybrid": "Hybrid", "graph": "Graph"}
CAT_LABELS = {"factoid": "factoid\n(semantic)", "keyword": "keyword\n(exact token)",
              "multi": "multi\n(aggregation)"}


def _kind(name: str) -> str:
    n = name.lower()
    return "vector" if ("vector" in n or "vecto" in n) else ("hybrid" if "hybr" in n else "graph")


def main() -> None:
    ap = argparse.ArgumentParser(description="MRR by question type and architecture.")
    ap.add_argument("--input", type=Path,
                    default=ROOT / "eval" / "reference" / "retrieval_results.json")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2",
                    help="embedder to plot (falls back to the first one in the snapshot)")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "per-category.svg")
    args = ap.parse_args()
    if not args.input.exists():
        sys.exit(f"❌ missing snapshot {args.input} — run eval.retrieval_eval, or pass --input")

    data = json.loads(args.input.read_text("utf-8"))
    res = data["results"]
    emb = args.embedder if args.embedder in res else next(iter(res))
    stacks = res[emb]["stacks"]

    cats = sorted({c for rep in stacks.values() for c in rep["by_type"]})
    mrr, err = {}, {}
    for name, rep in stacks.items():
        per_cat = [rep["by_type"][c] for c in cats]
        mrr[_kind(name)] = [m["mrr"] for m in per_cat]
        bounds = [m.get("mrr_ci95", (m["mrr"], m["mrr"])) for m in per_cat]
        err[_kind(name)] = np.array(
            [[max(0.0, m["mrr"] - lo) for m, (lo, _) in zip(per_cat, bounds)],
             [max(0.0, hi - m["mrr"]) for m, (_, hi) in zip(per_cat, bounds)]])
    has_ci = any(e.any() for e in err.values())

    x = np.arange(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, kind in enumerate(("vector", "hybrid", "graph")):
        bars = ax.bar(x + (i - 1) * width, mrr[kind], width, label=SHORT[kind], color=COLORS[kind],
                      yerr=err[kind] if has_ci else None, capsize=3,
                      error_kw={"elinewidth": 1, "ecolor": "#334155"})
        if has_ci:  # inside the bar: above it, the label would sit on the error bar
            ax.bar_label(bars, fmt="%.3f", fontsize=7, label_type="center", rotation=90,
                         color="white")
        else:
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    n_per_cat = {c: next(iter(stacks.values()))["by_type"][c].get("n") for c in cats}
    ax.set_xticklabels([CAT_LABELS.get(c, c) + (f"\nn={n_per_cat[c]}" if n_per_cat[c] else "")
                        for c in cats])
    ax.set_ylabel("MRR (higher = better)")
    ax.set_ylim(0, 1.05)
    ci_note = "; error bars = 95% bootstrap CI" if has_ci else ""
    ax.set_title(f"Retrieval MRR by query type\n(toy corpus, {emb}{ci_note})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"✅ {args.output}")


if __name__ == "__main__":
    main()
