"""Plot MRR by question type × architecture (tagged toy corpus).

Reads eval/retrieval_results.json (from retrieval_eval) and writes
docs/per-category.svg. Shows each architecture's character: Vector excels on
semantics (factoid), Hybrid on exact tokens (keyword), Graph stays robust via NER.
Requires the [notebooks] extra (matplotlib).

    python -m eval.retrieval_eval --embedders all-MiniLM-L6-v2
    python -m eval.plot_categories
"""

import json
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
    data = json.loads((ROOT / "eval" / "retrieval_results.json").read_text("utf-8"))
    res = data["results"]
    emb = "all-MiniLM-L6-v2" if "all-MiniLM-L6-v2" in res else next(iter(res))
    stacks = res[emb]["stacks"]

    cats = sorted({c for rep in stacks.values() for c in rep["by_type"]})
    mrr = {_kind(name): [rep["by_type"][c]["mrr"] for c in cats] for name, rep in stacks.items()}

    x = np.arange(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, kind in enumerate(("vector", "hybrid", "graph")):
        bars = ax.bar(x + (i - 1) * width, mrr[kind], width, label=SHORT[kind], color=COLORS[kind])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS.get(c, c) for c in cats])
    ax.set_ylabel("MRR (higher = better)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Retrieval by query type — each architecture's strength\n(toy corpus, {emb})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "docs" / "per-category.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
