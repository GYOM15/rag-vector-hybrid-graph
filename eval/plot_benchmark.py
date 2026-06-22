"""Trace nDCG@10 par corpus × architecture depuis les sorties de beir_eval.

Lit eval/beir_results.json (SciFact) et eval/beir_hotpot.json (HotpotQA) et écrit
docs/benchmark-results.svg. Nécessite l'extra [notebooks] (matplotlib).

    python -m eval.plot_benchmark
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"vector": "#3b82f6", "hybrid": "#22c55e", "graph": "#a855f7"}
SHORT = {"vector": "Vectoriel", "hybrid": "Hybride", "graph": "Graphe"}
SOURCES = [("SciFact\n(single-hop)", "beir_results.json"),
           ("HotpotQA\n(multi-hop)", "beir_hotpot.json")]


def _kind(name: str) -> str:
    n = name.lower()
    return "vector" if ("vector" in n or "vecto" in n) else ("hybrid" if "hybr" in n else "graph")


def main() -> None:
    corpora, ndcg = [], {"vector": [], "hybrid": [], "graph": []}
    for label, fname in SOURCES:
        data = json.loads((ROOT / "eval" / fname).read_text("utf-8"))
        corpora.append(label)
        for sname, m in data["stacks"].items():
            ndcg[_kind(sname)].append(m["ndcg@10"])

    x = np.arange(len(corpora))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, kind in enumerate(("vector", "hybrid", "graph")):
        bars = ax.bar(x + (i - 1) * width, ndcg[kind], width, label=SHORT[kind], color=COLORS[kind])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora)
    ax.set_ylabel("nDCG@10 (plus haut = mieux)")
    ax.set_ylim(0, 1)
    ax.set_title("Récupération : nDCG@10 par corpus et architecture\n(benchmarks BEIR, jugements humains)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "docs" / "benchmark-results.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
