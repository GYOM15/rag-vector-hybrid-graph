"""Plot nDCG@10 per corpus × architecture from the beir_eval outputs.

Reads eval/beir_results.json (SciFact) and eval/beir_hotpot.json (HotpotQA) and writes
docs/benchmark-results.svg. Requires the [notebooks] extra (matplotlib).

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
SHORT = {"vector": "Vector", "hybrid": "Hybrid", "graph": "Graph"}
SOURCES = [("SciFact\n(single-hop)", "beir_results.json"),
           ("HotpotQA\n(multi-hop)", "beir_hotpot.json"),
           ("NFCorpus\n(medical IR)", "beir_nfcorpus.json")]


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
    ax.set_ylabel("nDCG@10 (higher = better)")
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval: nDCG@10 by corpus and architecture\n(BEIR benchmarks, human relevance judgments)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "docs" / "benchmark-results.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
