"""Plot the retrieval comparison per embedder from retrieval_results.json.

Generates docs/retrieval-embedders.svg: overall MRR (bars grouped by architecture,
one cluster per embedder) + Vector hit@k curves (embedder effect).
Requires the [notebooks] extra (matplotlib).

    python -m eval.plot_retrieval
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


def _kind(name: str) -> str:
    n = name.lower()
    if "vector" in n or "vecto" in n:
        return "vector"
    if "hybr" in n:
        return "hybrid"
    return "graph"


def main() -> None:
    data = json.loads((ROOT / "eval" / "retrieval_results.json").read_text("utf-8"))
    embedders = data["config"]["embedders"]
    ks = data["config"]["ks"]
    results = data["results"]
    stacks = list(results[embedders[0]]["stacks"])
    kinds = [_kind(s) for s in stacks]
    labels = [e.split("/")[-1] for e in embedders]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(embedders))
    width = 0.25
    for i, (s, k) in enumerate(zip(stacks, kinds)):
        vals = [results[e]["stacks"][s]["overall"]["mrr"] for e in embedders]
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=SHORT[k], color=COLORS[k])
        ax1.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("MRR (overall)")
    ax1.set_ylim(0, 1)
    ax1.set_title("MRR by embedder and architecture")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    vec = stacks[kinds.index("vector")]
    styles = ["-o", "--s", ":^"]
    for e, label, style in zip(embedders, labels, styles):
        ys = [results[e]["stacks"][vec]["overall"][f"hit@{k}"] for k in ks]
        ax2.plot(ks, ys, style, label=label)
    ax2.set_xlabel("k")
    ax2.set_ylabel("hit@k")
    ax2.set_ylim(0, 1.02)
    ax2.set_xticks(ks)
    ax2.set_title("Vector: hit@k by embedder")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    n_q = data["config"].get("n_questions", "?")
    n_art = data["config"].get("n_articles", "?")
    fig.suptitle(f"The embedding model changes retrieval ({n_q} questions, {n_art} articles)",
                 fontsize=12)
    fig.tight_layout()
    out = ROOT / "docs" / "retrieval-embedders.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
