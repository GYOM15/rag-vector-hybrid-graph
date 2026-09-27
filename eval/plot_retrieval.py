"""Plot the retrieval comparison per embedder from a retrieval_eval snapshot.

Reads eval/reference/retrieval_results.json (--input overrides it) and writes
docs/retrieval-embedders.svg: overall MRR (bars grouped by architecture, one cluster per
embedder, with 95% bootstrap CIs when the snapshot has them) + Vector hit@k curves
(embedder effect). Requires the [notebooks] extra (matplotlib).

    python -m eval.plot_retrieval
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


def _kind(name: str) -> str:
    n = name.lower()
    if "vector" in n or "vecto" in n:
        return "vector"
    if "hybr" in n:
        return "hybrid"
    return "graph"


def main() -> None:
    ap = argparse.ArgumentParser(description="MRR per embedder × architecture + Vector hit@k.")
    ap.add_argument("--input", type=Path,
                    default=ROOT / "eval" / "reference" / "retrieval_results.json")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "retrieval-embedders.svg")
    args = ap.parse_args()
    if not args.input.exists():
        sys.exit(f"❌ missing snapshot {args.input} — run eval.retrieval_eval, or pass --input")

    data = json.loads(args.input.read_text("utf-8"))
    embedders = data["config"]["embedders"]
    ks = data["config"]["ks"]
    results = data["results"]
    stacks = list(results[embedders[0]]["stacks"])
    kinds = [_kind(s) for s in stacks]
    labels = [e.split("/")[-1] for e in embedders]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(embedders))
    width = 0.25
    overall = {s: [results[e]["stacks"][s]["overall"] for e in embedders] for s in stacks}
    has_ci = any("mrr_ci95" in o for rows in overall.values() for o in rows)
    for i, (s, k) in enumerate(zip(stacks, kinds)):
        vals = [o["mrr"] for o in overall[s]]
        bounds = [o.get("mrr_ci95", (o["mrr"], o["mrr"])) for o in overall[s]]
        yerr = np.array([[max(0.0, v - lo) for v, (lo, _) in zip(vals, bounds)],
                         [max(0.0, hi - v) for v, (_, hi) in zip(vals, bounds)]])
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=SHORT[k], color=COLORS[k],
                       yerr=yerr if has_ci else None, capsize=3,
                       error_kw={"elinewidth": 1, "ecolor": "#334155"})
        if has_ci:  # inside the bar: above it, the label would sit on the error bar
            ax1.bar_label(bars, fmt="%.2f", fontsize=7, label_type="center", rotation=90,
                          color="white")
        else:
            ax1.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("MRR (overall)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("MRR by embedder and architecture"
                  + ("\n(error bars = 95% bootstrap CI)" if has_ci else ""))
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
    fig.suptitle(f"Retrieval by embedding model ({n_q} questions, {n_art} articles)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"✅ {args.output}")


if __name__ == "__main__":
    main()
