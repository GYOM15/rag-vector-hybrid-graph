"""Plot nDCG@10 per corpus × architecture from the beir_eval snapshots.

Reads the committed snapshots eval/reference/beir_{scifact,hotpotqa,nfcorpus}.json (so the
committed figure can be regenerated from committed data; --scifact/--hotpotqa/--nfcorpus
override them) and writes docs/benchmark-results.svg. When a snapshot has 95% bootstrap
CIs ("ci95"), they are drawn as error bars; a snapshot without them (an older run) gets
no error bar and a "no CI" label, not a zero-width interval that would read as
"no uncertainty". Requires the [notebooks] extra (matplotlib).

    python -m eval.plot_benchmark
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
REFERENCE = ROOT / "eval" / "reference"
COLORS = {"vector": "#3b82f6", "hybrid": "#22c55e", "graph": "#a855f7"}
SHORT = {"vector": "Vector", "hybrid": "Hybrid", "graph": "Graph"}
# (option, x label, default snapshot)
SOURCES = [("scifact", "SciFact\n(single-hop)", "beir_scifact.json"),
           ("hotpotqa", "HotpotQA\n(multi-hop)", "beir_hotpotqa.json"),
           ("nfcorpus", "NFCorpus\n(medical IR)", "beir_nfcorpus.json")]


def _kind(name: str) -> str:
    n = name.lower()
    return "vector" if ("vector" in n or "vecto" in n) else ("hybrid" if "hybr" in n else "graph")


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"❌ missing snapshot {path} — run the eval, or pass its path explicitly")
    return json.loads(path.read_text("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="nDCG@10 bars per corpus and architecture.")
    for opt, label, fname in SOURCES:
        ap.add_argument(f"--{opt}", type=Path, default=REFERENCE / fname,
                        help=f"beir_eval output for {label.splitlines()[0]} (default: {fname})")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "benchmark-results.svg")
    args = ap.parse_args()

    corpora, with_ci = [], []
    ndcg = {k: [] for k in COLORS}
    err = {k: ([], []) for k in COLORS}  # (below, above) the mean, for yerr; NaN = no CI
    for opt, label, _ in SOURCES:
        data = _load(getattr(args, opt))
        stacks = data["stacks"].values()
        with_ci.append(all("ci95" in m for m in stacks))
        corpora.append(label)
        for sname, m in data["stacks"].items():
            kind, mean = _kind(sname), m["ndcg@10"]
            lo, hi = m.get("ci95", {}).get("ndcg@10", (np.nan, np.nan))
            ndcg[kind].append(mean)
            err[kind][0].append(max(0.0, mean - lo) if with_ci[-1] else np.nan)
            err[kind][1].append(max(0.0, hi - mean) if with_ci[-1] else np.nan)
    has_ci = any(with_ci)
    if has_ci:  # mixed old/new snapshots: say which corpora have no interval
        corpora = [c if ci else f"{c}\n(no CI)" for c, ci in zip(corpora, with_ci)]

    x = np.arange(len(corpora))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, kind in enumerate(("vector", "hybrid", "graph")):
        bars = ax.bar(x + (i - 1) * width, ndcg[kind], width, label=SHORT[kind], color=COLORS[kind],
                      yerr=np.array(err[kind]) if has_ci else None, capsize=3,
                      error_kw={"elinewidth": 1, "ecolor": "#334155"})
        if has_ci:  # inside the bar: above it, the label would sit on the error bar
            ax.bar_label(bars, fmt="%.3f", fontsize=7, label_type="center", rotation=90,
                         color="white")
        else:
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora)
    ax.set_ylabel("nDCG@10 (higher = better)")
    ax.set_ylim(0, 1)
    ci_note = ("; no CI in these snapshots" if not has_ci
               else "; error bars = 95% bootstrap CI over queries"
               + ("" if all(with_ci) else ", where the snapshot has one"))
    ax.set_title("Retrieval: nDCG@10 by corpus and architecture\n"
                 f"(BEIR benchmarks, human relevance judgments{ci_note})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"✅ {args.output}")


if __name__ == "__main__":
    main()
