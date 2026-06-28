"""Plot the system measurements from perf_results.json.

Two panels: (1) quality (nDCG@10) × median latency Pareto front, each
point annotated with its build cost; (2) throughput (req/s) by concurrency.
Writes docs/perf-pareto.svg. Requires the [notebooks] extra (matplotlib).

    python -m eval.plot_perf
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"vector": "#3b82f6", "hybrid": "#22c55e", "graph": "#a855f7"}
SHORT = {"vector": "Vector", "hybrid": "Hybrid", "graph": "Graph"}
BUILD_KEY = {"vector": "vector_total", "hybrid": "hybrid_total", "graph": "graph_total"}


def main() -> None:
    data = json.loads((ROOT / "eval" / "perf_results.json").read_text("utf-8"))
    stacks, build = data["stacks"], data["build_seconds"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: quality × latency Pareto (ideal = top-left).
    for kind, m in stacks.items():
        x, y = m["latency"]["median_ms"], m["ndcg@10"]
        ax1.scatter(x, y, s=160, color=COLORS[kind], zorder=3, edgecolor="white", linewidth=1.5)
        ax1.annotate(f"{SHORT[kind]}\n{build[BUILD_KEY[kind]]:.0f}s build",
                     (x, y), textcoords="offset points", xytext=(10, 6), fontsize=9)
    ax1.set_xlabel("median retrieval latency (ms) — lower is better →", fontsize=9)
    ax1.set_ylabel("nDCG@10 — higher is better ↑", fontsize=9)
    ax1.set_title("Quality × latency Pareto\n(↖ ideal; label = index build time)")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)

    # Panel 2: throughput by concurrency.
    conc = sorted((int(w) for w in next(iter(stacks.values()))["throughput_qps"]))
    for kind, m in stacks.items():
        ys = [m["throughput_qps"][str(w)] for w in conc]
        ax2.plot(conc, ys, "-o", color=COLORS[kind], label=SHORT[kind])
    ax2.set_xlabel("concurrent threads", fontsize=9)
    ax2.set_ylabel("throughput (queries / second)", fontsize=9)
    ax2.set_xticks(conc)
    ax2.set_title("Throughput vs concurrency\n(only Vector scales — FAISS releases the GIL)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    cfg = data["config"]
    fig.suptitle(f"Retrieval systems profile — {cfg['dataset']}, {cfg['n_docs']} docs "
                 f"(no LLM)", fontsize=12)
    fig.tight_layout()
    out = ROOT / "docs" / "perf-pareto.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
