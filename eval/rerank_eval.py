"""Reranking: quality **with vs without**, and latency cost — on a BEIR dataset.

For each architecture: we retrieve a large top-N, then score nDCG@10 (a) as is
and (b) after cross-encoder reranking down to the top-10. Answers two questions:
does reranking **improve** each architecture, and does it **even out** the gaps
between them? — and at what **latency cost** (the cross-encoder over N candidates).

The gains are small next to the query-sampling noise, so each stack also keeps its
per-query nDCG@10 (without / replace / fusion), 95% bootstrap CIs, and a paired test
of each reranking delta against the same queries without reranking (eval/stats.py).

On HotpotQA the corpus is the union of the paragraphs of the sampled questions, so
its size follows --max-queries (unlike beir_eval's fixed 500-question corpus) unless
--corpus-questions is set; `config.corpus` / `config.n_docs` record which one was used.

    python -m eval.rerank_eval --dataset scifact --candidates 50
    python -m eval.rerank_eval --dataset hotpotqa-distractor \
        --max-queries 100 --corpus-questions 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shared.ir_metrics import ndcg_at_k  # noqa: E402
from shared.reranker import CrossEncoderReranker  # noqa: E402
from pipeline import assemble_stacks  # noqa: E402

from eval.beir_eval import (  # noqa: E402
    _ranked_doc_ids, corpus_description, load_beir, load_hotpot_distractor,
)
from eval.provenance import finish_metadata, run_metadata  # noqa: E402
from eval.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

VARIANTS = ("base", "replace", "fusion")  # without reranking / cross-encoder only / RRF


def _kind(name: str) -> str:
    n = name.lower()
    return "Vector" if ("vecto" in n) else ("Hybrid" if "hybr" in n else "Graph")


def _rounded(res: dict, digits: int = 4) -> dict:
    return {k: (round(v, digits) if isinstance(v, float) else v) for k, v in res.items()}


def _summary(per_query: dict[str, dict[str, float]], rerank_ms: float) -> dict:
    """Means (same keys as before) + CIs + paired tests of each delta vs no reranking."""
    n = len(per_query)
    cols = {v: [q[v] for q in per_query.values()] for v in VARIANTS}
    mean = {v: sum(vals) / n for v, vals in cols.items()}
    ci95 = {}
    for v in VARIANTS:
        _, lo, hi = bootstrap_ci(cols[v])
        ci95[f"ndcg_{v}"] = [round(lo, 4), round(hi, 4)]
    return {
        "ndcg_base": round(mean["base"], 4),
        "ndcg_replace": round(mean["replace"], 4),
        "ndcg_fusion": round(mean["fusion"], 4),
        "delta_replace": round(mean["replace"] - mean["base"], 4),
        "delta_fusion": round(mean["fusion"] - mean["base"], 4),
        "rerank_ms_per_query": round(rerank_ms / n, 1),
        "n_queries": n,
        "ci95": ci95,
        # diff = reranked - base on the same queries (bootstrap CI + sign-flip p-value)
        "paired": {"delta_replace": _rounded(paired_bootstrap(cols["replace"], cols["base"])),
                   "delta_fusion": _rounded(paired_bootstrap(cols["fusion"], cols["base"]))},
        "per_query": {qid: {v: round(q[v], 4) for v in VARIANTS} for qid, q in per_query.items()},
    }


def run(dataset: str, candidates: int, max_queries: int, embedder: str, output: Path,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        corpus_questions: int = 0) -> dict:
    provenance = run_metadata()  # the code that runs is the code at the start
    if dataset == "hotpotqa-distractor":
        n_corpus = corpus_questions or max_queries or 500
        texts, metadata, queries_eval, qrels = load_hotpot_distractor(n_corpus)
        if max_queries:
            queries_eval = queries_eval[:max_queries]
        corpus = corpus_description(dataset, n_corpus)
    else:
        texts, metadata, queries_eval, qrels = load_beir(dataset)
        if max_queries:
            queries_eval = queries_eval[:max_queries]
        corpus = corpus_description(dataset)

    print(f"{dataset}: {len(texts)} docs, {len(queries_eval)} queries — indexing…", flush=True)
    stacks = assemble_stacks(texts, metadata, embedder=embedder)
    reranker = CrossEncoderReranker(reranker_model)
    # Warm-up: load (or download) the cross-encoder and run a first prediction before
    # any timing — otherwise that one-off cost is charged to the first stack's latency.
    reranker.rerank("warm-up", [{"text": "warm-up"}], top_k=1)

    report = {}
    for name, rag in stacks.items():
        per_query, rerank_ms = {}, 0.0
        for qid, query in queries_eval:
            results = rag.retriever.search(query, k=candidates)  # large top-N
            start = time.perf_counter()
            replaced = reranker.rerank(query, results, top_k=10, mode="replace")
            rerank_ms += (time.perf_counter() - start) * 1000
            fused = reranker.rerank(query, results, top_k=10, mode="fusion")
            per_query[qid] = {
                "base": ndcg_at_k(_ranked_doc_ids(results[:10]), qrels[qid], 10),  # no reranking
                "replace": ndcg_at_k(_ranked_doc_ids(replaced), qrels[qid], 10),  # cross-encoder
                "fusion": ndcg_at_k(_ranked_doc_ids(fused), qrels[qid], 10),  # RRF base + CE
            }
        report[_kind(name)] = _summary(per_query, rerank_ms)

    payload = {"config": {"dataset": dataset, "candidates": candidates, "embedder": embedder,
                          "n_queries": len(queries_eval), "n_docs": len(texts), "corpus": corpus,
                          "reranker": reranker.model_name,
                          "ci": "95% percentile bootstrap over queries (10k resamples, seed 0)",
                          "provenance": finish_metadata(provenance)},
               "stacks": report}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def spread(key: str) -> float:
        vals = [m[key] for m in report.values()]
        return max(vals) - min(vals)

    print(f"\n{dataset} · top-{candidates} → top-10 · {len(queries_eval)} queries (fusion = RRF k=60)\n")
    print(f"  {'arch':8} {'without':>7} {'replace':>9} {'fusion':>8}")
    for k, m in report.items():
        print(f"  {k:8} {m['ndcg_base']:7.3f} {m['ndcg_replace']:9.3f} {m['ndcg_fusion']:8.3f}")
    print("\n  paired Δ vs without (95% CI, sign-flip p)")
    for k, m in report.items():
        print(f"  {k:8} " + "  ".join(
            f"{d[6:]}={p['mean_diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] p={p['p_value']:.3f}"
            for d, p in m["paired"].items()))
    print(f"\n  gap between archs: without={spread('ndcg_base'):.3f}  "
          f"replace={spread('ndcg_replace'):.3f}  fusion={spread('ndcg_fusion'):.3f}")
    print(f"  reranking latency: ~{sum(m['rerank_ms_per_query'] for m in report.values()) / len(report):.0f} ms/query (CPU)")
    print(f"\n✅ Details written to {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-encoder reranking: quality with/without + latency (BEIR).")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--candidates", type=int, default=50, help="size of the top-N retrieved before reranking")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--corpus-questions", type=int, default=0,
                    help="hotpotqa-distractor only: questions whose paragraphs form the corpus "
                         "(0 = same as --max-queries, or 500; beir_eval uses 500)")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="cross-encoder model")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "rerank_results.json")
    args = ap.parse_args()
    run(args.dataset, args.candidates, args.max_queries, args.embedder, args.output, args.reranker,
        args.corpus_questions)


if __name__ == "__main__":
    main()
