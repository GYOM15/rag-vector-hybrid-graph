"""Reranking: quality **with vs without**, and latency cost — on a BEIR dataset.

For each architecture: we retrieve a large top-N, then score nDCG@10 (a) as is
and (b) after cross-encoder reranking down to the top-10. Answers two questions:
does reranking **improve** each architecture, and does it **even out** the gaps
between them? — and at what **latency cost** (the cross-encoder over N candidates).

    python -m eval.rerank_eval --dataset scifact --candidates 50
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

from eval.beir_eval import _ranked_doc_ids, load_beir, load_hotpot_distractor  # noqa: E402


def _kind(name: str) -> str:
    n = name.lower()
    return "Vector" if ("vecto" in n) else ("Hybrid" if "hybr" in n else "Graph")


def run(dataset: str, candidates: int, max_queries: int, embedder: str, output: Path,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> dict:
    if dataset == "hotpotqa-distractor":
        texts, metadata, queries_eval, qrels = load_hotpot_distractor(max_queries or 500)
    else:
        texts, metadata, queries_eval, qrels = load_beir(dataset)
        if max_queries:
            queries_eval = queries_eval[:max_queries]

    print(f"{dataset}: {len(texts)} docs, {len(queries_eval)} queries — indexing…", flush=True)
    stacks = assemble_stacks(texts, metadata, embedder=embedder)
    reranker = CrossEncoderReranker(reranker_model)

    report = {}
    for name, rag in stacks.items():
        base = repl = fus_sum = rerank_ms = 0.0
        for qid, query in queries_eval:
            results = rag.retriever.search(query, k=candidates)  # large top-N
            base += ndcg_at_k(_ranked_doc_ids(results[:10]), qrels[qid], 10)  # without reranking
            start = time.perf_counter()
            replaced = reranker.rerank(query, results, top_k=10, mode="replace")
            rerank_ms += (time.perf_counter() - start) * 1000
            repl += ndcg_at_k(_ranked_doc_ids(replaced), qrels[qid], 10)  # cross-encoder only
            fused = reranker.rerank(query, results, top_k=10, mode="fusion")
            fus_sum += ndcg_at_k(_ranked_doc_ids(fused), qrels[qid], 10)  # RRF base + cross-encoder
        n = len(queries_eval)
        report[_kind(name)] = {
            "ndcg_base": round(base / n, 4),
            "ndcg_replace": round(repl / n, 4),
            "ndcg_fusion": round(fus_sum / n, 4),
            "delta_replace": round((repl - base) / n, 4),
            "delta_fusion": round((fus_sum - base) / n, 4),
            "rerank_ms_per_query": round(rerank_ms / n, 1),
        }

    payload = {"config": {"dataset": dataset, "candidates": candidates, "embedder": embedder,
                          "n_queries": len(queries_eval), "reranker": reranker.model_name},
               "stacks": report}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def spread(key: str) -> float:
        vals = [m[key] for m in report.values()]
        return max(vals) - min(vals)

    print(f"\n{dataset} · top-{candidates} → top-10 · {len(queries_eval)} queries (fusion = RRF k=60)\n")
    print(f"  {'arch':8} {'without':>7} {'replace':>9} {'fusion':>8}")
    for k, m in report.items():
        print(f"  {k:8} {m['ndcg_base']:7.3f} {m['ndcg_replace']:9.3f} {m['ndcg_fusion']:8.3f}")
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
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="cross-encoder model")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "rerank_results.json")
    args = ap.parse_args()
    run(args.dataset, args.candidates, args.max_queries, args.embedder, args.output, args.reranker)


if __name__ == "__main__":
    main()
