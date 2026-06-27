"""Reranking : qualité **avec vs sans**, et coût en latence — sur un dataset BEIR.

Pour chaque architecture : on récupère un top-N large, on note nDCG@10 (a) tel quel
et (b) après reranking cross-encoder vers le top-10. Répond à deux questions :
le reranking **améliore-t-il** chaque archi, et **égalise-t-il** les écarts entre
elles ? — et à quel **coût en latence** (le cross-encoder sur N candidats).

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

from eval.beir_eval import _ranked_doc_ids, load_beir  # noqa: E402


def _kind(name: str) -> str:
    n = name.lower()
    return "Vector" if ("vecto" in n) else ("Hybrid" if "hybr" in n else "Graph")


def run(dataset: str, candidates: int, max_queries: int, embedder: str, output: Path) -> dict:
    texts, metadata, queries_eval, qrels = load_beir(dataset)
    if max_queries:
        queries_eval = queries_eval[:max_queries]

    print(f"{dataset} : {len(texts)} docs, {len(queries_eval)} requêtes — indexation…", flush=True)
    stacks = assemble_stacks(texts, metadata, embedder=embedder)
    reranker = CrossEncoderReranker()

    report = {}
    for name, rag in stacks.items():
        base = repl = fus_sum = rerank_ms = 0.0
        for qid, query in queries_eval:
            results = rag.retriever.search(query, k=candidates)  # top-N large
            base += ndcg_at_k(_ranked_doc_ids(results[:10]), qrels[qid], 10)  # sans reranking
            start = time.perf_counter()
            replaced = reranker.rerank(query, results, top_k=10, mode="replace")
            rerank_ms += (time.perf_counter() - start) * 1000
            repl += ndcg_at_k(_ranked_doc_ids(replaced), qrels[qid], 10)  # cross-encoder seul
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

    print(f"\n{dataset} · top-{candidates} → top-10 · {len(queries_eval)} requêtes (fusion = RRF k=60)\n")
    print(f"  {'archi':8} {'sans':>7} {'remplace':>9} {'fusion':>8}")
    for k, m in report.items():
        print(f"  {k:8} {m['ndcg_base']:7.3f} {m['ndcg_replace']:9.3f} {m['ndcg_fusion']:8.3f}")
    print(f"\n  écart entre archis : sans={spread('ndcg_base'):.3f}  "
          f"remplace={spread('ndcg_replace'):.3f}  fusion={spread('ndcg_fusion'):.3f}")
    print(f"  latence reranking : ~{sum(m['rerank_ms_per_query'] for m in report.values()) / len(report):.0f} ms/req (CPU)")
    print(f"\n✅ Détails écrits dans {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Reranking cross-encoder : qualité avec/sans + latence (BEIR).")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--candidates", type=int, default=50, help="taille du top-N récupéré avant reranking")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "rerank_results.json")
    args = ap.parse_args()
    run(args.dataset, args.candidates, args.max_queries, args.embedder, args.output)


if __name__ == "__main__":
    main()
