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
        base = rerank = rerank_ms = 0.0
        for qid, query in queries_eval:
            results = rag.retriever.search(query, k=candidates)  # top-N large
            base += ndcg_at_k(_ranked_doc_ids(results[:10]), qrels[qid], 10)  # sans reranking
            start = time.perf_counter()
            reranked = reranker.rerank(query, results, top_k=10)
            rerank_ms += (time.perf_counter() - start) * 1000
            rerank += ndcg_at_k(_ranked_doc_ids(reranked), qrels[qid], 10)  # avec reranking
        n = len(queries_eval)
        report[_kind(name)] = {
            "ndcg_base": round(base / n, 4),
            "ndcg_rerank": round(rerank / n, 4),
            "delta": round((rerank - base) / n, 4),
            "rerank_ms_per_query": round(rerank_ms / n, 1),
        }

    payload = {"config": {"dataset": dataset, "candidates": candidates, "embedder": embedder,
                          "n_queries": len(queries_eval), "reranker": reranker.model_name},
               "stacks": report}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    spreads = (max(m["ndcg_base"] for m in report.values()) - min(m["ndcg_base"] for m in report.values()),
               max(m["ndcg_rerank"] for m in report.values()) - min(m["ndcg_rerank"] for m in report.values()))
    print(f"\n{dataset} · top-{candidates} candidats → rerank top-10 · {len(queries_eval)} requêtes\n")
    print(f"  {'archi':8} {'sans':>7} {'avec':>7} {'Δ':>7}  {'+latence':>10}")
    for k, m in report.items():
        print(f"  {k:8} {m['ndcg_base']:7.3f} {m['ndcg_rerank']:7.3f} {m['delta']:+7.3f}  "
              f"{m['rerank_ms_per_query']:8.1f}ms")
    print(f"\n  écart entre archis : sans={spreads[0]:.3f} → avec={spreads[1]:.3f} "
          f"({'égalise' if spreads[1] < spreads[0] else 'pas d''égalisation'})")
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
