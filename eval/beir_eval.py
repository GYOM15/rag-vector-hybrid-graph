"""Évaluation BEIR (qrels) des 3 architectures : recall@k, nDCG@10, MRR — sans LLM.

Charge un dataset BEIR via HF datasets (corpus / queries / qrels), indexe le
corpus (chaque document = une unité), interroge les 3 retrievers et note contre
les jugements de pertinence humains. Métriques standard de la littérature IR.

    python -m eval.beir_eval --dataset scifact
    python -m eval.beir_eval --dataset scifact --max-queries 50   # smoke
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shared.ir_metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402

KS = (1, 5, 10)
K_MAX = 10


def load_beir(name: str, split: str = "test"):
    """Renvoie (texts, metadata, queries_eval, qrels) pour un dataset BEIR."""
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_rows = load_dataset(f"BeIR/{name}-qrels")[split]

    qrels: dict[str, dict] = {}
    for row in qrels_rows:
        if row["score"] > 0:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = float(row["score"])

    texts, metadata = [], []
    for doc in corpus:
        title = (doc.get("title") or "").strip()
        body = doc.get("text") or ""
        texts.append(f"{title}. {body}" if title else body)
        metadata.append({"doc_id": str(doc["_id"]), "title": title})

    query_text = {str(q["_id"]): q["text"] for q in queries}
    queries_eval = [(qid, query_text[qid]) for qid in qrels if qid in query_text]
    return texts, metadata, queries_eval, qrels


def _ranked_doc_ids(results: list[dict]) -> list[str]:
    """Ids de documents (dédupliqués, meilleur rang d'abord) depuis les chunks récupérés."""
    seen, ranked = set(), []
    for r in results:
        doc_id = r["metadata"].get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def run(name: str, embedder: str, max_queries: int, output: Path) -> dict:
    from pipeline import assemble_stacks

    texts, metadata, queries_eval, qrels = load_beir(name)
    if max_queries:
        queries_eval = queries_eval[:max_queries]
    print(f"{name} : {len(texts)} docs, {len(queries_eval)} requêtes jugées — indexation ({embedder})…")
    stacks = assemble_stacks(texts, metadata, embedder=embedder)

    report = {}
    for sname, rag in stacks.items():
        totals = {f"recall@{k}": 0.0 for k in KS} | {"ndcg@10": 0.0, "mrr": 0.0}
        for qid, qtext in queries_eval:
            ranked = _ranked_doc_ids(rag.retriever.search(qtext, k=K_MAX))
            rel = qrels[qid]
            for k in KS:
                totals[f"recall@{k}"] += recall_at_k(ranked, rel, k)
            totals["ndcg@10"] += ndcg_at_k(ranked, rel, 10)
            totals["mrr"] += reciprocal_rank(ranked, rel)
        n = len(queries_eval)
        report[sname] = {m: round(v / n, 4) for m, v in totals.items()} | {"n_queries": n}

    payload = {"config": {"dataset": name, "embedder": embedder, "n_docs": len(texts),
                          "n_queries": len(queries_eval)}, "stacks": report}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBEIR/{name} — {len(queries_eval)} requêtes · {len(texts)} docs · embedder {embedder}\n")
    for sname, m in report.items():
        print(f"  {sname}")
        print("    " + "  ".join(f"recall@{k}={m[f'recall@{k}']:.3f}" for k in KS)
              + f"  nDCG@10={m['ndcg@10']:.3f}  MRR={m['mrr']:.3f}")
    print(f"\n✅ Détails écrits dans {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Éval BEIR des 3 archis (recall@k, nDCG@10, MRR, sans LLM).")
    ap.add_argument("--dataset", default="scifact", help="nom du dataset BEIR (ex. scifact)")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--max-queries", type=int, default=0, help="limiter (0 = toutes)")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "beir_results.json")
    args = ap.parse_args()
    run(args.dataset, args.embedder, args.max_queries, args.output)


if __name__ == "__main__":
    main()
