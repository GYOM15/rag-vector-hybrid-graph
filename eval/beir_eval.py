"""Evaluation via relevance judgments (qrels) of the 3 architectures.

Loads a corpus + queries + qrels (BEIR via HF, or HotpotQA distractor for
multi-hop), indexes the corpus (each document = one unit), queries the 3
retrievers and scores against the human judgments: recall@k, nDCG@10, MRR. No LLM.

    python -m eval.beir_eval --dataset scifact
    python -m eval.beir_eval --dataset hotpotqa-distractor --max-queries 500
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
    """Returns (texts, metadata, queries_eval, qrels) for a BEIR dataset (HF)."""
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


def load_hotpot_distractor(n_questions: int = 500, split: str = "validation"):
    """Multi-hop corpus from HotpotQA (distractor).

    For a sample of questions, the union of their paragraphs (support +
    distractors) forms the corpus; the qrels are the "supporting facts" titles
    (the ~2 paragraphs to combine in order to answer).
    """
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    ds = ds.select(range(min(n_questions, len(ds))))

    corpus: dict[str, str] = {}
    queries_eval, qrels = [], {}
    for ex in ds:
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            corpus.setdefault(title, " ".join(sentences).strip())
        qrels[ex["id"]] = {t: 1.0 for t in set(ex["supporting_facts"]["title"])}
        queries_eval.append((ex["id"], ex["question"]))

    titles = list(corpus)
    texts = [corpus[t] for t in titles]
    metadata = [{"doc_id": t, "title": t} for t in titles]
    return texts, metadata, queries_eval, qrels


def _ranked_doc_ids(results: list[dict]) -> list[str]:
    """Document ids (deduplicated, best rank first) from the retrieved chunks."""
    seen, ranked = set(), []
    for r in results:
        doc_id = r["metadata"].get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def evaluate_corpus(name, texts, metadata, queries_eval, qrels, embedder, output) -> dict:
    """Indexes the corpus, queries the 3 retrievers, scores against the qrels."""
    from pipeline import assemble_stacks

    print(f"{name}: {len(texts)} docs, {len(queries_eval)} judged queries - indexing ({embedder})...")
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

    print(f"\n{name} - {len(queries_eval)} queries · {len(texts)} docs · embedder {embedder}\n")
    for sname, m in report.items():
        print(f"  {sname}")
        print("    " + "  ".join(f"recall@{k}={m[f'recall@{k}']:.3f}" for k in KS)
              + f"  nDCG@10={m['ndcg@10']:.3f}  MRR={m['mrr']:.3f}")
    print(f"\n✅ Details written to {output}")
    return payload


def run(dataset: str, embedder: str, max_queries: int, output: Path) -> dict:
    if dataset == "hotpotqa-distractor":
        loaded = load_hotpot_distractor(max_queries or 500)
    else:
        texts, metadata, queries_eval, qrels = load_beir(dataset)
        if max_queries:
            queries_eval = queries_eval[:max_queries]
        loaded = (texts, metadata, queries_eval, qrels)
    return evaluate_corpus(dataset, *loaded, embedder=embedder, output=output)


def main() -> None:
    ap = argparse.ArgumentParser(description="Qrels eval of the 3 architectures (recall@k, nDCG@10, MRR, no LLM).")
    ap.add_argument("--dataset", default="scifact", help="scifact, nfcorpus, ... or hotpotqa-distractor")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="limit (0 = all; for hotpotqa-distractor, sample size, default 500)")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "beir_results.json")
    args = ap.parse_args()
    run(args.dataset, args.embedder, args.max_queries, args.output)


if __name__ == "__main__":
    main()
