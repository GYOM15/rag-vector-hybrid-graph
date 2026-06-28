"""Retrieval -> answer loop: Exact-Match / F1 on HotpotQA gold answers.

For each architecture, on the **same sample**: nDCG@10 (retrieval quality) and
EM / F1 (generated answer quality), against the HotpotQA gold answers. No judge,
deterministic (the model runs at temperature 0). Lets you answer: "does better
retrieval lead to better answers?" and compare a small vs a large model
(--model).

    python -m eval.answer_eval --max-queries 100 --model llama3.2:1b
    python -m eval.answer_eval --max-queries 100 --model llama3.2:3b
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shared.answer_metrics import exact_match, f1_score  # noqa: E402
from shared.ir_metrics import ndcg_at_k  # noqa: E402
from pipeline import assemble_stacks  # noqa: E402

from eval.beir_eval import K_MAX, _ranked_doc_ids  # noqa: E402


def load_hotpot_with_answers(n_questions: int, split: str = "validation"):
    """Like the HotpotQA distractor loader, but also returns the gold answers.

    Returns (texts, metadata, queries, qrels, golds): the corpus = union of the
    paragraphs (support + distractors), qrels = support titles, golds[id] = answer.
    """
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    ds = ds.select(range(min(n_questions, len(ds))))

    corpus: dict[str, str] = {}
    queries, qrels, golds = [], {}, {}
    for ex in ds:
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            corpus.setdefault(title, " ".join(sentences).strip())
        qrels[ex["id"]] = {t: 1.0 for t in set(ex["supporting_facts"]["title"])}
        queries.append((ex["id"], ex["question"]))
        golds[ex["id"]] = ex["answer"]

    titles = list(corpus)
    texts = [corpus[t] for t in titles]
    metadata = [{"doc_id": t, "title": t} for t in titles]
    return texts, metadata, queries, qrels, golds


def run(max_queries: int, model: str | None, output: Path) -> dict:
    if model:
        os.environ["OLLAMA_MODEL"] = model

    texts, metadata, queries, qrels, golds = load_hotpot_with_answers(max_queries or 100)
    print(f"HotpotQA: {len(texts)} docs, {len(queries)} questions - indexing + generation "
          f"(model {model or os.getenv('OLLAMA_MODEL', '?')}, temperature 0)...", flush=True)
    stacks = assemble_stacks(texts, metadata)

    report = {}
    for sname, rag in stacks.items():
        agg = {"ndcg@10": 0.0, "em": 0.0, "f1": 0.0}
        for qid, qtext in queries:
            out = rag.query(qtext, k=K_MAX)
            agg["em"] += exact_match(out["answer"], golds[qid])
            agg["f1"] += f1_score(out["answer"], golds[qid])
            agg["ndcg@10"] += ndcg_at_k(_ranked_doc_ids(out["contexts"]), qrels[qid], 10)
        n = len(queries)
        report[sname] = {m: round(v / n, 4) for m, v in agg.items()} | {"n_queries": n}

    payload = {"config": {"dataset": "hotpotqa-distractor",
                          "model": model or os.getenv("OLLAMA_MODEL", "?"),
                          "n_docs": len(texts), "n_queries": len(queries)},
               "stacks": report}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nHotpotQA - gold answers · {len(queries)} questions · model {payload['config']['model']}\n")
    print(f"  {'architecture':28s} {'nDCG@10':>8s} {'EM':>7s} {'F1':>7s}")
    for sname, m in report.items():
        print(f"  {sname:28s} {m['ndcg@10']:8.3f} {m['em']:7.3f} {m['f1']:7.3f}")
    print(f"\n✅ Details written to {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="EM/F1 on HotpotQA gold answers (no judge, temperature 0).")
    ap.add_argument("--max-queries", type=int, default=100, help="number of questions (default 100)")
    ap.add_argument("--model", default=None, help="Ollama model (e.g. llama3.2:1b or llama3.2:3b)")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "answer_results.json")
    args = ap.parse_args()
    run(args.max_queries, args.model, args.output)


if __name__ == "__main__":
    main()
