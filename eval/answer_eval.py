"""Boucle récupération → réponse : Exact-Match / F1 sur les réponses gold HotpotQA.

Pour chaque architecture, sur le **même échantillon** : nDCG@10 (qualité de
récupération) et EM / F1 (qualité de la réponse générée), contre les réponses
gold de HotpotQA. Sans juge, déterministe (le modèle tourne à température 0).
Permet de répondre : « une meilleure récupération donne-t-elle de meilleures
réponses ? » et de comparer un petit vs un gros modèle (--model).

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
    """Comme le chargeur HotpotQA distractor, mais renvoie aussi les réponses gold.

    Renvoie (texts, metadata, queries, qrels, golds) : le corpus = union des
    paragraphes (support + distracteurs), qrels = titres support, golds[id] = réponse.
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
    print(f"HotpotQA : {len(texts)} docs, {len(queries)} questions — indexation + génération "
          f"(modèle {model or os.getenv('OLLAMA_MODEL', '?')}, température 0)…", flush=True)
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

    print(f"\nHotpotQA — réponses gold · {len(queries)} questions · modèle {payload['config']['model']}\n")
    print(f"  {'architecture':28s} {'nDCG@10':>8s} {'EM':>7s} {'F1':>7s}")
    for sname, m in report.items():
        print(f"  {sname:28s} {m['ndcg@10']:8.3f} {m['em']:7.3f} {m['f1']:7.3f}")
    print(f"\n✅ Détails écrits dans {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="EM/F1 des réponses HotpotQA gold (sans juge, température 0).")
    ap.add_argument("--max-queries", type=int, default=100, help="nombre de questions (défaut 100)")
    ap.add_argument("--model", default=None, help="modèle Ollama (p. ex. llama3.2:1b ou llama3.2:3b)")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "answer_results.json")
    args = ap.parse_args()
    run(args.max_queries, args.model, args.output)


if __name__ == "__main__":
    main()
