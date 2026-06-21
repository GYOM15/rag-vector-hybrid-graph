"""Évaluation de la RÉCUPÉRATION seule (sans LLM).

Pour chaque architecture et chaque k, mesure si le chunk contenant la réponse
(`gold`) est récupéré, et à quel rang : **hit@k** et **MRR**. Aucun LLM requis →
déterministe, gratuit, et immunisé contre la mémoire du modèle. Agrégé globalement
et par catégorie de question (`type`).

Exemple :
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --articles 100
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import build_stacks  # noqa: E402

KS = (1, 3, 5, 8, 10)


def _first_hit_rank(retriever, question: str, gold: str, k_max: int) -> int | None:
    """Rang (1-indexé) du 1er chunk contenant `gold`, ou None si absent du top-k_max."""
    gold = gold.lower()
    for rank, ctx in enumerate(retriever.search(question, k=k_max), 1):
        if gold in ctx["text"].lower():
            return rank
    return None


def _aggregate(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in KS}
    out["mrr"] = round(sum(1.0 / r for r in ranks if r) / n, 3)
    out["n"] = n
    return out


def run(n_articles: int, output: Path) -> dict:
    data = [d for d in json.loads((ROOT / "eval" / "questions.json").read_text("utf-8")) if d.get("gold")]
    types = [d.get("type", "?") for d in data]
    k_max = max(KS)

    print(f"Construction des stacks ({n_articles} articles)…")
    stacks = build_stacks(n_articles=n_articles)

    ranks = {name: [_first_hit_rank(rag.retriever, d["question"], d["gold"], k_max) for d in data]
             for name, rag in stacks.items()}

    # Diagnostic : gold jamais récupéré par AUCUNE archi (label douteux ou chunk introuvable).
    suspects = [d["question"] for i, d in enumerate(data)
                if all(ranks[name][i] is None for name in stacks)]

    cats = sorted(set(types))
    report = {}
    for name in stacks:
        report[name] = {
            "overall": _aggregate(ranks[name]),
            "by_type": {t: _aggregate([ranks[name][i] for i, tt in enumerate(types) if tt == t])
                        for t in cats},
        }

    payload = {
        "config": {"n_articles": n_articles, "n_questions": len(data), "ks": list(KS)},
        "stacks": report,
        "unretrieved": suspects,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRécupération — {len(data)} questions · {n_articles} articles\n")
    for name, rep in report.items():
        o = rep["overall"]
        print(name)
        print("  global   " + "  ".join(f"hit@{k}={o[f'hit@{k}']:.3f}" for k in KS) + f"  MRR={o['mrr']:.3f}")
        for t, m in rep["by_type"].items():
            print(f"  {t:8} " + "  ".join(f"hit@{k}={m[f'hit@{k}']:.3f}" for k in KS)
                  + f"  MRR={m['mrr']:.3f}  (n={m['n']})")
        print()
    if suspects:
        print(f"⚠ gold jamais récupéré ({len(suspects)}) — labels à vérifier :")
        for q in suspects:
            print(f"   - {q}")
    print(f"\n✅ Détails écrits dans {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Éval récupération RAG (hit@k, MRR, sans LLM).")
    ap.add_argument("--articles", type=int, default=100)
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "retrieval_results.json")
    args = ap.parse_args()
    run(args.articles, args.output)


if __name__ == "__main__":
    main()
