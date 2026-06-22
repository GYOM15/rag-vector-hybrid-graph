"""Évaluation de la RÉCUPÉRATION seule (sans LLM), sur un ou plusieurs embedders.

Pour chaque embedder × architecture × k, mesure si le chunk contenant la réponse
(`gold`) est récupéré, et à quel rang : hit@k et MRR. Aucun LLM requis →
déterministe, gratuit, immunisé contre la mémoire du modèle. Agrégé globalement et
par catégorie (`type`). Sert aussi à montrer le rôle du modèle d'embeddings.

Exemple :
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --embedders all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import build_stacks  # noqa: E402

KS = (1, 3, 5, 8, 10)
DEFAULT_EMBEDDERS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]


def _first_hit_rank(retriever, question: str, gold: str, k_max: int) -> int | None:
    """Rang (1-indexé) du 1er chunk contenant `gold`, ou None si hors top-k_max."""
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


def _eval_one(stacks: dict, data: list, types: list, cats: list, k_max: int):
    ranks = {name: [_first_hit_rank(rag.retriever, d["question"], d["gold"], k_max) for d in data]
             for name, rag in stacks.items()}
    suspects = [d["question"] for i, d in enumerate(data)
                if all(ranks[name][i] is None for name in stacks)]
    report = {
        name: {
            "overall": _aggregate(ranks[name]),
            "by_type": {t: _aggregate([ranks[name][i] for i, tt in enumerate(types) if tt == t])
                        for t in cats},
        }
        for name in stacks
    }
    return report, suspects


def run(n_articles: int, output: Path, embedders: list[str]) -> dict:
    data = [d for d in json.loads((ROOT / "eval" / "questions.json").read_text("utf-8")) if d.get("gold")]
    types = [d.get("type", "?") for d in data]
    cats = sorted(set(types))
    k_max = max(KS)

    results = {}
    for emb in embedders:
        print(f"\n=== Embedder : {emb} ({n_articles} articles) ===")
        stacks = build_stacks(n_articles=n_articles, embedder=emb)
        report, suspects = _eval_one(stacks, data, types, cats, k_max)
        results[emb] = {"stacks": report, "unretrieved": suspects}
        for name, rep in report.items():
            o = rep["overall"]
            print(f"  {name}")
            print("    global   " + "  ".join(f"hit@{k}={o[f'hit@{k}']:.3f}" for k in KS) + f"  MRR={o['mrr']:.3f}")
            for t, m in rep["by_type"].items():
                print(f"    {t:8} " + "  ".join(f"hit@{k}={m[f'hit@{k}']:.3f}" for k in KS) + f"  MRR={m['mrr']:.3f}")

    payload = {
        "config": {"n_articles": n_articles, "n_questions": len(data), "ks": list(KS), "embedders": embedders},
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Comparaison transverse : MRR global par architecture et par embedder.
    print("\n=== Comparaison embedders (MRR global) ===")
    stack_names = list(next(iter(results.values()))["stacks"])
    print("  " + "architecture".ljust(34) + "".join(e[:20].ljust(22) for e in embedders))
    for name in stack_names:
        row = "  " + name.ljust(34)
        row += "".join(f"{results[emb]['stacks'][name]['overall']['mrr']:.3f}".ljust(22) for emb in embedders)
        print(row)

    print(f"\n✅ Détails écrits dans {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Éval récupération RAG multi-embedder (hit@k, MRR, sans LLM).")
    ap.add_argument("--articles", type=int, default=100)
    ap.add_argument("--embedders", nargs="+", default=DEFAULT_EMBEDDERS)
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "retrieval_results.json")
    args = ap.parse_args()
    run(args.articles, args.output, args.embedders)


if __name__ == "__main__":
    main()
