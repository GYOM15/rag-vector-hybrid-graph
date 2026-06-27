"""Garde-fou anti-régression de la RÉCUPÉRATION (sans LLM, déterministe).

Construit les 3 architectures sur un corpus « doré » minuscule et FIXE
(`golden_corpus.json`), mesure le nDCG@5 de chacune, et le compare à une baseline
commitée (`baselines.json`). **Échoue (code de sortie 1)** si une architecture passe
sous sa baseline moins une tolérance — c'est ce qui en fait un vrai garde-fou en CI :
un changement qui casse silencieusement la récupération fait échouer la CI.

    python -m eval.check_regression            # vérifie vs baselines (CI)
    python -m eval.check_regression --update    # régénère eval/baselines.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GOLDEN = ROOT / "eval" / "golden_corpus.json"
BASELINES = ROOT / "eval" / "baselines.json"
K = 5
DEFAULT_TOLERANCE = 0.05


def _kind(name: str) -> str:
    n = name.lower()
    return "Vector" if ("vecto" in n) else ("Hybrid" if "hybr" in n else "Graph")


def measure(embedder: str = "all-MiniLM-L6-v2") -> dict[str, float]:
    """nDCG@5 moyen par architecture sur le corpus doré (déterministe)."""
    from shared.ir_metrics import ndcg_at_k
    from pipeline import assemble_stacks

    from eval.beir_eval import _ranked_doc_ids

    golden = json.loads(GOLDEN.read_text("utf-8"))
    texts = [d["text"] for d in golden["docs"]]
    metadata = [{"doc_id": d["doc_id"], "title": ""} for d in golden["docs"]]
    qrels = {q["qid"]: {r: 1.0 for r in q["relevant"]} for q in golden["queries"]}

    stacks = assemble_stacks(texts, metadata, embedder=embedder)
    scores = {}
    for name, rag in stacks.items():
        total = sum(
            ndcg_at_k(_ranked_doc_ids(rag.retriever.search(q["text"], k=K)), qrels[q["qid"]], K)
            for q in golden["queries"]
        )
        scores[_kind(name)] = round(total / len(golden["queries"]), 4)
    return scores


def check(scores: dict[str, float], baseline: dict) -> list[tuple]:
    """Renvoie la liste des régressions (archi, attendu, obtenu) ; vide si tout va bien."""
    tol = baseline["tolerance"]
    failures = []
    for arch, want in baseline["scores"].items():
        got = scores.get(arch, 0.0)
        if got < want - tol:
            failures.append((arch, want, got))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Garde-fou anti-régression récupération (corpus doré).")
    ap.add_argument("--update", action="store_true", help="régénère eval/baselines.json puis quitte")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = ap.parse_args()

    scores = measure()

    if args.update:
        payload = {"metric": f"ndcg@{K}", "tolerance": args.tolerance,
                   "embedder": "all-MiniLM-L6-v2", "scores": scores}
        BASELINES.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"✅ baselines régénérées → {BASELINES.name} : {scores}")
        return

    baseline = json.loads(BASELINES.read_text("utf-8"))
    tol = baseline["tolerance"]
    print(f"Garde-fou récupération · {baseline['metric']} · corpus doré · tolérance {tol}\n")
    print(f"  {'archi':8} {'baseline':>9} {'actuel':>8} {'Δ':>9}  verdict")
    for arch, want in baseline["scores"].items():
        got = scores.get(arch, 0.0)
        ok = got >= want - tol
        print(f"  {arch:8} {want:9.4f} {got:8.4f} {got - want:+9.4f}  {'OK' if ok else '❌ RÉGRESSION'}")

    failures = check(scores, baseline)
    if failures:
        detail = ", ".join(f"{a} {g:.4f} < {w:.4f}-{tol}" for a, w, g in failures)
        print(f"\n❌ {len(failures)} régression(s) : {detail}")
        sys.exit(1)
    print("\n✅ aucune régression")


if __name__ == "__main__":
    main()
