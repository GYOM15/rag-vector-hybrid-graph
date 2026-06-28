"""Anti-regression guardrail for RETRIEVAL (no LLM, deterministic).

Builds the 3 architectures on a tiny, FIXED "golden" corpus
(`golden_corpus.json`), measures the nDCG@5 of each, and compares it to a
committed baseline (`baselines.json`). **Fails (exit code 1)** if an architecture
drops below its baseline minus a tolerance - this is what makes it a real
guardrail in CI: a change that silently breaks retrieval makes CI fail.

    python -m eval.check_regression            # checks vs baselines (CI)
    python -m eval.check_regression --update    # regenerates eval/baselines.json
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
    """Average nDCG@5 per architecture on the golden corpus (deterministic)."""
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
    """Returns the list of regressions (arch, expected, got); empty if all is well."""
    tol = baseline["tolerance"]
    failures = []
    for arch, want in baseline["scores"].items():
        got = scores.get(arch, 0.0)
        if got < want - tol:
            failures.append((arch, want, got))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval anti-regression guardrail (golden corpus).")
    ap.add_argument("--update", action="store_true", help="regenerates eval/baselines.json then exits")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = ap.parse_args()

    scores = measure()

    if args.update:
        payload = {"metric": f"ndcg@{K}", "tolerance": args.tolerance,
                   "embedder": "all-MiniLM-L6-v2", "scores": scores}
        BASELINES.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"✅ baselines regenerated -> {BASELINES.name}: {scores}")
        return

    baseline = json.loads(BASELINES.read_text("utf-8"))
    tol = baseline["tolerance"]
    print(f"Retrieval guardrail · {baseline['metric']} · golden corpus · tolerance {tol}\n")
    print(f"  {'arch':8} {'baseline':>9} {'current':>8} {'Δ':>9}  verdict")
    for arch, want in baseline["scores"].items():
        got = scores.get(arch, 0.0)
        ok = got >= want - tol
        print(f"  {arch:8} {want:9.4f} {got:8.4f} {got - want:+9.4f}  {'OK' if ok else '❌ REGRESSION'}")

    failures = check(scores, baseline)
    if failures:
        detail = ", ".join(f"{a} {g:.4f} < {w:.4f}-{tol}" for a, w, g in failures)
        print(f"\n❌ {len(failures)} regression(s): {detail}")
        sys.exit(1)
    print("\n✅ no regression")


if __name__ == "__main__":
    main()
