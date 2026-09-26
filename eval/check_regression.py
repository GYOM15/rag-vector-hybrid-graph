"""Anti-regression guardrail for RETRIEVAL (no LLM, deterministic).

Builds the 3 architectures on a small, FIXED "golden" corpus
(`golden_corpus.json`), measures the nDCG@5 of each, and compares it to a
committed baseline (`baselines.json`). **Fails (exit code 1)** if an architecture
drops below its baseline minus a tolerance - this is what makes it a real
guardrail in CI: a change that silently breaks retrieval makes CI fail.

A guard where every stack scores 1.0 cannot see a component break: removing
Hybrid's BM25 or Graph's entity boost would still pass. So besides a few easy
queries, the corpus holds probes that only one component solves:
- lexical probes: the answer hinges on a rare exact token (a code, a surname
  variant) among near-duplicates - dense MiniLM ranks it poorly, BM25 nails it;
- entity probes: the relevant doc is tied to the query by a named entity but is
  semantically outranked by a distractor - Graph's entity boost lifts it.
Each stack thus has its own baseline below 1.0, and `--self-test` proves the guard
is sensitive: it disables each stack's own component in turn and fails unless
`check()` flags that stack.

    python -m eval.check_regression              # checks vs baselines (CI)
    python -m eval.check_regression --self-test  # proves each component is guarded
    python -m eval.check_regression --update     # regenerates eval/baselines.json
"""

import argparse
import json
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GOLDEN = ROOT / "eval" / "golden_corpus.json"
BASELINES = ROOT / "eval" / "baselines.json"
K = 5
DEFAULT_TOLERANCE = 0.05
EMBEDDER = "all-MiniLM-L6-v2"


def _kind(name: str) -> str:
    n = name.lower()
    return "Vector" if ("vecto" in n) else ("Hybrid" if "hybr" in n else "Graph")


def build(embedder: str = EMBEDDER) -> tuple[dict, dict]:
    """Builds the 3 retrievers on the golden corpus: ({arch: retriever}, golden)."""
    from pipeline import assemble_stacks

    golden = json.loads(GOLDEN.read_text("utf-8"))
    texts = [d["text"] for d in golden["docs"]]
    metadata = [{"doc_id": d["doc_id"], "title": ""} for d in golden["docs"]]
    stacks = assemble_stacks(texts, metadata, embedder=embedder)
    return {_kind(name): rag.retriever for name, rag in stacks.items()}, golden


def per_query_ndcg(retrievers: dict, golden: dict) -> dict[str, dict[str, float]]:
    """nDCG@5 of every golden query, per architecture: {arch: {qid: ndcg}}."""
    from eval.beir_eval import _ranked_doc_ids
    from shared.ir_metrics import ndcg_at_k

    return {
        arch: {q["qid"]: ndcg_at_k(_ranked_doc_ids(retr.search(q["text"], k=K)),
                                   {r: 1.0 for r in q["relevant"]}, K)
               for q in golden["queries"]}
        for arch, retr in retrievers.items()
    }


def score(retrievers: dict, golden: dict) -> dict[str, float]:
    """Average nDCG@5 per architecture (rounded, as stored in the baselines)."""
    return {arch: round(sum(per_q.values()) / len(per_q), 4)
            for arch, per_q in per_query_ndcg(retrievers, golden).items()}


def measure(embedder: str = EMBEDDER) -> dict[str, float]:
    """Average nDCG@5 per architecture on the golden corpus (deterministic)."""
    return score(*build(embedder))


def check(scores: dict[str, float], baseline: dict) -> list[tuple]:
    """Returns the list of regressions (arch, expected, got); empty if all is well."""
    tol = baseline["tolerance"]
    failures = []
    for arch, want in baseline["scores"].items():
        got = scores.get(arch, 0.0)
        if got < want - tol:
            failures.append((arch, want, got))
    return failures


# --- Self-test: every ablation must be caught -----------------------------------

def _scrambled_query_search(self, query: str, k: int):
    """VectorRetriever._search_index with the query embedding's dimensions shuffled
    (fixed seed): a query encoded in another space than the index, as a prefix or
    model mismatch would produce."""
    import faiss
    import numpy as np

    emb = np.asarray(self.embedding_model.encode_query(query), dtype=np.float32)
    emb = emb[np.random.default_rng(0).permutation(emb.size)][None, :].copy()
    faiss.normalize_L2(emb)
    return self.indexer.index.search(emb, k)


def ablations() -> dict[str, tuple[str, Callable]]:
    """{arch: (description, patch factory)}: each patch disables only that arch's own
    component, at query time — so the stacks are built once and reused."""
    from stack1_traditional.retriever import VectorRetriever
    from stack2_hybrid.retriever import HybridRetriever
    from stack3_graphrag import retriever as graph_retriever

    return {
        "Hybrid": ("BM25 ranking -> []",
                   lambda: mock.patch.object(HybridRetriever, "_bm25_ranking",
                                             lambda self, query, n: [])),
        "Graph": ("query entities -> []",
                  lambda: mock.patch.object(graph_retriever, "extract_entities",
                                            lambda text: [])),
        "Vector": ("query embedding shuffled",
                   lambda: mock.patch.object(VectorRetriever, "_search_index",
                                             _scrambled_query_search)),
    }


def self_test(baseline: dict, embedder: str = EMBEDDER) -> bool:
    """Runs the guard intact, then under each ablation. True iff the intact run passes
    AND each ablation makes `check()` flag the ablated architecture."""
    retrievers, golden = build(embedder)
    runs = [("-", "intact", nullcontext)] + [
        (arch, desc, patch) for arch, (desc, patch) in ablations().items()]
    archs = list(baseline["scores"])

    print(f"Guard self-test · {baseline['metric']} · tolerance {baseline['tolerance']}\n")
    print(f"  {'ablated':8} {'component':26}" + "".join(f"{a:>8}" for a in archs)
          + "   flagged        verdict")
    print(f"  {'baseline':8} {'':26}" + "".join(f"{baseline['scores'][a]:8.4f}" for a in archs))
    all_ok = True
    for arch, desc, patch in runs:
        with patch():
            scores = score(retrievers, golden)
        flagged = [a for a, _, _ in check(scores, baseline)]
        if arch == "-":
            ok, verdict = not flagged, ("OK" if not flagged else "❌ intact run already fails")
        else:
            ok = arch in flagged
            verdict = "caught" if ok else "❌ MISSED"
        all_ok &= ok
        print(f"  {arch:8} {desc:26}" + "".join(f"{scores.get(a, 0.0):8.4f}" for a in archs)
              + f"   {','.join(flagged) or '-':14} {verdict}")
    print("\n✅ every ablation is caught" if all_ok
          else "\n❌ the guard is blind to at least one component")
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval anti-regression guardrail (golden corpus).")
    ap.add_argument("--update", action="store_true", help="regenerates eval/baselines.json then exits")
    ap.add_argument("--self-test", action="store_true",
                    help="proves the guard catches each stack's own component being disabled")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test(json.loads(BASELINES.read_text("utf-8"))) else 1)

    scores = measure()

    if args.update:
        payload = {"metric": f"ndcg@{K}", "tolerance": args.tolerance,
                   "embedder": EMBEDDER, "scores": scores}
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
