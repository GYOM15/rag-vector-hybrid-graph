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
is sensitive: it disables each stack's own component in turn and fails unless the
guard flags that stack.

The mean over 18 queries absorbs a *partial* breakage: one query slipping from rank
1 to 2 costs 0.37 / 18 = 0.02, one probe losing BM25 or its entities 0.5 / 18 = 0.03,
both under the tolerance. So the baseline also stores every query's nDCG@5, and a
query dropping by more than `query_tolerance` fails too (`check_queries`). The guard is
deterministic and its margins are wide (a 3e-3 per-dimension noise on the query
embedding changes no per-query score), so this strictness is not a flakiness risk;
the self-test includes single-query ablations that only this check catches.

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
# Any rank loss of a relevant doc in the top 3 costs >= 0.069 (rank 3 -> 4).
DEFAULT_QUERY_TOLERANCE = 0.05
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


def mean_scores(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average nDCG@5 per architecture (rounded, as stored in the baselines)."""
    return {arch: round(sum(per_q.values()) / len(per_q), 4)
            for arch, per_q in per_query.items()}


def score(retrievers: dict, golden: dict) -> dict[str, float]:
    """Average nDCG@5 per architecture (rounded, as stored in the baselines)."""
    return mean_scores(per_query_ndcg(retrievers, golden))


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


def check_queries(per_query: dict[str, dict[str, float]], baseline: dict) -> list[tuple]:
    """Per-query regressions (arch, qid, expected, got): a golden query whose nDCG@5
    fell by more than `query_tolerance`. Empty for a baseline without per-query scores."""
    tol = baseline.get("query_tolerance", baseline["tolerance"])
    failures = []
    for arch, wants in baseline.get("per_query", {}).items():
        for qid, want in wants.items():
            got = per_query.get(arch, {}).get(qid, 0.0)
            if got < want - tol:
                failures.append((arch, qid, want, got))
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


def ablations(golden: dict) -> list[tuple[str, str, Callable]]:
    """[(arch, description, patch factory)]: each patch breaks only that arch's own
    component, at query time — so the stacks are built once and reused.

    The whole-component ablations move the arch's mean; the single-query ones (one
    lexical probe loses BM25, one entity probe its entities, one easy query slips from
    rank 1 to 2) are the partial breakages the mean absorbs, caught per query.
    """
    from stack1_traditional.retriever import VectorRetriever
    from stack2_hybrid.retriever import HybridRetriever
    from stack3_graphrag import retriever as graph_retriever

    first: dict[str, dict] = {}
    for q in golden["queries"]:
        first.setdefault(q["kind"], q)
    lex, ent, easy = first["lexical"], first["entity"], first["easy"]
    bm25, entities = HybridRetriever._bm25_ranking, graph_retriever.extract_entities
    vector_search = VectorRetriever.search

    def top2_swapped(self, query: str, k: int = 5) -> list[dict]:
        hits = vector_search(self, query, k)
        return [hits[1], hits[0], *hits[2:]] if query == easy["text"] and len(hits) > 1 else hits

    return [
        ("Hybrid", "BM25 ranking -> []",
         lambda: mock.patch.object(HybridRetriever, "_bm25_ranking", lambda self, query, n: [])),
        ("Graph", "query entities -> []",
         lambda: mock.patch.object(graph_retriever, "extract_entities", lambda text: [])),
        ("Vector", "query embedding shuffled",
         lambda: mock.patch.object(VectorRetriever, "_search_index", _scrambled_query_search)),
        ("Hybrid", f"no BM25 for {lex['qid']} only",
         lambda: mock.patch.object(
             HybridRetriever, "_bm25_ranking",
             lambda self, query, n: [] if query == lex["text"] else bm25(self, query, n))),
        ("Graph", f"no entities for {ent['qid']} only",
         lambda: mock.patch.object(
             graph_retriever, "extract_entities",
             lambda text: [] if text == ent["text"] else entities(text))),
        ("Vector", f"{easy['qid']}: rank 1 <-> 2",
         lambda: mock.patch.object(VectorRetriever, "search", top2_swapped)),
    ]


def _flags(per_query: dict[str, dict[str, float]], baseline: dict) -> dict[str, str]:
    """{arch: how it was flagged} — "mean" when its average fell, else the qids that did."""
    flags = {a: "mean" for a, _, _ in check(mean_scores(per_query), baseline)}
    for arch, qid, _, _ in check_queries(per_query, baseline):
        if flags.get(arch) != "mean":
            flags[arch] = f"{flags[arch]},{qid}" if arch in flags else qid
    return flags


def self_test(baseline: dict, embedder: str = EMBEDDER) -> bool:
    """Runs the guard intact, then under each ablation. True iff the intact run passes
    AND each ablation makes the guard (mean or per-query check) flag the ablated arch."""
    retrievers, golden = build(embedder)
    runs = [("-", "intact", nullcontext)] + ablations(golden)
    archs = list(baseline["scores"])

    print(f"Guard self-test · {baseline['metric']} · tolerance {baseline['tolerance']} "
          f"(mean), {baseline.get('query_tolerance', baseline['tolerance'])} (per query)\n")
    print(f"  {'ablated':8} {'component':26}" + "".join(f"{a:>8}" for a in archs)
          + "   flagged (by)          verdict")
    print(f"  {'baseline':8} {'':26}" + "".join(f"{baseline['scores'][a]:8.4f}" for a in archs))
    all_ok = True
    for arch, desc, patch in runs:
        with patch():
            per_query = per_query_ndcg(retrievers, golden)
        scores, flags = mean_scores(per_query), _flags(per_query, baseline)
        if arch == "-":
            ok, verdict = not flags, ("OK" if not flags else "❌ intact run already fails")
        else:
            ok = arch in flags
            verdict = "caught" if ok else "❌ MISSED"
        all_ok &= ok
        flagged = ", ".join(f"{a} ({by})" for a, by in flags.items()) or "-"
        print(f"  {arch:8} {desc:26}" + "".join(f"{scores.get(a, 0.0):8.4f}" for a in archs)
              + f"   {flagged:21} {verdict}")
    print("\n✅ every ablation is caught" if all_ok
          else "\n❌ the guard is blind to at least one component")
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval anti-regression guardrail (golden corpus).")
    ap.add_argument("--update", action="store_true", help="regenerates eval/baselines.json then exits")
    ap.add_argument("--self-test", action="store_true",
                    help="proves the guard catches each stack's own component being disabled")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--query-tolerance", type=float, default=DEFAULT_QUERY_TOLERANCE,
                    help="allowed nDCG@5 drop of any single golden query (with --update)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test(json.loads(BASELINES.read_text("utf-8"))) else 1)

    per_query = per_query_ndcg(*build())
    scores = mean_scores(per_query)

    if args.update:
        payload = {"metric": f"ndcg@{K}", "tolerance": args.tolerance,
                   "query_tolerance": args.query_tolerance, "embedder": EMBEDDER,
                   "scores": scores,
                   "per_query": {arch: {qid: round(v, 4) for qid, v in per_q.items()}
                                 for arch, per_q in per_query.items()}}
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
    query_failures = check_queries(per_query, baseline)
    if failures:
        detail = ", ".join(f"{a} {g:.4f} < {w:.4f}-{tol}" for a, w, g in failures)
        print(f"\n❌ {len(failures)} regression(s): {detail}")
    if query_failures:
        qtol = baseline.get("query_tolerance", tol)
        print(f"\n❌ {len(query_failures)} golden query(ies) dropped by more than {qtol}:")
        for arch, qid, want, got in query_failures:
            print(f"    {arch:8} {qid:5} {want:.4f} -> {got:.4f}")
    if failures or query_failures:
        sys.exit(1)
    per_query_note = (f" (and no golden query dropped by more than "
                      f"{baseline.get('query_tolerance', tol)})" if "per_query" in baseline else "")
    print(f"\n✅ no regression{per_query_note}")


if __name__ == "__main__":
    main()
