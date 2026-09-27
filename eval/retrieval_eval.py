"""Evaluation of RETRIEVAL alone (no LLM), over one or more embedders.

For each embedder × architecture × k, measures whether the chunk containing the
answer (`gold`) is retrieved, and at what rank: hit@k and MRR. No LLM required →
deterministic, free, immune to the model's memory. Aggregated globally and per
category (`type`), with n and a 95% bootstrap CI on MRR (27 questions: wide).
Also serves to show the role of the embedding model.

A retrieved chunk is a hit only if it comes from the question's source article
(`title`) AND contains the gold as whole words (see `_is_hit`). A raw substring test
over any article credited chunks that merely mention the question's own entity
("Zimbabwe" anywhere) or embed the gold in a longer word ("Wright" in "playwright").

Example:
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --embedders all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
"""

import argparse
import json
import re
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.provenance import finish_metadata, run_metadata  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402

KS = (1, 3, 5, 8, 10)
DEFAULT_EMBEDDERS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
QUESTIONS = ROOT / "eval" / "questions.json"
HIT_RULE = ("chunk title == item title (when given) AND gold matched case-insensitively "
            "as whole words")


@cache
def _gold_pattern(gold: str) -> re.Pattern:
    """Case-insensitive whole-word pattern for `gold`.

    Lookarounds instead of \\b: \\b needs a word character on its inner side, so it
    never matches at a gold's edge made of punctuation ("C++"); (?<!\\w)/(?!\\w) just
    forbid a letter or digit glued to either end. Inner whitespace matches any run of
    whitespace (a multi-word gold may wrap), and non-word characters inside the gold
    (the em-dash of "Mycology—Fungi") are matched literally.
    """
    body = r"\s+".join(re.escape(token) for token in gold.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _is_hit(ctx: dict, item: dict) -> bool:
    """True if the retrieved chunk `ctx` answers `item`: it comes from the item's
    source article (when the item names one) and contains its gold as whole words."""
    title = item.get("title")
    if title and ctx.get("metadata", {}).get("title") != title:
        return False
    return bool(_gold_pattern(item["gold"]).search(ctx["text"]))


def unfindable_golds(chunks: list[dict], data: list[dict]) -> list[dict]:
    """Items whose gold matches no chunk of the corpus under `_is_hit`: they can never
    be hit, so they would silently count as misses for every stack."""
    return [d for d in data if not any(_is_hit(c, d) for c in chunks)]


def _first_hit_rank(retriever, item: dict, k_max: int) -> int | None:
    """Rank (1-indexed) of the 1st chunk answering `item`, or None if outside top-k_max."""
    for rank, ctx in enumerate(retriever.search(item["question"], k=k_max), 1):
        if _is_hit(ctx, item):
            return rank
    return None


def _aggregate(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in KS}
    out["mrr"] = round(sum(1.0 / r for r in ranks if r) / n, 3)
    _, lo, hi = bootstrap_ci([1.0 / r if r else 0.0 for r in ranks])
    out["mrr_ci95"] = [round(lo, 3), round(hi, 3)]
    out["n"] = n
    return out


def _eval_one(stacks: dict, data: list, types: list, cats: list, k_max: int):
    ranks = {name: [_first_hit_rank(rag.retriever, d, k_max) for d in data]
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


def _load_questions() -> list[dict]:
    return [d for d in json.loads(QUESTIONS.read_text("utf-8")) if d.get("gold")]


def _load_corpus(n_articles: int) -> tuple[list[str], list[dict]]:
    from pipeline import load_chunks

    chunks = load_chunks(n_articles)
    return [c.text for c in chunks], [c.metadata for c in chunks]


def check_golds(n_articles: int, data: list[dict] | None = None) -> list[dict]:
    """Prints and returns the questions whose gold no chunk of the corpus can match."""
    data = data or _load_questions()
    texts, metadata = _load_corpus(n_articles)
    chunks = [{"text": t, "metadata": m} for t, m in zip(texts, metadata)]
    missing = unfindable_golds(chunks, data)
    print(f"Gold check ({n_articles} articles, {len(chunks)} chunks, {len(data)} questions): "
          + ("every gold is findable" if not missing else f"{len(missing)} unfindable"))
    for d in missing:
        print(f"  ❌ {d.get('title', '?')!r}: gold {d['gold']!r} — {d['question']}")
    return missing


def run(n_articles: int, output: Path, embedders: list[str]) -> dict:
    from pipeline import assemble_stacks

    provenance = run_metadata()  # the code that runs is the code at the start
    data = _load_questions()
    types = [d.get("type", "?") for d in data]
    cats = sorted(set(types))
    k_max = max(KS)

    # Chunked once for every embedder. assemble_stacks, not build_stacks: the latter
    # honours RERANK_MODE, which would silently evaluate reranked retrievers.
    texts, metadata = _load_corpus(n_articles)
    unfindable = unfindable_golds(
        [{"text": t, "metadata": m} for t, m in zip(texts, metadata)], data)
    for d in unfindable:
        print(f"⚠️  gold never matches its article (always a miss): {d['gold']!r}")

    results = {}
    for emb in embedders:
        print(f"\n=== Embedder: {emb} ({n_articles} articles) ===")
        stacks = assemble_stacks(texts, metadata, embedder=emb)
        report, suspects = _eval_one(stacks, data, types, cats, k_max)
        results[emb] = {"stacks": report, "unretrieved": suspects}
        for name, rep in report.items():
            o = rep["overall"]
            print(f"  {name}")
            print("    global   " + "  ".join(f"hit@{k}={o[f'hit@{k}']:.3f}" for k in KS)
                  + f"  MRR={o['mrr']:.3f} {o['mrr_ci95']} (n={o['n']})")
            for t, m in rep["by_type"].items():
                print(f"    {t:8} " + "  ".join(f"hit@{k}={m[f'hit@{k}']:.3f}" for k in KS)
                      + f"  MRR={m['mrr']:.3f} {m['mrr_ci95']} (n={m['n']})")

    payload = {
        "config": {"n_articles": n_articles, "n_chunks": len(texts), "n_questions": len(data),
                   "ks": list(KS), "embedders": embedders, "hit_rule": HIT_RULE,
                   "unfindable_golds": [d["question"] for d in unfindable],
                   "ci": "95% percentile bootstrap over questions (10k resamples, seed 0)",
                   "provenance": finish_metadata(provenance)},
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Cross-cutting comparison: global MRR per architecture and per embedder.
    print("\n=== Embedder comparison (global MRR) ===")
    stack_names = list(next(iter(results.values()))["stacks"])
    print("  " + "architecture".ljust(34) + "".join(e[:20].ljust(22) for e in embedders))
    for name in stack_names:
        row = "  " + name.ljust(34)
        row += "".join(f"{results[emb]['stacks'][name]['overall']['mrr']:.3f}".ljust(22) for emb in embedders)
        print(row)

    print(f"\n✅ Details written to {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-embedder RAG retrieval eval (hit@k, MRR, no LLM).")
    ap.add_argument("--articles", type=int, default=100)
    ap.add_argument("--embedders", nargs="+", default=DEFAULT_EMBEDDERS)
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "retrieval_results.json")
    ap.add_argument("--check-golds", action="store_true",
                    help="only check that every gold matches a chunk of its article "
                         "(no embedding); exits 1 otherwise")
    args = ap.parse_args()
    if args.check_golds:
        sys.exit(1 if check_golds(args.articles) else 0)
    run(args.articles, args.output, args.embedders)


if __name__ == "__main__":
    main()
