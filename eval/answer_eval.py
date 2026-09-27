"""Retrieval -> answer loop: Exact-Match / F1 / contains on HotpotQA gold answers.

For each architecture, on the **same sample**: nDCG@10 (retrieval quality) and
EM / F1 / contains (generated answer quality), against the HotpotQA gold answers. No
judge, deterministic (greedy decoding, temperature 0). Lets you answer: "does better
retrieval lead to better answers?" and compare a small vs a large model (--model).

EM/F1 compare the *whole* generation to a few-word gold, so on unconstrained answers
they mostly measure verbosity: a model that answers correctly in a full sentence
scores below a terser, less accurate one. Hence:
- `--prompt short` asks for the shortest answer span (or "unknown") - an eval-only
  template, so the app's prompt is untouched;
- `contains` (the normalized gold occurs in the answer as whole tokens, with guards
  against yes/no refusals, hedges and restated choice questions - see
  shared.answer_metrics.contains_answer) is reported next to EM/F1: it credits a
  correct but verbose answer. `answer_tokens` (mean answer length) sits next to it,
  so a model that "wins" contains by listing candidates shows up as verbose;
- every generation is saved under "per_query", so any claim can be audited, and the
  means come with 95% bootstrap CIs plus paired tests between stacks (eval/stats.py).

    python -m eval.answer_eval --max-queries 100 --model llama3.2:1b --prompt short
    python -m eval.answer_eval --max-queries 100 --model llama3.2:3b --prompt short
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shared.answer_metrics import (  # noqa: E402
    contains_answer,
    exact_match,
    f1_score,
    normalize_answer,
)
from shared.ir_metrics import ndcg_at_k  # noqa: E402
from shared.llm import active_config  # noqa: E402
from shared.prompts import DEFAULT_PROMPT_TEMPLATE  # noqa: E402

from eval.beir_eval import K_MAX, _ranked_doc_ids, corpus_description, short_name  # noqa: E402
from eval.provenance import finish_metadata, run_metadata  # noqa: E402
from eval.stats import bootstrap_ci, pairwise  # noqa: E402

# Same structure as DEFAULT_PROMPT_TEMPLATE (instructions, context, same question tail);
# only the instructions change: HotpotQA golds are short spans, so ask for one.
SHORT_PROMPT_TEMPLATE = """Answer the question based ONLY on the following context.
Reply with the shortest possible answer: a few words, a name, a date, or yes/no. Do not explain.
If the context does not contain the answer, reply "unknown".

Context:
{context}

Question: {question}

Answer:"""

PROMPTS = {"default": DEFAULT_PROMPT_TEMPLATE, "short": SHORT_PROMPT_TEMPLATE}
# Recorded in each snapshot: the definition of "contains" is part of what the numbers mean.
CONTAINS_RULE = ("normalized gold as a contiguous whole-token run of the normalized answer, "
                 "not next to 'or'/'nor'/'vs'; yes/no golds: the reply opens with the gold "
                 "word alone and its first sentence lacks the opposite; golds named in the "
                 "question: the reply opens with the gold, not followed by 'or'/'and'")
METRICS = ("ndcg@10", "em", "f1", "contains", "answer_tokens")


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


def score_answer(answer: str, gold: str, question: str = "") -> dict[str, float]:
    """EM / F1 / contains of one generated answer against its gold, plus its length.

    The question is passed to `contains_answer`: when the gold is one of the options
    the question names, merely restating the question must not count as an answer.
    """
    return {"em": exact_match(answer, gold), "f1": f1_score(answer, gold),
            "contains": contains_answer(answer, gold, question),
            "answer_tokens": float(len(normalize_answer(answer).split()))}


def run(max_queries: int, model: str | None, output: Path, prompt: str = "default",
        k: int = K_MAX) -> dict:
    from pipeline import assemble_stacks

    provenance = run_metadata()  # the code that runs is the code at the start
    if model:
        os.environ["OLLAMA_MODEL"] = model
    llm = active_config()  # what actually generates: the provider and *its* model
    if model and llm["provider"] != "ollama":
        print(f"⚠️  --model only sets OLLAMA_MODEL; generation uses {llm['provider']} "
              f"({llm['model']}).", flush=True)

    n_corpus = max_queries or 100
    texts, metadata, queries, qrels, golds = load_hotpot_with_answers(n_corpus)
    print(f"HotpotQA: {len(texts)} docs, {len(queries)} questions - indexing + generation "
          f"({llm['provider']} {llm['model']}, prompt {prompt}, temperature 0)...", flush=True)
    stacks = assemble_stacks(texts, metadata)
    for rag in stacks.values():
        rag.prompt_template = PROMPTS[prompt]

    per_query = [{"qid": qid, "question": question, "gold": golds[qid], "stacks": {}}
                 for qid, question in queries]
    report, cols = {}, {}
    for sname, rag in stacks.items():
        short = short_name(sname)
        cols[short] = {m: [] for m in METRICS}
        for rec in per_query:
            out = rag.query(rec["question"], k=k)
            retrieved = _ranked_doc_ids(out["contexts"])
            row = {"ndcg@10": ndcg_at_k(retrieved, qrels[rec["qid"]], 10)}
            row |= score_answer(out["answer"], rec["gold"], rec["question"])
            for m in METRICS:
                cols[short][m].append(row[m])
            rec["stacks"][short] = ({"answer": out["answer"]}
                                    | {m: round(v, 4) for m, v in row.items()}
                                    | {"retrieved": retrieved})
        n = len(per_query)
        ci95 = {}
        for m in METRICS:
            _, lo, hi = bootstrap_ci(cols[short][m])
            ci95[m] = [round(lo, 4), round(hi, 4)]
        report[sname] = ({m: round(sum(v) / n, 4) for m, v in cols[short].items()}
                         | {"n_queries": n, "ci95": ci95})

    payload = {
        "config": {"dataset": "hotpotqa-distractor", "provider": llm["provider"],
                   "model": llm["model"], "prompt": prompt, "prompt_template": PROMPTS[prompt],
                   "k": k, "decoding": "greedy (temperature 0)", "n_docs": len(texts),
                   "n_queries": len(queries),
                   "corpus": corpus_description("hotpotqa-distractor", n_corpus),
                   "contains_rule": CONTAINS_RULE,
                   "ci": "95% percentile bootstrap over questions (10k resamples, seed 0)",
                   "provenance": finish_metadata(provenance)},
        "stacks": report,
        "paired_metric": "f1",
        "paired": pairwise({s: c["f1"] for s, c in cols.items()}),
        "paired_contains": pairwise({s: c["contains"] for s, c in cols.items()}),
        "per_query": per_query,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    cfg = payload["config"]
    print(f"\nHotpotQA - gold answers · {len(queries)} questions · {cfg['provider']} "
          f"{cfg['model']} · prompt {prompt}\n")
    print(f"  {'architecture':28s} {'nDCG@10':>8s} {'EM':>7s} {'F1':>7s} {'contains':>9s}"
          f" {'tokens':>7s}   F1 95% CI")
    for sname, m in report.items():
        lo, hi = m["ci95"]["f1"]
        print(f"  {sname:28s} {m['ndcg@10']:8.3f} {m['em']:7.3f} {m['f1']:7.3f} "
              f"{m['contains']:9.3f} {m['answer_tokens']:7.1f}   [{lo:.3f}, {hi:.3f}]")
    print("\n  paired F1 (diff = first - second, 95% CI, sign-flip p)")
    for pair, r in payload["paired"].items():
        print(f"    {pair:14} {r['mean_diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
              f"p={r['p_value']:.4f}")
    print(f"\n✅ Details (with every generation) written to {output}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="EM/F1/contains on HotpotQA gold answers (no judge, temperature 0).")
    ap.add_argument("--max-queries", type=int, default=100, help="number of questions (default 100)")
    ap.add_argument("--model", default=None, help="Ollama model (e.g. llama3.2:1b or llama3.2:3b)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="default",
                    help="default = the app's prompt; short = ask for the shortest answer span")
    ap.add_argument("--k", type=int, default=K_MAX, help="contexts retrieved per question")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "answer_results.json")
    args = ap.parse_args()
    run(args.max_queries, args.model, args.output, args.prompt, args.k)


if __name__ == "__main__":
    main()
