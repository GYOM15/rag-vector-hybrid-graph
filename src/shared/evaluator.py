"""RAGAS evaluation and the stack benchmark loop.

The heavy imports (ragas, langchain-openai, datasets) are done **on demand**: importing
this module stays lightweight even without the `[eval]` extra installed. It is only
required when RAGAS is actually called.

The judge is configured EXPLICITLY (`JudgeConfig`): key, endpoint and model names are
handed to the judge objects, never read from or written to `os.environ`. The app serves
many visitors from one process, each with their own key: a key placed in the environment
would be used for — and billed to — every other visitor's run. For the same reason each
run builds its own metric objects: ragas attaches the judge to the metrics it is given
and detaches it afterwards, so two concurrent runs sharing ragas' module-level metric
instances would swap (or lose) each other's judge mid-run.
"""

import inspect
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .llm import _timeout

OPENAI_API_URL = "https://api.openai.com/v1"
_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
# Dataset column names: ragas 0.1 vs 0.2+ (question, answer, contexts, ground truth).
_V1_COLUMNS = ("question", "answer", "contexts", "ground_truth")
_V2_COLUMNS = ("user_input", "response", "retrieved_contexts", "reference")


@dataclass(frozen=True)
class JudgeConfig:
    """The RAGAS judge: an OpenAI-compatible chat model plus an embedding model (answer
    relevancy compares embeddings).

    `base_url` None -> OpenAI. The builders always pass an endpoint explicitly: left
    unset, the OpenAI client would fall back to OPENAI_BASE_URL from the environment —
    on a server whose own generation endpoint is set there, a visitor's key would go to
    it. The key is kept out of `repr` so a logged config cannot leak it.
    """

    api_key: str = field(repr=False)
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    base_url: str | None = None

    @property
    def endpoint(self) -> str:
        return (self.base_url or OPENAI_API_URL).strip().rstrip("/")


def ragas_columns(version: str) -> tuple[str, str, str, str]:
    """Column names ragas `version` expects for (question, answer, contexts, reference):
    0.1-style before 0.2, else the 0.2+ `SingleTurnSample` fields."""
    parts = [int(p) for p in re.findall(r"\d+", version)[:2]]
    major, minor = (parts + [0, 0])[:2]
    return _V1_COLUMNS if (major, minor) < (0, 2) else _V2_COLUMNS


def _require_key(judge: JudgeConfig | None) -> JudgeConfig:
    # An empty key must fail here: the OpenAI client treats "" as "unset" and would then
    # read OPENAI_API_KEY from the environment, i.e. use the server's key.
    if judge is None or not judge.api_key.strip():
        raise ValueError("RAGAS needs a judge API key (none given).")
    return judge


def _langchain_judge(judge: JudgeConfig) -> tuple:
    """`(ChatOpenAI, OpenAIEmbeddings)` for `judge`, every setting passed explicitly."""
    judge = _require_key(judge)
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    common = {"api_key": judge.api_key.strip(), "base_url": judge.endpoint,
              "timeout": _timeout(), "max_retries": 1}
    chat = ChatOpenAI(model=judge.model, temperature=0, **common)
    # Raw strings, not tiktoken ids: works with any OpenAI-compatible server (Ollama's
    # rejects token arrays) and needs no tokenizer download. Inputs here are short.
    embeddings = OpenAIEmbeddings(model=judge.embedding_model,
                                  check_embedding_ctx_length=False, **common)
    return chat, embeddings


def check_judge(judge: JudgeConfig) -> None:
    """One tiny chat call and one embedding call, so a wrong key, model or endpoint fails
    in seconds — before a benchmark spends minutes generating answers. Raises the
    client's error."""
    chat, embeddings = _langchain_judge(judge)
    chat.invoke("Reply with the word OK.", max_tokens=5)
    embeddings.embed_query("OK")


def _ragas_judge(judge: JudgeConfig, run_config) -> tuple:
    """The judge as ragas objects `(llm, embeddings)`, built for the installed ragas."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper

    chat, embeddings = _langchain_judge(judge)
    options = {"run_config": run_config}
    # Only OpenAI is known to honour `n` (several completions per request, used by answer
    # relevancy); elsewhere (e.g. Ollama) ask ragas to send n separate requests instead.
    if (judge.endpoint != OPENAI_API_URL
            and "bypass_n" in inspect.signature(LangchainLLMWrapper.__init__).parameters):
        options["bypass_n"] = True
    return LangchainLLMWrapper(chat, **options), LangchainEmbeddingsWrapper(embeddings)


def _dataset(questions, answers, contexts, ground_truths, version: str):
    """The evaluation set, with the column names of the installed ragas."""
    columns = ragas_columns(version)
    rows = [dict(zip(columns, row)) for row in zip(questions, answers, contexts, ground_truths)]
    if columns == _V1_COLUMNS:
        from datasets import Dataset

        return Dataset.from_dict({c: [r[c] for r in rows] for c in columns})
    from ragas import EvaluationDataset

    return EvaluationDataset.from_list(rows)


def _nan_mean(values) -> float | None:
    """Mean of the numeric, non-NaN `values` (a judge failure scores NaN); None if none."""
    kept = [float(v) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)]
    return sum(kept) / len(kept) if kept else None


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    judge: JudgeConfig | None = None,
) -> dict:
    """RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall +
    per_question) scored by `judge`.

    A metric no question could be scored on is None. Raises ValueError if the lists differ
    in length or `judge` has no key, RuntimeError if the judge scored nothing at all (every
    call failed: wrong key, model or endpoint).
    """
    n = len(questions)
    if not (n == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError(
            f"All input lists must have the same length. Got "
            f"questions={len(questions)}, answers={len(answers)}, "
            f"contexts={len(contexts)}, ground_truths={len(ground_truths)}."
        )
    judge = _require_key(judge)

    import ragas
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    # Fewer retries than ragas' default (10, up to 60 s apart): a bad key must not stall
    # the run for many minutes. Failed scores come back as NaN (raise_exceptions=False).
    run_config = RunConfig(max_retries=3, max_wait=10, max_workers=8)
    llm, embeddings = _ragas_judge(judge, run_config)
    metrics = [  # fresh instances, never shared across runs (see module docstring)
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    dataset = _dataset(questions, answers, contexts, ground_truths, ragas.__version__)
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings,
                      run_config=run_config, raise_exceptions=False)

    per_question = result.to_pandas().to_dict(orient="records")
    summary = {m: _nan_mean(row.get(m) for row in per_question) for m in _RAGAS_METRICS}
    if all(value is None for value in summary.values()):
        raise RuntimeError("The judge returned no usable score (every call failed): check "
                           "the key, the model names and the endpoint.")
    summary["per_question"] = per_question
    return summary


def _mean_quality(per_question: list[dict], indices: list[int]) -> dict:
    """Average of the RAGAS metrics over a subset of questions (by index)."""
    summary = {}
    for metric in _RAGAS_METRICS:
        mean = _nan_mean(per_question[i].get(metric) for i in indices)
        if mean is not None:
            summary[metric] = round(mean, 4)
    return summary


def evaluate_stacks(
    stacks: dict,
    questions: list[str],
    ground_truths: list[str],
    k: int = 5,
    types: list[str] | None = None,
    llm_fn: Callable[[str], str] | None = None,
    judge: JudgeConfig | None = None,
) -> dict[str, dict]:
    """Evaluate each stack (generation + latencies + RAGAS), overall and by `types` if provided.

    `llm_fn` overrides the stacks' generator for this run (the app passes the session's
    backend); `judge` is the RAGAS judge (None -> RAGAS is reported as failed). Returns
    {stack_name: metrics}; if RAGAS fails (no judge key, ragas not installed…), only the
    latencies + `ragas_error` ("<ExcType>: <message>"), so the caller can say *why*.
    Raises ValueError if `questions` is empty.
    """
    n = len(questions)
    if n == 0:
        raise ValueError("No questions to evaluate.")
    results: dict[str, dict] = {}

    for name, rag in stacks.items():
        answers, contexts, latencies = [], [], []
        retrieval = generation = total = 0.0
        for question in questions:
            r = rag.query(question, k=k, llm_fn=llm_fn)
            answers.append(r["answer"])
            contexts.append([c["text"] for c in r["contexts"]])
            latencies.append(r["latency_ms"])
            retrieval += r["retrieval_ms"]
            generation += r["generation_ms"]
            total += r["latency_ms"]

        metrics: dict = {}
        per_question: list[dict] = []
        try:
            full = evaluate_rag(questions, answers, contexts, ground_truths, judge=judge)
            per_question = full.pop("per_question", []) or []
            metrics = full
        except Exception as exc:  # RAGAS optional: keep the latencies, but record why
            metrics = {"ragas_error": f"{type(exc).__name__}: {exc}"}

        metrics["avg_retrieval_ms"] = round(retrieval / n, 2)
        metrics["avg_generation_ms"] = round(generation / n, 2)
        metrics["avg_latency_ms"] = round(total / n, 2)

        if types:
            valid = per_question if len(per_question) == n else []
            by_type: dict[str, dict] = {}
            for t in dict.fromkeys(types):  # unique categories, order preserved
                idx = [i for i, tt in enumerate(types) if tt == t]
                entry = {
                    "n": len(idx),
                    "avg_latency_ms": round(sum(latencies[i] for i in idx) / len(idx), 2),
                }
                if valid:
                    entry.update(_mean_quality(valid, idx))
                by_type[t] = entry
            metrics["by_type"] = by_type

        results[name] = metrics

    return results
