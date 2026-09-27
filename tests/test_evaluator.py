"""Tests for the RAGAS judge plumbing — no ragas, no langchain, no network.

`ragas`, `langchain_openai` (and `datasets`, for the 0.1 path) are replaced by fakes in
`sys.modules` that record what they are given. What must hold: the judge's key, endpoint
and model names reach the judge objects explicitly, never through `os.environ` (the app
serves many visitors, each with their own key, from one process), the metric objects are
fresh per run, and the dataset uses the column names of the installed ragas version.
"""

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared import evaluator  # noqa: E402
from shared.evaluator import JudgeConfig, ragas_columns  # noqa: E402

_NAN = float("nan")


class _Recorder:
    """Base of the fakes: keeps its constructor arguments."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class _FakeChat(_Recorder):
    def invoke(self, prompt, **kwargs):
        self.invoked = (prompt, kwargs)
        return "OK"


class _FakeEmbeddings(_Recorder):
    def embed_query(self, text):
        self.embedded = text
        return [0.0]


class _FakeLLMWrapper:
    def __init__(self, langchain_llm, run_config=None, bypass_n=False):
        self.langchain_llm, self.run_config, self.bypass_n = langchain_llm, run_config, bypass_n


class _FakeEmbeddingsWrapper:
    def __init__(self, embeddings, run_config=None):
        self.embeddings = embeddings


class _FakeMetric(_Recorder):
    name = "?"


def _metric(name):
    return type(name, (_FakeMetric,), {"name": name})


class _FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def to_pandas(self):
        rows = self.rows
        return types.SimpleNamespace(to_dict=lambda orient: rows)


@pytest.fixture
def fake_ragas(monkeypatch):
    """Fake ragas 0.4 + langchain_openai; `calls` records what `evaluate` received, and
    `scores` (one dict per question) is what it returns."""
    calls = types.SimpleNamespace(evaluate=[], datasets=[], scores=None)

    def evaluate(dataset, **kwargs):
        calls.evaluate.append((dataset, kwargs))
        return _FakeResult(calls.scores or [
            {"faithfulness": 1.0, "answer_relevancy": 0.5, "context_precision": _NAN,
             "context_recall": 0.25} for _ in dataset.rows])

    class EvaluationDataset:
        @classmethod
        def from_list(cls, rows):
            calls.datasets.append(rows)
            return types.SimpleNamespace(rows=rows)

    ragas = types.ModuleType("ragas")
    ragas.__version__ = "0.4.3"
    ragas.evaluate, ragas.EvaluationDataset = evaluate, EvaluationDataset
    metrics = types.ModuleType("ragas.metrics")
    for name, cls in (("Faithfulness", "faithfulness"), ("AnswerRelevancy", "answer_relevancy"),
                      ("ContextPrecision", "context_precision"),
                      ("ContextRecall", "context_recall")):
        setattr(metrics, name, _metric(cls))
    run_config = types.ModuleType("ragas.run_config")
    run_config.RunConfig = _Recorder
    llms, llms_base = types.ModuleType("ragas.llms"), types.ModuleType("ragas.llms.base")
    llms_base.LangchainLLMWrapper = _FakeLLMWrapper
    embeddings = types.ModuleType("ragas.embeddings")
    embeddings.LangchainEmbeddingsWrapper = _FakeEmbeddingsWrapper
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI, langchain_openai.OpenAIEmbeddings = _FakeChat, _FakeEmbeddings
    for module in (ragas, metrics, run_config, llms, llms_base, embeddings, langchain_openai):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    # The server's own settings, which a visitor's judge must never pick up.
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm.internal:8000/v1")
    return calls


def _run(judge, n=2):
    return evaluator.evaluate_rag([f"q{i}" for i in range(n)], [f"a{i}" for i in range(n)],
                                  [[f"c{i}"] for i in range(n)], [f"g{i}" for i in range(n)],
                                  judge=judge)


# ---------------------------------------------------------------------------
# Judge objects
# ---------------------------------------------------------------------------

def test_key_endpoint_and_models_reach_the_judge_objects(fake_ragas):
    before = dict(os.environ)
    _run(JudgeConfig(api_key=" sk-visitor ", model="gpt-4.1-mini",
                     embedding_model="text-embedding-3-large"))

    (_, kwargs), = fake_ragas.evaluate
    chat, embeddings = kwargs["llm"].langchain_llm, kwargs["embeddings"].embeddings
    assert chat.kwargs["api_key"] == embeddings.kwargs["api_key"] == "sk-visitor"
    # OpenAI, explicitly: never the OPENAI_BASE_URL of the environment (the server's).
    assert chat.kwargs["base_url"] == embeddings.kwargs["base_url"] == "https://api.openai.com/v1"
    assert chat.kwargs["model"] == "gpt-4.1-mini"
    assert embeddings.kwargs["model"] == "text-embedding-3-large"
    assert embeddings.kwargs["check_embedding_ctx_length"] is False
    assert kwargs["llm"].bypass_n is False             # OpenAI honours n
    assert kwargs["raise_exceptions"] is False
    assert dict(os.environ) == before                  # nothing written, nothing changed


def test_metrics_are_fresh_and_carry_the_judge(fake_ragas):
    judge = JudgeConfig(api_key="sk-visitor")
    _run(judge)
    _run(judge)
    (_, first), (_, second) = fake_ragas.evaluate
    names = [m.name for m in first["metrics"]]
    assert names == ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    assert all(m.kwargs["llm"] is first["llm"] for m in first["metrics"])
    assert first["metrics"][1].kwargs["embeddings"] is first["embeddings"]
    # New objects on every run: two concurrent runs never share (and swap) a judge.
    assert not {id(m) for m in first["metrics"]} & {id(m) for m in second["metrics"]}
    assert first["llm"] is not second["llm"]


def test_custom_judge_endpoint(fake_ragas):
    _run(JudgeConfig(api_key="ollama", model="qwen2.5:1.5b", embedding_model="nomic-embed-text",
                     base_url="http://localhost:11434/v1/"))
    (_, kwargs), = fake_ragas.evaluate
    assert kwargs["llm"].langchain_llm.kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["llm"].bypass_n is True              # n separate requests off OpenAI


@pytest.mark.parametrize("judge", [None, JudgeConfig(api_key=""), JudgeConfig(api_key="  ")])
def test_no_key_fails_before_any_client_is_built(monkeypatch, judge):
    # An empty key would make the OpenAI client fall back to OPENAI_API_KEY (the server's).
    monkeypatch.setitem(sys.modules, "ragas", None)
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    with pytest.raises(ValueError, match="judge API key"):
        _run(judge)
    with pytest.raises(ValueError, match="judge API key"):
        evaluator.check_judge(judge)


def test_check_judge_makes_one_chat_and_one_embedding_call(fake_ragas, monkeypatch):
    chat, embeddings = _FakeChat(), _FakeEmbeddings()
    built = []
    monkeypatch.setattr(evaluator, "_langchain_judge",
                        lambda judge: built.append(judge) or (chat, embeddings))
    judge = JudgeConfig(api_key="sk-visitor")
    evaluator.check_judge(judge)
    assert built == [judge]
    assert chat.invoked[1] == {"max_tokens": 5}  # a few tokens, not a benchmark
    assert embeddings.embedded == "OK"


def test_key_is_kept_out_of_the_repr():
    assert "sk-visitor" not in repr(JudgeConfig(api_key="sk-visitor"))


# ---------------------------------------------------------------------------
# Dataset columns and scores
# ---------------------------------------------------------------------------

def test_dataset_uses_the_02_plus_column_names(fake_ragas):
    _run(JudgeConfig(api_key="k"), n=1)
    assert fake_ragas.datasets == [[{"user_input": "q0", "response": "a0",
                                     "retrieved_contexts": ["c0"], "reference": "g0"}]]


def test_dataset_uses_the_01_column_names(fake_ragas, monkeypatch):
    monkeypatch.setattr(sys.modules["ragas"], "__version__", "0.1.21")
    received = []
    datasets = types.ModuleType("datasets")
    datasets.Dataset = types.SimpleNamespace(
        from_dict=lambda columns: received.append(columns) or types.SimpleNamespace(rows=[0]))
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    _run(JudgeConfig(api_key="k"), n=1)
    assert received == [{"question": ["q0"], "answer": ["a0"], "contexts": [["c0"]],
                         "ground_truth": ["g0"]}]


@pytest.mark.parametrize("version, expected", [
    ("0.1.21", ("question", "answer", "contexts", "ground_truth")),
    ("0.2.0rc1", ("user_input", "response", "retrieved_contexts", "reference")),
    ("0.4.3", ("user_input", "response", "retrieved_contexts", "reference")),
    ("1.0", ("user_input", "response", "retrieved_contexts", "reference")),
])
def test_ragas_columns(version, expected):
    assert ragas_columns(version) == expected


def test_scores_ignore_nan_and_all_nan_is_none(fake_ragas):
    fake_ragas.scores = [
        {"faithfulness": 1.0, "answer_relevancy": _NAN, "context_precision": _NAN,
         "context_recall": 0.0},
        {"faithfulness": 0.5, "answer_relevancy": 0.8, "context_precision": _NAN,
         "context_recall": None},
    ]
    out = _run(JudgeConfig(api_key="k"))
    assert out["faithfulness"] == 0.75
    assert out["answer_relevancy"] == 0.8
    assert out["context_precision"] is None             # no question scored
    assert out["context_recall"] == 0.0
    assert len(out["per_question"]) == 2


def test_a_judge_that_scored_nothing_is_an_error(fake_ragas):
    fake_ragas.scores = [dict.fromkeys(evaluator._RAGAS_METRICS, _NAN)] * 2
    with pytest.raises(RuntimeError, match="no usable score"):
        _run(JudgeConfig(api_key="bad-key"))


# ---------------------------------------------------------------------------
# evaluate_stacks forwards the judge
# ---------------------------------------------------------------------------

class _FakeRAG:
    def query(self, question, k=5, llm_fn=None):
        return {"answer": "a", "contexts": [{"text": "c"}],
                "retrieval_ms": 1.0, "generation_ms": 2.0, "latency_ms": 3.0}


def test_evaluate_stacks_forwards_the_judge(monkeypatch):
    seen = []

    def fake_evaluate_rag(questions, answers, contexts, ground_truths, judge=None):
        seen.append(judge)
        return {"faithfulness": 0.5, "per_question": [{"faithfulness": 0.5}] * len(questions)}

    monkeypatch.setattr(evaluator, "evaluate_rag", fake_evaluate_rag)
    judge = JudgeConfig(api_key="sk-visitor")
    out = evaluator.evaluate_stacks({"A": _FakeRAG(), "B": _FakeRAG()}, ["q1", "q2"],
                                    ["g1", "g2"], types=["x", "y"], judge=judge)
    assert seen == [judge, judge]
    assert out["A"]["by_type"]["x"] == {"n": 1, "avg_latency_ms": 3.0, "faithfulness": 0.5}


def test_evaluate_stacks_without_judge_reports_why(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", None)  # never reached
    out = evaluator.evaluate_stacks({"A": _FakeRAG()}, ["q1"], ["g1"])
    assert out["A"]["ragas_error"] == "ValueError: RAGAS needs a judge API key (none given)."
    assert out["A"]["avg_latency_ms"] == 3.0


def test_cli_judge_comes_from_the_environment(monkeypatch):
    pytest.importorskip("dotenv")
    pytest.importorskip("faiss")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent))
    from eval.benchmark import judge_from_env

    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    judge = judge_from_env("gpt-4o-mini", "text-embedding-3-small")
    assert (judge.api_key, judge.endpoint) == ("sk-owner", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert judge_from_env("qwen2.5:1.5b", "nomic-embed-text").endpoint == \
        "http://localhost:11434/v1"
