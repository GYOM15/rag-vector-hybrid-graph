"""Tests for the LLM interface — no backend is ever reached.

The network (`urllib.request.urlopen`), ollama and the HF loader are faked; the seq2seq
prompt fitting runs on a whitespace "tokenizer" (1 word = 1 token), so neither
transformers nor torch is needed.
"""

import json
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared import llm  # noqa: E402
from shared.prompts import QUESTION_MARKER, build_prompt  # noqa: E402

_QUESTION = "Where was Alan Turing born?"
_TAIL = f"\nQuestion: {_QUESTION}\n\nAnswer:"


class _WhitespaceTokenizer:
    """The minimal interface `fit_prompt` relies on: 1 word = 1 token."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(ids)


_TOK = _WhitespaceTokenizer()


def _prompt(n_chunks: int) -> str:
    contexts = [{"text": f"chunk{i} " + "filler " * 20, "metadata": {"title": f"T{i}"}}
                for i in range(n_chunks)]
    return build_prompt(_QUESTION, contexts)


# ---------------------------------------------------------------------------
# Prompt fitting (flan-t5's 512-token encoder; input cap of instruct models)
# ---------------------------------------------------------------------------

def test_marker_matches_the_prompt_template():
    # If the template changes, the fitting must still find where the question starts.
    prompt = _prompt(1)
    assert prompt[prompt.rfind(QUESTION_MARKER):] == _TAIL


def test_short_prompt_is_unchanged():
    prompt = _prompt(1)
    assert llm.fit_prompt(prompt, _TOK, 10_000) == prompt


def test_long_prompt_keeps_the_question_and_the_top_chunks():
    fitted = llm.fit_prompt(_prompt(50), _TOK, 100)            # ~1,300 tokens -> 100
    assert len(_TOK.encode(fitted)) == 100                      # fills the budget, no more
    assert fitted.endswith(_TAIL)                               # question kept verbatim
    assert fitted.startswith("Answer the question based ONLY")  # instruction kept
    assert "chunk0" in fitted and "chunk49" not in fitted       # head cut from its end


def test_without_marker_falls_back_to_plain_truncation():
    text = " ".join(f"w{i}" for i in range(50))
    assert llm.fit_prompt(text, _TOK, 10) == " ".join(f"w{i}" for i in range(10))


def test_question_longer_than_the_budget_keeps_its_end():
    question = " ".join(f"q{i}" for i in range(30))
    prompt = f"Context:\nabc\nQuestion: {question}\n\nAnswer:"
    assert llm.fit_prompt(prompt, _TOK, 5).split() == ["q26", "q27", "q28", "q29", "Answer:"]


def test_boundary_drift_is_re_measured():
    class _Drifting(_WhitespaceTokenizer):  # decoding adds a token (like a merge at the seam)
        def decode(self, ids, skip_special_tokens=False):
            return " ".join(ids) + " <x>"

    tok = _Drifting()
    fitted = llm.fit_prompt(_prompt(50), tok, 100)
    assert len(tok.encode(fitted)) <= 100
    assert fitted.endswith(_TAIL)


# ---------------------------------------------------------------------------
# HuggingFace backend (fake loader: no transformers)
# ---------------------------------------------------------------------------

class _FakeSeq2SeqTokenizer(_WhitespaceTokenizer):
    model_max_length = 30

    def num_special_tokens_to_add(self):
        return 1  # T5's </s>

    def __call__(self, text, return_tensors=None, max_length=None, truncation=False):
        self.encoded = self.encode(text)
        return {"input_ids": self.encoded}


class _FakeSeq2SeqModel:
    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return [["Maida", "Vale"]]


@pytest.fixture
def hf(monkeypatch):
    """A fake seq2seq model behind `_load_hf_model`; records every load."""
    tok, model, loads = _FakeSeq2SeqTokenizer(), _FakeSeq2SeqModel(), []

    def load(name):
        loads.append(name)
        time.sleep(0.05)  # widen the race window for the concurrency test
        return "seq2seq", tok, model

    monkeypatch.setattr(llm, "_load_hf_model", load)
    monkeypatch.setattr(llm, "_HF_CACHE", {})
    return types.SimpleNamespace(tok=tok, model=model, loads=loads)


def test_seq2seq_keeps_the_question_and_bounds_new_tokens(hf):
    answer = llm.call_llm(_prompt(20), provider="huggingface", model="fake-t5", max_length=64)
    assert answer == "Maida Vale"
    assert len(hf.tok.encoded) <= 30 - 1                           # fits the encoder
    assert " ".join(hf.tok.encoded).endswith("Alan Turing born? Answer:")  # question survived
    assert hf.model.kwargs == {"max_new_tokens": 64}                # answer length, not input


class _FakeCausalTokenizer(_WhitespaceTokenizer):
    """Chat template = 2 marker tokens around the user message."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return f"<user> {messages[0]['content']} <assistant>"

    def __call__(self, texts, return_tensors=None):
        self.encoded = self.encode(texts[0])
        return _CausalInputs(input_ids=self.encoded)


class _CausalInputs(dict):
    @property
    def input_ids(self):
        return types.SimpleNamespace(shape=(1, len(self["input_ids"])))


class _FakeCausalModel:
    def generate(self, input_ids, **kwargs):
        return [input_ids + ["Maida", "Vale"]]  # prompt + new tokens, like HF decoders


@pytest.fixture
def causal(monkeypatch):
    tok = _FakeCausalTokenizer()
    monkeypatch.setattr(llm, "_load_hf_model", lambda name: ("causal", tok, _FakeCausalModel()))
    monkeypatch.setattr(llm, "_HF_CACHE", {})
    monkeypatch.setattr(llm, "_CAUSAL_MAX_INPUT", 100)
    return tok


def test_causal_prompt_within_the_cap_is_unchanged(causal):
    prompt = _prompt(2)
    answer = llm.call_llm(prompt, provider="huggingface", model="fake-qwen")
    assert answer == "Maida Vale"  # only the new tokens are decoded
    assert causal.encoded == ["<user>", *_TOK.encode(prompt), "<assistant>"]


def test_causal_input_is_capped_and_keeps_the_question(causal):
    # A huge question (e.g. from a client bypassing the app's length check) must not reach
    # generate() unbounded: it runs under the global lock, blocking every other session.
    huge = _prompt(50).replace(_QUESTION, "blah " * 100_000 + _QUESTION)
    assert llm.call_llm(huge, provider="huggingface", model="fake-qwen") == "Maida Vale"
    assert len(causal.encoded) <= 100 + 2                        # cap + the template's markers
    assert " ".join(causal.encoded).endswith("born? Answer: <assistant>")


def test_hf_cache_keeps_a_single_model(hf):
    for name in ("a", "a", "b"):
        llm._hf_model(name)
    assert hf.loads == ["a", "b"]        # "a" loaded once, then reused
    assert list(llm._HF_CACHE) == ["b"]  # switching model evicted "a"


def test_concurrent_first_calls_load_the_model_once(hf):
    answers = []

    def ask():
        answers.append(llm.call_llm("Q", provider="huggingface", model="fake-t5"))

    threads = [threading.Thread(target=ask) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert answers == ["Maida Vale"] * 4
    assert hf.loads == ["fake-t5"]


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (fake network)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, content: str):
        self._body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(monkeypatch):
    """Captures the requests `_call_openai` would send; the env holds a *server* config."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "auth": request.get_header("Authorization"),
                      "timeout": timeout, "body": json.loads(request.data)})
        return _FakeResponse("ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://server.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)
    return calls


def test_openai_sends_to_the_given_endpoint_with_the_given_key(sent):
    answer = llm.call_llm("hi", provider="openai", model="m",
                          base_url="http://visitor.example/v1/", api_key="visitor-key")
    assert answer == "ok"
    assert sent[0]["url"] == "http://visitor.example/v1/chat/completions"
    assert sent[0]["auth"] == "Bearer visitor-key"
    assert sent[0]["body"]["model"] == "m"


def test_openai_falls_back_to_the_env_only_when_args_are_none(sent):
    llm.call_llm("hi", provider="openai")
    assert sent[0]["url"] == "http://server.example/v1/chat/completions"
    assert sent[0]["auth"] == "Bearer server-secret"


def test_openai_empty_key_does_not_pick_up_the_server_key(sent):
    llm.call_llm("hi", provider="openai", base_url="http://visitor.example/v1", api_key="")
    assert sent[0]["auth"] == "Bearer EMPTY"


def test_openai_call_has_a_timeout(sent, monkeypatch):
    llm.call_llm("hi", provider="openai")
    monkeypatch.setenv("LLM_TIMEOUT", "7")
    llm.call_llm("hi", provider="openai")
    assert [c["timeout"] for c in sent] == [120, 7]


def test_openai_rejects_non_http_urls(sent):
    with pytest.raises(ValueError, match="http"):
        llm.call_llm("hi", provider="openai", base_url="file:///etc/passwd")
    assert sent == []


# ---------------------------------------------------------------------------
# Ollama backend + dispatch
# ---------------------------------------------------------------------------

def test_ollama_client_gets_a_timeout(monkeypatch):
    created = []

    class _FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def chat(self, **kwargs):
            return {"message": {"content": "ok"}}

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(llm, "_OLLAMA_CLIENTS", {})
    monkeypatch.setenv("LLM_TIMEOUT", "30")
    assert llm.call_llm("hi", provider="ollama") == "ok"
    assert created[0]["timeout"] == 30


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        llm.call_llm("hi", provider="nope")
