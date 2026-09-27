"""Tests for the app's backend configuration — the multi-user security rules.

`app/backend_config.py` is pure (no Streamlit), so the rules are checked directly on
env dicts: public-demo parsing, the credential rule — the server's OPENAI_API_KEY must
never be sent to an endpoint a visitor chose — and the public-demo choices: preset
endpoints only, the visitor's own key only, allow-listed local models only.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "src")]

from backend_config import (  # noqa: E402
    API_PRESETS,
    DEFAULT_HF_PUBLIC_MODELS,
    OPENAI_API_URL,
    SERVER,
    BackendChoice,
    demo_articles,
    is_public_demo,
    judge_settings,
    llm_kwargs,
    preset_for_url,
    public_hf_models,
    ragas_max_questions,
    ragas_questions,
    resolve_api_key,
    server_base_url,
)

from shared.llm import call_llm  # noqa: E402

_URL = "http://vllm.internal:8000/v1"
_SERVER = {"LLM_PROVIDER": "openai", "OPENAI_BASE_URL": _URL,
           "OPENAI_API_KEY": "server-secret", "OPENAI_MODEL": "served"}


# ---------------------------------------------------------------------------
# PUBLIC_DEMO / DEMO_ARTICLES parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_public_demo_truthy(value):
    assert is_public_demo({"PUBLIC_DEMO": value})


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "public"])
def test_public_demo_falsy(value):
    assert not is_public_demo({"PUBLIC_DEMO": value})


def test_public_demo_off_when_unset():
    assert not is_public_demo({})


@pytest.mark.parametrize("value", [None, "", "public"])
def test_public_demo_fails_closed_on_a_hf_space(value):
    # HF sets SPACE_ID in every Space: a Space deployed without PUBLIC_DEMO stays locked.
    env = {"SPACE_ID": "gyom15/rag-vector-hybrid-graph"}
    if value is not None:
        env["PUBLIC_DEMO"] = value
    assert is_public_demo(env)


@pytest.mark.parametrize("value", ["0", "false", "Off", " no "])
def test_public_demo_explicit_opt_out_on_a_hf_space(value):
    assert not is_public_demo({"SPACE_ID": "gyom15/private-space", "PUBLIC_DEMO": value})


def test_demo_articles():
    assert demo_articles({}) == 500
    assert demo_articles({"DEMO_ARTICLES": "200"}) == 200


# ---------------------------------------------------------------------------
# Credential rule
# ---------------------------------------------------------------------------

def test_server_key_goes_to_the_server_endpoint():
    kwargs = llm_kwargs(BackendChoice("openai", "m", _URL + "/"), _SERVER)  # trailing "/" ok
    assert kwargs == {"provider": "openai", "model": "m", "base_url": _URL + "/",
                      "api_key": "server-secret"}


def test_empty_base_url_means_the_server_endpoint():
    kwargs = llm_kwargs(BackendChoice("openai"), _SERVER)
    assert (kwargs["base_url"], kwargs["api_key"]) == (_URL, "server-secret")


def test_custom_endpoint_without_key_never_gets_the_server_key():
    # The exfiltration case: a visitor aims the app at their own server, key left empty.
    kwargs = llm_kwargs(BackendChoice("openai", "m", "https://attacker.example/v1"), _SERVER)
    assert kwargs["base_url"] == "https://attacker.example/v1"
    assert kwargs["api_key"] == "EMPTY"
    assert "server-secret" not in kwargs.values()


@pytest.mark.parametrize("url", [
    "http://vllm.internal:8000",                     # other path
    "http://vllm.internal:8000/v1/../../evil",
    "http://vllm.internal:8000/v1?next=evil",
    "http://vllm.internal:8000/v1.attacker.example",
    "https://vllm.internal:8000/v1",                 # other scheme
    "http://vllm.internal:8001/v1",                  # other port
])
def test_near_miss_urls_are_not_the_server_endpoint(url):
    assert resolve_api_key(url, "", _SERVER) == "EMPTY"


def test_custom_endpoint_uses_the_key_typed_in_the_session():
    choice = BackendChoice("openai", "gpt-4o-mini", "https://api.openai.com/v1", " sk-visitor ")
    assert llm_kwargs(choice, _SERVER)["api_key"] == "sk-visitor"


def test_typed_key_also_wins_on_the_server_endpoint():
    assert resolve_api_key(_URL, "sk-visitor", _SERVER) == "sk-visitor"


def test_no_server_key_configured():
    env = {k: v for k, v in _SERVER.items() if k != "OPENAI_API_KEY"}
    assert resolve_api_key(_URL, "", env) == "EMPTY"


def test_default_server_endpoint_when_unset():
    env = {"OPENAI_API_KEY": "server-secret"}
    assert server_base_url(env) == "http://localhost:8000/v1"
    assert resolve_api_key("http://localhost:8000/v1", "", env) == "server-secret"
    assert resolve_api_key("http://evil.example/v1", "", env) == "EMPTY"


def test_exfiltration_attempt_sends_no_secret_on_the_wire(monkeypatch):
    # End to end: session kwargs -> call_llm -> the (faked) HTTP request actually sent.
    sent = []

    class _Response:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        sent.append((request.full_url, request.get_header("Authorization")))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")  # also in the real env
    choice = BackendChoice("openai", "m", "https://attacker.example/v1")
    call_llm("hi", **llm_kwargs(choice, _SERVER))
    assert sent == [("https://attacker.example/v1/chat/completions", "Bearer EMPTY")]


# ---------------------------------------------------------------------------
# Public demo: restricted choices
# ---------------------------------------------------------------------------

_PUBLIC = {**_SERVER, "PUBLIC_DEMO": "1"}
_OPENAI = API_PRESETS["OpenAI"].base_url


def test_no_choice_means_the_server_backend():
    env = {"LLM_PROVIDER": "huggingface"}
    assert llm_kwargs(None, env) == {"provider": "huggingface", "model": "google/flan-t5-base"}
    assert llm_kwargs(None, _PUBLIC) == {"provider": "openai", "model": "served",
                                         "base_url": _URL, "api_key": "server-secret"}


def test_public_server_option_is_the_server_config():
    # The server's own backend (and key, to its own endpoint), whatever else the session holds.
    visitor = BackendChoice(SERVER, "other-model", "https://attacker.example/v1", "sk-visitor")
    assert llm_kwargs(visitor, _PUBLIC) == {"provider": "openai", "model": "served",
                                            "base_url": _URL, "api_key": "server-secret"}


def test_public_server_option_uses_the_servers_hf_model():
    env = {"PUBLIC_DEMO": "yes", "LLM_PROVIDER": "huggingface",
           "HF_MODEL": "someone/not-in-the-allow-list"}
    assert llm_kwargs(BackendChoice(SERVER), env) == {
        "provider": "huggingface", "model": "someone/not-in-the-allow-list"}


@pytest.mark.parametrize("name", list(API_PRESETS))
def test_public_presets_with_the_visitors_key(name):
    preset = API_PRESETS[name]
    choice = BackendChoice("openai", " my-model ", preset.base_url + "/", " sk-visitor ")
    assert llm_kwargs(choice, _PUBLIC) == {"provider": "openai", "model": "my-model",
                                           "base_url": preset.base_url, "api_key": "sk-visitor"}


def test_public_preset_blank_model_means_its_default():
    choice = BackendChoice("openai", "  ", API_PRESETS["Groq"].base_url, "gsk-visitor")
    assert llm_kwargs(choice, _PUBLIC)["model"] == API_PRESETS["Groq"].default_model


@pytest.mark.parametrize("url", [
    "https://api.openai.com.evil.tld/v1",            # look-alike host
    "https://api.openai.com@evil.tld/v1",            # userinfo trick
    "https://evil.tld/https://api.openai.com/v1",
    "https://api.openai.com/v1/../../internal",
    "https://api.openai.com/v1?next=evil",
    "https://api.openai.com/v1.evil.tld",
    "https://api.openai.com/v2",
    "http://api.openai.com/v1",                      # other scheme
    "https://api.openai.com:8443/v1",                # other port
    "https://API.OPENAI.COM/v1",                     # exact match only
    _URL,                                            # the server's own (internal) endpoint
    "http://localhost:11434/v1",                     # a service next to the server
    "http://169.254.169.254/latest",                 # cloud metadata
    "",                                              # "the server's endpoint" (local mode only)
])
def test_public_rejects_any_url_outside_the_presets(url):
    assert preset_for_url(url) is None
    with pytest.raises(ValueError, match="Custom URLs are disabled"):
        llm_kwargs(BackendChoice("openai", "m", url, "sk-visitor"), _PUBLIC)


def test_public_preset_never_gets_the_server_key_even_on_the_servers_endpoint():
    # The owner serves OpenAI with their key; a visitor picks the same OpenAI preset.
    env = {"PUBLIC_DEMO": "1", "LLM_PROVIDER": "openai", "OPENAI_BASE_URL": _OPENAI,
           "OPENAI_API_KEY": "server-secret", "OPENAI_MODEL": "gpt-4o-mini"}
    with pytest.raises(ValueError, match="Enter your OpenAI API key") as err:
        llm_kwargs(BackendChoice("openai", "gpt-4o-mini", _OPENAI, ""), env)
    assert "server-secret" not in str(err.value)
    kwargs = llm_kwargs(BackendChoice("openai", "gpt-4o-mini", _OPENAI, "sk-visitor"), env)
    assert kwargs["api_key"] == "sk-visitor"
    assert llm_kwargs(BackendChoice(SERVER), env)["api_key"] == "server-secret"  # the owner's


@pytest.mark.parametrize("key", ["", "   ", "\t"])
def test_public_empty_key_is_an_error_not_a_fallback(key):
    with pytest.raises(ValueError, match="Enter your Mistral API key"):
        llm_kwargs(BackendChoice("openai", "m", API_PRESETS["Mistral"].base_url, key), _PUBLIC)


@pytest.mark.parametrize("key", ["sk-a b", "sk-a\r\nX-Evil: 1", "sk-é", "sk-" + "x" * 2000])
def test_public_malformed_key_is_rejected_without_echoing_it(key):
    with pytest.raises(ValueError, match="doesn't look valid") as err:
        llm_kwargs(BackendChoice("openai", "m", _OPENAI, key), _PUBLIC)
    assert key.strip() not in str(err.value)


@pytest.mark.parametrize("model", ["two words", "x" * 201, "bad\nname"])
def test_public_malformed_model_name_is_rejected(model):
    with pytest.raises(ValueError, match="model name"):
        llm_kwargs(BackendChoice("openai", model, _OPENAI, "sk-visitor"), _PUBLIC)


def test_public_hf_allow_list():
    for model in DEFAULT_HF_PUBLIC_MODELS:
        assert llm_kwargs(BackendChoice("huggingface", model), _PUBLIC) == {
            "provider": "huggingface", "model": model}
    assert llm_kwargs(BackendChoice("huggingface", " "), _PUBLIC)["model"] == \
        DEFAULT_HF_PUBLIC_MODELS[0]
    with pytest.raises(ValueError, match="must be one of"):
        llm_kwargs(BackendChoice("huggingface", "someone/huge-70b-model"), _PUBLIC)


def test_a_hf_space_without_public_demo_is_restricted_too():
    env = {**_SERVER, "SPACE_ID": "gyom15/rag-vector-hybrid-graph"}
    with pytest.raises(ValueError, match="must be one of"):
        llm_kwargs(BackendChoice("huggingface", "someone/huge-70b-model"), env)
    with pytest.raises(ValueError, match="Custom URLs are disabled"):
        llm_kwargs(BackendChoice("openai", "m", "https://attacker.example/v1"), env)


def test_hf_public_models_env_override():
    env = {**_PUBLIC, "HF_PUBLIC_MODELS": " org/a , org/b,,org/a , "}
    assert public_hf_models(env) == ("org/a", "org/b")
    assert llm_kwargs(BackendChoice("huggingface", "org/b"), env)["model"] == "org/b"
    with pytest.raises(ValueError, match="must be one of: org/a, org/b"):
        llm_kwargs(BackendChoice("huggingface", DEFAULT_HF_PUBLIC_MODELS[0]), env)
    assert public_hf_models({"HF_PUBLIC_MODELS": " , "}) == DEFAULT_HF_PUBLIC_MODELS


@pytest.mark.parametrize("provider", ["ollama", "vllm", ""])
def test_public_rejects_other_providers(provider):
    with pytest.raises(ValueError):
        llm_kwargs(BackendChoice(provider, "llama3.2:3b"), _PUBLIC)


def test_public_preset_on_the_wire_carries_only_the_visitors_key(monkeypatch):
    # End to end: session kwargs -> call_llm -> the (faked) HTTP request actually sent.
    sent = []

    class _Response:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        # urllib stores header names capitalized ("User-agent"). Groq and Together answer
        # urllib's default agent with a 403 before checking the key, so it must be ours.
        user_agent = request.get_header("User-agent") or ""
        assert user_agent and not user_agent.startswith("Python-urllib")
        sent.append((request.full_url, request.get_header("Authorization")))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")  # also in the real env
    groq = API_PRESETS["Groq"].base_url
    call_llm("hi", **llm_kwargs(BackendChoice("openai", "m", groq, "gsk-visitor"), _PUBLIC))
    assert sent == [(f"{groq}/chat/completions", "Bearer gsk-visitor")]
    with pytest.raises(ValueError):  # rejected before any request is built
        call_llm("hi", **llm_kwargs(BackendChoice("openai", "m", groq, ""), _PUBLIC))
    assert len(sent) == 1


def test_choices_never_touch_os_environ(monkeypatch):
    monkeypatch.setenv("PUBLIC_DEMO", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")
    before = dict(os.environ)
    llm_kwargs(BackendChoice("openai", "m", _OPENAI, "sk-visitor"))
    llm_kwargs(BackendChoice("huggingface", DEFAULT_HF_PUBLIC_MODELS[1]))
    judge_settings("sk-visitor")
    with pytest.raises(ValueError):
        llm_kwargs(BackendChoice("openai", "m", _OPENAI, ""))
    assert dict(os.environ) == before


# ---------------------------------------------------------------------------
# Live RAGAS benchmark: caps and judge settings
# ---------------------------------------------------------------------------

def test_ragas_question_caps():
    assert ragas_max_questions(_PUBLIC) == 10
    assert ragas_max_questions({}) == 50
    assert [ragas_questions(n, _PUBLIC) for n in (-3, 0, 1, 5, 10, 11, 10**9)] == \
        [1, 1, 1, 5, 10, 10, 10]
    assert [ragas_questions(n, {}) for n in (0, 30, 51)] == [1, 30, 50]


def test_public_judge_needs_the_visitors_key_and_is_openai():
    env = {**_PUBLIC, "OPENAI_BASE_URL": "http://vllm.internal:8000/v1"}
    with pytest.raises(ValueError, match="Enter your OpenAI API key") as err:
        judge_settings("", env=env)
    assert "server-secret" not in str(err.value)
    assert judge_settings(" sk-visitor ", env=env) == {
        "api_key": "sk-visitor", "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small", "base_url": OPENAI_API_URL}
    with pytest.raises(ValueError, match="can only be OpenAI"):
        judge_settings("sk-visitor", base_url="http://localhost:11434/v1", env=env)
    with pytest.raises(ValueError, match="judge model name"):
        judge_settings("sk-visitor", model="gpt 4", env=env)


def test_local_judge_key_scoping():
    env = {"OPENAI_API_KEY": "server-secret"}  # judge endpoint: OpenAI (no OPENAI_BASE_URL)
    assert judge_settings("", env=env)["api_key"] == "server-secret"
    assert judge_settings("", env=env)["base_url"] == OPENAI_API_URL
    assert judge_settings("sk-typed", env=env)["api_key"] == "sk-typed"
    ollama = judge_settings("", "qwen2.5:1.5b", "nomic-embed-text", "http://localhost:11434/v1",
                            env=env)
    assert ollama == {"api_key": "EMPTY", "model": "qwen2.5:1.5b",
                      "embedding_model": "nomic-embed-text",
                      "base_url": "http://localhost:11434/v1"}
    assert judge_settings("", env={})["api_key"] == ""  # no key: RAGAS reported as failed
    vllm = {**env, "OPENAI_BASE_URL": "http://vllm.internal:8000/v1"}
    assert judge_settings("", env=vllm)["base_url"] == "http://vllm.internal:8000/v1"
    assert judge_settings("", base_url=OPENAI_API_URL, env=vllm)["api_key"] == "EMPTY"


# ---------------------------------------------------------------------------
# Other providers
# ---------------------------------------------------------------------------

def test_non_openai_backends_carry_no_endpoint_or_key():
    kwargs = llm_kwargs(BackendChoice("ollama", "llama3.2:1b", "http://x", "k"), _SERVER)
    assert kwargs == {"provider": "ollama", "model": "llama3.2:1b"}


def test_blank_model_means_the_backend_default():
    assert llm_kwargs(BackendChoice("huggingface", "  "), {})["model"] is None


def test_local_mode_accepts_presets_like_any_url():
    groq = API_PRESETS["Groq"].base_url
    assert llm_kwargs(BackendChoice("openai", "m", groq, "gsk-typed"), _SERVER)["api_key"] == \
        "gsk-typed"
    assert llm_kwargs(BackendChoice("openai", "m", groq), _SERVER)["api_key"] == "EMPTY"


def test_local_server_option_is_the_server_config():
    assert llm_kwargs(BackendChoice(SERVER), _SERVER) == {
        "provider": "openai", "model": "served", "base_url": _URL, "api_key": "server-secret"}
