"""Server-side configuration of the Streamlit app — pure logic, no Streamlit.

The app also runs as a PUBLIC multi-user demo (Hugging Face Space), where every
visitor's session lives in the SAME Python process. Two rules follow, kept here (free
of Streamlit) so they are unit-tested:
  - a session's LLM backend choice becomes per-call `call_llm` kwargs; it is never
    written to `os.environ`, which would switch the backend — and leak the key — of
    every other visitor;
  - the server's OPENAI_API_KEY is only ever sent to the server-configured
    OPENAI_BASE_URL. A session pointed at another URL only gets a key typed in that
    session: otherwise a visitor could aim the app at their own server and collect
    the owner's secret.
With `PUBLIC_DEMO` on, visitors can't change the backend at all (no endpoint to
redirect, no model id to download): every session uses the server's. It fails closed:
on a Hugging Face Space it is on unless explicitly turned off.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from shared.llm import DEFAULT_OPENAI_BASE_URL, active_config

PROVIDERS = ("ollama", "openai", "huggingface")
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def is_public_demo(env: Mapping[str, str] | None = None) -> bool:
    """Whether visitors are barred from configuring the backend.

    `PUBLIC_DEMO` = 1/true/yes/on (any case) -> yes; 0/false/no/off -> no. Unset (or
    anything else): yes only on a Hugging Face Space, detected by the `SPACE_ID` that HF
    sets in every Space. Fail closed: a Space deployed without the variable stays locked;
    its owner opts out explicitly (PUBLIC_DEMO=0, e.g. a private Space).
    """
    env = _env(env)
    value = env.get("PUBLIC_DEMO", "").strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return bool(env.get("SPACE_ID", "").strip())


def demo_articles(env: Mapping[str, str] | None = None) -> int:
    """Corpus size the stacks are built with (`DEMO_ARTICLES`, default 500)."""
    return int(_env(env).get("DEMO_ARTICLES") or 500)


def server_base_url(env: Mapping[str, str] | None = None) -> str:
    """The server-configured OpenAI-compatible endpoint — the only one its key goes to."""
    return _env(env).get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL


@dataclass(frozen=True)
class BackendChoice:
    """A session's LLM backend (kept in `st.session_state`, never in `os.environ`)."""

    provider: str
    model: str = ""     # "" -> the backend's default (env)
    base_url: str = ""  # openai only; "" -> the server's endpoint
    api_key: str = ""   # openai only; a key typed in THIS session (never the server's)


def server_choice(env: Mapping[str, str] | None = None) -> BackendChoice:
    """The backend configured on the server (env) — the only one in public-demo mode."""
    config = active_config(_env(env))
    base_url = server_base_url(env) if config["provider"] == "openai" else ""
    return BackendChoice(config["provider"], config["model"], base_url)


def _same_endpoint(a: str, b: str) -> bool:
    # Same normalization as the openai backend applies before building the request URL.
    return a.strip().rstrip("/") == b.strip().rstrip("/")


def resolve_api_key(base_url: str, typed_key: str, env: Mapping[str, str] | None = None) -> str:
    """Key to send to `base_url`: the one typed in the session if any; else the server's
    key, but ONLY if `base_url` is the server's endpoint; else "EMPTY" (no key)."""
    env = _env(env)
    if typed_key.strip():
        return typed_key.strip()
    if _same_endpoint(base_url, server_base_url(env)):
        return env.get("OPENAI_API_KEY") or "EMPTY"
    return "EMPTY"


def llm_kwargs(choice: BackendChoice | None, env: Mapping[str, str] | None = None) -> dict:
    """`call_llm` kwargs for a session: its `choice`, or the server's backend when there is
    none or in public-demo mode (whatever the session holds is then ignored)."""
    env = _env(env)
    if choice is None or is_public_demo(env):
        choice = server_choice(env)
    kwargs = {"provider": choice.provider, "model": choice.model.strip() or None}
    if choice.provider == "openai":
        base_url = choice.base_url.strip() or server_base_url(env)
        kwargs["base_url"] = base_url
        kwargs["api_key"] = resolve_api_key(base_url, choice.api_key, env)
    return kwargs
