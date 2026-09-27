"""Server-side configuration of the Streamlit app — pure logic, no Streamlit.

The app also runs as a PUBLIC multi-user demo (Hugging Face Space), where every
visitor's session lives in the SAME Python process. The rules below are kept here (free
of Streamlit) so they are unit-tested:
  - a session's LLM backend choice becomes per-call `call_llm` kwargs; it is never
    written to `os.environ`, which would switch the backend — and leak the key — of
    every other visitor;
  - the server's OPENAI_API_KEY is only ever sent to the server-configured
    OPENAI_BASE_URL. A session pointed at another URL only gets a key typed in that
    session: otherwise a visitor could aim the app at their own server and collect
    the owner's secret.

With `PUBLIC_DEMO` on (the default on a Hugging Face Space), visitors still pick their
backend, but only among choices that cannot be turned against the server:
  - "server": the backend the Space is configured with (the default);
  - a hosted OpenAI-compatible API from a fixed preset table (`API_PRESETS`), with the
    VISITOR's own key — never the server's, even when the preset is the server's own
    endpoint. No free-form URL (the server would call any host a visitor names, internal
    ones included: SSRF) and no Ollama (a local-network service);
  - a small local Hugging Face model from an allow-list (`HF_PUBLIC_MODELS`), since each
    new model id is a download onto the Space's disk and a load into its memory.
Anything else raises ValueError with a message meant for the visitor: a choice is
rejected, never silently widened.
"""

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass

from shared.llm import DEFAULT_OPENAI_BASE_URL, active_config

PROVIDERS = ("ollama", "openai", "huggingface")
SERVER = "server"  # pseudo-provider: the backend configured on the server (env)
PUBLIC_CHOICES = (SERVER, "openai", "huggingface")
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ApiPreset:
    """A hosted OpenAI-compatible API a public-demo visitor may use with their own key.

    `default_model` only pre-fills the model field: the name is free text, sent in the
    request body to the preset's endpoint — nothing is downloaded on the server.
    """

    base_url: str
    default_model: str


# The ONLY endpoints a public-demo session can reach with a key (see module docstring).
API_PRESETS: dict[str, ApiPreset] = {
    "OpenAI": ApiPreset("https://api.openai.com/v1", "gpt-4o-mini"),
    "OpenRouter": ApiPreset("https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    "Groq": ApiPreset("https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
    "Together": ApiPreset("https://api.together.xyz/v1",
                          "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "Mistral": ApiPreset("https://api.mistral.ai/v1", "mistral-small-latest"),
}
OPENAI_API_URL = API_PRESETS["OpenAI"].base_url  # the RAGAS judge's endpoint on the demo

# Small enough for a free CPU Space; each one is a download + a load on first use.
DEFAULT_HF_PUBLIC_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/flan-t5-base",
)

# Live RAGAS benchmark: each question costs 3 generations (one per stack) + judge calls.
RAGAS_DEFAULT_QUESTIONS = 5
RAGAS_MAX_QUESTIONS_PUBLIC = 10
RAGAS_MAX_QUESTIONS_LOCAL = 50
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_EMBEDDINGS = "text-embedding-3-small"
# On the public demo, one live RAGAS run at a time (each holds the shared CPU for minutes).
# A module global, not an `st.cache_resource`: any browser can send Streamlit's
# `clear_cache` message, which empties every cached resource for all sessions and would
# hand out a new, free lock while a run still holds the old one. An imported module stays
# in `sys.modules` across reruns and cache clears (the Space runs without the file watcher).
RAGAS_RUN_LOCK = threading.Lock()

_MAX_NAME_CHARS = 200   # a model id, not a document
_MAX_KEY_CHARS = 1024   # real keys are < 200 chars; anything longer is not a key


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def is_public_demo(env: Mapping[str, str] | None = None) -> bool:
    """Whether visitors are restricted to the public-demo choices (see module docstring).

    `PUBLIC_DEMO` = 1/true/yes/on (any case) -> yes; 0/false/no/off -> no. Unset (or
    anything else): yes only on a Hugging Face Space, detected by the `SPACE_ID` that HF
    sets in every Space. Fail closed: a Space deployed without the variable stays
    restricted; its owner opts out explicitly (PUBLIC_DEMO=0, e.g. a private Space).
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


def public_hf_models(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Hugging Face models a public-demo visitor may load: `HF_PUBLIC_MODELS`
    (comma-separated) if set and non-blank, else `DEFAULT_HF_PUBLIC_MODELS`."""
    raw = _env(env).get("HF_PUBLIC_MODELS", "")
    models = tuple(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
    return models or DEFAULT_HF_PUBLIC_MODELS


@dataclass(frozen=True)
class BackendChoice:
    """A session's LLM backend (kept in `st.session_state`, never in `os.environ`)."""

    provider: str       # one of PROVIDERS, or SERVER
    model: str = ""     # "" -> the backend's default (env, or the preset's)
    base_url: str = ""  # openai only; "" -> the server's endpoint (local mode only)
    api_key: str = ""   # openai only; a key typed in THIS session (never the server's)


def server_choice(env: Mapping[str, str] | None = None) -> BackendChoice:
    """The backend configured on the server (env) — the "server" choice."""
    config = active_config(_env(env))
    base_url = server_base_url(env) if config["provider"] == "openai" else ""
    return BackendChoice(config["provider"], config["model"], base_url)


def _normalize_url(url: str) -> str:
    # Same normalization as the openai backend applies before building the request URL.
    return url.strip().rstrip("/")


def _same_endpoint(a: str, b: str) -> bool:
    return _normalize_url(a) == _normalize_url(b)


def preset_for_url(base_url: str) -> str | None:
    """Name of the preset whose endpoint is exactly `base_url` (after the trailing-slash
    normalization), else None. Exact match: a prefix or look-alike host is no preset."""
    for name, preset in API_PRESETS.items():
        if _same_endpoint(base_url, preset.base_url):
            return name
    return None


def resolve_api_key(base_url: str, typed_key: str, env: Mapping[str, str] | None = None) -> str:
    """Key to send to `base_url`: the one typed in the session if any; else the server's
    key, but ONLY if `base_url` is the server's endpoint; else "EMPTY" (no key)."""
    env = _env(env)
    if typed_key.strip():
        return typed_key.strip()
    if _same_endpoint(base_url, server_base_url(env)):
        return env.get("OPENAI_API_KEY") or "EMPTY"
    return "EMPTY"


def _check_name(value: str, what: str) -> str:
    """A model id typed by a visitor: non-empty, short, one printable token."""
    value = value.strip()
    if not value:
        raise ValueError(f"Enter a {what}.")
    if len(value) > _MAX_NAME_CHARS or not value.isprintable() or any(c.isspace() for c in value):
        raise ValueError(f"That {what} doesn't look valid (one word, at most "
                         f"{_MAX_NAME_CHARS} characters).")
    return value


def _check_visitor_key(key: str, what: str) -> str:
    """A key typed by a visitor: required (the server's is never lent), and plain ASCII
    with no spaces, since it travels in an HTTP header. The message never echoes it."""
    key = key.strip()
    if not key:
        raise ValueError(f"Enter your {what} API key: the public demo never uses the "
                         "server's.")
    if len(key) > _MAX_KEY_CHARS or not (key.isascii() and key.isprintable()) or " " in key:
        raise ValueError(f"That {what} API key doesn't look valid (unexpected characters).")
    return key


def _public_kwargs(choice: BackendChoice, env: Mapping[str, str]) -> dict:
    """`call_llm` kwargs for a public-demo session, or ValueError if the choice is not
    allowed there (see module docstring)."""
    if choice.provider == SERVER:
        return _local_kwargs(server_choice(env), env)
    if choice.provider == "openai":
        name = preset_for_url(choice.base_url)
        if name is None:
            raise ValueError("The public demo only reaches these APIs: "
                             + ", ".join(API_PRESETS) + ". Custom URLs are disabled.")
        preset = API_PRESETS[name]
        return {"provider": "openai",
                "model": _check_name(choice.model.strip() or preset.default_model,
                                     "model name"),
                "base_url": preset.base_url,  # the table's URL, not the session's string
                "api_key": _check_visitor_key(choice.api_key, name)}
    if choice.provider == "huggingface":
        allowed = public_hf_models(env)
        model = choice.model.strip() or allowed[0]
        if model not in allowed:
            raise ValueError("On the public demo, the local model must be one of: "
                             + ", ".join(allowed) + ".")
        return {"provider": "huggingface", "model": model}
    if choice.provider == "ollama":
        raise ValueError("Ollama isn't available on the public demo: pick a hosted API "
                         "(with your key) or a small local model.")
    raise ValueError(f"Unknown backend {choice.provider!r}.")


def _local_kwargs(choice: BackendChoice, env: Mapping[str, str]) -> dict:
    """`call_llm` kwargs outside public-demo mode: free provider, model and base URL; the
    server's key still only goes to the server's endpoint."""
    if choice.provider == SERVER:
        choice = server_choice(env)
    kwargs = {"provider": choice.provider, "model": choice.model.strip() or None}
    if choice.provider == "openai":
        base_url = choice.base_url.strip() or server_base_url(env)
        kwargs["base_url"] = base_url
        kwargs["api_key"] = resolve_api_key(base_url, choice.api_key, env)
    return kwargs


def llm_kwargs(choice: BackendChoice | None, env: Mapping[str, str] | None = None) -> dict:
    """`call_llm` kwargs for a session: its `choice`, or the server's backend when there
    is none. In public-demo mode the choice must be one of the allowed ones, else
    ValueError (a message for the visitor: show it, send nothing)."""
    env = _env(env)
    if choice is None:
        choice = BackendChoice(SERVER)
    if is_public_demo(env):
        return _public_kwargs(choice, env)
    return _local_kwargs(choice, env)


# ---------------------------------------------------------------------------
# Live RAGAS benchmark
# ---------------------------------------------------------------------------

def ragas_max_questions(env: Mapping[str, str] | None = None) -> int:
    """Question cap of one live RAGAS run: small on the public demo, where one run
    holds the shared CPU (3 generations per question) for everyone."""
    return RAGAS_MAX_QUESTIONS_PUBLIC if is_public_demo(env) else RAGAS_MAX_QUESTIONS_LOCAL


def ragas_questions(requested: int, env: Mapping[str, str] | None = None) -> int:
    """`requested` clamped to [1, cap]. Enforced on the server: a widget's min/max only
    bind the browser, and a crafted client can send any number."""
    return max(1, min(int(requested), ragas_max_questions(env)))


def judge_base_url(env: Mapping[str, str] | None = None) -> str:
    """Endpoint of the RAGAS judge the server's key belongs to (local mode): the
    configured OPENAI_BASE_URL, else OpenAI — as the command-line benchmark does."""
    return _env(env).get("OPENAI_BASE_URL") or OPENAI_API_URL


def judge_settings(
    api_key: str,
    model: str = "",
    embedding_model: str = "",
    base_url: str = "",
    env: Mapping[str, str] | None = None,
) -> dict:
    """Keyword arguments of the RAGAS judge (`shared.evaluator.JudgeConfig`) for a session.

    Public demo: the visitor's own OpenAI key is required and the endpoint is fixed to
    OpenAI (a URL outside it raises ValueError) — the server's key is never lent.
    Local mode: optional endpoint (e.g. Ollama or vLLM for tests; "" -> `judge_base_url`);
    the key typed in the session, else the server's key but only for `judge_base_url`,
    else "EMPTY" for another endpoint (a key-less local server) or "" (no key: the
    benchmark then reports RAGAS as failed and keeps the latencies).
    """
    env = _env(env)
    model = model.strip() or DEFAULT_JUDGE_MODEL
    embedding_model = embedding_model.strip() or DEFAULT_JUDGE_EMBEDDINGS
    if is_public_demo(env):
        if base_url.strip() and not _same_endpoint(base_url, OPENAI_API_URL):
            raise ValueError("On the public demo, the RAGAS judge can only be OpenAI.")
        return {"api_key": _check_visitor_key(api_key, "OpenAI"),
                "model": _check_name(model, "judge model name"),
                "embedding_model": _check_name(embedding_model, "embedding model name"),
                "base_url": OPENAI_API_URL}

    base_url = base_url.strip() or judge_base_url(env)
    if api_key.strip():
        key = api_key.strip()
    elif _same_endpoint(base_url, judge_base_url(env)):
        key = env.get("OPENAI_API_KEY", "")
    else:
        key = "EMPTY"
    return {"api_key": key, "model": model, "embedding_model": embedding_model,
            "base_url": base_url}
