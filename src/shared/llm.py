"""Unified, pluggable LLM interface.

Three backends, selected via the `provider` parameter or the
`LLM_PROVIDER` environment variable:
  - "ollama"      : local inference via Ollama (default).
  - "openai"      : any OpenAI-compatible endpoint (OpenAI, **vLLM**, etc.).
  - "huggingface" : local model — encoder-decoder (flan-t5) or instruct decoder
                    (Qwen2.5-Instruct…); the kind is auto-detected.

The environment is only the *default* configuration: a caller can override the
endpoint and key per call (`call_llm(..., base_url=, api_key=)`). The Streamlit app
relies on this — its visitors share one process, so a visitor's choice must never
be written to the process-wide environment.

The heavy dependencies (ollama, transformers) are imported **on demand**:
importing this module stays lightweight regardless of the backend actually used.
"""

import os
import threading
from collections.abc import Callable, Mapping

from .prompts import QUESTION_MARKER

DEFAULT_OPENAI_BASE_URL = "http://localhost:8000/v1"  # the vLLM server
_DEFAULT_TIMEOUT_S = 120.0
# flan-t5 was trained on 512-token inputs; also the cap when a tokenizer reports no limit.
_SEQ2SEQ_MAX_INPUT = 512
# Input cap of the instruct (causal) path. The app's largest prompt (k=10 chunks of ~500
# chars + a 500-char question) stays under ~2k tokens, so this only bites on abnormal input,
# which would otherwise run unbounded through generate() while holding _HF_LOCK.
_CAUSAL_MAX_INPUT = 4096

# At most ONE HF model in memory (a CPU Space can't hold several). The lock serializes
# loading AND generation: concurrent sessions don't load the same model twice, and on a
# CPU, parallel generations would only thrash.
_HF_CACHE: dict[str, tuple] = {}
_HF_LOCK = threading.Lock()
_OLLAMA_CLIENTS: dict[str, object] = {}


def call_llm(
    prompt: str,
    model: str | None = None,
    provider: str | None = None,
    max_length: int = 512,
    **options,
) -> str:
    """Generate an answer via the selected backend (`provider`/`LLM_PROVIDER`, default ollama).

    `model=None` -> backend default model. `options` are forwarded to the backend — for
    "openai": `base_url` / `api_key`, which override the env for this call only.
    Raises ValueError if the provider is unknown.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    try:
        handler = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider}. Choose from {sorted(_PROVIDERS)}.")
    return handler(prompt, model, max_length, **options)


# Environment variable and model default, per backend.
_MODEL_ENV = {"ollama": "OLLAMA_MODEL", "openai": "OPENAI_MODEL", "huggingface": "HF_MODEL"}
_MODEL_DEFAULT = {"ollama": "llama3.2:3b", "openai": "default", "huggingface": "google/flan-t5-base"}


def default_model(provider: str, env: Mapping[str, str] | None = None) -> str:
    """Model a backend uses when none is given: its env variable, else the built-in default."""
    env = os.environ if env is None else env
    return env.get(_MODEL_ENV.get(provider, "OLLAMA_MODEL"), _MODEL_DEFAULT.get(provider, "?"))


def active_config(env: Mapping[str, str] | None = None) -> dict:
    """Return the active LLM backend `{provider, model}` based on the environment.

    Used to label the benchmark results (to know which model generated them).
    """
    env = os.environ if env is None else env
    provider = env.get("LLM_PROVIDER", "ollama")
    return {"provider": provider, "model": default_model(provider, env)}


def _timeout() -> float:
    """Max seconds for one call to a remote backend (`LLM_TIMEOUT`, default 120): a dead
    or stalled endpoint must fail the request, not hang the app's worker forever."""
    return float(os.getenv("LLM_TIMEOUT") or _DEFAULT_TIMEOUT_S)


def _ollama_client(host: str):
    """Ollama client cached per host: its creation (including the SSL context) happens
    only once, not on every call -- faster, and more robust over long
    evaluation loops (avoids repeating a fragile I/O hundreds of times)."""
    if host not in _OLLAMA_CLIENTS:
        import ollama

        _OLLAMA_CLIENTS[host] = ollama.Client(host=host, timeout=_timeout())
    return _OLLAMA_CLIENTS[host]


def _call_ollama(prompt: str, model: str | None, max_length: int) -> str:
    """Local inference via Ollama (server defined by OLLAMA_URL).

    Temperature 0 (greedy decoding) -> **deterministic** generation: with an
    identical context, the same answer on every run. Essential for
    reproducible verdicts (otherwise a small model's answers drift from one run to the next).
    """
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    client = _ollama_client(os.getenv("OLLAMA_URL", "http://localhost:11434"))
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0})
    return response["message"]["content"]


def _call_openai(
    prompt: str,
    model: str | None,
    max_length: int,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """OpenAI-compatible endpoint (OpenAI, vLLM, etc.) via the standard library.

    `base_url` / `api_key` default to OPENAI_BASE_URL (default http://localhost:8000/v1,
    the vLLM server) / OPENAI_API_KEY — the env is read only when they are None — and
    the model to OPENAI_MODEL. No additional dependency.
    """
    import json
    import urllib.parse
    import urllib.request

    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = base_url.strip().rstrip("/")
    if urllib.parse.urlsplit(base_url).scheme not in ("http", "https"):  # no file://, ftp://…
        raise ValueError(f"The OpenAI base URL must be http(s), got {base_url!r}.")
    model = model or os.getenv("OPENAI_MODEL", "default")

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_length,
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key or 'EMPTY'}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_timeout()) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def fit_prompt(prompt: str, tokenizer, max_tokens: int, marker: str = QUESTION_MARKER) -> str:
    """Shorten `prompt` to at most `max_tokens` tokens WITHOUT losing the question.

    The template ends with the question (after `marker`), so a plain right-side
    truncation cuts off exactly what the model must answer. Instead, the tail (from the
    last `marker`) is kept whole and only the head (instruction + context) is cut, from
    its end: the instruction and the top-ranked chunks, which come first, survive.
    Without the marker, falls back to plain truncation. Used by both HF paths: to fit
    flan-t5's 512-token encoder, and as a safety cap on an instruct model's input.

    `tokenizer` only needs `encode(text, add_special_tokens=False) -> list` and
    `decode(ids, skip_special_tokens=True) -> str` (any HF tokenizer).
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt
    cut = prompt.rfind(marker)
    if cut == -1:
        return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    head, tail = prompt[:cut], prompt[cut:]
    head_ids = tokenizer.encode(head, add_special_tokens=False)
    tail_ids = tokenizer.encode(tail, add_special_tokens=False)
    room = max_tokens - len(tail_ids)
    # Decoding + re-joining can shift a token at the boundary: re-measure and shrink until
    # it fits, so the caller's safety-net truncation never has to bite into the question.
    while room > 0:
        fitted = tokenizer.decode(head_ids[:room], skip_special_tokens=True) + tail
        overflow = len(tokenizer.encode(fitted, add_special_tokens=False)) - max_tokens
        if overflow <= 0:
            return fitted
        room -= overflow
    # The question alone exceeds the budget: keep its end (with the "Answer:" cue).
    return tokenizer.decode(tail_ids[-max_tokens:], skip_special_tokens=True)


def _load_hf_model(model: str) -> tuple:
    """Load `(kind, tokenizer, model)`; the kind is auto-detected from the model config."""
    from transformers import (
        AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    if AutoConfig.from_pretrained(model).is_encoder_decoder:
        return "seq2seq", tokenizer, AutoModelForSeq2SeqLM.from_pretrained(model)
    return "causal", tokenizer, AutoModelForCausalLM.from_pretrained(model)


def _hf_model(model: str) -> tuple:
    """Cached `_load_hf_model`, keeping only the last model used. Call under `_HF_LOCK`."""
    if model not in _HF_CACHE:
        _HF_CACHE.clear()  # free the previous model BEFORE loading the next one
        _HF_CACHE[model] = _load_hf_model(model)
    return _HF_CACHE[model]


def _call_huggingface(prompt: str, model: str | None, max_length: int) -> str:
    """Local HuggingFace model, cached. Handles both encoder-decoder models
    (e.g. flan-t5) and instruct decoder/causal-LM models (e.g. Qwen2.5-Instruct);
    the kind is auto-detected from the model config."""
    model = model or os.getenv("HF_MODEL", "google/flan-t5-base")
    with _HF_LOCK:
        kind, tokenizer, llm_model = _hf_model(model)

        if kind == "causal":
            # Instruct decoder: render the chat template, decode only the new tokens. The
            # input is bounded first (question kept): generation holds the global lock.
            content = fit_prompt(prompt, tokenizer, _CAUSAL_MAX_INPUT)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt")
            outputs = llm_model.generate(**model_inputs, max_new_tokens=max_length, do_sample=False)
            new_tokens = outputs[0][model_inputs.input_ids.shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Encoder-decoder: the input must fit the encoder, question included (see
        # fit_prompt); `max_length` only bounds the *answer* (new tokens).
        limit = min(getattr(tokenizer, "model_max_length", _SEQ2SEQ_MAX_INPUT), _SEQ2SEQ_MAX_INPUT)
        text = fit_prompt(prompt, tokenizer, limit - tokenizer.num_special_tokens_to_add())
        inputs = tokenizer(text, return_tensors="pt", max_length=limit, truncation=True)
        outputs = llm_model.generate(**inputs, max_new_tokens=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


_PROVIDERS: dict[str, Callable[..., str]] = {
    "ollama": _call_ollama,
    "openai": _call_openai,
    "huggingface": _call_huggingface,
}
