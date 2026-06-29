"""Unified, pluggable LLM interface.

Three backends, selected via the `provider` parameter or the
`LLM_PROVIDER` environment variable:
  - "ollama"      : local inference via Ollama (default).
  - "openai"      : any OpenAI-compatible endpoint (OpenAI, **vLLM**, etc.).
  - "huggingface" : local model — encoder-decoder (flan-t5) or instruct decoder
                    (Qwen2.5-Instruct…); the kind is auto-detected.

The heavy dependencies (ollama, transformers) are imported **on demand**:
importing this module stays lightweight regardless of the backend actually used.
"""

import os
from typing import Callable

_HF_CACHE: dict[str, tuple] = {}
_OLLAMA_CLIENTS: dict[str, object] = {}


def call_llm(
    prompt: str,
    model: str | None = None,
    provider: str | None = None,
    max_length: int = 512,
) -> str:
    """Generate an answer via the selected backend (`provider`/`LLM_PROVIDER`, default ollama).

    `model=None` -> backend default model. Raises ValueError if the provider is unknown.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    try:
        handler = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider}. Choose from {sorted(_PROVIDERS)}.")
    return handler(prompt, model, max_length)


# Environment variable and model default, per backend.
_MODEL_ENV = {"ollama": "OLLAMA_MODEL", "openai": "OPENAI_MODEL", "huggingface": "HF_MODEL"}
_MODEL_DEFAULT = {"ollama": "llama3.2:3b", "openai": "default", "huggingface": "google/flan-t5-base"}


def active_config() -> dict:
    """Return the active LLM backend `{provider, model}` based on the environment.

    Used to label the benchmark results (to know which model generated them).
    """
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv(_MODEL_ENV.get(provider, "OLLAMA_MODEL"), _MODEL_DEFAULT.get(provider, "?"))
    return {"provider": provider, "model": model}


def _ollama_client(host: str):
    """Ollama client cached per host: its creation (including the SSL context) happens
    only once, not on every call -- faster, and more robust over long
    evaluation loops (avoids repeating a fragile I/O hundreds of times)."""
    if host not in _OLLAMA_CLIENTS:
        import ollama

        _OLLAMA_CLIENTS[host] = ollama.Client(host=host)
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


def _call_openai(prompt: str, model: str | None, max_length: int) -> str:
    """OpenAI-compatible endpoint (OpenAI, vLLM, etc.) via the standard library.

    Configured by OPENAI_BASE_URL (default http://localhost:8000/v1, the
    vLLM server), OPENAI_API_KEY and OPENAI_MODEL. No additional dependency.
    """
    import json
    import urllib.request

    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def _call_huggingface(prompt: str, model: str | None, max_length: int) -> str:
    """Local HuggingFace model, cached. Handles both encoder-decoder models
    (e.g. flan-t5) and instruct decoder/causal-LM models (e.g. Qwen2.5-Instruct);
    the kind is auto-detected from the model config."""
    from transformers import (
        AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
    )

    model = model or os.getenv("HF_MODEL", "google/flan-t5-base")
    if model not in _HF_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model)
        if AutoConfig.from_pretrained(model).is_encoder_decoder:
            _HF_CACHE[model] = ("seq2seq", tokenizer, AutoModelForSeq2SeqLM.from_pretrained(model))
        else:
            _HF_CACHE[model] = ("causal", tokenizer, AutoModelForCausalLM.from_pretrained(model))
    kind, tokenizer, llm_model = _HF_CACHE[model]

    if kind == "causal":  # instruct decoder: use the chat template, decode only the new tokens
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
        outputs = llm_model.generate(inputs, max_new_tokens=max_length, do_sample=False)
        return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = llm_model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


_PROVIDERS: dict[str, Callable[[str, str | None, int], str]] = {
    "ollama": _call_ollama,
    "openai": _call_openai,
    "huggingface": _call_huggingface,
}
