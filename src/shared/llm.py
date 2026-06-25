"""Interface LLM unifiée et pluggable.

Trois backends, choisis via le paramètre `provider` ou la variable
d'environnement `LLM_PROVIDER` :
  - "ollama"      : inférence locale via Ollama (défaut).
  - "openai"      : tout endpoint compatible OpenAI (OpenAI, **vLLM**, etc.).
  - "huggingface" : modèle encodeur-décodeur local (p. ex. flan-t5).

Les dépendances lourdes (ollama, transformers) sont importées **à la demande** :
importer ce module reste léger quel que soit le backend réellement utilisé.
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
    """Génère une réponse via le backend choisi (`provider`/`LLM_PROVIDER`, défaut ollama).

    `model=None` → modèle par défaut du backend. Lève ValueError si le provider est inconnu.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    try:
        handler = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider}. Choose from {sorted(_PROVIDERS)}.")
    return handler(prompt, model, max_length)


# Variable d'environnement et défaut du modèle, par backend.
_MODEL_ENV = {"ollama": "OLLAMA_MODEL", "openai": "OPENAI_MODEL", "huggingface": "HF_MODEL"}
_MODEL_DEFAULT = {"ollama": "llama3.2:3b", "openai": "default", "huggingface": "google/flan-t5-base"}


def active_config() -> dict:
    """Renvoie le backend LLM actif `{provider, model}` d'après l'environnement.

    Sert à étiqueter les résultats de benchmark (savoir quel modèle a généré).
    """
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv(_MODEL_ENV.get(provider, "OLLAMA_MODEL"), _MODEL_DEFAULT.get(provider, "?"))
    return {"provider": provider, "model": model}


def _ollama_client(host: str):
    """Client Ollama mis en cache par hôte : sa création (dont le contexte SSL) se fait
    une seule fois, pas à chaque appel — plus rapide, et plus robuste sur les longues
    boucles d'évaluation (évite de répéter une I/O fragile des centaines de fois)."""
    if host not in _OLLAMA_CLIENTS:
        import ollama

        _OLLAMA_CLIENTS[host] = ollama.Client(host=host)
    return _OLLAMA_CLIENTS[host]


def _call_ollama(prompt: str, model: str | None, max_length: int) -> str:
    """Inférence locale via Ollama (serveur défini par OLLAMA_URL).

    Température 0 (décodage glouton) → génération **déterministe** : à contexte
    identique, même réponse à chaque exécution. Indispensable pour des verdicts
    reproductibles (sinon les réponses d'un petit modèle flottent d'un run à l'autre).
    """
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    client = _ollama_client(os.getenv("OLLAMA_URL", "http://localhost:11434"))
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0})
    return response["message"]["content"]


def _call_openai(prompt: str, model: str | None, max_length: int) -> str:
    """Endpoint compatible OpenAI (OpenAI, vLLM, etc.) via la lib standard.

    Configuré par OPENAI_BASE_URL (défaut http://localhost:8000/v1, le serveur
    vLLM), OPENAI_API_KEY et OPENAI_MODEL. Aucune dépendance supplémentaire.
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
    """Modèle HuggingFace encodeur-décodeur local (p. ex. flan-t5), mis en cache."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = model or os.getenv("HF_MODEL", "google/flan-t5-base")
    if model not in _HF_CACHE:
        _HF_CACHE[model] = (
            AutoTokenizer.from_pretrained(model),
            AutoModelForSeq2SeqLM.from_pretrained(model),
        )
    tokenizer, llm_model = _HF_CACHE[model]
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = llm_model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


_PROVIDERS: dict[str, Callable[[str, str | None, int], str]] = {
    "ollama": _call_ollama,
    "openai": _call_openai,
    "huggingface": _call_huggingface,
}
