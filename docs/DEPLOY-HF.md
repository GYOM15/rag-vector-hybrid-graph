# Deploying the demo to Hugging Face Spaces

Hosts the Streamlit app (the **evaluation dashboard** + a working **Chat**) on a free
HF Space. The dashboard visualizes the committed reference results; the Chat generates
with a small local model (**flan-t5**) since the Space has no GPU and no Ollama.

## What runs where

| Feature | On the hosted Space |
|---|---|
| Dashboard — BEIR / Reranking / Systems / Answer quality | ✅ full (reads committed `eval/reference/*.json`) |
| Live Guard / Toy retrieval | ✅ builds on the Space CPU (fast) |
| Chat (3 architectures side by side) | ✅ retrieves + generates with **flan-t5-base** (CPU) — basic but works for any visitor, no setup |
| RAGAS (live) tab | ⚠️ needs `ragas` + an OpenAI key (not in the hosted requirements) → shows a graceful message; run it locally |

## Steps

### 1. Create the Space
- Hugging Face account → <https://huggingface.co> → **New** → **Space**.
- Owner = you · name = e.g. `rag-vector-hybrid-graph` · License = MIT.
- **SDK = Streamlit** · Hardware = **CPU basic (free)** · Visibility = Public → **Create**.

### 2. Configure the Space README
A Space is configured by a YAML header at the top of its `README.md`. Set:
```yaml
---
title: RAG Vector vs Hybrid vs Graph
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---
```
`app_file` points HF at our app; the rest is cosmetic.

### 3. Set the environment variables
Space → **Settings** → **Variables and secrets**:
- `LLM_PROVIDER = huggingface` → the Chat uses flan-t5 by default (no Ollama on the Space).
- `DEMO_ARTICLES = 200` *(optional)* → smaller corpus = faster first build.

### 4. Push the code
Two options:
- **Link GitHub**: Space → Settings → *Link a GitHub repository* and let it sync, **or**
- **Push manually** to the Space's own git repo:
  ```bash
  git clone https://huggingface.co/spaces/<you>/<space-name>
  # copy the project into it: src/  app/  eval/  docs/  .streamlit/  requirements.txt
  #   and the README.md carrying the YAML header from step 2
  cd <space-name> && git add -A && git commit -m "Deploy app" && git push
  ```
`requirements.txt` (already in this repo) tells the Space what to install — including the
spaCy NER model. Keep `.streamlit/config.toml` for the dark theme.

### 5. First load
On the first visit the Space builds the 3 stacks (download the Wikipedia articles +
embeddings + spaCy NER + BM25 + entity graph) → ~1–4 min, then cached. Free Spaces sleep
after inactivity, so the first visit after a sleep rebuilds.

## Notes
- **Memory**: the free tier (~16 GB) comfortably fits torch + faiss + spaCy + flan-t5.
- **Generation quality**: flan-t5-base is small — a demo. The full-quality, batched
  generation is the AWS/vLLM endpoint (see [`infra/DEPLOY.md`](../infra/DEPLOY.md)); once
  it is up, point the Space at it with `LLM_PROVIDER=openai` + `OPENAI_BASE_URL`.
- **Credentials**: creating the Space and pushing require *your* Hugging Face account —
  that part is yours to do; everything in this repo is ready.
