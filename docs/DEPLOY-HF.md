# Deploying the demo to Hugging Face Spaces — full reproducible guide

End-to-end, beginner-friendly walkthrough to host the Streamlit app (the **evaluation
dashboard** + a working **Chat**) on a **free** Hugging Face Space. Includes the exact
commands and the gotchas we actually hit, so it reproduces cleanly.

Running example used below: HF user `gyom15`, Space `rag-vector-hybrid-graph`
→ <https://huggingface.co/spaces/gyom15/rag-vector-hybrid-graph>. Replace with yours.

## What runs where

| Feature | On the hosted Space |
|---|---|
| Dashboard — BEIR / Reranking / Systems / Answer quality | ✅ full (reads committed `eval/reference/*.json`), **loads instantly** |
| Live Guard / Retrieval by type | ✅ builds the 3 stacks on first click (~1–3 min, then cached) |
| Chat (3 architectures side by side) | ✅ retrieves + generates with **flan-t5-base** (CPU) — works for any visitor, no setup; basic answers |
| RAGAS (live) tab | ⚠️ needs `ragas` + an OpenAI key (not in the hosted requirements) → graceful message; run locally |

The Streamlit app's `get_stacks()` is `@st.cache_resource`, and it's only called on the
first Chat/retrieval interaction — so the **initial page load is fast** (the dashboard is
pure JSON rendering); the corpus build is deferred and cached.

## Prerequisites
- A Hugging Face account — <https://huggingface.co> → Sign Up (verify your email).
- Git installed locally (macOS already has it).
- The project checked out locally.

---

## Step 1 — Create the Space (web UI, ~2 min)
1. Go to **<https://huggingface.co/new-space>** (or avatar → **New** → **Space**).
2. **Owner** = you · **Space name** = e.g. `rag-vector-hybrid-graph` · **License** = MIT.
3. **Select the SDK** → **Streamlit**.
4. **Hardware** → **CPU basic** (free) · **Visibility** → **Public**.
5. **Create Space**. You land on the Space page (tabs: *App*, *Files*, *Settings*, *Logs*).

## Step 2 — Create a Write access token (~1 min)
Avatar → **Settings** → **Access Tokens** → **New token** → name `deploy`, role **Write**
→ **Create** → **copy** it (`hf_…`). You'll use it as the git push password.

## Step 3 — Authenticate the CLI
```bash
hf auth login
```
> ⚠️ **Gotcha:** `huggingface-cli login` is **deprecated** (recent `huggingface_hub`); use
> **`hf auth login`**. It opens a device flow — visit the printed URL, enter the code, done.

## Step 4 — Clone the Space's git repo
The Space is its *own* git repo. Clone it somewhere **outside** your project:
```bash
cd ~
git clone https://huggingface.co/spaces/gyom15/rag-vector-hybrid-graph
cd rag-vector-hybrid-graph
```
> ⚠️ **Gotchas:** use your **real** username (not a literal `<placeholder>` — `< >` makes
> zsh try a file redirect and fail). And the Space must already exist (Step 1), otherwise
> the clone returns `Repository not found`.

## Step 5 — Copy the project files into the Space repo
```bash
SRC="/absolute/path/to/rag-vector-hybrid-graph"   # your project's path
cp -R "$SRC/src" "$SRC/app" "$SRC/eval" "$SRC/docs" "$SRC/.streamlit" \
      "$SRC/requirements.txt" "$SRC/.gitignore" .
rm -f app.py   # remove HF's starter demo app at the root; ours is app/streamlit_app.py
```
> Copying `.gitignore` matters: the scratch `eval/*.json` outputs stay uncommitted, but the
> committed `eval/reference/*.json` (the dashboard's numbers) **are** included. `cp` printing
> nothing means success.

## Step 6 — Write the Space README (with the right `app_file`)
HF configures a Space from a YAML header in its `README.md`. The default points at `app.py`;
ours is `app/streamlit_app.py`, so set it explicitly. Easiest — overwrite the README:
```bash
cat > README.md <<'EOF'
---
title: RAG Vector vs Hybrid vs Graph
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---

# RAG: Vector vs Hybrid vs Graph — hosted demo

Three retrieval architectures (Vector / Hybrid / Graph) compared, with an in-app
evaluation dashboard. Source: https://github.com/GYOM15/rag-vector-hybrid-graph
EOF
```

## Step 7 — Commit and push
```bash
git add -A
git commit -m "Deploy the RAG comparison app"
git push
```
> 🔑 If git prompts for credentials: **username** = your HF name, **password** = the **Write
> token** from Step 2 (not your account password).

The push triggers a build automatically.

## Step 8 — Set environment variables (web UI)
Space → **Settings** → **Variables and secrets** → **New variable** (a *Variable*, not a secret):

| Name | Value | Why |
|---|---|---|
| `LLM_PROVIDER` | `huggingface` | the Chat uses flan-t5 (no Ollama on the Space) |
| `DEMO_ARTICLES` | `200` | smaller corpus = faster first build |

Adding variables restarts the Space.

## Step 9 — Watch it build, then run
- Top of the page: **Building** (installs torch/faiss/transformers/spaCy — a few minutes) →
  **Running**.
- Open the **App** tab. The dashboard appears immediately; the Chat/retrieval build the 3
  stacks on first use.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'X'` at startup** (we hit this with `snowballstemmer`).
The hosted install comes only from `requirements.txt`. Make sure it lists every runtime
import — for this app: `streamlit, sentence-transformers, transformers, faiss-cpu,
rank-bm25, snowballstemmer, networkx, spacy, numpy, python-dotenv, datasets` + the spaCy
model wheel. Fix and re-push:
```bash
echo "the-missing-package" >> requirements.txt
git add requirements.txt && git commit -m "Add missing dependency" && git push
```
(`ollama` and `ragas` are intentionally absent — they're lazy-imported and not needed for
the default hosted path.)

**Build error (red) / runtime error.** Open the **Logs** tab; the traceback's bottom line is
the cause. Most failures are a missing dependency (fix as above) or a wrong `app_file`.

**The Space "sleeps".** Free Spaces pause after ~48 h of inactivity; the next visit wakes
them (~1–4 min rebuild). For an always-on demo, upgrade the Space hardware (paid).

## Notes
- **Memory**: the free tier (~16 GB) fits torch + faiss + spaCy + flan-t5.
- **Generation quality**: flan-t5-base is a small demo model. The full-quality, batched
  generation is the AWS/vLLM endpoint (see [`infra/DEPLOY.md`](../infra/DEPLOY.md)); once it
  is up, point the Space at it with `LLM_PROVIDER=openai` + `OPENAI_BASE_URL`.
- **Updating the Space later**: re-copy the changed files into the Space clone and
  `git add -A && git commit && git push`, or link the Space to GitHub for auto-sync.
- **Credentials**: creating the Space, the token, and pushing all use *your* HF account —
  no credentials live in this repo.
