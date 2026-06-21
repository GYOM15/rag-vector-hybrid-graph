<div align="center">

<pre>
██████╗  █████╗  ██████╗ 
██╔══██╗██╔══██╗██╔════╝ 
██████╔╝███████║██║  ███╗
██╔══██╗██╔══██║██║   ██║
██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 
</pre>

**Three RAG architectures, compared fairly.**

`Vector` · `Hybrid (BM25 + RRF)` · `Graph` — same corpus · chunking · prompt · LLM

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector-FAISS-009999)
![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM-FFD21E?logo=huggingface&logoColor=black)
![BM25](https://img.shields.io/badge/Lexical-BM25%20%2B%20RRF-1D9E75)
![networkx](https://img.shields.io/badge/Graph-networkx-2C5BB4)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS-E8543F)
![Ollama](https://img.shields.io/badge/LLM-Ollama-000000?logo=ollama&logoColor=white)
![vLLM](https://img.shields.io/badge/Serving-vLLM-FDB515)
![Ray](https://img.shields.io/badge/Scaling-Ray-028CF0)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/gyom15/rag-vector-hybrid-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/gyom15/rag-vector-hybrid-graph/actions)

[Architecture](#architecture) · [Quickstart](#quickstart) · [Evaluation](#evaluation-set) · [Scaling](#scaling)

</div>

---

Comparative study of three **Retrieval-Augmented Generation** architectures on the
**same corpus, chunking, prompt and LLM** — only the *retrieval strategy* changes,
so the comparison is fair (controlled variables).

| Stack | Retrieval | What it adds |
|------|-----------|--------------|
| **Vector** | dense similarity (FAISS) | semantic meaning |
| **Hybrid** | vector + BM25, fused by **RRF** | exact keywords (dates, names, codes) |
| **Graph** | entity graph (networkx) + vector seed + traversal | broader recall / relational |

## Architecture

![Architecture](docs/architecture.svg)

Only the **retriever** differs between stacks; chunking, embeddings, FAISS index,
prompt and LLM are shared. `pipeline.build_stacks()` is the single source of truth
used by both the app and the benchmark.

## A RAG answer has two stages

A wrong answer can come from **retrieval** (the right chunk never reaches the
context) *or* from **generation** (the chunk is there but the model misreads it).
Both must succeed — the matrix below shows why raising `k` alone or upgrading the
model alone is not enough:

![Retrieval × Generation](docs/retrieval-vs-generation.svg)

## Project structure

```
rag-vector-hybrid-graph/
├── src/
│   ├── shared/
│   ├── stack1_traditional/
│   ├── stack2_hybrid/
│   ├── stack3_graphrag/
│   └── pipeline.py
├── eval/
├── app/
├── tests/
└── docs/
```

`src/` is the library (shared core + the 3 stacks + `pipeline`), `eval/` the
benchmark, `app/` the Streamlit dashboard.

## Quickstart

### 1. Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core: library + Streamlit app
pip install -e ".[eval]"         # + RAGAS benchmark
pip install -e ".[dev]"          # + pytest / ruff
```

### 2. Pick an LLM backend

Generation needs an LLM, selected by `LLM_PROVIDER` (copy `.env.example` → `.env`):

| Provider | Env vars | Model type | When |
|---|---|---|---|
| `ollama` *(default)* | `OLLAMA_URL`, `OLLAMA_MODEL` | decoder LLM (llama3.2…) | local dev |
| `openai` | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | decoder LLM | OpenAI **or a vLLM server** |
| `huggingface` | `HF_MODEL` | **seq2seq only** (flan-t5…) | self-contained, no server |

```bash
ollama pull llama3.2:3b          # set OLLAMA_MODEL=llama3.2:3b
```

> `OPENAI_API_KEY` is also the **RAGAS** judge for the benchmark, whatever the
> generation backend.

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```
- **💬 Chat** — one question to the 3 architectures side by side, each with its own
  thread, latency and sources.
- **📊 Benchmark** — run the evaluation in-app (questions / `k` / OpenAI key) or view
  the last `eval/results.json`. Grouped charts compare quality and latency, overall
  and per question category.

### 4. Run the benchmark (CLI)

```bash
python -m eval.benchmark                 # all questions
python -m eval.benchmark --questions 15  # quick run
```
Writes `eval/results.json`. The app and the CLI share the same
`shared.evaluator.evaluate_stacks`.

## Evaluation set

`eval/questions.json` holds 40 hand-written questions, each tagged by **type** so the
benchmark shows *which architecture wins on which kind of question*:

| Type | Probes | Favours |
|---|---|---|
| `factoid` | paraphrased semantic fact | Vector |
| `keyword` | exact token (date, name, number) | Hybrid (BM25) |
| `multi` | aggregate several facts / broad recall | Graph / higher `k` |

> Small and corpus-bound → illustrative, not a production benchmark. RAGAS quality
> needs `OPENAI_API_KEY`; without it the benchmark still reports latencies.

## Scaling

> Work in progress — this section grows as the Ray + vLLM serving path lands.

- **vLLM** — high-throughput inference engine (PagedAttention, continuous batching)
  behind an **OpenAI-compatible API**, reachable through the `openai` provider with
  no code change.
- **Ray** — distributed orchestration: autoscaled replicas (Ray Serve LLM),
  multi-node parallelism, batch inference (Ray Data).

Path: prototype on **Ollama** (dev) → serve the same model class on **vLLM + Ray**
(prod), both via the `openai` provider.

## Tests

```bash
pytest -q
```
Cover the pure logic (recursive chunking, RRF fusion, entity extraction) and run
without the heavy ML stack, keeping CI fast.

## Data

[Simple English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
(`20231101.simple`), first 100 articles by default (`--articles` to change).

## License

[MIT](LICENSE) — see the `LICENSE` file.
