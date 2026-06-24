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
![spaCy](https://img.shields.io/badge/NER-spaCy-09A3D5?logo=spacy&logoColor=white)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS-E8543F)
![Ollama](https://img.shields.io/badge/LLM-Ollama-000000?logo=ollama&logoColor=white)
![vLLM](https://img.shields.io/badge/Serving-vLLM-FDB515)
![Ray](https://img.shields.io/badge/Scaling-Ray-028CF0)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/gyom15/rag-vector-hybrid-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/gyom15/rag-vector-hybrid-graph/actions)

[Architecture](#architecture) · [Quickstart](#quickstart) · [Evaluation](#evaluation) · [Roadmap](#roadmap)

</div>

---

Comparative study of three **Retrieval-Augmented Generation** architectures on the
**same corpus, chunking, prompt and LLM** — only the *retrieval strategy* changes,
so the comparison is fair (controlled variables).

| Stack | Retrieval | What it adds |
|------|-----------|--------------|
| **Vector** | dense similarity (FAISS) | semantic meaning |
| **Hybrid** | vector + BM25, fused by **RRF** | exact keywords (dates, names, codes) |
| **Graph** | spaCy NER → entity graph (networkx) + **local-search** (query entities via MENTIONS/RELATED_TO, IDF-weighted) | relational / multi-hop |

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
python -m spacy download en_core_web_sm   # NER model used by the graph stack
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

### 4. Reproduce the evaluation

Retrieval quality on standard IR benchmarks — **no LLM**, human relevance judgments:

```bash
python -m eval.beir_eval --dataset scifact                               # single-hop
python -m eval.beir_eval --dataset hotpotqa-distractor --max-queries 500  # multi-hop
python -m eval.retrieval_eval                                            # toy corpus + embedder comparison
```

Generation quality (RAGAS) — needs `OPENAI_API_KEY` as the judge:

```bash
python -m eval.benchmark --questions 15
```

## Evaluation

**Retrieval is evaluated without an LLM** — we measure whether each architecture
*retrieves the relevant documents*, using human relevance judgments (qrels) from
standard IR benchmarks. This isolates the retriever (immune to the LLM's memory),
is deterministic, and needs no API key. Metrics are pure, unit-tested functions
(`shared/ir_metrics.py`):

- **recall@k** — fraction of relevant docs found in the top-k.
- **nDCG@10** — top-10 ranking quality (rewards relevant docs ranked higher); the standard BEIR metric.
- **MRR** — 1 / rank of the first relevant doc.

**Datasets** (loaded from HuggingFace, each with its own human qrels):

- **BEIR** — a standard suite of information-retrieval benchmarks (each = a corpus + queries + relevance judgments).
- **SciFact** — scientific *claim verification*: ~5k abstracts, 300 queries; **single-hop** (the answer lives in one document).
- **HotpotQA** (distractor) — **multi-hop** QA: each question needs **≥2 documents combined**; we rank the supporting paragraphs among distractors.
- **qrels** — the human *relevance judgments*: for each query, which documents count as relevant. Metrics score the retrieved ranking against them.

### Results — nDCG@10 (human qrels)

![Benchmark results](docs/benchmark-results.svg)

| nDCG@10 | SciFact (single-hop) | HotpotQA (multi-hop) |
|---|---|---|
| **Hybrid** (BM25 + dense + RRF) | **0.711** | **0.778** |
| Vector (FAISS, MiniLM) | 0.648 | 0.749 |
| Graph (spaCy + local-search) | 0.591 | 0.484 |

**Takeaway:** the **hybrid** retriever is the robust winner on *both* corpora —
consistent with the BEIR literature (MiniLM ≈ 0.64, BM25 ≈ 0.665; RRF fusion lifts
to 0.711). The lightweight entity-graph underperforms on standard IR (its additive
entity boost adds noise on large corpora); the real GraphRAG advantage needs
LLM-extracted relations + community summaries, out of scope here. No free lunch —
and showing it *honestly* on standard benchmarks is the point.

Reproduce with [Quickstart §4](#4-reproduce-the-evaluation).

### By query type — where each architecture shines

On a tagged set (factoid = paraphrased semantic, keyword = exact token), each
architecture shows a distinct character (toy corpus, MRR):

![Per-query-type MRR](docs/per-category.svg)

| MRR | factoid (semantic) | keyword (exact token) |
|---|---|---|
| **Vector** | **0.885** | 0.602 |
| **Hybrid** | 0.875 | **0.845** |
| **Graph** | 0.854 | 0.830 |

- **Vector** — *semantic specialist*: best on factoid, but collapses on keyword (no lexical matching).
- **Hybrid** — *robust generalist*: wins keyword, near-best on factoid (why it tops the aggregate).
- **Graph** — *entity-robust*: spaCy NER recovers named-entity keyword queries (0.830) far better than pure Vector (0.602), without BM25.

> Small/easy corpus → indicative of *character*, not a ranking; the rigorous
> ranking is the BEIR table above.

### Seen live in the app

End-to-end, the same effect appears. Asked live (Simple Wikipedia, small local LLM),
all three retrieve the answer chunk — but on the keyword question only Hybrid answers:
its context is clean (all April), while Vector and Graph also pull December/August
events, and the noise tips the small model into *"not enough info"*. Retrieval
**precision** drives the final answer (and motivates measuring generation — see [Roadmap](#roadmap)).

![Live results across three questions](docs/live-questions.svg)

### Generation quality (optional)

`eval/questions.json` (40 questions tagged factoid / keyword / multi) drives a RAGAS
benchmark of answer quality (faithfulness, relevancy, context precision/recall) —
in the app's Benchmark tab or via `python -m eval.benchmark`. RAGAS uses an OpenAI
judge → needs `OPENAI_API_KEY`; without it only latencies are reported.

## Roadmap

Planned, not yet implemented:

- **Generation quality** — measure end-to-end answers (exact-match / F1 on HotpotQA gold answers, via the local LLM) to confirm whether *better retrieval → better answers*.
- **Serving at scale** — **vLLM** (PagedAttention, continuous batching) behind an OpenAI-compatible API, orchestrated by **Ray** (autoscaled replicas, Ray Data batch inference). Reachable through the existing `openai` provider with no code change — and it doubles as the remote LLM for a hosted demo.
- **Hosted demo** — a retrieval-first Streamlit demo on HF Spaces (generation wired to the vLLM endpoint above).
- **Breadth** — stronger embedders (bge/e5) on the BEIR datasets, plus more datasets (NFCorpus, FiQA).

## Tests

```bash
pytest -q
```
Cover the pure logic (chunking, RRF fusion, BM25 tokenizer, IR metrics) with only
light deps (snowballstemmer, spaCy) — no torch/faiss — keeping CI fast.

## Data

[Simple English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
(`20231101.simple`), first 100 articles by default (`--articles` to change).

## License

[MIT](LICENSE) — see the `LICENSE` file.
