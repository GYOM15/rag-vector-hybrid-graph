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
python -m eval.beir_eval --dataset scifact --output eval/beir_results.json                                 # single-hop
python -m eval.beir_eval --dataset hotpotqa-distractor --max-queries 500 --output eval/beir_hotpot.json     # multi-hop
python -m eval.beir_eval --dataset nfcorpus --output eval/beir_nfcorpus.json                               # medical IR (hard)
python -m eval.beir_eval --dataset scifact --embedder BAAI/bge-small-en-v1.5 --output eval/beir_scifact_bge.json  # embedder swap
python -m eval.retrieval_eval                                                                              # toy corpus + per-type
python -m eval.sweep_entity_norm                                                                           # held-out tuning of the graph normalization
python -m eval.plot_benchmark                                                                              # → docs/benchmark-results.svg
```

Answer quality end-to-end (EM/F1 on HotpotQA gold) — needs Ollama, deterministic (temperature 0):

```bash
python -m eval.answer_eval --max-queries 50 --model llama3.2:1b
python -m eval.answer_eval --max-queries 50 --model llama3.2:3b
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
- **NFCorpus** — a **medical/nutrition** IR benchmark (~3.6k docs) with many graded-relevant docs per query; a known-*hard* dataset where absolute scores are low for every retriever.
- **qrels** — the human *relevance judgments*: for each query, which documents count as relevant. Metrics score the retrieved ranking against them.

### Results — nDCG@10 (human qrels)

![Benchmark results](docs/benchmark-results.svg)

| nDCG@10 (MiniLM) | SciFact (single-hop) | HotpotQA (multi-hop) | NFCorpus (medical IR) |
|---|---|---|---|
| **Hybrid** (BM25 + dense + RRF) | **0.711** | **0.778** | **0.343** |
| Vector (FAISS, MiniLM) | 0.648 | 0.749 | 0.318 |
| Graph (spaCy + local-search) | 0.643 | 0.748 | 0.323 |

**Takeaway:** the **hybrid** retriever is the robust winner on *all three* corpora —
consistent with the BEIR literature (MiniLM ≈ 0.64, BM25 ≈ 0.665; RRF fusion lifts
to 0.711). NFCorpus is a deliberately hard benchmark (many graded-relevant docs per
query → low absolute nDCG for everyone), yet the ranking holds. The entity-graph
trails — but *why* it trailed turned out to be a fixable bug, not a fundamental
limit (next section).

Reproduce with [Quickstart §4](#4-reproduce-the-evaluation).

### Diagnosing and fixing the graph's failure at scale

The entity-graph first scored **0.484 on HotpotQA** — far below the others. Instead of
leaving it a strawman, I traced *why*: the score added an **unnormalized** sum of
entity-overlap IDF, so **hub documents** accumulated a huge boost and displaced focused
on-topic chunks. Concretely, on *"capital of Afghanistan?"* (500-article corpus) the
top result was the *"June"* calendar page — **53 entities, zero semantic similarity** —
because it cites dozens of countries that all sit in Afghanistan's entity neighborhood.
The noise grows with the corpus, exactly where GraphRAG should help.

The fix is one principled idea — **normalize the entity boost by the chunk's entity
richness** (BM25-style length normalization, [retriever.py](src/stack3_graphrag/retriever.py)):
a focused chunk beats a promiscuous hub, and vector similarity breaks ties. After the fix,
the *"Afghanistan"* query retrieves **5/5 on-topic** documents (was 2/5).

**Choosing the exact formula — without cheating.** Dividing by `n^p` (entities per chunk)
leaves one knob, `p`. I swept it (`none, log, 0.25, 0.5, 0.75, 1.0`) on a **held-out
validation split** (SciFact-train + NFCorpus-validation) and report on the **untouched
test split** — the test set never selects the formula
([sweep_entity_norm.py](eval/sweep_entity_norm.py)). My intuition (a *softer* penalty) was
**wrong**: a *stronger* one, `p=0.75`, won validation (mean nDCG 0.486, vs 0.481 for √,
0.479 for log, 0.453 for the unnormalized bug). That's the point of measuring instead of guessing.

| nDCG@10 (Graph), test split | before (bug) | **after (p=0.75)** | Δ |
|---|---|---|---|
| SciFact | 0.591 | **0.643** | +0.052 |
| HotpotQA (multi-hop) | 0.484 | **0.748** | **+0.264** |
| NFCorpus | 0.310 | **0.323** | +0.013 |

Vector and Hybrid are **byte-for-byte unchanged** (only the graph retriever was touched — a
clean regression check). The graph is now competitive, with the biggest gain on multi-hop
where hub noise hurt most. **Honest caveat:** at 0.748 the graph nearly *matches* the plain
vector retriever (0.749) — the strong normalization mostly stops it harming itself rather
than making it cleverer; its own edge stays on exact named-entity queries. A true GraphRAG
advantage would need LLM-extracted typed relations + community summaries (out of scope). For
fairness: the graph's other constants (`_VEC_SEEDS`, `_GRAPH_WEIGHT`, `_RELATED_DISCOUNT`)
were **not** tuned — only `p`, and only on held-out data.

### Embedder sensitivity

The retriever isn't tied to one embedder. Swapping MiniLM → **bge-small-en-v1.5** on SciFact:

| nDCG@10 (SciFact) | MiniLM | bge-small | Δ |
|---|---|---|---|
| Vector | 0.648 | 0.706 | +0.058 |
| **Hybrid** | **0.711** | **0.726** | +0.015 |
| Graph | 0.643 | 0.646 | +0.003 |

A stronger dense model helps — but unevenly. Pure **Vector** gains most (+0.058); **Hybrid**
moves little (+0.015, BM25 already carried the lexical signal the better embedder adds); the
**Graph** barely budges (+0.003) — its heavy entity-normalization makes the ranking less
sensitive to the embedder. Hybrid still wins: the embedder is a knob, not the verdict.

### By query type — where each architecture shines

On a tagged set (factoid = paraphrased semantic, keyword = exact token), each
architecture shows a distinct character (toy corpus, MRR):

![Per-query-type MRR](docs/per-category.svg)

| MRR | factoid (semantic) | keyword (exact token) |
|---|---|---|
| **Vector** | **0.885** | 0.602 |
| **Hybrid** | 0.875 | **0.845** |
| **Graph** (after fix) | **0.885** | 0.739 |

- **Vector** — *semantic specialist*: best on factoid, but collapses on keyword (no lexical matching).
- **Hybrid** — *robust generalist*: wins keyword, near-best on factoid (why it tops the aggregate).
- **Graph** — after the hub fix: now ties Vector on factoid (0.885) and still beats it on exact tokens via spaCy NER (0.739 vs 0.602). The entity-richness normalization trimmed its keyword edge (was 0.830) — an honest cost of the same fix that lifts the rigorous benchmarks.

> Small/easy corpus → indicative of *character*, not a ranking; the rigorous
> ranking is the BEIR table above.

### Retrieval → answer (end-to-end)

Does *better retrieval* yield *better answers*? Running the full pipeline (retrieve + generate)
and scoring the output against HotpotQA's gold answers with **Exact-Match / F1** (SQuAD-style,
deterministic at temperature 0, **no judge**) — 50 questions, `llama3.2` 1b vs 3b:

| | nDCG@10 | F1 (1b) | F1 (3b) |
|---|---|---|---|
| Vector | 0.803 | 0.079 | 0.158 |
| **Hybrid** | **0.812** | 0.059 | 0.165 |
| Graph | 0.779 | 0.067 | **0.209** |

Honest reading:
- **Model capability dominates** — the 3b roughly **2.5× the F1** of the 1b. That's the clear signal.
- **The per-architecture answer gaps are within noise** at n=50 (Graph's higher 3b-F1 despite the lowest nDCG is ~2 questions out of 50 — not a real win).
- So *better retrieval → better answer* **doesn't surface cleanly here**: with small local models on hard multi-hop QA, the **generator is the bottleneck** — good retrieval is necessary but not sufficient without a capable enough reader.
- Absolute EM/F1 are low by construction (small models; verbose answers vs 1–3-word gold, harsh on Exact-Match; multi-hop needs combining two documents).

**Scope note — deliberately local at this stage.** This ran ~300 generations (~20 min) on a laptop with **no batched serving**: fine for a one-off baseline, not for scale. The next stage — **vLLM + Ray** ([Roadmap](#roadmap)) — batches on GPU, making this eval fast *and* unlocking a larger, more capable reader (the setting where the retrieval→answer link should sharpen). Read the table as a current-stage baseline, not the last word.

### Generation quality (optional)

`eval/questions.json` (40 questions tagged factoid / keyword / multi) drives a RAGAS
benchmark of answer quality (faithfulness, relevancy, context precision/recall) —
in the app's Benchmark tab or via `python -m eval.benchmark`. RAGAS uses an OpenAI
judge → needs `OPENAI_API_KEY`; without it only latencies are reported.

## Roadmap

Planned, not yet implemented:

- **Performance & systems metrics** — isolated retrieval latency (warmup + repeats, median/IQR), a concurrency sweep (throughput, p50/p95/p99), separated index-build cost, and a quality × latency × cost Pareto. The differentiating axis — and the reason the slow local generation above matters.
- **Serving at scale** — **vLLM** (PagedAttention, continuous batching) behind an OpenAI-compatible API, orchestrated by **Ray** (autoscaled replicas, Ray Data batch inference). Reachable through the existing `openai` provider with no code change — it doubles as the remote LLM for a hosted demo *and* the larger, faster reader for the answer eval above.
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
(`20231101.simple`), first 500 articles by default (`--articles` to change).

## License

[MIT](LICENSE) — see the `LICENSE` file.
