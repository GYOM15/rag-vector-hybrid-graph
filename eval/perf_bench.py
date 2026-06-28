"""Performance/systems measurements of the 3 retrievers — **without a language model**.

Everything is deterministic and fast (we only query retrieval):
  1. index build cost, per component (embedding, FAISS, BM25, graph);
  2. isolated retrieval latency: warmup + repeats → median / p95 / p99;
  3. throughput (queries/s) under concurrency (sweep over the number of threads);
  4. nDCG@10 (quality) to trace the quality × latency Pareto front.

    python -m eval.perf_bench --dataset scifact --n-queries 200
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from shared.embeddings import EmbeddingModel  # noqa: E402
from shared.ir_metrics import ndcg_at_k  # noqa: E402
from shared.vector_index import FaissIndexer  # noqa: E402
from stack1_traditional import VectorRetriever  # noqa: E402
from stack2_hybrid import HybridRetriever  # noqa: E402
from stack3_graphrag import GraphRetriever, build_graph  # noqa: E402

from eval.beir_eval import _ranked_doc_ids, load_beir  # noqa: E402

K = 10
CONCURRENCY = (1, 2, 4, 8)


def _timed(fn):
    """Runs fn() and returns (result, elapsed seconds)."""
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def build_with_timing(texts, metadata, embedder):
    """Builds the 3 retrievers, timing each component."""
    embeddings = EmbeddingModel(embedder)
    vectors, t_embed = _timed(lambda: embeddings.encode(texts))

    indexer = FaissIndexer(dimension=embeddings.dimension)
    _, t_faiss = _timed(lambda: indexer.add(vectors, texts, metadata))

    vector, t_vec = _timed(lambda: VectorRetriever(indexer, embeddings))
    hybrid, t_hyb = _timed(lambda: HybridRetriever(indexer, embeddings))  # builds BM25
    graph_obj, t_graph_build = _timed(lambda: build_graph(texts, metadata))  # spaCy NER
    graph, t_graph_init = _timed(lambda: GraphRetriever(indexer, embeddings, graph_obj))

    shared = t_embed + t_faiss  # shared by all three
    build = {
        "shared_embed_faiss": round(shared, 2),
        "embed": round(t_embed, 2), "faiss": round(t_faiss, 2),
        "vector_total": round(shared + t_vec, 2),
        "hybrid_total": round(shared + t_hyb, 2),
        "hybrid_bm25": round(t_hyb, 2),
        "graph_total": round(shared + t_graph_build + t_graph_init, 2),
        "graph_ner_build": round(t_graph_build, 2),
    }
    return {"vector": vector, "hybrid": hybrid, "graph": graph}, build


def measure_latency(retriever, queries, warmup=10, repeats=3):
    """Per-query latency (ms): warmup then repeats → median/p95/p99/mean."""
    for q in queries[:warmup]:
        retriever.search(q, k=K)
    lats = []
    for q in queries:
        for _ in range(repeats):
            _, dt = _timed(lambda q=q: retriever.search(q, k=K))
            lats.append(dt * 1000)
    a = np.array(lats)
    return {"median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2),
            "mean_ms": round(float(a.mean()), 2), "n_calls": len(a)}


def measure_throughput(retriever, queries):
    """Throughput (queries/s) at different concurrency levels (threads)."""
    out = {}
    for workers in CONCURRENCY:
        def run():
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda q: retriever.search(q, k=K), queries))
        _, wall = _timed(run)
        out[str(workers)] = round(len(queries) / wall, 1)
    return out


def measure_ndcg(retriever, queries_eval, qrels):
    """Average nDCG@10 (quality axis of the Pareto front)."""
    total = sum(ndcg_at_k(_ranked_doc_ids(retriever.search(q, k=K)), qrels[qid], 10)
                for qid, q in queries_eval)
    return round(total / max(1, len(queries_eval)), 4)


def run(dataset, n_queries, embedder, output):
    texts, metadata, queries_eval, qrels = load_beir(dataset)
    q_texts = [q for _, q in queries_eval][:n_queries]

    print(f"{dataset}: {len(texts)} docs — build (timed)…", flush=True)
    retrievers, build = build_with_timing(texts, metadata, embedder)

    report = {}
    for name, retr in retrievers.items():
        print(f"  {name}: latency + throughput + nDCG…", flush=True)
        report[name] = {
            "ndcg@10": measure_ndcg(retr, queries_eval, qrels),
            "latency": measure_latency(retr, q_texts),
            "throughput_qps": measure_throughput(retr, q_texts),
        }

    payload = {"config": {"dataset": dataset, "n_docs": len(texts), "embedder": embedder,
                          "n_queries": len(q_texts), "k": K, "repeats": 3},
               "build_seconds": build, "stacks": report}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{dataset} · {len(texts)} docs · {len(q_texts)} queries (k={K})\n")
    print(f"  build (s): shared embed+faiss={build['shared_embed_faiss']}  "
          f"| +BM25={build['hybrid_bm25']}  | +graph(NER)={build['graph_ner_build']}")
    print(f"\n  {'arch':8} {'nDCG@10':>8} {'median':>9} {'p95':>8} {'p99':>8} "
          + "".join(f'qps@{w:<3}' for w in CONCURRENCY))
    for name, m in report.items():
        lat, tp = m["latency"], m["throughput_qps"]
        print(f"  {name:8} {m['ndcg@10']:8.3f} {lat['median_ms']:8.2f}ms {lat['p95_ms']:7.1f} "
              f"{lat['p99_ms']:7.1f} " + "".join(f"{tp[str(w)]:7.0f} " for w in CONCURRENCY))
    print(f"\n✅ Details written to {output}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Perf/systems of the 3 retrievers (no LLM).")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--n-queries", type=int, default=200, help="queries for latency/throughput")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "perf_results.json")
    args = ap.parse_args()
    run(args.dataset, args.n_queries, args.embedder, args.output)


if __name__ == "__main__":
    main()
