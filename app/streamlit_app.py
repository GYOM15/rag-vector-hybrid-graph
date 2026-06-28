"""Dashboard comparing the RAG architectures.

Two tabs:
  - "Live chat": 3 chats side by side (one per architecture). The same question
    goes to all three; each keeps its own thread.
  - "Evaluation": runs the evaluation (RAGAS + latencies) and visualizes it, or
    shows the last `eval/results.json`.

Note: the RAG has no conversational memory — each question is handled
independently (comparative chat log, not contextual multi-turn).

Run:  streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from pipeline import STACK_NAMES, build_stacks  # noqa: E402

from eval_dashboard import (  # noqa: E402
    _grouped_bar, render_answer, render_beir, render_regression_guard,
    render_reranking, render_systems, render_type_retrieval,
)

RESULTS_PATH = PROJECT_ROOT / "eval" / "results.json"
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
# Material icons (native to Streamlit) colored per architecture — consistent
# with the benchmark chart colors (blue / green / violet).
ICONS = {
    STACK_NAMES["vector"]: ":blue[:material/scatter_plot:]",
    STACK_NAMES["hybrid"]: ":green[:material/merge:]",
    STACK_NAMES["graph"]: ":violet[:material/hub:]",
}
# Monochrome (Material) bubble avatars: question = person, answer = the
# architecture icon (echoes the column header, without the color).
_ARCH_AVATAR = {
    STACK_NAMES["vector"]: ":material/scatter_plot:",
    STACK_NAMES["hybrid"]: ":material/merge:",
    STACK_NAMES["graph"]: ":material/hub:",
}
_USER_AVATAR = ":material/person:"


@st.cache_resource(show_spinner="Loading corpus, chunking and indexing…")
def get_stacks() -> dict:
    """Builds (once) the three RAG pipelines.

    `DEMO_ARTICLES` (env) caps the corpus size — useful to speed up a hosted demo
    (e.g. set 200 on Hugging Face Spaces); defaults to 500.
    """
    return build_stacks(n_articles=int(os.getenv("DEMO_ARTICLES", "500")))


def select_backend() -> None:
    """LLM backend selector (sidebar) — sets the env variables.

    `call_llm` reads them on every call, so switching backend does not rebuild
    the index (cached and independent of the LLM).
    """
    st.sidebar.header(":material/settings: LLM backend")
    providers = ["ollama", "openai", "huggingface"]
    default = os.getenv("LLM_PROVIDER", "ollama")  # a hosted demo sets huggingface (flan-t5)
    provider = st.sidebar.selectbox(
        "Provider", providers,
        index=providers.index(default) if default in providers else 0,
        help="ollama = local · openai = OpenAI/vLLM-compatible endpoint · "
             "huggingface = local flan-t5",
    )
    os.environ["LLM_PROVIDER"] = provider
    if provider == "ollama":
        os.environ["OLLAMA_MODEL"] = st.sidebar.text_input(
            "Ollama model", os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    elif provider == "openai":
        os.environ["OPENAI_BASE_URL"] = st.sidebar.text_input(
            "Base URL (vLLM / OpenAI)", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
        os.environ["OPENAI_MODEL"] = st.sidebar.text_input(
            "Model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
            "API key", os.getenv("OPENAI_API_KEY", ""), type="password") or os.getenv("OPENAI_API_KEY", "")
    else:
        os.environ["HF_MODEL"] = st.sidebar.text_input(
            "HF model", os.getenv("HF_MODEL", "google/flan-t5-base"))
    st.sidebar.caption(f"Active backend: **{provider}**")


def render_chat_tab() -> None:
    """3 chats side by side; one shared question goes to the 3 architectures."""
    st.caption("The same question is sent to the 3 architectures; each keeps its own thread. "
               "(No conversational memory: each question is independent.)")

    top = st.columns([3, 1], vertical_alignment="bottom")
    k = top[0].slider("Number of retrieved chunks (k)", 1, 10, 5)
    if top[1].button("Clear threads", icon=":material/delete:", width="stretch"):
        st.session_state.pop("chat", None)

    if "chat" not in st.session_state:
        st.session_state.chat = {name: [] for name in STACK_NAMES.values()}

    prompt = st.chat_input("Ask the 3 architectures a question…")
    if prompt:
        stacks = get_stacks()
        with st.spinner("Generating the 3 answers…"):
            for name in st.session_state.chat:
                st.session_state.chat[name].append({"role": "user", "content": prompt, "result": None})
                try:
                    r = stacks[name].query(prompt, k=k)
                    st.session_state.chat[name].append(
                        {"role": "assistant", "content": r["answer"], "result": r})
                except Exception as exc:
                    st.session_state.chat[name].append(
                        {"role": "assistant", "content": f"Error: {exc}", "result": None})

    for column, name in zip(st.columns(len(STACK_NAMES)), st.session_state.chat):
        with column, st.container(border=True):
            st.markdown(f"##### {ICONS.get(name, '')} {name}")
            for msg in st.session_state.chat[name]:
                avatar = _USER_AVATAR if msg["role"] == "user" else _ARCH_AVATAR.get(name)
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])
                    result = msg["result"]
                    if result is not None:
                        st.caption(f":material/schedule: {result['latency_ms']} ms · "
                                   f":material/description: {len(result['contexts'])} chunks")
                        with st.expander("Retrieved chunks"):
                            for i, ctx in enumerate(result["contexts"], 1):
                                meta = ctx["metadata"]
                                st.markdown(f"**[{i}]** *{meta.get('title', '')}* — score {ctx['score']:.4f}")
                                if meta.get("shared_entities"):
                                    st.caption("shared entities: " + ", ".join(meta["shared_entities"]))
                                text = ctx["text"]
                                st.write(text[:400] + ("…" if len(text) > 400 else ""))


def run_benchmark(n_questions: int, k: int) -> None:
    """Evaluates the 3 stacks (generation + RAGAS + latencies) and writes results.json."""
    try:
        from shared.evaluator import evaluate_stacks  # lazy import (ragas)
    except ImportError:
        st.error("RAGAS isn't installed in this deployment — run the benchmark locally (see README §4).")
        return

    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[:n_questions]
    questions = [d["question"] for d in data]
    ground_truths = [d.get("ground_truth") or d.get("answer") or "" for d in data]
    types = [d.get("type", "?") for d in data]
    try:
        with st.spinner(f"Benchmark on {len(questions)} questions (generation + RAGAS)… "
                        "this can take a few minutes."):
            metrics = evaluate_stacks(get_stacks(), questions, ground_truths, k=k, types=types)
        payload = {
            "config": {
                "n_articles": 500, "k": k, "n_questions": len(questions),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "stacks": metrics,
        }
        RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success("Benchmark done.")
    except Exception as exc:
        st.error(f"Benchmark failed: {exc}")


# Readable labels + stack colors (blue / green / violet, vector/hybrid/graph order).
_RAGAS_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}
_LATENCY_LABELS = {
    "avg_retrieval_ms": "Retrieval",
    "avg_generation_ms": "Generation",
    "avg_latency_ms": "Total",
}
def render_benchmark_results() -> None:
    """Shows the last results.json: comparison table + charts."""
    if not RESULTS_PATH.exists():
        st.info("No results yet. Run a benchmark above.")
        return

    import pandas as pd

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    st.caption(
        f"{cfg.get('n_questions', '?')} questions · {cfg.get('n_articles', '?')} articles "
        f"· generated on {cfg.get('generated_at', '?')}"
    )

    df = pd.DataFrame(data["stacks"])  # index = metrics, columns = stacks
    st.dataframe(df.T, width="stretch")  # stacks as rows (readable)

    ragas = [m for m in _RAGAS_LABELS if m in df.index]
    if ragas:
        st.subheader("Quality — RAGAS (higher = better, 0–1 score)")
        _grouped_bar(df.loc[ragas].rename(index=_RAGAS_LABELS),
                     "Metric", "Score", "Quality by metric", ".2f")
    else:
        st.warning("No RAGAS metrics (missing OpenAI key?). Only latencies are shown.")

    latency = [m for m in _LATENCY_LABELS if m in df.index]
    if latency:
        st.subheader("Mean latencies (lower = better)")
        _grouped_bar(df.loc[latency].rename(index=_LATENCY_LABELS),
                     "Step", "Milliseconds", "Latency by step", ".0f")

    _render_by_type(data["stacks"], pd)


def _render_by_type(stacks: dict, pd) -> None:
    """Breakdown by question category (factoid / keyword / multi)."""
    by_types = {name: m["by_type"] for name, m in stacks.items() if m.get("by_type")}
    if not by_types:
        return

    st.subheader("By question category")
    categories = sorted({t for bt in by_types.values() for t in bt})

    shown_quality = False
    for metric, label in _RAGAS_LABELS.items():
        table = {stack: {t: bt.get(t, {}).get(metric) for t in categories}
                 for stack, bt in by_types.items()}
        chart = pd.DataFrame(table)  # index = category, columns = stack
        if chart.notna().any().any():
            shown_quality = True
            _grouped_bar(chart, "Category", label,
                         f"{label} by category (higher = better)", ".2f")

    if not shown_quality:
        st.info("Per-category quality unavailable (RAGAS not computed, no OpenAI key). "
                "With a key, you'll see here which architecture wins per question type.")

    lat = {stack: {t: bt.get(t, {}).get("avg_latency_ms") for t in categories}
           for stack, bt in by_types.items()}
    chart_lat = pd.DataFrame(lat)
    if chart_lat.notna().any().any():
        _grouped_bar(chart_lat, "Category", "Latency (ms)",
                     "Mean latency by category (lower = better)", ".0f")


st.set_page_config(page_title="RAG comparison", layout="wide")
st.title("RAG architecture comparison")
st.caption("Vector · Hybrid · Graph — same corpus, same chunking, same prompt.")

select_backend()

tab_chat, tab_eval = st.tabs([":material/forum: Live chat", ":material/analytics: Evaluation"])

with tab_chat:
    render_chat_tab()

with tab_eval:
    st.caption("Evaluation results (reference snapshots in `eval/reference/`) + the live "
               "RAGAS benchmark. Heavy evals are re-run from the command line (README §4).")
    sub = st.tabs([":material/search: Retrieval (BEIR)", ":material/swap_vert: Reranking",
                   ":material/speed: Systems", ":material/question_answer: Answer quality",
                   ":material/shield: Guard (live)", ":material/category: Retrieval by type (live)",
                   ":material/fact_check: RAGAS (live)"])
    with sub[0]:
        render_beir()
    with sub[1]:
        render_reranking()
    with sub[2]:
        render_systems()
    with sub[3]:
        render_answer()
    with sub[4]:
        render_regression_guard()
    with sub[5]:
        render_type_retrieval(get_stacks)
    with sub[6]:
        st.subheader("RAGAS benchmark (generation + judging)")
        with st.form("bench_form"):
            c1, c2 = st.columns(2)
            n_q = c1.number_input("Number of questions", 1, 50, 5)
            k_b = c2.number_input("k (retrieved chunks)", 1, 10, 5)
            key = st.text_input(
                "OpenAI key for RAGAS (else taken from .env)",
                os.getenv("OPENAI_API_KEY", ""), type="password",
            )
            go = st.form_submit_button("Run the benchmark", type="primary", icon=":material/play_arrow:")
        if go:
            if key.strip():
                os.environ["OPENAI_API_KEY"] = key.strip()
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("No OpenAI key → only latencies will be computed (RAGAS skipped).")
            run_benchmark(int(n_q), int(k_b))
        st.divider()
        render_benchmark_results()
