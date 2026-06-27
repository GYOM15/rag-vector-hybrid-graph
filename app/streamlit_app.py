"""Tableau de bord de comparaison des architectures RAG.

Deux onglets :
  - « Chat en direct » : 3 chats côte à côte (un par architecture). Une même
    question part vers les trois ; chacune garde son fil.
  - « Benchmark » : lance l'évaluation (RAGAS + latences) et la visualise, ou
    affiche le dernier `eval/results.json`.

Note : le RAG est sans mémoire conversationnelle — chaque question est traitée
indépendamment (journal de chat comparatif, pas de multi-tour contextuel).

Lancer :  streamlit run app/streamlit_app.py
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
    render_reranking, render_systems, render_toy_retrieval,
)

RESULTS_PATH = PROJECT_ROOT / "eval" / "results.json"
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
# Icônes Material (natives Streamlit) colorées par architecture — cohérentes
# avec les couleurs des graphiques du benchmark (bleu / vert / violet).
ICONS = {
    STACK_NAMES["vector"]: ":blue[:material/scatter_plot:]",
    STACK_NAMES["hybrid"]: ":green[:material/merge:]",
    STACK_NAMES["graph"]: ":violet[:material/hub:]",
}


@st.cache_resource(show_spinner="Chargement du corpus, chunking et indexation…")
def get_stacks() -> dict:
    """Construit (une seule fois) les trois pipelines RAG."""
    return build_stacks()


def select_backend() -> None:
    """Sélecteur de backend LLM (barre latérale) — pose les variables d'env.

    `call_llm` les lit à chaque appel, donc changer de backend ne reconstruit pas
    l'index (mis en cache et indépendant du LLM).
    """
    st.sidebar.header("⚙️ Backend LLM")
    provider = st.sidebar.selectbox(
        "Provider", ["ollama", "openai", "huggingface"],
        help="ollama = local · openai = endpoint compatible OpenAI/vLLM · "
             "huggingface = flan-t5 local",
    )
    os.environ["LLM_PROVIDER"] = provider
    if provider == "ollama":
        os.environ["OLLAMA_MODEL"] = st.sidebar.text_input(
            "Modèle Ollama", os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    elif provider == "openai":
        os.environ["OPENAI_BASE_URL"] = st.sidebar.text_input(
            "Base URL (vLLM / OpenAI)", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
        os.environ["OPENAI_MODEL"] = st.sidebar.text_input(
            "Modèle", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
            "Clé API", os.getenv("OPENAI_API_KEY", ""), type="password") or os.getenv("OPENAI_API_KEY", "")
    else:
        os.environ["HF_MODEL"] = st.sidebar.text_input(
            "Modèle HF", os.getenv("HF_MODEL", "google/flan-t5-base"))
    st.sidebar.caption(f"Backend actif : **{provider}**")


def render_chat_tab() -> None:
    """3 chats côte à côte ; une question commune va vers les 3 architectures."""
    st.caption("Une même question est envoyée aux 3 architectures ; chacune garde son fil. "
               "(Pas de mémoire conversationnelle : chaque question est indépendante.)")

    top = st.columns([3, 1])
    k = top[0].slider("Nombre de chunks récupérés (k)", 1, 10, 5)
    if top[1].button("🗑️ Effacer les fils"):
        st.session_state.pop("chat", None)

    if "chat" not in st.session_state:
        st.session_state.chat = {name: [] for name in STACK_NAMES.values()}

    prompt = st.chat_input("Pose une question aux 3 architectures…")
    if prompt:
        stacks = get_stacks()
        with st.spinner("Génération des 3 réponses…"):
            for name in st.session_state.chat:
                st.session_state.chat[name].append({"role": "user", "content": prompt, "result": None})
                try:
                    r = stacks[name].query(prompt, k=k)
                    st.session_state.chat[name].append(
                        {"role": "assistant", "content": r["answer"], "result": r})
                except Exception as exc:
                    st.session_state.chat[name].append(
                        {"role": "assistant", "content": f"⚠️ {exc}", "result": None})

    for column, name in zip(st.columns(len(STACK_NAMES)), st.session_state.chat):
        with column:
            st.markdown(f"#### {ICONS.get(name, '')} {name}")
            for msg in st.session_state.chat[name]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    result = msg["result"]
                    if result is not None:
                        st.caption(f"⏱️ {result['latency_ms']} ms · {len(result['contexts'])} chunks")
                        with st.expander("Chunks récupérés"):
                            for i, ctx in enumerate(result["contexts"], 1):
                                meta = ctx["metadata"]
                                st.markdown(f"**[{i}]** *{meta.get('title', '')}* — score {ctx['score']:.4f}")
                                if meta.get("shared_entities"):
                                    st.caption("entités partagées : " + ", ".join(meta["shared_entities"]))
                                text = ctx["text"]
                                st.write(text[:400] + ("…" if len(text) > 400 else ""))


def run_benchmark(n_questions: int, k: int) -> None:
    """Évalue les 3 stacks (génération + RAGAS + latences) et écrit results.json."""
    from shared.evaluator import evaluate_stacks  # import lazy (ragas)

    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[:n_questions]
    questions = [d["question"] for d in data]
    ground_truths = [d.get("ground_truth") or d.get("answer") or "" for d in data]
    types = [d.get("type", "?") for d in data]
    try:
        with st.spinner(f"Benchmark sur {len(questions)} questions (génération + RAGAS)… "
                        "ça peut prendre quelques minutes."):
            metrics = evaluate_stacks(get_stacks(), questions, ground_truths, k=k, types=types)
        payload = {
            "config": {
                "n_articles": 500, "k": k, "n_questions": len(questions),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "stacks": metrics,
        }
        RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success("Benchmark terminé ✅")
    except Exception as exc:
        st.error(f"Échec du benchmark : {exc}")


# Étiquettes lisibles + couleurs des stacks (bleu / vert / violet, ordre vector/hybrid/graph).
_RAGAS_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}
_LATENCY_LABELS = {
    "avg_retrieval_ms": "Récupération",
    "avg_generation_ms": "Génération",
    "avg_latency_ms": "Total",
}
def render_benchmark_results() -> None:
    """Affiche le dernier results.json : tableau + graphiques de comparaison."""
    if not RESULTS_PATH.exists():
        st.info("Aucun résultat encore. Lance un benchmark ci-dessus.")
        return

    import pandas as pd

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    st.caption(
        f"{cfg.get('n_questions', '?')} questions · {cfg.get('n_articles', '?')} articles "
        f"· généré le {cfg.get('generated_at', '?')}"
    )

    df = pd.DataFrame(data["stacks"])  # index = métriques, colonnes = stacks
    st.dataframe(df.T, use_container_width=True)  # stacks en lignes (lisible)

    ragas = [m for m in _RAGAS_LABELS if m in df.index]
    if ragas:
        st.subheader("Qualité — RAGAS (plus haut = mieux, score 0–1)")
        _grouped_bar(df.loc[ragas].rename(index=_RAGAS_LABELS),
                     "Métrique", "Score", "Qualité par métrique", ".2f")
    else:
        st.warning("Pas de métriques RAGAS (clé OpenAI absente ?). Seules les latences sont affichées.")

    latency = [m for m in _LATENCY_LABELS if m in df.index]
    if latency:
        st.subheader("Latences moyennes (plus bas = mieux)")
        _grouped_bar(df.loc[latency].rename(index=_LATENCY_LABELS),
                     "Étape", "Millisecondes", "Latence par étape", ".0f")

    _render_by_type(data["stacks"], pd)


def _render_by_type(stacks: dict, pd) -> None:
    """Ventilation par catégorie de question (factoid / keyword / multi)."""
    by_types = {name: m["by_type"] for name, m in stacks.items() if m.get("by_type")}
    if not by_types:
        return

    st.subheader("Par catégorie de question")
    categories = sorted({t for bt in by_types.values() for t in bt})

    shown_quality = False
    for metric, label in _RAGAS_LABELS.items():
        table = {stack: {t: bt.get(t, {}).get(metric) for t in categories}
                 for stack, bt in by_types.items()}
        chart = pd.DataFrame(table)  # index = catégorie, colonnes = stack
        if chart.notna().any().any():
            shown_quality = True
            _grouped_bar(chart, "Catégorie", label,
                         f"{label} par catégorie (plus haut = mieux)", ".2f")

    if not shown_quality:
        st.info("Qualité par catégorie indisponible (RAGAS non calculé, pas de clé OpenAI). "
                "Avec une clé, tu verras ici quelle architecture gagne par type de question.")

    lat = {stack: {t: bt.get(t, {}).get("avg_latency_ms") for t in categories}
           for stack, bt in by_types.items()}
    chart_lat = pd.DataFrame(lat)
    if chart_lat.notna().any().any():
        _grouped_bar(chart_lat, "Catégorie", "Latence (ms)",
                     "Latence moyenne par catégorie (plus bas = mieux)", ".0f")


st.set_page_config(page_title="Comparaison RAG", layout="wide")
st.title("🔍 Comparaison des architectures RAG")
st.caption("Vectoriel · Hybride · Graphe — même corpus, même chunking, même prompt.")

select_backend()

tab_chat, tab_eval = st.tabs(["💬 Chat en direct", "🧪 Évaluation"])

with tab_chat:
    render_chat_tab()

with tab_eval:
    st.caption("Résultats des évaluations (instantanés de référence `eval/reference/`) + le "
               "benchmark RAGAS en direct. Les évals lourdes se relancent en ligne de commande (README §4).")
    sub = st.tabs(["📈 Récupération (BEIR)", "🔀 Reranking", "⚙️ Systèmes", "💬 Qualité réponse",
                   "🛡️ Garde-fou (live)", "🎯 Récupération jouet (live)", "🧪 RAGAS (live)"])
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
        render_toy_retrieval(get_stacks)
    with sub[6]:
        st.markdown("#### Lancer un benchmark RAGAS (génération + jugement)")
        with st.form("bench_form"):
            c1, c2 = st.columns(2)
            n_q = c1.number_input("Nombre de questions", 1, 50, 5)
            k_b = c2.number_input("k (chunks récupérés)", 1, 10, 5)
            key = st.text_input(
                "Clé OpenAI pour RAGAS (sinon prise depuis .env)",
                os.getenv("OPENAI_API_KEY", ""), type="password",
            )
            go = st.form_submit_button("▶️ Lancer le benchmark", type="primary")
        if go:
            if key.strip():
                os.environ["OPENAI_API_KEY"] = key.strip()
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("Pas de clé OpenAI → seules les latences seront calculées (RAGAS sauté).")
            run_benchmark(int(n_q), int(k_b))
        st.divider()
        render_benchmark_results()
