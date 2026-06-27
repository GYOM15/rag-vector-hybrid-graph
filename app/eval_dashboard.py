"""Tableau de bord d'évaluation : visualise les résultats des évals (snapshots commités).

Lit les instantanés de référence dans `eval/reference/` (versionnés, contrairement aux
sorties de scratch `eval/*.json`) et les rend en tableaux + graphiques. **Aucun calcul
lourd ici** — comme MLflow / Weights & Biases, on *visualise des runs déjà calculés*. Les
évals lourdes se relancent en ligne de commande (README §4) ; on régénère les snapshots
en copiant leurs sorties dans `eval/reference/`.
"""

import json
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = _ROOT / "eval" / "reference"
PERF_PARETO = _ROOT / "docs" / "perf-pareto.svg"

_STACK_COLORS = ["#3b82f6", "#22c55e", "#a855f7"]  # bleu, vert, violet
_ORDER = ["Vector", "Hybrid", "Graph"]


def _short(name: str) -> str:
    n = name.lower()
    return "Vector" if "vecto" in n else ("Hybrid" if "hybr" in n else "Graph")


def _load(name: str) -> dict | None:
    path = REFERENCE / name
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def _cols(df):
    """Garde et ordonne les colonnes architectures (Vector/Hybrid/Graph)."""
    return df[[c for c in _ORDER if c in df.columns]]


def _rows(df):
    """Ordonne les lignes architectures."""
    return df.reindex([c for c in _ORDER if c in df.index])


def _grouped_bar(matrix, x_title: str, y_title: str, title: str, value_format: str = ".2f") -> None:
    """Barres groupées (Altair). `matrix` : index = axe X, colonnes = séries."""
    import altair as alt

    order = list(matrix.columns)
    frame = matrix.copy()
    frame.index = frame.index.astype(str)
    frame.index.name = x_title
    long = (frame.reset_index()
            .melt(id_vars=x_title, var_name="Série", value_name=y_title)
            .dropna(subset=[y_title]))
    if long.empty:
        st.info("Pas de données à tracer.")
        return
    base = alt.Chart(long).encode(
        x=alt.X(f"{x_title}:N", title=x_title, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Série:N", sort=order),
        y=alt.Y(f"{y_title}:Q", title=y_title),
        color=alt.Color("Série:N",
                        scale=alt.Scale(domain=order, range=_STACK_COLORS[: len(order)]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip(f"{x_title}:N"), alt.Tooltip("Série:N"),
                 alt.Tooltip(f"{y_title}:Q", format=value_format)],
    )
    bars = base.mark_bar()
    labels = base.mark_text(dy=-4, fontSize=10, color="#555").encode(
        text=alt.Text(f"{y_title}:Q", format=value_format))
    st.altair_chart((bars + labels).properties(title=title, height=320), use_container_width=True)


# --- Récupération (BEIR) ------------------------------------------------------

_BEIR = {"SciFact": "beir_scifact.json", "NFCorpus": "beir_nfcorpus.json",
         "HotpotQA": "beir_hotpotqa.json"}


def render_beir() -> None:
    import pandas as pd

    loaded = {label: d for label, f in _BEIR.items() if (d := _load(f))}
    if not loaded:
        st.info("Aucun snapshot BEIR dans `eval/reference/`. "
                "Génère-les avec `python -m eval.beir_eval …` puis copie-les là.")
        return

    st.markdown("**Qualité de récupération** — nDCG@10, *sans LLM*, jugements humains (qrels).")
    ndcg = {ds: {_short(k): v["ndcg@10"] for k, v in d["stacks"].items()} for ds, d in loaded.items()}
    _grouped_bar(_cols(pd.DataFrame(ndcg).T), "Dataset", "nDCG@10",
                 "nDCG@10 par dataset (plus haut = mieux)", ".3f")
    st.caption("L'Hybride gagne sur les 3 corpus ; le Graphe est dernier "
               "(dominé au sens de Pareto sur l'IR standard).")

    st.divider()
    ds = st.selectbox("Détail d'un dataset", list(loaded.keys()), key="beir_ds")
    d = loaded[ds]
    c = d["config"]
    st.caption(f"{c.get('n_docs', '?')} docs · {c.get('n_queries', '?')} requêtes · "
               f"embedder {c.get('embedder', '?')}")
    metrics = ["recall@1", "recall@5", "recall@10", "ndcg@10", "mrr"]
    table = {_short(k): {m: v.get(m) for m in metrics} for k, v in d["stacks"].items()}
    st.dataframe(_rows(pd.DataFrame(table).T)[metrics], use_container_width=True)


# --- Reranking ----------------------------------------------------------------

_RERANK = {"SciFact": "rerank_scifact.json", "NFCorpus": "rerank_nfcorpus.json",
           "HotpotQA": "rerank_hotpotqa.json"}


def render_reranking() -> None:
    import pandas as pd

    loaded = {label: d for label, f in _RERANK.items() if (d := _load(f))}
    if not loaded:
        st.info("Aucun snapshot reranking dans `eval/reference/`.")
        return

    st.markdown("**Reranking cross-encoder** — *replace* (suit le ré-évaluateur seul) "
                "vs *fuse* (RRF du rang de base + du ré-évaluateur).")
    mean_delta = {}
    for ds, d in loaded.items():
        vals = list(d["stacks"].values())
        n = len(vals)
        mean_delta[ds] = {"replace": round(sum(v["delta_replace"] for v in vals) / n, 3),
                          "fuse (RRF)": round(sum(v["delta_fusion"] for v in vals) / n, 3)}
    _grouped_bar(pd.DataFrame(mean_delta).T[["replace", "fuse (RRF)"]], "Dataset",
                 "Δ nDCG@10 moyen", "Gain moyen du reranking (le gagnant change selon le dataset)", ".3f")
    st.caption("La fusion **ne généralise pas** : elle ne gagne que sur SciFact (où *replace* "
               "abîme l'Hybride fort). Sur NFCorpus/HotpotQA, *replace* gagne. → mesurer par dataset.")

    st.divider()
    ds = st.selectbox("Détail d'un dataset", list(loaded.keys()), key="rerank_ds")
    table = {_short(k): {"sans": v["ndcg_base"], "replace": v["ndcg_replace"], "fuse": v["ndcg_fusion"]}
             for k, v in loaded[ds]["stacks"].items()}
    df = _cols(pd.DataFrame(table)).reindex(["sans", "replace", "fuse"])
    _grouped_bar(df, "Variante", "nDCG@10", f"{ds} — sans / replace / fuse par architecture", ".3f")
    st.dataframe(df, use_container_width=True)


# --- Systèmes -----------------------------------------------------------------

def render_systems() -> None:
    import pandas as pd

    d = _load("perf_scifact.json")
    if not d:
        st.info("Aucun snapshot systèmes (`perf_scifact.json`).")
        return
    c = d["config"]
    st.markdown(f"**Coût & vitesse** — *sans LLM*, {c.get('n_docs', '?')} docs, "
                f"{c.get('n_queries', '?')} requêtes, {c.get('repeats', '?')} répétitions.")

    bs = d["build_seconds"]
    build = {"Vector": bs.get("vector_total"), "Hybrid": bs.get("hybrid_total"), "Graph": bs.get("graph_total")}
    _grouped_bar(_cols(pd.DataFrame({"Construction (s)": build}).T), "Étape", "Secondes",
                 "Coût de construction de l'index (plus bas = mieux)", ".0f")
    st.caption(f"NER spaCy du graphe ≈ {bs.get('graph_ner_build', 0):.0f}s — l'index le plus cher.")

    st.divider()
    lat = {_short(k): {"médiane": v["latency"]["median_ms"], "p95": v["latency"]["p95_ms"],
                       "p99": v["latency"]["p99_ms"]} for k, v in d["stacks"].items()}
    _grouped_bar(_cols(pd.DataFrame(lat)).reindex(["médiane", "p95", "p99"]), "Percentile",
                 "Latence (ms)", "Latence par requête (plus bas = mieux)", ".1f")

    st.divider()
    thr = {_short(k): v["throughput_qps"] for k, v in d["stacks"].items()}
    thr_df = pd.DataFrame(thr)
    thr_df.index = thr_df.index.astype(int)
    thr_df = _cols(thr_df.sort_index())
    st.markdown("**Débit vs concurrence** (req/s) — seul le Vectoriel scale (FAISS libère le GIL).")
    st.line_chart(thr_df, x_label="Requêtes concurrentes", y_label="req/s")

    if PERF_PARETO.exists():
        st.divider()
        st.markdown("**Front de Pareto** qualité × latence")
        st.image(str(PERF_PARETO))


# --- Qualité de réponse -------------------------------------------------------

def render_answer() -> None:
    import pandas as pd

    runs = {"llama3.2:1b": _load("answer_1b.json"), "llama3.2:3b": _load("answer_3b.json")}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        st.info("Aucun snapshot de qualité de réponse.")
        return
    st.markdown("**Qualité de réponse de bout en bout** — F1 (style SQuAD) sur réponses-or "
                "HotpotQA, génération déterministe (température 0).")
    f1 = {model: {_short(k): v["f1"] for k, v in d["stacks"].items()} for model, d in runs.items()}
    _grouped_bar(_rows(pd.DataFrame(f1)), "Architecture", "F1",
                 "F1 par architecture et par modèle", ".3f")
    st.caption("La **taille du modèle domine** (3b ≫ 1b) ; les écarts *entre architectures* "
               "sont dans le bruit (n=50) — à ce stade local, le générateur est le goulot.")
