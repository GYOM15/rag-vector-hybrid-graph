"""Tests for the dashboard's data-derived captions (pure functions, no streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import eval_insights as ei  # noqa: E402


def _beir(ndcg: dict[str, float], ci: dict | None = None, paired: dict | None = None) -> dict:
    names = {"Vector": "Stack 1 — Vector (FAISS)", "Hybrid": "Stack 2 — Hybrid (BM25 + RRF)",
             "Graph": "Stack 3 — Graph (networkx)"}
    stacks = {names[s]: {"ndcg@10": v} | ({"ci95": {"ndcg@10": ci[s]}} if ci else {})
              for s, v in ndcg.items()}
    return {"stacks": stacks} | ({"paired": paired} if paired else {})


def test_short_name_handles_every_key_style():
    assert ei.short_name("Stack 2 — Hybride (BM25 + RRF)") == "Hybrid"  # old French snapshots
    assert ei.short_name("vector") == "Vector"
    assert ei.short_name("Graph") == "Graph"


def test_oriented_pair_flips_a_reversed_pair():
    paired = {"Hybrid-Vector": {"mean_diff": 0.02, "lo": 0.01, "hi": 0.03, "p_value": 0.01}}
    assert ei.oriented_pair(paired, "Hybrid", "Vector")["mean_diff"] == 0.02
    flipped = ei.oriented_pair(paired, "Vector", "Hybrid")
    assert (flipped["mean_diff"], flipped["lo"], flipped["hi"]) == (-0.02, -0.03, -0.01)
    assert ei.oriented_pair(None, "Vector", "Hybrid") is None


def test_significance_needs_small_p_and_a_ci_excluding_zero():
    assert ei.is_significant({"p_value": 0.01, "lo": 0.01, "hi": 0.05})
    assert not ei.is_significant({"p_value": 0.20, "lo": 0.01, "hi": 0.05})
    assert not ei.is_significant({"p_value": 0.01, "lo": -0.01, "hi": 0.05})


def test_beir_caption_uses_the_paired_test_when_present():
    paired = {"Hybrid-Vector": {"mean_diff": 0.06, "lo": 0.03, "hi": 0.09, "p_value": 0.0002},
              "Graph-Vector": {"mean_diff": 0.0, "lo": -0.01, "hi": 0.01, "p_value": 0.9},
              "Graph-Hybrid": {"mean_diff": -0.06, "lo": -0.09, "hi": -0.03, "p_value": 0.0002}}
    text = ei.beir_caption({"SciFact": _beir({"Vector": 0.65, "Hybrid": 0.71, "Graph": 0.65},
                                             paired=paired)})
    assert text.startswith("Hybrid ranks first on this dataset")
    assert "significant, paired p=<0.001" in text


def test_beir_caption_falls_back_to_ci_overlap_then_to_unknown():
    ci = {"Vector": [0.60, 0.70], "Hybrid": [0.66, 0.76], "Graph": [0.55, 0.65]}
    with_ci = ei.beir_caption({"A": _beir({"Vector": 0.65, "Hybrid": 0.71, "Graph": 0.60}, ci=ci)})
    assert "within noise, 95% CIs overlap" in with_ci
    old = ei.beir_caption({"A": _beir({"Vector": 0.65, "Hybrid": 0.71, "Graph": 0.60}),
                           "B": _beir({"Vector": 0.80, "Hybrid": 0.70, "Graph": 0.60})})
    assert old.startswith("The leader changes by dataset")
    assert "significance unknown, no CI in this snapshot" in old


def test_rerank_caption_names_the_better_mode_per_dataset():
    snap = {"stacks": {"Vector": {"delta_replace": 0.05, "delta_fusion": 0.01},
                       "Hybrid": {"delta_replace": 0.03, "delta_fusion": 0.02}}}
    text = ei.rerank_caption({"HotpotQA": snap})
    assert text.startswith("*replace* gains more on this dataset")
    assert "replace +0.040 vs fusion +0.015" in text


def test_throughput_caption_reads_scaling_off_the_data():
    stacks = {"vector": {"throughput_qps": {"1": 90.0, "4": 120.0}},
              "hybrid": {"throughput_qps": {"1": 55.0, "4": 61.0}},
              "graph": {"throughput_qps": {"1": 67.0, "4": 90.0}}}
    text = ei.throughput_caption(stacks)
    assert "Graph +34%" in text and "Vector +33%" in text and "Hybrid +11%" in text
    assert text.endswith("Graph and Vector gain ≥ 20%.")  # not "only Vector scales"


def test_build_caption_names_the_costliest_index():
    text = ei.build_caption({"vector_total": 45, "hybrid_total": 63, "graph_total": 225,
                             "graph_ner_build": 180})
    assert text == "Graph is the costliest index to build (225s), of which spaCy NER ≈ 180s."


def test_answer_caption_ranks_runs_and_flags_old_snapshots():
    def run(model, f1s, **extra):
        return {"config": {"model": model} | extra,
                "stacks": {s: {"f1": v, "n_queries": 50} for s, v in zip(
                    ("Vector", "Hybrid", "Graph"), f1s)}}
    runs = {ei.run_label(r, "fallback"): r for r in (
        run("llama3.2:1b", [0.07, 0.05, 0.06]), run("llama3.2:3b", [0.15, 0.18, 0.20]),
        run("?", [0.1, 0.1, 0.1]))}
    assert list(runs) == ["llama3.2:1b", "llama3.2:3b", "fallback"]
    text = ei.answer_caption(runs)
    assert text.startswith("Mean F1 over the stacks: llama3.2:3b 0.177")
    assert "llama3.2:3b: Graph − Vector = +0.050 (significance unknown" in text
    assert "fallback: all stacks equal" in text


def test_run_label_mentions_a_non_default_prompt():
    assert ei.run_label({"config": {"model": "llama3.2:3b", "prompt": "short"}}, "x") \
        == "llama3.2:3b (short prompt)"
    assert ei.run_label({"config": {"model": "llama3.2:3b", "prompt": "default"}}, "x") \
        == "llama3.2:3b"


def test_provenance_note():
    assert ei.provenance_note({}) == ""
    cfg = {"provenance": {"git_sha": "4308fe0da5bc20b5", "git_dirty": True}}
    assert ei.provenance_note(cfg) == " · code 4308fe0 + uncommitted changes"
    cfg["provenance"] |= {"git_dirty": False, "code_changed_during_run": True}
    assert ei.provenance_note(cfg) == " · code 4308fe0 (checkout changed during the run)"


def test_label_runs_keeps_every_file_when_labels_collide():
    def snap(model, n):
        return {"config": {"model": model, "n_queries": n}, "stacks": {}}
    runs = ei.label_runs({"1b": snap("llama3.2:1b", 50), "1b_rerun": snap("llama3.2:1b", 4),
                          "3b": snap("llama3.2:3b", 50)})
    assert list(runs) == ["llama3.2:1b · 1b", "llama3.2:1b · 1b_rerun", "llama3.2:3b"]
    assert runs["llama3.2:1b · 1b_rerun"]["config"]["n_queries"] == 4


def test_answer_caption_shows_n_and_flags_unlike_or_verbose_runs():
    def run(model, f1s, n, **cfg):
        return {"config": {"model": model, "n_queries": n} | cfg,
                "stacks": {s: {"f1": v} for s, v in zip(("Vector", "Hybrid", "Graph"), f1s)}}
    short = run("llama3.2:1b", [0.30, 0.32, 0.31], 100, prompt="short")
    runs = ei.label_runs({"7b": run("?", [0.10, 0.11, 0.12], 50), "1b_short": short})
    text = ei.answer_caption(runs)
    assert text.startswith("Mean F1 over the stacks: llama3.2:1b (short prompt) 0.310 (n=100), "
                           "7b 0.110 (n=50)")
    assert "not a like-for-like ranking" in text
    # The old run (no "prompt" recorded) used the default prompt: F1 is flagged as verbosity.
    assert "7b used the default prompt" in text and "mostly measures verbosity" in text
    # Same setup everywhere, short prompt: no caveat.
    same = ei.answer_caption({"a": short, "b": run("llama3.2:3b", [0.4] * 3, 100, prompt="short")})
    assert "like-for-like" not in same and "verbosity" not in same
