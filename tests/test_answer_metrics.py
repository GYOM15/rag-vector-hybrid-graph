"""Tests des métriques de réponse (Exact Match / F1, style SQuAD)."""

from answer_metrics import exact_match, f1_score, normalize_answer


def test_normalize_strips_case_punct_articles():
    assert normalize_answer("The Titanic, 1912!") == "titanic 1912"
    assert normalize_answer("  A  RMS   ship ") == "rms ship"


def test_exact_match_ignores_surface_form():
    assert exact_match("Kabul", "kabul") == 1.0
    assert exact_match("The Kabul.", "Kabul") == 1.0
    assert exact_match("Kabul", "Herat") == 0.0


def test_f1_partial_overlap():
    # Réponse verbeuse vs gold court : EM=0 mais F1 partiel.
    assert exact_match("The capital is Kabul", "Kabul") == 0.0
    assert 0.0 < f1_score("The capital is Kabul", "Kabul") < 1.0
    assert f1_score("Kabul", "Kabul") == 1.0
    assert f1_score("Herat", "Kabul") == 0.0


def test_f1_handles_empty():
    assert f1_score("", "") == 1.0
    assert f1_score("", "Kabul") == 0.0


def test_yes_no_answers():
    assert exact_match("Yes.", "yes") == 1.0
    assert f1_score("no", "yes") == 0.0
