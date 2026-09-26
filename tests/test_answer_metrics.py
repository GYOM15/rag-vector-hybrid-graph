"""Tests for the answer metrics (Exact Match / F1, SQuAD-style)."""

from answer_metrics import contains_answer, exact_match, f1_score, normalize_answer


def test_normalize_strips_case_punct_articles():
    assert normalize_answer("The Titanic, 1912!") == "titanic 1912"
    assert normalize_answer("  A  RMS   ship ") == "rms ship"


def test_exact_match_ignores_surface_form():
    assert exact_match("Kabul", "kabul") == 1.0
    assert exact_match("The Kabul.", "Kabul") == 1.0
    assert exact_match("Kabul", "Herat") == 0.0


def test_f1_partial_overlap():
    # Verbose answer vs short gold: EM=0 but partial F1.
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


def test_contains_accepts_correct_verbose_answers():
    assert contains_answer("The capital of Afghanistan is Kabul.", "Kabul") == 1.0
    assert contains_answer("It was designed by Gustave Eiffel in 1889", "gustave eiffel") == 1.0
    assert contains_answer("Kabul", "Kabul") == 1.0
    # Normalization applies to both sides: case, punctuation, articles.
    assert contains_answer("Yes, it is.", "yes") == 1.0
    assert contains_answer("the RMS Titanic!", "The RMS Titanic") == 1.0


def test_contains_requires_whole_tokens():
    assert contains_answer("Kabuli pulao is a dish", "Kabul") == 0.0
    assert contains_answer("a playwright", "Wright") == 0.0
    assert contains_answer("1,000 metres", "1000") == 1.0  # punctuation stripped inside tokens


def test_contains_requires_a_contiguous_run():
    assert contains_answer("New and old York", "New York") == 0.0
    assert contains_answer("York, New", "New York") == 0.0
    assert contains_answer("in New York City", "New York") == 1.0


def test_contains_edge_cases():
    assert contains_answer("anything", "") == 0.0
    assert contains_answer("", "Kabul") == 0.0
    assert contains_answer("unknown", "Kabul") == 0.0
