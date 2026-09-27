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


# --- contains: guards against the wrong answers raw containment would credit ----

def test_contains_yes_no_needs_the_reply_to_open_with_the_gold():
    assert contains_answer("No, they are not both American.", "no") == 1.0
    assert contains_answer("Yes.", "yes") == 1.0
    assert contains_answer("**Yes** - both are directors", "yes") == 1.0
    # Refusals that merely mention "no" are not a "no" answer.
    assert contains_answer("I do not know; there is no information.", "no") == 0.0
    assert contains_answer("Unknown - no answer in the context", "no") == 0.0
    assert contains_answer("No information in the context.", "no") == 0.0
    assert contains_answer("No one knows.", "no") == 0.0
    # Wrong polarity, even when the gold word shows up later.
    assert contains_answer("Yes, but there is no proof.", "no") == 0.0


def test_contains_yes_no_hedges_score_zero_on_both_golds():
    for hedge in ("Yes and no.", "Yes, and no.", "yes or no", "Yes/no"):
        assert contains_answer(hedge, "yes") == 0.0, hedge
        assert contains_answer(hedge, "no") == 0.0, hedge


def test_contains_rejects_candidates_joined_by_an_alternative():
    assert contains_answer("1999, 2000, 2001 or 2002", "2001") == 0.0
    assert contains_answer("either 1999 or 2001", "2001") == 0.0
    assert contains_answer("Paris vs Lyon", "Paris") == 0.0
    # ... unless the gold also stands on its own somewhere.
    assert contains_answer("Paris or Lyon? It was Paris.", "Paris") == 1.0
    # Known limit (documented): a plain comma list still contains the gold.
    assert contains_answer("1999, 2000, 2001", "2001") == 1.0


def test_contains_on_choice_questions_needs_the_chosen_option_first():
    q = "Who is older, Annie Morton or Terry Richardson?"
    assert contains_answer("Terry Richardson", "Terry Richardson", q) == 1.0
    assert contains_answer("Terry Richardson is older.", "Terry Richardson", q) == 1.0
    assert contains_answer(q, "Terry Richardson", q) == 0.0  # restating the question
    assert contains_answer("Annie Morton or Terry Richardson", "Terry Richardson", q) == 0.0
    assert contains_answer("Annie Morton is older than Terry Richardson", "Terry Richardson",
                           q) == 0.0
    q = "Which band, Letters to Cleo or Screaming Trees, had more members?"
    assert contains_answer("Letters to Cleo had more members.", "Letters to Cleo", q) == 1.0
    assert contains_answer("Letters to Cleo or Screaming Trees", "Letters to Cleo", q) == 0.0
    assert contains_answer("Letters to Cleo and Screaming Trees", "Letters to Cleo", q) == 0.0
    # Without the question the plain rule applies (backward compatible call).
    assert contains_answer("Terry Richardson is older.", "Terry Richardson") == 1.0
