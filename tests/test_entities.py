"""Tests for the deterministic proper-name checker.

The precision guards get the most attention here. A name checker that cries
wolf is worse than none, because a reviewer learns to skip it.
"""

from __future__ import annotations

import json

import pytest

from compton_eval.entities import (
    EntityVocabulary,
    _distance,
    _norm,
    check_names,
    load_vocabulary,
    score_trace,
)


@pytest.fixture
def vocab(tmp_path):
    payload = {
        "officials": [
            {"full_name": "Lillie P. Darden", "display_name": "CM Darden",
             "title": "Council Member", "is_current": True,
             "aliases": ["Councilmember Darden", "Darden", "Madam Chair"]},
            {"full_name": "Emma Sharif", "display_name": "Mayor Sharif",
             "title": "Mayor", "is_current": True,
             "aliases": ["Mayor Sharif", "Mayor", "Madam Mayor", "Sharif"]},
            {"full_name": "Douglas Sanders", "display_name": "Treasurer Sanders",
             "title": "City Treasurer", "is_current": False, "aliases": ["Sanders"]},
        ],
        "corrections": [
            {"garbled": "Silbert Perkins", "corrected": "Selbert Perkins",
             "category": "firm", "confidence": "high"},
            {"garbled": "Attorney Maxie Fallon", "corrected": "Attorney Maxie Fallon",
             "category": "person", "confidence": "low"},
        ],
        "staff": [{"full_name": "Jeanine McIntyre", "role": "Division Chief"}],
        "vendors": ["Selbert Perkins Design", "Sanders Roberts LLP",
                    "Bowman Infrastructure Engineers Ltd."],
    }
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(payload))
    return EntityVocabulary.load(p)


# ------------------------------------------------------------- primitives

def test_norm_strips_case_accents_punctuation():
    assert _norm("Olivárez, Madruga!") == "olivarez madruga"
    assert _norm("  Multiple   Spaces ") == "multiple spaces"


def test_distance_basics():
    assert _distance("darden", "darton") == 2
    assert _distance("robert", "roberts") == 1
    assert _distance("abc", "xyzxyzxyz") > 2   # early abandon


# ---------------------------------------------------------- known garbles

def test_recorded_garble_is_caught(vocab):
    f = check_names("A contract with Silbert Perkins Design was approved.", vocab)
    high = [x for x in f if x.confidence == "high"]
    assert len(high) == 1
    assert high[0].suggested == "Selbert Perkins"


def test_correct_spelling_is_not_flagged(vocab):
    f = check_names("A contract with Selbert Perkins Design was approved.", vocab)
    assert [x for x in f if x.confidence == "high"] == []


def test_garble_counted_across_repeats(vocab):
    text = "Silbert Perkins bid. Later, Silbert Perkins won."
    f = [x for x in check_names(text, vocab) if x.confidence == "high"]
    assert f[0].count == 2


def test_self_mapping_correction_row_ignored(vocab):
    """asr_corrections contains placeholder rows where garbled == corrected."""
    assert _norm("attorney maxie fallon") not in vocab.garbles


def test_markdown_links_do_not_hide_garbles(vocab):
    text = "approved [Silbert Perkins Design](https://example.com) for branding"
    assert any(x.confidence == "high" for x in check_names(text, vocab))


# ------------------------------------------------- the alias-as-garble bug

def test_bare_title_is_never_a_garble(vocab):
    """Regression: 'Mayor' is an alias of Emma Sharif but is not a misspelling.

    An earlier version treated any alias not sharing the canonical surname as
    a garble, so the word 'mayor' was reported as a misspelling of 'Emma
    Sharif' and fired on four real summaries.
    """
    f = check_names("The mayor announced a childcare program.", vocab)
    assert [x for x in f if x.confidence == "high"] == []
    assert "mayor" not in vocab.garbles
    assert "madam mayor" not in vocab.garbles


def test_aliases_become_known_good(vocab):
    assert _norm("Councilmember Darden") in vocab.known_good
    assert _norm("Madam Chair") in vocab.known_good


# ------------------------------------------------------------ near misses

def test_near_miss_finds_unrecorded_garble(vocab):
    f = check_names("Councilmember Darton was absent for the vote.", vocab)
    review = [x for x in f if x.confidence == "review"]
    assert review and review[0].suggested == "Lillie P. Darden"


def test_distance_two_requires_a_title(vocab):
    """Without person context, distance 2 pairs unrelated words."""
    titled = check_names("Councilmember Darton was absent.", vocab)
    bare = check_names("The Darton report was filed.", vocab)
    assert any(x.confidence == "review" for x in titled)
    assert not any(x.confidence == "review" for x in bare)


def test_multiword_entity_is_not_split(vocab):
    """'Roberts' is distance 1 from a surname but part of a known firm."""
    f = check_names("The firm Sanders Roberts LLP presented.", vocab)
    assert not any(x.found.lower().startswith("robert") for x in f)


def test_known_good_token_never_flagged(vocab):
    f = check_names("Treasurer Sanders gave the report.", vocab)
    assert [x for x in f if x.confidence == "review"] == []


def test_civic_boilerplate_not_flagged(vocab):
    f = check_names("Residents cited California Penal Code 424 at the podium.", vocab)
    assert [x for x in f if x.confidence == "review"] == []


def test_near_misses_can_be_disabled(vocab):
    f = check_names("Councilmember Darton was absent.", vocab,
                    include_near_misses=False)
    assert all(x.confidence == "high" for x in f)


# ------------------------------------------------------------- the scorer

def test_score_is_binary_and_high_confidence_only(vocab):
    clean, _ = score_trace("Selbert Perkins Design was approved.", vocab)
    assert clean == 0
    bad, f = score_trace("Silbert Perkins Design was approved.", vocab)
    assert bad == 1 and f

    # a near-miss alone must not trip the metric
    near, _ = score_trace("Councilmember Darton was absent.", vocab)
    assert near == 0


def test_empty_text_is_clean(vocab):
    assert score_trace("", vocab)[0] == 0


def test_add_garble_extends_vocabulary(vocab):
    assert score_trace("Councilmember Darton was absent.", vocab)[0] == 0
    vocab.add_garble("Darton", "Darden")
    assert score_trace("Councilmember Darton was absent.", vocab)[0] == 1


# ------------------------------------------------------- shipped fixture

def test_shipped_vocabulary_loads():
    v = load_vocabulary()
    assert v.size["garbles"] > 20
    assert v.size["canonical"] > 1000
    assert "mayor" not in v.garbles          # the regression, on real data
    assert _norm("Emma Sharif") in v.known_good
