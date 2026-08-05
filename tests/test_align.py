"""Tests for judge alignment and bias correction.

The central property: when the judge's error rates are known, the correction
must recover the true rate that the raw judge output misses. That is the whole
claim, so it gets tested directly against a construction with a known answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from compton_eval.align import (
    GOOD_RATE,
    MIN_INFORMEDNESS,
    correct_rate,
    corrected_success_rate,
    measure_alignment,
    split_labeled,
)


def _judge_output(n_pass: int, n_fail: int, tpr: float, tnr: float, seed: int = 0):
    """Human labels plus judge predictions with prescribed TPR and TNR."""
    rng = np.random.default_rng(seed)
    human = np.array([True] * n_pass + [False] * n_fail)
    judge = np.empty_like(human)
    judge[:n_pass] = rng.random(n_pass) < tpr
    judge[n_pass:] = rng.random(n_fail) >= tnr
    return human, judge


# ------------------------------------------------------------- alignment

def test_perfect_judge():
    human = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    a = measure_alignment(human, human)
    assert a.tpr == 1.0 and a.tnr == 1.0
    assert a.informedness == pytest.approx(1.0)
    assert a.usable


def test_rates_computed_correctly():
    #        pass pass pass fail fail fail
    human = [1, 1, 1, 0, 0, 0]
    judge = [1, 1, 0, 0, 0, 1]     # 2/3 passes caught, 2/3 fails caught
    a = measure_alignment(human, judge)
    assert a.tpr == pytest.approx(2 / 3)
    assert a.tnr == pytest.approx(2 / 3)


def test_single_class_test_set_rejected():
    with pytest.raises(ValueError, match="only one class"):
        measure_alignment([1, 1, 1], [1, 1, 0])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="must match"):
        measure_alignment([1, 0, 1], [1, 0])


def test_warnings_flag_small_and_weak():
    human, judge = _judge_output(10, 10, tpr=0.6, tnr=0.6)
    a = measure_alignment(human, judge)
    joined = " ".join(a.warnings)
    assert "Pass examples" in joined and "Fail examples" in joined
    assert f"{GOOD_RATE:.0%}" in joined


def test_no_warnings_when_judge_is_good_and_set_is_big():
    human, judge = _judge_output(60, 60, tpr=0.97, tnr=0.97, seed=3)
    a = measure_alignment(human, judge)
    assert a.warnings == []


# ------------------------------------------------------- Rogan-Gladen

def test_correction_recovers_known_true_rate():
    """The book's worked setup: true rate 80%, imperfect judge.

    With TPR=0.9 and TNR=0.8, a judge run over a population that is truly
    80% pass observes 0.8*0.9 + 0.2*0.2 = 0.76. The correction must undo
    exactly that.
    """
    tpr, tnr, truth = 0.90, 0.80, 0.80
    p_obs = truth * tpr + (1 - truth) * (1 - tnr)
    assert p_obs == pytest.approx(0.76)
    assert correct_rate(p_obs, tpr, tnr) == pytest.approx(truth)


def test_correction_is_identity_for_a_perfect_judge():
    for p in (0.0, 0.25, 0.5, 0.99, 1.0):
        assert correct_rate(p, 1.0, 1.0) == pytest.approx(p)


def test_chance_level_judge_returns_nan():
    assert np.isnan(correct_rate(0.7, 0.5, 0.5))


def test_below_chance_judge_returns_nan():
    """Negative informedness must be refused, not sign-flipped into an answer."""
    assert np.isnan(correct_rate(0.7, 0.2, 0.3))


def test_barely_informative_judge_returns_nan():
    tpr, tnr = 0.5 + MIN_INFORMEDNESS / 4, 0.5 + MIN_INFORMEDNESS / 4
    assert 0 < tpr + tnr - 1 < MIN_INFORMEDNESS
    assert np.isnan(correct_rate(0.7, tpr, tnr))


def test_result_stays_in_unit_interval():
    for p in (0.0, 1.0):
        v = correct_rate(p, 0.7, 0.7)
        assert 0.0 <= v <= 1.0


# ------------------------------------------------ corrected_success_rate

def test_corrected_estimate_beats_raw_observation():
    """The point of the exercise: raw is biased, corrected is not."""
    tpr, tnr, truth = 0.85, 0.90, 0.70
    human, judge = _judge_output(200, 200, tpr=tpr, tnr=tnr, seed=1)

    m = 4000
    p_obs = truth * tpr + (1 - truth) * (1 - tnr)
    k = int(round(p_obs * m))

    r = corrected_success_rate(
        human, judge, n_unlabeled=m, n_judged_pass=k, n_resamples=2000
    )
    assert abs(r.corrected - truth) < abs(r.observed - truth)
    assert r.low <= truth <= r.high
    assert r.shift != 0


def test_interval_widens_with_a_worse_judge():
    truth = 0.75
    widths = []
    for tpr in (0.95, 0.65):
        human, judge = _judge_output(150, 150, tpr=tpr, tnr=0.95, seed=2)
        p_obs = truth * tpr + (1 - truth) * 0.05
        r = corrected_success_rate(
            human, judge, n_unlabeled=3000,
            n_judged_pass=int(round(p_obs * 3000)), n_resamples=2000,
        )
        widths.append(r.width)
    assert widths[1] > widths[0]


def test_batch_uncertainty_widens_the_interval():
    human, judge = _judge_output(100, 100, tpr=0.9, tnr=0.9, seed=4)
    kw = dict(n_unlabeled=60, n_judged_pass=42, n_resamples=2000)
    narrow = corrected_success_rate(human, judge, **kw)
    wide = corrected_success_rate(human, judge, include_batch_uncertainty=True, **kw)
    assert wide.width > narrow.width


def test_unusable_judge_reports_instead_of_guessing():
    human, judge = _judge_output(50, 50, tpr=0.5, tnr=0.5, seed=5)
    r = corrected_success_rate(human, judge, n_unlabeled=1000, n_judged_pass=500)
    assert np.isnan(r.corrected)
    assert "chance" in r.note
    assert np.isfinite(r.observed)


def test_invalid_batch_counts_rejected():
    human, judge = _judge_output(30, 30, tpr=0.9, tnr=0.9)
    with pytest.raises(ValueError, match="n_unlabeled must be positive"):
        corrected_success_rate(human, judge, n_unlabeled=0, n_judged_pass=0)
    with pytest.raises(ValueError, match="outside"):
        corrected_success_rate(human, judge, n_unlabeled=10, n_judged_pass=11)


def test_reproducible_across_runs():
    human, judge = _judge_output(80, 80, tpr=0.88, tnr=0.92, seed=6)
    kw = dict(n_unlabeled=500, n_judged_pass=350, n_resamples=1000, seed=99)
    a = corrected_success_rate(human, judge, **kw)
    b = corrected_success_rate(human, judge, **kw)
    assert (a.corrected, a.low, a.high) == (b.corrected, b.low, b.high)


# ------------------------------------------------------------- splits

def test_split_proportions_and_disjointness():
    tr, dv, te = split_labeled(1000, train=0.15, dev=0.425)
    assert len(tr) + len(dv) + len(te) == 1000
    assert not (set(tr) & set(dv)) and not (set(dv) & set(te)) and not (set(tr) & set(te))
    assert 130 <= len(tr) <= 170
    assert len(dv) > len(tr) and len(te) > len(tr)


def test_split_is_stratified():
    labels = np.array([True] * 300 + [False] * 100)
    tr, dv, te = split_labeled(400, labels=labels)
    for part in (tr, dv, te):
        frac = labels[part].mean()
        assert 0.68 <= frac <= 0.82, f"stratification drifted: {frac:.2f}"


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError, match="invalid split"):
        split_labeled(100, train=0.6, dev=0.6)


def test_split_rejects_length_mismatch():
    with pytest.raises(ValueError, match="!= n"):
        split_labeled(10, labels=[True] * 5)
