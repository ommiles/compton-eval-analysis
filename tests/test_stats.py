"""Tests for the statistical core.

These check properties that must hold rather than golden numbers, because
the point of the package is that the numbers are uncertain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compton_eval.bootstrap import bootstrap_mean, bootstrap_paired_delta
from compton_eval.compare import cliffs_delta, compare_run, holm_bonferroni
from compton_eval.load import EvalRun
from compton_eval.power import simulate_power
from compton_eval.reliability import check_repeatability, icc_2_1, krippendorff_alpha


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=30)
    ci = bootstrap_mean(x, seed=1)
    assert ci.low <= ci.point <= ci.high


def test_bootstrap_respects_bounds():
    """A proportion's CI must not escape [0, 1] — the failure mode a
    t-interval has on scores like these."""
    x = np.full(12, 0.97)
    x[0] = 0.90
    ci = bootstrap_mean(x, bounds=(0.0, 1.0), seed=1)
    assert ci.high <= 1.0
    assert ci.low >= 0.0


def test_constant_vector_is_degenerate_not_nan():
    """v0 scores exactly 0.0 on anomaly surfacing for every meeting."""
    ci = bootstrap_mean(np.zeros(12))
    assert ci.method == "degenerate"
    assert ci.point == ci.low == ci.high == 0.0
    assert not np.isnan(ci.point)


def test_empty_input_returns_empty_interval():
    ci = bootstrap_mean([])
    assert ci.n == 0 and ci.method == "empty"


def test_paired_delta_uses_pairing():
    """Perfectly correlated pairs with a constant offset have zero variance
    in their difference, so the paired interval must be tight — an unpaired
    analysis would report a wide one."""
    a = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4])
    b = a + 0.1
    d = bootstrap_paired_delta(a, b)
    assert d.point == pytest.approx(0.1)
    assert d.width == pytest.approx(0.0, abs=1e-9)


def test_paired_delta_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        bootstrap_paired_delta([1, 2, 3], [1, 2])


# ------------------------------------------------------------ cliffs delta

def test_cliffs_delta_bounds():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    d, label = cliffs_delta(a, b)
    assert d == pytest.approx(1.0)
    assert label == "large"

    d, _ = cliffs_delta(b, a)
    assert d == pytest.approx(-1.0)


def test_cliffs_delta_identical_is_zero():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    d, label = cliffs_delta(x, x)
    assert d == pytest.approx(0.0)
    assert label == "negligible"


# --------------------------------------------------------------------- holm

def test_holm_is_monotone_and_bounded():
    p = [0.001, 0.01, 0.03, 0.04, 0.9]
    adj = holm_bonferroni(p)
    assert all(0.0 <= v <= 1.0 for v in adj)
    # order preserved
    assert adj == sorted(adj)


def test_holm_is_less_conservative_than_bonferroni():
    p = [0.001, 0.02, 0.03]
    adj = holm_bonferroni(p)
    bonf = [min(v * len(p), 1.0) for v in p]
    assert all(a <= b + 1e-12 for a, b in zip(adj, bonf))
    assert any(a < b for a, b in zip(adj, bonf))


def test_holm_single_test_is_identity():
    assert holm_bonferroni([0.04])[0] == pytest.approx(0.04)


def test_holm_empty():
    assert holm_bonferroni([]) == []


# -------------------------------------------------------------------- power

def test_power_increases_with_sample_size():
    rng = np.random.default_rng(3)
    diffs = rng.normal(0.0, 0.15, size=20)
    sizes, powers = simulate_power(
        diffs, effect=0.15, sample_sizes=[10, 40, 100], n_simulations=600
    )
    assert powers[0] < powers[-1]
    assert powers[-1] > 0.8


def test_zero_effect_gives_power_near_alpha():
    """With no true effect, rejection rate should sit near alpha — this is
    the check that the simulation is not systematically optimistic."""
    rng = np.random.default_rng(4)
    diffs = rng.normal(0.0, 0.2, size=25)
    _, powers = simulate_power(
        diffs, effect=0.0, sample_sizes=[40], alpha=0.05, n_simulations=3000
    )
    assert powers[0] < 0.10


def test_power_needs_variance():
    with pytest.raises(ValueError, match="at least 2"):
        simulate_power([0.5], effect=0.1)


# -------------------------------------------------------------- reliability

def test_icc_perfect_agreement():
    x = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [2.0, 2.0]])
    assert icc_2_1(x) == pytest.approx(1.0, abs=1e-9)


def test_icc_no_agreement_is_low():
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 5, size=(40, 3))
    assert icc_2_1(x) < 0.3


def test_krippendorff_perfect_agreement():
    x = np.array([[2.0, 2.0], [4.0, 4.0], [5.0, 5.0], [1.0, 1.0]])
    assert krippendorff_alpha(x) == pytest.approx(1.0, abs=1e-9)


def test_reliability_reports_missing_data_honestly():
    """A single-pass run must not silently return a number."""
    df = pd.DataFrame(
        {
            "meeting_id": [1, 1, 2, 2],
            "variant": ["v0", "v1", "v0", "v1"],
            "tone_score": [2, 4, 3, 5],
        }
    )
    rel = check_repeatability(df)
    assert rel.available is False
    assert "no repeated judgments" in rel.message
    assert np.isnan(rel.icc)


def test_reliability_computes_when_passes_present():
    rows = []
    for m in range(8):
        for v in ("v0", "v1"):
            for p in range(3):
                rows.append(
                    {"meeting_id": m, "variant": v, "pass": p,
                     "tone_score": 2 + (v == "v1") * 2}
                )
    rel = check_repeatability(pd.DataFrame(rows))
    assert rel.available is True
    assert rel.n_raters == 3


# ----------------------------------------------------------------- compare

def _run(frame: pd.DataFrame) -> EvalRun:
    return EvalRun(run_id="test", scores=frame)


def test_identical_variants_are_not_significant():
    rows = []
    for m in range(12):
        val = 0.5 + m * 0.01
        rows.append({"meeting_id": m, "variant": "v1", "factual_precision": val})
        rows.append({"meeting_id": m, "variant": "v2", "factual_precision": val})
    out = compare_run(_run(pd.DataFrame(rows)), n_resamples=500)
    assert not out["significant"].any()
    assert "identical" in out.iloc[0]["note"]


def test_large_consistent_effect_is_significant():
    rows = []
    rng = np.random.default_rng(7)
    for m in range(12):
        base = rng.uniform(0.1, 0.3)
        rows.append({"meeting_id": m, "variant": "v0", "factual_precision": base})
        rows.append({"meeting_id": m, "variant": "v1", "factual_precision": base + 0.5})
    out = compare_run(_run(pd.DataFrame(rows)), n_resamples=500)
    assert out.iloc[0]["significant"]
    assert out.iloc[0]["effect"] == "large"


def test_incomplete_pairs_are_dropped_not_imputed():
    rows = [
        {"meeting_id": 1, "variant": "v0", "tone_score": 2},
        {"meeting_id": 1, "variant": "v1", "tone_score": 4},
        {"meeting_id": 2, "variant": "v0", "tone_score": 3},
        # meeting 2 has no v1 row
    ]
    run = _run(pd.DataFrame(rows))
    pairs, total = run.pairable("tone_score")
    assert pairs == 1 and total == 2
