"""Paired variant comparison: significance, effect size, multiplicity.

The design is a within-subjects comparison — every prompt variant is scored
on the same meetings — so the right test is a paired one. Using an unpaired
Mann-Whitney here would throw away the pairing and lose most of the power
the design was built to provide.

Three choices worth defending:

**Wilcoxon signed-rank, not a paired t-test.** The scores are bounded
proportions and an ordinal 1-5 judgment. Normality is not on offer at n = 12.

**Cliff's delta, not Cohen's d.** Cohen's d is a ratio of a mean difference
to a standard deviation, which presumes an interval scale. Cliff's delta
asks a question that survives ordinal data: pick a random meeting under each
variant, how much more often does one beat the other?

**Holm-Bonferroni, not raw p-values.** Four dimensions times three variant
pairs is twelve tests. At alpha = 0.05 you expect roughly one spurious
"significant" result per run by chance alone. Holm controls the family-wise
error rate without Bonferroni's conservatism.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .bootstrap import Interval, bootstrap_paired_delta
from .load import DIMENSIONS, EvalRun


@dataclass(frozen=True)
class Comparison:
    """One variant-pair comparison on one dimension."""

    dimension: str
    variant_a: str
    variant_b: str
    n_pairs: int
    mean_a: float
    mean_b: float
    delta: Interval
    cliffs_delta: float
    cliffs_label: str
    p_raw: float
    p_adjusted: float
    significant: bool
    note: str = ""

    @property
    def direction(self) -> str:
        if not np.isfinite(self.cliffs_delta) or self.cliffs_delta == 0:
            return "="
        return ">" if self.mean_b > self.mean_a else "<"


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Cliff's delta and its conventional magnitude label.

    delta = P(b > a) - P(a > b), on [-1, 1]. Zero means the two are
    indistinguishable by rank; +1 means every b beats every a.

    Thresholds follow Romano et al. (2006): |d| < 0.147 negligible,
    < 0.33 small, < 0.474 medium, else large.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan"), "undefined"

    # Pairwise sign comparison; O(n*m) is fine at these sizes.
    diff = np.sign(b[:, None] - a[None, :])
    d = float(diff.mean())

    mag = abs(d)
    if not np.isfinite(d):
        label = "undefined"
    elif mag < 0.147:
        label = "negligible"
    elif mag < 0.330:
        label = "small"
    elif mag < 0.474:
        label = "medium"
    else:
        label = "large"
    return d, label


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Returns adjusted p-values in the input order. Compare against alpha
    directly. Enforces monotonicity so an adjusted value never falls below
    one that preceded it in the sorted order.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return []

    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return [float(v) for v in adjusted]


def compare_run(
    run: EvalRun,
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Every variant pair on every dimension, multiplicity-corrected.

    Returns a tidy frame, one row per comparison, sorted so the significant
    results come first.
    """
    raw: list[Comparison] = []
    dims = [d for d in DIMENSIONS if d in run.scores.columns]

    for dim in dims:
        wide = run.dimension(dim)
        if wide.empty:
            continue
        variants = [v for v in run.variants if v in wide.columns]

        for va, vb in combinations(variants, 2):
            a = wide[va].to_numpy(dtype=float)
            b = wide[vb].to_numpy(dtype=float)
            n = a.size

            d_ci = bootstrap_paired_delta(
                a, b, n_resamples=n_resamples, seed=seed
            )
            cd, cd_label = cliffs_delta(a, b)

            note = ""
            diffs = b - a
            if n < 6:
                # Signed-rank cannot reach alpha=0.05 two-sided below n=6:
                # the smallest attainable p is 2/2**n.
                p = float("nan")
                note = f"n={n} too small for signed-rank (min attainable p = {2 / 2**n:.3f})"
            elif np.allclose(diffs, 0):
                p = 1.0
                note = "identical on every pair"
            else:
                try:
                    p = float(
                        stats.wilcoxon(
                            a, b, zero_method="wilcox", alternative="two-sided"
                        ).pvalue
                    )
                except ValueError as exc:  # all-zero differences, etc.
                    p = float("nan")
                    note = str(exc)

            raw.append(
                Comparison(
                    dimension=dim,
                    variant_a=va,
                    variant_b=vb,
                    n_pairs=n,
                    mean_a=float(np.nanmean(a)),
                    mean_b=float(np.nanmean(b)),
                    delta=d_ci,
                    cliffs_delta=cd,
                    cliffs_label=cd_label,
                    p_raw=p,
                    p_adjusted=float("nan"),
                    significant=False,
                    note=note,
                )
            )

    # Correct across the whole family of tests that actually ran.
    testable = [i for i, c in enumerate(raw) if np.isfinite(c.p_raw)]
    adj = holm_bonferroni([raw[i].p_raw for i in testable], alpha=alpha)
    adj_map = dict(zip(testable, adj))

    rows = []
    for i, c in enumerate(raw):
        p_adj = adj_map.get(i, float("nan"))
        rows.append(
            {
                "dimension": c.dimension,
                "variant_a": c.variant_a,
                "variant_b": c.variant_b,
                "n_pairs": c.n_pairs,
                "mean_a": c.mean_a,
                "mean_b": c.mean_b,
                "delta": c.delta.point,
                "delta_lo": c.delta.low,
                "delta_hi": c.delta.high,
                "cliffs_delta": c.cliffs_delta,
                "effect": c.cliffs_label,
                "p_raw": c.p_raw,
                "p_holm": p_adj,
                "significant": bool(np.isfinite(p_adj) and p_adj < alpha),
                "note": c.note,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["significant", "p_holm"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
