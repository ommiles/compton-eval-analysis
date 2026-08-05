"""Is the LLM judge measuring anything stable?

The tone dimension is scored by an LLM against fixed calibration anchors.
Anchoring pins the *scale* — it does not establish that the judge returns
the same answer twice. If the judge is unreliable, every tone comparison
downstream is noise dressed as measurement, and no amount of sample size
fixes it.

Reliability needs repeated judgments: the same (meeting, variant) scored
k > 1 times. A single-pass run cannot support it, so
:func:`check_repeatability` reports honestly that the data is missing
rather than computing something meaningless.

Two coefficients, because they answer different questions:

- **ICC(2,1)** — absolute agreement for a single rating. "If I score this
  once, how much of the variance is real signal?"
- **Krippendorff's alpha** — chance-corrected agreement that tolerates
  missing cells and ordinal data.

Interpretation follows Koo & Li (2016): ICC < 0.50 poor, < 0.75 moderate,
< 0.90 good, else excellent. For alpha, Krippendorff's own guidance is
>= 0.80 for firm conclusions, >= 0.667 for tentative ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reliability:
    """Agreement across repeated judgments of the same items."""

    n_items: int
    n_raters: int
    icc: float
    icc_label: str
    krippendorff_alpha: float
    alpha_label: str
    available: bool
    message: str = ""

    def __str__(self) -> str:
        if not self.available:
            return f"reliability unavailable: {self.message}"
        return (
            f"ICC(2,1)={self.icc:.3f} ({self.icc_label}), "
            f"alpha={self.krippendorff_alpha:.3f} ({self.alpha_label}), "
            f"{self.n_items} items x {self.n_raters} passes"
        )


def _icc_label(v: float) -> str:
    if not np.isfinite(v):
        return "undefined"
    if v < 0.50:
        return "poor"
    if v < 0.75:
        return "moderate"
    if v < 0.90:
        return "good"
    return "excellent"


def _alpha_label(v: float) -> str:
    if not np.isfinite(v):
        return "undefined"
    if v >= 0.80:
        return "acceptable"
    if v >= 0.667:
        return "tentative only"
    return "unacceptable"


def icc_2_1(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater.

    ``ratings`` is (n_items, n_raters). Follows Shrout & Fleiss (1979).
    """
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")

    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (residual**2).sum() / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom == 0:
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def krippendorff_alpha(ratings: np.ndarray, *, level: str = "ordinal") -> float:
    """Krippendorff's alpha for ordinal or interval data.

    ``ratings`` is (n_items, n_raters); NaN marks a missing judgment.
    """
    x = np.asarray(ratings, dtype=float)

    def metric(a: float, b: float) -> float:
        return (a - b) ** 2  # interval metric; adequate for a 1-5 anchored scale

    # Observed disagreement: within-item pairs.
    num_o, den_o = 0.0, 0
    for row in x:
        vals = row[~np.isnan(row)]
        m = vals.size
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    num_o += metric(vals[i], vals[j])
        den_o += m * (m - 1)
    if den_o == 0:
        return float("nan")
    d_o = num_o / den_o

    # Expected disagreement: all pairs across the whole pool.
    pool = x[~np.isnan(x)]
    n_pool = pool.size
    if n_pool < 2:
        return float("nan")
    num_e = sum(
        metric(pool[i], pool[j])
        for i in range(n_pool)
        for j in range(n_pool)
        if i != j
    )
    d_e = num_e / (n_pool * (n_pool - 1))

    if d_e == 0:
        return float("nan")
    return float(1 - d_o / d_e)


def check_repeatability(
    scores: pd.DataFrame,
    *,
    dimension: str = "tone_score",
    item_keys: tuple[str, ...] = ("meeting_id", "variant"),
) -> Reliability:
    """Measure judge reliability, or explain why it cannot be measured.

    Expects a ``pass`` (or ``replicate``) column identifying repeated
    scoring runs over the same items. Without one, returns
    ``available=False`` and the instruction needed to produce the data.
    """
    pass_col = next(
        (c for c in ("pass", "replicate", "rater", "judge_run") if c in scores.columns),
        None,
    )

    if pass_col is None:
        return Reliability(
            n_items=len(scores.groupby(list(item_keys))),
            n_raters=1,
            icc=float("nan"),
            icc_label="undefined",
            krippendorff_alpha=float("nan"),
            alpha_label="undefined",
            available=False,
            message=(
                "no repeated judgments in this run. Every (meeting, variant) "
                "was scored once. Re-run the tone judge k>=3 times over the "
                "same manifest, tagging each pass, then re-run this check."
            ),
        )

    wide = scores.pivot_table(
        index=list(item_keys), columns=pass_col, values=dimension
    )
    arr = wide.to_numpy(dtype=float)
    n_items, n_raters = arr.shape

    if n_raters < 2:
        return Reliability(
            n_items, n_raters, float("nan"), "undefined",
            float("nan"), "undefined", False,
            f"only one pass present in column '{pass_col}'",
        )

    icc = icc_2_1(arr[~np.isnan(arr).any(axis=1)])
    alpha = krippendorff_alpha(arr)
    return Reliability(
        n_items=n_items,
        n_raters=n_raters,
        icc=icc,
        icc_label=_icc_label(icc),
        krippendorff_alpha=alpha,
        alpha_label=_alpha_label(alpha),
        available=True,
    )
