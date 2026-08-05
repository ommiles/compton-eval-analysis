"""Bootstrap confidence intervals for bounded eval scores.

Why not a t-interval. Three of the four dimensions are proportions on
[0, 1] and the fourth is an ordinal 1-5 judgment. At n = 12 the sampling
distribution of their means is neither normal nor symmetric, and a t-interval
will happily hand back a bound like 1.04 for a quantity that cannot exceed 1.

The bias-corrected and accelerated (BCa) bootstrap fixes both problems it
can fix: it corrects for median bias and for skew that changes with the
parameter value. It still cannot invent information — with 12 meetings the
intervals come out wide, and that width is the finding.

Degenerate samples are common here and handled explicitly: ``v0`` scores
0.000 on ``anomaly_surface_rate`` for every meeting. A bootstrap over a
constant vector has zero variance, the acceleration term divides by zero,
and the honest answer is a degenerate interval rather than a NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    n: int
    method: str
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] ({pct}% CI, n={self.n})"


def bootstrap_mean(
    values: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    bounds: tuple[float, float] | None = None,
    seed: int = 0,
) -> Interval:
    """BCa bootstrap CI for the mean.

    Parameters
    ----------
    bounds:
        Optional (low, high) clamp reflecting the measurement scale. Applied
        to the interval, never to the point estimate — if a bound is doing
        real work, the interval was already up against the edge of what the
        metric can express.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size

    if n == 0:
        return Interval(np.nan, np.nan, np.nan, 0, "empty", confidence)

    point = float(x.mean())

    # A constant vector has no sampling variability to estimate. Say so
    # rather than emitting NaN from a 0/0 acceleration term.
    if n == 1 or np.allclose(x, x[0]):
        return Interval(point, point, point, n, "degenerate", confidence)

    res = stats.bootstrap(
        (x,),
        np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)

    # BCa can fail on tiny or near-degenerate samples; fall back rather than
    # returning NaN silently.
    method = "BCa"
    if not np.isfinite(low) or not np.isfinite(high):
        alpha = (1 - confidence) / 2
        boot = _resample_means(x, n_resamples, seed)
        low, high = np.quantile(boot, [alpha, 1 - alpha])
        low, high = float(low), float(high)
        method = "percentile (BCa failed)"

    if bounds is not None:
        lo_b, hi_b = bounds
        low = max(low, lo_b)
        high = min(high, hi_b)

    return Interval(point, low, high, n, method, confidence)


def bootstrap_paired_delta(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """BCa CI for the paired mean difference ``b - a``.

    Resamples *pairs*, not the two vectors independently. Resampling
    independently would discard the pairing and inflate the interval, which
    is the whole reason the meetings were held fixed across variants.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired vectors must match: {x.shape} vs {y.shape}")

    mask = ~(np.isnan(x) | np.isnan(y))
    d = y[mask] - x[mask]
    return bootstrap_mean(
        d, confidence=confidence, n_resamples=n_resamples, seed=seed
    )


def _resample_means(x: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    return x[idx].mean(axis=1)
