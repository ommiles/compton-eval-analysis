"""How many meetings does a run actually need?

The harness currently scores twelve meetings because twelve felt like
enough. This module replaces that intuition with a number.

Analytic power formulas for the Wilcoxon signed-rank test assume shapes the
data does not have (bounded proportions, heavy ties, floor effects — ``v0``
scores exactly 0.000 on anomaly surfacing for every meeting). So power is
estimated by simulation instead: resample paired differences from the
observed distribution, shift them to a specified effect size, and count how
often the test rejects.

A deliberate omission: there is no post-hoc power calculation here. Power
computed from an observed effect is a monotone transform of the p-value and
tells you nothing the p-value did not already say. The useful question is
prospective — *given an effect I would care about, how many meetings do I
need?* — and that is what this computes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .load import DIMENSIONS, EvalRun


@dataclass(frozen=True)
class PowerCurve:
    """Estimated power across sample sizes for one effect size."""

    dimension: str
    effect: float
    sample_sizes: list[int]
    power: list[float]
    alpha: float
    n_simulations: int

    def n_for(self, target: float = 0.80) -> int | None:
        """Smallest simulated n reaching ``target`` power, if any."""
        for n, p in zip(self.sample_sizes, self.power):
            if p >= target:
                return n
        return None


def simulate_power(
    differences: np.ndarray | list[float],
    *,
    effect: float,
    sample_sizes: list[int] | None = None,
    alpha: float = 0.05,
    n_simulations: int = 2_000,
    seed: int = 0,
    method: str = "asymptotic",
) -> tuple[list[int], list[float]]:
    """Empirical power for a paired signed-rank test.

    ``differences`` supplies the *shape* of the noise — its spread and skew
    are resampled — while ``effect`` sets the true mean difference being
    tested for. Centring the observed differences first means the simulation
    tests the effect you asked about, not the one that happened to occur.

    ``method`` is passed through to :func:`scipy.stats.wilcoxon` and defaults
    to the normal approximation so that every point on the curve is produced
    the same way. See the note in the body before changing it.
    """
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        raise ValueError("need at least 2 paired differences to model noise")

    if sample_sizes is None:
        sample_sizes = [6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 75, 100]

    centred = d - d.mean()
    rng = np.random.default_rng(seed)

    powers: list[float] = []
    for n in sample_sizes:
        idx = rng.integers(0, centred.size, size=(n_simulations, n))
        samples = centred[idx] + effect

        # One vectorised call over axis=1 rather than n_simulations separate
        # calls. Two details matter here.
        #
        # method="asymptotic" is pinned rather than left on "auto". scipy's
        # auto-selection uses the exact signed-rank distribution at small n
        # and the normal approximation above it, which would put a
        # discontinuity in the middle of the power curve — precisely the
        # region being read. Pinning one method keeps the curve comparable
        # across sample sizes, at the cost of being mildly conservative
        # below n = 20 where the exact test is slightly more powerful.
        # (It is also ~10,000x faster, because the exact path does not
        # vectorise over the simulation axis.)
        with np.errstate(invalid="ignore"):
            p = stats.wilcoxon(
                samples,
                axis=1,
                zero_method="wilcox",
                alternative="two-sided",
                method=method,
            ).pvalue
        p = np.asarray(p, dtype=float)

        # An all-zero difference vector yields no usable statistic; count it
        # as a non-rejection rather than dropping it, which would silently
        # shrink the denominator and overstate power.
        rejects = int(np.count_nonzero(np.isfinite(p) & (p < alpha)))
        powers.append(rejects / n_simulations)

    return list(sample_sizes), powers


def power_for_run(
    run: EvalRun,
    variant_a: str,
    variant_b: str,
    *,
    effects: dict[str, float] | None = None,
    alpha: float = 0.05,
    n_simulations: int = 2_000,
    seed: int = 0,
) -> list[PowerCurve]:
    """Power curves for one variant pair across all dimensions.

    ``effects`` sets the minimum difference worth detecting per dimension.
    The defaults are deliberately modest — a 5-point move on a bounded rate,
    a quarter-point on the 1-5 tone scale — because the expensive question
    is not "can I detect a huge win" but "can I detect a small regression".
    """
    if effects is None:
        effects = {
            "top_dollar_recall": 0.10,
            "anomaly_surface_rate": 0.10,
            "tone_score": 0.25,
            "factual_precision": 0.05,
        }

    curves: list[PowerCurve] = []
    skipped: list[tuple[str, str]] = []
    for dim in (d for d in DIMENSIONS if d in run.scores.columns):
        wide = run.dimension(dim)
        if wide.empty or variant_a not in wide or variant_b not in wide:
            continue

        diffs = (wide[variant_b] - wide[variant_a]).to_numpy(dtype=float)
        if np.allclose(diffs, diffs[0] if diffs.size else 0):
            # No variance to resample — a constant difference carries no
            # information about how noisy a future run would be. Record the
            # omission rather than dropping the dimension silently; a missing
            # line on the chart otherwise looks like an oversight.
            skipped.append(
                (dim, "identical on every pair" if np.allclose(diffs, 0)
                 else "constant difference on every pair")
            )
            continue

        effect = effects.get(dim, 0.10)
        sizes, powers = simulate_power(
            diffs,
            effect=effect,
            alpha=alpha,
            n_simulations=n_simulations,
            seed=seed,
        )
        curves.append(
            PowerCurve(dim, effect, sizes, powers, alpha, n_simulations)
        )

    # Attach for callers that want to report what was left out.
    power_for_run.skipped = skipped  # type: ignore[attr-defined]
    return curves


def power_table(curves: list[PowerCurve], target: float = 0.80) -> pd.DataFrame:
    """Summarise curves: meetings needed per dimension."""
    rows = []
    for c in curves:
        n = c.n_for(target)
        rows.append(
            {
                "dimension": c.dimension,
                "min_effect": c.effect,
                "target_power": target,
                "meetings_needed": n if n is not None else pd.NA,
                "power_at_12": next(
                    (p for s, p in zip(c.sample_sizes, c.power) if s == 12),
                    float("nan"),
                ),
            }
        )
    return pd.DataFrame(rows)
