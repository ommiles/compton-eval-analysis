"""Charts for eval analysis.

Design rules, applied consistently:

- Show the interval, not just the point. A bar chart of means is the exact
  visual that made "V2 beats V1" look obvious in the first place.
- Draw the individual meetings. At n = 12 the reader can and should see
  every observation; aggregate marks hide the one meeting driving a delta.
- Mark the zero line on difference plots. An interval crossing zero is the
  headline, and it should be readable without consulting a table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .bootstrap import bootstrap_mean  # noqa: E402
from .load import DIMENSIONS, EvalRun  # noqa: E402
from .power import PowerCurve  # noqa: E402

# Brand-neutral qualitative palette, colour-blind safe, legible on white.
PALETTE = ["#4C6EF5", "#F76707", "#0CA678", "#AE3EC9", "#F59F00"]
GRID = "#DEE2E6"
INK = "#212529"
MUTED = "#868E96"


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def plot_dimension_intervals(
    run: EvalRun, out: str | Path, *, n_resamples: int = 4_000
) -> Path:
    """Per-dimension variant means with bootstrap CIs and raw observations."""
    dims = [d for d in DIMENSIONS if d in run.scores.columns]
    fig, axes = plt.subplots(1, len(dims), figsize=(4.0 * len(dims), 4.4))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        wide = run.dimension(dim)
        variants = [v for v in run.variants if v in wide.columns]
        lo, hi = DIMENSIONS[dim]

        for i, v in enumerate(variants):
            vals = wide[v].to_numpy(dtype=float)
            ci = bootstrap_mean(
                vals, bounds=(lo, hi), n_resamples=n_resamples, seed=i
            )
            colour = PALETTE[i % len(PALETTE)]

            jitter = (np.random.default_rng(i).random(vals.size) - 0.5) * 0.16
            ax.scatter(
                np.full(vals.size, i) + jitter, vals,
                s=26, color=colour, alpha=0.35, edgecolors="none", zorder=2,
            )
            ax.errorbar(
                i, ci.point,
                yerr=[[ci.point - ci.low], [ci.high - ci.point]],
                fmt="o", color=colour, markersize=9, capsize=6,
                linewidth=2.2, zorder=3,
            )

        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([v.upper() for v in variants], fontsize=10, color=INK)
        ax.set_title(dim.replace("_", " "), fontsize=11, color=INK, pad=10)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        _style(ax)

    fig.suptitle(
        f"Variant means with 95% bootstrap CIs  ·  run {run.run_id[:10]}  ·  "
        f"dots are individual meetings",
        fontsize=11, color=INK, y=1.02,
    )
    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_paired_deltas(comparisons: pd.DataFrame, out: str | Path) -> Path:
    """Forest plot of paired differences. Intervals crossing zero are muted."""
    df = comparisons.dropna(subset=["delta"]).copy()
    if df.empty:
        raise ValueError("no comparisons to plot")

    df["label"] = (
        df["dimension"].str.replace("_", " ")
        + "   "
        + df["variant_a"].str.upper()
        + " → "
        + df["variant_b"].str.upper()
    )
    df = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(df) + 2.4))

    # Three states, because "CI excludes zero" and "survives multiplicity
    # correction" are different claims and this run contains rows where they
    # disagree. Colouring by CI alone would contradict the comparison table.
    SURVIVES, UNCORRECTED, NULL = PALETTE[2], PALETTE[4], MUTED

    for i, row in df.iterrows():
        crosses = row["delta_lo"] <= 0 <= row["delta_hi"]
        if row["significant"]:
            colour = SURVIVES
        elif not crosses:
            colour = UNCORRECTED
        else:
            colour = NULL
        ax.plot(
            [row["delta_lo"], row["delta_hi"]], [i, i],
            color=colour, linewidth=2.4, solid_capstyle="round", zorder=2,
        )
        ax.scatter(
            row["delta"], i, s=52, color=colour, zorder=3,
            edgecolors="white", linewidths=1.0,
        )

    ax.axvline(0, color=INK, linewidth=1.1, linestyle="--", alpha=0.55, zorder=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=9, color=INK)
    ax.set_xlabel("paired mean difference (95% bootstrap CI)", fontsize=10, color=INK)
    ax.set_title(
        "Which variant differences are real", fontsize=11.5, color=INK, pad=14
    )

    handles = [
        plt.Line2D([], [], color=SURVIVES, lw=2.6, marker="o", markersize=7,
                   label="survives Holm correction"),
        plt.Line2D([], [], color=UNCORRECTED, lw=2.6, marker="o", markersize=7,
                   label="CI excludes 0, but not after correcting for 12 tests"),
        plt.Line2D([], [], color=NULL, lw=2.6, marker="o", markersize=7,
                   label="indistinguishable"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5,
              loc="lower right", handlelength=1.8)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
    ax.yaxis.grid(False)

    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_power_curves(
    curves: list[PowerCurve], out: str | Path, *, target: float = 0.80
) -> Path:
    """Power vs sample size, with the current run size marked."""
    if not curves:
        raise ValueError("no power curves to plot")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i, c in enumerate(curves):
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(
            c.sample_sizes, c.power,
            marker="o", markersize=4.5, linewidth=2.0, color=colour,
            label=f"{c.dimension.replace('_', ' ')}  (Δ={c.effect:g})",
        )

    ax.axhline(target, color=INK, linestyle="--", linewidth=1.1, alpha=0.55)
    ax.text(
        ax.get_xlim()[1], target + 0.015, f"{int(target * 100)}% power",
        ha="right", fontsize=9, color=MUTED,
    )
    ax.axvline(12, color=PALETTE[1], linestyle=":", linewidth=1.6, alpha=0.8)
    ax.text(12.6, 0.04, "current run (n=12)", fontsize=9, color=PALETTE[1])

    ax.set_xlabel("meetings per variant", fontsize=10, color=INK)
    ax.set_ylabel("probability of detecting the effect", fontsize=10, color=INK)
    ax.set_ylim(0, 1.06)
    ax.set_title(
        "Meetings needed to detect a minimum effect worth caring about",
        fontsize=11.5, color=INK, pad=22,
    )
    # The curves are genuinely non-monotonic below ~n=16. That is the
    # signed-rank statistic being discrete, not simulation noise — the set of
    # attainable p-values is coarse at small n, so adding one meeting can
    # move the critical threshold the wrong way. Say so, or it reads as a bug.
    ax.annotate(
        "non-monotonic below n≈16: the signed-rank p-value grid is discrete at small n",
        xy=(0.5, 1.015), xycoords="axes fraction", ha="center",
        fontsize=8.5, color=MUTED, style="italic",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)

    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p
