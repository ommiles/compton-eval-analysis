"""Command line entry point.

    python -m compton_eval analyze <run-dir> --out report/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .compare import compare_run
from .load import load_run
from .plots import plot_dimension_intervals, plot_paired_deltas, plot_power_curves
from .power import power_for_run, power_table
from .reliability import check_repeatability


def _fmt_comparisons(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (no comparisons)\n"
    lines = []
    for _, r in df.iterrows():
        verdict = "REAL " if r["significant"] else "noise"
        ci = f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
        p = "  n/a" if pd.isna(r["p_holm"]) else f"{r['p_holm']:.4f}"
        lines.append(
            f"  {verdict}  {r['dimension']:22} {r['variant_a'].upper()}→"
            f"{r['variant_b'].upper()}  Δ={r['delta']:+.3f} {ci:>18}  "
            f"p_holm={p}  effect={r['effect']}"
        )
        if r["note"]:
            lines.append(f"           note: {r['note']}")
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nRun {run.run_id}")
    print(f"  meetings: {len(run.meetings)}   variants: {', '.join(run.variants)}")
    for dim in run.scores.columns:
        if dim in ("meeting_id", "variant"):
            continue
        pairs, total = run.pairable(dim)
        flag = "" if pairs == total else f"   ({total - pairs} incomplete, dropped)"
        print(f"  {dim:24} {pairs} complete pairs{flag}")

    print("\n── Variant comparisons " + "─" * 52)
    comparisons = compare_run(run, alpha=args.alpha, n_resamples=args.resamples)
    print(_fmt_comparisons(comparisons))
    comparisons.to_csv(out / "comparisons.csv", index=False)

    n_real = int(comparisons["significant"].sum()) if not comparisons.empty else 0
    n_total = len(comparisons)
    print(f"  {n_real} of {n_total} comparisons survive Holm correction at α={args.alpha}\n")

    print("── Judge reliability " + "─" * 54)
    rel = check_repeatability(run.scores)
    print(f"  {rel}\n")

    print("── Power " + "─" * 66)
    curves = power_for_run(
        run, args.baseline, args.candidate,
        alpha=args.alpha, n_simulations=args.simulations,
    )
    if curves:
        table = power_table(curves)
        for _, r in table.iterrows():
            need = "not reached by n=100" if pd.isna(r["meetings_needed"]) else f"{int(r['meetings_needed'])} meetings"
            print(
                f"  {r['dimension']:24} min Δ={r['min_effect']:<5g} "
                f"→ {need:<22} (power at n=12: {r['power_at_12']:.0%})"
            )
        table.to_csv(out / "power.csv", index=False)
    else:
        print(f"  no usable power curves for {args.baseline}→{args.candidate}")

    for dim, why in getattr(power_for_run, "skipped", []):
        print(f"  {dim:24} skipped — {why}")
    print()

    if not args.no_plots:
        print("── Charts " + "─" * 65)
        p1 = plot_dimension_intervals(run, out / "variant-intervals.png")
        print(f"  {p1}")
        p2 = plot_paired_deltas(comparisons, out / "paired-deltas.png")
        print(f"  {p2}")
        if curves:
            p3 = plot_power_curves(curves, out / "power-curves.png")
            print(f"  {p3}")
        print()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compton-eval",
        description="Statistical analysis for LLM summarization eval runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze one eval run")
    a.add_argument("run", help="run directory or scores.json")
    a.add_argument("--out", default="report", help="output directory")
    a.add_argument("--alpha", type=float, default=0.05)
    a.add_argument("--resamples", type=int, default=10_000)
    a.add_argument("--simulations", type=int, default=2_000)
    a.add_argument("--baseline", default="v1", help="power: baseline variant")
    a.add_argument("--candidate", default="v2", help="power: candidate variant")
    a.add_argument("--no-plots", action="store_true")
    a.set_defaults(func=analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
