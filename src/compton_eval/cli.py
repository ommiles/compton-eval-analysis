"""Command line entry point.

    python -m compton_eval analyze <run-dir> --out report/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import json

import numpy as np

from .align import corrected_success_rate, split_labeled
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


def align(args: argparse.Namespace) -> int:
    """Validate a judge against human labels, then correct a batch estimate."""
    payload = json.loads(Path(args.labels).read_text())
    rows = payload["test_set"]
    human = np.array([int(r["human"]) for r in rows])
    judge = np.array([int(r["judge"]) for r in rows])

    batch = payload.get("unlabeled", {})
    m = int(args.n_unlabeled or batch.get("n", 0))
    k = int(args.n_judged_pass if args.n_judged_pass is not None
            else batch.get("judged_pass", 0))

    mode = payload.get("failure_mode", Path(args.labels).stem)
    print(f"\nFailure mode: {mode}")
    print(f"  test set: {len(rows)} human-labeled traces\n")

    result = corrected_success_rate(
        human, judge, n_unlabeled=m, n_judged_pass=k,
        n_resamples=args.resamples,
        include_batch_uncertainty=args.batch_uncertainty,
        seed=args.seed,
    )
    a = result.alignment

    print("── Judge accuracy (frozen test set) " + "─" * 39)
    print(f"  {a}")
    for w in a.warnings:
        print(f"  ! {w}")
    print()

    print("── True rate over the unlabeled batch " + "─" * 37)
    print(f"  raw judge pass rate : {result.observed:.4f}  ({k}/{m})")
    if np.isfinite(result.corrected):
        print(f"  bias-corrected      : {result.corrected:.4f}  "
              f"[{result.low:.4f}, {result.high:.4f}]  "
              f"({int(result.confidence*100)}% CI)")
        print(f"  correction moved it : {result.shift:+.4f}")
    else:
        print(f"  bias-corrected      : unavailable — {result.note}")
    if result.note and np.isfinite(result.corrected):
        print(f"  note: {result.note}")
    print()

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "alignment.json").write_text(json.dumps({
            "failure_mode": mode,
            "tpr": a.tpr, "tnr": a.tnr,
            "informedness": a.informedness,
            "n_pass": a.n_pass, "n_fail": a.n_fail,
            "observed": result.observed,
            "corrected": result.corrected,
            "ci_low": result.low, "ci_high": result.high,
            "n_unlabeled": m, "n_judged_pass": k,
            "warnings": a.warnings,
        }, indent=2))
        print(f"  wrote {out / 'alignment.json'}\n")
    return 0


def split(args: argparse.Namespace) -> int:
    """Emit stratified train/dev/test indices for a labeled pool."""
    labels = json.loads(Path(args.labels).read_text())
    if isinstance(labels, dict):
        labels = labels.get("labels") or [int(r["human"]) for r in labels["test_set"]]
    y = np.array([int(v) for v in labels])

    tr, dv, te = split_labeled(y.size, labels=y, train=args.train, dev=args.dev,
                              seed=args.seed)
    print(f"\n  train {len(tr):4}  ({y[tr].sum()} pass / {(~y[tr].astype(bool)).sum()} fail)"
          "   few-shot candidates only")
    print(f"  dev   {len(dv):4}  ({y[dv].sum()} pass / {(~y[dv].astype(bool)).sum()} fail)"
          "   refine the prompt here")
    print(f"  test  {len(te):4}  ({y[te].sum()} pass / {(~y[te].astype(bool)).sum()} fail)"
          "   read once, after freezing\n")
    for name, part in (("dev", dv), ("test", te)):
        npass = int(y[part].sum()); nfail = len(part) - npass
        if npass < 30 or nfail < 30:
            print(f"  ! {name} has {npass} pass / {nfail} fail; want >=30 of each\n")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"train": tr.tolist(), "dev": dv.tolist(), "test": te.tolist()}, indent=2))
        print(f"  wrote {args.out}\n")
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

    g = sub.add_parser("align", help="validate a judge and correct a batch estimate")
    g.add_argument("labels", help="JSON with test_set [{human, judge}] and unlabeled counts")
    g.add_argument("--n-unlabeled", type=int, default=None)
    g.add_argument("--n-judged-pass", type=int, default=None)
    g.add_argument("--resamples", type=int, default=20_000)
    g.add_argument("--batch-uncertainty", action="store_true",
                   help="also resample the unlabeled batch (wider, honest at small m)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=None)
    g.set_defaults(func=align)

    s_ = sub.add_parser("split", help="stratified train/dev/test split of labeled traces")
    s_.add_argument("labels", help="JSON list of 0/1 human labels, or an align-style file")
    s_.add_argument("--train", type=float, default=0.15)
    s_.add_argument("--dev", type=float, default=0.425)
    s_.add_argument("--seed", type=int, default=0)
    s_.add_argument("--out", default=None)
    s_.set_defaults(func=split)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
