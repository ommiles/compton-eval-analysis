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
from .coding import (
    axial_prompt,
    blind_map,
    build_queue,
    census_queue,
    write_reading_doc,
    check_taxonomy,
    load_open_codes,
    load_taxonomy,
    load_traces,
    open_code_stub,
    prevalence,
    saturation_report,
)
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


def sample(args: argparse.Namespace) -> int:
    """Build a review queue and emit an open-coding worksheet."""
    traces = load_traces(args.run)

    if args.all:
        queue = census_queue(traces)
        print(f"\n  census: all {len(queue)} traces queued (no sampling bias; "
              "prevalence over this set is exact for the run)")
    else:
        queue = build_queue(traces, n=args.n, seed=args.seed)
        counts: dict[str, int] = {}
        for _, why in queue:
            counts[why] = counts.get(why, 0) + 1
        print(f"\n  {len(traces)} traces available, {len(queue)} queued")
        for why in ("failure-driven", "uncertainty", "random"):
            c = counts.get(why, 0)
            print(f"    {why:16} {c:3}  ({c / len(queue):.0%})")
        print(
            "\n  The random slice is the only unbiased read on overall quality.\n"
            "  Prevalence over the whole queue overstates failure by construction.\n"
        )

    key = None
    if args.blind:
        key = blind_map(traces, seed=args.seed)
        key_path = Path(args.out).with_name("key-do-not-open.json")
        key_path.write_text(json.dumps(key, indent=2))
        doc_path = write_reading_doc(traces, key, Path(args.out).with_name("reading-doc.md"))
        print(f"  blinded: variants shuffled behind letters per meeting")
        print(f"  reading doc: {doc_path}")
        print(f"  key (leave sealed until review is done): {key_path}")

    out = open_code_stub(queue, args.out, blind=key)
    print(f"  wrote {out}")
    print(
        "\n  Next: read each trace and fill in `note` and `acceptable`.\n"
        "  One short lowercase note on the FIRST failure. Do not diagnose or\n"
        "  propose fixes. This step is not delegated to a model.\n"
    )
    return 0


def saturation(args: argparse.Namespace) -> int:
    codes = load_open_codes(args.codes)
    payload = json.loads(Path(args.codes).read_text())
    n_queued = len(payload.get("traces", []))
    r = saturation_report(codes, n_queued)

    print(f"\n  coded {r['coded']}/{r['queued']}   problematic: {r['problems']}")
    print(f"  {r['advice']}\n")
    return 0


def axial(args: argparse.Namespace) -> int:
    codes = load_open_codes(args.codes)
    problems = [c for c in codes if not c.acceptable]
    if not problems:
        print("\n  No traces marked unacceptable — nothing to cluster.\n")
        return 0

    print(f"\n  {len(problems)} problematic notes.\n")
    print("  " + "─" * 66)
    print(axial_prompt(codes, args.system))
    print("  " + "─" * 66)
    print(
        "\n  Paste that into a model, then EDIT what comes back. It does not\n"
        "  know how the system works or how you would fix each failure, so it\n"
        "  merges modes that need different fixes. Save the reviewed result as\n"
        "  a taxonomy file with name/definition/examples per mode.\n"
    )
    return 0


def prevalence_cmd(args: argparse.Namespace) -> int:
    modes = load_taxonomy(args.taxonomy)
    warnings = check_taxonomy(modes)
    if warnings:
        print()
        for w in warnings:
            print(f"  ! {w}")

    payload = json.loads(Path(args.labels).read_text())
    labels = payload["labels"] if "labels" in payload else payload
    sampled_as = payload.get("sampled_as")

    rows = prevalence(labels, modes, sampled_as=sampled_as)
    print(f"\n  {len(labels)} labeled traces, {len(modes)} failure modes\n")
    for r in rows:
        line = (f"  {r['mode'][:34]:34} {r['rate']:.2f} "
                f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]  ({r['count']}/{r['n']})")
        if "unbiased_rate" in r:
            line += (f"   random-only: {r['unbiased_rate']:.2f} "
                     f"[{r['unbiased_ci_low']:.2f}, {r['unbiased_ci_high']:.2f}]"
                     f" (n={r['unbiased_n']})")
        print(line)
    print()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
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

    q = sub.add_parser("sample", help="build a blended review queue for open coding")
    q.add_argument("run", help="eval run directory containing variant artifacts")
    q.add_argument("--n", type=int, default=40)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", default="open-codes.json")
    q.add_argument("--all", action="store_true",
                   help="queue every trace (census) instead of sampling")
    q.add_argument("--blind", action="store_true",
                   help="hide variants behind shuffled letters; writes key + reading doc")
    q.set_defaults(func=sample)

    sat = sub.add_parser("saturation", help="has open coding gone far enough?")
    sat.add_argument("codes", help="filled-in open-coding worksheet")
    sat.set_defaults(func=saturation)

    ax = sub.add_parser("axial", help="emit the clustering prompt for your notes")
    ax.add_argument("codes", help="filled-in open-coding worksheet")
    ax.add_argument("--system", default="an LLM pipeline that summarizes city council meetings")
    ax.set_defaults(func=axial)

    pv = sub.add_parser("prevalence", help="failure-mode rates with intervals")
    pv.add_argument("labels", help="JSON of {trace_id: {mode: 0|1}}")
    pv.add_argument("--taxonomy", required=True)
    pv.add_argument("--out", default=None)
    pv.set_defaults(func=prevalence_cmd)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
