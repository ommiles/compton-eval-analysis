"""Error analysis: open coding, axial coding, and prevalence.

This is the step that comes *before* everything else in this package. The
four dimensions the harness scores were designed top-down, from what seemed
important. Grounded theory says the taxonomy should come the other way: read
traces, note what actually went wrong in plain language, then cluster those
notes into failure modes.

Following Husain & Shankar, *Evals for AI Engineers*, ch. 3 and ch. 10.

The loop:

1. **Sample** a review queue. Blended 50% failure-driven, 30% uncertainty,
   20% random (ch. 10). The random fifth is the part people skip, and it is
   the only unbiased read on overall quality.
2. **Open-code** each trace by hand. One short lowercase note on the *first*
   failure, plus a binary acceptable/unacceptable call. Not an LLM's job.
3. **Axial-code** the notes into failure modes. An LLM may help here, because
   the task is organizing observations you already made rather than deciding
   what counts as a failure.
4. **Label** every trace against the taxonomy, 1 or 0 per mode, and compute
   prevalence.

Two rules the module enforces rather than suggests. Open coding is refused
programmatically: :func:`open_code_stub` will not accept machine-generated
notes, because the whole value of the step is your judgment about what a
failure is. And judgments are binary — ch. 3 is explicit that Likert scales
without a detailed rubric produce lower inter-annotator agreement and higher
subjective variance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bootstrap import Interval, bootstrap_mean

#: Ch. 10's suggested blend for a review queue.
DEFAULT_BLEND = {"failure": 0.50, "uncertainty": 0.30, "random": 0.20}

#: Ch. 3: keep open coding until this many problematic traces are in hand
#: *and* new failure modes have stopped appearing.
SATURATION_MIN_PROBLEMS = 20

#: Ch. 3's rough target before a taxonomy is worth trusting.
TAXONOMY_MIN_MODES, TAXONOMY_MAX_MODES = 5, 8


@dataclass(frozen=True)
class Trace:
    """One reviewable artifact: a meeting summarized under one variant."""

    trace_id: str
    meeting_id: int
    variant: str
    text: str
    scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def deterministic(self) -> dict[str, float]:
        from .load import DETERMINISTIC

        return {k: v for k, v in self.scores.items() if k in DETERMINISTIC}


@dataclass
class OpenCode:
    """A hand-written observation about the first failure in one trace."""

    trace_id: str
    note: str
    acceptable: bool
    reviewer: str = ""

    def __post_init__(self) -> None:
        if not self.note or not self.note.strip():
            raise ValueError(
                f"{self.trace_id}: empty note. If nothing went wrong, say so "
                "('nothing wrong, summary matches the packet') rather than "
                "leaving it blank — a blank is indistinguishable from a skip."
            )


@dataclass
class FailureMode:
    """One binary failure mode from axial coding."""

    name: str
    definition: str
    examples: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.definition.strip():
            raise ValueError(
                f"{self.name}: needs a one-line definition. A mode without one "
                "cannot be applied consistently, by you or by a judge."
            )


# --------------------------------------------------------------- loading

def load_traces(run_dir: str | Path) -> list[Trace]:
    """Read variant artifacts and their scores out of an eval run directory."""
    d = Path(run_dir)
    scores_path = d / "scores.json"
    by_key: dict[tuple[int, str], dict] = {}
    if scores_path.exists():
        for row in json.loads(scores_path.read_text()).get("scores", []):
            by_key[(int(row["meeting_id"]), str(row["variant"]))] = row

    traces: list[Trace] = []
    for f in sorted(d.glob("*.json")):
        if f.name == "scores.json":
            continue
        try:
            a = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "meeting_id" not in a or "variant" not in a:
            continue

        mid, var = int(a["meeting_id"]), str(a["variant"])
        row = by_key.get((mid, var), {})
        scores = {
            k: float(v)
            for k, v in row.items()
            if isinstance(v, (int, float)) and k != "meeting_id"
        }
        traces.append(
            Trace(
                trace_id=f"{mid}-{var}",
                meeting_id=mid,
                variant=var,
                text=str(a.get("summary_en") or ""),
                scores=scores,
                meta={
                    "model": a.get("model"),
                    "prompt_version": a.get("prompt_version"),
                    "prompt_hash_summary": a.get("prompt_hash_summary"),
                    "n_chapters": len(a.get("chapters") or []),
                    "transcript_length": a.get("input_transcript_length"),
                    "anomaly_count": a.get("input_anomaly_count"),
                    "has_spanish": bool(a.get("summary_es")),
                    "error": a.get("error"),
                    "anomalies_flagged": row.get("anomalies_flagged"),
                    "tone_rationale": row.get("tone_rationale"),
                    "factual_errors": row.get("factual_errors"),
                },
            )
        )
    if not traces:
        raise FileNotFoundError(f"no variant artifacts found in {d}")
    return traces


# -------------------------------------------------------------- sampling

def _failure_score(t: Trace) -> float:
    """How badly a trace looks flagged by signals we already have.

    Higher means more suspicious. Only deterministic scores count here — using
    the LLM judge to pick what the human reviews would fold the judge's blind
    spots into the taxonomy meant to reveal them.
    """
    det = t.deterministic
    if t.meta.get("error"):
        return 1.0
    if not det:
        return 0.0
    return float(np.mean([1.0 - v for v in det.values()]))


def _uncertainty_score(t: Trace) -> float:
    """How much the available signals disagree with each other.

    Ch. 10: a trace that fails some criteria and passes others is borderline,
    and borderline traces are the informative ones to read.
    """
    det = list(t.deterministic.values())
    if len(det) < 2:
        return 0.0
    # Spread is maximal at 0.5, which is where the signals conflict most.
    return float(np.std(det) * 2.0)


def build_queue(
    traces: list[Trace],
    *,
    n: int = 40,
    blend: dict[str, float] | None = None,
    seed: int = 0,
) -> list[tuple[Trace, str]]:
    """Build a review queue, returning (trace, why_it_was_picked) pairs.

    The provenance tag matters: prevalence computed over a failure-driven
    queue is not the system's failure rate, and labeling the queue without
    recording how it was built invites exactly that mistake.
    """
    blend = blend or DEFAULT_BLEND
    if abs(sum(blend.values()) - 1.0) > 1e-6:
        raise ValueError(f"blend must sum to 1.0, got {sum(blend.values())}")
    n = min(n, len(traces))

    rng = np.random.default_rng(seed)
    picked: dict[str, str] = {}
    out: list[tuple[Trace, str]] = []

    def take(candidates: list[Trace], k: int, why: str) -> None:
        for t in candidates:
            if len(out) >= n or k <= 0:
                return
            if t.trace_id in picked:
                continue
            picked[t.trace_id] = why
            out.append((t, why))
            k -= 1

    by_failure = sorted(traces, key=_failure_score, reverse=True)
    take([t for t in by_failure if _failure_score(t) > 0],
         int(round(blend["failure"] * n)), "failure-driven")

    by_uncertainty = sorted(traces, key=_uncertainty_score, reverse=True)
    take([t for t in by_uncertainty if _uncertainty_score(t) > 0],
         int(round(blend["uncertainty"] * n)), "uncertainty")

    take(list(rng.permutation(np.array(traces, dtype=object))), n - len(out), "random")
    # Any shortfall in the first two strata spills into random, which keeps
    # the queue at n rather than silently returning fewer traces.
    return out


# ------------------------------------------------------------- blinding

def census_queue(traces: list[Trace]) -> list[tuple[Trace, str]]:
    """Every trace, tagged "census". With a population small enough to read
    in full, the blended queue is pointless — full coverage has no sampling
    bias, and prevalence over it is exact for the run."""
    return [(t, "census") for t in traces]


def blind_map(traces: list[Trace], *, seed: int = 0) -> dict[str, str]:
    """{display_id: trace_id}, variants shuffled behind letters per meeting.

    A reviewer who can see "v0" on a trace knows it came from the weak
    baseline prompt and will find what they expect to find. Letters remove
    the label; the shuffle removes position as a tell.
    """
    rng = np.random.default_rng(seed)
    by_meeting: dict[int, list[Trace]] = {}
    for t in traces:
        by_meeting.setdefault(t.meeting_id, []).append(t)

    key: dict[str, str] = {}
    for mid in sorted(by_meeting):
        group = sorted(by_meeting[mid], key=lambda t: t.variant)
        order = rng.permutation(len(group))
        for letter, idx in zip("ABCDEFGH", order):
            key[f"{mid}-{letter}"] = group[idx].trace_id
    return key


def unblind_codes(codes: list[OpenCode], key: dict[str, str]) -> list[OpenCode]:
    """Swap display ids back to trace ids after review is finished."""
    missing = [c.trace_id for c in codes if c.trace_id not in key]
    if missing:
        raise ValueError(f"ids not in the blind key: {missing[:5]}")
    return [
        OpenCode(key[c.trace_id], c.note, c.acceptable, c.reviewer)
        for c in codes
    ]


def write_reading_doc(
    traces: list[Trace],
    key: dict[str, str],
    out: str | Path,
) -> Path:
    """Markdown for the human to read: input facts plus blinded summaries.

    Includes what the meeting *contained* (transcript length, anomaly types
    the deterministic detectors flagged) and excludes every judgment the
    pipeline has already formed (scores, tone rationale, factual-error
    lists). Facts enable review; verdicts pre-empt it.
    """
    by_id = {t.trace_id: t for t in traces}
    by_meeting: dict[int, list[tuple[str, Trace]]] = {}
    for display, tid in key.items():
        mid = int(display.split("-")[0])
        by_meeting.setdefault(mid, []).append((display, by_id[tid]))

    lines = [
        "# Open-coding reading doc — round 1",
        "",
        "Read each summary as a Compton resident would. For each one, note in",
        "the worksheet the FIRST thing that goes wrong, in one short lowercase",
        "phrase, and mark it acceptable or not. Every row gets a note — write",
        '"nothing wrong" if nothing is. Do not diagnose causes or propose',
        "fixes. Variants are blinded; do not open the key file until done.",
        "",
    ]
    for mid in sorted(by_meeting):
        entries = sorted(by_meeting[mid])
        first = entries[0][1]
        anomalies = first.meta.get("anomalies_flagged") or []
        lines += [
            f"## Meeting {mid}",
            "",
            f"Transcript: {first.meta.get('transcript_length'):,} chars · "
            f"{first.meta.get('n_chapters')} chapters · "
            f"structured items: {first.meta.get('anomaly_count')} anomalies "
            f"flagged by detectors"
            + (f" ({', '.join(anomalies)})" if anomalies else ""),
            "",
        ]
        for display, t in entries:
            lines += [f"### {display}", "", t.text.strip() or "(empty summary)", ""]

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines))
    return p


# ----------------------------------------------------------- open coding

def open_code_stub(
    queue: list[tuple[Trace, str]],
    out: str | Path,
    *,
    blind: dict[str, str] | None = None,
) -> Path:
    """Write a worksheet for a human to fill in. Deliberately not automated.

    Ch. 3 recommends an LLM for axial coding but explicitly not for open
    coding, because that step needs your judgment and taste about what counts
    as a failure. This emits the empty structure and stops.
    """
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    if blind:
        reverse = {tid: disp for disp, tid in blind.items()}
        missing = [t.trace_id for t, _ in queue if t.trace_id not in reverse]
        if missing:
            raise ValueError(f"traces missing from blind map: {missing[:5]}")
        # Variant and scores are exactly the fields blinding exists to hide.
        rows = [
            {
                "trace_id": reverse[t.trace_id],
                "sampled_as": why,
                "meeting_id": t.meeting_id,
                "note": "",
                "acceptable": None,
            }
            for t, why in sorted(queue, key=lambda q: reverse[q[0].trace_id])
        ]
    else:
        rows = [
            {
                "trace_id": t.trace_id,
                "sampled_as": why,
                "meeting_id": t.meeting_id,
                "variant": t.variant,
                "scores": t.deterministic,
                "note": "",
                "acceptable": None,
            }
            for t, why in queue
        ]
    p.write_text(json.dumps({
        "_instructions": (
            "Read each trace and write ONE short lowercase note about the FIRST "
            "thing that went wrong, from the reader's perspective. Do not "
            "diagnose causes or propose fixes. Set acceptable to true or false; "
            "pick a side even when it feels borderline. Every row gets a note — "
            "write 'nothing wrong' if nothing is; leave it empty only if you "
            "skipped the trace. What counts as wrong: you are the instrument, "
            "not a rubric. Note whatever makes you wince as the end reader. "
            "Four questions catch most of it — misled (stated but not so, or "
            "attributed to no one)? shortchanged (the big number, the vote, the "
            "dissent missing)? spun (reads like the subject wrote it about "
            "itself)? lost (can't follow what happened)? For acceptable: would "
            "you ship this exact output tonight? Rough notes are fine; if "
            "something bugs you that you can't name, write that down too — "
            "those notes are what the current metrics can't see."
        ),
        "traces": rows,
    }, indent=2))
    return p


def load_open_codes(path: str | Path) -> list[OpenCode]:
    """Read a filled-in worksheet, skipping untouched rows."""
    payload = json.loads(Path(path).read_text())
    rows = payload.get("traces", payload if isinstance(payload, list) else [])
    codes = []
    for r in rows:
        if not (r.get("note") or "").strip():
            continue
        if r.get("acceptable") is None:
            raise ValueError(
                f"{r.get('trace_id')}: has a note but no acceptable judgment. "
                "Binary is the point — pick a side."
            )
        codes.append(OpenCode(
            trace_id=str(r["trace_id"]),
            note=str(r["note"]).strip(),
            acceptable=bool(r["acceptable"]),
            reviewer=str(r.get("reviewer", "")),
        ))
    return codes


def saturation_report(codes: list[OpenCode], n_queued: int) -> dict[str, object]:
    """Whether open coding has gone far enough to move on."""
    problems = [c for c in codes if not c.acceptable]
    reached = len(problems) >= SATURATION_MIN_PROBLEMS
    return {
        "coded": len(codes),
        "queued": n_queued,
        "problems": len(problems),
        "problems_needed": max(0, SATURATION_MIN_PROBLEMS - len(problems)),
        "saturation_likely": reached,
        "advice": (
            "Enough problematic traces to axial-code. Saturation also needs new "
            "failure modes to have stopped appearing — only you can judge that."
            if reached else
            f"Keep reading. {SATURATION_MIN_PROBLEMS - len(problems)} more "
            "problematic traces before the taxonomy is worth building."
        ),
    }


# ---------------------------------------------------------- axial coding

AXIAL_PROMPT = """\
Below is a list of open-ended annotations describing failures in {system}.
Please group them into a small set of coherent failure categories, where each
category captures similar types of mistakes. Each group should have a
one-line definition. Do not invent new failure types; only cluster based on
what is present in the notes.

{notes}
"""


def axial_prompt(codes: list[OpenCode], system: str) -> str:
    """The clustering prompt from ch. 3, filled with your notes.

    Returned as text rather than executed. The book is clear that LLM
    groupings "should not be accepted blindly" — the model does not know how
    the system works or how you would fix each failure, so it merges things
    that should stay split. Reviewing the output is the actual work.
    """
    problems = [c for c in codes if not c.acceptable]
    notes = "\n".join(f"- {c.note}" for c in problems)
    return AXIAL_PROMPT.format(system=system, notes=notes)


def load_taxonomy(path: str | Path) -> list[FailureMode]:
    payload = json.loads(Path(path).read_text())
    modes = payload.get("failure_modes", payload if isinstance(payload, list) else [])
    return [
        FailureMode(
            name=str(m["name"]),
            definition=str(m.get("definition", "")),
            examples=[str(e) for e in m.get("examples", [])],
        )
        for m in modes
    ]


def check_taxonomy(modes: list[FailureMode]) -> list[str]:
    """Structural warnings about a taxonomy before it gets used."""
    out: list[str] = []
    if len(modes) < TAXONOMY_MIN_MODES:
        out.append(
            f"{len(modes)} failure modes; ch. 3 lands at "
            f"{TAXONOMY_MIN_MODES}-{TAXONOMY_MAX_MODES}. Too few usually means "
            "categories were merged that have different root causes."
        )
    if len(modes) > TAXONOMY_MAX_MODES:
        out.append(
            f"{len(modes)} failure modes; more than {TAXONOMY_MAX_MODES} gets "
            "hard to apply consistently. Consider merging near-duplicates."
        )
    names = [m.name.lower() for m in modes]
    if len(set(names)) != len(names):
        out.append("duplicate mode names")
    generic = {"hallucination", "instruction following", "verbosity", "quality",
               "helpfulness", "relevance", "accuracy"}
    for m in modes:
        if m.name.lower() in generic:
            out.append(
                f"'{m.name}' is a generic category from LLM research, not "
                "something that emerged from your traces. Ch. 3 calls this out "
                "as the second most common mistake in error analysis."
            )
        if len(m.examples) == 0:
            out.append(f"'{m.name}' has no example traces; boundaries will drift")
    return out


# ------------------------------------------------------------ prevalence

def prevalence(
    labels: dict[str, dict[str, int]],
    modes: list[FailureMode],
    *,
    sampled_as: dict[str, str] | None = None,
    n_resamples: int = 10_000,
) -> list[dict[str, object]]:
    """Per-mode rates with bootstrap intervals.

    ``labels`` maps trace_id to {mode_name: 0 or 1}.

    If ``sampled_as`` is supplied, rates are also reported over the randomly
    sampled subset alone. That subset is the only one that estimates the
    system's real failure rate; the blended queue over-represents failures by
    construction, which is useful for finding modes and misleading for
    counting them.
    """
    ids = sorted(labels)
    out = []
    for m in modes:
        v = np.array([labels[i].get(m.name, 0) for i in ids], dtype=float)
        ci = bootstrap_mean(v, bounds=(0.0, 1.0), n_resamples=n_resamples)
        row: dict[str, object] = {
            "mode": m.name,
            "n": int(v.size),
            "count": int(v.sum()),
            "rate": float(v.mean()) if v.size else float("nan"),
            "ci_low": ci.low,
            "ci_high": ci.high,
        }
        if sampled_as:
            rid = [i for i in ids if sampled_as.get(i) == "random"]
            if rid:
                rv = np.array([labels[i].get(m.name, 0) for i in rid], dtype=float)
                rci = bootstrap_mean(rv, bounds=(0.0, 1.0), n_resamples=n_resamples)
                row.update(
                    unbiased_n=int(rv.size),
                    unbiased_rate=float(rv.mean()),
                    unbiased_ci_low=rci.low,
                    unbiased_ci_high=rci.high,
                )
        out.append(row)
    return sorted(out, key=lambda r: r["rate"], reverse=True)


def to_json(obj) -> str:
    """Serialize dataclasses in this module."""
    def default(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(type(o))

    return json.dumps(obj, indent=2, default=default)
