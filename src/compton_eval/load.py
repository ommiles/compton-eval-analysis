"""Load eval run artifacts into a tidy frame.

The upstream harness writes one ``scores.json`` per run:

    {"run_id": ..., "scores": [{"meeting_id": 1, "variant": "v0",
                                "top_dollar_recall": 0.4, ...}, ...]}

Every row is one (meeting, variant) pair, so the design is fully paired —
the same meetings are scored under every prompt variant. That pairing is
what licenses the signed-rank tests in :mod:`compton_eval.compare`, so it
is checked rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Scored dimensions and their measurement scale.
#: Bounded rates live on [0, 1]; the tone judge emits an ordinal 1-5.
DIMENSIONS: dict[str, tuple[float, float]] = {
    "top_dollar_recall": (0.0, 1.0),
    "anomaly_surface_rate": (0.0, 1.0),
    "tone_score": (1.0, 5.0),
    "factual_precision": (0.0, 1.0),
}

#: Dimensions produced by a deterministic checker rather than an LLM judge.
DETERMINISTIC = {"top_dollar_recall", "anomaly_surface_rate", "factual_precision"}


@dataclass(frozen=True)
class EvalRun:
    """One scoring run, tidied."""

    run_id: str
    scores: pd.DataFrame

    @property
    def variants(self) -> list[str]:
        return sorted(self.scores["variant"].unique())

    @property
    def meetings(self) -> list[int]:
        return sorted(self.scores["meeting_id"].unique())

    def dimension(self, dim: str) -> pd.DataFrame:
        """Wide frame for one dimension: index=meeting_id, columns=variants.

        Rows where any variant is missing are dropped, because a paired test
        needs the pair. The count of dropped rows is worth reporting — see
        :meth:`pairable`.
        """
        wide = self.scores.pivot(index="meeting_id", columns="variant", values=dim)
        return wide.dropna(how="any")

    def pairable(self, dim: str) -> tuple[int, int]:
        """(complete pairs, total meetings) for a dimension."""
        wide = self.scores.pivot(index="meeting_id", columns="variant", values=dim)
        return len(wide.dropna(how="any")), len(wide)


def load_run(path: str | Path) -> EvalRun:
    """Load a single ``scores.json`` (or a run directory containing one)."""
    p = Path(path)
    if p.is_dir():
        p = p / "scores.json"
    if not p.exists():
        raise FileNotFoundError(f"no scores.json at {p}")

    payload = json.loads(p.read_text())
    rows = payload.get("scores")
    if not rows:
        raise ValueError(f"{p} has no 'scores' array")

    df = pd.DataFrame(rows)

    missing = {"meeting_id", "variant"} - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing required columns: {sorted(missing)}")

    present = [d for d in DIMENSIONS if d in df.columns]
    if not present:
        raise ValueError(f"{p} has none of the known dimensions: {list(DIMENSIONS)}")

    keep = ["meeting_id", "variant", *present]
    df = df[keep].copy()
    df["meeting_id"] = df["meeting_id"].astype(int)
    df["variant"] = df["variant"].astype(str)
    for d in present:
        df[d] = pd.to_numeric(df[d], errors="coerce")

    dupes = df.duplicated(subset=["meeting_id", "variant"]).sum()
    if dupes:
        raise ValueError(
            f"{p} has {dupes} duplicate (meeting, variant) rows — "
            "the pairing is ambiguous, refusing to guess"
        )

    return EvalRun(run_id=payload.get("run_id", p.parent.name), scores=df)


def load_runs(root: str | Path) -> list[EvalRun]:
    """Load every run under ``eval-artifacts/runs/``, oldest first."""
    root = Path(root)
    candidates = sorted(root.glob("**/scores.json"))
    if not candidates:
        raise FileNotFoundError(f"no scores.json found under {root}")
    return [load_run(c) for c in candidates]
