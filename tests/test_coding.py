"""Tests for error analysis: sampling, open coding, taxonomy, prevalence.

Most of these check that the module refuses to let discipline slip — empty
notes, missing binary judgments, generic taxonomy categories, prevalence
computed over a deliberately biased queue.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from compton_eval.coding import (
    SATURATION_MIN_PROBLEMS,
    FailureMode,
    OpenCode,
    Trace,
    axial_prompt,
    build_queue,
    check_taxonomy,
    load_open_codes,
    load_traces,
    open_code_stub,
    prevalence,
    saturation_report,
)


def _trace(tid: str, **scores) -> Trace:
    mid, var = tid.split("-")
    return Trace(tid, int(mid), var, text=f"summary for {tid}", scores=scores)


def _population(n: int = 40) -> list[Trace]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        out.append(_trace(
            f"{i}-v1",
            top_dollar_recall=float(rng.random()),
            anomaly_surface_rate=float(rng.random()),
            factual_precision=float(rng.random()),
        ))
    return out


# -------------------------------------------------------------- sampling

def test_queue_follows_the_blend():
    q = build_queue(_population(60), n=20, seed=1)
    counts = {}
    for _, why in q:
        counts[why] = counts.get(why, 0) + 1
    assert len(q) == 20
    assert counts["failure-driven"] == 10
    assert counts["uncertainty"] == 6
    assert counts["random"] == 4


def test_queue_has_no_duplicates():
    q = build_queue(_population(60), n=30, seed=2)
    ids = [t.trace_id for t, _ in q]
    assert len(set(ids)) == len(ids)


def test_queue_always_includes_random_slice():
    """The random portion is the easy one to skip and the only unbiased one."""
    q = build_queue(_population(50), n=25, seed=3)
    assert any(why == "random" for _, why in q)


def test_queue_caps_at_population_size():
    q = build_queue(_population(7), n=40, seed=4)
    assert len(q) == 7


def test_failure_driven_picks_the_worst_first():
    traces = [
        _trace("1-v1", top_dollar_recall=1.0, factual_precision=1.0),
        _trace("2-v1", top_dollar_recall=0.0, factual_precision=0.0),
        _trace("3-v1", top_dollar_recall=0.9, factual_precision=0.9),
    ]
    q = build_queue(traces, n=1, blend={"failure": 1.0, "uncertainty": 0.0, "random": 0.0})
    assert q[0][0].trace_id == "2-v1"


def test_blend_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_queue(_population(10), n=5, blend={"failure": 0.5, "uncertainty": 0.2, "random": 0.2})


# ----------------------------------------------------------- open coding

def test_empty_note_rejected():
    with pytest.raises(ValueError, match="empty note"):
        OpenCode("1-v1", "", acceptable=False)
    with pytest.raises(ValueError, match="empty note"):
        OpenCode("1-v1", "   ", acceptable=True)


def test_worksheet_round_trip(tmp_path):
    q = build_queue(_population(20), n=6, seed=5)
    p = open_code_stub(q, tmp_path / "codes.json")
    payload = json.loads(p.read_text())
    assert len(payload["traces"]) == 6
    assert all(r["note"] == "" and r["acceptable"] is None for r in payload["traces"])
    assert "FIRST" in payload["_instructions"]

    for i, r in enumerate(payload["traces"]):
        r["note"] = f"note {i}"
        r["acceptable"] = i % 2 == 0
    p.write_text(json.dumps(payload))

    codes = load_open_codes(p)
    assert len(codes) == 6
    assert sum(1 for c in codes if not c.acceptable) == 3


def test_untouched_rows_are_skipped_not_counted(tmp_path):
    p = tmp_path / "codes.json"
    p.write_text(json.dumps({"traces": [
        {"trace_id": "1-v1", "note": "dropped the vendor name", "acceptable": False},
        {"trace_id": "2-v1", "note": "", "acceptable": None},
    ]}))
    assert len(load_open_codes(p)) == 1


def test_note_without_judgment_is_an_error(tmp_path):
    p = tmp_path / "codes.json"
    p.write_text(json.dumps({"traces": [
        {"trace_id": "1-v1", "note": "something felt off", "acceptable": None},
    ]}))
    with pytest.raises(ValueError, match="pick a side"):
        load_open_codes(p)


# ------------------------------------------------------------ saturation

def test_saturation_needs_enough_problems():
    few = [OpenCode(f"{i}-v1", "bad", acceptable=False) for i in range(5)]
    r = saturation_report(few, n_queued=40)
    assert not r["saturation_likely"]
    assert r["problems_needed"] == SATURATION_MIN_PROBLEMS - 5

    many = [OpenCode(f"{i}-v1", "bad", acceptable=False)
            for i in range(SATURATION_MIN_PROBLEMS)]
    assert saturation_report(many, n_queued=40)["saturation_likely"]


def test_acceptable_traces_do_not_count_toward_saturation():
    codes = [OpenCode(f"{i}-v1", "fine", acceptable=True) for i in range(50)]
    assert saturation_report(codes, 50)["problems"] == 0


# ---------------------------------------------------------- axial coding

def test_axial_prompt_uses_only_problem_notes():
    codes = [
        OpenCode("1-v1", "dropped the biggest contract", acceptable=False),
        OpenCode("2-v1", "reads fine", acceptable=True),
    ]
    p = axial_prompt(codes, "a council summarizer")
    assert "dropped the biggest contract" in p
    assert "reads fine" not in p
    assert "Do not invent new failure types" in p


# --------------------------------------------------------------- taxonomy

def test_definition_required():
    with pytest.raises(ValueError, match="one-line definition"):
        FailureMode("missing_vendor", "")


def test_generic_categories_flagged():
    modes = [
        FailureMode("hallucination", "makes things up", ["1-v1"]),
        FailureMode("verbosity", "too long", ["2-v1"]),
    ]
    joined = " ".join(check_taxonomy(modes))
    assert "generic category" in joined


def test_too_few_and_too_many_modes_flagged():
    one = [FailureMode("a", "d", ["x"])]
    assert any("failure modes" in w for w in check_taxonomy(one))
    many = [FailureMode(f"m{i}", "d", ["x"]) for i in range(12)]
    assert any("hard to apply consistently" in w for w in check_taxonomy(many))


def test_missing_examples_flagged():
    modes = [FailureMode(f"m{i}", "d", ["x"]) for i in range(5)]
    modes.append(FailureMode("m5", "d", []))
    assert any("no example traces" in w for w in check_taxonomy(modes))


def test_clean_taxonomy_passes():
    modes = [FailureMode(f"omits_{i}", "specific definition", [f"{i}-v1"]) for i in range(6)]
    assert check_taxonomy(modes) == []


# ------------------------------------------------------------ prevalence

def test_prevalence_counts_and_orders():
    modes = [FailureMode("a", "d", ["x"]), FailureMode("b", "d", ["x"])]
    labels = {
        "1-v1": {"a": 1, "b": 0},
        "2-v1": {"a": 1, "b": 0},
        "3-v1": {"a": 0, "b": 1},
        "4-v1": {"a": 1, "b": 0},
    }
    rows = prevalence(labels, modes, n_resamples=500)
    assert rows[0]["mode"] == "a"
    assert rows[0]["count"] == 3 and rows[0]["rate"] == pytest.approx(0.75)
    assert rows[1]["rate"] == pytest.approx(0.25)


def test_prevalence_separates_the_unbiased_slice():
    """Rate over a failure-driven queue is not the system's failure rate."""
    modes = [FailureMode("a", "d", ["x"])]
    labels = {f"{i}-v1": {"a": 1} for i in range(8)}
    labels.update({f"r{i}-v1": {"a": 0} for i in range(4)})
    sampled_as = {f"{i}-v1": "failure-driven" for i in range(8)}
    sampled_as.update({f"r{i}-v1": "random" for i in range(4)})

    rows = prevalence(labels, modes, sampled_as=sampled_as, n_resamples=500)
    assert rows[0]["rate"] == pytest.approx(8 / 12)
    assert rows[0]["unbiased_rate"] == pytest.approx(0.0)
    assert rows[0]["unbiased_n"] == 4


def test_prevalence_interval_within_bounds():
    modes = [FailureMode("a", "d", ["x"])]
    labels = {f"{i}-v1": {"a": i % 2} for i in range(30)}
    r = prevalence(labels, modes, n_resamples=1000)[0]
    assert 0.0 <= r["ci_low"] <= r["rate"] <= r["ci_high"] <= 1.0


# ------------------------------------------------------------- real data

def test_loads_real_eval_artifacts():
    run = ("/Users/o24s/Code/compton-civic-platform/eval-artifacts/runs/"
           "2026-04-15T23-05-34-891Z")
    try:
        traces = load_traces(run)
    except FileNotFoundError:
        pytest.skip("eval artifacts not present")
    assert len(traces) == 36
    t = traces[0]
    assert t.text and t.deterministic
    assert t.meta["model"] and t.meta["n_chapters"] > 0
