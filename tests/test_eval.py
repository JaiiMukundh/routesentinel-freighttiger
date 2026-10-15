"""Phase 6 eval tests: red-team 100%, silver verdict-accuracy smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import (  # noqa: E402
    EvalConfig,
    load_config,
    load_golden_notes,
    load_redteam_cases,
    load_silver,
    run_redteam_case,
    verdict_for,
)
from eval.inject_anomalies import run_injection_benchmark  # noqa: E402


def test_redteam_pass_rate_is_100_percent() -> None:
    cfg: EvalConfig = load_config()
    base = load_golden_notes()
    cases = load_redteam_cases()
    assert len(cases) == 9, f"expected 9 red-team cases, got {len(cases)}"
    failures = []
    for case in cases:
        ok, detail = run_redteam_case(base, case, cfg)
        if not ok:
            failures.append(f"{case.case_id}: {detail}")
    assert not failures, "red-team failures:\n" + "\n".join(failures)


def test_silver_verdict_accuracy_is_100_percent() -> None:
    """Smoke test on the current pipeline stub: reference-path verdicts equal silver."""
    cfg = load_config()
    notes = load_golden_notes()
    mismatches = []
    for row in load_silver(include_debatable=True):
        own = _own_pct(row.route, row.week_of.isoformat())
        flagged, matched = verdict_for(row.route, row.week_of, own, notes, cfg)
        if (flagged, matched) != (row.expected_flagged, row.expected_note_id):
            mismatches.append(f"{row.route} {row.week_of}: got {(flagged, matched)}")
    assert not mismatches, "silver verdict mismatches:\n" + "\n".join(mismatches)


def test_pipeline_output_matches_silver_when_present() -> None:
    """If outputs/output.csv exists, its non-reason columns must match silver."""
    import csv

    output_csv = Path(__file__).resolve().parent.parent / "outputs" / "output.csv"
    if not output_csv.exists():
        import pytest

        pytest.skip(
            "outputs/output.csv not present (pipeline stub); reference path covered elsewhere"
        )
    with output_csv.open(newline="") as fh:
        got = {
            (r["route"], r["week_of"]): (r["flagged"], r["matched_note_id"])
            for r in csv.DictReader(fh)
        }
    mismatches = []
    for row in load_silver(include_debatable=True):
        key = (row.route, row.week_of.isoformat())
        if got.get(key, ("?", "?")) != (row.expected_flagged, row.expected_note_id):
            mismatches.append(f"{key}: got {got.get(key)}")
    assert not mismatches, "pipeline output mismatches:\n" + "\n".join(mismatches)


def test_injection_benchmark_deterministic_and_monotone() -> None:
    cfg = load_config()
    first, targets_a = run_injection_benchmark(cfg=cfg)
    second, targets_b = run_injection_benchmark(cfg=cfg)
    assert targets_a == targets_b, "target selection not deterministic"
    assert [(r.magnitude, r.n_detected, r.recall) for r in first] == [
        (r.magnitude, r.n_detected, r.recall) for r in second
    ], "benchmark not deterministic across runs"
    recalls = [r.recall for r in first]
    pairs = zip(recalls, recalls[1:], strict=False)
    assert all(b >= a for a, b in pairs), f"recall not monotone: {recalls}"
    mags = [r.magnitude for r in first]
    assert mags == sorted(mags), "magnitudes not sorted"


def _own_pct(route: str, week_of: str) -> float:
    import pandas as pd

    ref = pd.read_csv(
        Path(__file__).resolve().parent.parent / "fixtures" / "reference_weekly_metrics.csv"
    )
    hit = ref[(ref["route"] == route) & (ref["week_of"] == week_of)]
    assert not hit.empty, f"unknown route-week {route} {week_of}"
    return float(hit.iloc[0]["own_pct"])
