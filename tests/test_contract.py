"""Phase 2: output-contract tests (AGENTS.md Output contract).

Covers header/order/regexes/blank-id rule, plus reproduction of the
formatting patterns in fixtures/sample_output_format_v2.csv.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.detect import find_candidates
from sentinel.output import (
    FLAGGED_JUSTIFIED,
    FLAGGED_YES,
    OUTPUT_COLUMNS,
    UNEXPLAINED_REASON,
    build_output_df,
    format_cost,
    format_row,
    format_vs_own,
    format_vs_similar,
    format_week,
    write_output,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"

EXPECTED_HEADER = (
    "route,week_of,cost_per_tonne_km,vs_own_history,"
    "vs_similar_routes,flagged,matched_note_id,reason"
)
COST_RE = re.compile(r"^\d+\.\d{2}$")
OWN_RE = re.compile(r"^[+-]\d+\.\d% vs this route's past average$")
SIMILAR_RE = re.compile(r"^[+-]\d+\.\d% vs similar-length routes this week$")
WEEK_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _synthetic_candidates() -> pd.DataFrame:
    """Small inline candidates frame (values invented for this test)."""
    return pd.DataFrame(
        [
            {
                "route": "Beta-Alpha",
                "week_of": "2025-02-10",
                "cpt": 3.456,
                "n_prior": 8,
                "own_pct": 0.12345,
                "peer_pct": -0.0234,
            },
            {
                "route": "Alpha-Beta",
                "week_of": "2025-02-10",
                "cpt": 2.5,
                "n_prior": 8,
                "own_pct": 0.08,
                "peer_pct": 0.15,
            },
            {
                "route": "Alpha-Beta",
                "week_of": "2025-02-03",
                "cpt": 2.505,
                "n_prior": 7,
                "own_pct": 0.071,
                "peer_pct": 0.101,
            },
        ]
    )


def _assert_contract(df: pd.DataFrame) -> None:
    assert list(df.columns) == OUTPUT_COLUMNS
    assert set(df["flagged"].unique().tolist()) <= {FLAGGED_YES, FLAGGED_JUSTIFIED}
    for _, row in df.iterrows():
        assert COST_RE.match(str(row["cost_per_tonne_km"])), row
        assert OWN_RE.match(str(row["vs_own_history"])), row
        assert SIMILAR_RE.match(str(row["vs_similar_routes"])), row
        assert WEEK_RE.match(str(row["week_of"])), row
        # matched_note_id empty iff flagged == "Yes"
        assert (str(row["matched_note_id"]) == "") == (row["flagged"] == FLAGGED_YES)
        keys = list(zip(df["week_of"].tolist(), df["route"].tolist(), strict=False))
        assert keys == sorted(keys)


def test_header_is_exact() -> None:
    out = build_output_df(_synthetic_candidates())
    assert ",".join(out.columns.tolist()) == EXPECTED_HEADER


def test_contract_patterns_and_order() -> None:
    _assert_contract(build_output_df(_synthetic_candidates()))


def test_default_rows_are_unexplained_yes() -> None:
    out = build_output_df(_synthetic_candidates())
    assert (out["flagged"] == FLAGGED_YES).all()
    assert (out["matched_note_id"] == "").all()
    assert (out["reason"] == UNEXPLAINED_REASON).all()


def test_justified_decision_is_honoured() -> None:
    frame = _synthetic_candidates()
    key = ("Alpha-Beta", "2025-02-10")
    decisions = {key: (FLAGGED_JUSTIFIED, "N999", "Matches note N999.")}
    out = build_output_df(frame, decisions)
    row = out[(out["route"] == "Alpha-Beta") & (out["week_of"] == "2025-02-10")].iloc[0]
    assert row["flagged"] == FLAGGED_JUSTIFIED
    assert row["matched_note_id"] == "N999"
    _assert_contract(out)


def test_format_row_rejects_blank_id_violations() -> None:
    with pytest.raises(ValueError):
        format_row("A-B", "2025-01-06", 3.0, 0.1, 0.2, flagged=FLAGGED_YES, matched_note_id="N001")
    with pytest.raises(ValueError):
        format_row(
            "A-B", "2025-01-06", 3.0, 0.1, 0.2, flagged=FLAGGED_JUSTIFIED, matched_note_id=""
        )
    with pytest.raises(ValueError):
        format_row("A-B", "2025-01-06", 3.0, 0.1, 0.2, flagged="Maybe")


def test_format_row_uses_unrounded_values() -> None:
    row = format_row("A-B", "2025-01-06", 4.172957, 0.355201, 0.210344)
    assert row["cost_per_tonne_km"] == "4.17"
    assert row["vs_own_history"] == "+35.5% vs this route's past average"
    assert row["vs_similar_routes"] == "+21.0% vs similar-length routes this week"


def test_format_week_accepts_datetime_types() -> None:
    assert format_week(pd.Timestamp("2025-01-06")) == "2025-01-06"
    assert format_week("2025-01-06") == "2025-01-06"
    assert format_cost(4.1) == "4.10"
    assert format_vs_own(-0.05) == "-5.0% vs this route's past average"
    assert format_vs_similar(0.0) == "+0.0% vs similar-length routes this week"


def test_sample_format_reproduction() -> None:
    """Every sample row re-emitted via the formatter keeps its formatting."""
    with open(FIXTURES / "sample_output_format_v2.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and list(rows[0].keys()) == OUTPUT_COLUMNS
    for sample in rows:
        cost = float(sample["cost_per_tonne_km"])
        own = float(sample["vs_own_history"].split("%")[0]) / 100.0
        peer = float(sample["vs_similar_routes"].split("%")[0]) / 100.0
        assert format_cost(cost) == sample["cost_per_tonne_km"]
        assert format_vs_own(own) == sample["vs_own_history"]
        assert format_vs_similar(peer) == sample["vs_similar_routes"]
        assert format_week(sample["week_of"]) == sample["week_of"]
        assert (sample["matched_note_id"] == "") == (sample["flagged"] == FLAGGED_YES)


def test_full_pipeline_reference_to_contract(tmp_path: Path) -> None:
    """Reference metrics → candidates → contract CSV, all patterns green."""
    reference = pd.read_csv(FIXTURES / "reference_weekly_metrics.csv")
    candidates = find_candidates(reference)
    dest = write_output(candidates, tmp_path / "output.csv")
    assert dest.exists()
    with open(dest, newline="") as handle:
        header = handle.readline().strip()
    assert header == EXPECTED_HEADER
    back = pd.read_csv(dest, dtype=str, keep_default_na=False)
    assert list(back.columns) == OUTPUT_COLUMNS
    expected = pd.read_csv(FIXTURES / "expected_candidates.csv")
    assert len(back) == len(candidates) == len(expected)
    assert len(back) > 0
    keys = list(zip(back["week_of"].tolist(), back["route"].tolist(), strict=False))
    assert keys == sorted(keys)
    for _, row in back.iterrows():
        assert COST_RE.match(row["cost_per_tonne_km"])
        assert OWN_RE.match(row["vs_own_history"])
        assert SIMILAR_RE.match(row["vs_similar_routes"])
        assert (row["matched_note_id"] == "") == (row["flagged"] == FLAGGED_YES)
