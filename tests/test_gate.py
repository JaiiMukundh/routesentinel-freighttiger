"""Phase 4 acceptance: gate rules, tie-break, and gold verdicts (--no-llm)."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.agent.gate import check, decide, decide_one  # noqa: E402
from sentinel.agent.graph import run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
CFG: dict[str, Any] = {"grace_days": 0, "absorption_weeks": 8}


def load_notes() -> list[dict[str, Any]]:
    with open(FIXTURES / "golden_notes.json", encoding="utf-8") as handle:
        notes: list[dict[str, Any]] = json.load(handle)
    for note in notes:
        note.setdefault("confidence", "high")
    return notes


def note_map(notes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(note["note_id"]): note for note in notes}


def cand(
    route: str,
    week_of: str,
    own_pct: float,
    peer_pct: float = 0.20,
    cpt: float = 3.0,
) -> dict[str, Any]:
    return {
        "route": route,
        "week_of": week_of,
        "cpt": cpt,
        "own_pct": own_pct,
        "peer_pct": peer_pct,
    }


NOTES = load_notes()
BY_ID = note_map(NOTES)


def failed_for(note_id: str, candidate: dict[str, Any]) -> list[str]:
    passed, failed = check(BY_ID[note_id], candidate, CFG)
    assert not passed
    return failed


# --- Distractors: each rejected for its specific rule ----------------------


def test_n004_rejected_g3() -> None:
    assert failed_for("N004", cand("Delhi-Jaipur", "2024-03-11", 0.05)) == ["G3"]


def test_n005_rejected_g3() -> None:
    assert failed_for("N005", cand("Mumbai-Delhi", "2024-07-29", 0.05)) == ["G3"]


def test_n006_rejected_g3() -> None:
    assert failed_for("N006", cand("Mumbai-Pune", "2025-09-22", 0.05)) == ["G3"]


def test_n007_rejected_g3_in_window() -> None:
    assert failed_for("N007", cand("Delhi-Jaipur", "2024-05-20", 0.05)) == ["G3"]


def test_n007_rejected_g2_and_g3_for_november() -> None:
    failed = failed_for("N007", cand("Delhi-Jaipur", "2024-11-11", 0.355201))
    assert failed == ["G2", "G3"]


def test_n008_rejected_g3() -> None:
    assert failed_for("N008", cand("Kolkata-Bhubaneswar", "2025-06-09", 0.05)) == ["G3"]


def test_n009_rejected_g3() -> None:
    assert failed_for("N009", cand("Chennai-Bangalore", "2025-03-17", 0.136898)) == ["G3"]


def test_n010_rejected_g3() -> None:
    assert failed_for("N010", cand("Mumbai-Pune", "2025-10-27", 0.05)) == ["G3"]


def test_n001_rejected_g2_for_march_10_week() -> None:
    assert failed_for("N001", cand("Chennai-Bangalore", "2025-03-10", 0.220771)) == ["G2"]


def test_n003_rejected_g2_absorption_for_september() -> None:
    # Open-ended N003 covers [2025-05-05, +8 weeks); September is absorbed out.
    assert failed_for("N003", cand("Mumbai-Pune", "2025-09-15", 0.092497)) == ["G2"]


# --- Remaining gates --------------------------------------------------------


def test_g1_rejects_out_of_scope_route() -> None:
    passed, failed = check(BY_ID["N005"], cand("Chennai-Bangalore", "2025-07-29", 0.05), CFG)
    assert not passed
    assert "G1" in failed


def test_g4_rejects_rise_beyond_twice_stated_max() -> None:
    # N003 states 5-7%: 0.20 > 2 * 7% while the September row (0.0925) fits.
    passed, failed = check(BY_ID["N003"], cand("Delhi-Jaipur", "2025-05-05", 0.20), CFG)
    assert not passed
    assert failed == ["G4"]


def test_g5_rejects_low_confidence() -> None:
    note = dict(BY_ID["N002"])
    note["confidence"] = "low"
    passed, failed = check(note, cand("Ahmedabad-Mumbai", "2025-01-20", 0.294731), CFG)
    assert not passed
    assert failed == ["G5"]


def test_grace_days_expand_the_window() -> None:
    candidate = cand("Chennai-Bangalore", "2025-03-10", 0.220771)
    assert check(BY_ID["N001"], candidate, CFG)[0] is False
    grace_cfg: dict[str, Any] = {"grace_days": 5, "absorption_weeks": 8}
    assert check(BY_ID["N001"], candidate, grace_cfg) == (True, [])


# --- Positive controls ------------------------------------------------------


def test_n001_justifies_feb_24_week() -> None:
    assert check(BY_ID["N001"], cand("Chennai-Bangalore", "2025-02-24", 0.314641), CFG) == (
        True,
        [],
    )


def test_n002_justifies_jan_20_week() -> None:
    assert check(BY_ID["N002"], cand("Ahmedabad-Mumbai", "2025-01-20", 0.294731), CFG) == (True, [])


def test_tiebreak_prefers_route_specific_over_all() -> None:
    generic: dict[str, Any] = {
        "note_id": "N000",
        "scope": "ALL",
        "effect": "increase",
        "start": "2025-02-24",
        "end": "2025-03-02",
        "open_ended": False,
        "mag": [None, None],
        "confidence": "high",
    }
    candidate = cand("Chennai-Bangalore", "2025-02-24", 0.314641)
    flagged, matched, _ = decide_one(candidate, [generic, BY_ID["N001"]], CFG)
    assert (flagged, matched) == ("No (justified)", "N001")


# --- Gold verdicts for every expected row -----------------------------------


def load_expected() -> list[dict[str, str]]:
    with open(FIXTURES / "expected_candidates.csv", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_candidates() -> list[dict[str, Any]]:
    rows = []
    for row in load_expected():
        rows.append(
            cand(
                row["route"],
                row["week_of"],
                float(row["own_pct"]),
                float(row["peer_pct"]),
                float(row["cost_per_tonne_km"]),
            )
        )
    return rows


def test_every_expected_row_gets_expected_verdict() -> None:
    expected = load_expected()
    results = decide(expected_candidates(), NOTES, CFG)
    assert len(results) == len(expected) == 13
    for row, result in zip(expected, results, strict=True):
        assert result["route"] == row["route"]
        assert result["week_of"] == row["week_of"]
        assert result["flagged"] == row["expected_flagged"], row
        assert result["matched_note_id"] == row["expected_note_id"], row


def test_debatable_rows_are_yes_under_defaults() -> None:
    for row, result in zip(load_expected(), decide(expected_candidates(), NOTES, CFG), strict=True):
        if row["tag"].startswith("debatable"):
            assert result["flagged"] == "Yes", row
            assert result["matched_note_id"] == "", row


def test_run_pipeline_matches_expected_and_writes_trace(tmp_path: Path) -> None:
    weekly = []
    for row in load_expected():
        weekly.append(
            {
                "route": row["route"],
                "week_of": row["week_of"],
                "cpt": float(row["cost_per_tonne_km"]),
                "n_prior": 8,
                "own_pct": float(row["own_pct"]),
                "peer_pct": float(row["peer_pct"]),
            }
        )
    weekly.append(
        {
            "route": "Mumbai-Pune",
            "week_of": "2025-09-01",
            "cpt": 3.5,
            "n_prior": 8,
            "own_pct": 0.01,
            "peer_pct": 0.02,
        }
    )
    db_path = tmp_path / "findings.sqlite"
    rows = run_pipeline(weekly, NOTES, CFG, db_path=db_path)
    assert len(rows) == 13
    assert [(row["week_of"], row["route"]) for row in rows] == sorted(
        (row["week_of"], row["route"]) for row in rows
    )
    for expected_row, row in zip(load_expected(), rows, strict=True):
        assert row["flagged"] == expected_row["expected_flagged"]
        assert row["matched_note_id"] == expected_row["expected_note_id"]
        if row["flagged"] == "Yes":
            assert row["matched_note_id"] == ""
    with sqlite3.connect(db_path) as conn:
        stored = conn.execute(
            "SELECT route, week_of, retrieved_ids, gated, matched_note_id, "
            "flagged FROM findings ORDER BY week_of, route"
        ).fetchall()
    assert len(stored) == 13
    for stored_row, row in zip(stored, rows, strict=True):
        assert stored_row[0] == row["route"]
        assert stored_row[1] == row["week_of"]
        assert stored_row[2]  # Non-empty retrieval trace.
        assert stored_row[3]  # Non-empty gate trace.
        assert stored_row[4] == row["matched_note_id"]
        assert stored_row[5] == row["flagged"]
