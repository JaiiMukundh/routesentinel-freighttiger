"""Phase 3 acceptance: rule extraction matches golden_notes.json 100%.

Compares scope, effect, start, end, open_ended and magnitude for every
gold-labelled note against the deterministic extractor over data/notes.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sentinel.notes.extract import extract_note  # noqa: E402

GOLD_FIELDS = ("scope", "effect", "start", "end", "open_ended", "mag")


def _load_note_rows() -> dict[str, dict[str, str]]:
    path = REPO_ROOT / "data" / "notes.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["note_id"]: row for row in csv.DictReader(handle)}


def _load_gold() -> list[dict[str, Any]]:
    path = REPO_ROOT / "fixtures" / "golden_notes.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _actual(note_id: str, rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    row = rows[note_id]
    ext = extract_note(
        note_id=note_id,
        applies_to=row["applies_to"],
        note_text=row["note"],
        note_date=date.fromisoformat(row["date"]),
    )
    scope: Any = ext.scope
    return {
        "scope": scope if scope == "ALL" else list(scope),
        "effect": ext.effect,
        "start": ext.start.isoformat(),
        "end": ext.end.isoformat() if ext.end else None,
        "open_ended": ext.open_ended,
        "mag": list(ext.mag),
    }


def test_rule_extraction_matches_gold_100_percent() -> None:
    rows = _load_note_rows()
    gold = _load_gold()
    assert len(gold) == 10, f"expected 10 gold notes, got {len(gold)}"
    mismatches: list[str] = []
    for entry in gold:
        got = _actual(entry["note_id"], rows)
        for field in GOLD_FIELDS:
            if got[field] != entry[field]:
                mismatches.append(
                    f"{entry['note_id']}.{field}: got {got[field]!r}, want {entry[field]!r}"
                )
    assert not mismatches, "extraction mismatches:\n" + "\n".join(mismatches)


def test_all_gold_extractions_are_high_confidence() -> None:
    rows = _load_note_rows()
    gold = _load_gold()
    for entry in gold:
        row = rows[entry["note_id"]]
        ext = extract_note(
            note_id=entry["note_id"],
            applies_to=row["applies_to"],
            note_text=row["note"],
            note_date=date.fromisoformat(row["date"]),
        )
        assert ext.confidence == "high", f"{entry['note_id']} confidence is low"


def test_scope_parsing_all_routes() -> None:
    rows = _load_note_rows()
    for note_id in ("N003", "N004", "N006", "N010"):
        row = rows[note_id]
        ext = extract_note(
            note_id=note_id,
            applies_to=row["applies_to"],
            note_text=row["note"],
            note_date=date.fromisoformat(row["date"]),
        )
        assert ext.scope == "ALL", f"{note_id} scope should be ALL"
