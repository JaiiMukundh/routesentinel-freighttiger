"""Deterministic regex/rule extractor for freight context notes.

No LLM, no embeddings, no randomness. All functions are pure: identical
inputs always produce identical outputs.

For each free-text note the extractor answers:
- scope: which routes it applies to (``"ALL"`` or an explicit route list),
- effect: whether the note claims costs rose (``increase``) or not,
- window: ``[start, end]`` plus an ``open_ended`` flag for ongoing changes,
- mag: an optional stated percentage range,
- cause: a coarse topic label.
"""

from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Effect = Literal["increase", "decrease", "none", "normalisation"]
Cause = Literal["weather", "festival", "fuel", "toll", "road", "demand", "other", "regulatory"]
Confidence = Literal["high", "low"]
Scope = Literal["ALL"] | list[str]

SINGLE_WEEK_DAYS = 6


class NoteExtraction(BaseModel):
    """Structured extraction of one context note."""

    note_id: str
    scope: Scope
    effect: Effect
    start: date
    end: date | None
    open_ended: bool
    mag: tuple[int | float | None, int | float | None]
    cause: Cause
    confidence: Confidence


_MONTHS: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Explicit date ranges such as "from Feb 24 to Mar 8" (year optional).
_RANGE_RE = re.compile(
    r"from\s+([A-Za-z]+\.?)\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?"
    r"\s+to\s+([A-Za-z]+\.?)\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?",
    re.IGNORECASE,
)

# Language signalling an ongoing change with no stated end.
_OPEN_ENDED_RES: list[re.Pattern[str]] = [
    re.compile(r"\bstart(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"until\s+further\s+notice", re.IGNORECASE),
    re.compile(r"\bongoing\b", re.IGNORECASE),
]

# Language signalling a short-lived event confined to roughly one week.
_LIMITED_WEEK_RES: list[re.Pattern[str]] = [
    re.compile(r"for\s+(?:about\s+|approximately\s+)?a\s+week", re.IGNORECASE),
    re.compile(r"\btemporary\b", re.IGNORECASE),
    re.compile(r"\bfestival\s+week\b", re.IGNORECASE),
]

# Polarity signals. Negation / normalisation / improvement patterns take
# precedence over increase patterns so that e.g. "costs were absorbed"
# is never read as a cost rise.
_NORMALISATION_RES: list[re.Pattern[str]] = [
    re.compile(r"return(?:s|ed)?\s+to\s+normal", re.IGNORECASE),
    re.compile(r"back\s+to\s+normal", re.IGNORECASE),
    re.compile(r"restor(?:ed|ation)\s+to\s+normal", re.IGNORECASE),
]

_DECREASE_RES: list[re.Pattern[str]] = [
    re.compile(r"\bimproved?\b", re.IGNORECASE),
    re.compile(r"costs?\s+(?:eased|declined|decreased|fell|dropped)", re.IGNORECASE),
    re.compile(r"rates?\s+(?:cut|reduced|lowered)", re.IGNORECASE),
]

_NONE_RES: list[re.Pattern[str]] = [
    re.compile(r"not\s+significantly\s+affected", re.IGNORECASE),
    re.compile(r"no\s+rate\s+change", re.IGNORECASE),
    re.compile(r"without\s+a?\s*rate\s+change", re.IGNORECASE),
    re.compile(r"\babsorb(?:ed|s)?\b", re.IGNORECASE),
    re.compile(r"not\s+part\s+of\s+this\s+dataset", re.IGNORECASE),
    re.compile(r"no\s+significant\s+disruptions?", re.IGNORECASE),
    re.compile(r"no\s+major\s+disruptions?", re.IGNORECASE),
    re.compile(r"remain(?:s|ed)?\s+stable", re.IGNORECASE),
    re.compile(r"remain(?:s|ed)?\s+normal", re.IGNORECASE),
]

_INCREASE_RES: list[re.Pattern[str]] = [
    re.compile(r"\bsurcharge\b", re.IGNORECASE),
    re.compile(r"higher\s+(?:trip\s+)?costs?", re.IGNORECASE),
    re.compile(r"push(?:ing|ed)?\s+up", re.IGNORECASE),
    re.compile(r"\brose\b", re.IGNORECASE),
    re.compile(r"\bdetours?\b", re.IGNORECASE),
    re.compile(r"costs?\s+(?:rose|increased?|went\s+up|climbed|surged)", re.IGNORECASE),
    re.compile(r"(?:price|rate)\s+hike", re.IGNORECASE),
]

# Cause signals, checked in order; first match wins.
_CAUSE_RULES: list[tuple[Cause, list[re.Pattern[str]]]] = [
    (
        "weather",
        [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"flood",
                r"cyclone",
                r"\bstorm\b",
                r"landslide",
                r"rainfall",
                r"monsoon",
                r"\bweather\b",
            ]
        ],
    ),
    (
        "festival",
        [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"festival",
                r"diwali",
                r"holi",
                r"\beid\b",
                r"durga",
                r"\bpuja\b",
                r"navratri",
            ]
        ],
    ),
    (
        "fuel",
        [re.compile(p, re.IGNORECASE) for p in [r"diesel", r"petrol", r"\bfuel\b", r"crude\s*oil"]],
    ),
    ("toll", [re.compile(r"\btoll\b", re.IGNORECASE)]),
    (
        "road",
        [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"maintenance",
                r"resurfacing",
                r"road\s+conditions?",
                r"pothole",
                r"construction\s+work",
            ]
        ],
    ),
    (
        "regulatory",
        [
            re.compile(p, re.IGNORECASE)
            for p in [r"mandate", r"compliance", r"regulation", r"\bpermit\b"]
        ],
    ),
    (
        "demand",
        [
            re.compile(p, re.IGNORECASE)
            for p in [r"\bdemand\b", r"truck\s+availability", r"\bcapacity\b"]
        ],
    ),
]

_RANGE_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*%")
_SINGLE_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _month_num(name: str) -> int | None:
    """Map a month name/abbreviation to its number, else None."""
    return _MONTHS.get(name.strip().rstrip(".").lower())


def _num(text: str) -> int | float:
    """Parse a numeric literal, keeping ints integral."""
    return float(text) if "." in text else int(text)


def extract_scope(applies_to: str) -> Scope:
    """Parse the ``applies_to`` column: "All Routes" -> "ALL", else [route]."""
    cleaned = applies_to.strip()
    if cleaned.lower() in {"all routes", "all", "*"}:
        return "ALL"
    routes = [part.strip() for part in cleaned.split(";") if part.strip()]
    return routes


def extract_effect(note_text: str) -> tuple[Effect, Confidence]:
    """Classify cost polarity; explicit negations beat increase keywords."""
    normalisation = any(p.search(note_text) for p in _NORMALISATION_RES)
    decrease = any(p.search(note_text) for p in _DECREASE_RES)
    none = any(p.search(note_text) for p in _NONE_RES)
    increase = any(p.search(note_text) for p in _INCREASE_RES)
    if normalisation:
        return ("normalisation", "high" if not increase else "low")
    if decrease:
        return ("decrease", "high" if not increase else "low")
    if none:
        return ("none", "high" if not increase else "low")
    if increase:
        return ("increase", "high")
    return ("none", "low")


def extract_window(note_text: str, note_date: date) -> tuple[date, date | None, bool]:
    """Infer the note's effective window.

    Start is always the note date. End comes from an explicit "from X to Y"
    range (year inferred from the note date when absent), else a single-week
    window for explicitly short-lived events, else None. Open-ended flags an
    ongoing change whose 8-week absorption window is applied later at the gate.
    """
    start = note_date
    match = _RANGE_RE.search(note_text)
    if match:
        end_month = _month_num(match.group(4))
        end_day = int(match.group(5))
        explicit_year = match.group(6)
        if end_month is None:
            return (start, None, False)
        year = int(explicit_year) if explicit_year else start.year
        end = date(year, end_month, end_day)
        if end < start:  # e.g. note in December describing early January.
            end = date(year + 1, end_month, end_day)
        return (start, end, False)
    if any(p.search(note_text) for p in _LIMITED_WEEK_RES):
        return (start, start + timedelta(days=SINGLE_WEEK_DAYS), False)
    if any(p.search(note_text) for p in _OPEN_ENDED_RES):
        return (start, None, True)
    return (start, None, False)


def extract_magnitude(note_text: str) -> tuple[int | float | None, int | float | None]:
    """Extract a stated percentage range, else (None, None)."""
    match = _RANGE_PCT_RE.search(note_text)
    if match:
        return (_num(match.group(1)), _num(match.group(2)))
    single = _SINGLE_PCT_RE.search(note_text)
    if single:
        value = _num(single.group(1))
        return (value, value)
    return (None, None)


def extract_cause(note_text: str) -> Cause:
    """Classify the note's topic; first matching rule wins, else "other"."""
    for cause, patterns in _CAUSE_RULES:
        if any(p.search(note_text) for p in patterns):
            return cause
    return "other"


def extract_note(
    note_id: str,
    applies_to: str,
    note_text: str,
    note_date: date | str,
) -> NoteExtraction:
    """Run the full deterministic rule pipeline for one note."""
    as_date = date.fromisoformat(note_date) if isinstance(note_date, str) else note_date
    effect, confidence = extract_effect(note_text)
    start, end, open_ended = extract_window(note_text, as_date)
    return NoteExtraction(
        note_id=note_id,
        scope=extract_scope(applies_to),
        effect=effect,
        start=start,
        end=end,
        open_ended=open_ended,
        mag=extract_magnitude(note_text),
        cause=extract_cause(note_text),
        confidence=confidence,
    )


def load_notes(csv_path: str | Path) -> list[NoteExtraction]:
    """Load and extract every note from a notes CSV file."""
    extractions: list[NoteExtraction] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            extractions.append(
                extract_note(
                    note_id=row["note_id"],
                    applies_to=row["applies_to"],
                    note_text=row["note"],
                    note_date=date.fromisoformat(row["date"]),
                )
            )
    extractions.sort(key=lambda item: item.note_id)
    return extractions
