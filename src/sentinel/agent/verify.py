"""Claim verifier for LLM-worded reasons.

Every checkable claim in ``reason`` must trace back to the fact sheet built
by :mod:`sentinel.agent.explain`:

- note ids (``N001`` …) must be the matched or closest-rejected note id;
- ISO dates must be the candidate week, the note start/end, or the
  rejected-note date;
- route names (``City-City``) must be the candidate route or a route in a
  fact-sheet scope;
- percentages must equal one of the fact-sheet percentages (own-history,
  peer, or stated note magnitude) within rounding tolerance;
- 2-decimal cost figures must equal the fact-sheet ``cost_per_tonne_km``.

Anything untraced fails verification and the caller falls back to the
deterministic template (after one retry). Bare integers, month names and
other qualitative wording are not treated as claims.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_NOTE_ID_RE = re.compile(r"\bN\d{3}\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ROUTE_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}(?:-[A-Z][A-Za-z]{2,})+\b")
_PCT_RE = re.compile(r"(?P<sign>[+-]?)(?P<num>\d+(?:\.\d+)?)\s*%")
_COST_RE = re.compile(r"(?<![\d.])\d+\.\d{2}(?![\d%])")

_WEEKDAY_PARTS = frozenset(
    {
        "Mon",
        "Tue",
        "Tues",
        "Wed",
        "Thu",
        "Thur",
        "Thurs",
        "Fri",
        "Sat",
        "Sun",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }
)

_PCT_TOL = 0.06  # 1-decimal rounding can err by at most 0.05
_COST_TOL = 0.005  # 2-decimal rounding can err by at most 0.005


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of :func:`verify_reason`."""

    ok: bool
    violations: tuple[str, ...] = ()


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [] if value in ("", "ALL") else [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str) and v not in ("", "ALL")]
    return []


def _allowed_ids(facts: Mapping[str, Any]) -> set[str]:
    ids = set()
    for key in ("matched_note_id", "note_id", "rejected_note_id"):
        value = facts.get(key)
        if isinstance(value, str) and value:
            ids.add(value)
    return ids


def _allowed_dates(facts: Mapping[str, Any]) -> set[str]:
    dates = set()
    for key in ("week_of", "note_date", "note_end", "rejected_note_date"):
        value = facts.get(key)
        if isinstance(value, str) and _DATE_RE.fullmatch(value):
            dates.add(value)
    return dates


def _allowed_routes(facts: Mapping[str, Any]) -> set[str]:
    routes = set()
    route = facts.get("route")
    if isinstance(route, str) and route:
        routes.add(route)
    for key in ("note_scope", "rejected_note_scope"):
        routes.update(_str_list(facts.get(key)))
    return routes


def _allowed_percentages(facts: Mapping[str, Any]) -> list[float]:
    pcts = []
    for key in ("own_pct", "peer_pct"):
        value = facts.get(key)
        if isinstance(value, (int, float)):
            pcts.append(float(value) * 100.0)
    for key in ("note_mag_low", "note_mag_high", "rejected_note_mag_low", "rejected_note_mag_high"):
        value = facts.get(key)
        if isinstance(value, (int, float)):
            pcts.append(float(value))
    for key in ("own_pct_display", "peer_pct_display"):
        value = facts.get(key)
        if isinstance(value, str):
            match = _PCT_RE.search(value)
            if match:
                pcts.append(float(match.group("num")))
    return pcts


def _is_route_token(token: str) -> bool:
    """Exclude weekday ranges such as ``Mon-Sun`` from route detection."""
    return not any(part in _WEEKDAY_PARTS for part in token.split("-"))


def verify_reason(reason: str, fact_sheet: Mapping[str, Any]) -> VerifyResult:
    """Check every checkable claim in ``reason`` against ``fact_sheet``."""
    violations: list[str] = []

    allowed_ids = _allowed_ids(fact_sheet)
    for note_id in sorted(set(_NOTE_ID_RE.findall(reason))):
        if note_id not in allowed_ids:
            violations.append(f"untraced note id: {note_id}")

    allowed_dates = _allowed_dates(fact_sheet)
    for date in sorted(set(_DATE_RE.findall(reason))):
        if date not in allowed_dates:
            violations.append(f"untraced date: {date}")

    allowed_routes = _allowed_routes(fact_sheet)
    found_routes = {m for m in _ROUTE_RE.findall(reason) if _is_route_token(m)}
    for route in sorted(found_routes):
        if route not in allowed_routes:
            violations.append(f"untraced route: {route}")

    allowed_pcts = _allowed_percentages(fact_sheet)
    for match in _PCT_RE.finditer(reason):
        sign = match.group("sign")
        if sign == "-" and match.start() > 0 and reason[match.start() - 1].isdigit():
            sign = "+"  # range dash as in "5-7%", not a negative number
        claimed = float(sign + match.group("num"))
        if not any(abs(claimed - allowed) <= _PCT_TOL for allowed in allowed_pcts):
            violations.append(f"untraced percentage: {match.group(0).strip()}")

    cost = fact_sheet.get("cost_per_tonne_km")
    for match in _COST_RE.finditer(reason):
        if not isinstance(cost, (int, float)):
            violations.append(f"untraced cost figure: {match.group(0)}")
        elif abs(float(match.group(0)) - round(float(cost), 2)) > _COST_TOL:
            violations.append(f"untraced cost figure: {match.group(0)}")

    violations.sort()
    return VerifyResult(ok=not violations, violations=tuple(violations))
