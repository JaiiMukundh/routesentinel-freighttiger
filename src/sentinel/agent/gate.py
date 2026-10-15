"""Deterministic evidence gate (G1-G5) with tie-break.

A note justifies a candidate week only if ALL gates pass:

- G1 scope: note scope is the route or ALL.
- G2 temporal: the Mon-Sun week overlaps the note window ``[start, end]``.
  Open-ended notes (``end`` missing and ``open_ended`` true) use
  ``[start, start + absorption_weeks)``. ``grace_days`` expands both sides.
- G3 polarity: ``effect == "increase"``.
- G4 magnitude: if the note states one, ``own_pct <= 2 * stated_max / 100``.
- G5 confidence: extraction confidence is not ``low``.

Tie-break among passing notes: route-specific over ALL, then larger overlap
(days), then later start, then lower note_id. Retrieval scores never justify
anything; this module decides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any

Note = Mapping[str, Any]
Candidate = Mapping[str, Any]
Cfg = Mapping[str, Any] | None

G1 = "G1"
G2 = "G2"
G3 = "G3"
G4 = "G4"
G5 = "G5"

FLAG_YES = "Yes"
FLAG_JUSTIFIED = "No (justified)"

GatedEntry = tuple[str, bool, list[str]]

_ALL_TOKENS = frozenset({"all", "all routes"})


def cfg_value(cfg: Cfg, *names: str, default: Any) -> Any:
    """Return the first present config key, else ``default``."""
    if cfg is None:
        return default
    for name in names:
        if name in cfg:
            return cfg[name]
    return default


def parse_date(value: str | date | datetime) -> date:
    """Parse an ISO date string (or pass through a date/datetime)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def format_date(value: str | date | datetime) -> str:
    """Normalise to ``YYYY-MM-DD``."""
    return parse_date(value).isoformat()


def _scope_is_all(scope: Any) -> bool:
    if scope is None:
        return False
    if isinstance(scope, str):
        return scope.strip().lower() in _ALL_TOKENS
    if isinstance(scope, (list, tuple, set, frozenset)):
        return any(isinstance(item, str) and item.strip().lower() in _ALL_TOKENS for item in scope)
    return False


def _scope_routes(scope: Any) -> list[str]:
    """Route-specific entries of a scope (empty when scope is ALL)."""
    if isinstance(scope, str):
        return [] if _scope_is_all(scope) else [scope]
    if isinstance(scope, (list, tuple, set, frozenset)):
        return [
            item
            for item in scope
            if isinstance(item, str) and item.strip().lower() not in _ALL_TOKENS
        ]
    return []


def is_route_specific(note: Note, route: str) -> bool:
    """True when the note names this route specifically (not via ALL)."""
    scope = note.get("scope")
    if _scope_is_all(scope):
        return False
    return route in _scope_routes(scope)


def note_window(note: Note, cfg: Cfg = None) -> tuple[date, date]:
    """Half-open effective window ``[start, end_exclusive)`` with grace applied.

    Closed notes use ``[start, end + 1 day]`` (end inclusive). Notes with no
    ``end`` and ``open_ended`` use ``[start, start + absorption_weeks)``;
    notes with no ``end`` and not open-ended are single-day.
    """
    grace = int(cfg_value(cfg, "grace_days", "GRACE_DAYS", default=0))
    absorb_weeks = int(cfg_value(cfg, "absorption_weeks", "ABSORPTION_WEEKS", default=8))
    start = parse_date(note["start"])
    end_raw = note.get("end")
    if end_raw is None:
        if bool(note.get("open_ended", True)):
            end_excl = start + timedelta(weeks=absorb_weeks)
        else:
            end_excl = start + timedelta(days=1)
    else:
        end_excl = parse_date(end_raw) + timedelta(days=1)
    delta = timedelta(days=grace)
    return (start - delta, end_excl + delta)


def week_window(candidate: Candidate) -> tuple[date, date]:
    """Half-open ``[monday, monday + 7 days)`` for the candidate week."""
    monday = parse_date(candidate["week_of"])
    return (monday, monday + timedelta(days=7))


def overlap_days(first: tuple[date, date], second: tuple[date, date]) -> int:
    """Whole overlapping days of two half-open windows (>= 0)."""
    latest_start = max(first[0], second[0])
    earliest_end = min(first[1], second[1])
    return max(0, (earliest_end - latest_start).days)


def _stated_max(mag: Any) -> float | None:
    """Largest stated percent magnitude, or None when unstated."""
    if mag is None:
        return None
    if isinstance(mag, bool):
        return None
    if isinstance(mag, (int, float)):
        return float(mag)
    if isinstance(mag, (list, tuple)):
        values = [
            float(item)
            for item in mag
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if not values:
            return None
        return max(values)
    return None


def check(note: Note, candidate: Candidate, cfg: Cfg = None) -> tuple[bool, list[str]]:
    """Apply G1-G5. Returns ``(passed, failed_rule_ids_in_order)``."""
    failed: list[str] = []
    route = str(candidate.get("route", ""))

    # G1 scope.
    scope = note.get("scope")
    if _scope_is_all(scope):
        g1 = True
    elif scope is None:
        g1 = False
    else:
        g1 = route in _scope_routes(scope)
    if not g1:
        failed.append(G1)

    # G2 temporal overlap.
    try:
        week = week_window(candidate)
        window = note_window(note, cfg)
        g2 = week[0] < window[1] and window[0] < week[1]
    except (KeyError, ValueError, TypeError):
        g2 = False
    if not g2:
        failed.append(G2)

    # G3 polarity.
    g3 = str(note.get("effect", "")).strip().lower() == "increase"
    if not g3:
        failed.append(G3)

    # G4 magnitude (only when the note states one).
    stated = _stated_max(note.get("mag"))
    if stated is None:
        g4 = True
    else:
        try:
            observed = float(candidate.get("own_pct", 0.0))
        except (TypeError, ValueError):
            observed = 0.0
        g4 = observed <= 2.0 * stated / 100.0
    if not g4:
        failed.append(G4)

    # G5 confidence.
    g5 = str(note.get("confidence", "high")).strip().lower() != "low"
    if not g5:
        failed.append(G5)

    return (not failed, failed)


def pick_best(candidate: Candidate, passing_notes: Sequence[Note], cfg: Cfg = None) -> Note | None:
    """Tie-break: route-specific > larger overlap > later start > lower id."""
    if not passing_notes:
        return None
    route = str(candidate.get("route", ""))
    week = week_window(candidate)

    def key(note: Note) -> tuple[int, int, int, str]:
        specific = is_route_specific(note, route)
        overlap = overlap_days(week, note_window(note, cfg))
        start = parse_date(note["start"]).toordinal()
        note_id = str(note.get("note_id", ""))
        return (0 if specific else 1, -overlap, -start, note_id)

    return min(passing_notes, key=key)


def decide_one(
    candidate: Candidate, notes: Sequence[Note], cfg: Cfg = None
) -> tuple[str, str, list[GatedEntry]]:
    """Decide one candidate. Returns ``(flagged, matched_note_id, gated)``.

    ``gated`` is in ``note_id`` order for determinism. ``matched_note_id``
    is empty unless some note passes every gate.
    """
    ordered = sorted(notes, key=lambda note: str(note.get("note_id", "")))
    gated: list[GatedEntry] = []
    passing: list[Note] = []
    for note in ordered:
        ok, failed = check(note, candidate, cfg)
        gated.append((str(note.get("note_id", "")), ok, failed))
        if ok:
            passing.append(note)
    best = pick_best(candidate, passing, cfg)
    if best is None:
        return (FLAG_YES, "", gated)
    return (FLAG_JUSTIFIED, str(best.get("note_id", "")), gated)


def decide(
    candidates: Sequence[Candidate], notes: Sequence[Note], cfg: Cfg = None
) -> list[dict[str, Any]]:
    """Decide every candidate. Each result has route/week/flagged/note/gated."""
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        flagged, matched, gated = decide_one(candidate, notes, cfg)
        results.append(
            {
                "route": str(candidate.get("route", "")),
                "week_of": format_date(candidate["week_of"]),
                "flagged": flagged,
                "matched_note_id": matched,
                "gated": gated,
            }
        )
    return results
