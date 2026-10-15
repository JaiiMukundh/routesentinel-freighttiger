"""Agent tools: retrieval, applicability, deterministic reasons, verification.

``search_notes`` only proposes candidates; :mod:`gate` disposes. Reason text
is built from a deterministic template so ``--no-llm`` output is stable.
``verify_claims`` is a stub returning True until the Phase 5 LLM verifier
lands; the graph already honours a False verdict via template fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .gate import (
    Candidate,
    Cfg,
    Note,
    cfg_value,
    check,
    format_date,
    is_route_specific,
    note_window,
    overlap_days,
    parse_date,
    week_window,
)


def index_notes(notes: Sequence[Note]) -> dict[str, Note]:
    """Map note_id -> note (later duplicates win; ids are unique in practice)."""
    return {str(note.get("note_id", "")): note for note in notes}


def search_notes(candidate: Candidate, notes: Sequence[Note], cfg: Cfg = None) -> list[str]:
    """Deterministically rank notes and return the top-k ids.

    Rank: route-specific scope first, then ALL scope, then out-of-scope;
    within each, larger temporal overlap, then later note start, then
    lower note_id. ``k`` comes from config (default 4).
    """
    top_k = int(cfg_value(cfg, "k", "top_k", default=4))
    route = str(candidate.get("route", ""))
    try:
        week = week_window(candidate)
    except (KeyError, ValueError, TypeError):
        return []

    ranked: list[tuple[tuple[int, int, int, str], str]] = []
    for note in notes:
        note_id = str(note.get("note_id", ""))
        scope = note.get("scope")
        if is_route_specific(note, route):
            scope_class = 0
        elif scope is not None and _is_all_scope(scope):
            scope_class = 1
        else:
            scope_class = 2
        try:
            overlap = overlap_days(week, note_window(note, cfg))
            start_ord = parse_date(note["start"]).toordinal()
        except (KeyError, ValueError, TypeError):
            overlap = 0
            start_ord = 0
        ranked.append(((scope_class, -overlap, -start_ord, note_id), note_id))
    ranked.sort(key=lambda item: item[0])
    return [note_id for _, note_id in ranked[: max(0, top_k)]]


def _is_all_scope(scope: Any) -> bool:
    if isinstance(scope, str):
        return scope.strip().lower() in {"all", "all routes"}
    if isinstance(scope, (list, tuple, set, frozenset)):
        return any(
            isinstance(item, str) and item.strip().lower() in {"all", "all routes"}
            for item in scope
        )
    return False


def check_applicability(note: Note, candidate: Candidate, cfg: Cfg = None) -> dict[str, Any]:
    """Structured wrapper over :func:`gate.check` with overlap info."""
    passed, failed = check(note, candidate, cfg)
    try:
        overlap = overlap_days(week_window(candidate), note_window(note, cfg))
    except (KeyError, ValueError, TypeError):
        overlap = 0
    return {
        "note_id": str(note.get("note_id", "")),
        "applicable": passed,
        "failed_rules": failed,
        "overlap_days": overlap,
    }


def closest_rejected_id(
    candidate: Candidate,
    notes_by_id: Mapping[str, Note],
    gated: list[tuple[str, bool, list[str]]],
    cfg: Cfg = None,
) -> str | None:
    """Id of the temporally closest rejected note, if any.

    Closeness is the absolute gap between the candidate Monday and the
    note start; ties break by lower note_id. Deterministic.
    """
    _ = cfg
    try:
        monday = parse_date(candidate["week_of"])
    except (KeyError, ValueError, TypeError):
        return None
    failed_ids = [note_id for note_id, ok, _ in gated if not ok]
    if not failed_ids:
        return None

    def key(note_id: str) -> tuple[int, str]:
        note = notes_by_id.get(note_id)
        if note is None:
            return (10**9, note_id)
        try:
            start = parse_date(note["start"])
        except (KeyError, ValueError, TypeError):
            return (10**9, note_id)
        return (abs((monday - start).days), note_id)

    return min(failed_ids, key=key)


def _scope_text(note: Note) -> str:
    scope = note.get("scope")
    if _is_all_scope(scope):
        return "all routes"
    if isinstance(scope, str):
        return scope
    if isinstance(scope, (list, tuple, set, frozenset)):
        return ", ".join(str(item) for item in sorted(scope))
    return "all routes"


def write_reason(
    candidate: Candidate,
    flagged: str,
    matched_note_id: str,
    notes_by_id: Mapping[str, Note],
    closest_note_id: str | None = None,
    failed_by_note: Mapping[str, list[str]] | None = None,
) -> str:
    """Deterministic template reason for a decided candidate."""
    week = format_date(candidate["week_of"])
    if flagged == "No (justified)" and matched_note_id in notes_by_id:
        note = notes_by_id[matched_note_id]
        start = format_date(note["start"])
        cause = str(note.get("cause", "event") or "event")
        effect = str(note.get("effect", "increase") or "increase")
        scope = _scope_text(note)
        return (
            f"Matches note {matched_note_id} dated {start}: recorded {cause} "
            f"{effect} on {scope} covers the week of {week}. "
            "The cost rise has a clear explanation."
        )
    if closest_note_id is not None and closest_note_id in notes_by_id:
        note = notes_by_id[closest_note_id]
        start = format_date(note["start"])
        failed = (failed_by_note or {}).get(closest_note_id, [])
        rules = ", ".join(failed) if failed else "all gates"
        return (
            f"The closest note ({closest_note_id}, dated {start}) does not "
            f"justify this rise (fails {rules}); no genuine justification "
            "found. Cost rise looks unexplained and worth a human review."
        )
    return (
        "No matching note found for this route or date range. "
        "Cost rise looks unexplained and worth a human review."
    )


def verify_claims(reason: str, fact_sheet: Mapping[str, Any]) -> bool:
    """Stub claim verifier: always True until the Phase 5 verifier lands.

    ``reason`` is template-generated from ``fact_sheet`` fields, so every
    cited id/date/route/number traces by construction.
    """
    _ = (reason, fact_sheet)
    return True
