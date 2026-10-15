"""Typed state machine: plan -> retrieve -> gate -> decide -> explain -> verify.

:func:`run_pipeline` filters candidate weeks, threads an
:class:`AgentState` per candidate through the stages, returns output rows in
the contract shape, and persists a retrieval trace to
``outputs/findings.sqlite`` (table ``findings``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .gate import (
    Candidate,
    Cfg,
    Note,
    cfg_value,
    check,
    decide_one,
    format_date,
)
from .state import AgentState
from .tools import (
    closest_rejected_id,
    index_notes,
    search_notes,
    verify_claims,
    write_reason,
)

OUTPUT_DB_DEFAULT = "outputs/findings.sqlite"


# --- Stages ---------------------------------------------------------------


def plan_step(state: AgentState, cfg: Cfg = None) -> AgentState:
    """Validate the state carries a usable (route, week)."""
    _ = cfg
    if not state.route:
        raise ValueError("AgentState.route must not be empty.")
    format_date(state.week_of)  # Raises on a bad date.
    return state


def retrieve_step(state: AgentState, notes: Sequence[Note], cfg: Cfg = None) -> AgentState:
    """Propose candidate notes via deterministic retrieval."""
    candidate = _state_candidate(state)
    return replace(state, retrieved_ids=search_notes(candidate, notes, cfg))


def gate_step(state: AgentState, notes_by_id: Mapping[str, Note], cfg: Cfg = None) -> AgentState:
    """Apply G1-G5 to each retrieved note, in retrieval order."""
    candidate = _state_candidate(state)
    gated: list[tuple[str, bool, list[str]]] = []
    for note_id in state.retrieved_ids:
        note = notes_by_id.get(note_id)
        if note is None:
            gated.append((note_id, False, ["G1"]))
            continue
        passed, failed = check(note, candidate, cfg)
        gated.append((note_id, passed, failed))
    return replace(state, gated=gated)


def decide_step(state: AgentState, notes_by_id: Mapping[str, Note], cfg: Cfg = None) -> AgentState:
    """Pick the tie-break winner among passing notes (or flag unexplained)."""
    candidate = _state_candidate(state)
    retrieved = [notes_by_id[note_id] for note_id in state.retrieved_ids if note_id in notes_by_id]
    flagged, matched, _ = decide_one(candidate, retrieved, cfg)
    return replace(state, flagged=flagged, matched_note_id=matched)


def explain_step(state: AgentState, notes_by_id: Mapping[str, Note], cfg: Cfg = None) -> AgentState:
    """Write the deterministic template reason."""
    failed = {note_id: list(rules) for note_id, ok, rules in state.gated if not ok}
    closest = closest_rejected_id(_state_candidate(state), notes_by_id, state.gated, cfg)
    reason = write_reason(
        _state_candidate(state),
        state.flagged,
        state.matched_note_id,
        notes_by_id,
        closest_note_id=closest,
        failed_by_note=failed,
    )
    return replace(state, reason=reason)


def verify_step(state: AgentState, notes_by_id: Mapping[str, Note], cfg: Cfg = None) -> AgentState:
    """Verify claims; on failure fall back to the plain template."""
    _ = cfg
    facts: dict[str, Any] = {
        "route": state.route,
        "week_of": state.week_of,
        "cpt": state.cpt,
        "own_pct": state.own_pct,
        "peer_pct": state.peer_pct,
        "flagged": state.flagged,
        "matched_note_id": state.matched_note_id,
        "retrieved_ids": list(state.retrieved_ids),
        "notes": {
            note_id: {
                "start": str(note.get("start", "")),
                "scope": note.get("scope"),
                "effect": note.get("effect"),
            }
            for note_id, note in notes_by_id.items()
        },
    }
    if verify_claims(state.reason, facts):
        return state
    fallback = write_reason(
        _state_candidate(state),
        state.flagged,
        state.matched_note_id,
        notes_by_id,
        closest_note_id=None,
    )
    return replace(state, reason=fallback)


def run_stages(state: AgentState, notes: Sequence[Note], cfg: Cfg = None) -> AgentState:
    """Run the full plan->...->verify chain for one candidate."""
    notes_by_id = index_notes(notes)
    state = plan_step(state, cfg)
    state = retrieve_step(state, notes, cfg)
    state = gate_step(state, notes_by_id, cfg)
    state = decide_step(state, notes_by_id, cfg)
    state = explain_step(state, notes_by_id, cfg)
    return verify_step(state, notes_by_id, cfg)


# --- Pipeline -------------------------------------------------------------


def _state_candidate(state: AgentState) -> Candidate:
    return {
        "route": state.route,
        "week_of": state.week_of,
        "cpt": state.cpt,
        "own_pct": state.own_pct,
        "peer_pct": state.peer_pct,
    }


def _as_records(weekly: Any) -> list[dict[str, Any]]:
    to_dict = getattr(weekly, "to_dict", None)
    if callable(to_dict):
        records = to_dict("records")
        return [dict(record) for record in records]
    if isinstance(weekly, Iterable):
        return [dict(record) for record in weekly]
    raise TypeError("weekly must be a DataFrame or an iterable of mappings.")


def _row_float(row: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return default


def is_candidate(row: Mapping[str, Any], cfg: Cfg = None) -> bool:
    """Candidate rule: enough history and both rises above threshold."""
    own_thr = float(cfg_value(cfg, "own_threshold", "OWN_THRESHOLD", default=0.07))
    peer_thr = float(cfg_value(cfg, "peer_threshold", "PEER_THRESHOLD", default=0.10))
    min_hist = int(cfg_value(cfg, "min_history", "MIN_HISTORY", default=4))
    if "n_prior" in row and row["n_prior"] not in (None, ""):
        try:
            if int(float(str(row["n_prior"]))) < min_hist:
                return False
        except (TypeError, ValueError):
            return False
    own = _row_float(row, "own_pct", default=float("nan"))
    peer = _row_float(row, "peer_pct", default=float("nan"))
    return own >= own_thr and peer >= peer_thr


def format_vs_own(own_pct: float) -> str:
    """``+35.5% vs this route's past average`` (sign always shown)."""
    return f"{own_pct:+.1%} vs this route's past average"


def format_vs_peer(peer_pct: float) -> str:
    """``+21.0% vs similar-length routes this week`` (sign always shown)."""
    return f"{peer_pct:+.1%} vs similar-length routes this week"


def run_pipeline(
    weekly: Any,
    notes: Sequence[Note],
    cfg: Cfg = None,
    db_path: str | Path = OUTPUT_DB_DEFAULT,
) -> list[dict[str, Any]]:
    """Filter candidates, run the graph per row, persist the trace.

    Returns output-contract rows sorted by ``(week_of, route)`` and writes
    table ``findings`` (with retrieval trace) to ``db_path``.
    """
    cfg = dict(cfg or {})
    records = _as_records(weekly)
    candidates = [row for row in records if is_candidate(row, cfg)]
    candidates.sort(
        key=lambda row: (
            format_date(row["week_of"]),
            str(row.get("route", "")),
        )
    )
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for record in candidates:
        state = AgentState(
            route=str(record.get("route", "")),
            week_of=format_date(record["week_of"]),
            cpt=_row_float(record, "cpt", "cost_per_tonne_km"),
            own_pct=_row_float(record, "own_pct"),
            peer_pct=_row_float(record, "peer_pct"),
        )
        state = run_stages(state, notes, cfg)
        rows.append(
            {
                "route": state.route,
                "week_of": state.week_of,
                "cost_per_tonne_km": round(state.cpt, 2),
                "vs_own_history": format_vs_own(state.own_pct),
                "vs_similar_routes": format_vs_peer(state.peer_pct),
                "flagged": state.flagged,
                "matched_note_id": state.matched_note_id,
                "reason": state.reason,
            }
        )
        trace.append(
            {
                "route": state.route,
                "week_of": state.week_of,
                "cpt": state.cpt,
                "own_pct": state.own_pct,
                "peer_pct": state.peer_pct,
                "retrieved_ids": list(state.retrieved_ids),
                "gated": [
                    [note_id, passed, list(failed)] for note_id, passed, failed in state.gated
                ],
                "matched_note_id": state.matched_note_id,
                "flagged": state.flagged,
                "reason": state.reason,
            }
        )
    _write_sqlite(trace, Path(db_path))
    return rows


def _write_sqlite(trace: list[dict[str, Any]], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS findings")
        conn.execute(
            "CREATE TABLE findings ("
            "route TEXT NOT NULL, "
            "week_of TEXT NOT NULL, "
            "cpt REAL NOT NULL, "
            "own_pct REAL NOT NULL, "
            "peer_pct REAL NOT NULL, "
            "retrieved_ids TEXT NOT NULL, "
            "gated TEXT NOT NULL, "
            "matched_note_id TEXT NOT NULL, "
            "flagged TEXT NOT NULL, "
            "reason TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO findings "
            "(route, week_of, cpt, own_pct, peer_pct, retrieved_ids, "
            "gated, matched_note_id, flagged, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    item["route"],
                    item["week_of"],
                    item["cpt"],
                    item["own_pct"],
                    item["peer_pct"],
                    json.dumps(item["retrieved_ids"], sort_keys=True),
                    json.dumps(item["gated"], sort_keys=True),
                    item["matched_note_id"],
                    item["flagged"],
                    item["reason"],
                )
                for item in trace
            ],
        )
        conn.commit()


__all__ = [
    "decide_step",
    "explain_step",
    "format_vs_own",
    "format_vs_peer",
    "gate_step",
    "is_candidate",
    "plan_step",
    "retrieve_step",
    "run_pipeline",
    "run_stages",
    "verify_step",
]
