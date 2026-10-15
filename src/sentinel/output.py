"""Output formatting for RouteSentinel (Phase 2).

Builds the exact contract CSV from candidate rows. Numbers are formatted
from unrounded values; verdicts/notes come from the caller (at this phase
every candidate defaults to ``Yes`` with the unexplained template reason —
the evidence gate in Phase 4 will supply real decisions).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

FLAGGED_YES = "Yes"
FLAGGED_JUSTIFIED = "No (justified)"
FLAGGED_VALUES = (FLAGGED_YES, FLAGGED_JUSTIFIED)

UNEXPLAINED_REASON = (
    "No matching note found for this route or date range. "
    "Cost rise looks unexplained and worth a human review."
)

OWN_SUFFIX = " vs this route's past average"
SIMILAR_SUFFIX = " vs similar-length routes this week"

COST_COLUMNS = ("cost_per_tonne_km", "cpt")

Decision = tuple[str, str, str | None]


def format_cost(cpt: float) -> str:
    """Format cost per tonne-km with exactly 2 decimals."""
    return f"{float(cpt):.2f}"


def format_vs_own(own_pct: float) -> str:
    """Format own-history rise, sign always shown, 1 decimal, unrounded input."""
    return f"{float(own_pct):+.1%}{OWN_SUFFIX}"


def format_vs_similar(peer_pct: float) -> str:
    """Format peer rise, sign always shown, 1 decimal, unrounded input."""
    return f"{float(peer_pct):+.1%}{SIMILAR_SUFFIX}"


def format_week(week_of: Any) -> str:
    """Format a week start as ``YYYY-MM-DD`` (accepts str/datetime/Timestamp)."""
    if isinstance(week_of, str) and len(week_of) == 10:
        return week_of
    parsed = pd.to_datetime(week_of, errors="coerce")
    if pd.isna(parsed):
        text = str(week_of)
        return text[:10] if len(text) > 10 else text
    return str(parsed.strftime("%Y-%m-%d"))


def unexplained_reason() -> str:
    """Template reason for a candidate with no justifying note."""
    return UNEXPLAINED_REASON


def justified_reason(
    note_id: str,
    note_date: str | None = None,
    cause: str | None = None,
) -> str:
    """Deterministic template reason for a candidate justified by a note."""
    when = f" dated {note_date}" if note_date else ""
    what = f": {cause}" if cause else ""
    return f"Matches note {note_id}{when}{what}. The cost rise has a clear explanation."


def _cost_column(candidates_df: pd.DataFrame) -> str:
    for column in COST_COLUMNS:
        if column in candidates_df.columns:
            return column
    raise ValueError(f"candidates_df needs one cost column among: {COST_COLUMNS}")


def format_row(
    route: Any,
    week_of: Any,
    cpt: float,
    own_pct: float,
    peer_pct: float,
    flagged: str = FLAGGED_YES,
    matched_note_id: str = "",
    reason: str | None = None,
) -> dict[str, str]:
    """Format one candidate as an output-contract row (ordered dict)."""
    if flagged not in FLAGGED_VALUES:
        raise ValueError(f"flagged must be one of {FLAGGED_VALUES}, got {flagged!r}")
    note_id = matched_note_id or ""
    if (flagged == FLAGGED_YES) != (note_id == ""):
        raise ValueError("matched_note_id must be empty iff flagged == 'Yes'")
    if reason is None:
        reason = UNEXPLAINED_REASON if flagged == FLAGGED_YES else justified_reason(note_id)
    return {
        "route": str(route),
        "week_of": format_week(week_of),
        "cost_per_tonne_km": format_cost(cpt),
        "vs_own_history": format_vs_own(own_pct),
        "vs_similar_routes": format_vs_similar(peer_pct),
        "flagged": flagged,
        "matched_note_id": note_id,
        "reason": str(reason),
    }


def build_output_df(
    candidates_df: pd.DataFrame,
    decisions: Mapping[tuple[str, str], Decision] | None = None,
) -> pd.DataFrame:
    """Build the contract output frame from candidates.

    ``decisions`` maps ``(route, week_of)`` → ``(flagged, note_id, reason)``;
    rows without an entry default to ``Yes`` with the unexplained template.
    Output is sorted by ``(week_of, route)``.
    """
    missing = [
        c
        for c in ("route", "week_of", "n_prior", "own_pct", "peer_pct")
        if c not in candidates_df.columns
    ]
    if missing:
        raise ValueError(f"candidates_df is missing required columns: {missing}")
    cost_col = _cost_column(candidates_df)

    rows: list[dict[str, str]] = []
    for _, record in candidates_df.iterrows():
        key = (str(record["route"]), format_week(record["week_of"]))
        if decisions is not None and key in decisions:
            flagged, note_id, reason = decisions[key]
        else:
            flagged, note_id, reason = FLAGGED_YES, "", None
        rows.append(
            format_row(
                record["route"],
                record["week_of"],
                float(record[cost_col]),
                float(record["own_pct"]),
                float(record["peer_pct"]),
                flagged=flagged,
                matched_note_id=note_id,
                reason=reason,
            )
        )
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values(["week_of", "route"], kind="mergesort").reset_index(drop=True)
    return out


def write_output(
    candidates_df: pd.DataFrame,
    path: str | Path,
    decisions: Mapping[tuple[str, str], Decision] | None = None,
) -> Path:
    """Write candidates to the contract CSV at ``path``; return the path."""
    out = build_output_df(candidates_df, decisions)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return dest
