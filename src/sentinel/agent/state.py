"""Typed agent state for the RouteSentinel evidence-gate pipeline.

The state machine in ``graph.py`` threads an :class:`AgentState` through
``plan -> retrieve -> gate -> decide -> explain -> verify``. Every field is
typed; there is no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Per-(route, week) pipeline state."""

    route: str = ""
    week_of: str = ""  # Monday of the candidate week, YYYY-MM-DD.
    cpt: float = 0.0  # Pooled cost per tonne-km for the week.
    own_pct: float = 0.0  # Fractional rise vs own history (e.g. 0.355 = +35.5%).
    peer_pct: float = 0.0  # Fractional rise vs same-type peers that week.
    retrieved_ids: list[str] = field(default_factory=list)
    # (note_id, passed_all_gates, failed_rule_ids) in retrieval order.
    gated: list[tuple[str, bool, list[str]]] = field(default_factory=list)
    matched_note_id: str = ""  # Empty unless justified.
    flagged: str = "Yes"  # Exactly "Yes" or "No (justified)".
    reason: str = ""
