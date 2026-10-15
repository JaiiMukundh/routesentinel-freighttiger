"""Candidate detection for RouteSentinel (Phase 2).

Deterministic rule only: a route-week is a *candidate* iff it has enough
history and clears both rise thresholds. The LLM never decides this.

Thresholds come from config (``config.yaml``); the defaults below mirror
the exact definitions in AGENTS.md and are used when no config is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

DEFAULT_MIN_HISTORY = 4
DEFAULT_OWN_THRESHOLD = 0.07
DEFAULT_PEER_THRESHOLD = 0.10

MIN_HISTORY_KEYS = ("min_history", "min_n_prior", "min_prior_weeks", "n_prior_min")
OWN_THRESHOLD_KEYS = (
    "own_threshold",
    "own_pct_threshold",
    "own_history_threshold",
    "own_pct_min",
)
PEER_THRESHOLD_KEYS = (
    "peer_threshold",
    "peer_pct_threshold",
    "similar_routes_threshold",
    "peer_pct_min",
)
NESTED_CONFIG_KEYS = ("detection", "thresholds", "candidate")

REQUIRED_COLUMNS = ("route", "week_of", "n_prior", "own_pct", "peer_pct")
COST_COLUMNS = ("cost_per_tonne_km", "cpt")


def _get_threshold(cfg: Any, keys: tuple[str, ...], default: float) -> float:
    """Read a numeric threshold from a config mapping/object.

    Accepts a plain mapping, an object with attributes, or ``None``
    (→ default). Also looks one level down under well-known nested
    section names so this stays compatible with whatever shape
    ``config.yaml`` loading produces.
    """
    if cfg is None:
        return default
    scopes: list[Any] = [cfg]
    if isinstance(cfg, Mapping):
        for section in NESTED_CONFIG_KEYS:
            nested = cfg.get(section)
            if isinstance(nested, Mapping):
                scopes.append(nested)
    for scope in scopes:
        for key in keys:
            if isinstance(scope, Mapping):
                if key in scope:
                    return float(scope[key])
            elif hasattr(scope, key):
                return float(getattr(scope, key))
    return default


def resolve_thresholds(cfg: Any = None) -> tuple[int, float, float]:
    """Return ``(min_history, own_threshold, peer_threshold)`` for a config."""
    min_history = int(_get_threshold(cfg, MIN_HISTORY_KEYS, float(DEFAULT_MIN_HISTORY)))
    own_threshold = _get_threshold(cfg, OWN_THRESHOLD_KEYS, DEFAULT_OWN_THRESHOLD)
    peer_threshold = _get_threshold(cfg, PEER_THRESHOLD_KEYS, DEFAULT_PEER_THRESHOLD)
    return min_history, own_threshold, peer_threshold


def is_candidate(
    n_prior: Any,
    own_pct: Any,
    peer_pct: Any,
    cfg: Any = None,
) -> bool:
    """Return True iff one route-week meets the candidate rule.

    Candidate iff ``n_prior >= min_history`` AND ``own_pct >= own_threshold``
    AND ``peer_pct >= peer_threshold``. Missing/NaN inputs never qualify.
    """
    min_history, own_threshold, peer_threshold = resolve_thresholds(cfg)
    try:
        if pd.isna(n_prior) or pd.isna(own_pct) or pd.isna(peer_pct):
            return False
        return bool(
            float(n_prior) >= min_history
            and float(own_pct) >= own_threshold
            and float(peer_pct) >= peer_threshold
        )
    except (TypeError, ValueError):
        return False


def find_candidates(weekly_df: pd.DataFrame, cfg: Any = None) -> pd.DataFrame:
    """Filter weekly metrics to candidate rows, sorted by ``(week_of, route)``.

    Expects per-(route, week) metrics columns (``route``, ``week_of``,
    ``n_prior``, ``own_pct``, ``peer_pct`` plus a cost column). The input
    frame is never mutated.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in weekly_df.columns]
    if missing:
        raise ValueError(f"weekly_df is missing required columns: {missing}")
    if not any(c in weekly_df.columns for c in COST_COLUMNS):
        raise ValueError(f"weekly_df needs one cost column among: {COST_COLUMNS}")

    min_history, own_threshold, peer_threshold = resolve_thresholds(cfg)
    n_prior = pd.to_numeric(weekly_df["n_prior"], errors="coerce")
    own_pct = pd.to_numeric(weekly_df["own_pct"], errors="coerce")
    peer_pct = pd.to_numeric(weekly_df["peer_pct"], errors="coerce")
    mask = (n_prior >= min_history) & (own_pct >= own_threshold) & (peer_pct >= peer_threshold)
    out = weekly_df.loc[mask.fillna(False)].copy()
    out = out.sort_values(["week_of", "route"], kind="mergesort").reset_index(drop=True)
    return out
