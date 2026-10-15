"""Weekly freight-cost metrics (Phase 1, no AI).

Definitions (AGENTS.md, exact):
- ``route = f"{origin}-{destination}"`` (directed).
- ``week_of = shipment_date - timedelta(days=shipment_date.weekday())``.
- ``cost_per_tonne_km`` per (route, week_of) = pooled ratio
  ``sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)``.
- own history: mean of the route's ``cpt`` over the previous up-to-8
  existing weeks strictly before the current week; ``n_prior`` counts
  those weeks; first week -> NaN. ``own_pct = cpt / hist - 1``.
- peers: unweighted mean of ``cpt`` of *other* routes with the same
  ``route_type`` in the same week; none -> NaN.
  ``peer_pct = cpt / peer - 1``.

Output sorted by ``(week_of, route)`` with explicit tie-breakers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.ingest import add_route_column, add_week_of_column

MAX_HISTORY_WEEKS = 8

WEEKLY_COLUMNS = [
    "route",
    "route_type",
    "week_of",
    "cpt",
    "n_prior",
    "own_pct",
    "peer_pct",
]


def _ensure_route_and_week(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "route" not in out.columns:
        out = add_route_column(out)
    if "week_of" not in out.columns:
        out = add_week_of_column(out)
    return out


def compute_pooled_cpt(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled ``cpt`` per (route, route_type, week_of), sorted deterministically."""
    src = _ensure_route_and_week(df)
    work = src.assign(
        _tonne_km=pd.to_numeric(src["quantity_tonnes"], errors="raise")
        * pd.to_numeric(src["distance_km"], errors="raise"),
    )
    agg = (
        work.groupby(["route", "route_type", "week_of"], sort=True)
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("_tonne_km", "sum"),
        )
        .reset_index()
    )
    agg["cpt"] = agg["total_cost"] / agg["total_tonne_km"]
    return agg[["route", "route_type", "week_of", "cpt"]]


def add_own_history(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add ``n_prior`` and ``own_pct`` using only strictly-prior weeks."""
    frames: list[pd.DataFrame] = []
    ordered = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)
    for _, group in ordered.groupby("route", sort=True):
        group = group.reset_index(drop=True)
        cpt = group["cpt"].to_numpy(dtype=float)
        n = len(cpt)
        hist = np.full(n, np.nan)
        n_prior = np.zeros(n, dtype=int)
        for i in range(n):
            lo = max(0, i - MAX_HISTORY_WEEKS)
            n_prior[i] = i - lo
            if i - lo > 0:
                hist[i] = float(np.mean(cpt[lo:i]))
        group = group.copy()
        group["n_prior"] = n_prior
        with np.errstate(divide="ignore", invalid="ignore"):
            group["own_pct"] = cpt / hist - 1.0
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def add_peer_comparison(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add ``peer_pct`` vs other same-type routes in the same week."""
    out = weekly.copy()
    peers = out.groupby(["route_type", "week_of"], sort=True)["cpt"]
    total = peers.transform("sum")
    count = peers.transform("size")
    peer_mean = (total - out["cpt"]) / (count - 1).replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["peer_pct"] = out["cpt"] / peer_mean - 1.0
    return out


def compute_weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: pooled cpt -> own history -> peers, sorted output."""
    weekly = compute_pooled_cpt(df)
    weekly = add_own_history(weekly)
    weekly = add_peer_comparison(weekly)
    weekly["n_prior"] = weekly["n_prior"].astype(int)
    return weekly.sort_values(["week_of", "route"]).reset_index(drop=True)[WEEKLY_COLUMNS]
