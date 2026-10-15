"""Ingest and validate raw shipment records (Phase 1, no AI).

Pure functions: validate values, derive ``route`` and ``week_of`` columns.
The LLM is never involved here; deterministic code only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ALLOWED_ROUTE_TYPES = frozenset({"Short", "Medium", "Long"})

REQUIRED_COLUMNS = (
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
)


def add_route_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive directed ``route`` as ``f"{origin}-{destination}"``."""
    out = df.copy()
    origin = out["origin"].astype("string")
    dest = out["destination"].astype("string")
    out["route"] = (origin + "-" + dest).astype(str)
    return out


def add_week_of_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Monday ``week_of`` from ``shipment_date``.

    ``week_of = shipment_date - timedelta(days=shipment_date.weekday())``.
    Stored as :class:`datetime.date`.
    """
    out = df.copy()
    dates = pd.to_datetime(out["shipment_date"], format="%Y-%m-%d", errors="raise")
    monday = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    out["week_of"] = monday.dt.date
    return out


def validate_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """Validate shipment values; raise :class:`ValueError` on violation.

    Checks: required columns present, no nulls in key columns, strictly
    positive ``quantity_tonnes`` / ``distance_km`` / ``freight_cost_inr``,
    ``route_type`` in ``{Short, Medium, Long}``, and ``route_type`` unique
    per route. Logs row counts via ``print``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    key_cols = [
        "origin",
        "destination",
        "route_type",
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
        "shipment_date",
    ]
    null_counts = {c: int(df[c].isna().sum()) for c in key_cols if df[c].isna().any()}
    if null_counts:
        raise ValueError(f"null values in key columns: {null_counts}")

    numeric_cols = ["quantity_tonnes", "distance_km", "freight_cost_inr"]
    out = df.copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="raise")
        n_bad = int((out[col] <= 0).sum())
        if n_bad:
            raise ValueError(f"column {col!r} has {n_bad} non-positive values")

    bad_types = sorted(set(out["route_type"].astype(str)) - ALLOWED_ROUTE_TYPES)
    if bad_types:
        raise ValueError(f"invalid route_type values: {bad_types}")

    if "route" not in out.columns:
        out = add_route_column(out)
    multi = out.groupby("route")["route_type"].nunique()
    offenders = sorted(multi[multi > 1].index.tolist())
    if offenders:
        raise ValueError(f"route_type not unique per route: {offenders}")

    print(
        f"[ingest] rows: {len(out)}, "
        f"routes: {out['route'].nunique()}, "
        f"shipments date range: {out['shipment_date'].min()}..{out['shipment_date'].max()}"
    )
    return out


def load_shipments(path: str | Path) -> pd.DataFrame:
    """Read, validate and enrich shipments from a CSV file."""
    df = pd.read_csv(path)
    print(f"[ingest] read {len(df)} rows from {path}")
    df = validate_shipments(df)
    df = add_week_of_column(df)
    n_weeks = df.groupby(["route", "week_of"]).ngroups
    print(f"[ingest] validated {len(df)} rows across {n_weeks} route-weeks")
    return df
