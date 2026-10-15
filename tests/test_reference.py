"""Reference cross-checks (Phase 1).

1. Pandas metrics must match ``fixtures/reference_weekly_metrics.csv``
   within 1e-6 on cpt/own_pct/peer_pct.
2. An independent DuckDB-SQL implementation must agree with pandas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.ingest import load_shipments  # noqa: E402
from sentinel.metrics import compute_weekly_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-6


def _pandas_metrics() -> pd.DataFrame:
    df = load_shipments(ROOT / "data" / "shipments.csv")
    out = compute_weekly_metrics(df)
    out = out.copy()
    out["week_of"] = out["week_of"].astype(str)
    return out


def test_matches_reference_fixture() -> None:
    got = _pandas_metrics()
    ref = pd.read_csv(ROOT / "fixtures" / "reference_weekly_metrics.csv", dtype={"week_of": str})

    assert len(got) == len(ref) == 728
    merged = got.merge(
        ref, on=["route", "week_of"], suffixes=("", "_ref"), how="outer", indicator=True
    )
    assert (merged["_merge"] == "both").all(), "route-week key sets differ"
    assert (merged["route_type"] == merged["route_type_ref"]).all()
    assert (merged["n_prior"] == merged["n_prior_ref"]).all()

    for col in ("cpt", "own_pct", "peer_pct"):
        a = merged[col].to_numpy(dtype=float)
        b = merged[f"{col}_ref"].to_numpy(dtype=float)
        assert (np.isnan(a) == np.isnan(b)).all(), f"NaN positions differ in {col}"
        mask = ~np.isnan(a)
        diff = np.abs(a[mask] - b[mask])
        assert diff.max() <= TOLERANCE, f"{col} max diff {diff.max()}"


SQL_WEEKLY_METRICS = """
WITH base AS (
    SELECT
        origin || '-' || destination AS route,
        route_type,
        (date_trunc('week', CAST(shipment_date AS DATE)))::DATE AS week_of,
        CAST(freight_cost_inr AS DOUBLE) AS cost,
        CAST(quantity_tonnes AS DOUBLE) * CAST(distance_km AS DOUBLE) AS tonne_km
    FROM shipments
),
weekly AS (
    SELECT route, route_type, week_of,
           SUM(cost) / SUM(tonne_km) AS cpt
    FROM base
    GROUP BY route, route_type, week_of
),
with_hist AS (
    SELECT
        route, route_type, week_of, cpt,
        COUNT(cpt) OVER w AS n_prior,
        AVG(cpt) OVER w AS hist
    FROM weekly
    WINDOW w AS (
        PARTITION BY route ORDER BY week_of
        ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
    )
)
SELECT
    route, route_type, week_of, cpt,
    CAST(n_prior AS INTEGER) AS n_prior,
    cpt / hist - 1 AS own_pct,
    cpt / (
        (SUM(cpt) OVER (PARTITION BY route_type, week_of) - cpt)
        / NULLIF(COUNT(*) OVER (PARTITION BY route_type, week_of) - 1, 0)
    ) - 1 AS peer_pct
FROM with_hist
ORDER BY week_of, route
"""


def test_duckdb_sql_agreement() -> None:
    raw = pd.read_csv(ROOT / "data" / "shipments.csv")
    got = compute_weekly_metrics(raw)
    with duckdb.connect() as con:
        con.register("shipments", raw)
        sql = con.execute(SQL_WEEKLY_METRICS).df()
    sql["week_of"] = pd.to_datetime(sql["week_of"]).dt.date
    sql = sql.sort_values(["week_of", "route"]).reset_index(drop=True)
    assert len(sql) == len(got) == 728
    assert (sql["route"] == got["route"]).all()
    assert (sql["route_type"] == got["route_type"]).all()
    assert (sql["week_of"] == got["week_of"]).all()
    assert (sql["n_prior"] == got["n_prior"]).all()
    for col in ("cpt", "own_pct", "peer_pct"):
        a = got[col].to_numpy(dtype=float)
        b = sql[col].to_numpy(dtype=float)
        assert (np.isnan(a) == np.isnan(b)).all(), f"NaN positions differ in {col}"
        mask = ~np.isnan(a)
        assert np.abs(a[mask] - b[mask]).max() <= 1e-9, f"{col} diverges"
