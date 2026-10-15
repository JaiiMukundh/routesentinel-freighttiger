"""Unit tests for ingest validation and weekly metrics (Phase 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.ingest import add_week_of_column, validate_shipments  # noqa: E402
from sentinel.metrics import compute_weekly_metrics  # noqa: E402


def _row(
    shipment_id: str = "S1",
    origin: str = "Mumbai",
    destination: str = "Pune",
    route_type: str = "Short",
    quantity_tonnes: float = 10.0,
    distance_km: float = 100.0,
    freight_cost_inr: float = 3000.0,
    shipment_date: str = "2025-01-06",
) -> dict[str, object]:
    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "route_type": route_type,
        "material": "Steel",
        "quantity_tonnes": quantity_tonnes,
        "distance_km": distance_km,
        "freight_cost_inr": freight_cost_inr,
        "shipment_date": shipment_date,
        "transporter": "T1",
    }


def _mondays(n: int, start: str = "2025-01-06") -> list[str]:
    base = pd.Timestamp(start)
    return [(base + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(n)]


# --- week bucketing ---


def test_week_boundaries_sunday_vs_monday() -> None:
    df = pd.DataFrame(
        [_row("S1", shipment_date="2025-01-05"), _row("S2", shipment_date="2025-01-06")]
    )
    out = add_week_of_column(df)
    sunday_week = out.loc[out["shipment_id"] == "S1", "week_of"].iloc[0]
    monday_week = out.loc[out["shipment_id"] == "S2", "week_of"].iloc[0]
    assert sunday_week != monday_week
    assert str(sunday_week) == "2024-12-30"
    assert str(monday_week) == "2025-01-06"


def test_week_bucket_monday_2024_12_30() -> None:
    df = pd.DataFrame(
        [
            _row("S1", shipment_date="2024-12-30"),  # Monday -> itself
            _row("S2", shipment_date="2024-12-31"),
            _row("S3", shipment_date="2025-01-05"),  # Sunday -> same Monday
        ]
    )
    out = add_week_of_column(df)
    assert set(map(str, out["week_of"])) == {"2024-12-30"}


# --- pooled ratio ---


def test_pooled_ratio_not_mean_of_ratios() -> None:
    # A: 3000 / (10*100) = 3.0 ; B: 5000 / (20*100) = 2.5
    # pooled = 8000/3000 = 2.6667 ; mean-of-ratios = 2.75
    df = pd.DataFrame(
        [
            _row("S1", quantity_tonnes=10.0, freight_cost_inr=3000.0),
            _row("S2", quantity_tonnes=20.0, freight_cost_inr=5000.0),
        ]
    )
    weekly = compute_weekly_metrics(df)
    assert len(weekly) == 1
    assert weekly["cpt"].iloc[0] == pytest.approx(8000.0 / 3000.0)
    assert weekly["cpt"].iloc[0] != pytest.approx(2.75)


# --- own history ---


def test_first_week_nan() -> None:
    df = pd.DataFrame([_row("S1", shipment_date=d) for d in _mondays(3)])
    weekly = compute_weekly_metrics(df)
    first = weekly.iloc[0]
    assert first["n_prior"] == 0
    assert pd.isna(first["own_pct"])


def test_fewer_than_8_priors() -> None:
    df = pd.DataFrame([_row(f"S{i}", shipment_date=d) for i, d in enumerate(_mondays(5))])
    weekly = compute_weekly_metrics(df)
    assert weekly["n_prior"].tolist() == [0, 1, 2, 3, 4]


def test_history_capped_at_8() -> None:
    costs = [3000.0 + 100.0 * i for i in range(10)]
    df = pd.DataFrame(
        [
            _row(f"S{i}", shipment_date=d, freight_cost_inr=c)
            for i, (d, c) in enumerate(zip(_mondays(10), costs, strict=True))
        ]
    )
    weekly = compute_weekly_metrics(df)
    assert weekly["n_prior"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    cpts = weekly["cpt"].to_numpy()
    assert weekly["own_pct"].iloc[9] == pytest.approx(cpts[9] / cpts[1:9].mean() - 1)


def test_no_lookahead() -> None:
    df = pd.DataFrame([_row(f"S{i}", shipment_date=d) for i, d in enumerate(_mondays(5))])
    before = compute_weekly_metrics(df)
    perturbed = df.copy()
    perturbed.loc[perturbed["shipment_date"] == _mondays(5)[-1], "freight_cost_inr"] *= 3
    after = compute_weekly_metrics(perturbed)
    # Prior-week rows are byte-identical; history does not see the future.
    pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4])
    # The perturbed week's own history equals the hand-computed prior mean.
    prior_mean = before["cpt"].iloc[:4].mean()
    assert after["own_pct"].iloc[4] == pytest.approx(after["cpt"].iloc[4] / prior_mean - 1)


# --- peers ---


def test_peer_excludes_self() -> None:
    df = pd.DataFrame(
        [
            _row("S1", origin="Mumbai", destination="Pune", freight_cost_inr=3000.0),
            _row("S2", origin="Delhi", destination="Jaipur", freight_cost_inr=6000.0),
        ]
    )
    weekly = compute_weekly_metrics(df)
    pune = weekly.loc[weekly["route"] == "Mumbai-Pune"].iloc[0]
    jaipur = weekly.loc[weekly["route"] == "Delhi-Jaipur"].iloc[0]
    # Peer of each route is exactly the OTHER route's cpt.
    assert pune["peer_pct"] == pytest.approx(pune["cpt"] / jaipur["cpt"] - 1)
    assert jaipur["peer_pct"] == pytest.approx(jaipur["cpt"] / pune["cpt"] - 1)


def test_single_route_type_week_peer_nan() -> None:
    df = pd.DataFrame(
        [
            _row("S1", origin="Mumbai", destination="Pune", route_type="Short"),
            _row("S2", origin="Delhi", destination="Chennai", route_type="Long"),
        ]
    )
    weekly = compute_weekly_metrics(df)
    assert weekly["peer_pct"].isna().all()


# --- determinism / shuffle invariance ---


def test_row_shuffle_invariance_fixed_seed() -> None:
    rows = [_row(f"S{i}", shipment_date=d) for i, d in enumerate(_mondays(6))]
    rows += [
        _row(f"T{i}", origin="Delhi", destination="Jaipur", shipment_date=d)
        for i, d in enumerate(_mondays(6))
    ]
    df = pd.DataFrame(rows)
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    pd.testing.assert_frame_equal(compute_weekly_metrics(df), compute_weekly_metrics(shuffled))


@given(st.permutations(list(range(6))))
@settings(max_examples=25)
def test_row_shuffle_invariance_hypothesis(perm: list[int]) -> None:
    rows = [_row(f"S{i}", shipment_date=d) for i, d in enumerate(_mondays(6))]
    df = pd.DataFrame(rows)
    shuffled = df.iloc[list(perm)].reset_index(drop=True)
    pd.testing.assert_frame_equal(compute_weekly_metrics(df), compute_weekly_metrics(shuffled))


# --- ingest validation ---


def test_validate_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        validate_shipments(pd.DataFrame([_row(quantity_tonnes=0.0)]))
    with pytest.raises(ValueError, match="non-positive"):
        validate_shipments(pd.DataFrame([_row(distance_km=-1.0)]))
    with pytest.raises(ValueError, match="non-positive"):
        validate_shipments(pd.DataFrame([_row(freight_cost_inr=0.0)]))


def test_validate_rejects_bad_route_type() -> None:
    with pytest.raises(ValueError, match="invalid route_type"):
        validate_shipments(pd.DataFrame([_row(route_type="Tiny")]))


def test_validate_rejects_split_route_type() -> None:
    df = pd.DataFrame([_row("S1", route_type="Short"), _row("S2", route_type="Medium")])
    with pytest.raises(ValueError, match="not unique per route"):
        validate_shipments(df)


def test_validate_route_directed() -> None:
    df = pd.DataFrame(
        [
            _row("S1", origin="A", destination="B", route_type="Short"),
            _row("S2", origin="B", destination="A", route_type="Short"),
        ]
    )
    out = validate_shipments(df)
    assert sorted(out["route"].unique().tolist()) == ["A-B", "B-A"]
