"""Phase 2: candidate-rule tests (AGENTS.md Definitions).

The (route, week_of) set produced by the rule must equal
fixtures/expected_candidates.csv exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.detect import find_candidates, is_candidate, resolve_thresholds

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
DATA = ROOT / "data"


def test_default_thresholds_match_definitions() -> None:
    assert resolve_thresholds(None) == (4, 0.07, 0.10)


def test_custom_config_thresholds_are_respected() -> None:
    cfg = {"min_history": 5, "own_threshold": 0.05, "peer_threshold": 0.20}
    assert resolve_thresholds(cfg) == (5, 0.05, 0.20)
    assert is_candidate(5, 0.06, 0.21, cfg) is True
    assert is_candidate(4, 0.06, 0.21, cfg) is False


def test_is_candidate_accepts_exact_boundaries() -> None:
    assert is_candidate(4, 0.07, 0.10) is True


def test_is_candidate_rejects_each_failing_condition() -> None:
    assert is_candidate(3, 0.07, 0.10) is False  # too little history
    assert is_candidate(4, 0.069999, 0.10) is False  # own rise too small
    assert is_candidate(4, 0.07, 0.099999) is False  # peer rise too small


def test_is_candidate_rejects_missing_values() -> None:
    nan = float("nan")
    assert is_candidate(nan, 0.5, 0.5) is False
    assert is_candidate(8, nan, 0.5) is False
    assert is_candidate(8, 0.5, nan) is False
    assert is_candidate(8, None, 0.5) is False
    assert is_candidate(None, 0.5, 0.5) is False


def test_find_candidates_match_expected_exactly() -> None:
    """Rule over the reference weekly metrics must yield the expected set."""
    reference = pd.read_csv(FIXTURES / "reference_weekly_metrics.csv")
    expected = pd.read_csv(FIXTURES / "expected_candidates.csv")

    candidates = find_candidates(reference)

    got = set(zip(candidates["route"].tolist(), candidates["week_of"].tolist(), strict=False))
    want = set(zip(expected["route"].tolist(), expected["week_of"].tolist(), strict=False))
    assert got == want
    assert len(candidates) == len(expected)


def test_find_candidates_sorted_by_week_then_route() -> None:
    reference = pd.read_csv(FIXTURES / "reference_weekly_metrics.csv")
    shuffled = reference.sample(frac=1.0, random_state=42)
    candidates = find_candidates(shuffled)
    keys = list(zip(candidates["week_of"].tolist(), candidates["route"].tolist(), strict=False))
    assert keys == sorted(keys)


def test_find_candidates_does_not_mutate_input() -> None:
    reference = pd.read_csv(FIXTURES / "reference_weekly_metrics.csv")
    before = reference.copy(deep=True)
    find_candidates(reference)
    pd.testing.assert_frame_equal(reference, before)


def test_find_candidates_missing_columns_raise() -> None:
    with pytest.raises(ValueError):
        find_candidates(pd.DataFrame({"route": ["A-B"]}))
    with pytest.raises(ValueError):
        find_candidates(
            pd.DataFrame(
                {
                    "route": ["A-B"],
                    "week_of": ["2025-01-06"],
                    "n_prior": [8],
                    "own_pct": [0.5],
                    "peer_pct": [0.5],
                }
            )
        )


def test_source_contains_no_fixture_values() -> None:
    """Hard rule: no value from the fixtures may be hardcoded in source."""
    expected = pd.read_csv(FIXTURES / "expected_candidates.csv")
    sources = [
        (ROOT / "src" / "sentinel" / "detect.py").read_text(),
        (ROOT / "src" / "sentinel" / "output.py").read_text(),
    ]
    for route in expected["route"].unique().tolist():
        for text in sources:
            assert route not in text, f"fixture route {route!r} hardcoded in source"


def test_find_candidates_on_live_metrics_if_available() -> None:
    """End-to-end wiring check: real metrics → same candidate set.

    Skips until the parallel metrics workstream exposes a recognised
    entry point; the reference-metrics test above is the binding check.
    """
    try:
        from sentinel import metrics as metrics_mod  # type: ignore[import]
    except ImportError:
        pytest.skip("sentinel.metrics not implemented yet (parallel workstream)")
        return
    for name in ("compute_weekly_metrics", "weekly_metrics", "compute_metrics"):
        fn = getattr(metrics_mod, name, None)
        if callable(fn):
            break
    else:
        pytest.skip("sentinel.metrics has no recognised entry point yet")
        return
    import inspect

    sig = inspect.signature(fn)
    shipments_path = DATA / "shipments.csv"
    if not shipments_path.exists():
        pytest.skip("data/shipments.csv not present")
        return
    try:
        from sentinel.ingest import load_shipments  # type: ignore[import]
    except ImportError:
        shipments_df = pd.read_csv(shipments_path)
    else:
        shipments_df = load_shipments(shipments_path)
    try:
        weekly = fn(shipments_df) if len(sig.parameters) > 0 else fn()
    except TypeError:
        pytest.skip("sentinel.metrics entry point signature not yet stable")
        return
    from sentinel.output import format_week

    expected = pd.read_csv(FIXTURES / "expected_candidates.csv")
    candidates = find_candidates(weekly)
    got = set(
        zip(
            candidates["route"].tolist(),
            [format_week(w) for w in candidates["week_of"].tolist()],
            strict=False,
        )
    )
    want = set(zip(expected["route"].tolist(), expected["week_of"].tolist(), strict=False))
    assert got == want
