"""Synthetic anomaly-injection benchmark (Phase 6).

Scales the cost-per-tonne-km of a fixed-seed selection of eligible,
currently-non-candidate route-weeks by a magnitude grid, re-applies the
deterministic candidate rule, and reports recall vs magnitude plus the
background false-positive rate per 100 route-weeks.

Deterministic: target selection uses random.Random(seed) over an explicitly
sorted key list; seed comes from config.yaml (default 42). No timestamps.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from eval import FIXTURES, EvalConfig, load_config  # noqa: E402

MAGNITUDE_GRID: tuple[float, ...] = (1.05, 1.08, 1.12, 1.20, 1.35)
N_TARGETS = 40


@dataclass(frozen=True)
class InjectionResult:
    magnitude: float
    n_injected: int
    n_detected: int
    recall: float
    fp_per_100: float
    n_candidates_total: int


def load_weekly_metrics() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "reference_weekly_metrics.csv")
    return df.sort_values(["week_of", "route"]).reset_index(drop=True)


def is_candidate(df: pd.DataFrame, cfg: EvalConfig) -> pd.Series:
    return (
        (df["n_prior"] >= cfg.min_history)
        & (df["own_pct"] >= cfg.own_threshold)
        & (df["peer_pct"] >= cfg.peer_threshold)
    )


def select_targets(df: pd.DataFrame, cfg: EvalConfig, n: int = N_TARGETS) -> list[tuple[str, str]]:
    """Eligible (n_prior ok) non-candidate route-weeks, seeded fixed selection."""
    base = df[~is_candidate(df, cfg)].copy()
    base = base[base["n_prior"] >= cfg.min_history]
    keys = sorted(zip(base["route"].tolist(), base["week_of"].tolist(), strict=True))
    rng = random.Random(cfg.seed)
    return sorted(rng.sample(keys, min(n, len(keys))))


def apply_magnitude(
    df: pd.DataFrame, targets: list[tuple[str, str]], magnitude: float, cfg: EvalConfig
) -> pd.DataFrame:
    """Scale target cpts; own history uses strictly-prior (unscaled) weeks.

    own_pct' = m * (1 + own_pct) - 1  (hist unchanged, NaN stays NaN).
    Peer means are recomputed per (route_type, week) from scaled cpts.
    """
    out = df.copy()
    target_set = set(targets)
    mask = pd.Series(
        [(r, w) in target_set for r, w in zip(out["route"], out["week_of"], strict=True)]
    )
    out["cpt_scaled"] = out["cpt"].where(~mask, out["cpt"] * magnitude)
    out["own_scaled"] = (1.0 + out["own_pct"]) * magnitude - 1.0
    out.loc[~mask, "own_scaled"] = out.loc[~mask, "own_pct"]
    peer_mean = out.groupby(["route_type", "week_of"])["cpt_scaled"].transform("mean")
    peer_n = out.groupby(["route_type", "week_of"])["cpt_scaled"].transform("size")
    # Unweighted mean of *other* routes with the same route_type in the same week.
    peer_excl_self = (peer_mean * peer_n - out["cpt_scaled"]) / (peer_n - 1)
    out["peer_scaled"] = out["cpt_scaled"] / peer_excl_self - 1.0
    out.loc[peer_n <= 1, "peer_scaled"] = float("nan")
    flagged = (
        (out["n_prior"] >= cfg.min_history)
        & (out["own_scaled"] >= cfg.own_threshold)
        & (out["peer_scaled"] >= cfg.peer_threshold)
    )
    out["flagged"] = flagged.fillna(False)
    return out


def run_injection_benchmark(
    magnitudes: tuple[float, ...] = MAGNITUDE_GRID,
    cfg: EvalConfig | None = None,
) -> tuple[list[InjectionResult], list[tuple[str, str]]]:
    cfg = cfg or load_config()
    df = load_weekly_metrics()
    targets = select_targets(df, cfg)
    target_set = set(targets)
    results: list[InjectionResult] = []
    for mag in sorted(magnitudes):
        scaled = apply_magnitude(df, targets, mag, cfg)
        detected = sum(
            1
            for (r, w) in targets
            if bool(scaled[(scaled["route"] == r) & (scaled["week_of"] == w)].iloc[0]["flagged"])
        )
        non_target = scaled[
            [
                (r, w) not in target_set
                for r, w in zip(scaled["route"], scaled["week_of"], strict=True)
            ]
        ]
        fp = int(non_target["flagged"].sum())
        fp_per_100 = 100.0 * fp / max(1, len(non_target))
        results.append(
            InjectionResult(
                magnitude=mag,
                n_injected=len(targets),
                n_detected=detected,
                recall=detected / max(1, len(targets)),
                fp_per_100=fp_per_100,
                n_candidates_total=int(scaled["flagged"].sum()),
            )
        )
    return results, targets


def format_table(results: list[InjectionResult]) -> str:
    lines = [
        "| magnitude | injected | detected | recall | fp_per_100_route_weeks | total_candidates |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| x{r.magnitude:.2f} | {r.n_injected} | {r.n_detected} "
            f"| {r.recall:.3f} | {r.fp_per_100:.2f} | {r.n_candidates_total} |"
        )
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    results, targets = run_injection_benchmark(cfg=cfg)
    print(f"seed={cfg.seed} targets={len(targets)} (eligible non-candidate route-weeks)")
    print(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
