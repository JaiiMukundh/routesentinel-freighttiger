"""Retrieval eval: Recall@4 and MRR for bm25 / dense / hybrid.

Gold mapping (labelled rows):
- Chennai-Bangalore 2025-02-24 -> N001
- Chennai-Bangalore 2025-03-03 -> N001
- Ahmedabad-Mumbai 2025-01-20 -> N002

Acceptance: hybrid returns each gold note in top-4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinel.notes.index import build_index  # noqa: E402
from sentinel.notes.retrieve import load_notes_csv, retrieve, retrieve_single_system  # noqa: E402

LABELS: list[tuple[dict[str, str], str]] = [
    ({"route": "Chennai-Bangalore", "week_of": "2025-02-24"}, "N001"),
    ({"route": "Chennai-Bangalore", "week_of": "2025-03-03"}, "N001"),
    ({"route": "Ahmedabad-Mumbai", "week_of": "2025-01-20"}, "N002"),
]

SYSTEMS = ("bm25", "dense", "hybrid")


@pytest.fixture(scope="module")
def index():
    return build_index(load_notes_csv(ROOT / "data" / "notes.csv"))


def rank_of(ranking: list[str], gold: str) -> int | None:
    try:
        return ranking.index(gold) + 1
    except ValueError:
        return None


def evaluate(index, system: str, k: int = 4) -> dict[str, float]:
    recalls: list[float] = []
    rrs: list[float] = []
    for candidate, gold in LABELS:
        n = len(index)
        if system == "hybrid":
            topk = retrieve(candidate, index, k=k)
            full = retrieve(candidate, index, k=n)
        else:
            topk = retrieve_single_system(candidate, index, system, k=k)
            full = retrieve_single_system(candidate, index, system, k=n)
        recalls.append(1.0 if gold in topk else 0.0)
        rank = rank_of(full, gold)
        rrs.append(1.0 / rank if rank else 0.0)
    return {
        "recall@4": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
    }


def test_hybrid_returns_each_gold_in_top4(index, capsys):
    metrics: dict[str, dict[str, float]] = {s: evaluate(index, s) for s in SYSTEMS}
    with capsys.disabled():
        print("\nRetrieval metrics (3 labelled queries, k=4):")
        print(f"{'system':<8}{'recall@4':>10}{'mrr':>10}")
        for system in SYSTEMS:
            m = metrics[system]
            print(f"{system:<8}{m['recall@4']:>10.3f}{m['mrr']:>10.3f}")
    for candidate, gold in LABELS:
        top4 = retrieve(candidate, index, k=4)
        assert gold in top4, f"hybrid missed {gold} for {candidate}: {top4}"
    assert metrics["hybrid"]["recall@4"] == pytest.approx(1.0)


def test_recall_mrr_ranges(index):
    for system in SYSTEMS:
        m = evaluate(index, system)
        assert 0.0 <= m["recall@4"] <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0


def test_rankings_deterministic(index):
    for candidate, _ in LABELS:
        first = retrieve(candidate, index, k=4)
        assert retrieve(candidate, index, k=4) == first
        rebuilt = build_index(load_notes_csv(ROOT / "data" / "notes.csv"))
        assert retrieve(candidate, rebuilt, k=4) == first
