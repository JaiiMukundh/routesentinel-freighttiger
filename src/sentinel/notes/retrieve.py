"""Multi-query hybrid retrieval with RRF fusion over the note index.

For each candidate ``(route, week)`` three queries are built (SPEC 6.2):
1. ``"{route} cost increase {week}"`` — route + week anchor.
2. ``"{cause hints} {route}"`` — generic disruption causes + route.
3. Note-text keywords — freight vocabulary shared with the notes.

Each query is run through both BM25 and dense search (exact flat search
over all notes); the resulting rankings are fused with Reciprocal Rank
Fusion (``k=60``) and tie-broken by ``(-fused_score, note_id)``.

``retrieve(candidate, index, k=4)`` returns ranked note ids. The
``bruteforce`` flag scores all notes; at this scale (n=10) every search
is already exact over all notes, so the flag is a kept-for-compat no-op.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .index import NoteIndex, build_index, full_bm25_scores, full_dense_scores

RRF_K = 60

# Generic cause hints; retrieval only proposes, the gate disposes.
CAUSE_HINTS = "flooding festival diesel surcharge detour disruption demand toll"

# Freight vocabulary overlapping the note texts (third query).
NOTE_KEYWORDS = (
    "highway truck transport freight costs rates corridor demand "
    "flooding detour surcharge diesel disruption repairs stable normal"
)

CandidateLike = dict[str, Any] | tuple[str, str] | list[str] | Any


def candidate_route_week(candidate: CandidateLike) -> tuple[str, str]:
    """Extract ``(route, week)`` strings from a dict, pair, or object."""
    if isinstance(candidate, dict):
        route = str(candidate.get("route", ""))
        week = str(candidate.get("week_of", candidate.get("week", candidate.get("weekof", ""))))
        return route, week
    if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
        return str(candidate[0]), str(candidate[1])
    route = str(getattr(candidate, "route", ""))
    week = str(getattr(candidate, "week_of", getattr(candidate, "week", "")))
    return route, week


def build_queries(candidate: CandidateLike) -> list[str]:
    """Build the three SPEC 6.2 queries for a ``(route, week)`` candidate."""
    route, week = candidate_route_week(candidate)
    route_spaced = route.replace("-", " ")
    return [
        f"{route} cost increase {week}",
        f"{CAUSE_HINTS} {route_spaced} {route}",
        NOTE_KEYWORDS,
    ]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked id lists (ranks are 1-based)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, nid in enumerate(ranking, start=1):
            fused[nid] = fused.get(nid, 0.0) + 1.0 / (k + rank)
    return fused


def _full_ranking(scores: list[float], note_ids: list[str]) -> list[str]:
    order = sorted(range(len(note_ids)), key=lambda i: (-scores[i], note_ids[i]))
    return [note_ids[i] for i in order]


def retrieve(
    candidate: CandidateLike,
    index: NoteIndex,
    k: int = 4,
    rrf_k: int = RRF_K,
    bruteforce: bool = False,  # noqa: FBT001, FBT002 - kept flag, exact at this scale
) -> list[str]:
    """Return the top-``k`` note ids for a candidate, fused across systems.

    ``bruteforce`` scores all notes; search here is always exact over all
    notes, so the flag only documents that intent.
    """
    _ = bruteforce  # exact flat search always scores every note
    rankings: list[list[str]] = []
    for query in build_queries(candidate):
        rankings.append(_full_ranking(full_bm25_scores(index, query), index.note_ids))
        rankings.append(_full_ranking(full_dense_scores(index, query), index.note_ids))
    fused = rrf_fuse(rankings, k=rrf_k)
    ordered = sorted(fused, key=lambda nid: (-fused[nid], nid))
    return ordered[: max(k, 0)]


def retrieve_single_system(
    candidate: CandidateLike,
    index: NoteIndex,
    system: str,
    k: int = 4,
    rrf_k: int = RRF_K,
) -> list[str]:
    """Top-``k`` ids using one system only (``"bm25"`` or ``"dense"``).

    Used by the retrieval eval to report per-system Recall@4/MRR.
    """
    if system not in ("bm25", "dense"):
        raise ValueError(f"unknown system: {system}")
    scorer = full_bm25_scores if system == "bm25" else full_dense_scores
    rankings = [_full_ranking(scorer(index, q), index.note_ids) for q in build_queries(candidate)]
    fused = rrf_fuse(rankings, k=rrf_k)
    ordered = sorted(fused, key=lambda nid: (-fused[nid], nid))
    return ordered[: max(k, 0)]


def load_notes_csv(path: str | Path) -> list[dict[str, str]]:
    """Load ``[{id, text}]`` from a ``note_id,date,applies_to,note`` CSV."""
    notes: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            notes.append({"id": row["note_id"], "text": row["note"]})
    return notes


def main(argv: list[str] | None = None) -> int:
    """CLI: ``retrieve --route R --week W [--k 4] [--bruteforce]``."""
    parser = argparse.ArgumentParser(description="Retrieve notes for a route-week candidate")
    parser.add_argument("--route", required=True)
    parser.add_argument("--week", required=True, help="week_of YYYY-MM-DD (Monday)")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--bruteforce", action="store_true", help="score all notes (exact at this scale)"
    )
    parser.add_argument("--notes-csv", default="data/notes.csv")
    args = parser.parse_args(argv)
    index = build_index(load_notes_csv(args.notes_csv))
    query = {"route": args.route, "week_of": args.week}
    for nid in retrieve(query, index, k=args.k, bruteforce=args.bruteforce):
        print(nid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
