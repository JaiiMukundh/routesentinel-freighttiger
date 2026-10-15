# Reproducibility report (Phase 7)

Pipeline: `canonical: PYTHONPATH=src python3 -m sentinel run`

Runs (fresh processes):

- run1 (default flags) -> `outputs/repro/run1.csv` (13 data rows)
- run2 (`--no-cache`) -> `outputs/repro/run2.csv` (13 data rows)
- run3 (`--no-llm`) -> `outputs/repro/run3.csv` (13 data rows)

SHA-256 of the 7 non-`reason` columns
(`route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id`):

- run1: `289b17a91744846ff9e9dbf612db28df4e7e3dedf6d8e60211d6fdd3c4cfa47f`
- run2: `289b17a91744846ff9e9dbf612db28df4e7e3dedf6d8e60211d6fdd3c4cfa47f`
- run3: `289b17a91744846ff9e9dbf612db28df4e7e3dedf6d8e60211d6fdd3c4cfa47f`

Diffs (7 columns):

- run1 vs run2: empty
- run1 vs run3: empty
Result: PASS — non-`reason` columns identical across 3 runs.
