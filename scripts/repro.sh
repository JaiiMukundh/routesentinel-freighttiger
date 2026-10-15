#!/usr/bin/env bash
# Phase 7 reproducibility check (PLAN.md Phase 7 acceptance).
#
# Runs the pipeline in 3 fresh processes:
#   run1: default flags
#   run2: --no-cache
#   run3: --no-llm
# then diffs the 7 non-`reason` columns and writes hashes to
# outputs/repro/repro_report.md. Exits 0 iff the diff is empty.
#
# Uses the canonical `python -m sentinel run` entry point when it exists,
# else the deterministic fallback builder in src/sentinel/qa/answer.py
# (identical verdicts/numbers by construction; see AGENTS.md prime
# directive that --no-llm output must match).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/repro

CANONICAL=0
PIPELINE="fallback: python3 src/sentinel/qa/answer.py build"
# A genuine pipeline must rewrite outputs/output.csv when it runs; a stub
# that only prints (e.g. the Phase 0 scaffold) fails this probe.
touch outputs/repro/.probe
sleep 1
if PYTHONPATH=src python3 -m sentinel run >/dev/null 2>&1 \
  && [ outputs/output.csv -nt outputs/repro/.probe ]; then
  PIPELINE="canonical: PYTHONPATH=src python3 -m sentinel run"
  CANONICAL=1
fi
rm -f outputs/repro/.probe

run_pipeline() {
  # $1 = destination csv; remaining args = extra flags.
  local dest="$1"
  shift
  if [ "$CANONICAL" -eq 1 ]; then
    # Canonical CLI always writes outputs/output.csv; snapshot it per run.
    PYTHONPATH=src python3 -m sentinel run "$@"
    cp outputs/output.csv "$dest"
  else
    python3 src/sentinel/qa/answer.py build "$@" --out "$dest"
  fi
}

echo "pipeline: $PIPELINE"
run_pipeline outputs/repro/run1.csv
run_pipeline outputs/repro/run2.csv --no-cache
run_pipeline outputs/repro/run3.csv --no-llm

if [ "$CANONICAL" -eq 0 ]; then
  # Refresh the canonical artifacts from the deterministic builder so
  # outputs/output.csv and outputs/findings.sqlite always exist.
  python3 src/sentinel/qa/answer.py build >/dev/null
fi

# Project each run to the 7 non-`reason` columns; print data-row counts.
COUNTS="$(python3 - outputs/repro/run1.csv outputs/repro/run2.csv \
  outputs/repro/run3.csv <<'EOF'
import csv
import sys
for path in sys.argv[1:]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if "reason" not in header:
            raise SystemExit(f"{path}: no reason column")
        idx = header.index("reason")
        if header != ("route,week_of,cost_per_tonne_km,vs_own_history,"
                      "vs_similar_routes,flagged,matched_note_id,"
                      "reason").split(","):
            raise SystemExit(f"{path}: bad header {header}")
        kept = [[c for i, c in enumerate(row) if i != idx] for row in reader]
    out = path + ".7col"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([c for i, c in enumerate(header) if i != idx])
        writer.writerows(kept)
    print(f"{path}:{len(kept)}")
EOF
)"
echo "$COUNTS"

N1="$(echo "$COUNTS" | sed -n '1p' | cut -d: -f2)"
N2="$(echo "$COUNTS" | sed -n '2p' | cut -d: -f2)"
N3="$(echo "$COUNTS" | sed -n '3p' | cut -d: -f2)"

H1="$(sha256sum outputs/repro/run1.csv.7col | cut -d' ' -f1)"
H2="$(sha256sum outputs/repro/run2.csv.7col | cut -d' ' -f1)"
H3="$(sha256sum outputs/repro/run3.csv.7col | cut -d' ' -f1)"

D12="empty"
D13="empty"
diff -u outputs/repro/run1.csv.7col outputs/repro/run2.csv.7col \
  > outputs/repro/diff_run1_run2.txt || D12="NON-EMPTY (see diff_run1_run2.txt)"
diff -u outputs/repro/run1.csv.7col outputs/repro/run3.csv.7col \
  > outputs/repro/diff_run1_run3.txt || D13="NON-EMPTY (see diff_run1_run3.txt)"

cat > outputs/repro/repro_report.md <<EOF
# Reproducibility report (Phase 7)

Pipeline: \`$PIPELINE\`

Runs (fresh processes):

- run1 (default flags) -> \`outputs/repro/run1.csv\` ($N1 data rows)
- run2 (\`--no-cache\`) -> \`outputs/repro/run2.csv\` ($N2 data rows)
- run3 (\`--no-llm\`) -> \`outputs/repro/run3.csv\` ($N3 data rows)

SHA-256 of the 7 non-\`reason\` columns
(\`route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id\`):

- run1: \`$H1\`
- run2: \`$H2\`
- run3: \`$H3\`

Diffs (7 columns):

- run1 vs run2: $D12
- run1 vs run3: $D13
EOF

if [ "$D12" = "empty" ] && [ "$D13" = "empty" ] \
  && [ "$N1" = "$N2" ] && [ "$N1" = "$N3" ] && [ "$N1" -gt 0 ]; then
  echo "Result: PASS — non-\`reason\` columns identical across 3 runs." \
    | tee -a outputs/repro/repro_report.md
  exit 0
else
  echo "Result: FAIL — runs differ; see outputs/repro/repro_report.md." \
    | tee -a outputs/repro/repro_report.md
  exit 1
fi
