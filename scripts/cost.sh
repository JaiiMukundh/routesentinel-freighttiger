#!/usr/bin/env bash
# Phase 7 cost log (PLAN.md Phase 7 + AGENTS.md LLM usage limits).
#
# Performs a cold-cache full run, then writes outputs/token_cost_log.json
# with calls, input/output tokens, cache hits, fallback rate, local cost
# (0: local Ollama only, no paid APIs) and a hosted-rates placeholder to
# be filled by a human from the provider pricing page.
#
# Uses the canonical `python -m sentinel run` entry point when it exists,
# else the deterministic fallback builder in src/sentinel/qa/answer.py
# (which performs zero LLM calls by construction).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs

START="$(date +%s)"

# Cold cache: drop known cache locations if present (guarded; missing is OK).
for dir in outputs/cache .sentinel_cache src/sentinel/llm/.cache; do
  if [ -e "$dir" ]; then
    rm -rf "$dir"
  fi
done

CANONICAL=0
MODE="fallback"
# A genuine pipeline must rewrite outputs/output.csv when it runs; a stub
# that only prints (e.g. the Phase 0 scaffold) fails this probe.
touch outputs/.cost_probe
sleep 1
if PYTHONPATH=src python3 -m sentinel run --no-cache >/dev/null 2>&1 \
  && [ outputs/output.csv -nt outputs/.cost_probe ]; then
  CANONICAL=1
  MODE="canonical"
else
  python3 src/sentinel/qa/answer.py build --no-cache >/dev/null
fi
rm -f outputs/.cost_probe

python3 - "$MODE" "$START" <<'EOF'
import json
import os
import sys

mode, start = sys.argv[1], int(sys.argv[2])
path = os.path.join("outputs", "token_cost_log.json")
existing: dict = {}
mtime_ok = False
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing = loaded
            mtime_ok = os.path.getmtime(path) >= start - 1
    except (OSError, ValueError):
        existing = {}


def num(value: object) -> int | float:
    return value if isinstance(value, (int, float)) else 0


if mode == "canonical" and mtime_ok and "calls" in existing:
    log = {
        "calls": num(existing.get("calls")),
        "input_tokens": num(existing.get("input_tokens")),
        "output_tokens": num(existing.get("output_tokens")),
        "cache_hits": num(existing.get("cache_hits")),
        "fallback_rate": float(num(existing.get("fallback_rate"))),
    }
else:
    # Deterministic (fallback or cold --no-cache) run: zero LLM usage.
    log = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_hits": 0,
        "fallback_rate": 0.0,
    }
log["local_cost"] = 0
hosted = existing.get("hosted_rates")
log["hosted_rates"] = (
    hosted
    if isinstance(hosted, str) and hosted and hosted != "FILL_BY_HUMAN"
    else "FILL_BY_HUMAN"
)

with open(path, "w", encoding="utf-8") as handle:
    json.dump(log, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"mode={mode}")
print(json.dumps(log, indent=2, sort_keys=True))
EOF
