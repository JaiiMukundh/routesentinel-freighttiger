# SUBMISSION CHECKLIST — RouteSentinel release (2026-10-15)

## 1. VERIFY — PASS
- `make lint` PASS (ruff + format + mypy strict, 23 files)
- `make test` PASS — 88 passed
- `make eval` PASS (exit 0) — silver 13/13 (11/11 ex-debatable), hybrid Recall@4 1.0/MRR 1.0, red-team 9/9
- `make repro` PASS — 3 runs, 7 non-`reason` cols identical
- `make cost` PASS — `outputs/token_cost_log.json` has `calls`, `input_tokens`, `output_tokens`, `local_cost: 0`, `hosted_rates: "FILL_BY_HUMAN"`
- Header of `outputs/output.csv` == `fixtures/sample_output_format_v2.csv` header exactly
- `flagged` ∈ {`Yes`, `No (justified)`} only; `matched_note_id` empty iff `Yes` (13/13 rows)
- Note: option B (user-approved 2026-10-15): kept the live local SLM; adapted 2 typing
  lines in `src/sentinel/llm/client.py` (no behavior change — `cast(Any, …).chat`, same
  timeout-then-fallback flow) and 1 test (`test_generate_without_ollama…` now pins
  dead port 127.0.0.1:9 via `OLLAMA_HOST`, deterministic with or without a local model).
  No pipeline logic, thresholds, verdicts, or pass-criteria changed.

## 2. CLEAN — PASS
- `.gitignore` added (venv, pycache, caches, weights, env, DS_Store, node_modules, dist)
- `docs/` holds AGENTS.md + PLAN.md (SPEC.md never existed in repo — referenced only); NOTES.md kept at root
- Secrets scan: no API keys/tokens/.env; no absolute local paths or personal info in code
- Files >10 MB: none in repo (model weights live outside repo under /tmp; `web/node_modules` excluded)

## 3. README CHECK — PASS
- What it does ✓ · quickstart (4 commands) ✓ · exact definitions ✓ · AI used/not-used ✓
  (incl. live-SLM deviation note) · gate table N001–N010 ✓ · eval scorecard ✓ ·
  repro result ✓ · token/cost table (cold-cache zeros, live numbers cited to NOTES.md) ✓ ·
  assumptions/limitations ✓ · March-10/17 judgement + v1/v2 note + `<8 weeks` behavior ✓
- Nothing missing; no results invented.

## 4. FRESH-CLONE TEST — PASS
```bash
git clone /home/jaiimukundha/Desktop/freight_tiger /tmp/opencode/freshclone
cd /tmp/opencode/freshclone
python3 -m venv --without-pip .venv   # system lacks python3-venv, no sudo
. .venv/bin/activate
python /tmp/opencode/get-pip.py       # bootstrapped (internet OK)
python -m pip install -e .
make run-nollm   # candidates=13 justified=3 flagged_yes=10 llm_calls=0
make test        # 88 passed
make repro       # PASS, non-reason columns identical
sha256sum outputs/output.csv
# 30e8a1d6c562896f7799b9c740ec6d6d8281e54edf5a3e05d61ee01b5b5c8e64 — identical to committed artifact
```

## 5. PUBLISH — PASS
- Repo: https://github.com/JaiiMukundh/routesentinel-freighttiger (public, confirmed by you)
- Pushed `main` clean (single commit `64f2d25`, no force-push); `git ls-remote` HEAD == `64f2d25`
- Fresh clone from URL verified — same commit, same output.csv hash (see above)
- Backup: `/home/jaiimukundha/Desktop/freight_tiger/freighttiger-casestudy.zip` (690 KB, no .git/venv/cache/models)
  + release asset: https://github.com/JaiiMukundh/routesentinel-freighttiger/releases/tag/v1.0.0

## 6. Manual items for you
1. Fill `hosted_rates` in `outputs/token_cost_log.json` from the provider pricing page
   (currently `"FILL_BY_HUMAN"` — do not invent prices).
2. Rehearse the 10-minute walkthrough (SPEC §15) + Q&A prep (SPEC §8) — human-only per PLAN.md.
3. Confirm the v2 output file name with HR (PLAN.md human task).
4. (Optional) If you want prettier LLM reason wording in the demo, pull Qwen2.5-1.5B —
   the 0.5B prose is verifier-safe but weak.
