# NOTES.md — RouteSentinel build log

## Phase 0 — Skeleton (2026-10-15)

- Created `config.yaml` (own_threshold=0.07, peer_threshold=0.10, min_history=4,
  max_history=8, absorption_weeks=8, grace_days=0, k_retrieval=4, rrf_k=60,
  model qwen2.5:3b-instruct-q4_K_M, temperature=0, top_k=1, seed=42, data/output
  paths), typed `src/sentinel/config.py` (`load_config`), `pyproject.toml`
  (python>=3.11 + allowed deps only, ruff+mypy config), `Makefile` (all 8 targets),
  typer CLI stub (`run [--no-llm] [--no-cache] [--bruteforce]`, prints
  "scaffold ok").
- Env note: system Python is 3.12 (AGENTS.md says 3.11); `requires-python >=3.11`
  covers both. ruff/mypy/pytest installed via pip (`--break-system-packages`);
  pandas/pydantic/typer/pyyaml were preinstalled. Heavy deps (duckdb, chromadb,
  fastembed, pandera, hypothesis, rank-bm25, ollama) are declared in pyproject but
  not installed yet — deferred to the phases that need them.
- `make lint` and `make test` (zero tests collected) pass. No fixture values
  hardcoded; nothing under `data/`/`fixtures/` touched.

## Phase 0 — verification update (2026-10-15)

- My files (`config.yaml`, `pyproject.toml`, `Makefile`, `src/sentinel/__init__.py`,
  `src/sentinel/__main__.py`, `src/sentinel/config.py`) are individually clean:
  `ruff check` + `ruff format --check` + `mypy --strict` all pass.
- Fixed two scaffold bugs found by execution: typer single-command dispatch meant
  `python -m sentinel run` was rejected, so an `eval` stub command was added (also
  gives `make eval` a real target); Makefile recipes now use `python3`/`python3 -m pip`
  since bare `python` does not exist in this environment.
- `make run`, `make run-nollm`, `make eval`, `make repro`, `make cost`, `make test`
  (23 passed — other workers' tests) all exit 0. `pip install -e . --no-deps` validates.
- Repo-wide `make lint` currently FAILS only on files outside my ownership, which I
  did not touch (parallel workers are actively editing; the set shifts between runs;
  at last check: B905 `zip()`-without-`strict` in `tests/test_candidates.py`; earlier
  also UP007 in `src/sentinel/notes/extract.py` and B905 in `tests/test_metrics.py` —
  the notes worker already fixed the UP007). Owned by the respective workers to fix.
- Env-only (not project deps, not added to pyproject): installed `types-PyYAML`
  and `pandas-stubs` so mypy can check the declared deps; heavy runtime deps
  (duckdb, chromadb, fastembed, rank-bm25, ollama, pandera, hypothesis) still
  intentionally not installed — deferred to the phases that need them.

## Phase 5 — LLM layer (2026-10-15)

- Files: `src/sentinel/llm/__init__.py` (re-exports), `src/sentinel/llm/client.py`
  (`OllamaClient`, model `qwen2.5:3b-instruct-q4_K_M`, temperature=0, top_k=1,
  seed=42, `generate(prompt, json_schema)` cached on sha256(prompt + model
  digest + params) under `outputs/.ollama_cache/`, one retry, raises
  `LLMUnavailable` on any outage — never paid APIs), `src/sentinel/llm/ledger.py`
  (token/call ledger at `outputs/token_cost_log.json`: calls, input/output
  tokens by whitespace split, cache_hits, fallbacks, fallback_rate, local_cost
  always 0; no timestamps), `src/sentinel/agent/explain.py`
  (`build_fact_sheet` allowlist-only facts, fact-sheet-only wording prompt with
  note-text-marked-untrusted, deterministic justified/unexplained/
  closest-rejected templates, word once + one retry then template),
  `src/sentinel/agent/verify.py` (claim verifier: every note id, ISO date,
  route, percentage, and 2-decimal cost figure must trace to the fact sheet),
  `tests/test_verify.py` (18 tests: stub emitting fake N011 + made-up cause
  falls back to template; verdict/numbers/ids identical LLM vs `--no-llm`).
- Ollama/model UNAVAILABLE in this environment (per AGENTS.md hard rule 5, no
  LLM output faked): `curl localhost:11434/api/tags` returns 000 (connection
  failed), and `python3 -c "import ollama"` raises ModuleNotFoundError. The
  `ollama` import is therefore deferred inside `OllamaClient._chat_once`, and
  pre-seeded cache entries are served without a server. All wording falls back
  silently to the deterministic template; the no-LLM path passes.
- `ruff check`, `ruff format --check`, `mypy --strict` clean on all 6 owned
  files; `pytest tests/test_verify.py` 18 passed; full suite 84 passed,
  1 skipped. `src/sentinel/__init__.py` holds only a docstring;
  `src/sentinel/agent/__init__.py` belongs to the gate worker (left untouched).

## Local SLM now live (user-requested, 2026-10-15)
- Real Ollama binary needs ~1.8GB + root; sandbox has neither. Instead:
  `scripts/slm_serve.py` (stdlib HTTP, Ollama-compatible `/api/chat`+`/api/tags`
  on 127.0.0.1:11434) serving Qwen2.5-0.5B-Instruct locally via
  torch 2.6/transformers 4.57 on CPU (greedy, seed 42). `pip install --user
  --break-system-packages ollama` provides the client; `ollama.chat()` works
  unchanged. No hosted/paid API touched.
- Deviation from AGENTS.md: 0.5B demo model instead of `qwen2.5:3b-instruct`;
  shim instead of Ollama binary. config.yaml keeps the canonical 3B default.
  Verdicts unaffected (LLM never decides).
- Live proof: 13/13 verdicts identical LLM vs --no-llm; 9 reasons model-worded
  (verifier-passed), 4 template fallbacks (verifier rejected, e.g. 0.5B wrote
  "approximately 10%" for a +29.5% row with no note id/date). Ledger now shows
  real usage: 14 calls, 1864 in / 1156 out tokens, fallback_rate 0.2, $0.
- Caveat: 0.5B prose is weak (verifier checks grounding, not eloquence). Next
  step if wanted: pull Qwen2.5-1.5B for better wording at ~1GB more disk.
