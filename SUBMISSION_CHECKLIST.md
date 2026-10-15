# SUBMISSION CHECKLIST — RouteSentinel release (2026-10-15)

Status: **BLOCKED at VERIFY — stopped per instructions, no further steps run.**

## 1. VERIFY

| Check | Result |
|---|---|
| `make lint` | **FAIL** — ruff clean; mypy 3 errors in `src/sentinel/llm/client.py` (unused `type: ignore` import + `ollama.chat(timeout=…)` kwarg missing in installed ollama 0.6.3 stubs; runtime has a TypeError fallback so behavior is unaffected) |
| `make test` | **FAIL** — 87 passed, 1 failed: `tests/test_verify.py::test_generate_without_ollama_raises_unavailable` expects `LLMUnavailable` when Ollama is unreachable, but the user-requested local SLM shim is live on 127.0.0.1:11434 so generation succeeds |
| `make eval` | **FAIL (knock-on)** — scorecard computes fully and `outputs/eval_report.md` is written (silver 13/13, red-team 9/9), but exit code is 1 because it gates on `tests_ok` (see above) |
| `make repro` | **PASS** — 3 runs, 7 non-`reason` columns byte-identical |
| `make cost` | **PASS** — `outputs/token_cost_log.json` written |
| CSV header vs `fixtures/sample_output_format_v2.csv` | **PASS** — identical columns, same order |
| `flagged` vocab | **PASS** — only `Yes` / `No (justified)` (13 rows) |
| blank-id rule (empty iff `Yes`) | **PASS** |
| `outputs/repro/` 3-run identity | **PASS** |
| `outputs/token_cost_log.json` schema | **PASS** — has `calls`, `input_tokens`, `output_tokens`, `local_cost: 0`, `hosted_rates: "FILL_BY_HUMAN"` |

Root cause (single, environmental, caused by the approved SLM demo): installing the
`ollama` 0.6.3 client + running the local shim changed the environment the tests
were written against (Ollama absent). No pipeline logic, thresholds, verdicts, or
test files were modified.

Side effect to know: `make cost` does a cold-cache run and reset
`outputs/token_cost_log.json` to zeros. The live-LLM numbers (14 calls, 1864 in /
1156 out tokens, fallback_rate 0.2, $0, 13/13 verdict parity) are preserved in
NOTES.md ("Local SLM now live").

### Decision needed (pick one — I changed nothing)
- **A.** Stop the SLM shim + uninstall the `ollama` client, re-run verify for a
  clean `--no-llm` release (mypy `type: ignore` becomes valid again; the
  unreachable-Ollama test passes). SLM demo can be re-added post-release.
- **B.** Keep the SLM and update the one test + two typing lines to the
  Ollama-present reality (I will not do this unasked — it touches tests).

## 2. CLEAN — NOT RUN (blocked)
## 3. README CHECK — NOT RUN (blocked)
## 4. FRESH-CLONE TEST — NOT RUN (blocked)
## 5. PUBLISH — NOT RUN (blocked)
- `gh auth status` not checked yet.
- No repo created, nothing pushed, no zip built.

## Manual items for you
1. Choose A or B above.
2. Fill `hosted_rates` in `outputs/token_cost_log.json` from the provider pricing
   page (currently `"FILL_BY_HUMAN"` — do not invent prices).
3. Confirm GitHub repo name + visibility when we reach PUBLISH.
