# RouteSentinel — FreightTiger

Deterministic CLI that flags unjustified freight-cost rises per route and explains them with retrieved notes. LLM never decides verdicts/numbers/note ids — it may only extract fields and word `reason` text, both verified. Full pipeline works with `--no-llm` and produces identical verdicts.

## Quickstart (4 commands)

```bash
python3 -m pip install -e .
make run-nollm                      # -> outputs/output.csv + outputs/findings.sqlite
make test                           # 88 passed
make repro                          # 3 runs, non-reason columns identical
```
Dashboard (optional): `cp outputs/output.csv web/public/output.csv && cd web && npm install && npm run dev`.

## Definitions (exact, per AGENTS.md)

- `route = "{origin}-{destination}"` directed; `route_type` from CSV, unique per route else raise.
- `week_of = shipment_date - timedelta(days=shipment_date.weekday())` (Monday).
- `cost_per_tonne_km` per (route, week) = `sum(cost) / sum(qty * dist)` (pooled ratio).
- Own history: mean of route's `cpt` over previous up-to-8 existing weeks strictly before current week; `n_prior = min(8, weeks before)`; first week NaN; `own_pct = cpt/hist - 1`.
- Peers: unweighted mean of `cpt` of *other* same-`route_type` routes same week; none → NaN; `peer_pct = cpt/peer - 1`.
- Candidate iff `n_prior >= 4 AND own_pct >= 0.07 AND peer_pct >= 0.10` (from `config.yaml`). Output = candidates only, sorted `(week_of, route)`.
- `<8 weeks` behaviour: use all available prior weeks (no padding/extrapolation); first week NaN can never be a candidate.

## Output contract (`outputs/output.csv`)

Header exactly: `route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason` — `week_of` YYYY-MM-DD, cost 2 decimals, `+X.X% vs this route's past average` / `+X.X% vs similar-length routes this week` (sign always, 1 decimal, from unrounded), `flagged` ∈ {`Yes`, `No (justified)`}, `matched_note_id` empty iff `Yes`. Format reference: `fixtures/sample_output_format_v2.csv` (v1/v2 file-name note: contract test reproduces its formatting; output file here is `outputs/output.csv`).

## How AI is used (and not used)

- Deterministic code decides verdicts, numbers, matched note ids (evidence gate G1–G5). Retrieval (BM25 + bge-small + RRF k=60) only proposes; similarity alone never justifies.
- LLM allowed only for: note extraction (once/note, cached on sha256), reason wording (candidates only, fact-sheet-only prompt, note text marked untrusted), one retry on verifier failure, Q&A stretch. Every call cached + logged to `outputs/token_cost_log.json`.
- Reason verifier: every note id/date/route/number must trace to the fact sheet; fail twice → deterministic template. Stub test proves fake `N011` + invented cause falls back.
- This environment: local SLM live via `scripts/slm_serve.py` (Qwen2.5-0.5B on CPU,
  Ollama-compatible shim, user-approved deviation from the canonical 3B — see NOTES.md);
  13/13 verdicts identical LLM vs `--no-llm`; verifier rejects bad wording → template.
  Committed `outputs/token_cost_log.json` is the cold-cache `make cost` artifact (zeros).

## Evidence gate (G1–G5)

| Note | Scope | Effect | Window | Verdict on candidates |
|------|-------|--------|--------|----------------------|
| N001 | Chennai-Bangalore | increase | 2025-02-24–2025-03-08 | justifies Feb-24 + Mar-03; Mar-10 fails **G2** (week starts after Mar-08) |
| N002 | Ahmedabad-Mumbai | increase | 2025-01-20 week | justifies Jan-20 |
| N003 | ALL, fuel +5–7% | increase | open-ended [May-05, +8w) | absorbed by history/peers; Mumbai-Pune Sep–Dec fails **G2** |
| N004 | ALL, toll | none (out-of-dataset) | — | **G3** reject everywhere |
| N005 | Mumbai-Delhi | none | — | **G3** reject |
| N006 | ALL | none (stable) | — | **G3** reject (Sep-15 decoy) |
| N007 | Delhi-Jaipur | decrease (improved) | — | **G3** (+G2) reject |
| N008 | Kolkata-Bhubaneswar | none (normal) | — | **G3** reject |
| N009 | Chennai-Bangalore | normalisation | Mar-17 | **G3** reject (Mar-17 decoy) |
| N010 | ALL | none (absorbed) | — | **G3** reject |

Tie-break: route-specific > ALL → larger overlap → later start → lower note_id. Unexplained rows cite closest *rejected* note in `reason` but keep `matched_note_id` empty.

March-10/17 judgement call: both `Yes` under defaults (tagged `debatable` in silver labels) — Mar-10 is 2 days past N001's window (strict G2, grace 0); Mar-17's only nearby note N009 is a normalisation, not a rise (G3). Widening the window or flipping polarity would need evidence from the Phase-6 threshold sweep — none found, so defaults stand.

## Eval scorecard

`make eval` → `outputs/eval_report.md`: unit tests PASS; silver verdict accuracy 100% (13/13 with debatable, 11/11 without); hybrid Recall@4 1.0 / MRR 1.0 (BM25 1.0/1.0, dense 1.0/0.78); red-team 9/9 100%; injection recall rises with magnitude (x1.05: 0.25 → x1.35: 0.975, fp/100 ≈ 1.8); threshold sweep 252 combos, default (7%, 10%) P=1.0 R=1.0.

## Repro + cost

- `make repro` (scripts/repro.sh): 3 fresh runs (default / `--no-cache` / `--no-llm`), diffs 7 non-`reason` cols → empty; hashes in `outputs/repro/repro_report.md`. PASS.
- `make cost` (scripts/cost.sh): cold-cache run → `outputs/token_cost_log.json` (human fills hosted rates from provider pricing).

| metric (committed cold-cache artifact) | value |
|---|---|
| llm calls | 0 |
| input / output tokens | 0 / 0 |
| cache hits | 0 |
| fallback_rate | 0.0 |
| local_cost | 0 |
| hosted_rates | `FILL_BY_HUMAN` (you must fill from the provider pricing page) |

Live-demo numbers (14 calls, 1864 in / 1156 out, fallback 0.2, $0) are recorded in NOTES.md, not in the committed artifact.

## Assumptions / limitations

- Monday-week bucketing; pooled ratio (not mean-of-ratios); unweighted LOO peers.
- Open-ended notes live 8 weeks (`absorption_weeks`); `GRACE_DAYS=0`.
- Magnitude gate: `own_pct <= 2 × stated max` (N003: 2×7%=14% cap — never binding here since N003 fails G2 first).
- Retrieval fallback: built-in BM25 + TF-IDF when `rank_bm25`/`fastembed` missing (same acceptance, warnings only).
- No LangChain/LlamaIndex/Kafka/Spark; only allowed deps in `pyproject.toml`.

## Layout

`src/sentinel/{config,ingest,metrics,detect,output}.py` · `notes/{extract,index,retrieve}.py` · `agent/{state,tools,gate,graph,explain,verify}.py` · `llm/{client,ledger}.py` · `qa/answer.py` · `eval/` · `tests/` · `data/` (shipments/notes copies; never edit sources) · `fixtures/` (read-only) · `web/` (TS dashboard) · `scripts/{repro,cost}.sh`.
