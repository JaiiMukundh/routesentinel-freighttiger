"""Phase 6 evaluation scorecard (RouteSentinel).

Prints one scorecard to stdout and writes outputs/eval_report.md with:
  unit-test status, silver precision/recall/verdict accuracy (with and
  without debatable rows), retrieval Recall@4/MRR, red-team pass rate,
  injection-benchmark table, threshold-sweep table.

Prefers the real pipeline (outputs/output.csv, src.sentinel.*) when present;
otherwise uses the deterministic reference path in eval/__init__.py and labels
it as such. Deterministic, sorted, no timestamps, no randomness.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from eval import (  # noqa: E402
    DATA,
    OUTPUTS,
    EvalConfig,
    Note,
    RedTeamCase,
    apply_case_notes,
    load_config,
    load_golden_notes,
    load_redteam_cases,
    load_silver,
    own_pct_of,
    run_redteam_case,
    verdict_for,
)
from eval.inject_anomalies import (  # noqa: E402
    MAGNITUDE_GRID,
    is_candidate,
    load_weekly_metrics,
    run_injection_benchmark,
)
from eval.inject_anomalies import (
    format_table as format_injection_table,
)

OUTPUT_CSV = OUTPUTS / "output.csv"
REPORT_MD = OUTPUTS / "eval_report.md"


def run_unit_tests() -> tuple[bool, str]:
    """Run the pytest suite; never raise (status is reported, not fatal)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=str(OUTPUTS.parent),
            capture_output=True,
            text=True,
            timeout=600,
        )
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
        # Strip durations: pytest prints "N passed in X.XXs", which varies run
        # to run. Counts are deterministic; timings are not (AGENTS.md rule 6).
        tail = re.sub(r" in \d+\.\d+s(?=\s*$)", "", tail)
        return proc.returncode == 0, tail or "(no output)"
    except Exception as exc:  # noqa: BLE001 - status must never crash eval
        return False, f"pytest could not run: {exc}"


def try_real_gate() -> tuple[object | None, str]:
    """Use src.sentinel.agent.gate.justify if importable and compatible."""
    try:
        from src.sentinel.agent import gate as real_gate

        if not hasattr(real_gate, "justify"):
            return None, "reference gate (real gate lacks justify)"
        return real_gate, "pipeline gate (src.sentinel.agent.gate)"
    except Exception:  # noqa: BLE001
        return None, "reference gate (AGENTS.md G1..G5; pipeline gate not importable)"


def pipeline_predicted(
    cfg: EvalConfig,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], tuple[str, str]], str]:
    """Candidate set + verdicts from outputs/output.csv when present."""
    df = pd.read_csv(OUTPUT_CSV, dtype=str).fillna("")
    cands = set(zip(df["route"].tolist(), df["week_of"].tolist(), strict=True))
    verdicts = {
        (r, w): (f, m)
        for r, w, f, m in zip(
            df["route"], df["week_of"], df["flagged"], df["matched_note_id"], strict=True
        )
    }
    return cands, verdicts, "pipeline output (outputs/output.csv)"


def reference_predicted(
    cfg: EvalConfig,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], tuple[str, str]], str]:
    """Candidate set + verdicts from the deterministic reference path."""
    ref = load_weekly_metrics()
    notes = load_golden_notes()
    cands: set[tuple[str, str]] = set()
    verdicts: dict[tuple[str, str], tuple[str, str]] = {}
    for _, row in ref[is_candidate(ref, cfg)].iterrows():
        key = (str(row["route"]), str(row["week_of"]))
        cands.add(key)
        flagged, matched = verdict_for(
            key[0], date.fromisoformat(key[1]), float(row["own_pct"]), notes, cfg
        )
        verdicts[key] = (flagged, matched)
    return (
        cands,
        verdicts,
        "reference path (candidate rule + reference gate; outputs/output.csv not present)",
    )


def pr_scores(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> tuple[float, float]:
    tp = len(pred & gold)
    return (tp / len(pred) if pred else 1.0, tp / len(gold) if gold else 1.0)


class SilverSubset(TypedDict):
    n_gold: int
    precision: float
    recall: float
    verdict_accuracy: float
    verdict_correct: int


class QueryRow(TypedDict):
    route: str
    week_of: str
    gold: str
    rank: int | None
    recall_at_k: bool


class SystemRow(TypedDict):
    system: str
    query_rows: list[QueryRow]
    recall_at_k: float
    mrr: float
    k: int


class SweepRow(TypedDict):
    own_pct: float
    peer_pct: float
    n_candidates: int
    precision: float
    recall: float
    is_default: bool


def silver_metrics(
    pred_cands: set[tuple[str, str]], verdicts: dict[tuple[str, str], tuple[str, str]]
) -> dict[str, SilverSubset]:
    full = load_silver(include_debatable=True)
    strict = load_silver(include_debatable=False)
    out: dict[str, SilverSubset] = {}
    for name, rows in (("with_debatable", full), ("without_debatable", strict)):
        gold_set = {(r.route, r.week_of.isoformat()) for r in rows}
        prec, rec = pr_scores(pred_cands, gold_set)
        correct = sum(
            1
            for r in rows
            if verdicts.get((r.route, r.week_of.isoformat()), ("?", "?"))
            == (r.expected_flagged, r.expected_note_id)
        )
        out[name] = {
            "n_gold": len(rows),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "verdict_accuracy": round(correct / len(rows), 4) if rows else 1.0,
            "verdict_correct": correct,
        }
    return out


def gate_agreement(
    cases: list[RedTeamCase], base_notes: list[Note], cfg: EvalConfig
) -> tuple[int, int, str]:
    """Cross-check reference verdicts against the pipeline gate (INFO only).

    Never fails the run: returns (agree, total, label). Any import or
    signature mismatch yields total == 0 with an explanatory label.
    """
    try:
        from src.sentinel.agent import gate as real_gate

        decide_one = real_gate.decide_one
    except Exception:  # noqa: BLE001
        return 0, 0, "pipeline gate not importable; cross-check skipped"
    agree = 0
    total = 0
    try:
        for case in cases:
            if case.kind != "gate":
                continue
            notes = apply_case_notes(base_notes, case)
            note_dicts = [
                {
                    "note_id": n.note_id,
                    "scope": "ALL" if n.is_all() else list(n.scope),
                    "effect": n.effect,
                    "start": n.start.isoformat(),
                    "end": n.end.isoformat() if n.end else None,
                    "open_ended": n.open_ended,
                    "mag": [n.mag_lo, n.mag_hi],
                    "confidence": n.confidence,
                }
                for n in notes
            ]
            for route, week_of in case.targets:
                own = own_pct_of(route, week_of)
                ref_flagged, ref_matched = verdict_for(
                    route, date.fromisoformat(week_of), own, notes, cfg
                )
                cand = {"route": route, "week_of": week_of, "own_pct": own}
                real_flagged, real_matched, _ = decide_one(cand, note_dicts, None)
                total += 1
                agree += (real_flagged, real_matched) == (ref_flagged, ref_matched)
        return agree, total, "pipeline gate (src.sentinel.agent.gate.decide_one)"
    except Exception as exc:  # noqa: BLE001
        return 0, 0, f"pipeline gate cross-check skipped ({exc})"


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def fallback_retrieve(query: str, k: int) -> list[str]:
    """Deterministic token-overlap ranker over note texts; tie-break note_id."""
    qtok = tokenize(query)
    scored = sorted(
        (
            (
                len(
                    qtok
                    & tokenize(
                        n.text
                        + " "
                        + n.note_id
                        + " "
                        + " ".join(n.scope if not n.is_all() else ("all routes",))
                    )
                ),
                n.note_id,
            )
            for n in load_golden_notes()
        ),
        key=lambda t: (-t[0], t[1]),
    )
    return [nid for _, nid in scored[:k]]


def score_rows(qrows: list[QueryRow], system: str, k: int) -> SystemRow:
    rec = sum(1 for r in qrows if r["recall_at_k"]) / max(1, len(qrows))
    mrr = sum(1.0 / r["rank"] for r in qrows if r["rank"] is not None) / max(1, len(qrows))
    return {
        "system": system,
        "query_rows": qrows,
        "recall_at_k": round(rec, 4),
        "mrr": round(mrr, 4),
        "k": k,
    }


def retrieval_metrics(cfg: EvalConfig) -> tuple[list[SystemRow], str]:
    """Recall@k and MRR over labelled rows (silver rows with a gold note).

    Uses the pipeline retriever (src.sentinel.notes.retrieve over a freshly
    built index) per system bm25/dense/hybrid; falls back to the deterministic
    token-overlap ranker only if the pipeline path raises.
    """
    labelled = [
        (r.route, r.week_of.isoformat(), r.expected_note_id)
        for r in load_silver()
        if r.expected_note_id
    ]
    try:
        from src.sentinel.notes.index import build_index
        from src.sentinel.notes.retrieve import (
            load_notes_csv,
            retrieve,
            retrieve_single_system,
        )

        index = build_index(load_notes_csv(DATA / "notes.csv"))
        table: list[SystemRow] = []
        for system in ("bm25", "dense", "hybrid"):
            qrows: list[QueryRow] = []
            for route, week_of, gold in sorted(labelled):
                cand = {"route": route, "week_of": week_of}
                if system == "hybrid":
                    ranked = retrieve(cand, index, k=cfg.retrieval_k)
                else:
                    ranked = retrieve_single_system(cand, index, system, k=cfg.retrieval_k)
                rank = ranked.index(gold) + 1 if gold in ranked else None
                qrows.append(
                    {
                        "route": route,
                        "week_of": week_of,
                        "gold": gold,
                        "rank": rank,
                        "recall_at_k": rank is not None,
                    }
                )
            table.append(score_rows(qrows, system, cfg.retrieval_k))
        return table, "pipeline retrieve (src.sentinel.notes.retrieve: bm25/dense/hybrid)"
    except Exception:  # noqa: BLE001 - pipeline path optional; fallback is deterministic
        qrows = []
        for route, week_of, gold in sorted(labelled):
            ranked = fallback_retrieve(
                f"{route} freight cost increase week of {week_of}", cfg.retrieval_k
            )
            rank = ranked.index(gold) + 1 if gold in ranked else None
            qrows.append(
                {
                    "route": route,
                    "week_of": week_of,
                    "gold": gold,
                    "rank": rank,
                    "recall_at_k": rank is not None,
                }
            )
        return [score_rows(qrows, "fallback-token-overlap", cfg.retrieval_k)], (
            "fallback token-overlap ranker (notes/retrieve not importable)"
        )


def threshold_sweep(cfg: EvalConfig) -> list[SweepRow]:
    """Pure-function sweep: own 4..15% x peer 5..25% (1pp steps), P/R vs silver."""
    ref = load_weekly_metrics()
    gold = {(r.route, r.week_of.isoformat()) for r in load_silver(include_debatable=True)}
    table: list[SweepRow] = []
    for own_bp in range(4, 16):
        for peer_bp in range(5, 26):
            mask = (
                (ref["n_prior"] >= cfg.min_history)
                & (ref["own_pct"] >= own_bp / 100.0)
                & (ref["peer_pct"] >= peer_bp / 100.0)
            )
            pred = set(
                zip(
                    ref.loc[mask, "route"].tolist(),
                    ref.loc[mask, "week_of"].tolist(),
                    strict=True,
                )
            )
            prec, rec = pr_scores(pred, gold)
            table.append(
                {
                    "own_pct": own_bp / 100.0,
                    "peer_pct": peer_bp / 100.0,
                    "n_candidates": len(pred),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "is_default": (
                        own_bp / 100.0 == cfg.own_threshold
                        and peer_bp / 100.0 == cfg.peer_threshold
                    ),
                }
            )
    return sorted(table, key=lambda d: (d["own_pct"], d["peer_pct"]))


def main() -> int:
    cfg = load_config()
    lines: list[str] = []
    report: list[str] = ["# RouteSentinel eval report (Phase 6)", ""]

    tests_ok, tests_tail = run_unit_tests()
    lines.append(f"unit tests: {'PASS' if tests_ok else 'FAIL'}")
    report += [
        "## Unit tests",
        f"Status: {'PASS' if tests_ok else 'FAIL'}",
        "",
        "```",
        tests_tail,
        "```",
        "",
    ]

    if OUTPUT_CSV.exists():
        pred_cands, verdicts, pred_source = pipeline_predicted(cfg)
    else:
        pred_cands, verdicts, pred_source = reference_predicted(cfg)
    _, gate_source = try_real_gate()
    gate_source = gate_source if OUTPUT_CSV.exists() else f"{pred_source}; {gate_source}"
    sil = silver_metrics(pred_cands, verdicts)
    for name in ("with_debatable", "without_debatable"):
        m = sil[name]
        lines.append(
            f"silver[{name}]: P={m['precision']:.4f} R={m['recall']:.4f} "
            f"verdict_acc={m['verdict_accuracy']:.4f} ({m['verdict_correct']}/{m['n_gold']})"
        )
    report += [
        "## Silver labels (fixtures/expected_candidates.csv)",
        f"Prediction source: {pred_source}.",
        f"Gate source: {gate_source}.",
        "",
        "| subset | n_gold | precision | recall | verdict_accuracy | correct |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("with_debatable", "without_debatable"):
        m = sil[name]
        report.append(
            f"| {name} | {m['n_gold']} | {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['verdict_accuracy']:.4f} | {m['verdict_correct']}/{m['n_gold']} |"
        )
    report.append("")

    retr, retr_source = retrieval_metrics(cfg)
    for sys_row in retr:
        lines.append(
            f"retrieval@{sys_row['k']}[{sys_row['system']}]: "
            f"recall={sys_row['recall_at_k']:.4f} mrr={sys_row['mrr']:.4f}"
        )
    lines[-1] += f" [{retr_source}]"
    report += [
        "## Retrieval",
        f"Source: {retr_source}.",
        f"Labelled rows: {len(retr[0]['query_rows'])} (silver rows with a gold note).",
        "",
        "| system | k | recall@k | mrr |",
        "| --- | --- | --- | --- |",
    ]
    for sys_row in retr:
        report.append(
            f"| {sys_row['system']} | {sys_row['k']} "
            f"| {sys_row['recall_at_k']:.4f} | {sys_row['mrr']:.4f} |"
        )
    report.append("")
    detail_rows = retr[0]["query_rows"]  # per-query detail from the first system row
    report += [
        "| route | week_of | gold_note | rank_in_topk | hit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for q in detail_rows:
        hit = "yes" if q["recall_at_k"] else "no"
        rank = q["rank"] if q["rank"] else "-"
        report.append(f"| {q['route']} | {q['week_of']} | {q['gold']} | {rank} | {hit} |")
    report.append("")

    base_notes = load_golden_notes()
    cases = load_redteam_cases()
    case_lines: list[str] = []
    passed = 0
    for case in cases:
        ok, detail = run_redteam_case(base_notes, case, cfg)
        passed += ok
        case_lines.append(f"  [{('PASS' if ok else 'FAIL')}] {case.case_id}: {detail}")
        lines.append(f"redteam {case.case_id}: {'PASS' if ok else 'FAIL'}")
    rate = passed / max(1, len(cases))
    lines.append(f"red-team pass rate: {passed}/{len(cases)} = {rate:.2%} (required 100%)")
    agree, agree_total, agree_label = gate_agreement(cases, base_notes, cfg)
    if agree_total:
        lines.append(f"pipeline-gate agreement: {agree}/{agree_total} [{agree_label}] (info only)")
    else:
        lines.append(f"pipeline-gate agreement: n/a [{agree_label}]")
    report += [
        "## Red team (eval/adversarial_cases.yaml)",
        f"Pass rate: {passed}/{len(cases)} = {rate:.2%} (required 100%).",
    ]
    if agree_total:
        report.append(
            f"Pipeline-gate agreement (info only): {agree}/{agree_total} [{agree_label}]."
        )
    else:
        report.append(f"Pipeline-gate agreement: n/a [{agree_label}].")
    report += [
        "",
        "```",
        *case_lines,
        "```",
        "",
    ]

    inj_results, _ = run_injection_benchmark(magnitudes=MAGNITUDE_GRID, cfg=cfg)
    lines.append(f"injection benchmark (seed={cfg.seed}):")
    for r in inj_results:
        lines.append(
            f"  x{r.magnitude:.2f}: recall={r.recall:.3f} "
            f"({r.n_detected}/{r.n_injected}) fp/100={r.fp_per_100:.2f}"
        )
    report += [
        "## Synthetic injection benchmark (eval/inject_anomalies.py)",
        f"Fixed seed {cfg.seed}; {inj_results[0].n_injected} eligible non-candidate "
        "route-weeks scaled; recall = fraction flagged; "
        "fp/100 = non-injected route-weeks flagged per 100.",
        "",
        format_injection_table(inj_results),
        "",
    ]

    sweep = threshold_sweep(cfg)
    default_rows = [d for d in sweep if d["is_default"]]
    d0: SweepRow | None = default_rows[0] if default_rows else None
    if d0 is not None:
        lines.append(
            f"threshold sweep: {len(sweep)} combos; "
            f"default (own=7%, peer=10%) -> n={d0['n_candidates']} "
            f"P={d0['precision']} R={d0['recall']}"
        )
    else:
        lines.append(f"threshold sweep: {len(sweep)} combos; default not found")
    report += [
        "## Threshold sweep (pure function over reference weekly metrics)",
        "Candidate rule recomputed per cell vs the 13-row silver candidate set "
        "(no full re-run). `*` marks the configured default.",
        "",
        "| own_thresh | peer_thresh | n_candidates | precision | recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in sweep:
        star = " *" if d["is_default"] else ""
        report.append(
            f"| {d['own_pct']:.2f}{star} | {d['peer_pct']:.2f}{star} "
            f"| {d['n_candidates']} | {d['precision']:.4f} | {d['recall']:.4f} |"
        )
    report.append("")

    print("\n".join(lines))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(report))
    print(f"report written to {REPORT_MD}")
    ok_all = tests_ok and rate == 1.0 and all(m["verdict_accuracy"] == 1.0 for m in sil.values())
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
