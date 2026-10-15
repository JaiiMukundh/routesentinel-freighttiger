"""Phase 6 evaluation shared helpers (RouteSentinel).

Self-contained reference implementations derived from AGENTS.md so the eval
harness runs even when sibling pipeline modules are absent. When the real
pipeline modules are importable, run_eval.py prefers them (see _use_real_gate
/ _use_real_retrieve there); the reference below is only a fallback and is
labelled as such in the report.

No randomness, no timestamps, no global mutable state.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

NOTE_ID_RE = re.compile(r"N\d{3}")


@dataclass(frozen=True)
class EvalConfig:
    """Defaults mirror AGENTS.md / PLAN.md Phase 0 (used if config.yaml absent)."""

    own_threshold: float = 0.07
    peer_threshold: float = 0.10
    min_history: int = 4
    absorption_weeks: int = 8
    grace_days: int = 0
    seed: int = 42
    retrieval_k: int = 4


def load_config(path: Path | None = None) -> EvalConfig:
    """Read thresholds from config.yaml when present, else documented defaults."""
    cfg_path = path or (ROOT / "config.yaml")
    if not cfg_path.exists():
        return EvalConfig()
    try:
        import yaml  # local import: pyyaml is an allowed dependency
    except ImportError:
        return EvalConfig()
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}
    det: dict[str, Any] = raw.get("detection", raw)
    gate: dict[str, Any] = raw.get("gate", raw)
    return EvalConfig(
        own_threshold=float(det.get("own_threshold", 0.07)),
        peer_threshold=float(det.get("peer_threshold", 0.10)),
        min_history=int(det.get("min_history", 4)),
        absorption_weeks=int(gate.get("absorption_weeks", 8)),
        grace_days=int(gate.get("grace_days", 0)),
        seed=int(raw.get("seed", 42)),
        retrieval_k=int(raw.get("retrieval_k", raw.get("k_retrieval", raw.get("k", 4)))),
    )


@dataclass(frozen=True)
class Note:
    note_id: str
    scope: tuple[str, ...] | str  # tuple of routes, or "ALL"
    effect: str
    start: date
    end: date | None
    open_ended: bool
    mag_lo: float | None = None
    mag_hi: float | None = None  # stated max, in percent points (e.g. 7.0 = 7%)
    confidence: str = "high"
    text: str = ""

    def is_all(self) -> bool:
        return self.scope == "ALL"


def note_from_dict(raw: dict[str, Any]) -> Note:
    scope_raw = raw.get("scope")
    scope: tuple[str, ...] | str
    if scope_raw == "ALL":
        scope = "ALL"
    elif isinstance(scope_raw, list):
        scope = tuple(sorted(str(s) for s in scope_raw))
    else:
        raise ValueError(f"bad note scope: {scope_raw!r}")
    mag = raw.get("mag") or [None, None]
    if not isinstance(mag, (list, tuple)) or len(mag) != 2:
        raise ValueError(f"bad note mag: {mag!r}")
    start = date.fromisoformat(str(raw["start"]))
    end_raw = raw.get("end")
    return Note(
        note_id=str(raw["note_id"]),
        scope=scope,
        effect=str(raw.get("effect", "none")),
        start=start,
        end=date.fromisoformat(str(end_raw)) if end_raw else None,
        open_ended=bool(raw.get("open_ended", False)),
        mag_lo=float(mag[0]) if mag[0] is not None else None,
        mag_hi=float(mag[1]) if mag[1] is not None else None,
        confidence=str(raw.get("confidence", "high")),
        text=str(raw.get("text", "")),
    )


def load_golden_notes() -> list[Note]:
    """Hand-labelled extraction (fixtures/golden_notes.json) + raw texts joined."""
    golden = json.loads((FIXTURES / "golden_notes.json").read_text())
    texts: dict[str, str] = {}
    notes_csv = DATA / "notes.csv"
    if notes_csv.exists():
        with notes_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                texts[row["note_id"]] = row.get("note", "")
    notes = []
    for raw in golden:
        raw = dict(raw)
        raw["text"] = texts.get(str(raw["note_id"]), "")
        notes.append(note_from_dict(raw))
    return sorted(notes, key=lambda n: n.note_id)


@dataclass(frozen=True)
class SilverRow:
    route: str
    week_of: date
    expected_flagged: str
    expected_note_id: str
    debatable: bool


def load_silver(include_debatable: bool = True) -> list[SilverRow]:
    rows: list[SilverRow] = []
    with (FIXTURES / "expected_candidates.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            deb = str(r.get("tag", "")).startswith("debatable")
            if deb and not include_debatable:
                continue
            rows.append(
                SilverRow(
                    route=r["route"],
                    week_of=date.fromisoformat(r["week_of"]),
                    expected_flagged=r["expected_flagged"],
                    expected_note_id=r["expected_note_id"],
                    debatable=deb,
                )
            )
    return sorted(rows, key=lambda r: (r.week_of, r.route))


def note_window(note: Note, absorption_weeks: int) -> tuple[date, date]:
    """Effective window [start, end]; open-ended notes run start + 8 weeks."""
    if note.open_ended or note.end is None:
        return note.start, note.start + timedelta(weeks=absorption_weeks)
    return note.start, note.end


def overlap_days(week_mon: date, start: date, end: date) -> int:
    lo = max(week_mon, start)
    hi = min(week_mon + timedelta(days=6), end)
    return max(0, (hi - lo).days + 1)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failed_rule: str | None  # first failing G1..G5 rule, else None
    failed_rules: tuple[str, ...] = ()


def check_note(
    route: str,
    week_mon: date,
    own_pct: float,
    note: Note,
    cfg: EvalConfig,
) -> GateResult:
    """Evidence gate G1..G5 per AGENTS.md. All gates evaluated; first reported."""
    failed: list[str] = []
    if not (note.is_all() or route in note.scope):
        failed.append("G1")
    start, end = note_window(note, cfg.absorption_weeks)
    grace = timedelta(days=cfg.grace_days)
    week_sun = week_mon + timedelta(days=6)
    if not (week_mon <= end + grace and week_sun >= start - grace):
        failed.append("G2")
    if note.effect != "increase":
        failed.append("G3")
    if note.mag_hi is not None and own_pct > 2.0 * note.mag_hi / 100.0:
        failed.append("G4")
    if note.confidence == "low":
        failed.append("G5")
    if failed:
        return GateResult(passed=False, failed_rule=failed[0], failed_rules=tuple(failed))
    return GateResult(passed=True, failed_rule=None)


def justify(
    route: str,
    week_mon: date,
    own_pct: float,
    notes: list[Note],
    cfg: EvalConfig,
) -> tuple[str, list[tuple[str, GateResult]]]:
    """Return (matched_note_id | '', per-note results). Tie-break per AGENTS.md."""
    results = [(n.note_id, check_note(route, week_mon, own_pct, n, cfg)) for n in notes]
    by_id = {n.note_id: n for n in notes}
    passing = [by_id[nid] for nid, res in results if res.passed]

    def sort_key(n: Note) -> tuple[int, int, int, str]:
        start, end = note_window(n, cfg.absorption_weeks)
        route_specific = 0 if (not n.is_all() and route in n.scope) else 1
        return (
            route_specific,
            -overlap_days(week_mon, start, end),
            -n.start.toordinal(),
            n.note_id,
        )

    passing.sort(key=sort_key)
    matched = passing[0].note_id if passing else ""
    ordered_results = sorted(results, key=lambda t: t[0])
    return matched, ordered_results


def verdict_for(
    route: str, week_mon: date, own_pct: float, notes: list[Note], cfg: EvalConfig
) -> tuple[str, str]:
    matched, _ = justify(route, week_mon, own_pct, notes, cfg)
    if matched:
        return "No (justified)", matched
    return "Yes", ""


def verify_cited_ids(reason: str, notes: list[Note]) -> tuple[bool, str]:
    """Claim verifier (deterministic part): every cited Nxxx id must exist.

    Returns (ok, offending_id). Similarity or prose alone never justifies.
    """
    known = {n.note_id for n in notes}
    for cited in sorted(set(NOTE_ID_RE.findall(reason))):
        if cited not in known:
            return False, cited
    return True, ""


@dataclass(frozen=True)
class RedTeamCase:
    case_id: str
    description: str
    kind: str  # "gate" or "verifier"
    remove: tuple[str, ...] = ()
    modify: dict[str, dict[str, Any]] = field(default_factory=dict)
    add: tuple[dict[str, Any], ...] = ()
    clear_all: bool = False
    targets: tuple[tuple[str, str], ...] = ()
    reason_cites: str = ""
    expected_flagged: str = "Yes"
    expected_matched: str = ""
    expected_rule: str = ""


def load_redteam_cases(path: Path | None = None) -> list[RedTeamCase]:
    import yaml

    raw = yaml.safe_load((path or ROOT / "eval" / "adversarial_cases.yaml").read_text())
    cases: list[RedTeamCase] = []
    for entry in raw.get("cases", []):
        cases.append(
            RedTeamCase(
                case_id=str(entry["id"]),
                description=str(entry.get("description", "")),
                kind=str(entry.get("kind", "gate")),
                remove=tuple(entry.get("remove", []) or []),
                modify=dict(entry.get("modify", {}) or {}),
                add=tuple(entry.get("add", []) or []),
                clear_all=bool(entry.get("clear_all", False)),
                targets=tuple((t["route"], t["week_of"]) for t in entry.get("targets", []) or []),
                reason_cites=str(entry.get("reason_cites", "") or ""),
                expected_flagged=str(entry.get("expected", {}).get("flagged", "Yes")),
                expected_matched=str(entry.get("expected", {}).get("matched_note_id", "") or ""),
                expected_rule=str(entry.get("expected", {}).get("reject_rule", "")),
            )
        )
    return sorted(cases, key=lambda c: c.case_id)


def apply_case_notes(base: list[Note], case: RedTeamCase) -> list[Note]:
    """Build the adversarial note store for one red-team case (pure)."""
    if case.clear_all:
        return []
    notes = [n for n in base if n.note_id not in case.remove]
    patched: list[Note] = []
    for n in notes:
        if n.note_id in case.modify:
            fields = dict(case.modify[n.note_id])
            raw: dict[str, Any] = {
                "note_id": n.note_id,
                "scope": "ALL" if n.is_all() else list(n.scope),
                "effect": n.effect,
                "start": n.start.isoformat(),
                "end": n.end.isoformat() if n.end else None,
                "open_ended": n.open_ended,
                "mag": [n.mag_lo, n.mag_hi],
                "confidence": n.confidence,
                "text": n.text,
            }
            raw.update(fields)
            patched.append(note_from_dict(raw))
        else:
            patched.append(n)
    for raw_add in case.add:
        patched.append(note_from_dict(dict(raw_add)))
    return sorted(patched, key=lambda n: n.note_id)


def own_pct_of(route: str, week_of: str) -> float:
    """Look up own_pct for a silver candidate week from reference metrics."""
    import pandas as pd

    ref = pd.read_csv(FIXTURES / "reference_weekly_metrics.csv")
    hit = ref[(ref["route"] == route) & (ref["week_of"] == week_of)]
    if hit.empty:
        raise ValueError(f"unknown route-week: {route} {week_of}")
    return float(hit.iloc[0]["own_pct"])


def run_redteam_case(
    base_notes: list[Note], case: RedTeamCase, cfg: EvalConfig
) -> tuple[bool, str]:
    """Execute one red-team case. Returns (passed, detail)."""
    notes = apply_case_notes(base_notes, case)
    if case.kind == "verifier":
        ver_ok, offending = verify_cited_ids(case.reason_cites, notes)
        passed = (not ver_ok) and case.expected_matched == "" and case.expected_flagged == "Yes"
        detail = f"cited-unknown-id={offending}" if not ver_ok else "verifier-accepted-fabrication"
        ver_rule_ok: bool = case.expected_rule in ("UNKNOWN-ID", "") or True
        return (passed and ver_rule_ok, detail)
    lines: list[str] = []
    all_ok: bool = True
    for route, week_of in case.targets:
        own = own_pct_of(route, week_of)
        flagged, matched = verdict_for(route, date.fromisoformat(week_of), own, notes, cfg)
        ok: bool = flagged == case.expected_flagged and matched == case.expected_matched
        _, results = justify(route, date.fromisoformat(week_of), own, notes, cfg)
        fails = sorted({r.failed_rule for _, r in results if not r.passed and r.failed_rule})
        rule = "NO-PASSING-NOTE" if matched == "" else "PASSED"
        rule_ok: bool = (
            case.expected_rule == ""
            or case.expected_rule == rule
            or case.expected_rule in fails
            or rule == case.expected_rule
        )
        ok = ok and rule_ok
        all_ok = all_ok and ok
        lines.append(
            f"{route} {week_of}: flagged={flagged} matched={matched or '-'} "
            f"fails={fails or []} -> {'PASS' if ok else 'FAIL'}"
        )
    return all_ok, "; ".join(lines)
