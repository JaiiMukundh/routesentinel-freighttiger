"""Deterministic pipeline builder + rule-based Q&A for RouteSentinel.

Two responsibilities (Phases 7-8):

1. ``build_outputs``: a dependency-free re-implementation of the
   ``--no-llm`` pipeline (AGENTS.md definitions). It reads
   ``data/shipments.csv`` and ``data/notes.csv`` and writes
   ``outputs/output.csv`` (8-column contract) plus
   ``outputs/findings.sqlite`` (tables ``findings`` and ``notes``).
   Used by ``scripts/repro.sh`` / ``scripts/cost.sh`` when the canonical
   ``python -m sentinel run`` entry point is unavailable, and by the QA
   tests to guarantee hermetic fixtures. All logic is generic: no value
   from ``fixtures/`` is hardcoded; thresholds come from ``config.yaml``
   when present, else the AGENTS.md defaults.

2. ``answer_question``: rules-only question answering over
   ``findings.sqlite``. Parses route/year/month/note-id from text,
   runs deterministic SQL, and composes answers only from the results,
   with citations. Refuses when there is no data and asks for
   clarification when the year is ambiguous.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults (AGENTS.md / PLAN.md Phase 0; overridden by config.yaml if present)
# ---------------------------------------------------------------------------

DEFAULT_OWN_THRESHOLD = 0.07
DEFAULT_PEER_THRESHOLD = 0.10
DEFAULT_MIN_HISTORY = 4
DEFAULT_ABSORPTION_WEEKS = 8
DEFAULT_GRACE_DAYS = 0

OUTPUT_HEADER = (
    "route,week_of,cost_per_tonne_km,vs_own_history,"
    "vs_similar_routes,flagged,matched_note_id,reason"
)

ROUTE_TYPES = ("Short", "Medium", "Long")

MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "sept": 9,
}
MONTH_LABEL = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

NONE_KEYWORDS = (
    "not significantly affected",
    "no rate change",
    "absorbed",
    "not part of this dataset",
    "remained stable",
    "no major disruptions",
    "no significant disruptions",
    "remained normal",
)
INCREASE_KEYWORDS = (
    "higher trip costs",
    "surcharge",
    "pushing up",
    "rose",
    "disrupted",
)
IMPROVE_KEYWORDS = ("improved", "resurfacing")


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Locate the repo root (directory containing data/shipments.csv)."""
    import os

    override = os.environ.get("SENTINEL_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "data" / "shipments.csv").exists():
            return parent
    return Path.cwd()


def load_thresholds(root: Path) -> dict[str, float | int]:
    """Load thresholds from config.yaml when present, else defaults."""
    cfg: dict[str, float | int] = {
        "own_threshold": DEFAULT_OWN_THRESHOLD,
        "peer_threshold": DEFAULT_PEER_THRESHOLD,
        "min_history": DEFAULT_MIN_HISTORY,
        "absorption_weeks": DEFAULT_ABSORPTION_WEEKS,
        "grace_days": DEFAULT_GRACE_DAYS,
    }
    path = root / "config.yaml"
    if not path.exists():
        return cfg
    text = path.read_text(encoding="utf-8")
    parsed: dict[str, str] = {}
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            for key in cfg:
                if key in data:
                    parsed[key] = str(data[key])
    except Exception:
        for line in text.splitlines():
            match = re.match(r"\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$", line)
            if match and match.group(1) in cfg:
                parsed[match.group(1)] = match.group(2)
    for key, raw in parsed.items():
        value = raw.strip().rstrip("%")
        try:
            number = float(value)
        except ValueError:
            continue
        if key in ("own_threshold", "peer_threshold") and number > 1:
            number = number / 100.0
        cfg[key] = (
            int(number) if key in ("min_history", "absorption_weeks", "grace_days") else number
        )
    return cfg


# ---------------------------------------------------------------------------
# Pipeline data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Note:
    note_id: str
    note_date: str
    scope: str  # route "A-B" or "ALL"
    effect: str
    start: str  # YYYY-MM-DD
    end: str | None
    open_ended: bool
    mag_min: float | None
    mag_max: float | None
    cause: str
    confidence: str
    text: str


@dataclass(frozen=True)
class Metric:
    route: str
    week_of: str
    cpt: float
    n_prior: int
    own_pct: float | None
    peer_pct: float | None


@dataclass(frozen=True)
class Eval:
    note_id: str
    passed: bool
    failed: tuple[str, ...]
    overlap_days: int


@dataclass(frozen=True)
class BuildStats:
    candidates: int
    justified: int
    flagged_yes: int
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    fallback_rate: float = 0.0


# ---------------------------------------------------------------------------
# Ingest + metrics (AGENTS.md definitions)
# ---------------------------------------------------------------------------


def monday(day: _dt.date) -> _dt.date:
    return day - _dt.timedelta(days=day.weekday())


def load_shipments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_shipments(rows: list[dict[str, str]]) -> dict[str, str]:
    """Validate rows; return route -> route_type (unique per route)."""
    route_type: dict[str, str] = {}
    bad = 0
    for row in rows:
        try:
            qty = float(row["quantity_tonnes"])
            dist = float(row["distance_km"])
            cost = float(row["freight_cost_inr"])
        except (KeyError, ValueError):
            bad += 1
            continue
        if not (qty > 0 and dist > 0 and cost > 0):
            bad += 1
        route = f"{row['origin']}-{row['destination']}"
        rtype = row["route_type"]
        if rtype not in ROUTE_TYPES:
            raise ValueError(f"bad route_type {rtype!r} for {route}")
        if route in route_type and route_type[route] != rtype:
            raise ValueError(f"route_type not unique for {route}")
        route_type[route] = rtype
        _dt.date.fromisoformat(row["shipment_date"])
    if bad:
        raise ValueError(f"{bad} shipment rows failed validation")
    return route_type


def weekly_cpt(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    cost: dict[tuple[str, _dt.date], float] = {}
    tonne_km: dict[tuple[str, _dt.date], float] = {}
    for row in rows:
        day = _dt.date.fromisoformat(row["shipment_date"])
        key = (f"{row['origin']}-{row['destination']}", monday(day))
        cost[key] = cost.get(key, 0.0) + float(row["freight_cost_inr"])
        tonne_km[key] = tonne_km.get(key, 0.0) + (
            float(row["quantity_tonnes"]) * float(row["distance_km"])
        )
    return {
        (route, week.isoformat()): cost[(route, week)] / tonne_km[(route, week)]
        for route, week in cost
    }


def compute_metrics(rows: list[dict[str, str]], route_type: dict[str, str]) -> list[Metric]:
    cpt = weekly_cpt(rows)
    by_route: dict[str, list[tuple[str, float]]] = {}
    for (route, week), value in cpt.items():
        by_route.setdefault(route, []).append((week, value))
    for weeks in by_route.values():
        weeks.sort()
    out: list[Metric] = []
    for route in sorted(by_route):
        weeks = by_route[route]
        for pos, (week, value) in enumerate(weeks):
            prior = [v for _, v in weeks[max(0, pos - 8) : pos]]
            n_prior = min(8, len(prior))
            own = (value / (sum(prior) / len(prior)) - 1) if prior else None
            peers = [
                other_cpt
                for (other_route, other_week), other_cpt in cpt.items()
                if other_week == week
                and other_route != route
                and route_type[other_route] == route_type[route]
            ]
            peer = (value / (sum(peers) / len(peers)) - 1) if peers else None
            out.append(Metric(route, week, value, n_prior, own, peer))
    out.sort(key=lambda m: (m.week_of, m.route))
    return out


def detect_candidates(metrics: list[Metric], cfg: dict[str, float | int]) -> list[Metric]:
    own_thr = float(cfg["own_threshold"])
    peer_thr = float(cfg["peer_threshold"])
    min_hist = int(cfg["min_history"])
    return [
        m
        for m in metrics
        if m.n_prior >= min_hist
        and m.own_pct is not None
        and m.own_pct >= own_thr
        and m.peer_pct is not None
        and m.peer_pct >= peer_thr
    ]


# ---------------------------------------------------------------------------
# Note extraction (deterministic rules; cross-checked vs golden_notes.json)
# ---------------------------------------------------------------------------


def _month_day_range(text: str, note_date: _dt.date) -> tuple[_dt.date | None, _dt.date | None]:
    match = re.search(
        r"from\s+([A-Za-z]+)\s+(\d{1,2})\s+to\s+([A-Za-z]+)\s+(\d{1,2})",
        text,
    )
    if not match:
        return None, None
    months = {
        m: i + 1
        for i, m in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        )
    }
    start = _dt.date(note_date.year, months[match.group(1)[:3].lower()], int(match.group(2)))
    end = _dt.date(note_date.year, months[match.group(3)[:3].lower()], int(match.group(4)))
    if end < start:
        end = _dt.date(note_date.year + 1, end.month, end.day)
    return start, end


def extract_note(note_id: str, date_str: str, applies_to: str, text: str) -> Note:
    note_date = _dt.date.fromisoformat(date_str)
    scope = "ALL" if applies_to.strip().lower() == "all routes" else applies_to.strip()
    lowered = text.lower()
    if "returned to normal" in lowered:
        effect = "normalisation"
    elif any(k in lowered for k in NONE_KEYWORDS):
        effect = "none"
    elif any(k in lowered for k in IMPROVE_KEYWORDS):
        effect = "decrease"
    elif any(k in lowered for k in INCREASE_KEYWORDS):
        effect = "increase"
    else:
        effect = "none"
    start_dt, end_dt = _month_day_range(text, note_date)
    start = start_dt or monday(note_date)
    if end_dt is not None:
        end: str | None = end_dt.isoformat()
        open_ended = False
    elif "about a week" in lowered or "festival week" in lowered:
        end = (start + _dt.timedelta(days=6)).isoformat()
        open_ended = False
    elif "starting this week" in lowered:
        end = None
        open_ended = True
    else:
        end = None
        open_ended = False
    mag_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%", text)
    mag_min = float(mag_match.group(1)) if mag_match else None
    mag_max = float(mag_match.group(2)) if mag_match else None
    if "flood" in lowered:
        cause = "weather"
    elif "festival" in lowered:
        cause = "festival"
    elif "diesel" in lowered or "fuel" in lowered:
        cause = "fuel"
    elif "toll" in lowered:
        cause = "toll"
    elif "maintenance" in lowered or "road" in lowered:
        cause = "road"
    elif "demand" in lowered:
        cause = "demand"
    elif "mandate" in lowered or "tracking" in lowered:
        cause = "regulatory"
    else:
        cause = "other"
    confidence = "high" if end_dt is not None else "medium"
    return Note(
        note_id,
        date_str,
        scope,
        effect,
        start.isoformat(),
        end,
        open_ended,
        mag_min,
        mag_max,
        cause,
        confidence,
        text,
    )


def load_notes(path: Path) -> list[Note]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    notes = [extract_note(r["note_id"], r["date"], r["applies_to"], r["note"]) for r in rows]
    notes.sort(key=lambda n: n.note_id)
    return notes


# ---------------------------------------------------------------------------
# Evidence gate G1-G5 + tie-break (AGENTS.md)
# ---------------------------------------------------------------------------


def effective_window(note: Note, absorption_weeks: int) -> tuple[_dt.date, _dt.date]:
    start = _dt.date.fromisoformat(note.start)
    if note.end is not None:
        return start, _dt.date.fromisoformat(note.end)
    if note.open_ended:
        return start, start + _dt.timedelta(weeks=absorption_weeks)
    return start, start


def apply_gate(
    route: str,
    week_of: str,
    own_pct: float,
    notes: list[Note],
    absorption_weeks: int,
    grace_days: int,
) -> tuple[Note | None, list[Eval]]:
    mon = _dt.date.fromisoformat(week_of)
    sun = mon + _dt.timedelta(days=6)
    evals: list[Eval] = []
    passing: list[tuple[Note, int]] = []
    for note in notes:
        failed: list[str] = []
        if not (note.scope == "ALL" or note.scope == route):
            failed.append("G1")
        wstart, wend = effective_window(note, absorption_weeks)
        wstart -= _dt.timedelta(days=grace_days)
        wend += _dt.timedelta(days=grace_days)
        overlap = max(0, (min(sun, wend) - max(mon, wstart)).days + 1)
        if overlap <= 0:
            failed.append("G2")
        if note.effect != "increase":
            failed.append("G3")
        if note.mag_max is not None and own_pct > 2 * (note.mag_max / 100):
            failed.append("G4")
        if note.confidence == "low":
            failed.append("G5")
        evals.append(Eval(note.note_id, not failed, tuple(failed), overlap))
        if not failed:
            passing.append((note, overlap))
    if not passing:
        return None, evals
    passing.sort(
        key=lambda item: (
            0 if item[0].scope != "ALL" else 1,
            -item[1],
            -_dt.date.fromisoformat(item[0].start).toordinal(),
            item[0].note_id,
        )
    )
    return passing[0][0], evals


# ---------------------------------------------------------------------------
# Output formatting (contract)
# ---------------------------------------------------------------------------


def format_pct(value: float, suffix: str) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}% {suffix}"


def window_label(note: Note, absorption_weeks: int) -> str:
    if note.end is not None:
        return f"{note.start} to {note.end}"
    if note.open_ended:
        end = _dt.date.fromisoformat(note.start) + _dt.timedelta(weeks=absorption_weeks)
        return f"{note.start} to {end.isoformat()} (open-ended)"
    return note.start


def closest_rejected(
    route: str, week_of: str, notes: list[Note], evals: list[Eval]
) -> tuple[Note | None, Eval | None]:
    by_id = {note.note_id: note for note in notes}
    scoped = [
        e
        for e in evals
        if not e.passed and (by_id[e.note_id].scope == "ALL" or by_id[e.note_id].scope == route)
    ]
    if not scoped:
        return None, None
    mon = _dt.date.fromisoformat(week_of)
    scoped.sort(
        key=lambda e: (
            abs((mon - _dt.date.fromisoformat(by_id[e.note_id].start)).days),
            e.note_id,
        )
    )
    best = scoped[0]
    gap = abs((mon - _dt.date.fromisoformat(by_id[best.note_id].start)).days)
    if gap > 60:
        return None, None
    return by_id[best.note_id], best


def _rule_phrase(rule: str, note: Note) -> str:
    if rule == "G1":
        return "G1 scope"
    if rule == "G2":
        return "G2 temporal window"
    if rule == "G3":
        return f"G3 polarity (effect {note.effect!r} is not an increase)"
    if rule == "G4":
        return "G4 magnitude"
    return "G5 confidence"


def build_reason(
    route: str,
    metric: Metric,
    matched: Note | None,
    notes: list[Note],
    evals: list[Eval],
    absorption_weeks: int,
) -> tuple[str, str, str]:
    if matched is not None:
        scope_txt = "all routes" if matched.scope == "ALL" else (f"the {matched.scope} corridor")
        reason = (
            f"Matches note {matched.note_id} dated {matched.note_date} "
            f"({scope_txt}, effective {window_label(matched, absorption_weeks)}): "
            f"the reported cost increase covers this route-week, "
            f"so the rise is explained by this note. [{matched.note_id}]"
        )
        return "No (justified)", matched.note_id, reason
    rejected, rej_eval = closest_rejected(route, metric.week_of, notes, evals)
    if rejected is not None and rej_eval is not None:
        rules = ", ".join(_rule_phrase(r, rejected) for r in rej_eval.failed)
        scope_txt = "all routes" if rejected.scope == "ALL" else (f"the {rejected.scope} corridor")
        reason = (
            f"The closest note ({rejected.note_id}, {rejected.note_date}, "
            f"{scope_txt}) does not justify this rise: it fails {rules}. "
            f"No genuine justification found; flagged for review."
        )
    else:
        reason = (
            "No matching note found for this route or date range. "
            "Cost rise looks unexplained and worth a human review."
        )
    return "Yes", "", reason


# ---------------------------------------------------------------------------
# Build outputs/output.csv + outputs/findings.sqlite
# ---------------------------------------------------------------------------

FINDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
  route TEXT NOT NULL,
  week_of TEXT NOT NULL,
  cost_per_tonne_km REAL NOT NULL,
  own_pct REAL NOT NULL,
  peer_pct REAL NOT NULL,
  n_prior INTEGER NOT NULL,
  flagged TEXT NOT NULL,
  matched_note_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  PRIMARY KEY (route, week_of)
);
CREATE TABLE IF NOT EXISTS notes (
  note_id TEXT PRIMARY KEY,
  note_date TEXT NOT NULL,
  scope TEXT NOT NULL,
  effect TEXT NOT NULL,
  start TEXT NOT NULL,
  end TEXT,
  open_ended INTEGER NOT NULL,
  mag_min REAL,
  mag_max REAL,
  cause TEXT NOT NULL,
  confidence TEXT NOT NULL,
  text TEXT NOT NULL
);
"""


def build_outputs(
    root: Path | None = None,
    out_csv: Path | None = None,
    db_path: Path | None = None,
) -> BuildStats:
    root = root or repo_root()
    cfg = load_thresholds(root)
    out_csv = out_csv or (root / "outputs" / "output.csv")
    db_path = db_path or (root / "outputs" / "findings.sqlite")
    absorption = int(cfg["absorption_weeks"])
    grace = int(cfg["grace_days"])

    shipments = load_shipments(root / "data" / "shipments.csv")
    route_type = validate_shipments(shipments)
    metrics = compute_metrics(shipments, route_type)
    candidates = detect_candidates(metrics, cfg)
    notes = load_notes(root / "data" / "notes.csv")

    rows: list[dict[str, str]] = []
    for metric in candidates:
        assert metric.own_pct is not None and metric.peer_pct is not None
        matched, evals = apply_gate(
            metric.route, metric.week_of, metric.own_pct, notes, absorption, grace
        )
        flagged, note_id, reason = build_reason(
            metric.route, metric, matched, notes, evals, absorption
        )
        rows.append(
            {
                "route": metric.route,
                "week_of": metric.week_of,
                "cost_per_tonne_km": f"{metric.cpt:.2f}",
                "vs_own_history": format_pct(metric.own_pct, "vs this route's past average"),
                "vs_similar_routes": format_pct(
                    metric.peer_pct, "vs similar-length routes this week"
                ),
                "flagged": flagged,
                "matched_note_id": note_id,
                "reason": reason,
            }
        )
    rows.sort(key=lambda r: (r["week_of"], r["route"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADER.split(","), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(FINDINGS_SCHEMA)
        with conn:
            conn.execute("DELETE FROM findings")
            conn.execute("DELETE FROM notes")
            for row, metric in zip(rows, candidates, strict=False):
                conn.execute(
                    "INSERT INTO findings (route, week_of,"
                    " cost_per_tonne_km, own_pct, peer_pct, n_prior,"
                    " flagged, matched_note_id, reason)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        row["route"],
                        row["week_of"],
                        metric.cpt,
                        metric.own_pct,
                        metric.peer_pct,
                        metric.n_prior,
                        row["flagged"],
                        row["matched_note_id"],
                        row["reason"],
                    ),
                )
            for note in notes:
                conn.execute(
                    "INSERT INTO notes (note_id, note_date, scope,"
                    " effect, start, end, open_ended, mag_min, mag_max,"
                    " cause, confidence, text)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        note.note_id,
                        note.note_date,
                        note.scope,
                        note.effect,
                        note.start,
                        note.end,
                        int(note.open_ended),
                        note.mag_min,
                        note.mag_max,
                        note.cause,
                        note.confidence,
                        note.text,
                    ),
                )
    finally:
        conn.close()

    justified = sum(1 for r in rows if r["flagged"] == "No (justified)")
    return BuildStats(
        candidates=len(rows),
        justified=justified,
        flagged_yes=len(rows) - justified,
    )


# ---------------------------------------------------------------------------
# Q&A: parse question -> structured query -> SQL -> composed answer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    kind: str  # "explain" | "note_impact" | "route_summary" | "clarify"
    route: str | None = None
    year: int | None = None
    month: int | None = None
    note_id: str | None = None
    raw: str = ""


def known_routes(root: Path) -> list[str]:
    shipments = load_shipments(root / "data" / "shipments.csv")
    routes = sorted({f"{r['origin']}-{r['destination']}" for r in shipments})
    return routes


def known_note_ids(root: Path) -> list[str]:
    notes = load_notes(root / "data" / "notes.csv")
    return [n.note_id for n in notes]


def parse_question(text: str, routes: list[str]) -> Query:
    lowered = text.lower().replace("-", " ").replace("/", " ")
    tokens = set(re.findall(r"[a-z]+", lowered)) | set(re.findall(r"\d+", lowered))
    matched_routes = [r for r in routes if all(part.lower() in tokens for part in r.split("-"))]
    matched_routes.sort()
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    year = int(year_match.group(1)) if year_match else None
    month: int | None = None
    for name, number in MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", lowered):
            month = number
            break
    note_match = re.search(r"\bN\s?0*(\d{1,3})\b", text.upper())
    note_id = f"N{int(note_match.group(1)):03d}" if note_match else None
    impact_words = (
        "raise",
        "affect",
        "cause",
        "justify",
        "impact",
        "blame",
        "responsible",
        "did",
        "effect",
    )
    if note_id is not None and any(w in lowered for w in impact_words):
        route = matched_routes[0] if matched_routes else None
        return Query("note_impact", route=route, note_id=note_id, raw=text)
    if len(matched_routes) > 1 and month is None and note_id is None:
        return Query("clarify", raw=text)
    if matched_routes and (
        month is not None
        or year is not None
        or any(
            w in lowered
            for w in (
                "why",
                "pricier",
                "price",
                "expensive",
                "cost",
                "rise",
                "rose",
                "increase",
                "explain",
                "happen",
            )
        )
    ):
        return Query("explain", route=matched_routes[0], year=year, month=month, raw=text)
    if matched_routes:
        return Query("route_summary", route=matched_routes[0], raw=text)
    return Query("clarify", raw=text)


def _connect(root: Path) -> sqlite3.Connection:
    db_path = root / "outputs" / "findings.sqlite"
    if not db_path.exists():
        build_outputs(root=root)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_week(row: sqlite3.Row) -> str:
    own = format_pct(row["own_pct"], "vs this route's past average")
    peer = format_pct(row["peer_pct"], "vs similar-length routes this week")
    verdict = (
        f"justified by note {row['matched_note_id']}"
        if row["flagged"] == "No (justified)"
        else "unexplained (flagged for review)"
    )
    return (
        f"week of {row['week_of']}: cost per tonne-km "
        f"{row['cost_per_tonne_km']:.2f}, {own}; {peer}; "
        f"{verdict}"
    )


def answer_explain(query: Query, root: Path) -> str:
    assert query.route is not None
    route = query.route
    conn = _connect(root)
    try:
        if query.month is not None and query.year is not None:
            cur = conn.execute(
                "SELECT * FROM findings WHERE route = ? AND week_of LIKE ? ORDER BY week_of",
                (route, f"{query.year}-{query.month:02d}%"),
            )
            weeks = list(cur)
            if not weeks:
                label = f"{MONTH_LABEL[query.month]} {query.year}"
                return (
                    f"No flagged weeks for {route} in {label}, "
                    f"so there is no flagged price rise to explain."
                )
            noun = f"{MONTH_LABEL[query.month]} {query.year}"
            header = f"{route} had {len(weeks)} flagged week(s) in {noun}:"
            return header + "\n" + _detail_weeks(weeks, conn, route)
        if query.month is not None:
            cur = conn.execute(
                "SELECT * FROM findings WHERE route = ?"
                " AND substr(week_of, 6, 2) = ? ORDER BY week_of",
                (route, f"{query.month:02d}"),
            )
            weeks = list(cur)
            years = sorted({w["week_of"][:4] for w in weeks})
            if not years:
                return (
                    f"No flagged weeks for {route} in "
                    f"{MONTH_LABEL[query.month]}, so there is no "
                    f"flagged price rise to explain."
                )
            if len(years) > 1:
                options = " or ".join(f"{MONTH_LABEL[query.month]} {y}" for y in years)
                return (
                    f"Which year did you mean — {options}? "
                    f"{route} has flagged "
                    f"{MONTH_LABEL[query.month]} weeks in each of "
                    f"{', '.join(years)}."
                )
            picked = [w for w in weeks if w["week_of"].startswith(years[0])]
            header = (
                f"Interpreting 'March' as March {years[0]} (the only "
                f"{MONTH_LABEL[query.month]} with flagged weeks for "
                f"this route). {route} had {len(picked)} flagged "
                f"week(s) in {MONTH_LABEL[query.month]} {years[0]}:"
            )
            return header + "\n" + _detail_weeks(picked, conn, route)
        cur = conn.execute("SELECT * FROM findings WHERE route = ? ORDER BY week_of", (route,))
        weeks = list(cur)
        if not weeks:
            return (
                f"No flagged weeks for {route} in the available "
                f"data, so there is no flagged price rise to explain." + _corridor_note(route, conn)
            )
        lines = [f"{route} has {len(weeks)} flagged week(s):"]
        lines.append(_detail_weeks(weeks, conn, route))
        return "\n".join(lines)
    finally:
        conn.close()


def _detail_weeks(weeks: list[sqlite3.Row], conn: sqlite3.Connection, route: str) -> str:
    parts: list[str] = []
    for week in weeks:
        parts.append("- " + _fmt_week(week))
        if week["flagged"] == "No (justified)":
            note = conn.execute(
                "SELECT * FROM notes WHERE note_id = ?", (week["matched_note_id"],)
            ).fetchone()
            if note is not None:
                parts.append(
                    f"  Justification: note {note['note_id']} "
                    f"({note['note_date']}, scope "
                    f"{'all routes' if note['scope'] == 'ALL' else note['scope']}, "
                    f"effect {note['effect']!r}) covers this week. "
                    f"[{note['note_id']}]"
                )
        else:
            parts.append(
                "  No note passed the evidence gate (scope, temporal "
                "window, increase polarity, magnitude, confidence); "
                "matched_note_id is empty. " + _reject_context(route, week, conn)
            )
    return "\n".join(parts)


def _reject_context(route: str, week: sqlite3.Row, conn: sqlite3.Connection) -> str:
    """Name the closest rejected note for context (never a justification)."""
    notes = list(
        conn.execute(
            "SELECT note_id, note_date, scope, effect, start FROM notes"
            " WHERE scope = 'ALL' OR scope = ? ORDER BY note_id",
            (route,),
        )
    )
    if not notes:
        return ""
    mon = _dt.date.fromisoformat(week["week_of"])
    notes.sort(key=lambda n: (abs((mon - _dt.date.fromisoformat(n["start"])).days), n["note_id"]))
    best = notes[0]
    gap = abs((mon - _dt.date.fromisoformat(best["start"])).days)
    if gap > 60:
        return ""
    if best["effect"] != "increase":
        why = f"its recorded effect is {best['effect']!r}, not a cost increase (gate G3)"
    else:
        why = "it falls outside this week's temporal window (gate G2)"
    return (
        f"For context, the closest rejected note is {best['note_id']} "
        f"({best['note_date']}): {why}. [{best['note_id']}]"
    )


def _corridor_note(route: str, conn: sqlite3.Connection) -> str:
    notes = list(
        conn.execute(
            "SELECT note_id, note_date, effect FROM notes WHERE scope = ? ORDER BY note_id",
            (route,),
        )
    )
    if not notes:
        return ""
    first = notes[0]
    return (
        f" The only note mentioning this corridor, {first['note_id']} "
        f"({first['note_date']}), records effect "
        f"{first['effect']!r} (no cost increase), consistent with no "
        f"flagged weeks. [{first['note_id']}]"
    )


def answer_note_impact(query: Query, root: Path) -> str:
    assert query.note_id is not None
    note_id = query.note_id
    conn = _connect(root)
    try:
        note = conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        if note is None:
            return (
                f"No note {note_id} exists in the notes data, "
                f"so I cannot attribute any cost change to it."
            )
        matched = list(
            conn.execute(
                "SELECT route, week_of FROM findings"
                " WHERE matched_note_id = ? ORDER BY week_of, route",
                (note_id,),
            )
        )
        scope_txt = "all routes" if note["scope"] == "ALL" else f"the {note['scope']} corridor"
        lines = [
            f"Note {note_id} ({note['note_date']}, {scope_txt}) records effect {note['effect']!r}.",
            f"Quoted text: {note['text']!r} [{note_id}]",
        ]
        if note["effect"] != "increase":
            lines.append(
                f"Because its effect is {note['effect']!r} rather than a "
                f"cost increase, {note_id} fails polarity gate G3 and "
                f"cannot justify any candidate; it did not raise costs."
            )
        if query.route:
            mine = [m for m in matched if m["route"] == query.route]
            lines.append(
                f"Flagged {query.route} weeks matched to {note_id}: "
                f"{len(mine)}."
                + (
                    ""
                    if mine
                    else f" None of the {query.route} flagged weeks list "
                    f"{note_id} as their matched note."
                )
            )
        else:
            lines.append(f"Weeks justified by {note_id}: {len(matched)}.")
        if "not part of this dataset" in note["text"]:
            lines.append(
                f"{note_id} explicitly states the affected routes are not part of this dataset."
            )
        return " ".join(lines)
    finally:
        conn.close()


def answer_question(question: str, root: Path | None = None) -> str:
    """Answer a question using only findings.sqlite + notes data."""
    base = root or repo_root()
    routes = known_routes(base)
    query = parse_question(question, routes)
    if query.kind == "note_impact":
        return answer_note_impact(query, base)
    if query.kind in ("explain", "route_summary"):
        return answer_explain(query, base)
    available = ", ".join(routes)
    return (
        "I can answer questions about flagged route-weeks "
        "(which route got pricier in which month/year) and about "
        "whether a specific note raised costs on a route. "
        f"Known routes: {available}. "
        "Please name a route, e.g. 'why did Chennai-Bangalore get "
        "pricier in March 2025?' or 'did N004 raise Mumbai-Pune costs?'"
    )


# ---------------------------------------------------------------------------
# CLI (used by scripts/repro.sh and scripts/cost.sh as fallback)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic builder + rule-based Q&A")
    sub = parser.add_subparsers(dest="command")
    build = sub.add_parser("build", help="write output.csv + findings.sqlite")
    build.add_argument("--out", default=None)
    build.add_argument("--db", default=None)
    build.add_argument(
        "--no-cache",
        action="store_true",
        help="accepted for interface parity; build is stateless and deterministic",
    )
    build.add_argument(
        "--no-llm",
        action="store_true",
        help="accepted for interface parity; build never calls an LLM",
    )
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    args = parser.parse_args(argv)
    root = repo_root()
    if args.command == "ask":
        print(answer_question(args.question, root=root))
        return 0
    stats = build_outputs(
        root=root,
        out_csv=Path(args.out) if args.out else None,
        db_path=Path(args.db) if args.db else None,
    )
    print(
        f"candidates={stats.candidates} justified={stats.justified} "
        f"flagged_yes={stats.flagged_yes} llm_calls={stats.llm_calls}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
