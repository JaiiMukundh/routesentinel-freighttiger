import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DEFAULT_CSV_URLS, loadOutputCsvWithFallback, parsePctLabel } from "./api";
import { EXAMPLE_QUESTIONS, answerQuestion, loadNotesCsv } from "./qa";
import type { OpsNote } from "./qa";
import type { FlagFilter, FlaggedRow, SortDir, SortKey } from "./types";

const SORT_LABELS: Record<SortKey, string> = {
  route: "Route",
  week_of: "Week",
  cost_per_tonne_km: "₹/t·km",
  vs_own_history: "vs own",
  vs_similar_routes: "vs peers",
  flagged: "Flag",
};

function sortValue(row: FlaggedRow, key: SortKey): string | number {
  switch (key) {
    case "route":
    case "week_of":
    case "flagged":
      return row[key];
    case "cost_per_tonne_km":
      return row.cost_per_tonne_km;
    case "vs_own_history":
      return parsePctLabel(row.vs_own_history);
    case "vs_similar_routes":
      return parsePctLabel(row.vs_similar_routes);
  }
}

function fmtPct(n: number): string {
  if (Number.isNaN(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
}

interface DeltaPoint {
  week: string;
  own: number;
  peer: number;
}

export default function App() {
  const [rows, setRows] = useState<FlaggedRow[] | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [flagFilter, setFlagFilter] = useState<FlagFilter>("all");
  const [routeQuery, setRouteQuery] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("week_of");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [chartRoute, setChartRoute] = useState<string | null>(null);
  const [notes, setNotes] = useState<OpsNote[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadOutputCsvWithFallback(DEFAULT_CSV_URLS)
      .then(({ rows: loaded, sourceUrl: url }) => {
        if (cancelled) return;
        setRows(loaded);
        setSourceUrl(url);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setLoadError(err instanceof Error ? err.message : String(err));
      });
    loadNotesCsv("./notes.csv")
      .then((n) => {
        if (!cancelled) setNotes(n);
      })
      .catch(() => {
        /* notes optional — QA degrades to rows-only citations */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const allRoutes = useMemo(() => {
    if (!rows) return [];
    return [...new Set(rows.map((r) => r.route))].sort();
  }, [rows]);

  useEffect(() => {
    if (!chartRoute && allRoutes.length > 0) setChartRoute(allRoutes[0]);
  }, [allRoutes, chartRoute]);

  const kpis = useMemo(() => {
    if (!rows || rows.length === 0) {
      return { total: 0, yes: 0, justified: 0, routes: 0, yesPct: 0, justPct: 0 };
    }
    const yes = rows.filter((r) => r.flagged === "Yes").length;
    const justified = rows.length - yes;
    return {
      total: rows.length,
      yes,
      justified,
      routes: new Set(rows.map((r) => r.route)).size,
      yesPct: (100 * yes) / rows.length,
      justPct: (100 * justified) / rows.length,
    };
  }, [rows]);

  const filtered = useMemo(() => {
    if (!rows) return [];
    const q = routeQuery.trim().toLowerCase();
    const out = rows.filter(
      (r) =>
        (flagFilter === "all" || r.flagged === flagFilter) &&
        (q === "" || r.route.toLowerCase().includes(q)),
    );
    const dir = sortDir === "asc" ? 1 : -1;
    out.sort((a, b) => {
      const va = sortValue(a, sortKey);
      const vb = sortValue(b, sortKey);
      if (va < vb) return -1 * dir;
      if (va > vb) return 1 * dir;
      // deterministic tie-break: (week_of, route)
      if (a.week_of !== b.week_of) return (a.week_of < b.week_of ? -1 : 1) * 1;
      return a.route < b.route ? -1 : 1;
    });
    return out;
  }, [rows, flagFilter, routeQuery, sortKey, sortDir]);

  const selected: FlaggedRow | null = useMemo(() => {
    if (!selectedKey || !rows) return null;
    return rows.find((r) => `${r.route}|${r.week_of}` === selectedKey) ?? null;
  }, [rows, selectedKey]);

  const lineData = useMemo(() => {
    if (!rows || !chartRoute) return [];
    return rows
      .filter((r) => r.route === chartRoute)
      .map((r) => ({ week: r.week_of.slice(5), full: r.week_of, cpt: r.cost_per_tonne_km }));
  }, [rows, chartRoute]);

  const deltaData: DeltaPoint[] = useMemo(() => {
    if (!rows || !chartRoute) return [];
    return rows
      .filter((r) => r.route === chartRoute)
      .map((r) => ({
        week: r.week_of.slice(5),
        own: parsePctLabel(r.vs_own_history),
        peer: parsePctLabel(r.vs_similar_routes),
      }));
  }, [rows, chartRoute]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <span className="hero-eyebrow">
          <span className="pulse" /> FreightTiger · freight-ops
        </span>
        <h1>RouteSentinel — FreightTiger</h1>
        <p>
          Flags unjustified freight-cost rises per route-week and explains them with retrieved
          corridor notes. Deterministic verdicts, evidence-gated justifications — this board reads
          the pipeline&apos;s <span className="mono">output.csv</span> candidates only.
        </p>
        <div className="hero-meta">
          {sourceUrl && (
            <span>
              source <code>{sourceUrl}</code>
            </span>
          )}
          <span>
            candidates only · sorted by <code>(week_of, route)</code>
          </span>
        </div>
      </header>

      {loadError && (
        <div className="error" role="alert">
          Could not load <span className="mono">output.csv</span>: {loadError}
          <br />
          For local dev, copy <span className="mono">outputs/output.csv</span> to{" "}
          <span className="mono">web/public/output.csv</span> and reload.
        </div>
      )}

      {!rows && !loadError && (
        <div className="loading">
          <div className="spinner" />
          Loading output.csv…
        </div>
      )}

      {rows && (
        <>
          <section className="kpis">
            <div className="kpi">
              <div className="kpi-label">Candidates</div>
              <div className="kpi-value">{kpis.total}</div>
              <div className="kpi-sub">route-weeks breaching own + peer thresholds</div>
            </div>
            <div className="kpi amber">
              <div className="kpi-label">Unexplained (Yes)</div>
              <div className="kpi-value">{kpis.yesPct.toFixed(1)}%</div>
              <div className="kpi-sub">
                {kpis.yes} of {kpis.total} flagged for human review
              </div>
            </div>
            <div className="kpi green">
              <div className="kpi-label">Justified</div>
              <div className="kpi-value">{kpis.justPct.toFixed(1)}%</div>
              <div className="kpi-sub">
                {kpis.justified} matched a passing evidence gate
              </div>
            </div>
            <div className="kpi">
              <div className="kpi-label">Routes covered</div>
              <div className="kpi-value">{kpis.routes}</div>
              <div className="kpi-sub">distinct directed corridors</div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Cost trajectory</h2>
              <span className="hint">₹/tonne·km over weeks + own/peer deltas</span>
            </div>
            <div className="route-pills">
              {allRoutes.map((r) => (
                <button
                  key={r}
                  className={r === chartRoute ? "pill active" : "pill"}
                  onClick={() => setChartRoute(r)}
                >
                  {r}
                </button>
              ))}
            </div>
            <div className="charts">
              <div className="chart-box">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={lineData} margin={{ top: 10, right: 18, bottom: 0, left: -8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
                    <XAxis dataKey="week" tick={{ fill: "#6b7a93", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#6b7a93", fontSize: 11 }} domain={["auto", "auto"]} />
                    <Tooltip
                      contentStyle={{
                        background: "#0b1220",
                        border: "1px solid rgba(255,255,255,0.12)",
                        borderRadius: 10,
                        fontSize: 12,
                      }}
                      labelFormatter={(_, payload) =>
                        payload && payload.length > 0
                          ? String(
                              (payload[0].payload as { full: string }).full,
                            )
                          : ""
                      }
                    />
                    <Line
                      type="monotone"
                      dataKey="cpt"
                      name="₹/t·km"
                      stroke="#38bdf8"
                      strokeWidth={2.5}
                      dot={{ r: 3, fill: "#38bdf8" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="chart-box">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={deltaData} margin={{ top: 10, right: 18, bottom: 0, left: -8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
                    <XAxis dataKey="week" tick={{ fill: "#6b7a93", fontSize: 11 }} />
                    <YAxis
                      tick={{ fill: "#6b7a93", fontSize: 11 }}
                      tickFormatter={(v: number) => `${v}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0b1220",
                        border: "1px solid rgba(255,255,255,0.12)",
                        borderRadius: 10,
                        fontSize: 12,
                      }}
                      formatter={(value) => [
                        `${fmtPct(Number(value))}`,
                        "",
                      ]}
                    />
                    <ReferenceLine y={0} stroke="rgba(255,255,255,0.25)" />
                    <Bar dataKey="own" name="vs own history" fill="#818cf8" radius={[4, 4, 0, 0]}>
                      {deltaData.map((d, i) => (
                        <Cell key={i} fill={d.own >= 0 ? "#fbbf24" : "#34d399"} />
                      ))}
                    </Bar>
                    <Bar dataKey="peer" name="vs peers" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          <section className="panel ask-panel">
            <div className="panel-head">
              <h2>Ask the evidence</h2>
              <span className="hint">grounded Q&A — answers only from loaded rows + notes</span>
            </div>
            <div className="ask-row">
              <label className="search ask-input">
                <span aria-hidden="true">?</span>
                <input
                  placeholder="e.g. why did Chennai-Bangalore get pricier in March 2025?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && rows) setAnswer(answerQuestion(question, rows, notes));
                  }}
                />
              </label>
              <button
                className="pill active"
                onClick={() => {
                  if (rows) setAnswer(answerQuestion(question, rows, notes));
                }}
              >
                Ask
              </button>
            </div>
            <div className="route-pills">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  className="pill"
                  onClick={() => {
                    setQuestion(q);
                    if (rows) setAnswer(answerQuestion(q, rows, notes));
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
            {answer && <div className="answer-box">{answer}</div>}
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Candidates</h2>
              <span className="hint">click a row for the evidence drawer</span>
            </div>
            <div className="toolbar">
              <label className="search">
                <span aria-hidden="true">⌕</span>
                <input
                  placeholder="Search route… (e.g. Delhi-Jaipur)"
                  value={routeQuery}
                  onChange={(e) => setRouteQuery(e.target.value)}
                />
              </label>
              <div className="seg" role="tablist" aria-label="flag filter">
                {(
                  [
                    ["all", "All"],
                    ["Yes", "Unexplained"],
                    ["No (justified)", "Justified"],
                  ] as [FlagFilter, string][]
                ).map(([v, label]) => (
                  <button
                    key={v}
                    className={flagFilter === v ? "active" : ""}
                    onClick={() => setFlagFilter(v)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <span className="count">
                {filtered.length} / {rows.length} rows
              </span>
            </div>

            {filtered.length === 0 ? (
              <div className="empty">No rows match the current filters.</div>
            ) : (
              <div className="table-wrap">
                <table className="grid">
                  <thead>
                    <tr>
                      {(Object.keys(SORT_LABELS) as SortKey[]).map((key) => (
                        <th key={key} onClick={() => toggleSort(key)}>
                          {SORT_LABELS[key]}{" "}
                          {sortKey === key ? (sortDir === "asc" ? "▲" : "▼") : ""}
                        </th>
                      ))}
                      <th>Note</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((r) => {
                      const key = `${r.route}|${r.week_of}`;
                      return (
                        <tr
                          key={key}
                          className={selectedKey === key ? "selected" : ""}
                          onClick={() => setSelectedKey(key)}
                        >
                          <td>
                            <strong>{r.route}</strong>
                          </td>
                          <td className="mono">{r.week_of}</td>
                          <td className="mono">{r.cost_per_tonne_km.toFixed(2)}</td>
                          <td className="mono">{r.vs_own_history}</td>
                          <td className="mono">{r.vs_similar_routes}</td>
                          <td>
                            <span
                              className={r.flagged === "Yes" ? "badge yes" : "badge justified"}
                            >
                              {r.flagged === "Yes" ? "● Yes" : "✓ Justified"}
                            </span>
                          </td>
                          <td>
                            {r.matched_note_id ? (
                              <span className="mono note-id">{r.matched_note_id}</span>
                            ) : (
                              <span className="mono" style={{ color: "#6b7a93" }}>
                                —
                              </span>
                            )}
                          </td>
                          <td>
                            <span className="reason-cell">{r.reason}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <div className="footer">
            <span>
              RouteSentinel · deterministic verdicts — the UI never decides flags, it only renders{" "}
              <span className="mono">outputs/output.csv</span>.
            </span>
            <span>Similarity scores never justify a row; only the 5-gate evidence check does.</span>
          </div>
        </>
      )}

      {selected && (
        <>
          <div className="drawer-backdrop" onClick={() => setSelectedKey(null)} />
          <aside className="drawer" role="dialog" aria-label="candidate detail">
            <div className="drawer-top">
              <div>
                <h3>{selected.route}</h3>
                <div className="week mono">week of {selected.week_of} (Mon–Sun)</div>
              </div>
              <button className="close-btn" onClick={() => setSelectedKey(null)} aria-label="Close">
                ✕
              </button>
            </div>
            <div style={{ marginTop: 12 }}>
              <span className={selected.flagged === "Yes" ? "badge yes" : "badge justified"}>
                {selected.flagged === "Yes" ? "● Yes — unexplained" : "✓ No (justified)"}
              </span>
            </div>
            <div className="metric-grid">
              <div className="metric">
                <div className="k">Cost / t·km</div>
                <div className="v mono">₹{selected.cost_per_tonne_km.toFixed(2)}</div>
              </div>
              <div className="metric">
                <div className="k">Matched note</div>
                <div className="v mono">
                  {selected.matched_note_id || <span style={{ color: "#6b7a93" }}>— none</span>}
                </div>
              </div>
              <div className="metric">
                <div className="k">vs own history</div>
                <div className="v mono">{selected.vs_own_history}</div>
              </div>
              <div className="metric">
                <div className="k">vs similar routes</div>
                <div className="v mono">{selected.vs_similar_routes}</div>
              </div>
            </div>
            <h4>Reason</h4>
            <div className="reason-box">{selected.reason}</div>
            <h4>Contract reminder</h4>
            <div style={{ fontSize: 12.5, color: "#aab6cc", lineHeight: 1.6 }}>
              <span className="mono">matched_note_id</span> is empty iff flagged is{" "}
              <span className="mono">Yes</span>. A reason may cite the closest rejected note, but
              only a row that passes all five evidence gates (scope · temporal · polarity ·
              magnitude · confidence) carries a note id.
            </div>
          </aside>
        </>
      )}
    </div>
  );
}
