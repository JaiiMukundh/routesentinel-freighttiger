import type { FlaggedRow } from "./types";

export interface OpsNote {
  note_id: string;
  date: string;
  applies_to: string;
  note: string;
}

const MONTHS: Record<string, string> = {
  january: "01",
  february: "02",
  march: "03",
  april: "04",
  may: "05",
  june: "06",
  july: "07",
  august: "08",
  september: "09",
  october: "10",
  november: "11",
  december: "12",
};

export async function loadNotesCsv(url: string): Promise<OpsNote[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  const text = await res.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) return [];
  // minimal CSV parse honoring quoted note field
  const out: OpsNote[] = [];
  for (const line of lines.slice(1)) {
    const m = /^([^,]+),([^,]+),([^,]+),"(.*)"\s*$/.exec(line) ?? /^([^,]+),([^,]+),([^,]+),(.*)$/.exec(line);
    if (!m) continue;
    out.push({ note_id: m[1].trim(), date: m[2].trim(), applies_to: m[3].trim(), note: (m[4] ?? "").trim() });
  }
  return out;
}

function findRoute(q: string, routes: string[]): string | null {
  const low = q.toLowerCase();
  const hits = routes.filter((r) => low.includes(r.toLowerCase()));
  if (hits.length === 1) return hits[0];
  // also accept "mumbai pune" without dash
  for (const r of routes) {
    const [a, b] = r.split("-");
    if (a && b && low.includes(a.toLowerCase()) && low.includes(b.toLowerCase())) return r;
  }
  return hits.length > 1 ? hits.sort()[0] : null;
}

function findYear(q: string): number | null {
  const m = /(19|20)\d{2}/.exec(q);
  return m ? Number(m[0]) : null;
}

function findMonth(q: string): string | null {
  const low = q.toLowerCase();
  for (const [name, num] of Object.entries(MONTHS)) {
    if (low.includes(name) || low.includes(name.slice(0, 3))) return num;
  }
  const m = /\b(0?[1-9]|1[0-2])\b/.exec(q);
  void m;
  return null;
}

function findNoteId(q: string): string | null {
  const m = /N0\d{2}/i.exec(q);
  return m ? m[0].toUpperCase() : null;
}

/** Grounded client-side Q&A over loaded dashboard rows + notes. Never invents facts. */
export function answerQuestion(q: string, rows: FlaggedRow[], notes: OpsNote[]): string {
  const routes = [...new Set(rows.map((r) => r.route))].sort();
  const noteId = findNoteId(q);
  const route = findRoute(q, routes);
  const month = findMonth(q);
  const year = findYear(q);

  if (noteId) {
    const note = notes.find((n) => n.note_id === noteId);
    if (!note) return `No note ${noteId} in the loaded notes. Known notes: ${notes.map((n) => n.note_id).join(", ")}.`;
    const matched = rows.filter((r) => r.matched_note_id === noteId);
    const scope = note.applies_to;
    if (route && !scope.toLowerCase().includes("all") && scope.toLowerCase() !== route.toLowerCase()) {
      return `Note ${noteId} (${note.date}, scope ${scope}) does not apply to ${route} — wrong route, so it cannot justify ${route} weeks. [${noteId}]`;
    }
    if (matched.length === 0) {
      const routeRows = route ? rows.filter((r) => r.route === route) : [];
      const suffix =
        route && routeRows.length > 0
          ? ` Flagged ${route} weeks matched to ${noteId}: 0.`
          : ` Flagged weeks matched to ${noteId}: 0.`;
      return `Note ${noteId} (${note.date}, scope ${scope}): "${note.note}" [${noteId}] It justifies no loaded candidate row (fails scope/temporal/polarity at the evidence gate for these weeks).${suffix}`;
    }
    const lines = matched.map(
      (r) => `- ${r.route} week of ${r.week_of}: ${r.cost_per_tonne_km.toFixed(2)} ₹/t·km, ${r.vs_own_history}; ${r.vs_similar_routes}.`,
    );
    return `Note ${noteId} (${note.date}, scope ${scope}) justifies ${matched.length} loaded week(s):\n${lines.join("\n")}\n[${noteId}]`;
  }

  if (!route) {
    return `Which route? Name one, e.g. 'why did Chennai-Bangalore get pricier in March 2025?' Known routes: ${routes.join(", ")}.`;
  }
  const routeRows = rows.filter((r) => r.route === route);
  if (routeRows.length === 0) {
    return `No flagged weeks for ${route} in the available data, so there is no flagged price rise to explain.`;
  }
  let scoped = routeRows;
  if (month && year) {
    scoped = routeRows.filter((r) => r.week_of.startsWith(`${year}-${month}`));
  } else if (month && !year) {
    const years = [...new Set(routeRows.map((r) => r.week_of.slice(0, 4)))].sort();
    if (years.length > 1) {
      return `Which year? ${route} has flagged weeks in ${years.join(" and ")} — e.g. 'why did ${route} get pricier in March ${years[years.length - 1]}?'`;
    }
    scoped = routeRows.filter((r) => r.week_of.slice(5, 7) === month);
  } else if (!month && year) {
    scoped = routeRows.filter((r) => r.week_of.startsWith(String(year)));
  }
  if (scoped.length === 0) {
    return `No flagged ${route} weeks for that period in the available data. Flagged ${route} weeks: ${routeRows.map((r) => r.week_of).join(", ")}.`;
  }
  const head = `${route} had ${scoped.length} flagged week(s)${month && year ? ` in ${year}-${month}` : ""}:`;
  const lines = scoped.map((r) => {
    const tag = r.flagged === "Yes" ? "unexplained (flagged for review)" : `justified by note ${r.matched_note_id} [${r.matched_note_id}]`;
    return `- week of ${r.week_of}: ${r.cost_per_tonne_km.toFixed(2)} ₹/t·km, ${r.vs_own_history}; ${r.vs_similar_routes}; ${tag}.`;
  });
  return `${head}\n${lines.join("\n")}`;
}

export const EXAMPLE_QUESTIONS = [
  "why did Chennai-Bangalore get pricier in March 2025?",
  "why did Kolkata-Bhubaneswar get pricier?",
  "did N004 raise Mumbai-Pune costs?",
];
