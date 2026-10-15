import Papa from "papaparse";
import type { FlaggedRow } from "./types";

const EXPECTED_COLUMNS = [
  "route",
  "week_of",
  "cost_per_tonne_km",
  "vs_own_history",
  "vs_similar_routes",
  "flagged",
  "matched_note_id",
  "reason",
] as const;

/** Parse a `+35.5% vs ...` label back to a number (35.5). Returns NaN when unparseable. */
export function parsePctLabel(label: string): number {
  const m = /([+-]?\d+(?:\.\d+)?)\s*%/.exec(label);
  return m ? Number(m[1]) : NaN;
}

function toFlaggedRow(raw: Record<string, string>): FlaggedRow | null {
  const route = (raw["route"] ?? "").trim();
  const week_of = (raw["week_of"] ?? "").trim();
  if (!route || !week_of) return null;
  const flagged = (raw["flagged"] ?? "").trim();
  if (flagged !== "Yes" && flagged !== "No (justified)") return null;
  return {
    route,
    week_of,
    cost_per_tonne_km: Number(raw["cost_per_tonne_km"] ?? NaN),
    vs_own_history: (raw["vs_own_history"] ?? "").trim(),
    vs_similar_routes: (raw["vs_similar_routes"] ?? "").trim(),
    flagged,
    matched_note_id: (raw["matched_note_id"] ?? "").trim(),
    reason: (raw["reason"] ?? "").trim(),
  };
}

function parseCsvText(csvText: string): FlaggedRow[] {
  const parsed = Papa.parse<Record<string, string>>(csvText, {
    header: true,
    skipEmptyLines: true,
  });
  const rows: FlaggedRow[] = [];
  for (const record of parsed.data) {
    const row = toFlaggedRow(record);
    if (row) rows.push(row);
  }
  rows.sort((a, b) =>
    a.week_of < b.week_of
      ? -1
      : a.week_of > b.week_of
        ? 1
        : a.route < b.route
          ? -1
          : a.route > b.route
            ? 1
            : 0,
  );
  return rows;
}

export function validateColumns(csvText: string): string[] {
  const head = csvText.split(/\r?\n/, 1)[0] ?? "";
  const actual = head.split(",").map((c) => c.trim());
  const missing = EXPECTED_COLUMNS.filter((c) => !actual.includes(c));
  return missing as string[];
}

/**
 * Fetch a CSV url and return typed rows sorted by (week_of, route).
 * Throws on network error or empty result.
 */
export async function loadOutputCsv(url: string): Promise<FlaggedRow[]> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`GET ${url} -> HTTP ${res.status}`);
  }
  const text = await res.text();
  const rows = parseCsvText(text);
  if (rows.length === 0) {
    throw new Error(`No data rows parsed from ${url}`);
  }
  return rows;
}

/**
 * Try each candidate URL in order (for dev vs. built-static layouts)
 * and return the first that yields rows.
 */
export async function loadOutputCsvWithFallback(urls: string[]): Promise<{
  rows: FlaggedRow[];
  sourceUrl: string;
}> {
  const errors: string[] = [];
  for (const url of urls) {
    try {
      const rows = await loadOutputCsv(url);
      return { rows, sourceUrl: url };
    } catch (err) {
      errors.push(`${url}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  throw new Error(`Could not load output.csv. Tried: ${errors.join(" | ")}`);
}

/** Default lookup order: vite `public/` first, then repo-relative paths. */
export const DEFAULT_CSV_URLS = [
  "./output.csv",
  "/output.csv",
  "outputs/output.csv",
  "/outputs/output.csv",
];
