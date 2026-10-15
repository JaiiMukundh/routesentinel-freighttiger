/**
 * Row shape of `outputs/output.csv` (the RouteSentinel output contract).
 *
 * Header, exactly:
 * route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason
 */
export interface FlaggedRow {
  route: string;
  /** Monday of the week, `YYYY-MM-DD`. */
  week_of: string;
  cost_per_tonne_km: number;
  /** e.g. `+35.5% vs this route's past average` (sign always shown, 1 decimal). */
  vs_own_history: string;
  /** e.g. `+21.0% vs similar-length routes this week`. */
  vs_similar_routes: string;
  /** Exactly `Yes` or `No (justified)`. */
  flagged: "Yes" | "No (justified)";
  /** Note id, or empty string iff flagged === "Yes". */
  matched_note_id: string;
  reason: string;
}

export type FlagFilter = "all" | "Yes" | "No (justified)";

export type SortKey =
  | "route"
  | "week_of"
  | "cost_per_tonne_km"
  | "vs_own_history"
  | "vs_similar_routes"
  | "flagged";

export type SortDir = "asc" | "desc";
