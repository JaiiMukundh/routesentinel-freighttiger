# RouteSentinel dashboard (web/)

TypeScript + Vite + React dashboard for RouteSentinel. No backend — it parses
`outputs/output.csv` client-side with PapaParse and renders KPIs, charts (recharts),
a filterable/sortable candidates table, and a row-click evidence drawer.

The UI never decides verdicts; it only renders the deterministic pipeline output.

## Run

```bash
cd web
npm install
npm run dev      # → http://localhost:5173
```

The app looks for the CSV at (first hit wins):

1. `./output.csv` and `/output.csv` — i.e. `web/public/output.csv` (recommended)
2. `outputs/output.csv` / `/outputs/output.csv` — repo-relative fallback

So for local dev:

```bash
# from the repo root
cp outputs/output.csv web/public/output.csv
cd web && npm install && npm run dev
```

> `web/public/` is git-ignored if you prefer; just copy the file there before `dev`/`build`.
> If no CSV is found the app shows an error panel instead of fake data.

## Build

```bash
cd web
npm install
npm run build      # tsc --noEmit && vite build → web/dist/
npm run preview    # serve the production build locally
```

## Files

| File | Purpose |
|---|---|
| `package.json` | react, react-dom, vite, typescript, papaparse, recharts |
| `vite.config.ts` | Vite + `@vitejs/plugin-react`, `base: "./"` for static hosting |
| `index.html` | `#root` mount + Inter font + module entry |
| `src/types.ts` | `FlaggedRow` (output contract), filter/sort types |
| `src/api.ts` | `loadOutputCsv(url)` via papaparse → rows sorted by `(week_of, route)` |
| `src/App.tsx` | KPI cards, charts, table, evidence drawer |
| `src/index.css` | hand-rolled dark glassy freight-ops theme (no CSS framework) |
| `src/main.tsx` | React mount wiring |
