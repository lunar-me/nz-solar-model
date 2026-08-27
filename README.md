# NZ Solar PV Model

A physics-first, idealized PV output model built on **pvlib-python**, driven by
**CAMS radiation** data for two New Zealand cities (**Auckland** and
**Christchurch**), exposed through a **FastAPI** backend and a small **React**
frontend for choosing a location, configuring a panel, and viewing the results.

Implements the plan: CAMS irradiance → solar position →
angle of incidence → Plane-of-Array (POA) irradiance → idealized DC/AC power →
15-minute energy, using established pvlib algorithms rather than hand-written
astronomy.

## Tabs

- **Daily** — 15-minute PV output, irradiance and sun geometry for a chosen date range.
- **Year** — monthly/weekly PV output for a calendar year.
- **Stability** — year-over-year PV output.
- **My money** — solar self-consumption & savings against Christchurch's real hourly electricity bill (locked to Christchurch).
- **Model money** — solar savings against *modelled* hourly consumption: your annual kWh spread over a region's real 2025 generation curve (`region_electricity_generation_2025_1h`), priced at a flat per-kWh rate plus a fixed daily charge. Auckland uses the Waikato region; Christchurch uses Canterbury.
- **Curves** — daily generation (MWh) of every region on each island, with per-region toggles, a log-scale option, and a daily-totals table.
- **Data quality** — CAMS radiation dataset quality report.

## Model boundary

The model computes, for every 15-minute interval:

```
timestamp, sun_elevation, sun_azimuth, AOI,
GHI, DHI, DNI, POA_direct, POA_diffuse, POA_ground, POA_global,
DC power (W), AC power (W), energy (Wh)
```

Inputs: location (lat/longitude/altitude pinned per location in `locations.py`,
since the Supabase table stores irradiance but not coordinates), panel
tilt/azimuth, rated power (kWp), albedo, transposition model, inverter
efficiency.

This v1 is **idealized** by design: no module-temperature,
no soiling/mismatch/shading/degradation losses. Each of those is an explicit
configuration knob to be added later without touching the core geometry.

## Coordinate conventions

- **Azimuth**: 0° = N, 90° = E, 180° = S, 270° = W (meteorological convention).
- **Tilt**: 0° = horizontal, 90° = vertical.
- Default for NZ is a north-ish facing tilt of 25°, but a south-facing roof is
  `azimuth=180`.

## Time handling

Both Supabase tables store datetimes in **UTC** (`cams_radiation.start_ts_utc`,
`christchurch_electricity_consumption.datetime_utc`,
`region_electricity_generation_2025_1h.datetime_utc`). The loader keeps a
timezone-aware UTC `DatetimeIndex`; pvlib solar position is computed on those
UTC stamps. The frontend displays them converted to **Pacific/Auckland** local
time (handles NZST/NZDT automatically). No naive datetimes are used internally.

## Units note (important)

The Supabase `cams_radiation` table already stores average irradiances in
**W/m²** (not Wh/m² per interval), so the loader passes them straight to pvlib
and energy is re-computed as `power_W × interval_hours`. The loader normalizes
the Supabase column names to the internal schema
(`ghi / dhi / dni / *_clear / reliability`).

## Repository layout

```
data/            Legacy CAMS CSVs — reference only (the app reads Supabase)
api/             FastAPI backend — also Vercel's Python entrypoint
  index.py         FastAPI app (exposes `app` + serves the built SPA)
  locations.py     location registry (the "switch" keyed by name)
  supabase_client.py  Supabase/PostgREST access (paginated, parallel fetches)
  loader.py        Supabase → normalized, W/m², UTC-indexed DataFrame
  engine.py        pvlib solar position / POA / idealized PV + summary
  schemas.py       Pydantic request models
requirements.txt Python dependencies (used locally and on Vercel)
vercel.json     Vercel config (minimal — the Python framework preset auto-detects)
.python-version 3.12 (pins the Python runtime Vercel installs)
frontend/        React (Vite) app: pick location, configure panel, view charts
tests/           pytest suite (physics + data validation; uses Supabase)
```

## Run it

### Backend (FastAPI)

The backend reads all data from **Supabase**. Create a `.env` in the repo root
with your publishable key (already gitignored):

```
SUPABASE_URL=<project>.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
```

Then:

```bash
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API.

### Frontend (React, dev)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api → :8000)
```

Production build: `npm run build` (outputs to `frontend/dist`).

### Serving the frontend from the API

The FastAPI app also serves the built SPA from `frontend/dist` (Vercel routes
every request to the same function). Build it once, then the API serves the app
at `http://localhost:8000/`:

```bash
cd frontend && npm install && npm run build
```

## Deploy to Vercel

The whole app runs as **one Vercel Python Function**. Vercel's Python framework
preset auto-detects FastAPI from the root `requirements.txt`, finds the
entrypoint `api/index.py` (exposing `app`), and routes *every* request to it —
so the same function serves the `/api/*` endpoints and the built frontend. No
`vercel.json` builds are required; the minimal `vercel.json` here is a no-op.

1. Build the frontend so the API can serve it:
   `cd frontend && npm install && npm run build` (outputs to `frontend/dist`).
2. From the project root: `vercel link`, then `vercel deploy` (or push to the
   Git integration).
3. In **Vercel → Project → Settings → Environment Variables** add:
   `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` for Production / Preview /
   Development.
4. Open the deployment — `/api/health` returns `{"status":"ok"}`, and `/`
   serves the SPA.

Notes:

- Vercel **never reads your local `.env`**. The app reads `SUPABASE_URL` /
  `SUPABASE_PUBLISHABLE_KEY` from the environment; `python-dotenv` only loads a
  local `.env` when present, so the same code runs locally and on Vercel.
- Data loads are **lazy** (`lru_cache`), so importing the app never triggers a
  Supabase request at startup — a bare `/api/health` works even if the
  credentials are wrong (it doesn't touch Supabase).
- `.python-version` pins Python 3.12 so the installed `pandas`/`numpy`/`pvlib`
  wheels match your local build.

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | health check |
| `GET /api/locations` | the two switchable locations + metadata |
| `GET /api/radiation/{location}?start&end&limit` | normalised radiation (W/m²) |
| `POST /api/simulate` | run the PV model; body `{location, start, end, panel}` |
| `POST /api/money` | solar self-consumption & savings over Christchurch's electricity year |
| `POST /api/model-money` | solar savings against modelled hourly consumption (region 2025 curve) |
| `POST /api/model-money/daily` | hourly detail for one day of the Model-money model |
| `POST /api/curves/daily` | daily generation (MWh) of every region on both islands |
| `GET /api/data-quality` | CAMS radiation data-quality report |

`panel` = `{tilt, azimuth, rated_power_kwp, albedo, transposition_model,
inverter_efficiency}`. `transposition_model` ∈ `perez` (default), `haydavies`,
`isotropic`.

## Data quality — computed in Postgres

The `GET /api/data-quality` endpoint used to download the whole multi-year
`cams_radiation` table and aggregate it in pandas, which was slow. The report
is now computed **server-side in Postgres** by a function
`public.get_data_quality(location)` that returns
the complete report as one JSON object; the app just proxies the result.

The SQL is in [`supabase/get_data_quality.sql`](supabase/get_data_quality.sql).
Run it once in the **Supabase SQL editor** to install the function:

```sql
-- open supabase/get_data_quality.sql and run it (it's idempotent)
```

Until the function is installed, the endpoint **falls back** to the old in-app
computation (so the app keeps working). The two solar-zenith-dependent checks
(`GHI ≈ DHI + DNI·cos(z)` residual and `BHI ≤ GHI`) are intentionally dropped —
they required pvlib solar geometry that doesn't map trivially to SQL — and the
API returns `ghi_conservation` / `bhi_le_ghi_violations` as `null` (the UI shows
`n/a`).

## Tests

```bash
python -m pytest tests -q
```

Validates: timezone-aware UTC indexing, multi-year loading, zero output at
night, positive solstice energy with rated-power clipping, summer > winter,
flat-panel POA ≈ GHI, idealized linear power, summary shape, the shared
self-consumption / savings helpers, the fixed daily-charge model, and that the
two locations map to distinct coordinate sets.
