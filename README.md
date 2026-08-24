# NZ Solar PV Model

A physics-first, idealized PV output model built on **pvlib-python**, driven by
**CAMS radiation** data for two New Zealand cities (**Auckland** and
**Christchurch**), exposed through a **FastAPI** backend and a small **React**
frontend for choosing a location, configuring a panel, and viewing the results.

Implements the plan in [`idea.md`](idea.md): CAMS irradiance → solar position →
angle of incidence → Plane-of-Array (POA) irradiance → idealized DC/AC power →
15-minute energy, using established pvlib algorithms rather than hand-written
astronomy.

## Model boundary

The model computes, for every 15-minute interval:

```
timestamp, sun_elevation, sun_azimuth, AOI,
GHI, DHI, DNI, POA_direct, POA_diffuse, POA_ground, POA_global,
DC power (W), AC power (W), energy (Wh)
```

Inputs: location (lat/lon/altitude read from the CAMS file header), panel
tilt/azimuth, rated power (kWp), albedo, transposition model, inverter
efficiency.

This v1 is **idealized** by design (idea.md §9, §29): no module-temperature,
no soiling/mismatch/shading/degradation losses. Each of those is an explicit
configuration knob to be added later without touching the core geometry.

## Coordinate conventions (idea.md §3)

- **Azimuth**: 0° = N, 90° = E, 180° = S, 270° = W (meteorological convention).
- **Tilt**: 0° = horizontal, 90° = vertical.
- Default for NZ is a north-ish facing tilt of 25°, but a south-facing roof is
  `azimuth=180`.

## Time handling (idea.md §4)

CAMS exports timestamps in **UTC**. The loader keeps a timezone-aware UTC
`DatetimeIndex`; pvlib solar position is computed on those UTC stamps. The
frontend displays them converted to **Pacific/Auckland** local time (handles
NZST/NZDT automatically). No naive datetimes are used internally.

## Units note (important)

CAMS exports **irradiations** (Wh/m²) integrated over each 15-minute interval,
not instantaneous irradiances. The loader converts each interval to an average
**irradiance in W/m²** (divide by the interval length in hours) so pvlib gets
what it expects, then energy is re-computed as `power_W × interval_hours`.
The loader also normalizes CAMS names to the internal schema
(`ghi / dhi / dni / *_clear / reliability`) per idea.md §13.

## Repository layout

```
data/            CAMS radiation CSVs (Auckland, Christchurch) — unchanged
backend/
  app/
    locations.py  location registry (the "switch" keyed by name)
    loader.py     CAMS CSV → normalized, W/m², UTC-indexed DataFrame
    engine.py     pvlib solar position / POA / idealized PV + summary
    schemas.py    Pydantic request models
    main.py       FastAPI app (caches each loaded dataset)
  requirements.txt
frontend/        React (Vite) app: pick location, configure panel, view charts
tests/           pytest suite (physics + data validation)
```

## Run it

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API.

### Frontend (React, dev)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api → :8000)
```

Production build: `npm run build` (outputs to `frontend/dist`).

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | health check |
| `GET /api/locations` | the two switchable locations + metadata |
| `GET /api/radiation/{location}?start&end&limit` | normalised radiation (W/m²) |
| `POST /api/simulate` | run the PV model; body `{location, start, end, panel}` |

`panel` = `{tilt, azimuth, rated_power_kwp, albedo, transposition_model,
inverter_efficiency}`. `transposition_model` ∈ `perez` (default), `haydavies`,
`isotropic`.

## Tests

```bash
python -m pytest tests -q
```

Validates: timezone-aware UTC indexing, multi-year loading, zero output at
night, positive solstice energy with rated-power clipping, summer > winter,
flat-panel POA ≈ GHI, idealized linear power, summary shape, and that the two
locations map to distinct coordinate sets.
