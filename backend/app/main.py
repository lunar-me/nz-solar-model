"""FastAPI application exposing the location-switchable PV model.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .engine import PanelConfig, run_simulation, summarize
from .loader import load_radiation
from .locations import LOCATIONS, get_location
from .schemas import SimulateRequest

app = FastAPI(
    title="NZ Solar PV Model",
    description="Physics-first idealized PV output model over CAMS radiation "
                "data, switchable between Auckland and Christchurch.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=len(LOCATIONS))
def _cached_radiation(location_key: str) -> pd.DataFrame:
    loc = get_location(location_key)
    return load_radiation(loc.file)


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    return df.loc[start_ts:end_ts]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _cached_meta(key: str) -> dict:
    try:
        return _cached_radiation(key).attrs.get("metadata", {})
    except Exception:
        return {}


@app.get("/api/locations")
def list_locations() -> list[dict]:
    return [
        {
            "key": loc.key,
            "name": loc.name,
            "region": loc.region,
            "metadata": _cached_meta(loc.key),
        }
        for loc in LOCATIONS.values()
    ]


@app.get("/api/radiation/{location}")
def get_radiation(
    location: str,
    start: str | None = Query(None),
    end: str | None = Query(None),
    limit: int = Query(5000, ge=1, le=50000),
) -> dict:
    """Return normalised radiation (W/m2) for a location, optionally a date slice."""
    if location not in LOCATIONS:
        raise HTTPException(404, detail=f"Unknown location '{location}'")
    df = _slice(_cached_radiation(location), start, end)
    if len(df) > limit:
        step = max(len(df) // limit, 1)
        df = df.iloc[::step]
    rows = df.reset_index().rename(columns={"index": "timestamp"})
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows["timestamp_local"] = df.index.tz_convert("Pacific/Auckland").strftime(
        "%Y-%m-%d %H:%M")
    cols = ["timestamp", "timestamp_local", "ghi", "dhi", "dni",
            "ghi_clear", "dhi_clear", "dni_clear", "reliability"]
    return {
        "location": location,
        "metadata": df.attrs.get("metadata", {}),
        "columns": cols,
        "rows": rows[cols].to_dict("records"),
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest) -> dict:
    """Run the idealized PV model for a location + panel over a date range."""
    if req.location not in LOCATIONS:
        raise HTTPException(404, detail=f"Unknown location '{req.location}'")

    radiation = _slice(_cached_radiation(req.location), req.start, req.end)
    if radiation.empty:
        raise HTTPException(400, detail="No data in the requested date range.")

    meta = radiation.attrs.get("metadata", {})
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )

    result = run_simulation(
        radiation,
        panel=panel,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    summary = summarize(result, panel.rated_power_kwp)

    base_cols = [
        "timestamp", "timestamp_local", "sun_elevation_deg", "sun_azimuth_deg", "aoi_deg",
        "poa_direct", "poa_diffuse", "poa_ground", "poa_global",
        "dc_power_w", "ac_power", "ac_power_clear", "energy_wh", "energy_clear_wh",
    ]
    if req.include_radiation:
        cols = ["timestamp", "timestamp_local", "ghi", "dhi", "dni"] + base_cols[2:]
    else:
        cols = base_cols

    return {
        "location": req.location,
        "metadata": meta,
        "panel": req.panel.model_dump(),
        "summary": summary,
        "timeseries": result[cols].to_dict("records"),
    }
