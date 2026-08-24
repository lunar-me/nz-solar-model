"""FastAPI application exposing the location-switchable PV model.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import statistics

from .engine import PanelConfig, run_simulation, summarize, aggregate_energy
from .loader import load_radiation
from .locations import LOCATIONS, get_location
from .schemas import AggregateRequest, SimulateRequest, StabilityRequest

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


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Slice a UTC-indexed frame by start/end (str or tz-aware Timestamp)."""

    def coerce(v):
        if v is None:
            return None
        ts = pd.Timestamp(v)
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC")
        return ts.tz_localize("UTC")

    return df.loc[coerce(start):coerce(end)]


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


@app.post("/api/aggregate")
def aggregate(req: AggregateRequest) -> dict:
    """Aggregate PV output over a full NZ calendar year into monthly/weekly kWh."""
    if req.location not in LOCATIONS:
        raise HTTPException(404, detail=f"Unknown location '{req.location}'")

    from datetime import date, timedelta

    AK = "Pacific/Auckland"
    radiation = _cached_radiation(req.location)
    meta = radiation.attrs.get("metadata", {})
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )

    if req.period == "week":
        # Slice over the FULL ISO weeks (Mon..Sun) intersecting the year, so
        # every weekly bar is a complete week rather than a partial Jan-1 cut.
        first_mon = date.fromisocalendar(req.year, 1, 1)
        last_iso = date(req.year, 12, 31).isocalendar()
        last_mon = date.fromisocalendar(last_iso[0], last_iso[1], 1)
        last_sun = last_mon + timedelta(days=6)
        start = pd.Timestamp(first_mon, tz=AK)
        end = pd.Timestamp(last_sun + timedelta(days=1), tz=AK)
        rad = radiation.loc[start.tz_convert("UTC"):end.tz_convert("UTC")]
        rad = rad[rad.index < end.tz_convert("UTC")]
        result = run_simulation(
            rad, panel=panel,
            latitude=meta.get("latitude", 0.0),
            longitude=meta.get("longitude", 0.0),
            altitude=meta.get("altitude", 0.0),
        )
        # Annual summary stays the exact calendar year (Jan 1 -> Jan 1).
        cs = pd.Timestamp(f"{req.year}-01-01", tz=AK)
        ce = pd.Timestamp(f"{req.year + 1}-01-01", tz=AK)
        cal = result.loc[cs.tz_convert("UTC"):ce.tz_convert("UTC")]
        cal = cal[cal.index < ce.tz_convert("UTC")]
        if cal.empty:
            raise HTTPException(400, detail=f"No data for year {req.year}.")
        summary = summarize(cal, panel.rated_power_kwp)
        buckets = aggregate_energy(result, "week")
    else:
        start = pd.Timestamp(f"{req.year}-01-01", tz=AK)
        end = pd.Timestamp(f"{req.year + 1}-01-01", tz=AK)
        rad = radiation.loc[start.tz_convert("UTC"):end.tz_convert("UTC")]
        rad = rad[rad.index < end.tz_convert("UTC")]
        if rad.empty:
            raise HTTPException(400, detail=f"No data for year {req.year}.")
        result = run_simulation(
            rad, panel=panel,
            latitude=meta.get("latitude", 0.0),
            longitude=meta.get("longitude", 0.0),
            altitude=meta.get("altitude", 0.0),
        )
        summary = summarize(result, panel.rated_power_kwp)
        buckets = aggregate_energy(result, "month")

    return {
        "location": req.location,
        "year": req.year,
        "period": req.period,
        "metadata": meta,
        "panel": req.panel.model_dump(),
        "summary": summary,
        "buckets": buckets,
    }



@app.post("/api/stability")
def stability(req: StabilityRequest) -> dict:
    """Year-over-year PV totals (real + no-cloud) for every year in the dataset."""
    if req.location not in LOCATIONS:
        raise HTTPException(404, detail=f"Unknown location '{req.location}'")

    radiation = _cached_radiation(req.location)
    meta = radiation.attrs.get("metadata", {})
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )

    AK = "Pacific/Auckland"
    years = []
    for y in range(2020, 2026):  # all calendar years in the dataset
        start = pd.Timestamp(f"{y}-01-01", tz=AK)
        end = pd.Timestamp(f"{y + 1}-01-01", tz=AK)
        rad = radiation.loc[start.tz_convert("UTC"):end.tz_convert("UTC")]
        rad = rad[rad.index < end.tz_convert("UTC")]
        if rad.empty:
            continue
        res = run_simulation(
            rad, panel=panel,
            latitude=meta.get("latitude", 0.0),
            longitude=meta.get("longitude", 0.0),
            altitude=meta.get("altitude", 0.0),
        )
        real_kwh = float(res["energy_wh"].sum() / 1000.0)
        clear_kwh = float(res["energy_clear_wh"].sum() / 1000.0)
        loss = max(clear_kwh - real_kwh, 0.0)
        years.append({
            "year": y,
            "total_energy_kwh": round(real_kwh, 3),
            "total_energy_clear_kwh": round(clear_kwh, 3),
            "cloud_loss_kwh": round(loss, 3),
            "cloud_loss_pct": round(loss / clear_kwh * 100.0, 2) if clear_kwh else 0.0,
        })

    reals = [y["total_energy_kwh"] for y in years]
    mean = float(statistics.mean(reals))
    std = float(statistics.stdev(reals)) if len(reals) > 1 else 0.0
    cv = std / mean * 100.0 if mean else 0.0
    mn = min(reals)
    mx = max(reals)

    yoy = []
    variations = 0
    for i in range(len(years) - 1):
        a = years[i]["total_energy_kwh"]
        b = years[i + 1]["total_energy_kwh"]
        chg = (b - a) / a * 100.0 if a else 0.0
        yoy.append({"from": years[i]["year"], "to": years[i + 1]["year"],
                    "change_pct": round(chg, 2)})
        if abs(chg) >= 5.0:
            variations += 1

    metrics = {
        "mean_kwh": round(mean, 1),
        "std_kwh": round(std, 1),
        "cv_pct": round(cv, 1),
        "min_kwh": round(mn, 1),
        "min_year": years[reals.index(mn)]["year"],
        "max_kwh": round(mx, 1),
        "max_year": years[reals.index(mx)]["year"],
        "range_kwh": round(mx - mn, 1),
        "transitions": len(yoy),
        "variations": variations,
        "yoy": yoy,
    }

    return {
        "location": req.location,
        "metadata": meta,
        "panel": req.panel.model_dump(),
        "years": years,
        "metrics": metrics,
    }

