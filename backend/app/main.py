"""FastAPI application exposing the location-switchable PV model.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import statistics

from .engine import (PanelConfig, run_simulation, summarize, aggregate_energy,
                     data_quality_report)
from .loader import load_radiation, load_electricity
from .locations import LOCATIONS, get_location, DATA_DIR
from .schemas import (AggregateRequest, SimulateRequest, StabilityRequest,
                      MoneyRequest)

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


@lru_cache(maxsize=len(LOCATIONS))
def _cached_radiation_hourly(location_key: str) -> pd.DataFrame:
    """15-min radiation resampled to hourly, for the coarse aggregation views.

    Monthly/weekly/yearly totals agree with the 15-min data to <1%, but run
    ~3.5x faster because plane-of-array is computed over 4x fewer intervals.
    """
    rad = _cached_radiation(location_key)
    hourly = rad.resample("1h").mean()
    hourly.attrs["interval_h"] = 1.0
    hourly.attrs["metadata"] = rad.attrs.get("metadata", {})
    return hourly


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
        "cloud_index",
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
    radiation = _cached_radiation_hourly(req.location)
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

    radiation = _cached_radiation_hourly(req.location)
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



@lru_cache(maxsize=1)
def _christchurch_hourly_radiation() -> pd.DataFrame:
    """Combined 1-hour Christchurch radiation (2020-2025 + 2026 files)."""
    f1 = DATA_DIR / "CAMS Radiation - 1h - Christchurch - 20200101 - 20251231.csv"
    f2 = DATA_DIR / "CAMS Radiation - 1h - Christchurch - 20260101 - 20260823.csv"
    rads = [load_radiation(f1), load_radiation(f2)]
    rad = pd.concat(rads).sort_index()
    rad = rad[~rad.index.duplicated(keep="first")]
    rad.attrs["interval_h"] = 1.0
    return rad


@app.post("/api/money")
def money(req: MoneyRequest) -> dict:
    """Solar self-consumption & savings against the fixed Christchurch year."""
    if req.location != "christchurch":
        raise HTTPException(400, detail="'My money' tab is locked to Christchurch.")

    el = load_electricity()  # hourly consumption (kWh) + cost ($), UTC index
    meta = _christchurch_hourly_radiation().attrs.get("metadata", {})
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )

    start = el.index.min()
    end = el.index.max()
    rad = _christchurch_hourly_radiation().loc[start:end]
    result = run_simulation(
        rad, panel=panel,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    solar = (result["energy_wh"] / 1000.0).rename("solar_kwh")

    df = pd.concat([el, solar], axis=1).sort_index()
    df["solar_kwh"] = df["solar_kwh"].fillna(0.0)
    df["consumption_kwh"] = df["consumption_kwh"].fillna(0.0)
    df["cost_$"] = df["cost_$"].fillna(0.0)

    cons = df["consumption_kwh"].to_numpy()
    sol = df["solar_kwh"].to_numpy()
    cost = df["cost_$"].to_numpy()

    self_kwh = np.minimum(cons, sol)
    excess_kwh = np.maximum(sol - cons, 0.0)
    grid_kwh = np.maximum(cons - sol, 0.0)
    frac = np.divide(self_kwh, cons, out=np.zeros_like(cons), where=cons > 0)
    savings = frac * cost
    net = cost - savings

    df["self_consumed_kwh"] = self_kwh
    df["excess_kwh"] = excess_kwh
    df["grid_kwh"] = grid_kwh
    df["savings_$"] = savings
    df["net_$"] = net

    total_cost = float(cost.sum())
    total_cons = float(cons.sum())
    # Value wasted solar at each hour's OWN effective rate (that hour's bill
    # divided by its usage) — no assumed/average electricity rate.
    hourly_rate = np.divide(cost, cons, out=np.zeros_like(cost), where=cons > 0)
    df["waste_$"] = df["excess_kwh"] * hourly_rate

    solar_kwh = float(sol.sum())
    self_sum = float(self_kwh.sum())
    excess_sum = float(excess_kwh.sum())
    grid_sum = float(grid_kwh.sum())
    savings_sum = float(savings.sum())
    waste_sum = float(df["waste_$"].sum())

    totals = {
        "consumption_kwh": round(total_cons, 1),
        "solar_kwh": round(solar_kwh, 1),
        "self_consumed_kwh": round(self_sum, 1),
        "excess_kwh": round(excess_sum, 1),
        "grid_import_kwh": round(grid_sum, 1),
        "self_consumption_pct": round(self_sum / solar_kwh * 100, 1) if solar_kwh else 0.0,
        "solar_coverage_pct": round(self_sum / total_cons * 100, 1) if total_cons else 0.0,
        "cost_without_solar_$": round(total_cost, 2),
        "cost_with_solar_$": round(total_cost - savings_sum, 2),
        "savings_$": round(savings_sum, 2),
        "savings_pct": round(savings_sum / total_cost * 100, 1) if total_cost else 0.0,
        "wasted_value_$": round(waste_sum, 2),
    }

    # Monthly aggregation (NZ-local months).
    local = df.index.tz_convert("Pacific/Auckland").to_period("M")
    g = df.groupby(local).agg({
        "consumption_kwh": "sum", "solar_kwh": "sum",
        "self_consumed_kwh": "sum", "excess_kwh": "sum", "grid_kwh": "sum",
        "cost_$": "sum", "savings_$": "sum", "waste_$": "sum",
    })
    monthly = []
    for p in g.index:
        r = g.loc[p]
        monthly.append({
            "month": str(p),
            "label": p.strftime("%b %Y"),
            "consumption_kwh": round(float(r["consumption_kwh"]), 1),
            "solar_kwh": round(float(r["solar_kwh"]), 1),
            "self_consumed_kwh": round(float(r["self_consumed_kwh"]), 1),
            "excess_kwh": round(float(r["excess_kwh"]), 1),
            "grid_kwh": round(float(r["grid_kwh"]), 1),
            "cost_$": round(float(r["cost_$"]), 2),
            "savings_$": round(float(r["savings_$"]), 2),
            "waste_$": round(float(r["waste_$"]), 2),
        })

    return {
        "location": "christchurch",
        "metadata": meta,
        "panel": req.panel.model_dump(),
        "totals": totals,
        "monthly": monthly,
    }



@app.get("/api/data-quality")
def data_quality(location: str = Query(...)) -> dict:
    """Data-quality report for a location's CAMS dataset (idea.md #14)."""
    if location not in LOCATIONS:
        raise HTTPException(404, detail=f"Unknown location '{location}'")
    rad = _cached_radiation(location)
    meta = rad.attrs.get("metadata", {})
    report = data_quality_report(
        rad,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    report["location"] = location
    report["metadata"] = meta
    return report

