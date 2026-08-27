"""FastAPI application exposing the location-switchable PV model.

Local dev (run from the project root):
    uvicorn api.index:app --reload --port 8000

Vercel deploy: Vercel's Python framework preset detects the ``app`` object in
``api/index.py`` (a supported entrypoint) and routes *every* request to it, so
this app also serves the built Vite frontend from ``frontend/dist``. No
``vercel.json`` builds are required; only ``SUPABASE_URL`` /
``SUPABASE_PUBLISHABLE_KEY`` need to be set as Vercel environment variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
import pandas as pd
import numpy as np
import statistics
import warnings

from .engine import (PanelConfig, run_simulation, summarize, aggregate_energy,
                     data_quality_report)
from .loader import (load_radiation_from_supabase, load_electricity_from_supabase,
                     load_region_generation_from_supabase, _utc_iso)
from .supabase_client import (RPCFunctionNotFoundError, fetch_data_quality,
                              fetch_region_generation_by_island)
from .locations import LOCATIONS
from .schemas import (AggregateRequest, SimulateRequest, StabilityRequest,
                      MoneyRequest, ModelMoneyRequest, ModelMoneyDailyRequest,
                      CurvesDailyRequest)

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
    return load_radiation_from_supabase(location_key)


@lru_cache(maxsize=64)
def _cached_range_radiation(location: str, start_iso: str | None,
                            end_iso: str | None) -> pd.DataFrame:
    """Radiation for a (location, start, end) range, cached.

    Radiation depends only on *location + date range* — not on panel settings.
    Caching it means changing tilt/azimuth/inverter efficiency re-runs the
    simulation but does NOT refetch Supabase.
    """
    return load_radiation_from_supabase(location, start=start_iso, end=end_iso)


def _range_radiation(location: str, start, end) -> pd.DataFrame:
    """Range-limited radiation fetch, normalised to UTC-ISO keys for caching."""
    return _cached_range_radiation(location, _utc_iso(start), _utc_iso(end))


@lru_cache(maxsize=1)
def _cached_electricity() -> pd.DataFrame:
    """Hourly Christchurch consumption, fetched once and reused (panel-independent)."""
    return load_electricity_from_supabase()


# Maps a location key to (display label, component regions) whose 2025
# electricity-generation curves model the location's hourly usage. Auckland uses
# the Waikato region; Christchurch uses Canterbury.
LOCATION_REGIONS = {
    "auckland": ("Waikato", ["Waikato"]),
    "christchurch": ("Canterbury", ["Canterbury"]),
}


@lru_cache(maxsize=sum(len(v[1]) for v in LOCATION_REGIONS.values()))
def _cached_region_generation(region: str) -> pd.DataFrame:
    """Hourly 2025 generation share for one region, fetched once and reused."""
    return load_region_generation_from_supabase(region)


def _location_usage_percent(location: str) -> pd.DataFrame:
    """Combined hourly usage_percent for a location's component region(s).

    Each component region's ``usage_percent`` sums to ~100 over the year (its
    share of that region's annual electricity), so averaging the components
    keeps the combined annual total ~100 — i.e. modelled annual consumption still
    equals the user's annual kWh.
    """
    entry = LOCATION_REGIONS.get(location)
    if entry is None:
        raise KeyError(location)
    _, regions = entry
    combined = _cached_region_generation(regions[0]).copy()
    for region in regions[1:]:
        combined["usage_percent"] = (
            combined["usage_percent"] + _cached_region_generation(region)["usage_percent"]
        )
    combined["usage_percent"] = combined["usage_percent"] / len(regions)
    return combined


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


def _hourly_radiation(location_key: str, start=None, end=None) -> pd.DataFrame:
    """Fetch a UTC date range from Supabase and resample to hourly.

    Only the requested slice is pulled (PostgREST ``start_ts_utc`` filters),
    so /api/aggregate does not download the whole multi-year dataset.
    """
    rad = _range_radiation(location_key, start, end)
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
    """Location metadata from the registry (no Supabase / dataset fetch).

    The coordinates are pinned in ``locations.py``; reading them must not pull
    the entire multi-year radiation dataset just to answer ``/api/locations``.
    """
    loc = LOCATIONS.get(key)
    if loc is None:
        return {}
    return {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "altitude": loc.altitude,
    }


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
    df = _range_radiation(location, start, end)
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

    radiation = _range_radiation(req.location, req.start, req.end)
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
    meta = _cached_meta(req.location)
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )

    # Determine the exact UTC range we need, then fetch ONLY that slice rather
    # than the entire multi-year dataset.
    if req.period == "week":
        # Full ISO weeks (Mon..Sun) intersecting the year, so every weekly bar
        # is a complete week rather than a partial Jan-1 cut.
        first_mon = date.fromisocalendar(req.year, 1, 1)
        last_iso = date(req.year, 12, 31).isocalendar()
        last_mon = date.fromisocalendar(last_iso[0], last_iso[1], 1)
        last_sun = last_mon + timedelta(days=6)
        start = pd.Timestamp(first_mon, tz=AK)
        end = pd.Timestamp(last_sun + timedelta(days=1), tz=AK)
    else:
        start = pd.Timestamp(f"{req.year}-01-01", tz=AK)
        end = pd.Timestamp(f"{req.year + 1}-01-01", tz=AK)

    radiation = _hourly_radiation(
        req.location, start=start.tz_convert("UTC"), end=end.tz_convert("UTC"))

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

    if req.period == "week":
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



def _self_consumption_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-hour self-consumption / savings columns to an hourly frame.

    Expects hourly columns ``consumption_kwh``, ``solar_kwh`` and ``cost_$``.
    Adds ``self_consumed_kwh``, ``excess_kwh``, ``grid_kwh``, ``savings_$``,
    ``net_$`` and ``waste_$`` (wasted solar valued at each hour's own effective
    rate = cost / consumption, so a flat per-kWh price works too).
    """
    df = df.copy()
    cons = df["consumption_kwh"].to_numpy()
    sol = df["solar_kwh"].to_numpy()
    cost = df["cost_$"].to_numpy()

    self_kwh = np.minimum(cons, sol)
    excess_kwh = np.maximum(sol - cons, 0.0)
    grid_kwh = np.maximum(cons - sol, 0.0)
    frac = np.divide(self_kwh, cons, out=np.zeros_like(cons), where=cons > 0)
    savings = frac * cost
    net = cost - savings
    hourly_rate = np.divide(cost, cons, out=np.zeros_like(cost), where=cons > 0)

    df["self_consumed_kwh"] = self_kwh
    df["excess_kwh"] = excess_kwh
    df["grid_kwh"] = grid_kwh
    df["savings_$"] = savings
    df["net_$"] = net
    df["waste_$"] = df["excess_kwh"] * hourly_rate
    return df


def _money_totals(df: pd.DataFrame) -> dict:
    """Aggregate an hourly self-consumption frame into headline totals."""
    solar_kwh = float(df["solar_kwh"].sum())
    total_cons = float(df["consumption_kwh"].sum())
    total_cost = float(df["cost_$"].sum())
    self_sum = float(df["self_consumed_kwh"].sum())
    savings_sum = float(df["savings_$"].sum())
    return {
        "consumption_kwh": round(total_cons, 1),
        "solar_kwh": round(solar_kwh, 1),
        "self_consumed_kwh": round(self_sum, 1),
        "excess_kwh": round(float(df["excess_kwh"].sum()), 1),
        "grid_import_kwh": round(float(df["grid_kwh"].sum()), 1),
        "self_consumption_pct": round(self_sum / solar_kwh * 100, 1) if solar_kwh else 0.0,
        "solar_coverage_pct": round(self_sum / total_cons * 100, 1) if total_cons else 0.0,
        "cost_without_solar_$": round(total_cost, 2),
        "cost_with_solar_$": round(total_cost - savings_sum, 2),
        "savings_$": round(savings_sum, 2),
        "savings_pct": round(savings_sum / total_cost * 100, 1) if total_cost else 0.0,
        "wasted_value_$": round(float(df["waste_$"].sum()), 2),
    }


def _money_monthly(df: pd.DataFrame) -> list[dict]:
    """Aggregate an hourly self-consumption frame into NZ-local calendar months."""
    with warnings.catch_warnings():
        # tz-aware DatetimeIndex -> Period drops tz by design; we already
        # converted to NZ local time, so silence pandas' informational warning.
        warnings.filterwarnings(
            "ignore",
            message="Converting to PeriodArray/Index representation will drop timezone information",
        )
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
    return monthly


def _add_daily_charges(df: pd.DataFrame, daily_charge: float) -> pd.DataFrame:
    """Spread a flat daily fixed charge across each NZ-local day's hours.

    The fixed charge (a set daily connection fee) is added to every hour's
    ``cost_$`` by splitting that day's total equally among the day's hours, so
    each NZ calendar day contributes exactly ``daily_charge`` to total cost.

    This is intended to be called *after* ``_self_consumption_columns``, so the
    fixed charge is added only to the dollar totals — it never feeds the
    per-hour solar self-consumption/savings math, because it is paid regardless
    of how much electricity is used (and therefore can never be "saved" by
    solar).
    """
    if not daily_charge:
        return df
    df = df.copy()
    nz_date = df.index.tz_convert("Pacific/Auckland").normalize()
    counts = nz_date.value_counts()          # hours per NZ day (24; 23/25 on DST days)
    per_hour = daily_charge / counts.reindex(nz_date).to_numpy()
    df["cost_$"] = df["cost_$"] + per_hour
    return df


def _parse_nz_day(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse a NZ-local ``YYYY-MM-DD`` into ``(start_utc, end_utc)`` for that day.

    Raises :class:`HTTPException` (422) for an invalid, empty or non-2025 date —
    including the ``NaT`` case that ``pd.Timestamp("")`` silently produces.
    """
    try:
        day = pd.Timestamp(date_str, tz="Pacific/Auckland")
    except Exception:
        raise HTTPException(422, detail=f"Invalid date '{date_str}' (expected YYYY-MM-DD).")
    if pd.isna(day):
        raise HTTPException(422, detail=f"Invalid date '{date_str}' (expected YYYY-MM-DD).")
    start_nz = day.normalize()
    end_nz = start_nz + pd.Timedelta(days=1)
    if start_nz.year != 2025:
        raise HTTPException(422, detail="Day must be within the 2025 NZ year.")
    return start_nz.tz_convert("UTC"), end_nz.tz_convert("UTC")


@app.post("/api/money")
def money(req: MoneyRequest) -> dict:
    """Solar self-consumption & savings over the Christchurch electricity year.

    Electricity consumption always uses the Christchurch dataset (Auckland
    consumption isn't available yet), but the *solar* side uses the selected
    location's radiation over that same hourly year.
    """
    el = _cached_electricity()  # hourly consumption (kWh) + cost ($), UTC index (cached)
    if el.empty:
        raise HTTPException(
            400, detail="No Christchurch electricity consumption data available."
        )

    # Fetch the selected location's radiation only across the electricity year
    # (UTC range), rather than the entire 2020-2026 dataset, to keep memory +
    # load time small.  Radiation is resampled to hourly to align with the
    # hourly electricity consumption.
    start = el.index.min()
    end = el.index.max()
    rad = _hourly_radiation(req.location, start=start, end=end)
    if rad.empty:
        raise HTTPException(
            400,
            detail="No radiation data overlaps the Christchurch electricity period.",
        )
    meta = rad.attrs.get("metadata", {})
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )

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
    with warnings.catch_warnings():
        # tz-aware DatetimeIndex -> Period drops tz by design; we already
        # converted to NZ local time, so silence pandas' informational warning.
        warnings.filterwarnings(
            "ignore",
            message="Converting to PeriodArray/Index representation will drop timezone information",
        )
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
        "location": req.location,
        "metadata": meta,
        "panel": req.panel.model_dump(),
        "totals": totals,
        "monthly": monthly,
    }



@app.post("/api/model-money")
def model_money(req: ModelMoneyRequest) -> dict:
    """Solar savings against *modelled* hourly consumption (region 2025 curve).

    Instead of a real household's consumption (as in ``/api/money``), hourly
    usage is modelled by spreading the user's annual kWh over the selected
    region's real 2025 electricity-generation curve (``usage_percent`` per hour),
    priced at a flat per-kWh rate incl. GST, then fed through the same
    self-consumption / solar-savings calculation as ``/api/money``.
    """
    entry = LOCATION_REGIONS.get(req.location)
    if entry is None:
        raise HTTPException(404, detail=f"Unknown location '{req.location}'")
    region = entry[0]  # display label, e.g. "Auckland + Northland + Waikato"

    gen = _location_usage_percent(req.location)  # combined hourly usage_percent (NZ-processed)
    if gen.empty:
        raise HTTPException(400, detail=f"No generation data for {req.location} in 2025.")

    # Fetch the selected city's radiation across the region's 2025 period and
    # resample to hourly so it aligns with the modelled hourly consumption.
    start = gen.index.min()
    end = gen.index.max()
    rad = _hourly_radiation(req.location, start=start, end=end)
    if rad.empty:
        raise HTTPException(
            400, detail="No radiation overlaps the region 2025 generation period."
        )
    meta = rad.attrs.get("metadata", {})
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )

    result = run_simulation(
        rad, panel=panel,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    solar = (result["energy_wh"] / 1000.0).rename("solar_kwh")

    # --- model hourly consumption & cost from annual usage + region curve ---
    pct = gen["usage_percent"]
    consumption = req.annual_kwh * pct / 100.0
    cost = req.kwh_price_gst * req.annual_kwh * pct / 100.0
    model = pd.DataFrame(
        {"consumption_kwh": consumption, "cost_$": cost}, index=gen.index
    )

    df = pd.concat([model, solar], axis=1).sort_index()
    df["solar_kwh"] = df["solar_kwh"].fillna(0.0)
    df = df.dropna(subset=["consumption_kwh", "cost_$"])

    df = _self_consumption_columns(df)
    # Fixed daily connection fee — added only to dollar totals after the
    # per-hour solar savings are derived, so it never looks "savable".
    df = _add_daily_charges(df, req.daily_charge)
    totals = _money_totals(df)
    monthly = _money_monthly(df)

    return {
        "location": req.location,
        "region": region,
        "annual_kwh": req.annual_kwh,
        "kwh_price_gst": req.kwh_price_gst,
        "daily_charge": req.daily_charge,
        "period": {"start": str(gen.index.min()), "end": str(gen.index.max())},
        "metadata": meta,
        "panel": req.panel.model_dump(),
        "totals": totals,
        "monthly": monthly,
    }


@app.post("/api/model-money/daily")
def model_money_daily(req: ModelMoneyDailyRequest) -> dict:
    """Hourly detail for a single NZ day of the Model-money model.

    Returns that day's modelled hourly solar generation (+ no-cloud reference),
    electricity consumption, solar-saved (self-consumed) and solar-wasted
    (excess) curves so the frontend can chart a single day without re-running
    the whole year. Consumption is modelled from the annual usage spread over
    the region's 2025 generation curve (as in /api/model-money); solar is
    simulated over just that day's radiation.
    """
    entry = LOCATION_REGIONS.get(req.location)
    if entry is None:
        raise HTTPException(404, detail=f"Unknown location '{req.location}'")
    region = entry[0]  # display label
    gen = _location_usage_percent(req.location)
    if gen.empty:
        raise HTTPException(400, detail=f"No generation data for {req.location} in 2025.")

    start_utc, end_utc = _parse_nz_day(req.date)

    # Solar for that NZ day (simulated only over that day's radiation).
    rad = _hourly_radiation(req.location, start=start_utc, end=end_utc)
    if rad.empty:
        raise HTTPException(400, detail="No radiation data for the requested day.")
    meta = rad.attrs.get("metadata", {})
    panel = PanelConfig(
        tilt=req.panel.tilt,
        azimuth=req.panel.azimuth,
        rated_power_kwp=req.panel.rated_power_kwp,
        albedo=req.panel.albedo,
        transposition_model=req.panel.transposition_model,
        inverter_efficiency=req.panel.inverter_efficiency,
    )
    result = run_simulation(
        rad, panel=panel,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    solar = (result["energy_wh"] / 1000.0).rename("solar_kwh")
    solar_clear = (result["energy_clear_wh"] / 1000.0).rename("solar_clear_kwh")

    # Modelled consumption for that NZ day (last hour boundary excluded).
    day_gen = gen.loc[start_utc:end_utc - pd.Timedelta(hours=1)]
    consumption = req.annual_kwh * day_gen["usage_percent"] / 100.0
    cost = req.kwh_price_gst * req.annual_kwh * day_gen["usage_percent"] / 100.0
    model = pd.DataFrame(
        {"consumption_kwh": consumption, "cost_$": cost}, index=day_gen.index
    )

    df = pd.concat([model, solar, solar_clear], axis=1).sort_index()
    df["solar_kwh"] = df["solar_kwh"].fillna(0.0)
    df["solar_clear_kwh"] = df["solar_clear_kwh"].fillna(0.0)
    df = df.dropna(subset=["consumption_kwh", "cost_$"])
    df = _self_consumption_columns(df)  # adds self_consumed_kwh ("saved") / excess_kwh ("wasted")
    df = df.sort_index()

    # Emit hours in NZ-local wall time.
    local = df.index.tz_convert("Pacific/Auckland")
    cols = ["consumption_kwh", "solar_kwh", "solar_clear_kwh",
            "self_consumed_kwh", "excess_kwh"]
    vals = df[cols].to_numpy()
    daily = []
    for i, ts in enumerate(local):
        v = vals[i]
        daily.append({
            "time": ts.strftime("%Y-%m-%d %H:%M"),
            "hour": ts.strftime("%H:%M"),
            "consumption_kwh": round(float(v[0]), 3),
            "solar_kwh": round(float(v[1]), 3),
            "solar_clear_kwh": round(float(v[2]), 3),
            "self_consumed_kwh": round(float(v[3]), 3),
            "excess_kwh": round(float(v[4]), 3),
        })

    return {
        "location": req.location,
        "region": region,
        "date": req.date,
        "daily": daily,
        "metadata": meta,
    }


@app.post("/api/curves/daily")
def curves_daily(req: CurvesDailyRequest) -> dict:
    """Daily hourly generation (MWh) for every region of both islands.

    Reads ``region_electricity_generation_2025_1h`` for one NZ day and returns,
    per island, the hourly usage in MWh for each region, so the frontend can draw
    a multi-line daily chart per island.
    """
    start_utc, end_utc = _parse_nz_day(req.date)

    rows = fetch_region_generation_by_island(start=start_utc, end=end_utc)
    if not rows:
        raise HTTPException(400, detail="No generation data for the requested day.")

    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    # The fetch's end filter is inclusive, so it also returns the first hour of
    # the NEXT day. That hour maps to NZ "00:00" too, which would double-count
    # the 00:00 bucket in the pivot — drop it so each NZ hour appears once.
    df = df[df["datetime_utc"] < end_utc]
    df["usage_mwh"] = pd.to_numeric(df["usage_kwh"], errors="coerce") / 1000.0
    df["hour"] = df["datetime_utc"].dt.tz_convert("Pacific/Auckland").dt.strftime("%H:%M")

    hours = [f"{h:02d}:00" for h in range(24)]
    islands = {}
    for island in ("North Island", "South Island"):
        sub = df[df["island"] == island]
        regions = sorted(sub["region"].dropna().unique().tolist())
        piv = sub.pivot_table(index="hour", columns="region",
                              values="usage_mwh", aggfunc="sum").reindex(hours)
        rows_out = []
        for h in hours:
            row = {"hour": h}
            for r in regions:
                v = piv.loc[h, r] if r in piv.columns else None
                row[r] = round(float(v), 3) if pd.notna(v) else None
            rows_out.append(row)
        islands[island] = {"regions": regions, "rows": rows_out}

    return {"date": req.date, "islands": islands}


@app.get("/api/data-quality")
def data_quality(location: str = Query(...)) -> dict:
    """Data-quality report for a location's CAMS dataset (idea.md #14).

    The heavy aggregation runs server-side in Postgres via the
    ``get_data_quality`` RPC (``supabase/get_data_quality.sql``), so the app
    doesn't download the whole multi-year table. If that function hasn't been
    created in Supabase yet, fall back to the in-app computation (normalised to
    the same JSON shape, with the two solar-zenith checks dropped).
    """
    if location not in LOCATIONS:
        raise HTTPException(404, detail=f"Unknown location '{location}'")
    loc = LOCATIONS[location]
    meta = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "altitude": loc.altitude,
    }
    try:
        report = fetch_data_quality(loc.supabase_name)
    except (RPCFunctionNotFoundError, httpx.HTTPStatusError, httpx.RequestError):
        # Fall back to the in-app computation on ANY RPC failure — the function
        # isn't installed yet (404), a transient Supabase 5xx slipped through the
        # single retry, or the network call failed. The fallback is slower (it
        # pulls the whole multi-year table) but always returns a valid report.
        rad = _cached_radiation(location)
        report = data_quality_report(
            rad, loc.latitude, loc.longitude, loc.altitude
        )
        # Normalise the fallback to the SQL report's shape (solar checks dropped).
        report["radiation"]["ghi_conservation"] = {
            "mean_residual": None,
            "max_abs_residual": None,
        }
        report["radiation"]["bhi_le_ghi_violations"] = None
        report["checks"] = [
            c for c in report["checks"] if "cosz" not in c["msg"]
        ]
    report["location"] = location
    report["metadata"] = meta
    return report


# --- Static frontend (built SPA) -------------------------------------------
# Vercel's Python framework preset routes *every* request to this FastAPI app,
# so the app itself serves the built Vite frontend from frontend/dist. Mounting
# StaticFiles at "/" (with html=True) handles the SPA and its hashed assets;
# the /api/*, /docs and /openapi.json routes above were registered first, so
# they take precedence over the mount.
#
# The frontend must be built first (`cd frontend && npm run build`). If dist is
# absent the mount is skipped so the API-only app still boots locally.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=_FRONTEND_DIST, html=True),
        name="frontend",
    )

