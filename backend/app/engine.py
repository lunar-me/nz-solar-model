"""Physics-first PV engine built on pvlib.

Pipeline (from idea.md):
    CAMS irradiance -> solar position -> AOI -> Plane-of-Array (POA)
    -> idealized DC/AC power -> 15-minute energy.

Model is deliberately idealized (v1): no module-temperature, no soiling/
mismatch/shading losses, no inverter losses beyond a flat efficiency.  All of
those are explicit configuration knobs to be added later without touching the
core geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pvlib import irradiance, solarposition

TRANSPOSITION_MODELS = ("perez", "haydavies", "isotropic")


@dataclass
class PanelConfig:
    """Orientation + size of one (fixed) PV array."""

    tilt: float = 25.0            # degrees from horizontal (0 = flat)
    azimuth: float = 0.0           # degrees from North, meteorological convention
    rated_power_kwp: float = 5.0  # STC rated power (kWp)
    albedo: float = 0.2           # ground albedo for the reflected component
    transposition_model: str = "perez"
    inverter_efficiency: float = 0.95  # modern string inverters: ~95-98%

    @property
    def rated_power_w(self) -> float:
        return self.rated_power_kwp * 1000.0

    def __post_init__(self) -> None:
        if self.transposition_model not in TRANSPOSITION_MODELS:
            raise ValueError(
                f"transposition_model must be one of {TRANSPOSITION_MODELS}"
            )


def run_simulation(
    radiation: pd.DataFrame,
    panel: PanelConfig,
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
    solpos: pd.DataFrame | None = None,
    dni_extra: pd.Series | None = None,
) -> pd.DataFrame:
    """Run the idealized PV simulation for a normalised radiation DataFrame.

    `solpos` / `dni_extra` may be precomputed (cached) to avoid recomputing the
    expensive solar geometry on every call; they are re-indexed to `radiation`.

    Returns a DataFrame indexed by the same UTC timestamps with columns:
        timestamp, sun_elevation_deg, sun_azimuth_deg, aoi_deg,
        ghi, dhi, dni, poa_direct, poa_diffuse, poa_ground, poa_global,
        dc_power_w, ac_power, energy_wh
    """
    times = radiation.index
    interval_h = radiation.attrs.get("interval_h", 0.25)

    if solpos is None:
        solpos = solarposition.get_solarposition(
            times, latitude=latitude, longitude=longitude, altitude=altitude
        )
    else:
        solpos = solpos.reindex(times)
    zen = solpos["apparent_zenith"].astype(float)
    elev = solpos["apparent_elevation"].astype(float)
    az = solpos["azimuth"].astype(float)

    aoi = irradiance.aoi(
        surface_tilt=panel.tilt,
        surface_azimuth=panel.azimuth,
        solar_zenith=zen,
        solar_azimuth=az,
    )

    if dni_extra is None:
        dni_extra = irradiance.get_extra_radiation(times)
    else:
        dni_extra = dni_extra.reindex(times)
    poa = irradiance.get_total_irradiance(
        surface_tilt=panel.tilt,
        surface_azimuth=panel.azimuth,
        solar_zenith=zen,
        solar_azimuth=az,
        dni=radiation["dni"],
        ghi=radiation["ghi"],
        dhi=radiation["dhi"],
        dni_extra=dni_extra,
        model=panel.transposition_model,
        albedo=panel.albedo,
    )

    dc_w = panel.rated_power_w * poa["poa_global"].clip(lower=0.0) / 1000.0
    dc_w = dc_w.clip(upper=panel.rated_power_w)
    # Night-time guards (idea.md #7): no output when the sun is below the
    # horizon, and none when there is no horizontal irradiance.
    dc_w = dc_w.where((elev > 0.0) & (radiation["ghi"] > 0.0), 0.0)

    ac_w = dc_w * panel.inverter_efficiency
    energy_wh = ac_w * interval_h

    # Clear-sky (no-clouds) reference: same geometry, but using the clear-sky
    # irradiance components, so we can overlay "what the panel would make if
    # there were no clouds" on top of the actual output.
    poa_clear = irradiance.get_total_irradiance(
        surface_tilt=panel.tilt,
        surface_azimuth=panel.azimuth,
        solar_zenith=zen,
        solar_azimuth=az,
        dni=radiation["dni_clear"],
        ghi=radiation["ghi_clear"],
        dhi=radiation["dhi_clear"],
        dni_extra=dni_extra,
        model=panel.transposition_model,
        albedo=panel.albedo,
    )
    dc_clear_w = (panel.rated_power_w * poa_clear["poa_global"].clip(lower=0.0)
                  / 1000.0)
    dc_clear_w = dc_clear_w.clip(upper=panel.rated_power_w)
    dc_clear_w = dc_clear_w.where((elev > 0.0) & (radiation["ghi_clear"] > 0.0),
                                  0.0)
    ac_clear_w = dc_clear_w * panel.inverter_efficiency
    energy_clear_wh = ac_clear_w * interval_h

    # Enforce the no-cloud (clear-sky) ceiling: real output must never exceed it.
    # At low sun a north-facing panel can otherwise show real > clear because the
    # diffuse transposition reacts to the (higher) cloudy-sky DHI; clamping makes
    # "no cloud" a true upper bound.
    dc_w = dc_w.clip(upper=dc_clear_w)
    ac_w = dc_w * panel.inverter_efficiency
    energy_wh = ac_w * interval_h

    out = pd.DataFrame(index=times)
    out["timestamp"] = times.strftime("%Y-%m-%dT%H:%M:%SZ")  # UTC
    # New Zealand local wall time (UTC -> Pacific/Auckland), for display axes.
    out["timestamp_local"] = times.tz_convert("Pacific/Auckland").strftime(
        "%Y-%m-%d %H:%M")
    out["sun_elevation_deg"] = elev.round(4)
    out["sun_azimuth_deg"] = az.round(4)
    out["aoi_deg"] = aoi.round(4)
    out["ghi"] = radiation["ghi"].round(4)
    out["dhi"] = radiation["dhi"].round(4)
    out["dni"] = radiation["dni"].round(4)
    out["poa_direct"] = poa["poa_direct"].round(4)
    out["poa_diffuse"] = poa["poa_diffuse"].round(4)
    out["poa_ground"] = poa["poa_ground_diffuse"].round(4)
    out["poa_global"] = poa["poa_global"].round(4)
    out["dc_power_w"] = dc_w.round(3)
    out["ac_power"] = ac_w.round(3)
    out["ac_power_clear"] = ac_clear_w.round(3)
    out["energy_wh"] = energy_wh.round(3)
    out["energy_clear_wh"] = energy_clear_wh.round(3)
    # Cloud index: GHI / clear-sky GHI (1.0 = clear, <1.0 = cloud).
    # Night (no sunlight) is left as NaN so it is blank on charts and excluded
    # from monthly/weekly means — including 0 would drag the average down.
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = radiation["ghi"] / radiation["ghi_clear"]
    ci = ci.where((radiation["ghi_clear"] > 0.0) & (elev > 0.0), np.nan)
    ci = ci.clip(lower=0.0, upper=2.0)
    out["cloud_index"] = ci.round(4)
    return out


def summarize(result: pd.DataFrame, rated_kwp: float) -> dict:
    """Aggregate a simulation result into a small set of headline numbers."""
    energy_kwh = float(result["energy_wh"].sum() / 1000.0)
    clear_kwh = float(result["energy_clear_wh"].sum() / 1000.0)
    peak_kw = float(result["ac_power"].max() / 1000.0)
    n = int(len(result))
    if n > 1:
        dt = float(result.index.to_series().diff().median() / np.timedelta64(1, "h"))
    else:
        dt = 0.0
    return {
        "total_energy_kwh": round(energy_kwh, 3),
        "total_energy_clear_kwh": round(clear_kwh, 3),
        "peak_power_kw": round(peak_kw, 3),
        "specific_yield_kwh_per_kwp": round(energy_kwh / rated_kwp, 3)
        if rated_kwp else 0.0,
        "interval_count": n,
        "period_hours": round(float(dt * n), 3),
        "mean_ac_power_w": round(float(result["ac_power"].mean()), 3),
    }


def aggregate_energy(result: pd.DataFrame, period: str) -> list[dict]:
    """Aggregate the 15-min AC energy into 'month' or 'week' buckets.

    Periods are taken in NZ-local time so the buckets align with calendar
    months / ISO weeks as seen in New Zealand. Each bucket reports the real
    energy, the clear-sky (no-cloud) energy, the difference (no_cloud_extra),
    and the share of the annual total. Weekly buckets also carry the Monday
    (week-starting) date as an ISO string.
    """
    from datetime import date as _date

    local = result.index.tz_convert("Pacific/Auckland")
    real = result["energy_wh"].to_numpy()
    clear = result["energy_clear_wh"].to_numpy()
    ci = result["cloud_index"].to_numpy()

    if period == "month":
        per = local.to_period("M")
        sr = pd.Series(real, index=per).groupby(level=0).sum()
        sc = pd.Series(clear, index=per).groupby(level=0).sum()
        sm_ci = pd.Series(ci, index=per).groupby(level=0).mean()
        keys = [str(p) for p in sr.index]
        labels = [f"{p.strftime('%b %Y')}" for p in sr.index]
        week_start = [None] * len(sr)
    elif period == "week":
        cal = local.isocalendar()
        df = pd.DataFrame({"y": cal.year.to_numpy(), "w": cal.week.to_numpy(),
                           "re": real, "ce": clear, "ci": ci})
        g = df.groupby(["y", "w"], sort=True).agg({"re": "sum", "ce": "sum", "ci": "mean"})
        sr, sc, sm_ci = g["re"], g["ce"], g["ci"]
        keys = [f"{y}-W{w:02d}" for y, w in g.index]
        labels = [f"{y} W{w:02d}" for y, w in g.index]
        week_start = [_date.fromisocalendar(y, w, 1).isoformat() for y, w in g.index]
    else:
        raise ValueError(f"period must be 'month' or 'week', got {period!r}")

    total = float(sr.sum())
    out = []
    for i, key in enumerate(keys):
        re_wh = float(sr.values[i])
        ce_wh = float(sc.values[i])
        out.append({
            "key": key,
            "label": labels[i],
            "energy_kwh": round(re_wh / 1000.0, 3),
            "energy_clear_kwh": round(ce_wh / 1000.0, 3),
            "no_cloud_extra": round(max(ce_wh - re_wh, 0.0) / 1000.0, 3),
            "cloud_index": round(float(sm_ci.values[i]), 3),
            "share": round(re_wh / total, 4) if total else 0.0,
            "week_start": week_start[i],
        })

    # Always show 52 weekly bars: if the ISO year has 53 weeks (e.g. 2020, 2025),
    # simply ignore the 53rd week rather than merging it.
    if period == "week" and len(out) == 53:
        out = out[:-1]

    return out




def data_quality_report(
    radiation: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
) -> dict:
    """Build a data-quality report for a normalised radiation dataset.

    Checks (idea.md #14):
      * time: resolution, expected vs actual rows, duplicate timestamps, gaps
      * radiation: negative values, GHI ~ DHI + DNI*cos(zenith), plausibility
      * reliability: distribution of the reliability flag
    """
    idx = radiation.index
    n = int(len(radiation))
    interval_h = float(radiation.attrs.get("interval_h", 0.25))
    span_h = float((idx.max() - idx.min()).total_seconds() / 3600.0) if n > 1 else 0.0
    expected = int(round(span_h / interval_h)) + 1 if n > 1 else n

    diffs = idx.to_series().diff().dt.total_seconds().dropna()
    gaps = diffs[diffs > interval_h * 3600 * 1.5]
    biggest = gaps.sort_values(ascending=False).head(5)
    gap_list = [
        {"after": str(ts_gap), "hours": round(hours / 3600.0, 2)}
        for ts_gap, hours in biggest.items()
    ]
    duplicates = int(idx.duplicated().sum())
    missing = max(expected - n, 0)

    rad_cols = ["ghi", "dhi", "dni", "ghi_clear", "dhi_clear", "dni_clear"]
    negatives = {c: int((radiation[c] < 0).sum()) for c in rad_cols}
    ranges = {
        c: [round(float(radiation[c].min()), 1), round(float(radiation[c].max()), 1)]
        for c in rad_cols
    }

    # GHI conservation: GHI ~= DHI + BHI, with BHI = DNI * cos(zenith).
    zen = solarposition.get_solarposition(
        idx, latitude=latitude, longitude=longitude, altitude=altitude
    )["apparent_zenith"]
    bhi = radiation["dni"] * np.cos(np.radians(zen))
    residual = radiation["ghi"] - (radiation["dhi"] + bhi)

    rel = radiation["reliability"]
    rel_report = {
        "min": round(float(rel.min()), 3),
        "median": round(float(rel.median()), 3),
        "below_1": int((rel < 1.0).sum()),
        "below_0_5": int((rel < 0.5).sum()),
        "low_pct": round(float((rel < 1.0).sum()) / n * 100.0, 2) if n else 0.0,
    }

    checks = []
    for c, cnt in negatives.items():
        if cnt:
            checks.append({"level": "error", "msg": f"{c}: {cnt} negative values"})
    if duplicates:
        checks.append({"level": "warn", "msg": f"{duplicates} duplicate timestamps"})
    if missing:
        checks.append({"level": "warn", "msg": f"{missing} missing intervals"})
    for g in gap_list:
        checks.append({"level": "warn",
                       "msg": f"gap of {g['hours']}h after {g['after']}"})
    if abs(float(residual.mean())) > 5.0:
        checks.append({"level": "warn",
                       "msg": f"GHI-(DHI+DNI*cosz) mean residual {float(residual.mean()):.1f} W/m2"})
    if rel_report["low_pct"] > 5.0:
        checks.append({"level": "info",
                       "msg": f"{rel_report['low_pct']}% of intervals have reliability < 1.0"})

    status = "good" if not any(c["level"] == "error" for c in checks) else "issues"
    return {
        "span": {
            "start": str(idx.min()),
            "end": str(idx.max()),
            "interval_h": interval_h,
            "rows": n,
        },
        "time": {
            "expected_intervals": expected,
            "rows": n,
            "duplicates": duplicates,
            "missing_intervals": missing,
            "completeness_pct": round(n / expected * 100.0, 2) if expected else 100.0,
            "gaps": gap_list,
            "timezone_utc_aware": bool(idx.tz is not None),
        },
        "radiation": {
            "negatives": negatives,
            "ranges": ranges,
            "ghi_conservation": {
                "mean_residual": round(float(residual.mean()), 3),
                "max_abs_residual": round(float(residual.abs().max()), 3),
            },
            "dhi_le_ghi_violations": int((radiation["dhi"] > radiation["ghi"] + 0.01).sum()),
            "bhi_le_ghi_violations": int((bhi > radiation["ghi"] + 0.01).sum()),
        },
        "reliability": rel_report,
        "checks": checks,
        "status": status,
    }

