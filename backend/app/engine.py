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
    rated_power_kwp: float = 1.0  # STC rated power (kWp)
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
) -> pd.DataFrame:
    """Run the idealized PV simulation for a normalised radiation DataFrame.

    Returns a DataFrame indexed by the same UTC timestamps with columns:
        timestamp, sun_elevation_deg, sun_azimuth_deg, aoi_deg,
        ghi, dhi, dni, poa_direct, poa_diffuse, poa_ground, poa_global,
        dc_power_w, ac_power, energy_wh
    """
    times = radiation.index
    interval_h = radiation.attrs.get("interval_h", 0.25)

    solpos = solarposition.get_solarposition(
        times, latitude=latitude, longitude=longitude, altitude=altitude
    )
    zen = solpos["apparent_zenith"].astype(float)
    elev = solpos["apparent_elevation"].astype(float)
    az = solpos["azimuth"].astype(float)

    aoi = irradiance.aoi(
        surface_tilt=panel.tilt,
        surface_azimuth=panel.azimuth,
        solar_zenith=zen,
        solar_azimuth=az,
    )

    dni_extra = irradiance.get_extra_radiation(times)
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
