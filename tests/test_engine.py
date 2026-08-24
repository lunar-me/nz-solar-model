"""Unit tests for the PV engine + CAMS loader.

Run from the repo root:
    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.engine import PanelConfig, run_simulation, summarize  # noqa: E402
from app.loader import load_radiation  # noqa: E402
from app.locations import LOCATIONS, get_location  # noqa: E402

ALL_LOCATIONS = sorted(LOCATIONS)


@pytest.fixture(scope="module")
def radiation() -> dict:
    """Load each location exactly once and cache by key."""
    return {key: load_radiation(get_location(key).file) for key in ALL_LOCATIONS}


@pytest.fixture(scope="module")
def auckland(radiation):
    return radiation["auckland"]


def test_load_metadata(radiation):
    for key, df in radiation.items():
        meta = df.attrs["metadata"]
        assert meta["latitude"] < 0 and meta["longitude"] > 0
        assert 0 < df.attrs["interval_h"] <= 1.0
        assert df.index.tz is not None  # timezone-aware (UTC)
        assert len(df) > 50_000  # multi-year dataset


def test_night_is_zero(radiation):
    panel = PanelConfig()
    for key, df in radiation.items():
        meta = df.attrs["metadata"]
        night = df.loc[["2020-06-21 12:00"]]  # local midnight in NZ winter
        result = run_simulation(night, panel, meta["latitude"],
                                meta["longitude"], meta["altitude"])
        assert (result["dc_power_w"] == 0).all()
        assert (result["energy_wh"] == 0).all()


def test_solstice_energy_positive_and_clipped(radiation):
    panel = PanelConfig()
    for key, df in radiation.items():
        meta = df.attrs["metadata"]
        day_df = df.loc["2020-12-21 18:00":"2020-12-22 18:00"]
        result = run_simulation(day_df, panel, meta["latitude"],
                                meta["longitude"], meta["altitude"])
        assert result["energy_wh"].sum() > 0
        assert result["ac_power"].max() <= panel.rated_power_w + 1e-6


def test_summer_exceeds_winter(auckland):
    panel = PanelConfig()
    meta = auckland.attrs["metadata"]
    summer = run_simulation(
        auckland.loc["2020-12-21 00:00":"2020-12-22 00:00"], panel,
        meta["latitude"], meta["longitude"], meta["altitude"])
    winter = run_simulation(
        auckland.loc["2020-06-21 00:00":"2020-06-22 00:00"], panel,
        meta["latitude"], meta["longitude"], meta["altitude"])
    assert summer["energy_wh"].sum() > winter["energy_wh"].sum()


def test_horizontal_panel_equals_ghi(auckland):
    """tilt=0 -> POA tracks GHI closely on a bright day."""
    panel = PanelConfig(tilt=0.0, azimuth=180.0)
    meta = auckland.attrs["metadata"]
    bright = auckland.loc["2020-01-10 00:00":"2020-01-10 02:00"]
    result = run_simulation(bright, panel, meta["latitude"],
                            meta["longitude"], meta["altitude"])
    assert np.allclose(result["poa_global"].values, result["ghi"].values,
                       atol=20.0)


def test_idealized_power_linear_in_poa(auckland):
    """DC power == rated * poa/1000, clipped at rated."""
    panel = PanelConfig(rated_power_kwp=1.0)
    meta = auckland.attrs["metadata"]
    day = auckland.loc["2020-01-10 00:00":"2020-01-10 06:00"]
    result = run_simulation(day, panel, meta["latitude"],
                            meta["longitude"], meta["altitude"])
    expected = np.minimum(result["poa_global"] * 1.0, panel.rated_power_w)
    assert np.allclose(result["dc_power_w"], expected, atol=1.0)


def test_summary_shape(auckland):
    meta = auckland.attrs["metadata"]
    result = run_simulation(
        auckland.loc["2020-01-10 00:00":"2020-01-10 06:00"], PanelConfig(),
        meta["latitude"], meta["longitude"], meta["altitude"])
    s = summarize(result, 1.0)
    assert s["total_energy_kwh"] > 0
    assert s["peak_power_kw"] > 0


def test_locations_switch_registry():
    """The two CAMS files map to distinct locations with distinct coords."""
    from app.locations import get_location
    ak = load_radiation(get_location("auckland").file)
    ch = load_radiation(get_location("christchurch").file)
    assert ak.attrs["metadata"]["latitude"] != ch.attrs["metadata"]["latitude"]


def test_clear_sky_power_column(auckland):
    """Engine exposes a no-clouds AC power series, and real never exceeds it."""
    meta = auckland.attrs["metadata"]
    day = auckland.loc["2020-01-10 00:00":"2020-01-10 06:00"]
    result = run_simulation(day, PanelConfig(), meta["latitude"],
                            meta["longitude"], meta["altitude"])
    assert "ac_power_clear" in result.columns
    assert (result["ac_power"] <= result["ac_power_clear"] + 1e-6).all()
    assert result["ac_power_clear"].sum() >= result["ac_power"].sum() - 1e-3



def test_aggregate_month_and_week(auckland):
    """Monthly and weekly buckets cover the NZ year and sum to the annual total."""
    from app.engine import aggregate_energy
    meta = auckland.attrs["metadata"]
    # NZ-local calendar year (like the API): Jan 1 00:00 -> Jan 1 next year, exclusive.
    start = pd.Timestamp("2020-01-01", tz="Pacific/Auckland")
    end = pd.Timestamp("2021-01-01", tz="Pacific/Auckland")
    year = auckland.loc[start.tz_convert("UTC"):end.tz_convert("UTC")]
    year = year[year.index < end.tz_convert("UTC")]
    result = run_simulation(year, PanelConfig(), meta["latitude"],
                            meta["longitude"], meta["altitude"])
    annual_kwh = result["energy_wh"].sum() / 1000.0

    months = aggregate_energy(result, "month")
    weeks = aggregate_energy(result, "week")
    assert len(months) == 12
    assert len(weeks) == 52
    assert abs(sum(m["energy_kwh"] for m in months) - annual_kwh) < 0.01
    # Week 53 is ignored, so the weekly sum never exceeds the annual total.
    assert sum(w["energy_kwh"] for w in weeks) <= annual_kwh + 0.01
    # clear-sky energy and weekly week-starting dates are present
    assert all(m["energy_clear_kwh"] >= m["energy_kwh"] for m in months)
    assert all(m["no_cloud_extra"] >= 0 for m in months)
    assert all(w["week_start"] for w in weeks)

