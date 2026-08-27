"""Unit tests for the PV engine + Supabase loaders.

The engine is tested against real radiation data pulled from Supabase, so these
tests require network access and valid credentials in ``.env`` (SUPABASE_URL /
SUPABASE_PUBLISHABLE_KEY).

Run from the repo root:
    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.engine import PanelConfig, run_simulation, summarize  # noqa: E402
from api.loader import load_radiation_from_supabase  # noqa: E402
from api.locations import LOCATIONS  # noqa: E402

ALL_LOCATIONS = sorted(LOCATIONS)


@pytest.fixture(scope="module")
def radiation() -> dict:
    """Load each location exactly once from Supabase and cache by key."""
    return {key: load_radiation_from_supabase(key) for key in ALL_LOCATIONS}


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
    """The two locations map to distinct coordinates via Supabase."""
    ak = load_radiation_from_supabase("auckland")
    ch = load_radiation_from_supabase("christchurch")
    assert ak.attrs["metadata"]["latitude"] != ch.attrs["metadata"]["latitude"]
    assert ak.attrs["metadata"]["longitude"] != ch.attrs["metadata"]["longitude"]


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
    from api.engine import aggregate_energy
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



def test_data_quality_report(auckland):
    """The data-quality report flags the full dataset as internally consistent."""
    from api.engine import data_quality_report
    meta = auckland.attrs["metadata"]
    rep = data_quality_report(auckland, meta["latitude"], meta["longitude"],
                              meta["altitude"])
    assert rep["time"]["rows"] > 100_000
    assert rep["time"]["duplicates"] == 0
    assert rep["radiation"]["negatives"]["ghi"] == 0
    assert abs(rep["radiation"]["ghi_conservation"]["mean_residual"]) < 1.0
    assert rep["radiation"]["dhi_le_ghi_violations"] == 0



def test_cloud_index_column(auckland):
    """The engine exposes a cloud index (GHI / GHI_clear), 0 at night."""
    from api.engine import aggregate_energy
    meta = auckland.attrs["metadata"]
    day = auckland.loc["2020-01-10 00:00":"2020-01-10 06:00"]
    result = run_simulation(day, PanelConfig(), meta["latitude"],
                            meta["longitude"], meta["altitude"])
    assert "cloud_index" in result.columns
    assert (result["cloud_index"].dropna() >= 0).all()
    # night (no sunlight) should be NaN so it is blank / excluded
    night = result[result["sun_elevation_deg"] <= 0]
    if len(night):
        assert night["cloud_index"].isna().all()
    # aggregation buckets carry the mean cloud index too (use a quarter-year)
    q = auckland.loc["2020-01-01":"2020-03-31"]
    res = run_simulation(q, PanelConfig(), meta["latitude"],
                         meta["longitude"], meta["altitude"])
    months = aggregate_energy(res, "month")
    assert all("cloud_index" in m for m in months)
    assert all(0 <= m["cloud_index"] <= 1.5 for m in months)


def test_self_consumption_helpers_synthetic():
    """The shared self-consumption / savings helpers used by /api/money and
    /api/model-money compute correctly on synthetic data (no network)."""
    from api.index import _self_consumption_columns, _money_totals, _money_monthly

    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({
        "consumption_kwh": [10.0, 0.0, 5.0],
        "solar_kwh": [8.0, 4.0, 0.0],
        "cost_$": [3.5, 0.0, 1.75],
    }, index=idx)

    df = _self_consumption_columns(df)
    # hour0: consume 10, solar 8  -> self 8, excess 0, grid 2, savings 8/10*3.5=2.8
    # hour1: consume 0,  solar 4  -> self 0, excess 4, grid 0, savings 0
    # hour2: consume 5,  solar 0  -> self 0, excess 0, grid 5, savings 0
    assert list(df["self_consumed_kwh"]) == [8.0, 0.0, 0.0]
    assert list(df["excess_kwh"]) == [0.0, 4.0, 0.0]
    assert list(df["grid_kwh"]) == [2.0, 0.0, 5.0]
    assert list(df["savings_$"]) == [pytest.approx(2.8), 0.0, 0.0]

    totals = _money_totals(df)
    assert totals["consumption_kwh"] == 15.0
    assert totals["solar_kwh"] == 12.0
    assert totals["savings_$"] == 2.8
    assert totals["cost_without_solar_$"] == 5.25
    assert totals["cost_with_solar_$"] == 2.45
    # wasted solar (4 kWh) is valued at hour1's effective rate (cost/consumption
    # = 0 because consumption was 0), so waste is $0 at that hour.
    assert totals["wasted_value_$"] == 0.0

    months = _money_monthly(df)
    assert len(months) == 1
    assert months[0]["savings_$"] == 2.8


def test_model_money_math():
    """Modelled hourly consumption/cost = annual_kwh * usage_percent / 100, and
    the cost totals annual_kwh * price (the region loader is exercised live in
    test_load_region_generation_nz_time)."""
    import numpy as np

    # usage_percent is a percentage number; annual=8000, price=0.35.
    annual, price = 8000.0, 0.35
    pct = pd.Series([0.5, 1.0, 2.5])  # sums to 4.0 (percent)
    consumption = annual * pct / 100.0
    cost = price * annual * pct / 100.0
    assert np.allclose(consumption, [40.0, 80.0, 200.0])
    assert np.allclose(cost, [14.0, 28.0, 70.0])
    assert np.isclose(consumption.sum(), annual * pct.sum() / 100.0)
    assert np.isclose(cost.sum(), price * annual * pct.sum() / 100.0)


def test_add_daily_charges():
    """A fixed daily charge adds $/day to the dollar totals only — it never
    touches the per-hour kWh or the solar self-consumption / savings columns."""
    from api.index import _self_consumption_columns, _add_daily_charges, _money_totals

    # 48 hourly rows = exactly 2 full NZ days: starting 2025-06-01 12:00 UTC
    # (= 2025-06-02 00:00 NZ) for 48h covers NZ 06-02 and 06-03.
    idx = pd.date_range("2025-06-01 12:00", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({
        "consumption_kwh": [10.0] * 48,
        "solar_kwh": [5.0] * 48,
        "cost_$": [3.5] * 48,   # variable cost
    }, index=idx)

    base = _self_consumption_columns(df)
    with_charge = _add_daily_charges(base.copy(), 1.50)
    totals_base = _money_totals(base)
    totals_charge = _money_totals(with_charge)

    # kWh columns unchanged
    assert (base["consumption_kwh"] == with_charge["consumption_kwh"]).all()
    assert (base["solar_kwh"] == with_charge["solar_kwh"]).all()
    assert (base["self_consumed_kwh"] == with_charge["self_consumed_kwh"]).all()
    # savings unchanged (fixed charge is not savable)
    assert totals_base["savings_$"] == totals_charge["savings_$"]
    # 2 days * 1.5 = 3.0 added to the dollar totals
    assert abs(totals_charge["cost_without_solar_$"] - totals_base["cost_without_solar_$"] - 3.0) < 1e-9
    assert abs(totals_charge["cost_with_solar_$"] - totals_base["cost_with_solar_$"] - 3.0) < 1e-9


def test_load_region_generation_nz_time():
    """The region 2025 loader converts UTC->NZ time so usage_percent sums to ~100
    and the index spans the full NZ 2025 calendar year (network-backed)."""
    from api.loader import load_region_generation_from_supabase

    for region in ("Auckland", "Canterbury"):
        df = load_region_generation_from_supabase(region)
        assert "usage_percent" in df.columns
        nz = df.index.tz_convert("Pacific/Auckland")
        assert nz.min().year == 2025 and nz.max().year == 2025
        # usage_percent is a percentage (0..100 scale) of annual use -> ~100 total
        assert 99.0 < df["usage_percent"].sum() <= 100.5
        assert df["usage_percent"].min() >= 0.0


