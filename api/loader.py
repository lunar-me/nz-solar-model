"""Data loaders — Supabase-only.

Radiation and electricity are read exclusively from the Supabase project
(``cams_radiation`` and ``christchurch_electricity_consumption``).  Both tables
store datetimes as UTC; the loaders normalise them into the internal schema
from idea.md (ghi / dhi / dni / *clear / reliability) on a timezone-aware UTC
index.  The legacy CAMS ``.csv`` files are reference only and are not read by
the app.

Internal schema (units: W/m2 unless noted):
    index        : DatetimeIndex, UTC, timezone-aware
    ghi          : global horizontal
    dhi          : diffuse horizontal
    dni          : direct normal  (from CAMS BNI)
    ghi_clear    : clear-sky global horizontal
    dhi_clear    : clear-sky diffuse horizontal
    dni_clear    : clear-sky direct normal (from CAMS clear-sky BNI)
    reliability  : proportion of reliable data in the interval (0..1, unscaled)
"""
from __future__ import annotations

import pandas as pd

from .locations import get_location
from .supabase_client import fetch_electricity, fetch_region_generation, fetch_radiation


def _utc_iso(v) -> str | None:
    """Coerce a timestamp to an ISO-8601 UTC string for PostgREST filters.

    ``None`` -> ``None``.  A tz-aware pandas Timestamp is converted to UTC;
    naive values are assumed to already be UTC (both Supabase tables store UTC).
    """
    if v is None:
        return None
    ts = pd.Timestamp(v)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    else:
        ts = ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_radiation_from_supabase(location_key: str, start=None, end=None) -> pd.DataFrame:
    """Load normalised W/m2 radiation for a location from Supabase.

    The Supabase ``cams_radiation`` table already stores average irradiances
    in W/m2 (unlike the raw CAMS CSVs, which are Wh/m2 per interval and need
    scaling), so no interval scaling is applied here.  Returns the internal
    schema described in the module docstring, with ``interval_h`` and
    ``metadata`` (latitude / longitude / altitude) attributes.

    ``start`` / ``end`` (tz-aware UTC Timestamps or ISO strings) restrict the
    fetched range to avoid pulling the whole multi-year dataset when only a
    slice is needed (e.g. the money tab's single electricity year).
    """
    loc = get_location(location_key)
    s = _utc_iso(start)
    e = _utc_iso(end)
    rows = fetch_radiation(loc.supabase_name, start=s, end=e)
    if not rows:
        raise FileNotFoundError(
            f"No radiation data for location '{location_key}' in Supabase "
            f"(table cams_radiation, location='{loc.supabase_name}')."
        )

    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["start_ts_utc"], utc=True)
    idx.name = None
    df.index = idx
    out = pd.DataFrame(index=idx)
    # Assign by position (.to_numpy()) — index-alignment would otherwise
    # produce all-NaN because df's RangeIndex never matches the timestamp index.
    out["ghi"] = pd.to_numeric(df["ghi"], errors="coerce").to_numpy()
    out["dhi"] = pd.to_numeric(df["dhi"], errors="coerce").to_numpy()
    out["dni"] = pd.to_numeric(df["bni"], errors="coerce").to_numpy()  # BNI = direct normal
    out["ghi_clear"] = pd.to_numeric(df["clear_sky_ghi"], errors="coerce").to_numpy()
    out["dhi_clear"] = pd.to_numeric(df["clear_sky_dhi"], errors="coerce").to_numpy()
    out["dni_clear"] = pd.to_numeric(df["clear_sky_bni"], errors="coerce").to_numpy()
    out["reliability"] = pd.to_numeric(df["reliability"], errors="coerce").to_numpy()

    out = out[~out.index.duplicated(keep="first")].sort_index()
    interval_h = float(
        out.index.to_series().diff().dt.total_seconds().median() / 3600.0
    )
    # The Supabase table stores CAMS *irradiations* (Wh/m2 per interval), exactly
    # like the raw CSV export. Convert to average W/m2 for pvlib by dividing by
    # the interval length in hours (e.g. x4 for 15-min data). `reliability` is a
    # proportion and is NOT scaled.
    scale = 1.0 / interval_h
    for col in ("ghi", "dhi", "dni", "ghi_clear", "dhi_clear", "dni_clear"):
        out[col] = out[col] * scale
    out.attrs["interval_h"] = interval_h
    out.attrs["metadata"] = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "altitude": loc.altitude,
    }
    return out


def load_electricity_from_supabase() -> pd.DataFrame:
    """Load hourly Christchurch consumption from Supabase.

    Returns a DataFrame indexed by timezone-aware UTC with columns
    ``consumption_kwh`` and ``cost_$``.  The Supabase ``datetime_utc`` values
    are naive UTC, so they are localized to UTC here.
    """
    rows = fetch_electricity()
    if not rows:
        raise FileNotFoundError(
            "No electricity data in Supabase (table "
            "christchurch_electricity_consumption)."
        )

    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["datetime_utc"], utc=True)
    idx.name = None
    df.index = idx
    out = pd.DataFrame(
        {
            "consumption_kwh": pd.to_numeric(df["usage_kWh"], errors="coerce").to_numpy(),
            "cost_$": pd.to_numeric(df["dollars"], errors="coerce").to_numpy(),
        },
        index=idx,
    )
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def load_region_generation_from_supabase(region: str) -> pd.DataFrame:
    """Load hourly generation share for one region, processed in NZ local time.

    The Supabase ``region_electricity_generation_2025_1h`` table stores one row
    per UTC hour, with ``usage_percent`` = the percentage (0..100 scale) of that
    region's whole-year electricity use that fell in the hour (it sums to 100
    across the year). A NZ wall-clock hour can span two UTC hours (and one NZ
    hour vanishes at the DST spring-forward), so the timestamps are converted to
    ``Pacific/Auckland``, re-binned to clean NZ-local 1h buckets, then converted
    back to UTC. That keeps the hourly index aligned with the (also hourly)
    radiation / solar series, because NZ's offset from UTC is always a whole
    number of hours.

    Returns a DataFrame indexed by UTC with a single ``usage_percent`` column.
    """
    rows = fetch_region_generation(region)
    if not rows:
        raise FileNotFoundError(
            f"No generation data for region '{region}' in Supabase "
            f"(table region_electricity_generation_2025_1h)."
        )

    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["datetime_utc"], utc=True)
    idx.name = None
    df.index = idx
    out = pd.DataFrame(
        {
            "usage_percent": pd.to_numeric(
                df["usage_percent"], errors="coerce"
            ).to_numpy(),
        },
        index=idx,
    )
    out = out[~out.index.duplicated(keep="first")].sort_index()

    # --- process in NZ local time (the table's timestamps are UTC) ---
    out.index = out.index.tz_convert("Pacific/Auckland")
    # Re-bin onto clean 1h NZ buckets. The DST spring-forward hour does not exist
    # in the source data, so it comes back as NaN and is filled with 0 (that
    # wall-clock hour never happened, so no consumption is assigned to it).
    # The source is already hourly, so each 1h bucket holds exactly one value.
    out = out.resample("1h").sum().fillna(0.0)
    out.index = out.index.tz_convert("UTC")

    # Normalise so usage_percent sums to 100 over the year. The source table's
    # scale is not guaranteed (usage_percent may be a 0..1 fraction or a 0..100
    # percentage); the model's hourly `annual * usage_percent / 100` needs the
    # annual total to be ~100 so the modelled consumption equals the user's kWh.
    total = float(out["usage_percent"].sum())
    if total > 0:
        out["usage_percent"] = out["usage_percent"] / total * 100.0
    return out
