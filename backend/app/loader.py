"""CAMS radiation CSV loader / normalizer.

CAMS exports irradiations integrated over each 15-minute interval in Wh/m2,
prefixed by a '#'-comment header. We normalise to the internal schema from
idea.md (ghi / dhi / dni / *clear / reliability) and convert interval
irradiations to average irradiances in W/m2 for pvlib, using the timestamps
(UTC) as the (timezone-aware) index.

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

from pathlib import Path

import pandas as pd

# Order of columns in the CAMS export (after the '#' header block).
CAMS_COLUMNS = [
    "Observation_period",
    "TOA",
    "Clear sky GHI",
    "Clear sky BHI",
    "Clear sky DHI",
    "Clear sky BNI",
    "GHI",
    "BHI",
    "DHI",
    "BNI",
    "Reliability",
]

# Radiation columns that are irradiations (Wh/m2 per interval) -> scaled to W/m2.
_IRRADIANCE_COLS = {
    "TOA",
    "Clear sky GHI",
    "Clear sky BHI",
    "Clear sky DHI",
    "Clear sky BNI",
    "GHI",
    "BHI",
    "DHI",
    "BNI",
}


def parse_metadata(path: Path) -> dict:
    """Pull latitude / longitude / altitude out of the CAMS header block."""
    meta: dict = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("#"):
                break
            if line.startswith("# Latitude"):
                meta["latitude"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("# Longitude"):
                meta["longitude"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("# Altitude"):
                meta["altitude"] = float(line.split(":", 1)[1].strip())
    return meta


def load_radiation(path: Path) -> pd.DataFrame:
    """Load a CAMS CSV and return the normalised, W/m2 DataFrame (UTC index)."""
    raw = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=CAMS_COLUMNS,
        comment="#",
        encoding="utf-8-sig",
        dtype=str,
        engine="python",
    )

    start = raw["Observation_period"].str.split("/").str[0]
    index = pd.to_datetime(start, utc=True)
    index.name = None  # avoid leaking the source column name onto the index
    raw.index = index
    raw = raw.drop(columns=["Observation_period"])

    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw[~raw.index.duplicated(keep="first")].sort_index()

    # Median interval length (hours) -> Wh/m2 per interval -> average W/m2.
    interval_h = float(
        raw.index.to_series().diff().dt.total_seconds().median() / 3600.0
    )
    scale = 1.0 / interval_h
    for col in _IRRADIANCE_COLS:
        raw[col] = raw[col] * scale

    out = pd.DataFrame(index=raw.index)
    out["ghi"] = raw["GHI"]
    out["dhi"] = raw["DHI"]
    out["dni"] = raw["BNI"]  # BNI is direct-normal irradiation
    out["ghi_clear"] = raw["Clear sky GHI"]
    out["dhi_clear"] = raw["Clear sky DHI"]
    out["dni_clear"] = raw["Clear sky BNI"]
    out["reliability"] = raw["Reliability"]  # proportion, NOT scaled

    out.attrs["interval_h"] = interval_h
    out.attrs["metadata"] = parse_metadata(path)
    return out
