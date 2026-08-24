"""Location registry.

Each entry maps a short key to the local CAMS CSV for that location.  The
authoritative coordinates/altitude are parsed from the file header itself by
the loader (self-describing data), but the registry is what lets the UI and API
"switch between locations" by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class Location:
    key: str
    name: str
    region: str
    file: Path


LOCATIONS: dict[str, Location] = {
    "auckland": Location(
        key="auckland",
        name="Auckland",
        region="North Island",
        file=DATA_DIR / "CAMS Radiation - Auckland - 20200101 - 20251231.csv",
    ),
    "christchurch": Location(
        key="christchurch",
        name="Christchurch",
        region="South Island",
        file=DATA_DIR / "CAMS Radiation - Christchurch - 20200101 - 20251231.csv",
    ),
}


def get_location(key: str) -> Location:
    if key not in LOCATIONS:
        raise KeyError(
            f"Unknown location '{key}'. Available: {sorted(LOCATIONS)}"
        )
    return LOCATIONS[key]
