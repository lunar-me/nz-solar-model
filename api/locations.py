"""Location registry (Supabase-only).

Each entry maps a short API key to a location's metadata (display name, region,
coordinates/altitude, and the ``location`` value used in the Supabase
``cams_radiation`` table).  The CAMS ``.csv`` files that once fed the model are
legacy / reference only — the app reads radiation and electricity exclusively
from Supabase.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    key: str
    name: str
    region: str
    # Value of the `location` column in the Supabase cams_radiation table.
    supabase_name: str = ""
    # Coordinates/altitude are not stored in the Supabase table, so they are
    # pinned here (they match the CAMS CSV headers for each location).
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0


LOCATIONS: dict[str, Location] = {
    "auckland": Location(
        key="auckland",
        name="Auckland",
        region="North Island",
        supabase_name="Auckland",
        latitude=-36.7341,
        longitude=174.7081,
        altitude=51.00,
    ),
    "christchurch": Location(
        key="christchurch",
        name="Christchurch",
        region="South Island",
        supabase_name="Christchurch",
        latitude=-43.5372,
        longitude=172.7049,
        altitude=8.00,
    ),
}


def get_location(key: str) -> Location:
    if key not in LOCATIONS:
        raise KeyError(
            f"Unknown location '{key}'. Available: {sorted(LOCATIONS)}"
        )
    return LOCATIONS[key]
