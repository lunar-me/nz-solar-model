"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PanelConfigIn(BaseModel):
    tilt: float = Field(25.0, ge=0.0, le=90.0, description="Degrees from horizontal")
    azimuth: float = Field(0.0, ge=0.0, le=360.0,
                           description="Degrees from North (0=N,90=E,180=S,270=W)")
    rated_power_kwp: float = Field(1.0, gt=0.0)
    albedo: float = Field(0.2, ge=0.0, le=1.0)
    transposition_model: Literal["perez", "haydavies", "isotropic"] = "perez"
    inverter_efficiency: float = Field(0.95, gt=0.0, le=1.0)


class SimulateRequest(BaseModel):
    location: str = Field(..., description="Location key, e.g. 'auckland'")
    start: str | None = Field(None, description="Start ISO timestamp (UTC). Default = start of data.")
    end: str | None = Field(None, description="End ISO timestamp (UTC, exclusive). Default = end of data.")
    panel: PanelConfigIn = Field(default_factory=PanelConfigIn)
    include_radiation: bool = Field(True, description="Echo GHI/DHI/DNI in the timeseries")


class AggregateRequest(BaseModel):
    """Annual aggregation over a full calendar year (monthly or weekly buckets)."""
    location: str = Field(..., description="Location key, e.g. 'auckland'")
    year: int = Field(..., ge=2020, le=2025, description="Calendar year to aggregate")
    period: Literal["month", "week"] = "month"
    panel: PanelConfigIn = Field(default_factory=PanelConfigIn)


class StabilityRequest(BaseModel):
    """Year-over-year stability: totals for every year in the dataset."""
    location: str = Field(..., description="Location key, e.g. 'auckland'")
    panel: PanelConfigIn = Field(default_factory=PanelConfigIn)


class MoneyRequest(BaseModel):
    """Solar self-consumption / savings over the fixed Christchurch year."""
    location: str = Field("christchurch", description="Locked to Christchurch")
    price_per_kwh: float | None = Field(
        None, gt=0.0,
        description="$/kWh used to value wasted solar; defaults to the bill's effective rate")
    panel: PanelConfigIn = Field(default_factory=PanelConfigIn)

