"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PanelConfigIn(BaseModel):
    tilt: float = Field(25.0, ge=0.0, le=90.0, description="Degrees from horizontal")
    azimuth: float = Field(0.0, ge=0.0, le=360.0,
                           description="Degrees from North (0=N,90=E,180=S,270=W)")
    rated_power_kwp: float = Field(5.0, gt=0.0)
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
    """Solar self-consumption / savings over the fixed Christchurch year.

    Electricity consumption always comes from the Christchurch dataset; the
    ``location`` selects which city's radiation drives the solar side.
    """
    location: str = Field("christchurch",
                          description="Location key; solar output uses this city's radiation, "
                                      "electricity consumption is always Christchurch")
    panel: PanelConfigIn = Field(default_factory=PanelConfigIn)


class ModelMoneyRequest(BaseModel):
    """Solar savings against *modelled* hourly consumption.

    Unlike :class:`MoneyRequest` (which uses Christchurch's real consumption),
    hourly usage is modelled by spreading the user's annual kWh over the
    selected region's real 2025 electricity-generation curve, priced at a flat
    per-kWh rate (incl. GST), then fed through the same self-consumption /
    solar-savings calculation.
    """
    location: str = Field("christchurch",
                          description="Location key; maps to the region(s) that model hourly usage "
                                      "(Auckland=Waikato, Christchurch=Canterbury) "
                                      "and drives the solar side")
    annual_kwh: float = Field(10000.0, gt=0.0,
                              description="User's annual electricity consumption in kWh")
    kwh_price_gst: float = Field(0.35, gt=0.0,
                                 description="Price per kWh including GST, in NZ dollars")
    daily_charge: float = Field(1.50, ge=0.0,
                                description="Fixed daily connection fee in NZ dollars, added to "
                                            "every day's cost regardless of consumption")
    panel: PanelConfigIn = Field(default_factory=PanelConfigIn)


class ModelMoneyDailyRequest(ModelMoneyRequest):
    """Hourly detail for a single NZ day of the Model-money model.

    Extends :class:`ModelMoneyRequest` with a specific NZ-local day so the
    frontend can chart that day's hourly solar / consumption / savings curves
    without re-running the whole year.
    """
    date: str = Field("2025-08-18",
                      description="NZ-local date (YYYY-MM-DD) within 2025 to show hourly detail for")


class CurvesDailyRequest(BaseModel):
    """Daily hourly generation (MWh) for every region of both islands."""
    date: str = Field("2025-08-18",
                      description="NZ-local date (YYYY-MM-DD) within 2025 to chart")

