"""HTTP schemas for batch scan, cross-version recomparison and snapshots."""

from __future__ import annotations

from pydantic import BaseModel, Field

from risk_engine.api.schemas import MarketDataSchema, PortfolioSchema


class ScanCellSchema(BaseModel):
    portfolio: PortfolioSchema
    market: MarketDataSchema
    scenario_name: str
    correlation_name: str | None = None
    version: str = "v1"


class CrossProductScanRequest(BaseModel):
    # A batch of portfolios evaluated against one shared market history and
    # a batch of already-registered scenarios.
    portfolios: list[PortfolioSchema]
    market: MarketDataSchema
    scenario_names: list[str] = Field(min_length=1)
    correlation_name: str | None = None
    version: str = "v1"
    max_items: int | None = None
    max_workers: int | None = None
    timeout_seconds: float | None = None


class ScanRequest(BaseModel):
    cells: list[ScanCellSchema]
    max_items: int | None = None
    max_workers: int | None = None
    timeout_seconds: float | None = None
    title: str = "risk scan snapshot"


class HistoricalConclusionSchema(BaseModel):
    portfolio: PortfolioSchema
    market: MarketDataSchema
    stamped_version: str
    conclusion: dict
    window: int = Field(default=20, ge=1)
    confidence: float = Field(default=0.99, gt=0.5, lt=1.0)
    horizon_days: int = Field(default=1, ge=1)


class RecompareRequest(BaseModel):
    items: list[HistoricalConclusionSchema]
    target_version: str
    allow_breaking: bool = False
    max_items: int | None = None
    max_workers: int | None = None
    timeout_seconds: float | None = None
