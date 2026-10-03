"""Pydantic request/response schemas for the HTTP layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HoldingSchema(BaseModel):
    instrument: str
    quantity: float


class PortfolioSchema(BaseModel):
    name: str
    holdings: list[HoldingSchema]


class MarketDataSchema(BaseModel):
    dates: list[str]
    prices: dict[str, list[float]]


class CorrelationSchema(BaseModel):
    name: str
    instruments: list[str]
    matrix: list[list[float]]
    version: str = "v1"


class AssessRequest(BaseModel):
    portfolio: PortfolioSchema
    market: MarketDataSchema
    window: int = Field(default=20, ge=1)
    confidence: float = Field(default=0.99, gt=0.5, lt=1.0)
    horizon_days: int = Field(default=1, ge=1)
    correlation: CorrelationSchema | None = None
    correlation_name: str | None = None
    version: str = "v1"


class BatchAssessRequest(BaseModel):
    requests: list[AssessRequest]
    max_items: int | None = None
    timeout_seconds: float | None = None


class ScanJobSchema(BaseModel):
    portfolio: PortfolioSchema
    market: MarketDataSchema
    scenario_names: list[str]


class ScanRequest(BaseModel):
    jobs: list[ScanJobSchema]
    max_items: int | None = None
    timeout_seconds: float | None = None


class ReassessJobSchema(BaseModel):
    portfolio: PortfolioSchema
    market: MarketDataSchema
    stamped_version: str
    target_version: str
    original: dict
    window: int = Field(default=20, ge=1)
    confidence: float = Field(default=0.99, gt=0.5, lt=1.0)
    horizon_days: int = Field(default=1, ge=1)
    correlation: CorrelationSchema | None = None
    correlation_name: str | None = None
    allow_breaking: bool = False


class ReassessRequest(BaseModel):
    jobs: list[ReassessJobSchema]
    max_items: int | None = None
    timeout_seconds: float | None = None


class ErrorResponse(BaseModel):
    code: str
    reason: str
