"""Pydantic request/response schemas for the HTTP layer.

Numeric ranges (window, confidence, horizon) are intentionally *not*
constrained here: the engine validates them per item and raises typed
errors, so that in a batch one out-of-range item fails only its own
slot instead of rejecting the whole request at the schema boundary.
Only structural shape (types, required fields) is enforced here.
"""

from __future__ import annotations

from pydantic import BaseModel


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
    window: int = 20
    confidence: float = 0.99
    horizon_days: int = 1
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
    window: int = 20
    confidence: float = 0.99
    horizon_days: int = 1
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
