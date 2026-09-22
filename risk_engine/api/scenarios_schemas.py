"""Schemas for stress-scenario endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from risk_engine.api.schemas import (
    CorrelationSchema,
    MarketDataSchema,
    PortfolioSchema,
)


class FactorShockSchema(BaseModel):
    instrument: str
    relative_move: float


class ScenarioSchema(BaseModel):
    name: str
    shocks: list[FactorShockSchema]
    version: str = "v1"


class ComposeSchema(BaseModel):
    result_name: str
    names: list[str] = Field(min_length=1)
    version: str = "v1"


class ScenarioEvaluateRequest(BaseModel):
    portfolio: PortfolioSchema
    market: MarketDataSchema
    scenario: ScenarioSchema | None = None
    scenario_name: str | None = None
    correlation: CorrelationSchema | None = None


class CorrelationComposeSchema(BaseModel):
    result_name: str
    names: list[str] = Field(min_length=1)
    version: str = "v1"
