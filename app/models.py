"""API / domain data models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --- portfolio & market data ------------------------------------------------------

class Position(BaseModel):
    symbol: str
    quantity: float = Field(description="unit quantity (shares); may be negative")


class Portfolio(BaseModel):
    name: str
    positions: list[Position]


class PriceSeries(BaseModel):
    symbol: str
    prices: list[float] = Field(description="chronologically ordered prices, oldest first")


# --- risk requests / responses ----------------------------------------------------

class ConfidenceLevel(BaseModel):
    confidence: float = Field(gt=0.5, lt=1.0)
    window: int = Field(gt=1, description="number of most recent return observations to use")


class RiskSummary(BaseModel):
    portfolio: str
    window: int
    observations_used: int
    confidence: float
    portfolio_value: float
    historical_var: float
    historical_es: float
    parametric_var: float
    parametric_es: float


# --- scenarios --------------------------------------------------------------------

class ScenarioType(str, Enum):
    factor_shock = "factor_shock"
    composite = "composite"


class FactorShock(BaseModel):
    """A single-factor relative shock, e.g. equity -0.10 means -10%."""

    factor: str
    shock: float


class ScenarioDefinition(BaseModel):
    name: str
    type: ScenarioType
    shocks: list[FactorShock] = []
    components: list[str] = []  # names of named scenarios, for composite
    description: str = ""


class ScenarioResult(BaseModel):
    scenario: str
    base_value: float
    stressed_value: float
    pnl: float
    pnl_pct: float
    factor_contributions: dict[str, float]


# --- correlation assumptions -------------------------------------------------------

class CorrelationOverride(BaseModel):
    factors: list[str]
    matrix: list[list[float]]


# --- versioning -------------------------------------------------------------------

class VersionedParameterSet(BaseModel):
    name: str
    version: int
    rule_version: int
    params: dict
    compatible_with: list[int] = []


class ScenarioEvaluation(BaseModel):
    scenario: str
    version: int
    rule_version: int
    reproducible: bool
    result: ScenarioResult | None = None
    note: str = ""


# --- batch ------------------------------------------------------------------------

class BatchItem(BaseModel):
    id: str
    portfolio: Portfolio
    confidence: ConfidenceLevel


class BatchEntry(BaseModel):
    id: str
    ok: bool
    reason: str = ""
    detail: str = ""
    result: RiskSummary | None = None
