"""Exposure aggregation and tail-risk metrics (historical + parametric)."""

from __future__ import annotations

from risk_engine.models import (
    CorrelationAssumption,
    MarketData,
    Portfolio,
    RiskReport,
    TailRiskEstimate,
)
from risk_engine.series import Matrix, Vector


def portfolio_returns(portfolio: Portfolio, instruments: tuple[str, ...],
                      asset_returns: Matrix) -> Vector:
    raise NotImplementedError


def exposure_breakdown(
    portfolio: Portfolio, market: MarketData
) -> tuple[float, float, dict[str, float]]:
    raise NotImplementedError


def historical_tail(
    portfolio: Portfolio, instruments: tuple[str, ...], asset_returns: Matrix,
    confidence: float, horizon_days: int,
) -> TailRiskEstimate:
    raise NotImplementedError


def parametric_tail(
    portfolio: Portfolio,
    instruments: tuple[str, ...],
    asset_returns: Matrix,
    confidence: float,
    horizon_days: int,
    correlation: CorrelationAssumption | None,
) -> tuple[TailRiskEstimate, float]:
    raise NotImplementedError


def build_report(
    portfolio: Portfolio,
    market: MarketData,
    *,
    window: int,
    confidence: float = 0.99,
    horizon_days: int = 1,
    correlation: CorrelationAssumption | None = None,
    version: str = "v1",
) -> RiskReport:
    raise NotImplementedError
