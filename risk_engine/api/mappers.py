"""Translate HTTP schemas into domain objects."""

from __future__ import annotations

from datetime import date

from risk_engine.api.schemas import (
    AssessRequest,
    CorrelationSchema,
    MarketDataSchema,
    PortfolioSchema,
)
from risk_engine.models import (
    CorrelationAssumption,
    FactorShock,
    Holding,
    MarketData,
    Portfolio,
    StressScenario,
)
from risk_engine.service import AssessmentJob


def to_portfolio(schema: PortfolioSchema) -> Portfolio:
    return Portfolio(
        name=schema.name,
        holdings=tuple(
            Holding(instrument=h.instrument, quantity=float(h.quantity))
            for h in schema.holdings
        ),
    )


def to_market(schema: MarketDataSchema) -> MarketData:
    return MarketData(
        dates=tuple(date.fromisoformat(d) for d in schema.dates),
        prices={inst: list(series) for inst, series in schema.prices.items()},
    )


def to_correlation(schema: CorrelationSchema) -> CorrelationAssumption:
    return CorrelationAssumption(
        name=schema.name,
        instruments=tuple(schema.instruments),
        matrix=tuple(tuple(row) for row in schema.matrix),
        version=schema.version,
    )


def to_job(schema: AssessRequest, correlation=None) -> AssessmentJob:
    return AssessmentJob(
        portfolio=to_portfolio(schema.portfolio),
        market=to_market(schema.market),
        window=schema.window,
        confidence=schema.confidence,
        horizon_days=schema.horizon_days,
        correlation=(
            correlation
            if correlation is not None
            else (to_correlation(schema.correlation) if schema.correlation else None)
        ),
        version=schema.version,
    )
