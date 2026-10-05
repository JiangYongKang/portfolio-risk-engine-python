"""Translate HTTP schemas into domain objects.

Any malformed raw input (e.g. an unparseable date string) is converted
into a typed ``RiskEngineError`` here, at the boundary, so that in a
batch only the offending item fails — raw ``ValueError``/``TypeError``
from parsing never aborts the whole batch.
"""

from __future__ import annotations

from datetime import date

from risk_engine.api.schemas import (
    AssessRequest,
    CorrelationSchema,
    MarketDataSchema,
    PortfolioSchema,
)
from risk_engine.errors import InvalidInputError, RiskEngineError
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
    try:
        return Portfolio(
            name=schema.name,
            holdings=tuple(
                Holding(instrument=h.instrument, quantity=float(h.quantity))
                for h in schema.holdings
            ),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"portfolio '{schema.name}': malformed holding: {exc}"
        ) from exc


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"malformed date {raw!r}: expected ISO format 'YYYY-MM-DD'"
        ) from exc


def to_market(schema: MarketDataSchema) -> MarketData:
    return MarketData(
        dates=tuple(_parse_date(d) for d in schema.dates),
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
