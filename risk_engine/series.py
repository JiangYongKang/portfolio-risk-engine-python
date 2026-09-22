"""Price-series preparation: alignment, returns, window validation."""

from __future__ import annotations

from datetime import date

from risk_engine.models import InstrumentId, MarketData, Portfolio

Matrix = list[list[float]]
Vector = list[float]


def aligned_price_matrix(
    portfolio: Portfolio,
    market: MarketData,
) -> tuple[tuple[date, ...], tuple[InstrumentId, ...], Matrix]:
    raise NotImplementedError


def simple_returns(prices: Matrix) -> Matrix:
    raise NotImplementedError


def window_slice(returns: Matrix, window: int) -> Matrix:
    raise NotImplementedError
