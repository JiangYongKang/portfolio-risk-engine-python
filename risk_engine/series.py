"""Price-series preparation: alignment, returns, window validation."""

from __future__ import annotations

from datetime import date

from risk_engine.errors import (
    InsufficientDataError,
    InvalidWindowError,
    UnknownInstrumentError,
)
from risk_engine.models import InstrumentId, MarketData, Portfolio

Matrix = list[list[float]]
Vector = list[float]

MIN_OBSERVATIONS = 2  # need at least 2 prices -> 1 return


def aligned_price_matrix(
    portfolio: Portfolio,
    market: MarketData,
) -> tuple[tuple[date, ...], tuple[InstrumentId, ...], Matrix]:
    """Align portfolio instruments to market dates.

    Returns (dates, instruments, price_matrix) where price_matrix is
    rows=dates x cols=instruments.  Raises typed errors for unknown
    instruments, ragged series, or insufficient history.
    """
    if not portfolio.holdings:
        raise InsufficientDataError("portfolio has no holdings")
    if len(market.dates) < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"market data has {len(market.dates)} dates, need >= {MIN_OBSERVATIONS}"
        )

    instruments = tuple(dict.fromkeys(portfolio.instruments))
    n_dates = len(market.dates)
    columns: list[list[float]] = []
    for inst in instruments:
        series = market.prices.get(inst)
        if series is None:
            raise UnknownInstrumentError(
                f"no market data for instrument '{inst}'"
            )
        if len(series) != n_dates:
            raise InsufficientDataError(
                f"series for '{inst}' has {len(series)} points, expected {n_dates}"
            )
        if any(p <= 0 for p in series):
            raise InsufficientDataError(
                f"series for '{inst}' contains non-positive prices"
            )
        columns.append([float(p) for p in series])

    matrix: Matrix = [
        [columns[j][i] for j in range(len(instruments))] for i in range(n_dates)
    ]
    return tuple(market.dates), instruments, matrix


def simple_returns(prices: Matrix) -> Matrix:
    """Row-wise simple returns: r[t] = p[t]/p[t-1] - 1."""
    if len(prices) < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"need >= {MIN_OBSERVATIONS} price rows to compute returns"
        )
    out: Matrix = []
    for prev, cur in zip(prices, prices[1:]):
        out.append([c / p - 1.0 for p, c in zip(prev, cur)])
    return out


def window_slice(returns: Matrix, window: int) -> Matrix:
    """Take the most recent ``window`` return rows, with validation."""
    if window < 1:
        raise InvalidWindowError(f"window must be >= 1, got {window}")
    if len(returns) < window:
        raise InsufficientDataError(
            f"window {window} exceeds available {len(returns)} return observations"
        )
    return returns[-window:]
