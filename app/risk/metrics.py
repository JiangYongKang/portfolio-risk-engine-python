"""Historical-simulation and parametric tail-risk metrics.

Conventions (applied identically by every code path so that serial and
batch evaluations are bit-for-bit consistent):

* ``confidence`` is the *loss* quantile, e.g. 0.95 reports the loss level
  exceeded only 5% of the time; reported VaR is a non-negative loss number.
* Historical simulation uses the window's empirical loss distribution:
  ``VaR_h`` is its ``confidence`` quantile (linear interpolation, numpy
  default), ``ES_h`` the mean of losses at or above that quantile.
* Parametric VaR assumes jointly normal factor returns:
  ``VaR_p = z * sigma_p - mu_p`` (drift adjusted), ``ES_p`` uses the
  normal tail expectation ``sigma_p * phi(z) / (1 - confidence) - mu_p``.
"""

from __future__ import annotations

import math

import numpy as np

from app.errors import (
    EmptyPortfolioError,
    InsufficientHistoryError,
    InvalidPositionError,
    InvalidWindowError,
    MatrixDimensionError,
    SymbolNotFoundError,
)
from app.market_data import MarketDataStore
from app.models import Portfolio, RiskSummary
from app.risk.covariance import correlation_to_covariance, validate_matrix


def simple_returns(prices) -> np.ndarray:
    """Period-over-period simple returns; requires >= 2 finite prices."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.shape[0] < 2:
        raise InvalidWindowError("need at least 2 prices to form a return")
    return p[1:] / p[:-1] - 1.0


def _portfolio_weights(portfolio: Portfolio, latest: dict[str, float]):
    if not portfolio.positions:
        raise EmptyPortfolioError("portfolio contains no positions")
    values: dict[str, float] = {}
    for pos in portfolio.positions:
        if not math.isfinite(pos.quantity) or pos.quantity == 0.0:
            raise InvalidPositionError(
                f"position in {pos.symbol!r} must be finite and non-zero"
            )
        if pos.symbol not in latest:
            raise SymbolNotFoundError(f"no market data for symbol {pos.symbol!r}")
        values[pos.symbol] = values.get(pos.symbol, 0.0) + pos.quantity * latest[pos.symbol]
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise InvalidPositionError(
            "portfolio market value must be finite and positive "
            "(netting short positions is not supported for risk aggregation)"
        )
    return values, total


def historical_tail(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return ``(VaR, ES)`` from an empirical loss distribution."""
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var - 1e-15]
    es = float(np.mean(tail))
    return max(var, 0.0), max(es, 0.0)


def parametric_tail(
    mu: np.ndarray, cov: np.ndarray, weights: np.ndarray, confidence: float
) -> tuple[float, float]:
    """Return drift-adjusted normal ``(VaR, ES)`` for the weighted portfolio."""
    z = _zscore(confidence)
    mu_p = float(weights @ mu)
    var_p = float(z * np.sqrt(weights @ cov @ weights) - mu_p)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    es_p = float(np.sqrt(weights @ cov @ weights) * phi / (1.0 - confidence) - mu_p)
    return max(var_p, 0.0), max(es_p, 0.0)


def _zscore(confidence: float) -> float:
    # inverse standard-normal CDF (loss tail = 1 - confidence)
    return _ndtri(confidence)


def _ndtri(p: float) -> float:
    # Acklam-style inverse normal CDF, deterministic and dependency-free.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def compute_risk(
    portfolio: Portfolio,
    store: MarketDataStore,
    window: int,
    confidence: float,
    correlation_override: list[list[float]] | None = None,
) -> RiskSummary:
    """Aggregate exposures and compute both tail-risk families.

    When ``correlation_override`` is supplied it replaces the empirical
    correlations used by the **parametric** family (volatilities are kept
    from the window); the matrix is strictly validated first.  Historical
    simulation is unaffected because it replays actual joint history.
    """
    if not isinstance(window, int) or window < 2:
        raise InvalidWindowError(f"window must be an integer >= 2, got {window!r}")
    if not (0.5 < confidence < 1.0):
        raise InvalidWindowError(f"confidence must lie in (0.5, 1), got {confidence}")

    symbols = [p.symbol for p in portfolio.positions]
    for sym in symbols:
        if not store.has_symbol(sym):
            raise SymbolNotFoundError(f"no market data for symbol {sym!r}")

    latest = {sym: store.prices(sym)[-1] for sym in set(symbols)}
    values, total_value = _portfolio_weights(portfolio, latest)

    aligned: list[np.ndarray] = []
    for sym in values:
        prices = store.prices(sym)
        if len(prices) - 1 < window:
            raise InsufficientHistoryError(
                f"symbol {sym!r} has {len(prices) - 1} return observations; "
                f"window requires {window}"
            )
        r = simple_returns(prices[-window - 1:])
        aligned.append(r)
    rets = np.column_stack(aligned)  # rows = dates, cols = symbols (sorted set)
    syms = list(values)
    w = np.array([values[s] / total_value for s in syms], dtype=float)

    # Historical simulation: P&L of weighted basket per historical date.
    portfolio_r = rets @ w
    losses = -portfolio_r
    h_var, h_es = historical_tail(losses, confidence)

    # Parametric: sample mean / covariance of the same window.
    mu = rets.mean(axis=0)
    cov = np.cov(rets, rowvar=False, ddof=1)
    if cov.ndim == 0:  # single asset
        cov = cov.reshape(1, 1)
    if correlation_override is not None:
        n = len(syms)
        corr = validate_matrix(correlation_override, syms)
        if corr.shape != (n, n):
            raise MatrixDimensionError(
                f"correlation override is {corr.shape[0]}x{corr.shape[0]}, "
                f"need {n}x{n} for portfolio symbols {syms}"
            )
        vols = np.sqrt(np.diag(cov))
        cov = correlation_to_covariance(corr, vols)
    p_var, p_es = parametric_tail(mu, cov, w, confidence)

    return RiskSummary(
        portfolio=portfolio.name,
        window=window,
        observations_used=window,
        confidence=confidence,
        portfolio_value=float(total_value),
        historical_var=float(h_var * total_value),
        historical_es=float(h_es * total_value),
        parametric_var=float(p_var * total_value),
        parametric_es=float(p_es * total_value),
    )
