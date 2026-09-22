"""Exposure aggregation and tail-risk metrics (historical + parametric)."""

from __future__ import annotations

import math
from datetime import date

from risk_engine.correlation import build_covariance
from risk_engine.errors import (
    ComputationError,
    InsufficientDataError,
    InvalidParameterError,
    InvalidWindowError,
    RiskEngineError,
)
from risk_engine.models import (
    CorrelationAssumption,
    MarketData,
    Portfolio,
    RiskReport,
    TailRiskEstimate,
)
from risk_engine.series import (
    aligned_price_matrix,
    simple_returns,
    window_slice,
    Matrix,
    Vector,
)

Z_99 = 2.3263478740408408


def _erf_inv(x: float) -> float:
    # Peter Acklam style inverse normal CDF via inverse erf is fiddly;
    # use a small Newton inversion of erf for the single value needed.
    import math as _m

    # Winitzki approximation then Newton polish
    if abs(x) >= 1.0:
        return math.copysign(8.0, x)
    a = 0.147
    log = _m.log(1 - x * x)
    term = 2 / (_m.pi * a) + log / 2
    y = _m.copysign(_m.sqrt(_m.sqrt(term * term - log / a) - term), x)
    for _ in range(6):
        ey = _m.erf(y)
        der = (2 / _m.sqrt(_m.pi)) * _m.exp(-y * y)
        if der == 0:
            break
        y -= (ey - x) / der
    return y


def z_for_confidence(confidence: float) -> float:
    """One-sided standard normal quantile (loss) for given confidence."""
    if not (0.5 < confidence < 1.0):
        raise InvalidParameterError(
            f"confidence must be within (0.5, 1.0), got {confidence}"
        )
    return math.sqrt(2.0) * _erf_inv(2 * confidence - 1)


def exposure_breakdown(
    portfolio: Portfolio, market: MarketData
) -> tuple[float, float, dict[str, float]]:
    """(gross, net, per-instrument) notional exposures at last prices."""
    if not portfolio.holdings:
        raise InsufficientDataError("portfolio has no holdings")
    if len(market.dates) == 0:
        raise InsufficientDataError("market data has no dates")
    last_prices = {inst: float(series[-1]) for inst, series in market.prices.items()}
    per: dict[str, float] = {}
    gross = 0.0
    net = 0.0
    for h in portfolio.holdings:
        price = last_prices.get(h.instrument)
        if price is None:
            from risk_engine.errors import UnknownInstrumentError

            raise UnknownInstrumentError(
                f"no market data for instrument '{h.instrument}'"
            )
        notional = h.quantity * price
        per[h.instrument] = per.get(h.instrument, 0.0) + notional
        gross += abs(notional)
        net += notional
    return gross, net, per


def _quantile_loss(losses: list[float], confidence: float) -> float:
    """Historical quantile using the deterministic 'lower rank' convention.

    VaR is the quantile of the empirical loss distribution.  We use the
    lower-rank definition: index = ceil(confidence * n) - 1 after ascending
    sort.  This is fully deterministic and independent of numpy/interp.
    """
    ordered = sorted(losses)
    n = len(ordered)
    idx = math.ceil(confidence * n) - 1
    idx = min(max(idx, 0), n - 1)
    return ordered[idx]


def _amount_pnl_series(
    portfolio: Portfolio,
    instruments: tuple[str, ...],
    prices: Matrix,
    asset_returns: Matrix,
) -> tuple[list[float], list[float]]:
    """Per-period portfolio PnL amounts and per-instrument weights."""
    last_prices = [prices[-1][j] for j in range(len(instruments))]
    qty = {h.instrument: 0.0 for h in portfolio.holdings}
    for h in portfolio.holdings:
        qty[h.instrument] += h.quantity
    amounts = [qty[inst] * last_prices[j] for j, inst in enumerate(instruments)]
    pnl = [
        sum(amounts[j] * asset_returns[t][j] for j in range(len(instruments)))
        for t in range(len(asset_returns))
    ]
    return pnl, amounts


def historical_tail(
    portfolio: Portfolio,
    instruments: tuple[str, ...],
    prices: Matrix,
    asset_returns: Matrix,
    confidence: float,
    horizon_days: int,
) -> TailRiskEstimate:
    if not asset_returns:
        raise InsufficientDataError("no return observations for historical VaR")
    if horizon_days < 1:
        raise InvalidWindowError(f"horizon_days must be >= 1, got {horizon_days}")
    pnl, _ = _amount_pnl_series(portfolio, instruments, prices, asset_returns)
    losses = [-v for v in pnl]
    scale = math.sqrt(horizon_days)  # deterministic horizon scaling
    var1 = _quantile_loss(losses, confidence)
    # ES = mean of losses at or beyond VaR threshold
    tail = [v for v in losses if v >= var1 - 1e-15]
    es1 = sum(tail) / len(tail)
    return TailRiskEstimate(
        confidence=confidence,
        horizon_days=horizon_days,
        var=var1 * scale,
        expected_shortfall=es1 * scale,
    )


def parametric_tail(
    portfolio: Portfolio,
    instruments: tuple[str, ...],
    prices: Matrix,
    asset_returns: Matrix,
    confidence: float,
    horizon_days: int,
    correlation: CorrelationAssumption | None,
) -> tuple[TailRiskEstimate, float]:
    if horizon_days < 1:
        raise InvalidWindowError(f"horizon_days must be >= 1, got {horizon_days}")
    n_obs = len(asset_returns)
    if n_obs < 2:
        raise InsufficientDataError(
            f"need >= 2 return observations for parametric VaR, got {n_obs}"
        )
    _, amounts = _amount_pnl_series(
        portfolio, instruments, prices, asset_returns
    )

    n_assets = len(instruments)
    means = [0.0] * n_assets
    for row in asset_returns:
        for j in range(n_assets):
            means[j] += row[j] / n_obs
    variances = [0.0] * n_assets
    for row in asset_returns:
        for j in range(n_assets):
            d = row[j] - means[j]
            variances[j] += d * d / (n_obs - 1)
    vols = [math.sqrt(v) for v in variances]

    try:
        if correlation is None:
            port_var = sum(
                (amounts[j] * vols[j]) ** 2 for j in range(n_assets)
            )
        else:
            cov = build_covariance(vols, correlation, instruments)
            port_var = 0.0
            for a in range(n_assets):
                for b in range(n_assets):
                    port_var += amounts[a] * amounts[b] * cov[a][b]
        if port_var < 0 or not math.isfinite(port_var):
            raise ComputationError(
                f"portfolio variance invalid: {port_var}"
            )
        sigma1 = math.sqrt(port_var)
        z = z_for_confidence(confidence)
        scale = math.sqrt(horizon_days)
        var = z * sigma1 * scale
        es = (
            math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / (1 - confidence)
        ) * sigma1 * scale
    except ComputationError:
        raise
    except RiskEngineError:
        # typed correlation/window errors must propagate with their own code
        raise
    except Exception as exc:  # never leak raw numeric errors
        raise ComputationError(f"parametric risk computation failed: {exc}") from exc

    return (
        TailRiskEstimate(
            confidence=confidence,
            horizon_days=horizon_days,
            var=var,
            expected_shortfall=es,
        ),
        sigma1,
    )


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
    if window < 1:
        raise InvalidWindowError(f"window must be >= 1, got {window}")
    if not (0.5 < confidence < 1.0):
        raise InvalidParameterError(
            f"confidence must be within (0.5, 1.0), got {confidence}"
        )
    dates, instruments, prices = aligned_price_matrix(portfolio, market)
    all_returns = simple_returns(prices)
    returns = window_slice(all_returns, window)
    # prices aligned with the chosen window (window returns -> window+1 prices)
    window_prices = prices[-(window + 1):]

    gross, net, per = exposure_breakdown(portfolio, market)
    hist = historical_tail(
        portfolio, instruments, window_prices, returns,
        confidence, horizon_days,
    )
    param, sigma1 = parametric_tail(
        portfolio, instruments, window_prices, returns,
        confidence, horizon_days, correlation,
    )

    return RiskReport(
        portfolio=portfolio.name,
        version=version,
        window=window,
        as_of=dates[-1],
        historical=hist,
        parametric=param,
        portfolio_volatility=sigma1,
        gross_exposure=gross,
        net_exposure=net,
        exposure_by_instrument=per,
    )
