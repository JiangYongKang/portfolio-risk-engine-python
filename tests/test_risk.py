"""Risk metrics: data rejection, determinism, reproducibility, edge cases."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from risk_engine.errors import (
    InsufficientDataError,
    InvalidParameterError,
    InvalidWindowError,
    UnknownInstrumentError,
)
from risk_engine.models import (
    CorrelationAssumption,
    Holding,
    MarketData,
    Portfolio,
)
from risk_engine.risk import build_report, z_for_confidence

from .conftest import log


def test_unknown_instrument_rejected(market, portfolio):
    p = Portfolio("bad", (Holding("ZZ", 1.0),))
    log.info("case=unknown-instrument holdings=['ZZ']; criterion: UNKNOWN_INSTRUMENT")
    with pytest.raises(UnknownInstrumentError):
        build_report(p, market, window=10)


def test_window_zero_rejected(market, portfolio):
    log.info("case=window-zero window=0; criterion: INVALID_WINDOW")
    with pytest.raises(InvalidWindowError):
        build_report(portfolio, market, window=0)


def test_window_negative_rejected(market, portfolio):
    log.info("case=window-negative window=-3; criterion: INVALID_WINDOW")
    with pytest.raises(InvalidWindowError):
        build_report(portfolio, market, window=-3)


def test_window_exceeds_history_rejected(market, portfolio):
    log.info("case=window-too-big window=500 returns=59; criterion: INSUFFICIENT_DATA")
    with pytest.raises(InsufficientDataError):
        build_report(portfolio, market, window=500)


def test_too_few_prices_rejected():
    dates = (date(2026, 1, 1),)
    m = MarketData(dates, {"A": [100.0], "B": [50.0], "C": [70.0]})
    p = Portfolio("p", (Holding("A", 1.0), Holding("B", 1.0), Holding("C", 1.0)))
    log.info("case=too-few-prices n_dates=1; criterion: INSUFFICIENT_DATA")
    with pytest.raises(InsufficientDataError):
        build_report(p, m, window=1)


def test_ragged_series_rejected(market, portfolio):
    m = MarketData(market.dates, {"A": list(market.prices["A"])[:-1],
                                  "B": list(market.prices["B"]),
                                  "C": list(market.prices["C"])})
    log.info("case=ragged-series A=59 B=60 C=60; criterion: INSUFFICIENT_DATA")
    with pytest.raises(InsufficientDataError):
        build_report(portfolio, m, window=10)


def test_non_positive_price_rejected(market, portfolio):
    prices = {k: list(v) for k, v in market.prices.items()}
    prices["A"][5] = 0.0
    m = MarketData(market.dates, prices)
    log.info("case=zero-price; criterion: INSUFFICIENT_DATA")
    with pytest.raises(InsufficientDataError):
        build_report(portfolio, m, window=10)


def test_results_are_deterministic(market, portfolio):
    r1 = build_report(portfolio, market, window=20)
    r2 = build_report(portfolio, market, window=20)
    log.info("case=determinism; criterion: repeated calls bit-identical")
    assert r1 == r2


def test_historical_and_parametric_both_present(market, portfolio):
    r = build_report(portfolio, market, window=20, confidence=0.99)
    log.info(
        "case=both-tail-methods hist_VaR=%s param_VaR=%s",
        r.historical.var, r.parametric.var,
    )
    assert math.isfinite(r.historical.var)
    assert math.isfinite(r.parametric.var)
    assert r.historical.confidence == 0.99
    assert r.parametric.confidence == 0.99


def test_es_is_at_least_var_parametric(market, portfolio):
    r = build_report(portfolio, market, window=30, confidence=0.99)
    log.info("case=es>=var param VaR=%s ES=%s; criterion: ES >= VaR",
             r.parametric.var, r.parametric.expected_shortfall)
    assert r.parametric.expected_shortfall >= r.parametric.var


def test_horizon_scales_by_sqrt_t(market, portfolio):
    r1 = build_report(portfolio, market, window=30, horizon_days=1)
    r4 = build_report(portfolio, market, window=30, horizon_days=4)
    log.info("case=sqrt-time h=1 vs 4; criterion: x2")
    assert r4.parametric.var == pytest.approx(2 * r1.parametric.var)
    assert r4.historical.var == pytest.approx(2 * r1.historical.var)


def test_correlation_changes_parametric_not_historical(market, portfolio):
    base = build_report(portfolio, market, window=30)
    corr = CorrelationAssumption(
        "hi", ("A", "B", "C"),
        ((1.0, 0.3, 0.2), (0.3, 1.0, 0.1), (0.2, 0.1, 1.0)),
    )
    with_corr = build_report(portfolio, market, window=30, correlation=corr)
    log.info(
        "case=corr-separation hist unchanged=%s param changed=%s",
        base.historical.var == with_corr.historical.var,
        base.parametric.var != with_corr.parametric.var,
    )
    assert with_corr.historical.var == base.historical.var
    assert with_corr.parametric.var != base.parametric.var


def test_confidence_quantiles_are_monotonic(market, portfolio):
    r95 = build_report(portfolio, market, window=40, confidence=0.95)
    r99 = build_report(portfolio, market, window=40, confidence=0.99)
    log.info("case=quantile-monotonic VaR95=%s <= VaR99=%s",
             r95.parametric.var, r99.parametric.var)
    assert r99.parametric.var >= r95.parametric.var


def test_z_quantile_matches_normal_table():
    log.info("case=z-table z99; criterion: |error| < 1e-9")
    assert z_for_confidence(0.99) == pytest.approx(2.3263478740, abs=1e-9)
    with pytest.raises(InvalidParameterError):
        z_for_confidence(0.5)
