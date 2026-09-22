"""Tests for exposure aggregation and historical / parametric tail risk."""

from __future__ import annotations

import math

import pytest

from app.errors import (
    EmptyPortfolioError,
    MatrixDimensionError,
    InsufficientHistoryError,
    InvalidPositionError,
    InvalidWindowError,
    SymbolNotFoundError,
)
from app.models import Portfolio, Position, PriceSeries
from app.risk.metrics import compute_risk
from app.market_data import MarketDataStore
from tests.conftest import log, sample_portfolio, seeded_store


def test_basic_risk_both_families():
    store = seeded_store(n=120)
    pf = sample_portfolio()
    r = compute_risk(pf, store, 60, 0.95)
    log.info(
        "input=2-asset portfolio value=%.2f window=60 c=0.95 -> hVaR=%.4f hES=%.4f pVaR=%.4f pES=%.4f",
        r.portfolio_value, r.historical_var, r.historical_es,
        r.parametric_var, r.parametric_es,
    )
    assert r.portfolio_value > 0
    assert r.observations_used == 60
    assert r.historical_var >= 0 and r.historical_es >= r.historical_var
    assert r.parametric_var >= 0 and r.parametric_es >= r.parametric_var
    log.info("judgement: both ES dominate their VaR; non-negative losses")


def test_deterministic_repeat_runs():
    store = seeded_store(n=80)
    r1 = compute_risk(sample_portfolio(), store, 40, 0.97)
    r2 = compute_risk(sample_portfolio(), store, 40, 0.97)
    assert r1 == r2
    log.info("repeated identical inputs -> byte-identical RiskSummary")


def test_window_exactly_matches_available_history():
    store = seeded_store(n=31)  # 30 returns
    r = compute_risk(sample_portfolio(), store, 30, 0.95)
    assert r.observations_used == 30
    log.info("boundary: window == available returns (30) accepted")


def test_history_one_return_short_is_rejected():
    store = seeded_store(n=31)  # 30 returns
    log.info("input: 30 returns available, window=31 requested")
    with pytest.raises(InsufficientHistoryError) as ei:
        compute_risk(sample_portfolio(), store, 31, 0.95)
    assert ei.value.reason == "insufficient_history"
    log.info("rejected reason=insufficient_history detail=%s", ei.value.detail)


def test_unknown_symbol_is_rejected():
    store = seeded_store(n=50)
    pf = Portfolio(name="x", positions=[Position(symbol="NOPE", quantity=1)])
    log.info("input: position in symbol without price data 'NOPE'")
    with pytest.raises(SymbolNotFoundError) as ei:
        compute_risk(pf, store, 10, 0.95)
    assert ei.value.reason == "symbol_not_found"
    log.info("rejected reason=symbol_not_found")


def test_empty_portfolio_rejected():
    store = seeded_store(n=50)
    with pytest.raises(EmptyPortfolioError):
        compute_risk(Portfolio(name="x", positions=[]), store, 10, 0.95)
    log.info("empty portfolio rejected reason=empty_portfolio")


@pytest.mark.parametrize("window", [0, 1, -5, 2.5])
def test_invalid_window_shapes(window):
    store = seeded_store(n=30)
    log.info("input window=%r (n_prices=30 -> 29 returns)", window)
    with pytest.raises(InvalidWindowError):
        compute_risk(sample_portfolio(), store, window, 0.95)
    log.info("rejected reason=invalid_window")


def test_window_beyond_history_distinguishable():
    store = seeded_store(n=30)  # 29 returns
    log.info("input: structurally valid window=60 but only 29 returns")
    with pytest.raises(InsufficientHistoryError) as ei:
        compute_risk(sample_portfolio(), store, 60, 0.95)
    assert ei.value.reason == "insufficient_history"
    log.info("rejected reason=insufficient_history (distinct from invalid_window)")


@pytest.mark.parametrize("confidence", [0.5, 0.999, 1.0, 0.0])
def test_confidence_boundaries(confidence):
    store = seeded_store(n=120)
    log.info("input confidence=%r", confidence)
    if 0.5 < confidence < 1.0:
        r = compute_risk(sample_portfolio(), store, 60, confidence)
        assert r.confidence == confidence
        log.info("accepted (strictly inside (0.5, 1))")
    else:
        with pytest.raises(InvalidWindowError):
            compute_risk(sample_portfolio(), store, 60, confidence)
        log.info("rejected reason=invalid_window")


def test_extreme_scenario_monotonic_crash():
    """A strictly monotonically falling series: VaR must reflect the crash."""
    store = MarketDataStore()
    prices = [100.0 * (0.97 ** i) for i in range(40)]
    store.load(PriceSeries(symbol="AAA", prices=prices))
    pf = Portfolio(name="crash", positions=[Position(symbol="AAA", quantity=1)])
    r = compute_risk(pf, store, 20, 0.95)
    log.info(
        "monotonic -3%%/period: hVaR/value=%.4f pVaR/value=%.4f",
        r.historical_var / r.portfolio_value,
        r.parametric_var / r.portfolio_value,
    )
    assert r.historical_var / r.portfolio_value > 0.02
    log.info("judgement: historical VaR exceeds one-period -2%% move")


def test_zero_variance_series_parametric_var_is_zero():
    store = MarketDataStore()
    store.load(PriceSeries(symbol="AAA", prices=[100.0] * 30))
    pf = Portfolio(name="flat", positions=[Position(symbol="AAA", quantity=1)])
    r = compute_risk(pf, store, 20, 0.95)
    log.info("flat prices -> hVaR=%.6g pVaR=%.6g pES=%.6g",
             r.historical_var, r.parametric_var, r.parametric_es)
    assert r.historical_var == 0.0
    assert r.parametric_var == 0.0
    assert r.parametric_es == 0.0
    log.info("judgement: zero-variance inputs yield zero tail risk, no NaN")


def test_zero_quantity_position_rejected():
    store = seeded_store(n=50)
    pf = Portfolio(name="x", positions=[Position(symbol="AAA", quantity=0)])
    with pytest.raises(InvalidPositionError):
        compute_risk(pf, store, 10, 0.95)
    log.info("zero-quantity position rejected reason=invalid_position")


def test_correlation_override_changes_parametric_not_historical():
    # Two identical-volatility, perfectly co-moving symbols: empirical corr == 1.
    store = MarketDataStore()
    prices = [100.0 * (1.0 + 0.01 * ((i * 37) % 7 - 3)) for i in range(100)]
    store.load(PriceSeries(symbol="AAA", prices=prices))
    store.load(PriceSeries(symbol="BBB", prices=prices))
    pf = Portfolio(name="p", positions=[
        Position(symbol="AAA", quantity=50), Position(symbol="BBB", quantity=50)])
    base = compute_risk(pf, store, 60, 0.95)
    zero_corr = compute_risk(pf, store, 60, 0.95, [[1.0, 0.0], [0.0, 1.0]])
    log.info(
        "empirical corr=1: pVaR empirical=%.4f pVaR override-zero=%.4f | "
        "hVaR base=%.4f override=%.4f",
        base.parametric_var, zero_corr.parametric_var,
        base.historical_var, zero_corr.historical_var,
    )
    assert zero_corr.historical_var == base.historical_var
    assert zero_corr.parametric_var < base.parametric_var
    # VaR = z*sigma - mu; both runs share mu, so compare the sigma terms only.
    # Independent equal-vol assets halve variance vs perfectly correlated:
    # (V_zero + muV)/(V_base + muV) should equal 1/sqrt(2).
    ratio = zero_corr.parametric_var / base.parametric_var
    log.info("VaR ratio zero/perfect=%.6f (1/sqrt(2)=%.6f)",
             ratio, 1.0 / math.sqrt(2.0))
    assert ratio == pytest.approx(1.0 / math.sqrt(2.0), abs=6e-3)
    log.info("judgement: override affects only parametric VaR; historical replay fixed")


def test_correlation_override_dimension_mismatch_rejected():
    store = seeded_store(n=100)
    pf = sample_portfolio()  # 2 distinct symbols
    log.info("correlation override 3x3 supplied for 2-symbol portfolio")
    with pytest.raises(MatrixDimensionError) as ei:
        compute_risk(pf, store, 60, 0.95,
                     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert ei.value.reason == "matrix_dimension_mismatch"
    log.info("rejected reason=%s", ei.value.reason)


def test_correlation_override_non_pd_rejected():
    from app.errors import MatrixNotPositiveDefiniteError

    store = seeded_store(n=100)
    with pytest.raises(MatrixNotPositiveDefiniteError):
        compute_risk(sample_portfolio(), store, 60, 0.95,
                     [[1.0, 1.0], [1.0, 1.0]])
    log.info("singular correlation override rejected matrix_not_positive_definite")
