"""Stress scenarios: extreme/boundary moves, composition, self-consistency."""

from __future__ import annotations

import pytest

from risk_engine.errors import UnknownInstrumentError
from risk_engine.models import FactorShock, StressScenario
from risk_engine.scenarios import (
    compose_scenarios,
    evaluate_scenario,
    single_factor_perturbation,
)

from .conftest import log


def _sc(name, *pairs):
    return StressScenario(name, tuple(FactorShock(i, m) for i, m in pairs))


def test_single_factor_matches_scenario(portfolio, market):
    move = -0.10
    single = single_factor_perturbation(portfolio, market, "A", move)
    result = evaluate_scenario(portfolio, market, _sc("s", ("A", move)))
    log.info("case=single-vs-scenario single=%s scenario=%s", single, result.pnl)
    assert result.pnl == single
    assert result.consistent_with_single_factor


def test_extreme_total_loss(portfolio, market):
    result = evaluate_scenario(portfolio, market, _sc("wipeout", ("A", -1.0), ("B", -1.0), ("C", -1.0)))
    gross = sum(abs(v) for v in result.pnl_by_instrument.values())
    log.info("case=total-loss pnl=%s gross=%s", result.pnl, gross)
    # every position fully written off
    for inst, v in result.pnl_by_instrument.items():
        qty = sum(h.quantity for h in portfolio.holdings if h.instrument == inst)
        price = market.prices[inst][-1]
        assert v == pytest.approx(-qty * price)


def test_extreme_rally(portfolio, market):
    result = evaluate_scenario(portfolio, market, _sc("rally", ("A", 2.0)))
    qty = 10.0
    price = market.prices["A"][-1]
    log.info("case=rally +200%% pnl=%s", result.pnl)
    assert result.pnl == pytest.approx(qty * price * 2.0)


def test_zero_move_is_zero_pnl(portfolio, market):
    result = evaluate_scenario(portfolio, market, _sc("flat", ("A", 0.0), ("B", 0.0)))
    log.info("case=zero-move pnl=%s", result.pnl)
    assert result.pnl == 0.0


def test_short_position_sign(portfolio, market):
    # B is held short (-20); a -5% drop is a gain.
    result = evaluate_scenario(portfolio, market, _sc("short_gain", ("B", -0.05)))
    price = market.prices["B"][-1]
    log.info("case=short-sign pnl=%s expect=%s", result.pnl, -20 * price * -0.05)
    assert result.pnl == pytest.approx(-20 * price * -0.05)


def test_composition_is_exactly_additive(portfolio, market):
    s1 = _sc("eq", ("A", -0.10), ("B", -0.05))
    s2 = _sc("credit", ("C", -0.20), ("A", -0.05))
    combo = compose_scenarios("combo", s1, s2)
    r1 = evaluate_scenario(portfolio, market, s1)
    r2 = evaluate_scenario(portfolio, market, s2)
    rc = evaluate_scenario(portfolio, market, combo)
    log.info("case=compose-additive combo=%s parts=%s", rc.pnl, r1.pnl + r2.pnl)
    assert rc.pnl == r1.pnl + r2.pnl
    assert rc.consistent_with_single_factor


def test_composition_order_does_not_change_pnl(portfolio, market):
    s1 = _sc("a", ("A", -0.10), ("B", -0.05))
    s2 = _sc("b", ("C", -0.20))
    c1 = evaluate_scenario(portfolio, market, compose_scenarios("x", s1, s2))
    c2 = evaluate_scenario(portfolio, market, compose_scenarios("y", s2, s1))
    log.info("case=compose-order c1=%s c2=%s", c1.pnl, c2.pnl)
    assert c1.pnl == c2.pnl


def test_unknown_shocked_instrument_rejected(portfolio, market):
    log.info("case=unknown-shock instrument=ZZ; criterion: UNKNOWN_INSTRUMENT")
    with pytest.raises(UnknownInstrumentError):
        evaluate_scenario(portfolio, market, _sc("bad", ("ZZ", -0.1)))


def test_shock_to_unheld_instrument_rejected(portfolio, market):
    # 'D' exists in market but not in portfolio
    from datetime import date, timedelta
    from risk_engine.models import MarketData
    m = MarketData(market.dates, {**market.prices, "D": [10.0] * len(market.dates)})
    log.info("case=unheld-shock instrument=D; criterion: UNKNOWN_INSTRUMENT")
    with pytest.raises(UnknownInstrumentError):
        evaluate_scenario(portfolio, m, _sc("bad", ("D", -0.1)))


def test_factor_moves_aggregated_for_display(portfolio, market):
    sc = _sc("dup", ("A", -0.1), ("A", -0.2))
    result = evaluate_scenario(portfolio, market, sc)
    log.info("case=dup-factor moves=%s", result.factor_moves)
    assert result.factor_moves["A"] == pytest.approx(-0.3)
    assert result.consistent_with_single_factor
