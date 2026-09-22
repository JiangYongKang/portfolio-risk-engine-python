"""Tests for named/composable stress scenarios and factor consistency."""

from __future__ import annotations

import math

import pytest

from app.errors import (
    EmptyPortfolioError,
    ScenarioDefinitionError,
    ScenarioNotFoundError,
)
from app.models import (
    FactorShock,
    Portfolio,
    Position,
    ScenarioDefinition,
    ScenarioType,
)
from app.risk.scenarios import ScenarioEngine
from tests.conftest import log, two_factor_engine


def _pf():
    return Portfolio(
        name="p",
        positions=[Position(symbol="AAA", quantity=1000), Position(symbol="BBB", quantity=2000)],
    )


def test_single_shock_manual_expected():
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="eq10", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="equity", shock=-0.10)]))
    r = engine.evaluate("eq10", _pf())
    # AAA: 1000 * -10% = -100; BBB: 2000 * 0.6 * -10% = -120
    log.info("equity -10%%: pnl=%.2f contributions=%s", r.pnl, r.factor_contributions)
    assert r.pnl == pytest.approx(-220.0)
    assert r.factor_contributions["equity"] == pytest.approx(-220.0)
    assert r.pnl_pct == pytest.approx(-220.0 / 3000.0)


def test_factor_contributions_reconstruct_total():
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="multi", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="equity", shock=-0.12),
                FactorShock(factor="credit", shock=0.05)]))
    r = engine.evaluate("multi", _pf())
    total = sum(r.factor_contributions.values())
    log.info("multi-factor: total=%.6f sum(contribs)=%.6f", r.pnl, total)
    assert math.isclose(total, r.pnl, rel_tol=1e-12, abs_tol=1e-9)
    assert r.stressed_value - r.base_value == pytest.approx(r.pnl)
    log.info("judgement: per-factor perturbations are self-consistent with aggregate")


def test_composite_equals_sum_of_components():
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="eq", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="equity", shock=-0.10)]))
    engine.define(ScenarioDefinition(
        name="cr", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="credit", shock=0.05)]))
    engine.define(ScenarioDefinition(
        name="combo", type=ScenarioType.composite, components=["eq", "cr"]))
    r_eq = engine.evaluate("eq", _pf())
    r_cr = engine.evaluate("cr", _pf())
    r_combo = engine.evaluate("combo", _pf())
    log.info("combo pnl=%.4f vs eq+cr=%.4f", r_combo.pnl, r_eq.pnl + r_cr.pnl)
    assert r_combo.pnl == pytest.approx(r_eq.pnl + r_cr.pnl)


def test_unknown_scenario_rejected():
    engine = two_factor_engine()
    with pytest.raises(ScenarioNotFoundError):
        engine.evaluate("ghost", _pf())
    log.info("undefined scenario rejected reason=scenario_not_found")


def test_unknown_factor_rejected():
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="fx", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="fx_eurusd", shock=-0.10)]))
    log.info("shocked factor fx_eurusd not mapped to any position")
    with pytest.raises(ScenarioDefinitionError) as ei:
        engine.evaluate("fx", _pf())
    assert ei.value.reason == "scenario_definition_invalid"
    log.info("rejected reason=%s", ei.value.reason)


def test_duplicate_factor_in_definition_rejected():
    with pytest.raises(ScenarioDefinitionError):
        ScenarioEngine().define(ScenarioDefinition(
            name="dup", type=ScenarioType.factor_shock,
            shocks=[FactorShock(factor="equity", shock=-0.1),
                    FactorShock(factor="equity", shock=0.1)]))
    log.info("duplicate factor in one scenario rejected")


def test_nested_composite_rejected():
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="eq", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="equity", shock=-0.1)]))
    engine.define(ScenarioDefinition(
        name="c1", type=ScenarioType.composite, components=["eq"]))
    log.info("attempt composite referencing another composite c1")
    with pytest.raises(ScenarioDefinitionError):
        engine.define(ScenarioDefinition(
            name="c2", type=ScenarioType.composite, components=["c1"]))
    log.info("nested composite rejected reason=scenario_definition_invalid")


def test_empty_portfolio_rejected():
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="eq", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="equity", shock=-0.1)]))
    with pytest.raises(EmptyPortfolioError):
        engine.evaluate("eq", Portfolio(name="x", positions=[]))
    log.info("empty portfolio rejected in scenario engine")


def test_extreme_shock_boundary():
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="wipeout", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="equity", shock=-1.0)]))
    r = engine.evaluate("wipeout", Portfolio(
        name="p", positions=[Position(symbol="AAA", quantity=1000)]))
    log.info("-100%% shock: stressed=%.4f pnl=%.4f", r.stressed_value, r.pnl)
    assert r.stressed_value == 0.0
    assert r.pnl_pct == -1.0


def test_identity_mapping_by_default():
    """Symbols without an explicit factor map map 1:1 to a factor named after them."""
    engine = ScenarioEngine()
    engine.define(ScenarioDefinition(
        name="aaa", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="AAA", shock=-0.07)]))
    r = engine.evaluate("aaa", Portfolio(
        name="p", positions=[Position(symbol="AAA", quantity=500)]))
    assert r.pnl == pytest.approx(-35.0)
    log.info("default identity factor mapping AAA->AAA pnl=%.2f", r.pnl)
