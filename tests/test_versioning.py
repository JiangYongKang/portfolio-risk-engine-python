"""Tests for parameter-set versioning and reproducibility policy."""

from __future__ import annotations

import pytest

from app.errors import VersionIncompatibleError, VersionNotFoundError
from app.models import (
    FactorShock,
    Portfolio,
    Position,
    ScenarioDefinition,
    ScenarioType,
    VersionedParameterSet,
)
from app.versioning import ParameterRegistry, ScenarioSnapshot
from tests.conftest import log, two_factor_engine


def _setup(rule_version: int = 1):
    engine = two_factor_engine()
    engine.define(ScenarioDefinition(
        name="eq", type=ScenarioType.factor_shock,
        shocks=[FactorShock(factor="equity", shock=-0.10)]))
    pf = Portfolio(name="p", positions=[Position(symbol="AAA", quantity=1000)])
    reg = ParameterRegistry(engine, current_rule_version=rule_version)
    ps = VersionedParameterSet(
        name="baseline", version=1, rule_version=1, params={"window": 30},
    )
    reg.register(ps, ScenarioSnapshot(scenario_name="eq", portfolio=pf))
    return reg


def test_original_rule_reproduces_identically():
    reg = _setup(rule_version=1)
    a = reg.evaluate_versioned("baseline", 1)
    b = reg.evaluate_versioned("baseline", 1)
    assert a.reproducible is True
    assert a.result == b.result
    log.info("same rule re-evaluation identical; reproducible=%s", a.reproducible)


def test_declared_compatible_new_rule_reproduces():
    reg = _setup(rule_version=2)
    # mark the v1 set compatible with rule generation 2
    ps = reg.get("baseline", 1).model_copy(update={"compatible_with": [2]})
    reg._sets[("baseline", 1)] = ps
    ev = reg.evaluate_versioned("baseline", 1)
    log.info("rule v2 with compatibility declared -> reproducible=%s note=%s",
             ev.reproducible, ev.note)
    assert ev.reproducible is True
    assert ev.result.pnl == -100.0


def test_undeclared_new_rule_is_rejected_then_accepted():
    reg = _setup(rule_version=2)
    log.info("v1 set under rule v2 without compatibility declaration")
    with pytest.raises(VersionIncompatibleError) as ei:
        reg.evaluate_versioned("baseline", 1)
    assert ei.value.reason == "version_incompatible"
    log.info("rejected reason=%s detail=%s", ei.value.reason, ei.value.detail)
    ev = reg.evaluate_versioned("baseline", 1, accept_incompatible=True)
    assert ev.reproducible is False
    assert "incompatibility accepted" in ev.note
    log.info("explicit acceptance -> reproducible=False, note marks rule change")


def test_missing_version_distinguishable():
    reg = _setup()
    with pytest.raises(VersionNotFoundError):
        reg.evaluate_versioned("baseline", 99)
    log.info("unknown version rejected reason=version_not_found")


def test_versions_are_immutable():
    reg = _setup()
    with pytest.raises(VersionIncompatibleError):
        reg.register(VersionedParameterSet(
            name="baseline", version=1, rule_version=1, params={}))
    log.info("re-registering same (name, version) rejected (immutability)")
