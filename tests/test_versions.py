"""Versioned parameter sets: reproducibility and compatibility gates."""

from __future__ import annotations

import pytest

from risk_engine.errors import IncompatibleVersionError, UnknownVersionError
from risk_engine.service import RiskService
from risk_engine.versions import RulesVersion

from .conftest import log


def _svc() -> RiskService:
    svc = RiskService()
    svc.register_version(
        RulesVersion("v2", "2026-06-01", "v1", frozenset({"v1"}), "same quantile rule")
    )
    svc.register_version(
        RulesVersion("v3", "2026-09-01", "v2", frozenset(), "new ES convention")
    )
    return svc


def test_same_version_reproduces(portfolio, market):
    svc = _svc()
    r1 = svc.assess(portfolio, market, window=20, version="v1")
    r2 = svc.reassess_under_version(
        portfolio, market, stamped_version="v1", target_version="v1", window=20
    )
    log.info("case=same-version; criterion: identical conclusions")
    assert r1.historical.var == r2.historical.var
    assert r1.parametric.var == r2.parametric.var


def test_compatible_new_version_reproduces(portfolio, market):
    svc = _svc()
    old = svc.assess(portfolio, market, window=20, version="v1")
    new = svc.reassess_under_version(
        portfolio, market, stamped_version="v1", target_version="v2", window=20
    )
    log.info("case=compatible-upgrade v1->v2; criterion: identical, re-stamped")
    assert new.version == "v2"
    assert new.historical.var == old.historical.var
    assert new.parametric.var == old.parametric.var


def test_incompatible_version_refused(portfolio, market):
    svc = _svc()
    log.info("case=incompatible v1->v3; criterion: INCOMPATIBLE_VERSION")
    with pytest.raises(IncompatibleVersionError) as ei:
        svc.reassess_under_version(
            portfolio, market, stamped_version="v1", target_version="v3", window=20
        )
    assert ei.value.code == "INCOMPATIBLE_VERSION"


def test_breaking_change_requires_explicit_opt_in(portfolio, market):
    svc = _svc()
    r = svc.reassess_under_version(
        portfolio, market, stamped_version="v1", target_version="v3",
        window=20, allow_breaking=True,
    )
    log.info("case=explicit-breaking; criterion: allowed, re-stamped v3")
    assert r.version == "v3"


def test_unknown_version_rejected(portfolio, market):
    svc = _svc()
    log.info("case=unknown-version v9; criterion: UNKNOWN_VERSION")
    with pytest.raises(UnknownVersionError):
        svc.assess(portfolio, market, window=20, version="v9")


def test_duplicate_registration_rejected():
    svc = _svc()
    with pytest.raises(ValueError):
        svc.register_version(
            RulesVersion("v2", "2026-07-01", "v1", frozenset({"v1"}))
        )


def test_scenario_versions_are_stamped(portfolio, market):
    from risk_engine.models import FactorShock, StressScenario
    svc = _svc()
    sc = StressScenario("eq", (FactorShock("A", -0.1),), version="v2")
    result = svc.run_scenario(portfolio, market, sc)
    log.info("case=scenario-stamp version=%s", result.version)
    assert result.version == "v2"
