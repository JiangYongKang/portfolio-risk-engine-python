"""Cross-version batch recomparison: REPRODUCED / CHANGED / REJECTED,
order stability, per-item isolation, explicit breaking opt-in."""

from __future__ import annotations

import copy

import pytest

from risk_engine.errors import IncompatibleVersionError
from risk_engine.recompare import (
    CHANGED,
    REJECTED,
    REPRODUCED,
    HistoricalConclusion,
    RecompareLimits,
    conclusions_equal,
)
from risk_engine.service import RiskService, report_to_dict
from risk_engine.versions import RulesVersion

from .conftest import log


def _svc() -> RiskService:
    svc = RiskService()
    # v2 declares v1 reproducible (same rule)
    svc.register_version(
        RulesVersion("v2", "2026-06-01", "v1", frozenset({"v1"}), "same rule")
    )
    # v3 does NOT declare v1/v2 reproducible (new convention)
    svc.register_version(
        RulesVersion("v3", "2026-09-01", "v2", frozenset(), "new ES convention")
    )
    return svc


def _hc(svc, portfolio, market, version="v1", **overrides) -> HistoricalConclusion:
    report = svc.assess(portfolio, market, window=20, version=version)
    payload = report_to_dict(report)
    payload.update(overrides)
    return HistoricalConclusion(
        portfolio=portfolio,
        market=market,
        stamped_version=version,
        conclusion=payload,
        window=20,
    )


def test_reproduced_under_compatible_version(portfolio, market):
    svc = _svc()
    items = [_hc(svc, portfolio, market)]
    out = svc.recompare_conclusions(items, target_version="v2")
    log.info("case=reproduced v1->v2 verdict=%s", out[0].verdict)
    assert len(out) == 1
    assert out[0].verdict == REPRODUCED
    assert out[0].new_conclusion["version"] == "v2"
    assert conclusions_equal(items[0].conclusion, out[0].new_conclusion)


def test_rejected_when_breaking_not_approved(portfolio, market):
    svc = _svc()
    items = [_hc(svc, portfolio, market)]
    out = svc.recompare_conclusions(items, target_version="v3")
    log.info("case=rejected v1->v3 no-optin verdict=%s err=%s",
             out[0].verdict, out[0].error and out[0].error["code"])
    assert out[0].verdict == REJECTED
    assert out[0].error["code"] == "INCOMPATIBLE_VERSION"
    # old conclusion retained untouched, nothing new written
    assert out[0].new_conclusion is None
    assert out[0].old_conclusion is items[0].conclusion


def test_changed_after_explicit_approval(portfolio, market):
    svc = _svc()
    # tamper the historical numbers to simulate a genuinely different
    # conclusion under the new convention
    items = [_hc(svc, portfolio, market)]
    old = copy.deepcopy(items[0].conclusion)
    old["parametric"]["var"] = old["parametric"]["var"] + 123.0
    items = [HistoricalConclusion(
        portfolio=portfolio, market=market, stamped_version="v1",
        conclusion=old, window=20,
    )]
    out = svc.recompare_conclusions(
        items, target_version="v3", allow_breaking=True
    )
    log.info("case=changed v1->v3 approved verdict=%s", out[0].verdict)
    assert out[0].verdict == CHANGED
    assert out[0].old_conclusion["parametric"]["var"] != \
        out[0].new_conclusion["parametric"]["var"]
    assert out[0].new_conclusion["version"] == "v3"
    # input historical conclusion object is never mutated
    assert items[0].conclusion["parametric"]["var"] == old["parametric"]["var"]


def test_mixed_batch_isolated_and_ordered(portfolio, market):
    svc = _svc()
    reproduced = _hc(svc, portfolio, market)
    failing_ref = reproduced
    from risk_engine.models import Holding, Portfolio
    boom_portfolio = Portfolio("boom", (Holding("ZZ", 1.0),))
    failing = HistoricalConclusion(
        portfolio=boom_portfolio, market=market, stamped_version="v1",
        conclusion=failing_ref.conclusion, window=20,
    )
    changed_old = copy.deepcopy(reproduced.conclusion)
    changed_old["historical"]["var"] += 50.0
    changed = HistoricalConclusion(
        portfolio=portfolio, market=market, stamped_version="v1",
        conclusion=changed_old, window=20,
    )
    items = [reproduced, reproduced, failing, changed]
    # approval given for the whole batch, but verdicts still depend on
    # whether numbers actually move
    out = svc.recompare_conclusions(
        items, target_version="v3", allow_breaking=True,
        limits=RecompareLimits(max_items=16, timeout_seconds=10),
    )
    verdicts = [(r.index, r.portfolio, r.verdict) for r in out]
    log.info("case=mixed %s", verdicts)
    assert [r.index for r in out] == [0, 1, 2, 3]
    # breaking approved, but numbers are bit-identical -> still REPRODUCED
    assert out[0].verdict == REPRODUCED
    assert out[1].verdict == REPRODUCED
    # item 2 fails on unknown instrument -> REJECTED, siblings unaffected
    assert out[2].verdict == REJECTED
    assert out[2].error["code"] == "UNKNOWN_INSTRUMENT"
    assert out[3].verdict == CHANGED
    assert [r.portfolio for r in out] == ["fund-x", "fund-x", "boom", "fund-x"]


def test_rejected_does_not_require_approval_for_one_bad_item(portfolio, market):
    # one rejected item (breaking not approved) must not change the verdict
    # of an adjacent reproducible item, and order must hold.
    svc = _svc()
    ok = _hc(svc, portfolio, market)
    items = [ok, ok, ok]
    out = svc.recompare_conclusions(items, target_version="v3")
    log.info("case=batch-rejected-neighbours verdicts=%s", [r.verdict for r in out])
    assert [r.verdict for r in out] == [REJECTED, REJECTED, REJECTED]
    out2 = svc.recompare_conclusions([ok, ok], target_version="v2")
    assert [r.verdict for r in out2] == [REPRODUCED, REPRODUCED]


def test_capacity_limit_rejects_whole_recompare_batch(portfolio, market):
    from risk_engine.errors import BatchSizeLimitError

    svc = _svc()
    items = [_hc(svc, portfolio, market) for _ in range(5)]
    log.info("case=recompare-capacity n=5 max=3")
    with pytest.raises(BatchSizeLimitError):
        svc.recompare_conclusions(
            items, target_version="v2",
            limits=RecompareLimits(max_items=3),
        )
