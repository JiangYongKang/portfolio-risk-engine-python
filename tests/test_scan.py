"""Batch scan: portfolios x registered scenarios in one ordered batch."""

from __future__ import annotations

import pytest

from risk_engine.batch import BatchLimits
from risk_engine.errors import BatchSizeLimitError
from risk_engine.models import FactorShock, Holding, Portfolio, StressScenario
from risk_engine.scan import ScanJob, expand_pairs
from risk_engine.service import RiskService

from .conftest import log


@pytest.fixture
def svc(market) -> RiskService:
    s = RiskService()
    s.register_scenario(StressScenario("eq_drop", (FactorShock("A", -0.10),)))
    s.register_scenario(
        StressScenario("combo", (FactorShock("A", -0.05), FactorShock("B", 0.08)))
    )
    s.register_scenario(StressScenario("c_shock", (FactorShock("C", 0.03),)))
    return s


def _jobs(market, n_portfolios=3):
    names = ("eq_drop", "combo", "c_shock")
    return [
        ScanJob(
            Portfolio(f"fund-{i}", (Holding("A", 1.0 + i), Holding("B", -2.0), Holding("C", 1.0))),
            market,
            names,
        )
        for i in range(n_portfolios)
    ]


def test_expand_pairs_portfolio_major_order(market):
    jobs = _jobs(market, 2)
    pairs = expand_pairs(jobs)
    log.info("case=expand n=%d; criterion: portfolio-major order", len(pairs))
    assert len(pairs) == 6
    assert [p[0].name for p in pairs] == ["fund-0"] * 3 + ["fund-1"] * 3
    assert [p[2] for p in pairs[:3]] == ["eq_drop", "combo", "c_shock"]


def test_scan_matches_individual_runs(svc, market):
    jobs = _jobs(market, 3)
    out = svc.scan(jobs, BatchLimits(max_items=64, max_workers=4, timeout_seconds=10))
    log.info("case=scan-vs-single n=%d; criterion: identical to one-by-one", len(out))
    assert len(out) == 9
    k = 0
    for job in jobs:
        for name in job.scenario_names:
            single = svc.run_scenario(job.portfolio, job.market, svc.get_scenario(name))
            item = out[k]
            assert item.ok and item.index == k
            assert item.result["portfolio"] == job.portfolio.name
            assert item.result["scenario"] == name
            assert item.result["pnl"] == single.pnl
            assert item.result["pnl_by_instrument"] == dict(single.pnl_by_instrument)
            k += 1


def test_scan_order_stable_across_worker_counts(svc, market):
    jobs = _jobs(market, 4)
    out1 = svc.scan(jobs, BatchLimits(max_workers=1, timeout_seconds=10))
    out8 = svc.scan(jobs, BatchLimits(max_workers=8, timeout_seconds=10))
    log.info("case=scan-order workers=1 vs 8; criterion: same order/values")
    assert [r.result for r in out1] == [r.result for r in out8]
    assert [r.index for r in out8] == list(range(12))


def test_scan_unknown_scenario_isolated(svc, market):
    jobs = [
        ScanJob(
            Portfolio("p", (Holding("A", 1.0), Holding("B", 1.0), Holding("C", 1.0))),
            market,
            ("eq_drop", "not_registered", "combo"),
        )
    ]
    out = svc.scan(jobs, BatchLimits(max_items=8, timeout_seconds=5))
    codes = [None if r.ok else r.error["code"] for r in out]
    log.info("case=scan-isolation codes=%s; criterion: only bad slot fails", codes)
    assert [r.ok for r in out] == [True, False, True]
    assert out[1].error["code"] == "UNKNOWN_SCENARIO"
    assert out[0].result["scenario"] == "eq_drop"
    assert out[2].result["scenario"] == "combo"


def test_scan_size_limit_counts_expanded_pairs(svc, market):
    jobs = _jobs(market, 2)  # 2 portfolios x 3 scenarios = 6 pairs
    log.info("case=scan-size-limit pairs=6 max=5; criterion: BATCH_SIZE_LIMIT")
    with pytest.raises(BatchSizeLimitError):
        svc.scan(jobs, BatchLimits(max_items=5, timeout_seconds=5))


def test_scan_timeout_marks_only_unfinished(svc, market, monkeypatch):
    import time

    from risk_engine import service as service_mod

    real = svc._run_scan_pair

    def slow_pair(pair):
        if pair[2] == "combo":
            time.sleep(0.3)
        return real(pair)

    monkeypatch.setattr(svc, "_run_scan_pair", slow_pair)
    jobs = _jobs(market, 1)  # eq_drop(fast), combo(slow), c_shock(fast)
    out = svc.scan(jobs, BatchLimits(max_items=8, max_workers=3, timeout_seconds=0.15))
    codes = [None if r.ok else r.error["code"] for r in out]
    log.info("case=scan-timeout codes=%s; criterion: fast pairs kept, slow timed out", codes)
    assert out[0].ok and out[0].result["scenario"] == "eq_drop"
    assert out[1].ok is False and out[1].error["code"] == "EVALUATION_TIMEOUT"
    assert out[2].ok and out[2].result["scenario"] == "c_shock"
