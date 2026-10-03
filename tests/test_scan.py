"""Batch scan (portfolios x registered scenarios): order, equivalence,
isolation, capacity/time limits."""

from __future__ import annotations

import time

import pytest

from risk_engine.errors import BatchSizeLimitError
from risk_engine.models import FactorShock, StressScenario
from risk_engine.scan import ScanCell, ScanLimits, cross_product_cells
from risk_engine.service import RiskService

from .conftest import log


def _svc(market) -> RiskService:
    svc = RiskService()
    svc.register_scenario(
        StressScenario("eq_drop", (FactorShock("A", -0.10),))
    )
    svc.register_scenario(
        StressScenario("credit", (FactorShock("B", -0.05),))
    )
    svc.register_scenario(
        StressScenario("rates_up", (FactorShock("C", 0.02),))
    )
    return svc


def test_cross_product_order_is_fixed(market):
    from risk_engine.models import Holding, Portfolio

    ports = {
        "p1": (Portfolio("p1", (Holding("A", 1.0),)), market),
        "p2": (Portfolio("p2", (Holding("B", 2.0),)), market),
    }
    cells = cross_product_cells(ports, ["eq_drop", "credit", "rates_up"])
    order = [(c.portfolio.name, c.scenario_name) for c in cells]
    log.info("case=cross-order cells=%s", order)
    assert order == [
        ("p1", "eq_drop"), ("p1", "credit"), ("p1", "rates_up"),
        ("p2", "eq_drop"), ("p2", "credit"), ("p2", "rates_up"),
    ]


def test_scan_matches_single_runs_and_keeps_order(market):
    from risk_engine.models import Holding, Portfolio

    svc = _svc(market)
    ports = {
        f"p{i}": (
            Portfolio(
                f"p{i}",
                (Holding("A", 1.0 + i), Holding("B", -2.0), Holding("C", 3.0)),
            ),
            market,
        )
        for i in range(4)
    }
    names = ["eq_drop", "credit", "rates_up"]
    cells = cross_product_cells(ports, names)
    out = svc.scan(cells, ScanLimits(max_items=64, max_workers=8, timeout_seconds=10))

    # serial, single-run ground truth
    serial = [
        svc.run_scenario(c.portfolio, c.market, svc.get_scenario(c.scenario_name))
        for c in cells
    ]
    log.info("case=scan-vs-single n=%d; criterion: equal pnl+order", len(cells))
    assert len(out) == 12
    for i, item in enumerate(out):
        assert item.ok, item.error
        assert item.index == i
        assert item.result["pnl"] == serial[i].pnl
        assert item.result["scenario"] == cells[i].scenario_name
        assert item.result["input"]["portfolio"] == cells[i].portfolio.name
        assert item.result["version"] == "v1"


def test_scan_unknown_scenario_isolated(market):
    from risk_engine.models import Holding, Portfolio

    svc = _svc(market)
    p = Portfolio("p", (Holding("A", 1.0), Holding("B", 1.0)))
    cells = [
        ScanCell(p, market, "eq_drop"),
        ScanCell(p, market, "nope"),
        ScanCell(p, market, "credit"),
    ]
    out = svc.scan(cells, ScanLimits(max_items=16, timeout_seconds=10))
    flags = [(r.index, r.ok, (r.error or {}).get("code")) for r in out]
    log.info("case=scan-isolation %s", flags)
    assert flags == [
        (0, True, None),
        (1, False, "UNKNOWN_SCENARIO"),
        (2, True, None),
    ]


def test_scan_capacity_limit_rejects_whole_batch(market):
    from risk_engine.models import Holding, Portfolio

    svc = _svc(market)
    p = Portfolio("p", (Holding("A", 1.0),))
    cells = [ScanCell(p, market, "eq_drop") for _ in range(5)]
    log.info("case=scan-capacity n=5 max=3; criterion: BATCH_SIZE_LIMIT")
    with pytest.raises(BatchSizeLimitError):
        svc.scan(cells, ScanLimits(max_items=3))


def test_scan_completed_cells_kept_when_deadline_hits():
    # A slow cell must not cause already-finished cells to be dropped or
    # mislabelled as timeouts.
    from risk_engine.models import Holding, MarketData, Portfolio
    from datetime import date, timedelta

    dates = tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(3))
    market = MarketData(dates, {"A": [100.0, 101.0, 102.0]})
    svc = _svc(market)
    p = Portfolio("p", (Holding("A", 1.0),))

    calls = {"n": 0}

    def resolver(cell):
        calls["n"] += 1
        # every second call is slow beyond the batch budget
        if calls["n"] % 2 == 0:
            time.sleep(0.3)
        else:
            time.sleep(0.01)
        return {"pnl": float(calls["n"])}

    cells = [ScanCell(p, market, "eq_drop") for _ in range(4)]
    from risk_engine.scan import scan_cells

    out = scan_cells(
        cells, resolver,
        limits=ScanLimits(max_items=16, max_workers=4, timeout_seconds=0.12),
    )
    codes = [(r.index, r.ok, (r.error or {}).get("code"), r.result) for r in out]
    log.info("case=scan-deadline %s", codes)
    # fast cells must remain ok with their computed value preserved
    fast = [r for r in out if r.index in (0, 2)]
    assert all(r.ok and r.result is not None for r in fast)
    # slow cells are timeouts, never silently dropped
    slow = [r for r in out if r.index in (1, 3)]
    assert all(not r.ok and r.error["code"] == "EVALUATION_TIMEOUT" for r in slow)
    assert [r.index for r in out] == [0, 1, 2, 3]
