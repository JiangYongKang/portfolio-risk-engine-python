"""Snapshot determinism and worst-case rankings."""

from __future__ import annotations

import json

import pytest

from risk_engine.batch import BatchLimits
from risk_engine.models import FactorShock, Holding, Portfolio, StressScenario
from risk_engine.scan import ScanJob
from risk_engine.service import RiskService
from risk_engine.snapshot import (
    SnapshotEntry,
    build_snapshot,
    canonical_json,
    rank_worst,
)

from .conftest import log


@pytest.fixture
def svc(market) -> RiskService:
    s = RiskService()
    s.register_scenario(StressScenario("eq_drop", (FactorShock("A", -0.10),)))
    s.register_scenario(StressScenario("b_rally", (FactorShock("B", 0.20),)))
    s.register_scenario(StressScenario("mixed", (FactorShock("A", -0.04),
                                                 FactorShock("C", 0.05))))
    return s


def _jobs(market):
    return [
        ScanJob(Portfolio("alpha", (Holding("A", 10.0), Holding("B", 1.0),
                                    Holding("C", 2.0))), market,
                ("eq_drop", "b_rally", "mixed")),
        ScanJob(Portfolio("beta", (Holding("A", 3.0), Holding("B", -8.0),
                                   Holding("C", 1.0))), market,
                ("eq_drop", "b_rally", "mixed")),
    ]


def test_snapshot_byte_identical_across_runs(svc, market):
    jobs = _jobs(market)
    snaps = {
        canonical_json(svc.scan_snapshot(jobs, BatchLimits(max_workers=w,
                                                           timeout_seconds=10)))
        for w in (1, 4, 8)
    }
    log.info("case=snapshot-determinism workers=1/4/8; criterion: one byte string")
    assert len(snaps) == 1


def test_snapshot_entries_carry_summary_version_rationale(svc, market):
    snap = svc.scan_snapshot(_jobs(market), BatchLimits(timeout_seconds=10))
    e = snap["entries"][0]
    log.info("case=entry-fields rationale=%s", e["rationale"])
    assert e["input_summary"] == {
        "portfolio": "alpha", "scenario": "eq_drop", "holdings": 3, "dates": 60,
    }
    assert e["version"] == "v1"
    assert "eq_drop" in e["rationale"] and "pnl=" in e["rationale"]
    assert snap["format"] == "risk-engine-snapshot/1"
    assert snap["count"] == 6


def test_snapshot_records_failures_with_reason(svc, market):
    jobs = [ScanJob(Portfolio("g", (Holding("A", 1.0),)), market,
                    ("eq_drop", "ghost"))]
    snap = svc.scan_snapshot(jobs, BatchLimits(timeout_seconds=5))
    bad = snap["entries"][1]
    log.info("case=failed-entry rationale=%s", bad["rationale"])
    assert bad["ok"] is False
    assert bad["error"]["code"] == "UNKNOWN_SCENARIO"
    assert "UNKNOWN_SCENARIO" in bad["rationale"]
    # failed entries are excluded from rankings
    assert all(i["name"] != "ghost"
               for i in snap["rankings"]["worst_by_scenario"]["items"])


def test_rankings_direction_and_tie_break(svc, market):
    snap = svc.scan_snapshot(_jobs(market), BatchLimits(timeout_seconds=10))
    by_p = snap["rankings"]["worst_by_portfolio"]
    by_s = snap["rankings"]["worst_by_scenario"]
    log.info("case=rankings by_portfolio=%s by_scenario=%s",
             by_p["items"], by_s["items"])
    # worst = most negative pnl first
    assert by_p["metric"] == "pnl" and "ascending" in by_p["order"]
    assert by_p["items"][0]["name"] == "alpha"   # 10x A hit hardest by eq_drop
    assert by_p["items"][0]["value"] <= by_p["items"][1]["value"]
    assert by_s["items"][0]["name"] == "eq_drop"
    assert [i["rank"] for i in by_s["items"]] == [1, 2, 3]


def test_ranking_tie_break_is_deterministic():
    def entry(i, name, pnl):
        return SnapshotEntry(
            index=i, kind="scenario",
            input_summary={"portfolio": name, "scenario": "s"},
            version="v1", ok=True, rationale="",
            result={"pnl": pnl}, error=None,
        )

    entries = [entry(0, "gamma", -5.0), entry(1, "alpha", -5.0),
               entry(2, "beta", -5.0), entry(3, "delta", -1.0)]
    ranked = rank_worst(entries, group_by="portfolio")
    log.info("case=tie-break items=%s", ranked["items"])
    # equal pnl -> alphabetical by name, independent of submission order
    assert [i["name"] for i in ranked["items"]] == ["alpha", "beta", "gamma",
                                                    "delta"]
    assert "tie_break" in ranked


def test_canonical_json_stable_and_rejects_nan():
    obj = {"b": 1, "a": [0.1 + 0.2, {"z": None, "y": "x"}]}
    log.info("case=canonical-json; criterion: sorted keys, exact floats, no NaN")
    assert canonical_json(obj) == canonical_json(json.loads(canonical_json(obj)))
    assert canonical_json(obj).startswith('{"a"')
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
