"""Readable snapshot export: byte-stability, readability, truncation, isolation."""

from __future__ import annotations

import time

import pytest

from risk_engine.batch import BatchLimits, run_batch
from risk_engine.errors import BatchSizeLimitError, InvalidParameterError
from risk_engine.models import FactorShock, Holding, Portfolio, StressScenario
from risk_engine.scan import ScanJob
from risk_engine.service import RiskService
from risk_engine.snapshot import canonical_json, render_text

from .conftest import log


@pytest.fixture
def svc(market) -> RiskService:
    s = RiskService()
    s.register_scenario(StressScenario("eq_drop", (FactorShock("A", -0.10),)))
    s.register_scenario(StressScenario("b_rally", (FactorShock("B", 0.20),)))
    return s


def _jobs(market):
    return [
        ScanJob(Portfolio("alpha", (Holding("A", 10.0), Holding("B", 1.0))),
                market, ("eq_drop", "b_rally")),
        ScanJob(Portfolio("beta", (Holding("A", 3.0), Holding("B", -8.0))),
                market, ("eq_drop", "b_rally")),
    ]


def test_text_export_byte_identical_across_runs_and_workers(svc, market):
    jobs = _jobs(market)
    texts = {
        svc.scan_snapshot_text(jobs, BatchLimits(max_workers=w, timeout_seconds=10))
        for w in (1, 4, 8)
    }
    again = svc.scan_snapshot_text(jobs, BatchLimits(max_workers=4,
                                                     timeout_seconds=10))
    log.info("case=text-determinism workers=1/4/8 repeat=2; "
             "criterion: one byte string across all exports")
    assert len(texts) == 1
    assert again in texts


def test_text_export_is_self_explanatory(svc, market):
    jobs = _jobs(market) + [
        ScanJob(Portfolio("gamma", (Holding("A", 1.0),)), market, ("ghost",))
    ]
    text = svc.scan_snapshot_text(jobs, BatchLimits(timeout_seconds=10))
    log.info("case=text-readable lines=%d; criterion: header/summary/version/"
             "rationale/rejection-reason/category all present", len(text.splitlines()))
    assert "RISK ENGINE RESULT SNAPSHOT" in text
    assert "format: risk-engine-snapshot/1" in text
    # input summary, rules version and rationale per entry
    assert "portfolio=alpha" in text and "scenario=eq_drop" in text
    assert "rules version: v1" in text
    assert "rationale: scenario 'eq_drop' evaluated under rules v1" in text
    # rejected entry: typed error + category + reason, human readable
    assert "ENTRY 4 [scenario] REJECTED" in text
    assert "code=UNKNOWN_SCENARIO" in text
    assert "category=invalid_input" in text
    assert "scenario 'ghost' is not registered" in text
    assert "failures by category: invalid_input=1" in text
    # rankings rendered with explicit metric/order
    assert "worst_by_portfolio" in text and "metric=pnl" in text


def test_text_export_has_no_timestamps(svc, market):
    jobs = _jobs(market)
    t1 = svc.scan_snapshot_text(jobs, BatchLimits(timeout_seconds=10))
    time.sleep(0.02)
    t2 = svc.scan_snapshot_text(jobs, BatchLimits(timeout_seconds=10))
    log.info("case=no-wallclock; criterion: exports 20ms apart are identical")
    assert t1 == t2


class _SlowFirstService(RiskService):
    """First pair sleeps past the deadline; the rest are instant."""

    def _run_scan_pair(self, pair):
        if pair[2] == "eq_drop" and pair[0].name == "alpha":
            time.sleep(0.3)
        return super()._run_scan_pair(pair)


def test_snapshot_keeps_completed_results_when_deadline_passes(market):
    svc = _SlowFirstService()
    svc.register_scenario(StressScenario("eq_drop", (FactorShock("A", -0.10),)))
    svc.register_scenario(StressScenario("b_rally", (FactorShock("B", 0.20),)))
    jobs = _jobs(market)
    # 2 workers: the slow first pair occupies one worker past the deadline
    # while the other three pairs finish on the second worker in time.
    limits = BatchLimits(max_workers=2, timeout_seconds=0.15)
    snap = svc.scan_snapshot(jobs, limits)
    flags = [(e["index"], e["ok"], (e["error"] or {}).get("code"),
              (e["error"] or {}).get("category")) for e in snap["entries"]]
    log.info("case=snapshot-truncation flags=%s; criterion: finished entries "
             "kept verbatim, unfinished marked timeout", flags)
    # entry 0 blew the budget; it is the only timed-out slot
    assert snap["entries"][0]["ok"] is False
    assert snap["entries"][0]["error"]["code"] == "EVALUATION_TIMEOUT"
    assert snap["entries"][0]["error"]["category"] == "timeout"
    # every later entry finished before the deadline was noticed and is
    # preserved verbatim — identical to a run with a generous budget
    full = svc.scan_snapshot(jobs, BatchLimits(max_workers=1, timeout_seconds=30))
    for i in range(1, len(snap["entries"])):
        assert snap["entries"][i] == full["entries"][i]
    # the text export reflects the same truncation, deterministically
    text = render_text(snap)
    assert "EVALUATION_TIMEOUT" in text and "category=timeout" in text
    assert "outcomes: 3 ok, 1 rejected" in text


def test_failed_slots_from_unresolvable_jobs_keep_positions(svc, market):
    jobs = _jobs(market)
    error = {"code": "INVALID_INPUT", "category": "invalid_input",
             "reason": "malformed date '2026-13-99'"}
    summary = {"portfolio": "broken", "scenario": "eq_drop",
               "holdings": 1, "dates": 60}
    # a job that could not be resolved contributes failed slots at its
    # original expansion positions (here: between the two good jobs)
    failed = {2: (error, summary)}
    snap = svc.scan_snapshot(jobs, BatchLimits(timeout_seconds=10),
                             failed_slots=failed)
    idx = [e["index"] for e in snap["entries"]]
    log.info("case=failed-slots indices=%s; criterion: failed slot sits at "
             "its original position, others shift but stay ordered", idx)
    assert idx == [0, 1, 2, 3, 4]
    assert snap["count"] == 5
    bad = snap["entries"][2]
    assert bad["ok"] is False and bad["error"]["code"] == "INVALID_INPUT"
    assert bad["input_summary"]["portfolio"] == "broken"
    assert "INVALID_INPUT" in bad["rationale"]
    # good neighbours are untouched
    assert snap["entries"][1]["ok"] and snap["entries"][3]["ok"]
    # failed entries stay out of the rankings
    assert all(i["name"] != "broken"
               for i in snap["rankings"]["worst_by_portfolio"]["items"])


def test_failure_categories_are_distinguishable():
    def evaluate(x):
        if x == "bad-input":
            raise InvalidParameterError("confidence out of range")
        if x == "boom":
            raise RuntimeError("numeric kaboom")
        if x == "slow":
            time.sleep(0.3)
        return {"v": x}

    out = run_batch(
        ["ok", "bad-input", "boom", "slow"],
        evaluate,
        BatchLimits(max_items=8, max_workers=4, timeout_seconds=0.15),
    )
    seen = {r.index: (r.ok, None if r.ok else (r.error["code"], r.error["category"]))
            for r in out}
    log.info("case=categories %s; criterion: ok/invalid_input/internal/timeout "
             "all distinct", seen)
    assert seen[0] == (True, None)
    assert seen[1] == (False, ("INVALID_PARAMETER", "invalid_input"))
    assert seen[2] == (False, ("COMPUTATION_ERROR", "internal"))
    assert seen[3] == (False, ("EVALUATION_TIMEOUT", "timeout"))
    # the fourth family: over-limit batches are rejected up-front
    try:
        run_batch([1, 2], evaluate, BatchLimits(max_items=1))
        raise AssertionError("expected BatchSizeLimitError")
    except BatchSizeLimitError as err:
        log.info("case=category-limit code=%s category=%s", err.code, err.category)
        assert err.to_dict()["category"] == "resource_limit"


def test_empty_inputs_are_handled(svc, market):
    empty_jobs = [ScanJob(Portfolio("solo", (Holding("A", 1.0),)), market, ())]
    snap = svc.scan_snapshot(empty_jobs, BatchLimits(timeout_seconds=5))
    text = render_text(snap)
    log.info("case=empty-scan entries=%d; criterion: valid empty snapshot, "
             "stable serialization", snap["count"])
    assert snap["count"] == 0 and snap["entries"] == []
    assert "entries: 0" in text
    assert canonical_json(snap) == canonical_json(svc.scan_snapshot(
        empty_jobs, BatchLimits(timeout_seconds=5)))
    assert svc.scan([], BatchLimits()) == []
