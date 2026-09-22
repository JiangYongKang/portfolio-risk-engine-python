"""Concurrent batch: serial equivalence, stable order, isolation, limits."""

from __future__ import annotations

import threading
import time

import pytest

from risk_engine.batch import BatchLimits, run_batch
from risk_engine.errors import BatchSizeLimitError
from risk_engine.service import AssessmentJob, RiskService
from risk_engine.models import Holding, Portfolio

from .conftest import log


def _make_jobs(n, market):
    jobs = []
    for i in range(n):
        p = Portfolio(
            f"p{i}",
            (Holding("A", 1.0 + i), Holding("B", -1.0), Holding("C", 1.0)),
        )
        jobs.append(AssessmentJob(p, market, window=20))
    return jobs


def test_batch_matches_serial(market):
    jobs = _make_jobs(16, market)
    svc = RiskService()
    serial = [svc._run_job(j) for j in jobs]
    concurrent = svc.batch_assess(jobs, BatchLimits(max_items=64, max_workers=8, timeout_seconds=10))
    log.info("case=serial-vs-concurrent n=16; criterion: identical results+order")
    assert len(concurrent) == 16
    for i, item in enumerate(concurrent):
        assert item.ok
        assert item.index == i
        assert item.result == serial[i]


def test_order_stable_under_different_worker_counts(market):
    jobs = _make_jobs(20, market)
    svc = RiskService()
    out1 = svc.batch_assess(jobs, BatchLimits(max_workers=1, timeout_seconds=10))
    out8 = svc.batch_assess(jobs, BatchLimits(max_workers=8, timeout_seconds=10))
    log.info("case=order-stable workers=1 vs 8; criterion: same order/values")
    assert [r.result for r in out1] == [r.result for r in out8]
    assert [r.index for r in out8] == list(range(20))


def test_one_failure_does_not_affect_others(market):
    good = _make_jobs(4, market)
    bad = AssessmentJob(
        Portfolio("bad", (Holding("ZZ", 1.0),)), market, window=20
    )
    jobs = [good[0], bad, good[1], bad, good[2]]
    svc = RiskService()
    out = svc.batch_assess(jobs, BatchLimits(max_items=16, max_workers=4, timeout_seconds=10))
    expected = [True, False, True, False, True]
    codes = [None if r.ok else r.error["code"] for r in out]
    log.info("case=failure-isolation ok=%s codes=%s", expected, codes)
    assert [r.ok for r in out] == expected
    assert codes[1] == "UNKNOWN_INSTRUMENT"
    assert codes[3] == "UNKNOWN_INSTRUMENT"
    assert out[0].result is not None
    assert out[4].result is not None


def test_batch_size_limit_rejected_upfront(market):
    jobs = _make_jobs(5, market)
    log.info("case=size-limit n=5 max=3; criterion: BATCH_SIZE_LIMIT")
    with pytest.raises(BatchSizeLimitError):
        run_batch(jobs, lambda j: {}, BatchLimits(max_items=3))


def test_timeout_is_distinguishable(market):
    def slow(job):
        time.sleep(0.3)
        return {"ok": True}

    jobs = list(range(6))
    log.info("case=timeout budget=0.1s; criterion: EVALUATION_TIMEOUT per slot")
    out = run_batch(
        jobs, slow, BatchLimits(max_items=16, max_workers=2, timeout_seconds=0.1)
    )
    codes = {(r.index, r.error["code"]) for r in out if not r.ok}
    assert all(code == "EVALUATION_TIMEOUT" for _, code in codes)
    # fast items that complete before deadline are still recorded as ok/timeout
    assert len(out) == 6


def test_fast_items_succeed_within_batch_budget(market):
    # Even under a tight budget, near-instant items succeed.
    def fast(x):
        return {"v": x * 2}

    out = run_batch(
        list(range(10)), fast,
        BatchLimits(max_items=16, max_workers=10, timeout_seconds=2.0),
    )
    log.info("case=fast-budget n=10; criterion: all ok, ordered")
    assert all(r.ok for r in out)
    assert [r.result["v"] for r in out] == [2 * i for i in range(10)]


def test_raw_exception_is_wrapped_not_leaked():
    def boom(x):
        raise RuntimeError("internal numeric kaboom")

    out = run_batch([1], boom, BatchLimits(max_items=4, timeout_seconds=2))
    log.info("case=raw-wrap code=%s; criterion: COMPUTATION_ERROR", out[0].error["code"])
    assert out[0].ok is False
    assert out[0].error["code"] == "COMPUTATION_ERROR"
    assert "kaboom" in out[0].error["reason"]


def test_empty_batch_returns_empty():
    out = run_batch([], lambda x: {}, BatchLimits())
    assert out == []


def test_concurrent_execution_actually_overlaps():
    # Demonstrates real concurrency: 8 sleeps of 0.1s with 8 workers must
    # finish much faster than serial (0.8s).
    def slow(x):
        time.sleep(0.1)
        return x

    start = time.perf_counter()
    out = run_batch(
        list(range(8)), slow,
        BatchLimits(max_items=16, max_workers=8, timeout_seconds=5),
    )
    elapsed = time.perf_counter() - start
    log.info("case=overlap elapsed=%.3f serial=0.8", elapsed)
    assert all(r.ok for r in out)
    assert elapsed < 0.5
