"""Tests for concurrent batch evaluation: ordering, isolation, limits."""

from __future__ import annotations

import time

import pytest

from app.batch import evaluate_batch, evaluate_serial
from app.errors import BatchTimeoutError, BatchTooLargeError
from app.models import (
    BatchItem,
    ConfidenceLevel,
    Portfolio,
    Position,
)
from tests.conftest import log, sample_portfolio, seeded_store
from app.risk.metrics import compute_risk


def _items(n: int, bad_at=()) -> list[BatchItem]:
    store = seeded_store(n=80)
    items: list[BatchItem] = []
    for i in range(n):
        if i in bad_at:
            pf = Portfolio(name=f"bad{i}", positions=[Position(symbol="MISSING", quantity=1)])
        else:
            pf = sample_portfolio(f"pf{i}")
        items.append(BatchItem(
            id=f"item-{i:02d}", portfolio=pf,
            confidence=ConfidenceLevel(confidence=0.95, window=20 + (i % 5)),
        ))
    return items, store


def test_concurrent_matches_serial_and_ordering():
    items, store = _items(12)

    def runner(item: BatchItem):
        return compute_risk(item.portfolio, store,
                            item.confidence.window, item.confidence.confidence)

    serial = evaluate_serial(items, runner)
    concurrent = evaluate_batch(items, runner, max_workers=4)
    ids_c = [e.id for e in concurrent]
    assert ids_c == [it.id for it in items], ids_c
    log.info("order preserved across threads: %s", ids_c)
    for s, c in zip(serial, concurrent):
        assert s.ok == c.ok and s.reason == c.reason
        assert s.result == c.result
    log.info("judgement: concurrent results equal serial results field-by-field")


def test_single_failure_does_not_affect_others():
    items, store = _items(8, bad_at=(2, 5))

    def runner(item: BatchItem):
        return compute_risk(item.portfolio, store,
                            item.confidence.window, item.confidence.confidence)

    out = evaluate_batch(items, runner, max_workers=4)
    assert len(out) == 8
    status = {e.id: (e.ok, e.reason) for e in out}
    assert status["item-02"] == (False, "symbol_not_found")
    assert status["item-05"] == (False, "symbol_not_found")
    assert all(status[f"item-{i:02d}"][0] for i in [0, 1, 3, 4, 6, 7])
    log.info("input: failing indices 2,5; failures isolated, reasons distinguishable")


def test_batch_size_limit_rejected_upfront():
    items, store = _items(3)
    log.info("batch of 3 submitted with max_items=2")
    with pytest.raises(BatchTooLargeError) as ei:
        evaluate_batch(items, lambda it: None, max_items=2)
    assert ei.value.reason == "batch_too_large"
    log.info("rejected reason=batch_too_large (up-front, nothing executed)")


def test_duplicate_ids_rejected():
    from app.errors import BatchRejectedError

    items, store = _items(2)
    items[1].id = items[0].id
    with pytest.raises(BatchRejectedError):
        evaluate_batch(items, lambda it: None)
    log.info("duplicate item ids rejected reason=batch_rejected")


def test_timeout_marks_unfinished_items():
    def slow_runner(item: BatchItem):
        time.sleep(0.3)
        return "done"

    items, _ = _items(4)
    t0 = time.monotonic()
    out = evaluate_batch(items, slow_runner, timeout=0.1, max_workers=1)
    elapsed = time.monotonic() - t0
    timed_out = [e.id for e in out if e.reason == BatchTimeoutError.reason]
    log.info("4 slow items, 0.1s budget, 1 worker -> timed-out ids=%s elapsed=%.2fs",
             timed_out, elapsed)
    assert timed_out, "at least one item must be reported as timeout"
    assert [e.id for e in out] == [it.id for it in items]
    assert elapsed < 1.0
    log.info("judgement: timeout distinguishable, order retained, returns promptly")


def test_empty_batch_is_valid():
    assert evaluate_batch([], lambda it: None) == []
    log.info("empty batch returns empty list")
