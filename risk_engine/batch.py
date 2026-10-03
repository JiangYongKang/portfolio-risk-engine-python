"""Concurrent batch evaluation with stable order and isolated failures.

Guarantees
----------
* Output order matches request order exactly (indexed results).
* Concurrent results equal serial results because evaluated callables are
  treated as pure functions; the engine itself holds no per-request state.
* One request raising never aborts the batch: its slot carries a typed
  error; every other slot is unaffected.
* Over-limit batches are rejected up-front with ``BATCH_SIZE_LIMIT``;
  per-item time overruns produce ``EVALUATION_TIMEOUT`` in that slot.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from time import perf_counter

from risk_engine.errors import (
    BatchSizeLimitError,
    ComputationError,
    RiskEngineError,
)
from risk_engine.models import BatchItemResult


@dataclass(frozen=True)
class BatchLimits:
    max_items: int = 128
    max_workers: int = 8
    timeout_seconds: float = 5.0


def _evaluate_safely(evaluate: Callable[[object], dict], request: object) -> dict:
    try:
        return evaluate(request)
    except RiskEngineError:
        raise
    except Exception as exc:  # wrap, never leak raw numeric/runtime errors
        raise ComputationError(f"unexpected evaluation failure: {exc}") from exc


def run_batch(
    requests: list,
    evaluate: Callable[[object], dict],
    limits: BatchLimits | None = None,
) -> list[BatchItemResult]:
    limits = limits or BatchLimits()
    n = len(requests)
    if n > limits.max_items:
        raise BatchSizeLimitError(
            f"batch has {n} requests, limit is {limits.max_items}"
        )
    if n == 0:
        return []
    if limits.max_workers < 1 or limits.timeout_seconds <= 0:
        raise BatchSizeLimitError("invalid batch limits configuration")

    results: list[BatchItemResult | None] = [None] * n

    def _work(idx: int, req: object) -> tuple[int, dict | None, dict | None]:
        try:
            return idx, _evaluate_safely(evaluate, req), None
        except RiskEngineError as err:
            return idx, None, err.to_dict()

    deadline = perf_counter() + limits.timeout_seconds
    with ThreadPoolExecutor(
        max_workers=min(limits.max_workers, n),
        thread_name_prefix="risk-batch",
    ) as pool:
        futures = [pool.submit(_work, i, req) for i, req in enumerate(requests)]
        for i, fut in enumerate(futures):
            # A future that already completed keeps its result even if the
            # batch deadline has passed: only genuinely unfinished items
            # are marked EVALUATION_TIMEOUT.
            if fut.done():
                idx, payload, error = fut.result()
                results[idx] = BatchItemResult(
                    index=idx, ok=error is None, result=payload, error=error
                )
                continue
            remaining = deadline - perf_counter()
            if remaining <= 0:
                results[i] = BatchItemResult(
                    index=i,
                    ok=False,
                    error={
                        "code": "EVALUATION_TIMEOUT",
                        "reason": f"batch time budget {limits.timeout_seconds}s "
                        "exceeded",
                    },
                )
                continue
            try:
                idx, payload, error = fut.result(timeout=remaining)
            except FutureTimeout:
                results[i] = BatchItemResult(
                    index=i,
                    ok=False,
                    error={
                        "code": "EVALUATION_TIMEOUT",
                        "reason": f"item exceeded time budget",
                    },
                )
                continue
            results[idx] = BatchItemResult(
                index=idx,
                ok=error is None,
                result=payload,
                error=error,
            )

    return [r if r is not None else BatchItemResult(i, False, None, {
        "code": "EVALUATION_TIMEOUT",
        "reason": "no result produced",
    }) for i, r in enumerate(results)]
