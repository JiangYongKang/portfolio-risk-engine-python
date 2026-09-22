"""Concurrent batch evaluation with stable ordering and resource limits.

* Ordering: results are returned in exactly the input order; concurrency
  never reorders output.
* Isolation: a failure in one item is captured as a ``BatchEntry`` with
  ``ok=False`` and the domain ``reason``; other items still complete.
* Limits: requests larger than ``max_items`` are rejected up front with
  ``batch_too_large``; a wall-clock budget ``timeout`` is enforced and
  unfinished items are reported as ``batch_timeout``.
* Determinism: the pure numeric path is identical to serial evaluation;
  threads only schedule independent stateless calls.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable

from app.errors import (
    BatchRejectedError,
    BatchTimeoutError,
    BatchTooLargeError,
    RiskEngineError,
)
from app.models import BatchEntry, BatchItem

DEFAULT_MAX_ITEMS = 128
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_WORKERS = 8


def evaluate_batch(
    items: list[BatchItem],
    runner: Callable[[BatchItem], object],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[BatchEntry]:
    if not isinstance(items, list):
        raise BatchRejectedError("batch payload must be a list")
    if len(items) > max_items:
        raise BatchTooLargeError(
            f"batch size {len(items)} exceeds limit {max_items}"
        )
    if timeout <= 0 or max_workers <= 0:
        raise BatchRejectedError("timeout and max_workers must be positive")

    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise BatchRejectedError(f"duplicate batch item id: {it.id!r}")
        seen.add(it.id)

    if not items:
        return []

    results: dict[str, BatchEntry] = {}
    deadline = time.monotonic() + timeout
    workers = min(max_workers, len(items))

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    future_to_id = {pool.submit(runner, item): item.id for item in items}
    try:
        remaining = deadline - time.monotonic()
        for fut in concurrent.futures.as_completed(future_to_id, timeout=remaining):
            item_id = future_to_id[fut]
            try:
                value = fut.result()
            except RiskEngineError as exc:
                results[item_id] = BatchEntry(
                    id=item_id, ok=False, reason=exc.reason, detail=str(exc)
                )
            except Exception:  # never leak implementation exceptions
                results[item_id] = BatchEntry(
                    id=item_id,
                    ok=False,
                    reason="internal_error",
                    detail="evaluation failed for an undisclosed internal reason",
                )
            else:
                results[item_id] = BatchEntry(id=item_id, ok=True, result=value)
    except TimeoutError:
        pass  # remaining futures are reported as batch_timeout below
    finally:
        # do not block on unfinished work, and stop queued items
        pool.shutdown(wait=False, cancel_futures=True)

    ordered: list[BatchEntry] = []
    for item in items:
        if item.id in results:
            ordered.append(results[item.id])
        else:
            ordered.append(
                BatchEntry(
                    id=item.id,
                    ok=False,
                    reason=BatchTimeoutError.reason,
                    detail=(
                        f"item did not complete within the {timeout:.3g}s budget"
                    ),
                )
            )
    return ordered


def evaluate_serial(
    items: list[BatchItem],
    runner: Callable[[BatchItem], object],
) -> list[BatchEntry]:
    """Serial reference implementation; used by tests to assert equivalence."""
    out: list[BatchEntry] = []
    for item in items:
        out.append(_safe_entry(item.id, runner, item))
    return out


def _safe_entry(item_id: str, runner: Callable[[BatchItem], object], item: BatchItem) -> BatchEntry:
    try:
        value = runner(item)
    except RiskEngineError as exc:
        return BatchEntry(id=item_id, ok=False, reason=exc.reason, detail=str(exc))
    except Exception:
        return BatchEntry(
            id=item_id,
            ok=False,
            reason="internal_error",
            detail="evaluation failed for an undisclosed internal reason",
        )
    return BatchEntry(id=item_id, ok=True, result=value)
