"""Concurrent batch evaluation with stable order and isolated failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from risk_engine.models import BatchItemResult


@dataclass
class BatchLimits:
    max_items: int = 128
    max_workers: int = 8
    timeout_seconds: float = 5.0


def run_batch(
    requests: list,
    evaluate: Callable[[object], dict],
    limits: BatchLimits | None = None,
) -> list[BatchItemResult]:
    raise NotImplementedError
