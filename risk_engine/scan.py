"""Batch scanning of many portfolios x many registered scenarios.

A scan expands a set of (portfolio, scenario) cells, evaluates each cell
through the same path as a single scenario evaluation, and returns cells
in submission order with per-cell isolated failures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping

from risk_engine.batch import BatchLimits, run_batch
from risk_engine.errors import BatchSizeLimitError, InvalidParameterError
from risk_engine.models import (
    BatchItemResult,
    CorrelationAssumption,
    MarketData,
    Portfolio,
    Version,
)


@dataclass(frozen=True)
class ScanCell:
    """One (portfolio, scenario) pair to evaluate.

    Scenarios are referenced by their registered name so that a batch of
    already-registered scenarios can be scanned against.
    """

    portfolio: Portfolio
    market: MarketData
    scenario_name: str
    correlation_name: str | None = None
    version: Version = "v1"


@dataclass(frozen=True)
class ScanLimits:
    max_items: int = 256
    max_workers: int = 8
    timeout_seconds: float = 10.0

    def to_batch_limits(self) -> BatchLimits:
        return BatchLimits(
            max_items=self.max_items,
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
        )


def cross_product_cells(
    portfolios: Mapping[str, tuple[Portfolio, MarketData]],
    scenario_names: list[str],
    *,
    correlation_name: str | None = None,
    version: str = "v1",
) -> list[ScanCell]:
    """Expand portfolios x scenarios in fixed order.

    Ordering: outer loop over portfolios (insertion order), inner loop
    over ``scenario_names``.  This is the submission order of the scan.
    """
    if not scenario_names:
        raise InvalidParameterError("scan requires at least one scenario name")
    if not portfolios:
        raise InvalidParameterError("scan requires at least one portfolio")
    cells: list[ScanCell] = []
    for name, (portfolio, market) in portfolios.items():
        for scenario_name in scenario_names:
            cells.append(
                ScanCell(
                    portfolio=portfolio,
                    market=market,
                    scenario_name=scenario_name,
                    correlation_name=correlation_name,
                    version=version,
                )
            )
    return cells


def scan_cells(
    cells: list[ScanCell],
    resolver: Callable[[ScanCell], dict],
    *,
    limits: ScanLimits | None = None,
) -> list[BatchItemResult]:
    """Evaluate scan cells concurrently with isolated failures.

    ``resolver`` maps a cell to its result dict; in the service it resolves
    the registered scenario/correlation names and runs the exact single
    scenario evaluation path, guaranteeing scan == single-run results.

    Over-capacity batches are rejected up-front with ``BATCH_SIZE_LIMIT``.
    """
    limits = limits or ScanLimits()
    n = len(cells)
    if n > limits.max_items:
        raise BatchSizeLimitError(
            f"scan has {n} cells (portfolios x scenarios), limit is "
            f"{limits.max_items}"
        )
    return run_batch(cells, resolver, limits.to_batch_limits())
