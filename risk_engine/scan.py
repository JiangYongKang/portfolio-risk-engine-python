"""Batch scan: many portfolios x many registered scenarios, one shot.

Semantics
---------
* The scan is expanded into (portfolio, scenario) pairs in submission
  order (portfolio-major: for each job, scenarios in listed order).
* Expansion reuses :func:`risk_engine.batch.run_batch`, so ordering,
  serial-equivalence and per-pair failure isolation are inherited.
* The item limit applies to the *expanded* pair count; the time budget
  applies to the whole scan.  Pairs that finished before the deadline
  keep their results (see run_batch).
"""

from __future__ import annotations

from dataclasses import dataclass

from risk_engine.models import MarketData, Portfolio


@dataclass(frozen=True)
class ScanJob:
    """One portfolio scanned against a list of registered scenario names."""

    portfolio: Portfolio
    market: MarketData
    scenario_names: tuple[str, ...]


def expand_pairs(jobs: list[ScanJob]) -> list[tuple[Portfolio, MarketData, str]]:
    """Flatten scan jobs into ordered (portfolio, market, scenario) pairs."""
    return [
        (job.portfolio, job.market, name)
        for job in jobs
        for name in job.scenario_names
    ]
