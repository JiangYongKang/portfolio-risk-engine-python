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
* A job whose inputs cannot even be parsed (``InvalidScanJob``, e.g. a
  malformed market date at the HTTP boundary) expands into one failed
  item per listed scenario carrying that job's typed error; every other
  job is unaffected and the expanded order is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

from risk_engine.errors import RiskEngineError
from risk_engine.models import MarketData, Portfolio


@dataclass(frozen=True)
class ScanJob:
    """One portfolio scanned against a list of registered scenario names."""

    portfolio: Portfolio
    market: MarketData
    scenario_names: tuple[str, ...]


@dataclass(frozen=True)
class InvalidScanJob:
    """A job whose raw inputs failed to parse before evaluation.

    Carries whatever could be salvaged for the input summary plus the
    typed error; expands into one failed item per scenario name so the
    (job x scenario) indexing of the batch stays intact.
    """

    portfolio_name: str
    holdings: int
    dates: int
    scenario_names: tuple[str, ...]
    error: RiskEngineError


@dataclass(frozen=True)
class ScanItem:
    """One expanded (portfolio, scenario) pair, or a pre-failed slot."""

    portfolio_name: str
    scenario_name: str
    holdings: int
    dates: int
    portfolio: Portfolio | None = None
    market: MarketData | None = None
    error: RiskEngineError | None = None

    @property
    def input_summary(self) -> dict:
        return {
            "portfolio": self.portfolio_name,
            "scenario": self.scenario_name,
            "holdings": self.holdings,
            "dates": self.dates,
        }


def expand_pairs(jobs: list[ScanJob]) -> list[tuple[Portfolio, MarketData, str]]:
    """Flatten scan jobs into ordered (portfolio, market, scenario) pairs."""
    return [
        (job.portfolio, job.market, name)
        for job in jobs
        for name in job.scenario_names
    ]


def expand_items(jobs: list[ScanJob | InvalidScanJob]) -> list[ScanItem]:
    """Flatten (possibly partially invalid) jobs into ordered scan items."""
    items: list[ScanItem] = []
    for job in jobs:
        if isinstance(job, InvalidScanJob):
            items.extend(
                ScanItem(
                    portfolio_name=job.portfolio_name,
                    scenario_name=name,
                    holdings=job.holdings,
                    dates=job.dates,
                    error=job.error,
                )
                for name in job.scenario_names
            )
        else:
            items.extend(
                ScanItem(
                    portfolio_name=job.portfolio.name,
                    scenario_name=name,
                    holdings=len(job.portfolio.holdings),
                    dates=len(job.market.dates),
                    portfolio=job.portfolio,
                    market=job.market,
                )
                for name in job.scenario_names
            )
    return items
