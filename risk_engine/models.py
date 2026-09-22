"""Shared data contracts for the risk engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Sequence


InstrumentId = str
Version = str


@dataclass(frozen=True)
class Holding:
    instrument: InstrumentId
    quantity: float  # units; mark-to-market PnL = quantity * price change


@dataclass(frozen=True)
class Portfolio:
    name: str
    holdings: tuple[Holding, ...]

    @property
    def instruments(self) -> tuple[InstrumentId, ...]:
        return tuple(h.instrument for h in self.holdings)


@dataclass(frozen=True)
class MarketData:
    """Aligned daily price histories keyed by instrument.

    Every series must be aligned to ``dates`` and have the same length.
    """

    dates: tuple[date, ...]
    prices: Mapping[InstrumentId, Sequence[float]]

    def __post_init__(self) -> None:  # pragma: no cover - real checks live in series
        object.__setattr__(self, "prices", dict(self.prices))


@dataclass(frozen=True)
class TailRiskEstimate:
    confidence: float
    horizon_days: int
    var: float          # loss, positive number, in portfolio currency
    expected_shortfall: float


@dataclass(frozen=True)
class RiskReport:
    portfolio: str
    version: Version
    window: int
    as_of: date
    historical: TailRiskEstimate
    parametric: TailRiskEstimate
    portfolio_volatility: float          # 1-day parametric portfolio sigma
    gross_exposure: float
    net_exposure: float
    exposure_by_instrument: Mapping[InstrumentId, float]


@dataclass(frozen=True)
class CorrelationAssumption:
    """Named, composable correlation assumption applied to a factor set."""

    name: str
    instruments: tuple[InstrumentId, ...]
    matrix: tuple[tuple[float, ...], ...]
    version: Version = "v1"


@dataclass(frozen=True)
class FactorShock:
    """A single-factor relative move, e.g. instrument 'SPX' shocked -0.10."""

    instrument: InstrumentId
    relative_move: float


@dataclass(frozen=True)
class StressScenario:
    name: str
    shocks: tuple[FactorShock, ...]
    version: Version = "v1"


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    version: Version
    pnl: float
    pnl_by_instrument: Mapping[InstrumentId, float]
    factor_moves: Mapping[InstrumentId, float]
    consistent_with_single_factor: bool


@dataclass(frozen=True)
class BatchItemResult:
    index: int
    ok: bool
    result: dict | None = None
    error: dict | None = None
