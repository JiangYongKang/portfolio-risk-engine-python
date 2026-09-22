"""High-level facade combining risk, scenarios, versions and batch."""

from __future__ import annotations

from risk_engine.batch import BatchLimits
from risk_engine.models import (
    CorrelationAssumption,
    MarketData,
    Portfolio,
    RiskReport,
    ScenarioResult,
    StressScenario,
)


class RiskService:
    def __init__(self) -> None:
        self._scenarios: dict[str, StressScenario] = {}

    def register_version(self, rules) -> None:
        raise NotImplementedError

    def assess(
        self,
        portfolio: Portfolio,
        market: MarketData,
        *,
        window: int,
        confidence: float = 0.99,
        horizon_days: int = 1,
        correlation: CorrelationAssumption | None = None,
        version: str = "v1",
    ) -> RiskReport:
        raise NotImplementedError

    def run_scenario(
        self,
        portfolio: Portfolio,
        market: MarketData,
        scenario: StressScenario,
        correlation: CorrelationAssumption | None = None,
    ) -> ScenarioResult:
        raise NotImplementedError

    def batch_assess(
        self, requests: list, limits: BatchLimits | None = None
    ) -> list:
        raise NotImplementedError
