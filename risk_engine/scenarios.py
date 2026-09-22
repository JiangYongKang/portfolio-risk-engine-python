"""Named/composable stress scenarios and factor perturbations."""

from __future__ import annotations

from risk_engine.models import (
    CorrelationAssumption,
    MarketData,
    Portfolio,
    ScenarioResult,
    StressScenario,
)


def compose_scenarios(name: str, *scenarios: StressScenario) -> StressScenario:
    raise NotImplementedError


def single_factor_perturbation(
    portfolio: Portfolio, market: MarketData, instrument: str, move: float
) -> float:
    raise NotImplementedError


def evaluate_scenario(
    portfolio: Portfolio,
    market: MarketData,
    scenario: StressScenario,
    correlation: CorrelationAssumption | None = None,
) -> ScenarioResult:
    raise NotImplementedError
