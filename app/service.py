"""Application-level facade wiring stores, engines and the registry."""

from __future__ import annotations

from dataclasses import dataclass

from app.market_data import MarketDataStore
from app.models import (
    BatchItem,
    CorrelationOverride,
    Portfolio,
    RiskSummary,
)
from app.risk.covariance import validate_matrix
from app.risk.metrics import compute_risk
from app.risk.scenarios import ScenarioEngine
from app.versioning import ParameterRegistry
from app.batch import evaluate_batch

DEFAULT_RULE_VERSION = 1


@dataclass
class AppState:
    store: MarketDataStore
    scenarios: ScenarioEngine
    registry: ParameterRegistry

    def risk(
        self,
        portfolio: Portfolio,
        window: int,
        confidence: float,
        correlation_override: list[list[float]] | None = None,
    ) -> RiskSummary:
        return compute_risk(
            portfolio, self.store, window, confidence, correlation_override
        )

    def run_batch(self, items: list[BatchItem], timeout: float, max_items: int):
        def runner(item: BatchItem) -> RiskSummary:
            return compute_risk(
                item.portfolio,
                self.store,
                item.confidence.window,
                item.confidence.confidence,
            )

        return evaluate_batch(items, runner, timeout=timeout, max_items=max_items)

    def apply_correlation_override(self, override: CorrelationOverride) -> list[list[float]]:
        # Strict validation only; the validated matrix is echoed back.
        return [list(map(float, row)) for row in validate_matrix(
            override.matrix, override.factors
        )]


def build_app_state() -> AppState:
    engine = ScenarioEngine()
    return AppState(
        store=MarketDataStore(),
        scenarios=engine,
        registry=ParameterRegistry(engine, current_rule_version=DEFAULT_RULE_VERSION),
    )
