"""Shared fixtures and logging helpers.

Every test logs a one-line input summary plus the assertion rationale so
that a test report explains *why* a case was accepted or rejected.
"""

from __future__ import annotations

import logging
import math

import pytest

from app.main import create_app
from app.market_data import MarketDataStore
from app.models import Portfolio, Position, PriceSeries
from app.service import AppState
from app.risk.scenarios import ScenarioEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("risk-tests")


@pytest.fixture
def engine_state() -> AppState:
    app = create_app()
    return app.state.engine


def seeded_store(symbols=("AAA", "BBB"), n: int = 120) -> MarketDataStore:
    store = MarketDataStore()
    for k, sym in enumerate(symbols):
        prices = [
            100.0 * math.exp(0.0008 * (i % 9 - 4) + 0.0004 * k * (i % 7 - 3))
            for i in range(n)
        ]
        store.load(PriceSeries(symbol=sym, prices=prices))
    return store


def sample_portfolio(name: str = "p") -> Portfolio:
    return Portfolio(
        name=name,
        positions=[
            Position(symbol="AAA", quantity=100),
            Position(symbol="BBB", quantity=50),
        ],
    )


def two_factor_engine() -> ScenarioEngine:
    engine = ScenarioEngine()
    engine.set_factor_map("AAA", {"equity": 1.0})
    engine.set_factor_map("BBB", {"equity": 0.6, "credit": 0.4})
    return engine
