"""Shared fixtures + input-summary logging for every test.

Each test logs a compact summary of its inputs and the criterion that
decides pass/fail, satisfying the "explainable verdict" requirement.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest

from risk_engine.models import Holding, MarketData, Portfolio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("tests")


@pytest.fixture
def market() -> MarketData:
    dates = tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(60))
    # Deterministic, strictly positive series with differing dynamics.
    a = [100.0 + 0.3 * i for i in range(60)]
    b = [50.0 + ((i * 7) % 11) * 0.4 for i in range(60)]
    c = [70.0 + 0.1 * i + (0.5 if i % 6 == 0 else 0.0) for i in range(60)]
    return MarketData(dates, {"A": a, "B": b, "C": c})


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(
        "fund-x",
        (Holding("A", 10.0), Holding("B", -20.0), Holding("C", 5.0)),
    )


@pytest.fixture
def long_market() -> MarketData:
    """Longer history used for horizon/extreme tests."""
    dates = tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(260))
    a = [100.0 + 0.2 * i for i in range(260)]
    b = [50.0 + 0.05 * i + ((i * 3) % 7) * 0.3 for i in range(260)]
    c = [70.0 + 0.08 * i for i in range(260)]
    return MarketData(dates, {"A": a, "B": b, "C": c})


def summary(**kw) -> None:
    log.info("INPUT SUMMARY %s | decision: %s", kw.get("case"), kw)
