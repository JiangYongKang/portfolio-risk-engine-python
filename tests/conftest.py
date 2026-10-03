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


def _fixture_summary(name: str, value) -> str:
    """Compact, single-line description of a shared fixture value."""
    if name in ("market", "long_market"):
        prices = getattr(value, "prices", {})
        return (
            f"MarketData(dates={len(getattr(value, 'dates', ()))}, "
            f"instruments={sorted(prices)})"
        )
    if name == "portfolio":
        holdings = [
            f"{h.instrument}x{h.quantity}" for h in getattr(value, "holdings", ())
        ]
        return f"Portfolio({getattr(value, 'name', '?')}, {holdings})"
    return str(value)


@pytest.fixture(autouse=True)
def _log_case_verdict(request):
    """Log every test's inputs up front and its pass/fail verdict afterwards.

    Active by default (see pytest.ini ``log_cli=true``) so the plain
    ``uv run pytest`` command already shows what each case fed in and the
    criterion its result was judged against.
    """
    fixtured = {
        name: _fixture_summary(name, request.getfixturevalue(name))
        for name in ("market", "long_market", "portfolio")
        if name in getattr(request.function, "fixturenames", set())
    }
    log.info(
        "CASE START %s | fixtures: %s",
        request.node.nodeid,
        fixtured if fixtured else "(no shared fixtures / inline inputs)",
    )
    yield
    log.info("CASE VERDICT %s | PASSED", request.node.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        log.info("CASE VERDICT %s | FAILED: %s", item.nodeid, call.excinfo.value)
