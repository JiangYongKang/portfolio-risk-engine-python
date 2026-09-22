"""Tests for market-data ingestion validation."""

from __future__ import annotations

import pytest

from app.errors import InvalidPriceError
from app.market_data import MarketDataStore
from app.models import PriceSeries
from tests.conftest import log


@pytest.mark.parametrize(
    "prices,summary",
    [
        ([100.0], "only one price"),
        ([100.0, -1.0], "negative price"),
        ([100.0, 0.0], "zero price"),
        ([100.0, float("nan")], "NaN price"),
        ([100.0, float("inf")], "infinite price"),
    ],
)
def test_invalid_price_series_rejected(prices, summary):
    store = MarketDataStore()
    log.info("input prices=%s (%s)", prices, summary)
    with pytest.raises(InvalidPriceError) as ei:
        store.load(PriceSeries(symbol="AAA", prices=prices))
    assert ei.value.reason == "invalid_price_series"
    log.info("rejected reason=%s", ei.value.reason)


def test_store_returns_defensive_copy():
    store = MarketDataStore()
    store.load(PriceSeries(symbol="AAA", prices=[1.0, 2.0, 3.0]))
    out = store.prices("AAA")
    out[0] = 999.0
    assert store.prices("AAA")[0] == 1.0
    log.info("stored series cannot be mutated through returned copy")
