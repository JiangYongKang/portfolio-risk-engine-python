"""In-memory market data repository.

Prices are stored per symbol as plain lists, chronologically ordered
(oldest first).  The store itself performs only basic value validation;
window and sufficiency checks belong to the risk layer.
"""

from __future__ import annotations

import math

from app.errors import InvalidPriceError
from app.models import PriceSeries


class MarketDataStore:
    def __init__(self) -> None:
        self._prices: dict[str, list[float]] = {}

    def load(self, series: PriceSeries) -> None:
        """Register or replace a symbol's price series."""
        values = list(series.prices)
        if len(values) < 2:
            raise InvalidPriceError(
                f"symbol {series.symbol!r} requires at least 2 prices, got {len(values)}"
            )
        for v in values:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise InvalidPriceError(
                    f"symbol {series.symbol!r} contains a non-finite price"
                )
            if v <= 0:
                raise InvalidPriceError(
                    f"symbol {series.symbol!r} contains a non-positive price: {v}"
                )
        self._prices[series.symbol] = values

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._prices

    def prices(self, symbol: str) -> list[float]:
        """Return a defensive copy of the stored prices."""
        return list(self._prices[symbol])

    def symbols(self) -> list[str]:
        return sorted(self._prices)
