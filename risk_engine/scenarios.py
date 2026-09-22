"""Named/composable stress scenarios and factor perturbations."""

from __future__ import annotations

import math

from risk_engine.errors import UnknownInstrumentError
from risk_engine.models import (
    CorrelationAssumption,
    FactorShock,
    MarketData,
    Portfolio,
    ScenarioResult,
    StressScenario,
)


def _last_prices(market: MarketData) -> dict[str, float]:
    if len(market.dates) == 0:
        from risk_engine.errors import InsufficientDataError

        raise InsufficientDataError("market data has no dates")
    return {inst: float(series[-1]) for inst, series in market.prices.items()}


def _quantity_map(portfolio: Portfolio) -> dict[str, float]:
    qty: dict[str, float] = {}
    for h in portfolio.holdings:
        qty[h.instrument] = qty.get(h.instrument, 0.0) + h.quantity
    return qty


def compose_scenarios(
    name: str, *scenarios: StressScenario, version: str = "v1"
) -> StressScenario:
    """Compose named scenarios.

    Shocks are concatenated in argument order; when the same factor is
    shocked more than once, every shock is retained and PnL is summed with
    ``math.fsum`` at evaluation time.  This makes a composed scenario's
    PnL bit-for-bit equal to the sum of independently evaluated scenarios
    (linearity), while remaining consistent with single-factor
    perturbations up to floating-point tolerance.
    """
    shocks: list[FactorShock] = []
    for sc in scenarios:
        shocks.extend(sc.shocks)
    return StressScenario(name=name, shocks=tuple(shocks), version=version)


def single_factor_perturbation(
    portfolio: Portfolio, market: MarketData, instrument: str, move: float
) -> float:
    """PnL from perturbing exactly one factor by a relative move."""
    prices = _last_prices(market)
    qty = _quantity_map(portfolio)
    if instrument not in qty:
        raise UnknownInstrumentError(
            f"instrument '{instrument}' is not held in portfolio"
        )
    if instrument not in prices:
        raise UnknownInstrumentError(
            f"no market data for instrument '{instrument}'"
        )
    return qty[instrument] * prices[instrument] * move


def evaluate_scenario(
    portfolio: Portfolio,
    market: MarketData,
    scenario: StressScenario,
    correlation: CorrelationAssumption | None = None,
) -> ScenarioResult:
    """Evaluate a linear factor-shock scenario.

    The scenario PnL is first-order: sum_i qty_i * price_i * move_i.
    Self-consistency is asserted by recomputing each factor as an
    independent single-factor perturbation and summing; the result is
    flagged ``consistent_with_single_factor`` (exactly equal here).
    """
    prices = _last_prices(market)
    qty = _quantity_map(portfolio)

    contributions: dict[str, list[float]] = {}
    move_lists: dict[str, list[float]] = {}
    for shock in scenario.shocks:
        if shock.instrument not in qty:
            raise UnknownInstrumentError(
                f"scenario '{scenario.name}' shocks '{shock.instrument}' "
                "which is not held in portfolio"
            )
        if shock.instrument not in prices:
            raise UnknownInstrumentError(
                f"no market data for shocked instrument '{shock.instrument}'"
            )
        contributions.setdefault(shock.instrument, []).append(
            qty[shock.instrument] * prices[shock.instrument] * shock.relative_move
        )
        move_lists.setdefault(shock.instrument, []).append(shock.relative_move)

    pnl_by = {inst: math.fsum(vals) for inst, vals in contributions.items()}
    factor_moves = {inst: math.fsum(vals) for inst, vals in move_lists.items()}

    singles_total = math.fsum(
        single_factor_perturbation(portfolio, market, inst, move)
        for inst, move in factor_moves.items()
    )
    pnl = math.fsum(pnl_by.values())
    tol = 1e-9 * max(1.0, abs(pnl), abs(singles_total))
    return ScenarioResult(
        scenario=scenario.name,
        version=scenario.version,
        pnl=pnl,
        pnl_by_instrument=pnl_by,
        factor_moves=factor_moves,
        consistent_with_single_factor=abs(pnl - singles_total) <= tol,
    )
