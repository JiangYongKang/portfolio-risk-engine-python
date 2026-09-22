"""Named, composable stress scenarios.

A :class:`FactorShock` is a *relative* return shock to one named factor
(e.g. ``equity: -0.10`` means equity-sensitive positions fall 10%).  Each
portfolio symbol is mapped to factors with loadings (sensitivities); the
shock applied to a position value is

    r_i = sum_f loading[i, f] * shock[f].

Composite scenarios aggregate named component scenarios by summing their
per-factor shocks.  Factor contributions are computed independently of the
aggregated P&L (same loadings, one factor at a time), so the summed
contributions equal the reported total by construction.
"""

from __future__ import annotations

import math

from app.errors import (
    EmptyPortfolioError,
    InvalidPositionError,
    ScenarioDefinitionError,
    ScenarioNotFoundError,
)
from app.models import Portfolio, ScenarioDefinition, ScenarioResult


class ScenarioEngine:
    def __init__(self) -> None:
        self._definitions: dict[str, ScenarioDefinition] = {}
        # symbol -> {factor: loading}; default factor == symbol with loading 1.0
        self._factor_map: dict[str, dict[str, float]] = {}

    # --- configuration -------------------------------------------------------

    def set_factor_map(self, symbol: str, loadings: dict[str, float]) -> None:
        for f, l in loadings.items():
            if not math.isfinite(l):
                raise ScenarioDefinitionError(
                    f"loading for {symbol!r}/{f!r} must be finite"
                )
        self._factor_map[symbol] = dict(loadings)

    def define(self, definition: ScenarioDefinition) -> None:
        name = definition.name
        if definition.type.value == "factor_shock":
            factors = [s.factor for s in definition.shocks]
            if len(factors) != len(set(factors)):
                raise ScenarioDefinitionError(
                    f"scenario {name!r}: duplicate factors are not allowed"
                )
            for s in definition.shocks:
                if not math.isfinite(s.shock):
                    raise ScenarioDefinitionError(
                        f"scenario {name!r}: shock for {s.factor!r} must be finite"
                    )
        else:
            if not definition.components:
                raise ScenarioDefinitionError(
                    f"composite scenario {name!r} must list at least one component"
                )
            for comp in definition.components:
                if comp == name:
                    raise ScenarioDefinitionError(
                        f"scenario {name!r} cannot include itself"
                    )
                if comp not in self._definitions:
                    raise ScenarioDefinitionError(
                        f"scenario {name!r} references unknown component {comp!r}"
                    )
                if self._definitions[comp].type.value != "factor_shock":
                    raise ScenarioDefinitionError(
                        f"component {comp!r} must be a factor_shock scenario "
                        "(nested composites are not allowed)"
                    )
        self._definitions[name] = definition

    def get(self, name: str) -> ScenarioDefinition:
        if name not in self._definitions:
            raise ScenarioNotFoundError(f"scenario {name!r} is not defined")
        return self._definitions[name]

    def list_scenarios(self) -> list[str]:
        return sorted(self._definitions)

    # --- evaluation ----------------------------------------------------------

    def resolve_shocks(self, name: str) -> dict[str, float]:
        """Return the effective per-factor shock vector for a named scenario."""
        d = self.get(name)
        if d.type.value == "factor_shock":
            return {s.factor: s.shock for s in d.shocks}
        merged: dict[str, float] = {}
        for comp in d.components:
            for factor, shock in self.resolve_shocks(comp).items():
                merged[factor] = merged.get(factor, 0.0) + shock
        return merged

    def evaluate(
        self,
        name: str,
        portfolio: Portfolio,
        factor_map: dict[str, dict[str, float]] | None = None,
    ) -> ScenarioResult:
        if not portfolio.positions:
            raise EmptyPortfolioError("portfolio contains no positions")

        effective_map = {**self._factor_map}
        if factor_map:
            effective_map.update(factor_map)

        shocks = self.resolve_shocks(name)

        # position market values are provided through quantity * unit value;
        # stress scenarios operate on current market value, so quantity here
        # is interpreted directly as monetary exposure per symbol.
        exposures: dict[str, float] = {}
        for pos in portfolio.positions:
            if not math.isfinite(pos.quantity):
                raise InvalidPositionError(
                    f"position in {pos.symbol!r} must be finite"
                )
            exposures[pos.symbol] = exposures.get(pos.symbol, 0.0) + pos.quantity
        base_value = sum(exposures.values())
        if not math.isfinite(base_value) or base_value <= 0.0:
            raise InvalidPositionError(
                "portfolio base value must be finite and positive"
            )

        # Every shocked factor must be traceable to at least one position.
        loadings_by_symbol = {
            sym: effective_map.get(sym, {sym: 1.0}) for sym in exposures
        }
        covered: set[str] = set()
        for ld in loadings_by_symbol.values():
            covered.update(ld)
        unknown = sorted(set(shocks) - covered)
        if unknown:
            raise ScenarioDefinitionError(
                f"scenario {name!r}: factors {unknown} are not mapped to any position"
            )
        def symbol_return(symbol: str, factor: str | None) -> float:
            ld = loadings_by_symbol[symbol]
            if factor is None:
                return sum(l * shocks.get(f, 0.0) for f, l in ld.items())
            return ld.get(factor, 0.0) * shocks.get(factor, 0.0)

        contributions: dict[str, float] = {}
        for factor in shocks:
            pnl = sum(expo * symbol_return(sym, factor) for sym, expo in exposures.items())
            contributions[factor] = pnl

        stressed_value = sum(
            expo * (1.0 + symbol_return(sym, None)) for sym, expo in exposures.items()
        )
        pnl = stressed_value - base_value
        # identity check: contributions must reconstruct the total
        reconstructed = sum(contributions.values())
        if not math.isclose(reconstructed, pnl, rel_tol=1e-12, abs_tol=1e-9):
            raise ScenarioDefinitionError(
                f"internal consistency failed: contributions {reconstructed} "
                f"!= total pnl {pnl}"
            )

        return ScenarioResult(
            scenario=name,
            base_value=float(base_value),
            stressed_value=float(stressed_value),
            pnl=float(pnl),
            pnl_pct=float(pnl / base_value),
            factor_contributions={f: float(v) for f, v in contributions.items()},
        )
