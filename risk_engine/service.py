"""High-level facade combining risk, scenarios, versions and batch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from risk_engine.batch import BatchLimits, run_batch
from risk_engine.correlation import combine_block_diagonal
from risk_engine.errors import (
    IncompatibleVersionError,
    UnknownAssumptionError,
    UnknownScenarioError,
)
from risk_engine.models import (
    BatchItemResult,
    CorrelationAssumption,
    MarketData,
    Portfolio,
    RiskReport,
    ScenarioResult,
    StressScenario,
)
from risk_engine.recompare import (
    HistoricalConclusion,
    RecompareItem,
    RecompareLimits,
    recompare,
)
from risk_engine.risk import build_report
from risk_engine.scenarios import (
    compose_scenarios,
    evaluate_scenario,
)
from risk_engine.scan import ScanCell, ScanLimits, cross_product_cells, scan_cells
from risk_engine.snapshot import Snapshot, build_snapshot
from risk_engine.versions import RulesVersion, VersionRegistry, default_registry


@dataclass(frozen=True)
class AssessmentJob:
    portfolio: Portfolio
    market: MarketData
    window: int = 20
    confidence: float = 0.99
    horizon_days: int = 1
    correlation: CorrelationAssumption | None = None
    version: str = "v1"


class RiskService:
    def __init__(self, registry: VersionRegistry | None = None) -> None:
        self._versions = registry or default_registry()
        self._scenarios: dict[str, StressScenario] = {}
        self._correlations: dict[str, CorrelationAssumption] = {}

    # --- versioning -------------------------------------------------------

    def register_version(self, rules: RulesVersion) -> None:
        self._versions.register(rules)

    @property
    def versions(self) -> VersionRegistry:
        return self._versions

    # --- named scenarios / assumptions -----------------------------------

    def register_scenario(self, scenario: StressScenario, *, overwrite: bool = False) -> None:
        if scenario.name in self._scenarios and not overwrite:
            raise ValueError(f"scenario '{scenario.name}' already registered")
        self._scenarios[scenario.name] = scenario

    def get_scenario(self, name: str) -> StressScenario:
        try:
            return self._scenarios[name]
        except KeyError:
            raise UnknownScenarioError(f"scenario '{name}' is not registered")

    def register_correlation(self, corr: CorrelationAssumption, *, overwrite: bool = False) -> None:
        if corr.name in self._correlations and not overwrite:
            raise ValueError(f"correlation '{corr.name}' already registered")
        self._correlations[corr.name] = corr

    def get_correlation(self, name: str) -> CorrelationAssumption:
        try:
            return self._correlations[name]
        except KeyError:
            raise UnknownAssumptionError(
                f"correlation assumption '{name}' is not registered"
            )

    def compose_correlation(
        self, result_name: str, names: list[str], *, version: str = "v1"
    ) -> CorrelationAssumption:
        assumptions = [self.get_correlation(n) for n in names]
        composed = combine_block_diagonal(
            result_name, assumptions, version=version
        )
        self.register_correlation(composed)
        return composed

    def compose_named(
        self, result_name: str, names: list[str], *, version: str = "v1"
    ) -> StressScenario:
        scenarios = tuple(self.get_scenario(n) for n in names)
        composed = compose_scenarios(result_name, *scenarios, version=version)
        self.register_scenario(composed)
        return composed

    # --- risk -------------------------------------------------------------

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
        self._versions.get(version)  # unknown version rejected
        return build_report(
            portfolio,
            market,
            window=window,
            confidence=confidence,
            horizon_days=horizon_days,
            correlation=correlation,
            version=version,
        )

    def reassess_under_version(
        self,
        portfolio: Portfolio,
        market: MarketData,
        *,
        stamped_version: str,
        target_version: str,
        window: int,
        confidence: float = 0.99,
        horizon_days: int = 1,
        correlation: CorrelationAssumption | None = None,
        allow_breaking: bool = False,
    ) -> RiskReport:
        """Reproduce an old conclusion under newer rules.

        Refuses (IncompatibleVersionError) unless ``target`` declares the
        old version reproducible or ``allow_breaking`` is explicitly set.
        """
        self._versions.get(stamped_version)
        self._versions.get(target_version)
        self._versions.check_compatible(
            stamped_version, target_version, allow_breaking=allow_breaking
        )
        return self.assess(
            portfolio,
            market,
            window=window,
            confidence=confidence,
            horizon_days=horizon_days,
            correlation=correlation,
            version=target_version,
        )

    # --- scenarios --------------------------------------------------------

    def run_scenario(
        self,
        portfolio: Portfolio,
        market: MarketData,
        scenario: StressScenario,
        correlation: CorrelationAssumption | None = None,
    ) -> ScenarioResult:
        return evaluate_scenario(portfolio, market, scenario, correlation)

    # --- batch ------------------------------------------------------------

    def batch_assess(
        self, jobs: list[AssessmentJob], limits: BatchLimits | None = None
    ) -> list[BatchItemResult]:
        return run_batch(jobs, self._run_job, limits)

    def _run_job(self, job: AssessmentJob) -> dict:
        report = self.assess(
            job.portfolio,
            job.market,
            window=job.window,
            confidence=job.confidence,
            horizon_days=job.horizon_days,
            correlation=job.correlation,
            version=job.version,
        )
        return report_to_dict(report)

    # --- batch scan: portfolios x registered scenarios --------------------

    def scan(
        self,
        cells: list[ScanCell],
        limits: ScanLimits | None = None,
    ) -> list[BatchItemResult]:
        """Scan (portfolio, registered-scenario) cells with isolated errors."""
        return scan_cells(cells, self._run_scan_cell, limits=limits)

    def cross_product_scan(
        self,
        portfolios: dict[str, tuple[Portfolio, MarketData]],
        scenario_names: list[str],
        *,
        correlation_name: str | None = None,
        version: str = "v1",
        limits: ScanLimits | None = None,
    ) -> list[BatchItemResult]:
        """Convenience: expand portfolios x scenarios then scan in order."""
        cells = cross_product_cells(
            portfolios,
            scenario_names,
            correlation_name=correlation_name,
            version=version,
        )
        return self.scan(cells, limits)

    def export_scan_snapshot(
        self,
        cells: list[ScanCell],
        *,
        title: str = "risk scan snapshot",
        limits: ScanLimits | None = None,
    ) -> Snapshot:
        """Run a scan and package it as a deterministic snapshot."""
        results = self.scan(cells, limits)
        return build_snapshot(results, title=title)

    def _run_scan_cell(self, cell: ScanCell) -> dict:
        # Same code path as a single run_scenario call: resolve names,
        # then evaluate, so batch and single results always agree.
        scenario = self.get_scenario(cell.scenario_name)
        correlation: CorrelationAssumption | None = None
        if cell.correlation_name is not None:
            correlation = self.get_correlation(cell.correlation_name)
        result = evaluate_scenario(
            cell.portfolio, cell.market, scenario, correlation
        )
        payload = scenario_to_dict(result)
        payload["input"] = self._scan_input_summary(cell)
        payload["rationale"] = self._scenario_rationale(result)
        return payload

    @staticmethod
    def _scenario_rationale(result: ScenarioResult) -> str:
        direction = "loss" if result.pnl < 0 else "profit"
        return (
            f"linear first-order scenario PnL={result.pnl:.6g} "
            f"({direction}); single-factor self-consistency="
            f"{result.consistent_with_single_factor}"
        )

    @staticmethod
    def _scan_input_summary(cell: ScanCell) -> dict:
        return {
            "portfolio": cell.portfolio.name,
            "holdings": len(cell.portfolio.holdings),
            "scenario": cell.scenario_name,
            "correlation": cell.correlation_name,
            "dates": len(cell.market.dates),
            "version": cell.version,
        }

    # --- cross-version recomparison ---------------------------------------

    def recompare_conclusions(
        self,
        items: list[HistoricalConclusion],
        *,
        target_version: str,
        allow_breaking: bool = False,
        limits: RecompareLimits | None = None,
    ) -> list[RecompareItem]:
        """Batch recompute historical conclusions under a target version."""
        return recompare(
            items,
            self._recompute_gated,
            target_version=target_version,
            allow_breaking=allow_breaking,
            limits=limits,
        )

    def _recompute_gated(
        self,
        item: HistoricalConclusion,
        target_version: str,
        allow_breaking: bool,
    ) -> dict:
        # The compatibility gate refuses breaking upgrades unless the
        # caller explicitly approves, so an old conclusion can never be
        # silently replaced under changed rules.
        self._versions.check_compatible(
            item.stamped_version,
            target_version,
            allow_breaking=allow_breaking,
        )
        report = self.assess(
            item.portfolio,
            item.market,
            window=item.window,
            confidence=item.confidence,
            horizon_days=item.horizon_days,
            version=target_version,
        )
        return report_to_dict(report)


def _iso(d: date) -> str:
    return d.isoformat()


def _estimate_dict(e) -> dict:
    return {
        "confidence": e.confidence,
        "horizon_days": e.horizon_days,
        "var": e.var,
        "expected_shortfall": e.expected_shortfall,
    }


def report_to_dict(report: RiskReport) -> dict:
    return {
        "portfolio": report.portfolio,
        "version": report.version,
        "window": report.window,
        "as_of": _iso(report.as_of),
        "historical": _estimate_dict(report.historical),
        "parametric": _estimate_dict(report.parametric),
        "portfolio_volatility": report.portfolio_volatility,
        "gross_exposure": report.gross_exposure,
        "net_exposure": report.net_exposure,
        "exposure_by_instrument": dict(report.exposure_by_instrument),
    }


def scenario_to_dict(result: ScenarioResult) -> dict:
    return {
        "scenario": result.scenario,
        "version": result.version,
        "pnl": result.pnl,
        "pnl_by_instrument": dict(result.pnl_by_instrument),
        "factor_moves": dict(result.factor_moves),
        "consistent_with_single_factor": result.consistent_with_single_factor,
    }
