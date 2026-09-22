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
from risk_engine.risk import build_report
from risk_engine.scenarios import (
    compose_scenarios,
    evaluate_scenario,
)
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
