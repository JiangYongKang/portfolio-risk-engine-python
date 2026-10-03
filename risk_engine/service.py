"""High-level facade combining risk, scenarios, versions and batch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from risk_engine.batch import BatchLimits, run_batch
from risk_engine.correlation import combine_block_diagonal
from risk_engine.errors import (
    IncompatibleVersionError,
    RiskEngineError,
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
from risk_engine.reassess import (
    VERDICT_CHANGED,
    VERDICT_REJECTED,
    VERDICT_REPRODUCED,
    ReassessJob,
    conclusions_equal,
)
from risk_engine.risk import build_report
from risk_engine.scenarios import (
    compose_scenarios,
    evaluate_scenario,
)
from risk_engine.scan import ScanJob, expand_pairs
from risk_engine.snapshot import SnapshotEntry, build_snapshot, rank_worst
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

    # --- cross-version reassessment ---------------------------------------

    def reassess_one(self, job: ReassessJob) -> dict:
        """Recompute one stamped conclusion under a target version.

        Never raises for expected refusals: version/input problems become
        a ``REJECTED`` verdict carrying the typed error.
        """
        try:
            report = self.reassess_under_version(
                job.portfolio,
                job.market,
                stamped_version=job.stamped_version,
                target_version=job.target_version,
                window=job.window,
                confidence=job.confidence,
                horizon_days=job.horizon_days,
                correlation=job.correlation,
                allow_breaking=job.allow_breaking,
            )
        except RiskEngineError as err:
            return {
                "verdict": VERDICT_REJECTED,
                "portfolio": job.portfolio.name,
                "stamped_version": job.stamped_version,
                "target_version": job.target_version,
                "error": err.to_dict(),
                "recomputed": None,
            }
        recomputed = report_to_dict(report)
        verdict = (
            VERDICT_REPRODUCED
            if conclusions_equal(job.original, recomputed)
            else VERDICT_CHANGED
        )
        return {
            "verdict": verdict,
            "portfolio": job.portfolio.name,
            "stamped_version": job.stamped_version,
            "target_version": job.target_version,
            "error": None,
            "recomputed": recomputed,
        }

    def batch_reassess(
        self, jobs: list[ReassessJob], limits: BatchLimits | None = None
    ) -> list[BatchItemResult]:
        """Batch cross-version reassessment: ordered, isolated verdicts."""
        return run_batch(jobs, self.reassess_one, limits)

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

    def scan(
        self, jobs: list[ScanJob], limits: BatchLimits | None = None
    ) -> list[BatchItemResult]:
        """Scan portfolios x registered scenarios as one ordered batch.

        Results are flattened in submission order (portfolio-major).  An
        unknown scenario name or a failing pair occupies only its own
        slot; every other pair is unaffected.
        """
        pairs = expand_pairs(jobs)
        return run_batch(pairs, self._run_scan_pair, limits)

    def scan_snapshot(
        self, jobs: list[ScanJob], limits: BatchLimits | None = None
    ) -> dict:
        """Run a scan and freeze the results into a ranked snapshot.

        The returned document is deterministic: serialize it with
        ``risk_engine.snapshot.canonical_json`` for byte-stable export.
        """
        pairs = expand_pairs(jobs)
        results = run_batch(pairs, self._run_scan_pair, limits)
        entries: list[SnapshotEntry] = []
        for (portfolio, market, scenario_name), item in zip(pairs, results):
            summary = {
                "portfolio": portfolio.name,
                "scenario": scenario_name,
                "holdings": len(portfolio.holdings),
                "dates": len(market.dates),
            }
            if item.ok:
                version = item.result["version"]
                rationale = (
                    f"scenario '{scenario_name}' evaluated under rules "
                    f"{version} on portfolio '{portfolio.name}': "
                    f"pnl={item.result['pnl']!r}; "
                    f"consistent_with_single_factor="
                    f"{item.result['consistent_with_single_factor']}"
                )
            else:
                version = ""
                rationale = (
                    f"rejected: {item.error['code']} — {item.error['reason']}"
                )
            entries.append(
                SnapshotEntry(
                    index=item.index,
                    kind="scenario",
                    input_summary=summary,
                    version=version,
                    ok=item.ok,
                    rationale=rationale,
                    result=item.result,
                    error=item.error,
                )
            )
        rankings = {
            "worst_by_portfolio": rank_worst(entries, group_by="portfolio"),
            "worst_by_scenario": rank_worst(entries, group_by="scenario"),
        }
        return build_snapshot(entries, rankings=rankings)

    def _run_scan_pair(self, pair: tuple) -> dict:
        portfolio, market, scenario_name = pair
        scenario = self.get_scenario(scenario_name)  # unknown -> per-slot error
        result = evaluate_scenario(portfolio, market, scenario)
        payload = scenario_to_dict(result)
        payload["portfolio"] = portfolio.name
        return payload

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
