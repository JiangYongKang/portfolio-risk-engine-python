"""HTTP routes exposing risk assessment and batch evaluation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from risk_engine.errors import (
    ComputationError,
    ResourceLimitError,
    RiskEngineError,
)
from risk_engine.api.mappers import to_correlation, to_job, to_market, to_portfolio
from risk_engine.api.scenarios_schemas import (
    ComposeSchema,
    CorrelationComposeSchema,
    ScenarioEvaluateRequest,
    ScenarioSchema,
)
from risk_engine.api.schemas import (
    AssessRequest,
    BatchAssessRequest,
    CorrelationSchema,
    ReassessRequest,
    ScanRequest,
)
from risk_engine.models import FactorShock, StressScenario
from risk_engine.service import report_to_dict, scenario_to_dict

logger = logging.getLogger("risk_engine.api")

router = APIRouter(prefix="/api/risk", tags=["risk"])


def _service(request: Request):
    return request.app.state.risk_service


def _error_response(err: RiskEngineError, status_code: int = 422) -> JSONResponse:
    logger.warning("risk request rejected: code=%s reason=%s", err.code, err.reason)
    return JSONResponse(
        status_code=status_code,
        content={"error": err.to_dict()},
    )


def _http_status(err: RiskEngineError) -> int:
    if isinstance(err, ResourceLimitError):
        return 429
    if isinstance(err, ComputationError):
        return 500
    return 422


def _resolve_correlation(svc, payload):
    """Inline matrix wins over a registered name; both given is rejected."""
    if payload.correlation is not None and payload.correlation_name:
        raise RiskEngineError(
            "provide either 'correlation' or 'correlation_name', not both",
            code="INVALID_PARAMETER",
        )
    if payload.correlation is not None:
        return to_correlation(payload.correlation)
    if payload.correlation_name:
        return svc.get_correlation(payload.correlation_name)
    return None


def _summary(payload) -> dict:
    """Compact input summary for structured logs."""
    if isinstance(payload, AssessRequest):
        return {
            "type": "assess",
            "portfolio": payload.portfolio.name,
            "holdings": len(payload.portfolio.holdings),
            "dates": len(payload.market.dates),
            "window": payload.window,
            "confidence": payload.confidence,
            "horizon_days": payload.horizon_days,
            "version": payload.version,
            "correlation": payload.correlation.name if payload.correlation else None,
        }
    if isinstance(payload, ScenarioEvaluateRequest):
        return {
            "type": "scenario",
            "portfolio": payload.portfolio.name,
            "scenario": payload.scenario.name if payload.scenario else payload.scenario_name,
            "shocks": len(payload.scenario.shocks) if payload.scenario else None,
        }
    return {"type": type(payload).__name__}


@router.post("/assess")
def assess(payload: AssessRequest, request: Request):
    logger.info("assess request summary=%s", _summary(payload))
    svc = _service(request)
    try:
        correlation = _resolve_correlation(svc, payload)
        report = svc.assess(
            to_portfolio(payload.portfolio),
            to_market(payload.market),
            window=payload.window,
            confidence=payload.confidence,
            horizon_days=payload.horizon_days,
            correlation=correlation,
            version=payload.version,
        )
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return report_to_dict(report)


@router.post("/batch")
def batch(payload: BatchAssessRequest, request: Request):
    logger.info(
        "batch request size=%d max_items=%s timeout=%s",
        len(payload.requests), payload.max_items, payload.timeout_seconds,
    )
    svc = _service(request)
    from risk_engine.batch import BatchLimits

    limits = BatchLimits(
        max_items=payload.max_items or BatchLimits().max_items,
        timeout_seconds=payload.timeout_seconds or BatchLimits().timeout_seconds,
    )
    from risk_engine.models import BatchItemResult

    # Resolve each request independently so one bad reference does not
    # abort the batch; resolution failures occupy their own ordered slot.
    jobs = []
    pre_failures: dict[int, dict] = {}
    for i, req in enumerate(payload.requests):
        try:
            jobs.append(to_job(req, _resolve_correlation(svc, req)))
        except RiskEngineError as err:
            jobs.append(None)
            pre_failures[i] = err.to_dict()
    try:
        results = svc.batch_assess([j for j in jobs if j is not None], limits)
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))

    # Re-insert pre-resolution failures at their original indices.
    merged: list = []
    k = 0
    for i in range(len(payload.requests)):
        if i in pre_failures:
            merged.append(
                {"index": i, "ok": False, "result": None, "error": pre_failures[i]}
            )
        else:
            r = results[k]
            k += 1
            merged.append(
                {"index": i, "ok": r.ok, "result": r.result, "error": r.error}
            )
    return {
        "count": len(merged),
        "results": merged,
    }


@router.post("/scan")
def scan(payload: ScanRequest, request: Request):
    """Batch scan: portfolios x registered scenarios, one ordered batch."""
    logger.info(
        "scan request jobs=%d pairs=%d max_items=%s timeout=%s",
        len(payload.jobs),
        sum(len(j.scenario_names) for j in payload.jobs),
        payload.max_items,
        payload.timeout_seconds,
    )
    svc = _service(request)
    from risk_engine.batch import BatchLimits
    from risk_engine.scan import ScanJob

    limits = BatchLimits(
        max_items=payload.max_items or BatchLimits().max_items,
        timeout_seconds=payload.timeout_seconds or BatchLimits().timeout_seconds,
    )
    jobs = [
        ScanJob(
            portfolio=to_portfolio(j.portfolio),
            market=to_market(j.market),
            scenario_names=tuple(j.scenario_names),
        )
        for j in payload.jobs
    ]
    try:
        results = svc.scan(jobs, limits)
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return {
        "count": len(results),
        "results": [
            {"index": r.index, "ok": r.ok, "result": r.result, "error": r.error}
            for r in results
        ],
    }


@router.post("/scan/snapshot")
def scan_snapshot(payload: ScanRequest, request: Request):
    """Run a scan and return a deterministic, ranked result snapshot."""
    logger.info("scan snapshot request jobs=%d", len(payload.jobs))
    svc = _service(request)
    from risk_engine.batch import BatchLimits
    from risk_engine.scan import ScanJob

    limits = BatchLimits(
        max_items=payload.max_items or BatchLimits().max_items,
        timeout_seconds=payload.timeout_seconds or BatchLimits().timeout_seconds,
    )
    jobs = [
        ScanJob(
            portfolio=to_portfolio(j.portfolio),
            market=to_market(j.market),
            scenario_names=tuple(j.scenario_names),
        )
        for j in payload.jobs
    ]
    try:
        return svc.scan_snapshot(jobs, limits)
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))


@router.post("/reassess")
def reassess(payload: ReassessRequest, request: Request):
    """Batch cross-version reassessment with per-item verdicts."""
    logger.info(
        "reassess request jobs=%d targets=%s",
        len(payload.jobs),
        sorted({j.target_version for j in payload.jobs}),
    )
    svc = _service(request)
    from risk_engine.batch import BatchLimits
    from risk_engine.reassess import ReassessJob

    limits = BatchLimits(
        max_items=payload.max_items or BatchLimits().max_items,
        timeout_seconds=payload.timeout_seconds or BatchLimits().timeout_seconds,
    )
    # Resolve each job independently; a bad reference fails only its slot.
    jobs = []
    pre_failures: dict[int, dict] = {}
    for i, j in enumerate(payload.jobs):
        try:
            correlation = _resolve_correlation(svc, j)
            jobs.append(
                ReassessJob(
                    portfolio=to_portfolio(j.portfolio),
                    market=to_market(j.market),
                    stamped_version=j.stamped_version,
                    target_version=j.target_version,
                    original=j.original,
                    window=j.window,
                    confidence=j.confidence,
                    horizon_days=j.horizon_days,
                    correlation=correlation,
                    allow_breaking=j.allow_breaking,
                )
            )
        except RiskEngineError as err:
            jobs.append(None)
            pre_failures[i] = err.to_dict()
    try:
        results = svc.batch_reassess([j for j in jobs if j is not None], limits)
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))

    merged: list = []
    k = 0
    for i in range(len(payload.jobs)):
        if i in pre_failures:
            merged.append(
                {"index": i, "ok": False, "result": None, "error": pre_failures[i]}
            )
        else:
            r = results[k]
            k += 1
            merged.append(
                {"index": i, "ok": r.ok, "result": r.result, "error": r.error}
            )
    return {"count": len(merged), "results": merged}


@router.post("/scenarios")
def register_scenario(payload: ScenarioSchema, request: Request):
    svc = _service(request)
    scenario = StressScenario(
        name=payload.name,
        shocks=tuple(
            FactorShock(instrument=s.instrument, relative_move=s.relative_move)
            for s in payload.shocks
        ),
        version=payload.version,
    )
    try:
        svc.register_scenario(scenario)
    except ValueError as err:
        return _error_response(RiskEngineError(str(err), code="SCENARIO_EXISTS"), 409)
    return {"registered": scenario.name, "shocks": len(scenario.shocks)}


@router.post("/scenarios/compose")
def compose(payload: ComposeSchema, request: Request):
    svc = _service(request)
    try:
        composed = svc.compose_named(
            payload.result_name, payload.names, version=payload.version
        )
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return {"registered": composed.name, "names": payload.names,
            "shocks": len(composed.shocks)}


@router.post("/scenarios/evaluate")
def evaluate(payload: ScenarioEvaluateRequest, request: Request):
    logger.info("scenario request summary=%s", _summary(payload))
    svc = _service(request)
    try:
        if payload.scenario is not None:
            scenario = StressScenario(
                name=payload.scenario.name,
                shocks=tuple(
                    FactorShock(s.instrument, s.relative_move)
                    for s in payload.scenario.shocks
                ),
                version=payload.scenario.version,
            )
        elif payload.scenario_name is not None:
            scenario = svc.get_scenario(payload.scenario_name)
        else:
            return _error_response(
                RiskEngineError(
                    "provide either 'scenario' or 'scenario_name'",
                    code="MISSING_SCENARIO",
                ),
                422,
            )
        result = svc.run_scenario(
            to_portfolio(payload.portfolio),
            to_market(payload.market),
            scenario,
            to_correlation(payload.correlation) if payload.correlation else None,
        )
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return scenario_to_dict(result)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/correlations")
def register_correlation(payload: CorrelationSchema, request: Request):
    svc = _service(request)
    assumption = to_correlation(payload)
    try:
        # validate eagerly so an invalid matrix is rejected at registration
        from risk_engine.correlation import validate_matrix

        validate_matrix(
            [list(r) for r in assumption.matrix], assumption.instruments
        )
        svc.register_correlation(assumption)
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    except ValueError:
        return _error_response(
            RiskEngineError(
                f"correlation '{payload.name}' already registered",
                code="ASSUMPTION_EXISTS",
            ),
            409,
        )
    return {"registered": assumption.name, "instruments": len(assumption.instruments)}


@router.post("/correlations/compose")
def compose_correlations(payload: "CorrelationComposeSchema", request: Request):
    svc = _service(request)
    try:
        composed = svc.compose_correlation(
            payload.result_name, payload.names, version=payload.version
        )
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return {
        "registered": composed.name,
        "instruments": list(composed.instruments),
        "from": payload.names,
    }
