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
    ScenarioEvaluateRequest,
    ScenarioSchema,
)
from risk_engine.api.schemas import AssessRequest, BatchAssessRequest
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
        report = svc.assess(
            to_portfolio(payload.portfolio),
            to_market(payload.market),
            window=payload.window,
            confidence=payload.confidence,
            horizon_days=payload.horizon_days,
            correlation=to_correlation(payload.correlation) if payload.correlation else None,
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
    try:
        results = svc.batch_assess([to_job(r) for r in payload.requests], limits)
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return {
        "count": len(results),
        "results": [
            {
                "index": r.index,
                "ok": r.ok,
                "result": r.result,
                "error": r.error,
            }
            for r in results
        ],
    }


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
