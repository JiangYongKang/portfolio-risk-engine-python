"""FastAPI application entry point.

HTTP is a thin transport layer: all domain failures are
:class:`~app.errors.RiskEngineError` subclasses rendered uniformly as
``{"error": {"reason", "detail"}}`` so clients can distinguish causes by a
stable code; raw numeric/implementation exceptions never escape.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.errors import RiskEngineError
from app.models import (
    BatchItem,
    ConfidenceLevel,
    CorrelationOverride,
    Portfolio,
    PriceSeries,
    ScenarioDefinition,
    VersionedParameterSet,
)
from app.service import build_app_state
from app.versioning import ScenarioSnapshot


# --- request wrappers -------------------------------------------------------------

class RiskRequest(BaseModel):
    portfolio: Portfolio
    confidence: ConfidenceLevel
    correlation_override: list[list[float]] | None = Field(
        default=None,
        description="optional NxN correlation matrix, N = number of distinct "
        "portfolio symbols; replaces empirical correlations for parametric VaR/ES",
    )


class ScenarioEvaluateRequest(BaseModel):
    portfolio: Portfolio
    factor_map: dict[str, dict[str, float]] | None = None


class BatchRequest(BaseModel):
    items: list[BatchItem]
    timeout: float = Field(default=10.0, gt=0)
    max_items: int = Field(default=128, gt=0)


class RegisterVersionRequest(BaseModel):
    parameter_set: VersionedParameterSet
    scenario_name: str
    portfolio: Portfolio
    factor_map: dict[str, dict[str, float]] | None = None


class EvaluateVersionRequest(BaseModel):
    accept_incompatible: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="Portfolio Risk Engine", version="0.1.0")
    state = build_app_state()
    app.state.engine = state

    @app.exception_handler(RiskEngineError)
    async def _domain_error_handler(_: Request, exc: RiskEngineError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": {"reason": exc.reason, "detail": exc.detail}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"reason": "invalid_request",
                               "detail": "request payload failed validation",
                               "issues": exc.errors()}},
        )

    @app.exception_handler(Exception)
    async def _unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # log server-side for diagnosis, but never leak internals to the caller
        logging.getLogger("portfolio-risk-engine").exception(
            "unexpected error on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={"error": {"reason": "internal_error",
                               "detail": "an unexpected internal error occurred"}},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/market/prices", status_code=201)
    def load_prices(series: PriceSeries) -> dict:
        state.store.load(series)
        return {"loaded": series.symbol, "observations": len(series.prices)}

    @app.post("/risk")
    def risk(req: RiskRequest) -> dict:
        return state.risk(
            req.portfolio,
            req.confidence.window,
            req.confidence.confidence,
            req.correlation_override,
        ).model_dump()

    @app.post("/scenarios", status_code=201)
    def define_scenario(definition: ScenarioDefinition) -> dict:
        state.scenarios.define(definition)
        return {"defined": definition.name, "type": definition.type.value}

    @app.post("/scenarios/{name}/evaluate")
    def evaluate_scenario(name: str, req: ScenarioEvaluateRequest) -> dict:
        return state.scenarios.evaluate(
            name, req.portfolio, req.factor_map
        ).model_dump()

    @app.post("/correlation/validate")
    def validate_correlation(override: CorrelationOverride) -> dict:
        matrix = state.apply_correlation_override(override)
        return {"factors": override.factors, "matrix": matrix, "accepted": True}

    @app.post("/batch")
    def batch(req: BatchRequest) -> dict:
        entries = state.run_batch(req.items, req.timeout, req.max_items)
        return {
            "count": len(entries),
            "results": [e.model_dump() for e in entries],
        }

    @app.post("/parameters", status_code=201)
    def register_version(req: RegisterVersionRequest) -> dict:
        snapshot = ScenarioSnapshot(
            scenario_name=req.scenario_name,
            portfolio=req.portfolio,
            factor_map=req.factor_map or {},
        )
        state.registry.register(req.parameter_set, snapshot)
        ps = req.parameter_set
        return {"registered": ps.name, "version": ps.version, "rule_version": ps.rule_version}

    @app.post("/parameters/{name}/versions/{version}/evaluate")
    def evaluate_version(name: str, version: int, req: EvaluateVersionRequest) -> dict:
        return state.registry.evaluate_versioned(
            name, version, accept_incompatible=req.accept_incompatible
        ).model_dump()

    return app


app = create_app()
