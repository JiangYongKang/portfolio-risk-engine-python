"""HTTP routes exposing risk assessment and batch evaluation."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from risk_engine.errors import RiskEngineError
from risk_engine.api.schemas import AssessRequest, BatchAssessRequest

router = APIRouter(prefix="/api/risk", tags=["risk"])


def _service(request: Request):
    return request.app.state.risk_service


def _error_response(err: RiskEngineError) -> JSONResponse:
    raise NotImplementedError


@router.post("/assess")
def assess(payload: AssessRequest, request: Request) -> dict:
    raise NotImplementedError


@router.post("/batch")
def batch(payload: BatchAssessRequest, request: Request) -> dict:
    raise NotImplementedError


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
