"""HTTP routes: portfolio x scenario scan, snapshots, cross-version recompare."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from risk_engine.api.mappers import (
    to_historical_conclusion,
    to_market,
    to_portfolio,
    to_scan_cell,
)
from risk_engine.api.scan_schemas import (
    CrossProductScanRequest,
    RecompareRequest,
    ScanRequest,
)
from risk_engine.errors import RiskEngineError
from risk_engine.recompare import RecompareLimits
from risk_engine.scan import ScanLimits, cross_product_cells
from risk_engine.snapshot import build_snapshot

logger = logging.getLogger("risk_engine.api")

router = APIRouter(prefix="/api/risk", tags=["risk"])


def _service(request: Request):
    return request.app.state.risk_service


def _error_response(err: RiskEngineError, status_code: int = 422):
    from fastapi.responses import JSONResponse

    logger.warning("scan request rejected: code=%s reason=%s", err.code, err.reason)
    return JSONResponse(
        status_code=status_code,
        content={"error": err.to_dict()},
    )


def _http_status(err: RiskEngineError) -> int:
    from risk_engine.errors import ComputationError, ResourceLimitError

    if isinstance(err, ResourceLimitError):
        return 429
    if isinstance(err, ComputationError):
        return 500
    return 422


def _scan_limits(payload) -> ScanLimits:
    defaults = ScanLimits()
    return ScanLimits(
        max_items=payload.max_items or defaults.max_items,
        max_workers=payload.max_workers or defaults.max_workers,
        timeout_seconds=payload.timeout_seconds or defaults.timeout_seconds,
    )


def _slot_dict(slot) -> dict:
    return {
        "index": slot.index,
        "ok": slot.ok,
        "result": slot.result,
        "error": slot.error,
    }


@router.post("/scan")
def scan(payload: CrossProductScanRequest, request: Request):
    """Batch scan: many portfolios x many registered scenarios."""
    svc = _service(request)
    logger.info(
        "scan portfolios=%d scenarios=%d",
        len(payload.portfolios), len(payload.scenario_names),
    )
    market = to_market(payload.market)
    portfolios = {
        p.name: (to_portfolio(p), market) for p in payload.portfolios
    }
    try:
        cells = cross_product_cells(
            portfolios,
            payload.scenario_names,
            correlation_name=payload.correlation_name,
            version=payload.version,
        )
        results = svc.scan(cells, _scan_limits(payload))
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return {
        "count": len(results),
        "results": [_slot_dict(s) for s in results],
    }


@router.post("/scan/snapshot")
def scan_snapshot(payload: ScanRequest, request: Request):
    """Run an explicit-cell scan and return a standalone HTML snapshot."""
    svc = _service(request)
    try:
        cells = [to_scan_cell(c) for c in payload.cells]
        results = svc.scan(cells, _scan_limits(payload))
        snapshot = build_snapshot(results, title=payload.title)
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return HTMLResponse(content=snapshot.html, media_type="text/html")


@router.post("/recompare")
def recompare(payload: RecompareRequest, request: Request):
    """Batch cross-version recomparison of historical conclusions."""
    svc = _service(request)
    logger.info(
        "recompare items=%d target=%s allow_breaking=%s",
        len(payload.items), payload.target_version, payload.allow_breaking,
    )
    defaults = RecompareLimits()
    limits = RecompareLimits(
        max_items=payload.max_items or defaults.max_items,
        max_workers=payload.max_workers or defaults.max_workers,
        timeout_seconds=payload.timeout_seconds or defaults.timeout_seconds,
    )
    items = [to_historical_conclusion(i) for i in payload.items]
    try:
        results = svc.recompare_conclusions(
            items,
            target_version=payload.target_version,
            allow_breaking=payload.allow_breaking,
            limits=limits,
        )
    except RiskEngineError as err:
        return _error_response(err, _http_status(err))
    return {
        "count": len(results),
        "results": [
            {
                "index": r.index,
                "portfolio": r.portfolio,
                "stamped_version": r.stamped_version,
                "target_version": r.target_version,
                "verdict": r.verdict,
                "reason": r.reason,
                "old_conclusion": r.old_conclusion,
                "new_conclusion": r.new_conclusion,
                "error": r.error,
            }
            for r in results
        ],
    }
