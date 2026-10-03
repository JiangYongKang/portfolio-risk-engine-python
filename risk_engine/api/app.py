"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from risk_engine.api.routes import router
from risk_engine.api.scan_routes import router as scan_router
from risk_engine.errors import RiskEngineError
from risk_engine.service import RiskService

logger = logging.getLogger("risk_engine")


def create_app() -> FastAPI:
    app = FastAPI(title="Portfolio Risk Engine", version="1.1")
    app.state.risk_service = RiskService()
    app.include_router(router)
    app.include_router(scan_router)

    @app.get("/")
    def read_root() -> dict:
        return {"service": "portfolio-risk-engine", "status": "ok"}

    @app.exception_handler(RiskEngineError)
    async def risk_error_handler(request: Request, exc: RiskEngineError):
        return JSONResponse(status_code=422, content={"error": exc.to_dict()})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        # Never leak stack traces / raw numeric internals to callers.
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "reason": "internal computation error; see service logs",
                }
            },
        )

    return app
