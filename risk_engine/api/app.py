"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from risk_engine.api.routes import router
from risk_engine.service import RiskService


def create_app() -> FastAPI:
    app = FastAPI(title="Portfolio Risk Engine")
    app.state.risk_service = RiskService()
    app.include_router(router)

    @app.get("/")
    def read_root() -> dict:
        return {"service": "portfolio-risk-engine", "status": "ok"}

    return app
