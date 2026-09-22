"""Domain errors with stable, distinguishable reason codes.

Callers (HTTP layer, batch runner) map these to error responses; low-level
numeric exceptions from numpy must never leak out directly.
"""

from __future__ import annotations


class RiskEngineError(Exception):
    """Base class for all domain errors.

    ``reason`` is a stable machine-readable code; ``detail`` carries a
    human-readable explanation safe to expose to callers.
    """

    reason: str = "engine_error"
    http_status: int = 400

    def __init__(self, detail: str = "") -> None:
        self.detail = detail or self.reason
        super().__init__(self.detail)


# --- market data / window errors -------------------------------------------------

class SymbolNotFoundError(RiskEngineError):
    reason = "symbol_not_found"
    http_status = 404


class InsufficientHistoryError(RiskEngineError):
    reason = "insufficient_history"
    http_status = 422


class InvalidWindowError(RiskEngineError):
    reason = "invalid_window"
    http_status = 422


class EmptyPortfolioError(RiskEngineError):
    reason = "empty_portfolio"
    http_status = 422


class InvalidPositionError(RiskEngineError):
    reason = "invalid_position"
    http_status = 422


# --- correlation / covariance errors ---------------------------------------------

class MatrixDimensionError(RiskEngineError):
    reason = "matrix_dimension_mismatch"
    http_status = 422


class MatrixAsymmetricError(RiskEngineError):
    reason = "matrix_asymmetric"
    http_status = 422


class MatrixNotPositiveDefiniteError(RiskEngineError):
    reason = "matrix_not_positive_definite"
    http_status = 422


class MatrixInvalidValueError(RiskEngineError):
    reason = "matrix_invalid_value"
    http_status = 422


# --- scenario errors -------------------------------------------------------------

class ScenarioNotFoundError(RiskEngineError):
    reason = "scenario_not_found"
    http_status = 404


class ScenarioDefinitionError(RiskEngineError):
    reason = "scenario_definition_invalid"
    http_status = 422


# --- versioning errors -----------------------------------------------------------

class VersionNotFoundError(RiskEngineError):
    reason = "version_not_found"
    http_status = 404


class VersionIncompatibleError(RiskEngineError):
    reason = "version_incompatible"
    http_status = 409


# --- batch / resource errors -----------------------------------------------------

class BatchTooLargeError(RiskEngineError):
    reason = "batch_too_large"
    http_status = 429


class BatchTimeoutError(RiskEngineError):
    reason = "batch_timeout"
    http_status = 408


class BatchRejectedError(RiskEngineError):
    reason = "batch_rejected"
    http_status = 422


class InvalidPriceError(RiskEngineError):
    reason = "invalid_price_series"
    http_status = 422
