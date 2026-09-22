"""Typed error taxonomy for the risk engine.

Every rejection carries a machine-readable ``code`` and a human-readable
``reason`` so callers can distinguish failure causes.  Internal numeric
errors are never leaked raw to callers; they are wrapped into
``ComputationError`` (or a more specific subclass) at module boundaries.
"""

from __future__ import annotations


class RiskEngineError(Exception):
    """Base class for all risk-engine errors."""

    code: str = "RISK_ENGINE_ERROR"

    def __init__(self, reason: str, *, code: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        if code is not None:
            self.code = code

    def to_dict(self) -> dict:
        return {"code": self.code, "reason": self.reason}


# --- data / input problems -------------------------------------------------

class InsufficientDataError(RiskEngineError):
    code = "INSUFFICIENT_DATA"


class InvalidWindowError(RiskEngineError):
    code = "INVALID_WINDOW"


class InvalidParameterError(RiskEngineError):
    code = "INVALID_PARAMETER"


class UnknownInstrumentError(RiskEngineError):
    code = "UNKNOWN_INSTRUMENT"


class UnknownScenarioError(RiskEngineError):
    code = "UNKNOWN_SCENARIO"


class UnknownVersionError(RiskEngineError):
    code = "UNKNOWN_VERSION"


# --- correlation / covariance matrix problems ------------------------------

class CorrelationMatrixError(RiskEngineError):
    code = "CORRELATION_MATRIX_INVALID"


class DimensionMismatchError(CorrelationMatrixError):
    code = "DIMENSION_MISMATCH"


class AsymmetricMatrixError(CorrelationMatrixError):
    code = "ASYMMETRIC_MATRIX"


class NotPositiveDefiniteError(CorrelationMatrixError):
    code = "NOT_POSITIVE_DEFINITE"


class InvalidCorrelationValueError(CorrelationMatrixError):
    code = "INVALID_CORRELATION_VALUE"


# --- versioning ------------------------------------------------------------

class IncompatibleVersionError(RiskEngineError):
    code = "INCOMPATIBLE_VERSION"


# --- resources -------------------------------------------------------------

class ResourceLimitError(RiskEngineError):
    code = "RESOURCE_LIMIT_EXCEEDED"


class EvaluationTimeoutError(ResourceLimitError):
    code = "EVALUATION_TIMEOUT"


class BatchSizeLimitError(ResourceLimitError):
    code = "BATCH_SIZE_LIMIT"


# --- internal computation --------------------------------------------------

class ComputationError(RiskEngineError):
    """Wraps unexpected numeric failures so raw exceptions never leak."""

    code = "COMPUTATION_ERROR"
