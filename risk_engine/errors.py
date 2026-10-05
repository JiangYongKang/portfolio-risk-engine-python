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


class InvalidInputError(RiskEngineError):
    """Malformed input payload (e.g. unparseable date, wrong shape).

    Distinct from semantic rejections (unknown names, bad matrices): the
    payload itself could not be read.  In batches only the offending slot
    fails with this code; every other slot is unaffected.
    """

    code = "INVALID_INPUT"


class UnknownInstrumentError(RiskEngineError):
    code = "UNKNOWN_INSTRUMENT"


class UnknownScenarioError(RiskEngineError):
    code = "UNKNOWN_SCENARIO"


class UnknownAssumptionError(RiskEngineError):
    code = "UNKNOWN_ASSUMPTION"


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


# --- failure classes ---------------------------------------------------------
#
# Every failure code maps to exactly one coarse class so callers can tell
# the four failure kinds apart at a glance:
#
# * ``invalid_input``  — the request itself is malformed or inconsistent
#   (bad dates, unknown names, invalid matrices, bad parameters, ...);
# * ``timeout``        — the item did not finish within the time budget;
# * ``over_limit``     — a declared resource limit was exceeded;
# * ``internal_error`` — an unexpected engine-side failure (already wrapped).

CLASS_INVALID_INPUT = "invalid_input"
CLASS_TIMEOUT = "timeout"
CLASS_OVER_LIMIT = "over_limit"
CLASS_INTERNAL_ERROR = "internal_error"

_CLASS_BY_CODE = {
    "EVALUATION_TIMEOUT": CLASS_TIMEOUT,
    "BATCH_SIZE_LIMIT": CLASS_OVER_LIMIT,
    "RESOURCE_LIMIT_EXCEEDED": CLASS_OVER_LIMIT,
    "COMPUTATION_ERROR": CLASS_INTERNAL_ERROR,
    "INTERNAL_ERROR": CLASS_INTERNAL_ERROR,
}


def classify_failure(code: str) -> str:
    """Coarse failure class for an error code.

    Everything not explicitly timeout/over-limit/internal is an input
    problem (the 422 family), so unknown codes default to
    ``invalid_input``.
    """
    return _CLASS_BY_CODE.get(code, CLASS_INVALID_INPUT)
