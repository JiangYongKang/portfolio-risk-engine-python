"""Typed error taxonomy for the risk engine.

Every rejection carries a machine-readable ``code`` and a human-readable
``reason`` so callers can distinguish failure causes.  On top of the
specific codes, every error also carries a coarse ``category`` so the
four failure families can be told apart at a glance:

* ``invalid_input``   — the request itself is malformed or inconsistent
  (bad dates, unknown names, invalid matrices, version refusals, ...);
* ``resource_limit``  — the batch/item budget was exceeded
  (``BATCH_SIZE_LIMIT``);
* ``timeout``         — the time budget was exceeded
  (``EVALUATION_TIMEOUT``);
* ``internal``        — unexpected engine-side failure, wrapped so raw
  exceptions never leak (``COMPUTATION_ERROR``).

Internal numeric errors are never leaked raw to callers; they are wrapped
into ``ComputationError`` (or a more specific subclass) at module
boundaries.
"""

from __future__ import annotations

# Coarse failure families (stable strings, safe to compare/persist).
CATEGORY_INVALID_INPUT = "invalid_input"
CATEGORY_RESOURCE_LIMIT = "resource_limit"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_INTERNAL = "internal"


class RiskEngineError(Exception):
    """Base class for all risk-engine errors."""

    code: str = "RISK_ENGINE_ERROR"
    category: str = CATEGORY_INVALID_INPUT

    def __init__(
        self,
        reason: str,
        *,
        code: str | None = None,
        category: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        if code is not None:
            self.code = code
        if category is not None:
            self.category = category

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "category": self.category,
            "reason": self.reason,
        }


# --- data / input problems -------------------------------------------------

class InsufficientDataError(RiskEngineError):
    code = "INSUFFICIENT_DATA"


class InvalidWindowError(RiskEngineError):
    code = "INVALID_WINDOW"


class InvalidParameterError(RiskEngineError):
    code = "INVALID_PARAMETER"


class InvalidInputError(RiskEngineError):
    """Malformed raw input (e.g. an unparseable date string)."""

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
    category = CATEGORY_RESOURCE_LIMIT


class EvaluationTimeoutError(ResourceLimitError):
    code = "EVALUATION_TIMEOUT"
    category = CATEGORY_TIMEOUT


class BatchSizeLimitError(ResourceLimitError):
    code = "BATCH_SIZE_LIMIT"


# --- internal computation --------------------------------------------------

class ComputationError(RiskEngineError):
    """Wraps unexpected numeric failures so raw exceptions never leak."""

    code = "COMPUTATION_ERROR"
    category = CATEGORY_INTERNAL
