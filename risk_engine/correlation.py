"""Correlation/covariance validation and covariance construction."""

from __future__ import annotations

from risk_engine.models import CorrelationAssumption, InstrumentId

Matrix = list[list[float]]


def validate_matrix(matrix: Matrix, instruments: tuple[InstrumentId, ...],
                    *, kind: str = "correlation") -> None:
    raise NotImplementedError


def build_covariance(
    vols: list[float], correlation: CorrelationAssumption,
    instruments: tuple[InstrumentId, ...],
) -> Matrix:
    raise NotImplementedError


def cholesky(matrix: Matrix) -> Matrix:
    """Raise NotPositiveDefiniteError rather than leak numeric errors."""
    raise NotImplementedError
