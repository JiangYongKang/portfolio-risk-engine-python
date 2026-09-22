"""Validation and construction of correlation / covariance matrices.

Matrices are validated strictly and rejected (never silently repaired) when:

* dimensions are missing or inconsistent      -> matrix_dimension_mismatch
* entries are non-finite / out of range       -> matrix_invalid_value
* the matrix is not symmetric                 -> matrix_asymmetric
* the matrix is not positive definite         -> matrix_not_positive_definite

Low-level ``numpy.linalg`` exceptions are translated into the domain errors
above so implementation details never leak to callers.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from app.errors import (
    MatrixAsymmetricError,
    MatrixDimensionError,
    MatrixInvalidValueError,
    MatrixNotPositiveDefiniteError,
)

# relative tolerance for structural checks (symmetry, unit diagonal)
_SYM_TOL = 1e-10
# eigenvalues at or below this fraction of the spectral scale are treated
# as non-positive-definite (also catches exact zero eigenvalues)
_PSD_TOL = 1e-10


class MatrixKind(str, Enum):
    correlation = "correlation"
    covariance = "covariance"


def _to_array(matrix) -> np.ndarray:
    if matrix is None:
        raise MatrixDimensionError("matrix is required")
    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2 or matrix.size == 0:
            raise MatrixDimensionError("matrix must be a non-empty 2-D table")
        if matrix.shape[0] != matrix.shape[1]:
            raise MatrixDimensionError(f"matrix must be square, got shape {matrix.shape}")
        return np.array(matrix, dtype=float, copy=True)
    if not isinstance(matrix, (list, tuple)) or len(matrix) == 0:
        raise MatrixDimensionError("matrix must be a non-empty list of rows")
    n = len(matrix)
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise MatrixDimensionError(
                f"every row must be a list of length {n}; found row of length "
                f"{len(row) if hasattr(row, '__len__') else 'n/a'}"
            )
    try:
        arr = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MatrixInvalidValueError(f"matrix is not numeric: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise MatrixDimensionError("matrix must be a non-empty 2-D table")
    if arr.shape[0] != arr.shape[1]:
        raise MatrixDimensionError(f"matrix must be square, got shape {arr.shape}")
    return arr


def validate_matrix(
    matrix,
    labels=None,
    *,
    kind: MatrixKind | str = MatrixKind.correlation,
) -> np.ndarray:
    """Validate a correlation (default) or covariance matrix.

    Returns an immutable read-only ``numpy.ndarray`` when valid.
    ``labels`` optionally supplies the factor names; a mismatch with the
    matrix size raises ``matrix_dimension_mismatch``.
    """
    kind = MatrixKind(kind)
    arr = _to_array(matrix)

    if labels is not None:
        labels = list(labels)
        if len(labels) != arr.shape[0]:
            raise MatrixDimensionError(
                f"expected {len(labels)} labels but matrix is "
                f"{arr.shape[0]}x{arr.shape[0]}"
            )
        if len(set(labels)) != len(labels):
            raise MatrixInvalidValueError("factor labels must be unique")

    if not np.all(np.isfinite(arr)):
        raise MatrixInvalidValueError("matrix contains NaN or infinite entries")

    scale = float(np.max(np.abs(arr))) if arr.size else 1.0
    tol = _SYM_TOL * max(1.0, scale)
    if not np.allclose(arr, arr.T, atol=tol, rtol=0.0):
        max_asym = float(np.max(np.abs(arr - arr.T)))
        raise MatrixAsymmetricError(
            f"matrix is asymmetric; max |A - A.T| = {max_asym:.3g} exceeds tol {tol:.3g}"
        )

    diag = np.diag(arr)
    if kind is MatrixKind.correlation:
        if not np.allclose(diag, 1.0, atol=_SYM_TOL):
            bad = int(np.argmax(np.abs(diag - 1.0)))
            raise MatrixInvalidValueError(
                f"correlation diagonal must be 1.0; entry [{bad},{bad}] = {diag[bad]}"
            )
        if np.any(arr < -1.0 - _SYM_TOL) or np.any(arr > 1.0 + _SYM_TOL):
            raise MatrixInvalidValueError("correlation entries must lie in [-1, 1]")
    else:
        if np.any(diag <= 0.0):
            raise MatrixInvalidValueError("covariance diagonal must be strictly positive")

    _require_positive_definite(arr)
    arr.setflags(write=False)
    return arr


def _require_positive_definite(arr: np.ndarray) -> None:
    try:
        eigvals = np.linalg.eigvalsh(arr)
    except np.linalg.LinAlgError as exc:
        raise MatrixNotPositiveDefiniteError(
            f"eigen-decomposition failed: {exc}"
        ) from exc
    eig_min = float(eigvals[0])
    eig_scale = max(1.0, float(np.max(np.abs(eigvals))))
    limit = _PSD_TOL * eig_scale
    if eig_min <= limit:
        raise MatrixNotPositiveDefiniteError(
            f"smallest eigenvalue {eig_min:.6g} is not greater than tolerance {limit:.6g}"
        )


def correlation_to_covariance(corr, vols) -> np.ndarray:
    """Convert ``C`` to ``D C D`` with per-factor volatilities ``vols``."""
    c = validate_matrix(corr, kind=MatrixKind.correlation)
    try:
        d = np.asarray(vols, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MatrixInvalidValueError(f"volatilities are not numeric: {exc}") from exc
    if d.ndim != 1 or d.shape[0] != c.shape[0]:
        raise MatrixDimensionError(
            f"expected {c.shape[0]} volatilities, got {d.shape}"
        )
    if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        raise MatrixInvalidValueError("volatilities must be finite and positive")
    cov = (d[:, None] * c) * d[None, :]
    cov.setflags(write=False)
    return cov


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Sample covariance of column-oriented return observations."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise MatrixDimensionError("need at least 2 observations to estimate covariance")
    if not np.all(np.isfinite(arr)):
        raise MatrixInvalidValueError("returns contain NaN or infinite values")
    cov = np.cov(arr, rowvar=False, ddof=1)
    if not math.isfinite(float(np.max(np.abs(cov)))):
        raise MatrixInvalidValueError("covariance estimate is not finite")
    return cov
