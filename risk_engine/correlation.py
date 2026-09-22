"""Correlation/covariance validation and covariance construction."""

from __future__ import annotations

import math

from risk_engine.errors import (
    AsymmetricMatrixError,
    ComputationError,
    DimensionMismatchError,
    DimensionMismatchError,
    InvalidCorrelationValueError,
    NotPositiveDefiniteError,
)
from risk_engine.models import CorrelationAssumption, InstrumentId

Matrix = list[list[float]]

SYMMETRY_TOL = 1e-9
PD_TOL = 1e-12
_CORR_BOUND = 1.0 + 1e-9


def _as_matrix(matrix: Matrix | tuple[tuple[float, ...], ...]) -> Matrix:
    m = [list(row) for row in matrix]
    return m


def validate_matrix(
    matrix: Matrix | tuple[tuple[float, ...], ...],
    instruments: tuple[InstrumentId, ...],
    *,
    kind: str = "correlation",
) -> Matrix:
    """Validate a symmetric positive-(semi)definite matrix.

    Checks in a fixed, distinguishable order:
      1. dimensions match ``instruments`` (square, aligned)
      2. values finite and within legal bounds
      3. symmetry
      4. positive-definiteness (Cholesky)
    Nothing is silently repaired; raw numeric exceptions are wrapped.
    """
    n = len(instruments)
    m = _as_matrix(matrix)

    # 1. dimensions
    if len(m) != n:
        raise DimensionMismatchError(
            f"{kind} matrix has {len(m)} rows but {n} instruments are referenced"
        )
    for i, row in enumerate(m):
        if len(row) != n:
            raise DimensionMismatchError(
                f"{kind} matrix row {i} has {len(row)} columns, expected {n}"
            )

    # 2. values
    for i, row in enumerate(m):
        for j, value in enumerate(row):
            if not math.isfinite(value):
                raise InvalidCorrelationValueError(
                    f"{kind} matrix contains non-finite value at [{i}][{j}]"
                )
            if kind == "correlation":
                if value < -_CORR_BOUND or value > _CORR_BOUND:
                    raise InvalidCorrelationValueError(
                        f"correlation [{i}][{j}]={value} outside [-1, 1]"
                    )
        if kind == "correlation" and not math.isclose(
            m[i][i], 1.0, abs_tol=1e-9
        ):
            raise InvalidCorrelationValueError(
                f"correlation diagonal [{i}][{i}]={m[i][i]} must equal 1.0"
            )

    # 3. symmetry
    for i in range(n):
        for j in range(i + 1, n):
            if abs(m[i][j] - m[j][i]) > SYMMETRY_TOL:
                raise AsymmetricMatrixError(
                    f"{kind} matrix not symmetric: "
                    f"[{i}][{j}]={m[i][j]} vs [{j}][{i}]={m[j][i]}"
                )

    # 4. positive-definiteness (PSD accepted for correlations by clamping
    #    only through Cholesky tolerance; zero variance is allowed)
    cholesky(m)
    return m


def cholesky(matrix: Matrix) -> Matrix:
    """Lower-triangular Cholesky factor.

    Raises NotPositiveDefiniteError for non-PD input.  Numeric surprises
    are wrapped into ComputationError so callers never see raw exceptions.
    """
    n = len(matrix)
    lower: Matrix = [[0.0] * n for _ in range(n)]
    try:
        for i in range(n):
            for j in range(i + 1):
                acc = matrix[i][j] - sum(
                    lower[i][k] * lower[j][k] for k in range(j)
                )
                if i == j:
                    if acc <= PD_TOL:
                        raise NotPositiveDefiniteError(
                            f"matrix not positive definite at pivot {i} "
                            f"(residual {acc:.3e})"
                        )
                    lower[i][j] = math.sqrt(acc)
                else:
                    lower[i][j] = acc / lower[j][j] if lower[j][j] != 0 else 0.0
    except NotPositiveDefiniteError:
        raise
    except (ZeroDivisionError, ValueError, OverflowError) as exc:
        raise ComputationError(f"matrix decomposition failed: {exc}") from exc
    return lower


def build_covariance(
    vols: list[float],
    correlation: CorrelationAssumption,
    instruments: tuple[InstrumentId, ...],
) -> Matrix:
    """Construct covariance = diag(vol) @ corr @ diag(vol) with order mapping.

    The assumption's instrument order is mapped onto ``instruments``;
    mismatched instrument sets are rejected (no silent reordering unless
    the same set is present).
    """
    if set(correlation.instruments) != set(instruments):
        raise DimensionMismatchError(
            "correlation instruments "
            f"{list(correlation.instruments)} do not match "
            f"portfolio instruments {list(instruments)}"
        )

    corr = validate_matrix(correlation.matrix, correlation.instruments)
    idx = {inst: i for i, inst in enumerate(correlation.instruments)}
    pos = [idx[inst] for inst in instruments]
    vol_by_inst = dict(zip(instruments, vols))

    n = len(instruments)
    cov: Matrix = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            va = vol_by_inst[instruments[a]]
            vb = vol_by_inst[instruments[b]]
            cov[a][b] = corr[pos[a]][pos[b]] * va * vb
    return cov


def combine_block_diagonal(
    name: str,
    assumptions: list[CorrelationAssumption],
    *,
    version: str = "v1",
) -> CorrelationAssumption:
    """Combine assumptions over **disjoint** factor blocks.

    Blocks with overlapping instruments cannot be combined unambiguously
    and are rejected with ``DIMENSION_MISMATCH`` rather than averaged.
    Cross-block correlations are set to zero (block-diagonal construction),
    ordered by first appearance of instruments, which is deterministic.
    Each input assumption is validated first.
    """
    seen: set[InstrumentId] = set()
    ordered: list[InstrumentId] = []
    blocks: list[Matrix] = []
    for assump in assumptions:
        m = validate_matrix(assump.matrix, assump.instruments)
        overlap = set(assump.instruments) & seen
        if overlap:
            raise DimensionMismatchError(
                "cannot combine overlapping correlation blocks on "
                f"{sorted(overlap)}"
            )
        seen.update(assump.instruments)
        ordered.extend(assump.instruments)
        blocks.append(m)

    n = len(ordered)
    combined: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        combined[i][i] = 1.0
    offset = 0
    for m in blocks:
        k = len(m)
        for a in range(k):
            for b in range(k):
                combined[offset + a][offset + b] = m[a][b]
        offset += k
    return CorrelationAssumption(
        name=name,
        instruments=tuple(ordered),
        matrix=tuple(tuple(row) for row in combined),
        version=version,
    )
