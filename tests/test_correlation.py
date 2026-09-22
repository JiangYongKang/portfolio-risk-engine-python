"""Illegal correlation/covariance inputs must be rejected by cause."""

from __future__ import annotations

import pytest

from risk_engine import correlation as corr
from risk_engine.errors import (
    AsymmetricMatrixError,
    DimensionMismatchError,
    InvalidCorrelationValueError,
    NotPositiveDefiniteError,
)
from risk_engine.models import CorrelationAssumption

from .conftest import log


def _case(name, matrix, insts, expected):
    log.info(
        "case=%s insts=%s matrix=%s -> expect %s; criterion: exact typed code",
        name, list(insts), matrix, expected,
    )


def test_dimension_row_count_mismatch():
    insts = ("A", "B")
    matrix = [[1.0, 0.0]]  # 1 row, 2 instruments
    _case("dim-rows", matrix, insts, "DIMENSION_MISMATCH")
    with pytest.raises(DimensionMismatchError) as ei:
        corr.validate_matrix(matrix, insts)
    assert ei.value.code == "DIMENSION_MISMATCH"


def test_dimension_column_count_mismatch():
    insts = ("A", "B")
    matrix = [[1.0, 0.0], [0.0]]  # ragged row
    _case("dim-cols", matrix, insts, "DIMENSION_MISMATCH")
    with pytest.raises(DimensionMismatchError) as ei:
        corr.validate_matrix(matrix, insts)
    assert ei.value.code == "DIMENSION_MISMATCH"


def test_value_out_of_bounds():
    insts = ("A", "B")
    matrix = [[1.0, 1.4], [1.4, 1.0]]
    _case("value-oob", matrix, insts, "INVALID_CORRELATION_VALUE")
    with pytest.raises(InvalidCorrelationValueError) as ei:
        corr.validate_matrix(matrix, insts)
    assert ei.value.code == "INVALID_CORRELATION_VALUE"


def test_value_non_finite():
    insts = ("A", "B")
    matrix = [[1.0, float("nan")], [float("nan"), 1.0]]
    _case("value-nan", matrix, insts, "INVALID_CORRELATION_VALUE")
    with pytest.raises(InvalidCorrelationValueError):
        corr.validate_matrix(matrix, insts)


def test_diagonal_not_one():
    insts = ("A", "B")
    matrix = [[0.9, 0.0], [0.0, 1.0]]
    _case("diag", matrix, insts, "INVALID_CORRELATION_VALUE")
    with pytest.raises(InvalidCorrelationValueError):
        corr.validate_matrix(matrix, insts)


def test_asymmetric_rejected():
    insts = ("A", "B")
    matrix = [[1.0, 0.5], [0.2, 1.0]]
    _case("asymmetric", matrix, insts, "ASYMMETRIC_MATRIX")
    with pytest.raises(AsymmetricMatrixError) as ei:
        corr.validate_matrix(matrix, insts)
    assert ei.value.code == "ASYMMETRIC_MATRIX"


def test_non_positive_definite_rejected():
    insts = ("A", "B", "C")
    matrix = [
        [1.0, -0.6, -0.6],
        [-0.6, 1.0, -0.6],
        [-0.6, -0.6, 1.0],
    ]
    _case("npd", matrix, insts, "NOT_POSITIVE_DEFINITE")
    with pytest.raises(NotPositiveDefiniteError) as ei:
        corr.validate_matrix(matrix, insts)
    assert ei.value.code == "NOT_POSITIVE_DEFINITE"


def test_valid_positive_definite_passes():
    insts = ("A", "B", "C")
    matrix = [
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.1],
        [0.2, 0.1, 1.0],
    ]
    _case("valid", matrix, insts, "no error")
    assert corr.validate_matrix(matrix, insts) == matrix


def test_covariance_instrument_set_mismatch():
    insts = ("A", "B")
    ca = CorrelationAssumption("x", ("A", "C"), ((1.0, 0.0), (0.0, 1.0)))
    log.info("case=cov-set-mismatch; criterion: set inequality rejected")
    with pytest.raises(DimensionMismatchError):
        corr.build_covariance([0.01, 0.02], ca, insts)


def test_covariance_reorder_is_consistent():
    # The same assumption expressed in a different instrument order must
    # produce the same covariance in canonical portfolio order.
    ca_ab = CorrelationAssumption(
        "ab", ("A", "B"), ((1.0, 0.3), (0.3, 1.0))
    )
    ca_ba = CorrelationAssumption(
        "ba", ("B", "A"), ((1.0, 0.3), (0.3, 1.0))
    )
    cov1 = corr.build_covariance([0.02, 0.01], ca_ab, ("A", "B"))
    cov2 = corr.build_covariance([0.02, 0.01], ca_ba, ("A", "B"))
    log.info("case=cov-reorder cov_ab=%s cov_ba=%s", cov1, cov2)
    for i in range(2):
        for j in range(2):
            assert cov1[i][j] == pytest.approx(cov2[i][j], abs=1e-15)


def test_cholesky_factor_reconstructs_matrix():
    matrix = [[4.0, 2.0], [2.0, 5.0]]
    l = corr.cholesky(matrix)
    reconstructed = [
        [sum(l[i][k] * l[j][k] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]
    for i in range(2):
        for j in range(2):
            assert reconstructed[i][j] == pytest.approx(matrix[i][j])


def test_block_diagonal_combine_zero_cross_block():
    a = CorrelationAssumption("a", ("A", "B"), ((1.0, 0.3), (0.3, 1.0)))
    c = CorrelationAssumption("c", ("C",), ((1.0,),))
    combined = corr.combine_block_diagonal("ac", [a, c])
    log.info("case=block-diagonal instruments=%s", list(combined.instruments))
    assert combined.instruments == ("A", "B", "C")
    m = combined.matrix
    # within-block preserved
    assert m[0][1] == m[1][0] == 0.3
    # cross-block zeroed, identity diagonal
    assert m[0][2] == m[1][2] == m[2][0] == m[2][1] == 0.0
    # combined must remain valid PD
    corr.validate_matrix(
        [list(r) for r in m], combined.instruments
    )


def test_block_diagonal_overlap_rejected():
    a = CorrelationAssumption("a", ("A", "B"), ((1.0, 0.3), (0.3, 1.0)))
    b = CorrelationAssumption("b", ("B", "C"), ((1.0, 0.2), (0.2, 1.0)))
    log.info("case=block-overlap blocks on {A,B} and {B,C}")
    with pytest.raises(DimensionMismatchError):
        corr.combine_block_diagonal("bad", [a, b])
