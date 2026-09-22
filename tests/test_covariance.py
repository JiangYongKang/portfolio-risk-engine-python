"""Tests for strict correlation / covariance matrix validation."""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import MatrixDimensionError, MatrixInvalidValueError
from app.risk.covariance import (
    MatrixKind,
    correlation_to_covariance,
    sample_covariance,
    validate_matrix,
)
from tests.conftest import log  # noqa: E402


VALID_2X2 = [[1.0, 0.5], [0.5, 1.0]]


@pytest.mark.parametrize(
    "matrix,expected_reason,summary",
    [
        (VALID_2X2, None, "valid symmetric PD 2x2 correlation"),
        ([[1.0, 0.5]], "matrix_dimension_mismatch", "rectangular 1x2 table"),
        ([[1.0, 0.5], [0.5]], "matrix_dimension_mismatch", "ragged rows"),
        ([[1.0, 0.5], [0.6, 1.0]], "matrix_asymmetric", "A[0,1]=0.5 != A[1,0]=0.6"),
        (
            [[1.0, 1.0000001], [1.0000001, 1.0]],
            "matrix_invalid_value",
            "entry 1.0000001 outside [-1,1]",
        ),
        ([[1.0, 1.0], [1.0, 1.0]], "matrix_not_positive_definite", "perfectly collinear"),
        ([[1.0, 1.1], [1.1, 1.0]], "matrix_invalid_value", "entry outside [-1,1]"),
        ([[1.0, float("nan")], [float("nan"), 1.0]], "matrix_invalid_value", "NaN entry"),
        ([[2.0, 0.0], [0.0, 1.0]], "matrix_invalid_value", "non-unit diagonal"),
        ([[]], "matrix_dimension_mismatch", "empty matrix"),
    ],
)
def test_matrix_validation_reasons(matrix, expected_reason, summary):
    from app.errors import RiskEngineError

    log.info("input=%s (%s)", matrix, summary)
    if expected_reason is None:
        out = validate_matrix(matrix)
        log.info("accepted; rationale: symmetric, unit diagonal, eigmin>0")
        assert out.shape == (2, 2)
        assert not out.flags.writeable
        return
    with pytest.raises(RiskEngineError) as ei:
        validate_matrix(matrix)
    assert ei.value.reason == expected_reason
    log.info("rejected reason=%s rationale=%s", expected_reason, summary)


def test_label_dimension_mismatch():
    log.info("input=2x2 matrix with 3 labels")
    with pytest.raises(MatrixDimensionError) as ei:
        validate_matrix(VALID_2X2, labels=["a", "b", "c"])
    assert ei.value.reason == "matrix_dimension_mismatch"
    log.info("rejected reason=%s", ei.value.reason)


def test_covariance_kind_accepts_identity_variance():
    cov = [[0.04, 0.01], [0.01, 0.09]]
    log.info("input=%s kind=covariance", cov)
    out = validate_matrix(cov, kind=MatrixKind.covariance)
    assert out[0, 0] == pytest.approx(0.04)
    log.info("accepted covariance; positive diagonal, symmetric PD")


def test_covariance_kind_rejects_nonpositive_diagonal():
    with pytest.raises(MatrixInvalidValueError):
        validate_matrix([[0.0, 0.0], [0.0, 1.0]], kind=MatrixKind.covariance)
    log.info("zero diagonal covariance rejected as matrix_invalid_value")


def test_correlation_to_covariance_roundtrip():
    corr = validate_matrix(VALID_2X2)
    cov = correlation_to_covariance(VALID_2X2, [0.2, 0.3])
    log.info("corr=%s vols=%s -> cov=%s", VALID_2X2, [0.2, 0.3], cov.tolist())
    assert cov[0, 1] == pytest.approx(0.5 * 0.2 * 0.3)
    implied = cov / np.outer([0.2, 0.3], [0.2, 0.3])
    assert np.allclose(implied, corr)


def test_correlation_to_covariance_dimension_mismatch():
    with pytest.raises(MatrixDimensionError):
        correlation_to_covariance(VALID_2X2, [0.2])
    log.info("1 volatility vs 2 factors rejected as dimension mismatch")


def test_sample_covariance_from_returns():
    rng = np.random.default_rng(7)
    rets = rng.normal(size=(50, 3))
    cov = sample_covariance(rets)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    log.info("sample covariance of 50x3 returns shape=%s symmetric", cov.shape)
