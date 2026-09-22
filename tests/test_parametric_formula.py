"""Closed-form checks for the parametric normal tail metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.risk.metrics import parametric_tail, _zscore
from tests.conftest import log


def test_zscore_matches_known_normal_quantiles():
    cases = {0.90: 1.2815515655446004, 0.95: 1.6448536269514722,
             0.975: 1.959963984540054, 0.99: 2.3263478740408408}
    for c, expected in cases.items():
        assert _zscore(c) == pytest.approx(expected, abs=2e-9)
    log.info("inverse-normal quantiles match reference values: %s", cases)


def test_single_asset_zero_drift_closed_form():
    mu = np.array([0.0])
    cov = np.array([[0.01]])  # sigma = 0.1
    w = np.array([1.0])
    c = 0.95
    var, es = parametric_tail(mu, cov, w, c)
    z = 1.6448536269514722
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    assert var == pytest.approx(z * 0.1, abs=1e-9)
    assert es == pytest.approx(0.1 * phi / 0.05, abs=1e-9)
    log.info("closed-form zero-drift: VaR=%.6f ES=%.6f", var, es)


def test_drift_adjustment_shifts_both_metrics():
    cov = np.array([[0.01]])
    w = np.array([1.0])
    c = 0.95
    var0, es0 = parametric_tail(np.array([0.0]), cov, w, c)
    var_up, es_up = parametric_tail(np.array([0.02]), cov, w, c)
    log.info("mu=0 VaR=%.6f ES=%.6f ; mu=0.02 VaR=%.6f ES=%.6f",
             var0, es0, var_up, es_up)
    assert var_up == max(var0 - 0.02, 0.0)
    assert es_up == pytest.approx(es0 - 0.02, abs=1e-12)


def test_two_asset_diversification_reduces_var():
    mu = np.zeros(2)
    cov_indep = np.array([[0.04, 0.0], [0.0, 0.04]])
    cov_perf = np.array([[0.04, 0.04], [0.04, 0.04]])
    w = np.array([0.5, 0.5])
    v_indep, _ = parametric_tail(mu, cov_indep, w, 0.95)
    v_perf, _ = parametric_tail(mu, cov_perf, w, 0.95)
    log.info("VaR independent=%.6f perfectly-correlated=%.6f", v_indep, v_perf)
    assert v_indep < v_perf
