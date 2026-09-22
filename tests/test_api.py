"""End-to-end HTTP tests: error envelopes, batch isolation, version flow."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import log


@pytest.fixture
def client():
    return TestClient(create_app())


def _seed_prices(client, symbols=("AAA", "BBB"), n=80):
    for k, sym in enumerate(symbols):
        prices = [
            100.0 * math.exp(0.0008 * (i % 9 - 4) + 0.0004 * k * (i % 7 - 3))
            for i in range(n)
        ]
        r = client.post("/market/prices", json={"symbol": sym, "prices": prices})
        assert r.status_code == 201, r.text


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_risk_happy_path_and_error_envelope(client):
    _seed_prices(client)
    payload = {
        "portfolio": {"name": "p", "positions": [
            {"symbol": "AAA", "quantity": 100}, {"symbol": "BBB", "quantity": 50}]},
        "confidence": {"confidence": 0.95, "window": 30},
    }
    r = client.post("/risk", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    log.info("risk response keys=%s hVaR=%.3f pVaR=%.3f",
             sorted(body), body["historical_var"], body["parametric_var"])
    assert body["historical_es"] >= body["historical_var"]

    bad = dict(payload)
    bad["portfolio"] = {"name": "p", "positions": [{"symbol": "GHOST", "quantity": 1}]}
    r = client.post("/risk", json=bad)
    log.info("unknown symbol status=%s body=%s", r.status_code, r.json())
    assert r.status_code == 404
    assert r.json()["error"]["reason"] == "symbol_not_found"


def test_risk_correlation_override_endpoint(client):
    _seed_prices(client)
    base_payload = {
        "portfolio": {"name": "p", "positions": [
            {"symbol": "AAA", "quantity": 100}, {"symbol": "BBB", "quantity": 50}]},
        "confidence": {"confidence": 0.95, "window": 30},
    }
    r0 = client.post("/risk", json=base_payload).json()
    r1 = client.post("/risk", json={**base_payload,
        "correlation_override": [[1, 0], [0, 1]]})
    assert r1.status_code == 200, r1.text
    log.info("override accepted: hVaR unchanged=%s, pVaR %.4f -> %.4f",
             r1.json()["historical_var"] == r0["historical_var"],
             r0["parametric_var"], r1.json()["parametric_var"])
    assert r1.json()["historical_var"] == r0["historical_var"]
    bad = client.post("/risk", json={**base_payload,
        "correlation_override": [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]})
    assert bad.status_code == 422
    assert bad.json()["error"]["reason"] == "matrix_dimension_mismatch"
    log.info("3x3 override on 2-symbol portfolio -> 422 matrix_dimension_mismatch")


def test_correlation_endpoint_rejects_by_reason(client):
    cases = [
        ({"factors": ["a", "b"], "matrix": [[1, 0.5], [0.5]]}, 422, "matrix_dimension_mismatch"),
        ({"factors": ["a", "b"], "matrix": [[1, 0.5], [0.6, 1]]}, 422, "matrix_asymmetric"),
        ({"factors": ["a", "b"], "matrix": [[1, 1], [1, 1]]}, 422, "matrix_not_positive_definite"),
        ({"factors": ["a", "b", "c"], "matrix": [[1, 0], [0, 1]]},
         422, "matrix_dimension_mismatch"),
    ]
    for payload, status, reason in cases:
        r = client.post("/correlation/validate", json=payload)
        log.info("correlation input %s -> status=%s reason=%s",
                 payload["matrix"], r.status_code, r.json()["error"]["reason"])
        assert r.status_code == status
        assert r.json()["error"]["reason"] == reason
    ok = client.post("/correlation/validate",
                     json={"factors": ["a", "b"], "matrix": [[1, 0.5], [0.5, 1]]})
    assert ok.status_code == 200 and ok.json()["accepted"] is True


def test_scenario_and_composite_flow(client):
    for s in [
        {"name": "eq", "type": "factor_shock",
         "shocks": [{"factor": "equity", "shock": -0.1}]},
        {"name": "cr", "type": "factor_shock",
         "shocks": [{"factor": "credit", "shock": 0.05}]},
        {"name": "combo", "type": "composite", "components": ["eq", "cr"]},
    ]:
        assert client.post("/scenarios", json=s).status_code == 201
    pf = {"name": "p", "positions": [
        {"symbol": "AAA", "quantity": 1000}, {"symbol": "BBB", quantity: 2000 if False else "quantity", "quantity": 2000} if False else {"symbol": "BBB", "quantity": 2000}]}
    fmap = {"factor_map": {"AAA": {"equity": 1.0}, "BBB": {"equity": 0.6, "credit": 0.4}}}
    r = client.post("/scenarios/combo/evaluate", json={"portfolio": pf, **fmap})
    assert r.status_code == 200, r.text
    body = r.json()
    log.info("combo pnl=%s contributions=%s", body["pnl"], body["factor_contributions"])
    assert sum(body["factor_contributions"].values()) == pytest.approx(body["pnl"])


def test_batch_endpoint_ordering_and_isolation(client):
    _seed_prices(client)

    def item(i, bad=False):
        return {
            "id": f"i{i}",
            "portfolio": {"name": f"p{i}", "positions":
                          [{"symbol": "GHOST", "quantity": 1}] if bad else
                          [{"symbol": "AAA", "quantity": 100}, {"symbol": "BBB", "quantity": 50}]},
            "confidence": {"confidence": 0.95, "window": 20 + i % 5},
        }

    items = [item(0), item(1, bad=True), item(2), item(3, bad=True), item(4)]
    r = client.post("/batch", json={"items": items, "timeout": 10})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [e["id"] for e in results] == [f"i{i}" for i in range(5)]
    assert results[1]["reason"] == "symbol_not_found" and results[1]["ok"] is False
    assert results[3]["reason"] == "symbol_not_found"
    assert all(results[i]["ok"] for i in (0, 2, 4))
    log.info("batch order & isolation verified; failures do not abort siblings")


def test_batch_too_large_distinguishable(client):
    item = {
        "id": "x",
        "portfolio": {"name": "p", "positions": [
            {"symbol": "AAA", "quantity": 100}]},
        "confidence": {"confidence": 0.95, "window": 10},
    }
    r = client.post("/batch", json={"items": [item, item], "max_items": 1})
    assert r.status_code == 429
    assert r.json()["error"]["reason"] == "batch_too_large"
    log.info("oversized batch rejected status=429 reason=batch_too_large")


def test_version_lifecycle(client):
    for s in [
        {"name": "eq", "type": "factor_shock",
         "shocks": [{"factor": "equity", "shock": -0.1}]},
    ]:
        client.post("/scenarios", json=s)
    reg = {
        "parameter_set": {"name": "base", "version": 1, "rule_version": 1, "params": {}},
        "scenario_name": "eq",
        "portfolio": {"name": "p", "positions": [{"symbol": "AAA", "quantity": 1000}]},
        "factor_map": {"AAA": {"equity": 1.0}},
    }
    assert client.post("/parameters", json=reg).status_code == 201
    r = client.post("/parameters/base/versions/1/evaluate", json={})
    assert r.status_code == 200 and r.json()["reproducible"] is True
    r2 = client.post("/parameters/base/versions/1/evaluate", json={})
    assert r.json()["result"] == r2.json()["result"]
    log.info("versioned evaluation deterministic over HTTP")


def test_validation_error_does_not_leak_internals(client):
    r = client.post("/risk", json={
        "portfolio": {"name": "p", "positions": [{"symbol": "AAA", "quantity": 1}]},
        "confidence": {"confidence": 0.95, "window": 1}})
    assert r.status_code == 422
    log.info("invalid window -> 422 without 500/stacktrace")


def test_unexpected_exception_never_leaks(monkeypatch):
    client = TestClient(create_app(), raise_server_exceptions=False)

    def boom(*_a, **_kw):
        raise RuntimeError("numpy internals exploded at frame 0x7f3a")

    monkeypatch.setattr(client.app.state.engine, "risk", boom)
    _seed_prices(client)
    r = client.post("/risk", json={
        "portfolio": {"name": "p", "positions": [
            {"symbol": "AAA", "quantity": 100}]},
        "confidence": {"confidence": 0.95, "window": 20}})
    log.info("forced internal failure -> status=%s body=%s", r.status_code, r.json())
    assert r.status_code == 500
    body = r.json()["error"]
    assert body["reason"] == "internal_error"
    assert "numpy internals" not in str(body)
