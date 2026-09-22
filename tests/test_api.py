"""HTTP integration: error codes over the wire, batch isolation, scenarios."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from main import app

from .conftest import log

logging.disable(logging.NOTSET)
client = TestClient(app)


def _dates(n=60):
    return [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]


def _valid_body(corr=None):
    body = {
        "portfolio": {
            "name": "fund-x",
            "holdings": [
                {"instrument": "A", "quantity": 10},
                {"instrument": "B", "quantity": -5},
            ],
        },
        "model_invalid": False,
        "market": {
            "dates": _dates(),
            "prices": {
                "A": [100 + 0.3 * i for i in range(60)],
                "B": [50 + (i % 5) * 0.4 for i in range(60)],
            },
        },
        "window": 20,
    }
    if corr is not None:
        body["correlation"] = corr
    return body


def test_assess_ok():
    r = client.post("/api/risk/assess", json=_valid_body())
    log.info("case=http-ok status=%s", r.status_code)
    assert r.status_code == 200
    d = r.json()
    assert d["version"] == "v1"
    assert "historical" in d and "parametric" in d


def test_unknown_instrument_http():
    b = _valid_body()
    b["portfolio"]["holdings"].append({"instrument": "Z", "quantity": 1})
    r = client.post("/api/risk/assess", json=b)
    log.info("case=http-unknown status=%s code=%s", r.status_code, r.json()["error"]["code"])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNKNOWN_INSTRUMENT"


def test_invalid_window_http():
    b = _valid_body()
    b["window"] = 999
    r = client.post("/api/risk/assess", json=b)
    log.info("case=http-window status=%s code=%s", r.status_code, r.json()["error"]["code"])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INSUFFICIENT_DATA"


def test_each_correlation_error_over_http():
    cases = {
        "DIMENSION_MISMATCH": {"name": "x", "instruments": ["A"], "matrix": [[1]]},
        "INVALID_CORRELATION_VALUE": {
            "name": "x", "instruments": ["A", "B"], "matrix": [[1, 1.5], [1.5, 1]]
        },
        "ASYMMETRIC_MATRIX": {
            "name": "x", "instruments": ["A", "B"], "matrix": [[1, 0.5], [0.2, 1]]
        },
    }
    for expected, corr in cases.items():
        r = client.post("/api/risk/assess", json=_valid_body(corr))
        log.info("case=http-corr-%s status=%s code=%s", expected, r.status_code, r.json()["error"]["code"])
        assert r.status_code == 422
        assert r.json()["error"]["code"] == expected


def test_batch_isolates_and_preserves_order():
    good = _valid_body()
    bad = _valid_body()
    bad["window"] = 999
    r = client.post("/api/risk/batch", json={"requests": [good, bad, good]})
    d = r.json()
    flags = [(x["index"], x["ok"], (x["error"] or {}).get("code")) for x in d["results"]]
    log.info("case=http-batch %s", flags)
    assert r.status_code == 200
    assert flags == [
        (0, True, None),
        (1, False, "INSUFFICIENT_DATA"),
        (2, True, None),
    ]


def test_batch_size_limit_http():
    r = client.post(
        "/api/risk/batch",
        json={"requests": [_valid_body() for _ in range(3)], "max_items": 2},
    )
    log.info("case=http-batch-limit status=%s code=%s", r.status_code, r.json()["error"]["code"])
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "BATCH_SIZE_LIMIT"


def test_scenario_evaluate_http():
    b = _valid_body()
    payload = {
        "portfolio": b["portfolio"],
        "market": b["market"],
        "scenario": {
            "name": "drop",
            "shocks": [
                {"instrument": "A", "relative_move": -0.1},
                {"instrument": "B", "relative_move": -0.05},
            ],
        },
    }
    r = client.post("/api/risk/scenarios/evaluate", json=payload)
    d = r.json()
    log.info("case=http-scenario pnl=%s consistent=%s", d["pnl"], d["consistent_with_single_factor"])
    assert r.status_code == 200
    assert d["consistent_with_single_factor"]


def test_register_and_compose_named_scenario_http():
    client.post(
        "/api/risk/scenarios",
        json={"name": "eq", "shocks": [{"instrument": "A", "relative_move": -0.1}]},
    )
    client.post(
        "/api/risk/scenarios",
        json={"name": "credit", "shocks": [{"instrument": "B", "relative_move": -0.05}]},
    )
    r = client.post(
        "/api/risk/scenarios/compose",
        json={"result_name": "combo", "names": ["eq", "credit"]},
    )
    log.info("case=http-compose status=%s body=%s", r.status_code, r.json())
    assert r.status_code == 200
    b = _valid_body()
    r = client.post(
        "/api/risk/scenarios/evaluate",
        json={"portfolio": b["portfolio"], "market": b["market"], "scenario_name": "combo"},
    )
    assert r.status_code == 200
    assert r.json()["scenario"] == "combo"


def test_missing_scenario_reference_http():
    b = _valid_body()
    r = client.post(
        "/api/risk/scenarios/evaluate",
        json={"portfolio": b["portfolio"], "market": b["market"], "scenario_name": "nope"},
    )
    log.info("case=http-missing-scenario code=%s", r.json()["error"]["code"])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNKNOWN_SCENARIO"
