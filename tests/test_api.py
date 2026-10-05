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


def test_scan_endpoint_order_isolation_and_limit():
    # register two scenarios (unique names to avoid clashing across tests)
    for name, shock in [("scan_drop", -0.1), ("scan_rally", 0.05)]:
        client.post(
            "/api/risk/scenarios",
            json={"name": name,
                  "shocks": [{"instrument": "A", "relative_move": shock}]},
        )
    job = {
        "portfolio": {"name": "fund-s",
                      "holdings": [{"instrument": "A", "quantity": 10}]},
        "market": {"dates": _dates(3),
                   "prices": {"A": [100.0, 101.0, 102.0]}},
        "scenario_names": ["scan_drop", "scan_missing", "scan_rally"],
    }
    r = client.post("/api/risk/scan", json={"jobs": [job, job]})
    d = r.json()
    log.info("case=http-scan status=%s count=%s", r.status_code, d.get("count"))
    assert r.status_code == 200
    assert d["count"] == 6
    oks = [item["ok"] for item in d["results"]]
    assert oks == [True, False, True, True, False, True]
    assert d["results"][1]["error"]["code"] == "UNKNOWN_SCENARIO"
    assert d["results"][0]["result"]["scenario"] == "scan_drop"
    assert d["results"][5]["result"]["scenario"] == "scan_rally"
    # repeated submission is deterministic
    r2 = client.post("/api/risk/scan", json={"jobs": [job, job]})
    assert r2.json() == d
    # expanded pair count is what the item limit sees
    r3 = client.post("/api/risk/scan",
                     json={"jobs": [job, job], "max_items": 5})
    assert r3.status_code == 429
    assert r3.json()["error"]["code"] == "BATCH_SIZE_LIMIT"


def test_reassess_endpoint_verdicts():
    from risk_engine.versions import RulesVersion

    svc = app.state.risk_service
    for v, parents in [("v2", {"v1"}), ("v3", set())]:
        try:
            svc.register_version(
                RulesVersion(v, "2026-06-01", "v1", frozenset(parents))
            )
        except ValueError:
            pass  # already registered by an earlier run of this test

    body = _valid_body()
    original = client.post("/api/risk/assess", json=body).json()

    def job(target, allow=False):
        return {
            "portfolio": body["portfolio"],
            "market": body["market"],
            "stamped_version": "v1",
            "target_version": target,
            "original": original,
            "window": body["window"],
            "allow_breaking": allow,
        }

    r = client.post("/api/risk/reassess",
                    json={"jobs": [job("v2"), job("v3"), job("v3", allow=True)]})
    d = r.json()
    verdicts = [item["result"]["verdict"] for item in d["results"]]
    log.info("case=http-reassess verdicts=%s", verdicts)
    assert r.status_code == 200
    assert verdicts == ["REPRODUCED", "REJECTED", "REPRODUCED"]
    assert d["results"][1]["result"]["error"]["code"] == "INCOMPATIBLE_VERSION"
    assert d["results"][2]["result"]["recomputed"]["version"] == "v3"


def test_scan_snapshot_endpoint_deterministic():
    for name, shock in [("snap_drop", -0.1), ("snap_rally", 0.05)]:
        client.post(
            "/api/risk/scenarios",
            json={"name": name,
                  "shocks": [{"instrument": "A", "relative_move": shock}]},
        )
    job = {
        "portfolio": {"name": "fund-p",
                      "holdings": [{"instrument": "A", "quantity": 4}]},
        "market": {"dates": _dates(3), "prices": {"A": [10.0, 11.0, 12.0]}},
        "scenario_names": ["snap_drop", "snap_rally"],
    }
    body = {"jobs": [job]}
    r1 = client.post("/api/risk/scan/snapshot", json=body)
    r2 = client.post("/api/risk/scan/snapshot", json=body)
    log.info("case=http-snapshot status=%s", r1.status_code)
    assert r1.status_code == 200
    d = r1.json()
    assert d["format"] == "risk-engine-snapshot/1"
    assert d["count"] == 2
    assert r1.content == r2.content  # byte-identical over the wire
    assert d["rankings"]["worst_by_portfolio"]["items"][0]["name"] == "fund-p"
    assert d["rankings"]["worst_by_scenario"]["items"][0]["name"] == "snap_drop"
    assert "rationale" in d["entries"][0]


def test_batch_bad_date_isolated_http():
    good = _valid_body()
    bad = _valid_body()
    bad["market"] = {"dates": ["2026-13-99"], "prices": {"A": [1.0]}}
    r = client.post("/api/risk/batch", json={"requests": [good, bad, good]})
    d = r.json()
    flags = [(x["index"], x["ok"], (x["error"] or {}).get("code"))
             for x in d["results"]]
    log.info(
        "case=http-batch-bad-date flags=%s; criterion: only the malformed "
        "slot fails with INVALID_INPUT, order preserved",
        flags,
    )
    assert r.status_code == 200
    assert flags == [(0, True, None), (1, False, "INVALID_INPUT"), (2, True, None)]


def test_scan_bad_date_job_isolated_http():
    client.post(
        "/api/risk/scenarios",
        json={"name": "iso_drop",
              "shocks": [{"instrument": "A", "relative_move": -0.1}]},
    )
    good_job = {
        "portfolio": {"name": "iso-ok",
                      "holdings": [{"instrument": "A", "quantity": 2}]},
        "market": {"dates": _dates(3), "prices": {"A": [10.0, 11.0, 12.0]}},
        "scenario_names": ["iso_drop"],
    }
    bad_job = {
        "portfolio": {"name": "iso-broken",
                      "holdings": [{"instrument": "A", "quantity": 1}]},
        "market": {"dates": ["not-a-date"], "prices": {"A": [1.0]}},
        "scenario_names": ["iso_drop", "iso_drop"],
    }
    r = client.post("/api/risk/scan",
                    json={"jobs": [good_job, bad_job, good_job]})
    flags = [(x["index"], x["ok"], (x["error"] or {}).get("code"))
             for x in r.json()["results"]]
    log.info(
        "case=http-scan-bad-date flags=%s; criterion: broken job fails its "
        "own two slots only, others unaffected",
        flags,
    )
    assert r.status_code == 200
    assert flags == [
        (0, True, None),
        (1, False, "INVALID_INPUT"),
        (2, False, "INVALID_INPUT"),
        (3, True, None),
    ]
    # the failure lands in the snapshot with summary, rationale, class
    r2 = client.post("/api/risk/scan/snapshot",
                     json={"jobs": [good_job, bad_job]})
    entries = r2.json()["entries"]
    broken = entries[1]
    log.info(
        "case=http-scan-bad-date-snapshot entry=%s; criterion: failed entry "
        "carries summary + INVALID_INPUT rationale + failure_class",
        {k: broken[k] for k in ("ok", "failure_class", "rationale")},
    )
    assert broken["ok"] is False
    assert broken["failure_class"] == "invalid_input"
    assert "INVALID_INPUT" in broken["rationale"]
    assert broken["input_summary"]["portfolio"] == "iso-broken"
    assert entries[0]["ok"] is True and entries[0]["version"] == "v1"


def test_reassess_bad_date_isolated_http():
    body = _valid_body()
    original = client.post("/api/risk/assess", json=body).json()

    def job(market):
        return {
            "portfolio": body["portfolio"],
            "market": market,
            "stamped_version": "v1",
            "target_version": "v1",
            "original": original,
            "window": body["window"],
        }

    bad_market = {"dates": ["2026-02-30"], "prices": body["market"]["prices"]}
    r = client.post("/api/risk/reassess",
                    json={"jobs": [job(body["market"]), job(bad_market)]})
    flags = [(x["index"], x["ok"], (x["error"] or {}).get("code"))
             for x in r.json()["results"]]
    log.info(
        "case=http-reassess-bad-date flags=%s; criterion: bad date fails "
        "only its own slot with INVALID_INPUT",
        flags,
    )
    assert r.status_code == 200
    assert flags == [(0, True, None), (1, False, "INVALID_INPUT")]
    assert r.json()["results"][0]["result"]["verdict"] == "REPRODUCED"


def test_scan_snapshot_export_endpoint_readable_and_byte_stable():
    client.post(
        "/api/risk/scenarios",
        json={"name": "exp_drop",
              "shocks": [{"instrument": "A", "relative_move": -0.1}]},
    )
    body = {
        "jobs": [{
            "portfolio": {"name": "exp-fund",
                          "holdings": [{"instrument": "A", "quantity": 4}]},
            "market": {"dates": _dates(3), "prices": {"A": [10.0, 11.0, 12.0]}},
            "scenario_names": ["exp_drop"],
        }]
    }
    r1 = client.post("/api/risk/scan/snapshot/export", json=body)
    r2 = client.post("/api/risk/scan/snapshot/export", json=body)
    log.info(
        "case=http-export status=%s bytes=%d; criterion: readable text, "
        "byte-identical across repeated exports",
        r1.status_code, len(r1.content),
    )
    assert r1.status_code == 200
    assert r1.headers["content-type"].startswith("text/plain")
    assert r1.content == r2.content
    text = r1.text
    assert text.startswith("# Risk Engine Scan Snapshot")
    assert "format: risk-engine-snapshot/1" in text
    assert "portfolio=exp-fund" in text and "scenario=exp_drop" in text
    assert "rules=v1" in text and "why:" in text
    assert "### worst_by_portfolio" in text


def test_assess_validation_boundary_http():
    bad_conf = _valid_body()
    bad_conf["confidence"] = 1.5
    r1 = client.post("/api/risk/assess", json=bad_conf)
    bad_window = _valid_body()
    bad_window["window"] = 0
    r2 = client.post("/api/risk/assess", json=bad_window)
    log.info(
        "case=http-validation confidence=1.5 -> %s, window=0 -> %s; "
        "criterion: both rejected 422",
        r1.status_code, r2.status_code,
    )
    assert r1.status_code == 422
    assert r2.status_code == 422


def test_batch_and_scan_empty_http():
    r1 = client.post("/api/risk/batch", json={"requests": []})
    r2 = client.post("/api/risk/scan", json={"jobs": []})
    r3 = client.post("/api/risk/scan/snapshot", json={"jobs": []})
    log.info(
        "case=http-empty batch=%s scan=%s snapshot=%s; criterion: empty in, "
        "well-formed empty out",
        r1.status_code, r2.status_code, r3.status_code,
    )
    assert r1.status_code == 200 and r1.json() == {"count": 0, "results": []}
    assert r2.status_code == 200 and r2.json() == {"count": 0, "results": []}
    assert r3.status_code == 200
    assert r3.json()["count"] == 0 and r3.json()["entries"] == []
