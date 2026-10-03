"""HTTP integration for scan, snapshot export and cross-version recompare."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from main import app

from .conftest import log

client = TestClient(app)


def _market(n=60):
    from datetime import date, timedelta
    dates = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]
    return {
        "dates": dates,
        "prices": {
            "A": [100 + 0.3 * i for i in range(n)],
            "B": [50 + (i % 5) * 0.4 for i in range(n)],
            "C": [70 + 0.1 * i for i in range(n)],
        },
    }


def _portfolio(name, insts=("A", "B", "C")):
    qty = {"A": 10, "B": -5, "C": 4}
    return {
        "name": name,
        "holdings": [
            {"instrument": i, "quantity": qty[i]} for i in insts
        ],
    }


def _register_scenarios():
    for name, inst, move in (("eq", "A", -0.1), ("credit", "B", -0.05)):
        r = client.post(
            "/api/risk/scenarios",
            json={"name": name,
                  "shocks": [{"instrument": inst, "relative_move": move}]},
        )
        assert r.status_code in (200, 409), r.text


def test_scan_endpoint_order_isolation_and_equivalence():
    _register_scenarios()
    body = {
        "portfolios": [_portfolio("f1"), _portfolio("f2"), _portfolio("f3")],
        "market": _market(),
        "scenario_names": ["eq", "credit"],
    }
    r = client.post("/api/risk/scan", json=body)
    d = r.json()
    log.info("case=http-scan status=%s count=%s", r.status_code, d.get("count"))
    assert r.status_code == 200
    assert d["count"] == 6
    # fixed submission order: portfolio outer, scenario inner
    order = [
        (x["result"]["input"]["portfolio"], x["result"]["scenario"])
        for x in d["results"]
    ]
    assert order == [
        ("f1", "eq"), ("f1", "credit"),
        ("f2", "eq"), ("f2", "credit"),
        ("f3", "eq"), ("f3", "credit"),
    ]
    # equivalence with single-run endpoint
    single = client.post(
        "/api/risk/scenarios/evaluate",
        json={
            "portfolio": _portfolio("f2"),
            "market": _market(),
            "scenario_name": "credit",
        },
    ).json()
    cell = d["results"][3]["result"]
    assert cell["pnl"] == single["pnl"]


def test_scan_unknown_scenario_is_slot_error():
    body = {
        "portfolios": [_portfolio("f1")],
        "market": _market(),
        "scenario_names": ["eq", "ghost"],
    }
    r = client.post("/api/risk/scan", json=body)
    slots = [(x["index"], x["ok"], (x["error"] or {}).get("code"))
             for x in r.json()["results"]]
    log.info("case=http-scan-isolation %s", slots)
    assert slots == [(0, True, None), (1, False, "UNKNOWN_SCENARIO")]


def test_scan_capacity_limit_http():
    body = {
        "portfolios": [_portfolio("f1"), _portfolio("f2")],
        "market": _market(),
        "scenario_names": ["eq", "credit"],  # 4 cells
        "max_items": 3,
    }
    r = client.post("/api/risk/scan", json=body)
    log.info("case=http-scan-limit status=%s code=%s",
             r.status_code, r.json()["error"]["code"])
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "BATCH_SIZE_LIMIT"


def test_snapshot_endpoint_returns_deterministic_html():
    body = {
        "cells": [
            {
                "portfolio": _portfolio("f1"),
                "market": _market(),
                "scenario_name": "eq",
            },
            {
                "portfolio": _portfolio("f1"),
                "market": _market(),
                "scenario_name": "credit",
            },
        ],
        "title": "http snapshot",
    }
    r1 = client.post("/api/risk/scan/snapshot", json=body)
    r2 = client.post("/api/risk/scan/snapshot", json=body)
    log.info("case=http-snapshot bytes=%d", len(r1.text))
    assert r1.status_code == 200
    assert r1.headers["content-type"].startswith("text/html")
    assert r1.text == r2.text
    assert r1.text.lstrip().startswith("<!DOCTYPE html>")
    assert "Worst by portfolio" in r1.text


def test_recompare_endpoint_three_verdicts():
    # register v2 (compatible) and v3 (breaking)
    from risk_engine.service import RiskService
    from risk_engine.versions import RulesVersion

    svc = app.state.risk_service
    for v, parents in (("v2", frozenset({"v1"})),):
        try:
            svc.register_version(
                RulesVersion(v, "2026-06-01", "v1", parents, "same")
            )
        except ValueError:
            pass
    try:
        svc.register_version(
            RulesVersion("v3", "2026-09-01", "v2", frozenset(), "new")
        )
    except ValueError:
        pass

    # produce an old v1 conclusion over HTTP
    old = client.post(
        "/api/risk/assess",
        json={
            "portfolio": _portfolio("f1"),
            "market": _market(),
            "window": 20,
        },
    ).json()

    # 1) compatible -> REPRODUCED
    r = client.post("/api/risk/recompare", json={
        "items": [{
            "portfolio": _portfolio("f1"),
            "market": _market(),
            "stamped_version": "v1",
            "conclusion": old,
            "window": 20,
        }],
        "target_version": "v2",
    })
    log.info("case=http-recompare reproduced=%s", r.json()["results"][0]["verdict"])
    assert r.json()["results"][0]["verdict"] == "REPRODUCED"

    # 2) breaking without approval -> REJECTED
    r = client.post("/api/risk/recompare", json={
        "items": [{
            "portfolio": _portfolio("f1"),
            "market": _market(),
            "stamped_version": "v1",
            "conclusion": old,
            "window": 20,
        }],
        "target_version": "v3",
    })
    item = r.json()["results"][0]
    log.info("case=http-recompare rejected verdict=%s code=%s",
             item["verdict"], item["error"]["code"])
    assert item["verdict"] == "REJECTED"
    assert item["error"]["code"] == "INCOMPATIBLE_VERSION"

    # 3) explicit approval + tampered old numbers -> CHANGED
    changed_old = dict(old)
    changed_old["historical"] = dict(old["historical"])
    changed_old["historical"]["var"] += 99.0
    r = client.post("/api/risk/recompare", json={
        "items": [{
            "portfolio": _portfolio("f1"),
            "market": _market(),
            "stamped_version": "v1",
            "conclusion": changed_old,
            "window": 20,
        }],
        "target_version": "v3",
        "allow_breaking": True,
    })
    item = r.json()["results"][0]
    log.info("case=http-recompare changed verdict=%s", item["verdict"])
    assert item["verdict"] == "CHANGED"
    assert item["new_conclusion"]["version"] == "v3"
