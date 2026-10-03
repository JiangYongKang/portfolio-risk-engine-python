"""Deterministic snapshots + two-direction worst-case rankings."""

from __future__ import annotations

import json

from risk_engine.models import FactorShock, Holding, Portfolio, StressScenario
from risk_engine.scan import ScanCell, ScanLimits, cross_product_cells
from risk_engine.service import RiskService
from risk_engine.snapshot import (
    RANK_METRIC,
    TIE_BREAK,
    build_snapshot,
    canonical_json,
    rank_by_portfolio,
    rank_by_scenario,
)

from .conftest import log


def _svc() -> RiskService:
    svc = RiskService()
    svc.register_scenario(StressScenario("s1", (FactorShock("A", -0.10),)))
    svc.register_scenario(StressScenario("s2", (FactorShock("A", -0.30),)))
    svc.register_scenario(StressScenario("s3", (FactorShock("B", -0.20),)))
    return svc


def _cells(market):
    p1 = Portfolio("alpha", (Holding("A", 100.0), Holding("B", 10.0)))
    p2 = Portfolio("beta", (Holding("A", 50.0), Holding("B", 80.0)))
    return cross_product_cells(
        {"alpha": (p1, market), "beta": (p2, market)},
        ["s1", "s2", "s3"],
    )


def test_snapshot_byte_identical_across_repeats(market):
    svc = _svc()
    cells = _cells(market)
    snap1 = svc.export_scan_snapshot(cells, title="stable")
    snap2 = svc.export_scan_snapshot(cells, title="stable")
    log.info("case=snapshot-determinism len=%d", len(snap1.html))
    assert snap1.html == snap2.html
    assert snap1.canonical_json == snap2.canonical_json
    # no environmental leakage: no timestamp-ish fields
    data = json.loads(snap1.canonical_json)
    assert set(data) >= {"cells", "rank_by_portfolio", "rank_by_scenario",
                         "ranking_metric", "tie_break"}


def test_rankings_worst_pnl_and_order(market):
    svc = _svc()
    out = svc.scan(_cells(market), ScanLimits(max_items=32, timeout_seconds=10))
    snap = build_snapshot(out)
    d = snap.data
    log.info("case=rank portfolio=%s scenario=%s",
             d["rank_by_portfolio"], d["rank_by_scenario"])
    # metric + tie-break are documented
    assert d["ranking_metric"] == RANK_METRIC
    assert "lexicographically" in d["tie_break"]

    rp = d["rank_by_portfolio"]
    # worst for each portfolio is s2 (A -30%); sorted worst-first
    assert [r["name"] for r in rp] == ["alpha", "beta"]
    assert rp[0]["worst_partner"] == "s2"
    pnls_p = [r["worst_pnl"] for r in rp]
    assert pnls_p == sorted(pnls_p)

    rs = d["rank_by_scenario"]
    # s2 hits A with -30%: alpha holds more A -> alpha is worst partner
    s2 = next(r for r in rs if r["name"] == "s2")
    assert s2["worst_partner"] == "alpha"
    pnls_s = [r["worst_pnl"] for r in rs]
    assert pnls_s == sorted(pnls_s)


def test_ranking_ties_broken_by_name(market):
    # Two portfolios with identical exposure -> identical worst PnL;
    # ranking must fall back to name ordering.
    svc = _svc()
    same1 = Portfolio("zeta", (Holding("A", 100.0),))
    same2 = Portfolio("alpha", (Holding("A", 100.0),))
    cells = cross_product_cells(
        {"zeta": (same1, market), "alpha": (same2, market)}, ["s1", "s2"]
    )
    out = svc.scan(cells, ScanLimits(max_items=16, timeout_seconds=10))
    entries = rank_by_portfolio(build_snapshot(out).data["cells"])
    log.info("case=rank-tie names=%s", [e.name for e in entries])
    assert entries[0].worst_pnl == entries[1].worst_pnl
    assert [e.name for e in entries] == ["alpha", "zeta"]


def test_cell_rows_carry_input_version_rationale(market):
    svc = _svc()
    snap = svc.export_scan_snapshot(_cells(market))
    for cell in snap.data["cells"]:
        assert cell["ok"]
        assert cell["input"]["portfolio"] in ("alpha", "beta")
        assert cell["input"]["scenario"] in ("s1", "s2", "s3")
        assert cell["version"] == "v1"
        assert "PnL" in cell["rationale"]
    log.info("case=cell-metadata sample=%s", snap.data["cells"][0]["input"])


def test_failed_cells_excluded_from_rankings_but_listed(market):
    svc = _svc()
    good = Portfolio("ok", (Holding("A", 10.0),))
    bad = Portfolio("bad", (Holding("ZZ", 10.0),))
    cells = [
        ScanCell(good, market, "s1"),
        ScanCell(bad, market, "s1"),
    ]
    snap = svc.export_scan_snapshot(cells)
    names = [r["name"] for r in snap.data["rank_by_portfolio"]]
    log.info("case=rank-excludes-failed names=%s", names)
    assert names == ["ok"]
    statuses = [(c["index"], c["ok"], (c["error"] or {}).get("code"))
                for c in snap.data["cells"]]
    assert statuses == [(0, True, None), (1, False, "UNKNOWN_INSTRUMENT")]


def test_snapshot_html_is_self_contained(market, tmp_path):
    svc = _svc()
    snap = svc.export_scan_snapshot(_cells(market), title="local view")
    assert snap.html.lstrip().startswith("<!DOCTYPE html>")
    assert "<script id=\"snapshot-data\"" in snap.html
    assert "application/json" in snap.html
    # opens locally with no external assets
    assert "http://" not in snap.html and "https://" not in snap.html
    path = tmp_path / "snap.html"
    path.write_text(snap.html, encoding="utf-8")
    reread = path.read_text(encoding="utf-8")
    assert reread == snap.html
    log.info("case=html-file bytes=%d", len(reread))


def test_canonical_json_is_key_sorted():
    payload = {"b": 1, "a": {"z": 1, "y": [1, 2]}}
    encoded = canonical_json(payload)
    assert encoded == '{"a":{"y":[1,2],"z":1},"b":1}'
    log.info("case=canonical-json %s", encoded)
