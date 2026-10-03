"""Cross-version batch reassessment: REPRODUCED / CHANGED / REJECTED."""

from __future__ import annotations

import pytest

from risk_engine.batch import BatchLimits
from risk_engine.reassess import ReassessJob, conclusions_equal
from risk_engine.service import RiskService, report_to_dict
from risk_engine.versions import RulesVersion

from .conftest import log


@pytest.fixture
def svc() -> RiskService:
    s = RiskService()
    s.register_version(
        RulesVersion("v2", "2026-06-01", "v1", frozenset({"v1"}),
                     notes="compatible refinement")
    )
    s.register_version(
        RulesVersion("v3", "2026-07-01", "v2", frozenset(),
                     notes="breaking: quantile rule changed")
    )
    return s


def _job(portfolio, market, stamped, target, original, **kw):
    return ReassessJob(
        portfolio, market, stamped, target, original, window=10, **kw
    )


def test_verdict_reproduced(svc, portfolio, market):
    original = report_to_dict(svc.assess(portfolio, market, window=10, version="v1"))
    out = svc.batch_reassess([_job(portfolio, market, "v1", "v2", original)])
    log.info("case=reproduced verdict=%s", out[0].result["verdict"])
    assert out[0].ok and out[0].result["verdict"] == "REPRODUCED"
    assert out[0].result["recomputed"]["version"] == "v2"


def test_verdict_changed_when_conclusion_differs(svc, portfolio, market):
    # original was produced with a different window: recomputation under
    # the job's parameters cannot reproduce it -> CHANGED.
    original = report_to_dict(svc.assess(portfolio, market, window=15, version="v1"))
    out = svc.batch_reassess([_job(portfolio, market, "v1", "v2", original)])
    log.info("case=changed verdict=%s", out[0].result["verdict"])
    assert out[0].ok and out[0].result["verdict"] == "CHANGED"


def test_verdict_rejected_without_opt_in(svc, portfolio, market):
    original = report_to_dict(svc.assess(portfolio, market, window=10, version="v1"))
    out = svc.batch_reassess([_job(portfolio, market, "v1", "v3", original)])
    r = out[0].result
    log.info("case=rejected code=%s", r["error"]["code"])
    assert r["verdict"] == "REJECTED"
    assert r["error"]["code"] == "INCOMPATIBLE_VERSION"
    assert r["recomputed"] is None


def test_breaking_recompute_requires_explicit_opt_in(svc, portfolio, market):
    original = report_to_dict(svc.assess(portfolio, market, window=10, version="v1"))
    job = _job(portfolio, market, "v1", "v3", original, allow_breaking=True)
    out = svc.batch_reassess([job])
    r = out[0].result
    log.info("case=breaking-opt-in verdict=%s version=%s",
             r["verdict"], r["recomputed"]["version"])
    assert r["verdict"] in ("REPRODUCED", "CHANGED")
    assert r["recomputed"]["version"] == "v3"  # re-stamped, never silent


def test_batch_isolation_and_order(svc, portfolio, market):
    original = report_to_dict(svc.assess(portfolio, market, window=10, version="v1"))
    changed_original = report_to_dict(
        svc.assess(portfolio, market, window=15, version="v1")
    )
    jobs = [
        _job(portfolio, market, "v1", "v2", original),               # REPRODUCED
        _job(portfolio, market, "v1", "v3", original),               # REJECTED
        _job(portfolio, market, "v1", "v2", changed_original),       # CHANGED
        _job(portfolio, market, "v9", "v2", original),               # REJECTED (unknown)
        _job(portfolio, market, "v1", "v2", original),               # REPRODUCED
    ]
    out = svc.batch_reassess(jobs, BatchLimits(max_items=8, max_workers=4,
                                               timeout_seconds=10))
    verdicts = [r.result["verdict"] for r in out]
    log.info("case=batch-verdicts %s; criterion: order kept, failures isolated",
             verdicts)
    assert [r.index for r in out] == list(range(5))
    assert verdicts == ["REPRODUCED", "REJECTED", "CHANGED", "REJECTED",
                        "REPRODUCED"]
    assert out[3].result["error"]["code"] == "UNKNOWN_VERSION"


def test_conclusions_equal_ignores_version_stamp():
    a = {"version": "v1", "var": 1.5, "nested": {"x": 0.1 + 0.2}}
    b = {"version": "v2", "var": 1.5, "nested": {"x": 0.1 + 0.2}}
    c = {"version": "v2", "var": 1.5, "nested": {"x": 0.3000000000000001}}
    log.info("case=compare; criterion: version ignored, floats exact")
    assert conclusions_equal(a, b)
    assert not conclusions_equal(a, c)
