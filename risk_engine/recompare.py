"""Cross-version recomparison of historical risk conclusions.

Each item carries the conclusion originally produced under an old rules
version.  The item is recomputed under a target version and classified:

* ``REPRODUCED``  - recomputation allowed and the new conclusion equals
  the old one bit-for-bit;
* ``CHANGED``     - recomputation only possible with an explicit breaking
  opt-in, and the new conclusion differs (never applied silently);
* ``REJECTED``    - recomputation not permitted / failed; the historical
  conclusion is left untouched.

Per-item failures never affect siblings and submission order is fixed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from risk_engine.batch import run_batch
from risk_engine.models import MarketData, Portfolio, Version


# Verdict codes -------------------------------------------------------------

REPRODUCED = "REPRODUCED"
CHANGED = "CHANGED"
REJECTED = "REJECTED"


@dataclass(frozen=True)
class HistoricalConclusion:
    """A previously issued risk conclusion to be re-examined.

    ``conclusion`` is the canonical result payload (as produced by
    ``report_to_dict``) that the recomputed value is compared against.
    """

    portfolio: Portfolio
    market: MarketData
    stamped_version: Version
    conclusion: dict
    window: int = 20
    confidence: float = 0.99
    horizon_days: int = 1


@dataclass(frozen=True)
class RecompareItem:
    index: int
    portfolio: str
    stamped_version: Version
    target_version: Version
    verdict: str
    reason: str
    old_conclusion: dict | None = None
    new_conclusion: dict | None = None
    error: dict | None = None


@dataclass(frozen=True)
class RecompareLimits:
    max_items: int = 256
    max_workers: int = 8
    timeout_seconds: float = 10.0

    def to_batch_limits(self):
        from risk_engine.batch import BatchLimits

        return BatchLimits(
            max_items=self.max_items,
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
        )


def conclusions_equal(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Bit-for-bit equality of canonical conclusion payloads.

    The ``version`` stamp is intentionally excluded: an identical numerical
    conclusion re-stamped under a compatible new version still reproduces.
    """
    keys = set(old) | set(new)
    for key in keys:
        if key == "version":
            continue
        if old.get(key) != new.get(key):
            return False
    return True


def _classify(
    item: HistoricalConclusion,
    target_version: str,
    allow_breaking: bool,
    new_payload: dict,
) -> RecompareItem:
    if conclusions_equal(item.conclusion, new_payload):
        return RecompareItem(
            index=-1,
            portfolio=item.portfolio.name,
            stamped_version=item.stamped_version,
            target_version=target_version,
            verdict=REPRODUCED,
            reason=(
                "recomputed conclusion matches the historical one "
                "(all metrics bit-for-bit equal); restamped under "
                f"{target_version}"
            ),
            old_conclusion=item.conclusion,
            new_conclusion=new_payload,
        )
    # Numbers differ. If the upgrade was a breaking one this must have
    # required an explicit opt-in; the difference is reported as CHANGED,
    # never silently overwriting the old conclusion.
    breaking = allow_breaking and item.stamped_version != target_version
    return RecompareItem(
        index=-1,
        portfolio=item.portfolio.name,
        stamped_version=item.stamped_version,
        target_version=target_version,
        verdict=CHANGED,
        reason=(
            "metrics differ under the new rules; the historical conclusion "
            "is NOT overwritten"
            + (" (breaking recompute explicitly approved)" if breaking else "")
        ),
        old_conclusion=item.conclusion,
        new_conclusion=new_payload,
    )


def recompare(
    items: list[HistoricalConclusion],
    evaluator: Callable[[HistoricalConclusion, str, bool], dict],
    *,
    target_version: str,
    allow_breaking: bool = False,
    limits: RecompareLimits | None = None,
) -> list[RecompareItem]:
    """Recompute and classify a batch of historical conclusions.

    ``evaluator(item, target_version, allow_breaking)`` performs the gated
    recomputation and returns the new canonical payload, or raises a
    :class:`RiskEngineError` (e.g. ``INCOMPATIBLE_VERSION`` when a breaking
    upgrade was not approved).  One item's rejection/failure never affects
    another and output order matches submission order.
    """
    limits = limits or RecompareLimits()

    def _work(req: tuple[HistoricalConclusion, int]) -> dict:
        hc, idx = req
        payload = evaluator(hc, target_version, allow_breaking)
        classified = _classify(hc, target_version, allow_breaking, payload)
        # serialize via a plain dict for the generic batch runner
        return {
            "index": idx,
            "portfolio": classified.portfolio,
            "stamped_version": classified.stamped_version,
            "target_version": classified.target_version,
            "verdict": classified.verdict,
            "reason": classified.reason,
            "old_conclusion": classified.old_conclusion,
            "new_conclusion": classified.new_conclusion,
            "error": None,
        }

    raw = run_batch(
        [(hc, i) for i, hc in enumerate(items)],
        _work,
        limits.to_batch_limits(),
    )

    out: list[RecompareItem] = []
    for slot in raw:
        hc = items[slot.index]
        if slot.ok:
            d = slot.result
            out.append(
                RecompareItem(
                    index=slot.index,
                    portfolio=d["portfolio"],
                    stamped_version=d["stamped_version"],
                    target_version=d["target_version"],
                    verdict=d["verdict"],
                    reason=d["reason"],
                    old_conclusion=d["old_conclusion"],
                    new_conclusion=d["new_conclusion"],
                    error=None,
                )
            )
        else:
            err = slot.error or {}
            verdict = REJECTED
            out.append(
                RecompareItem(
                    index=slot.index,
                    portfolio=hc.portfolio.name,
                    stamped_version=hc.stamped_version,
                    target_version=target_version,
                    verdict=verdict,
                    reason=(
                        f"recomputation not allowed/failed ({err.get('code')}); "
                        "historical conclusion left unchanged"
                    ),
                    old_conclusion=hc.conclusion,
                    new_conclusion=None,
                    error=err,
                )
            )
    return out
