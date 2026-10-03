"""Cross-version batch reassessment with per-item verdicts.

Each job carries a conclusion previously stamped at ``stamped_version``
plus the inputs that produced it.  The batch recomputes every item under
``target_version`` and classifies it:

* ``REPRODUCED`` — recomputed conclusion is bit-for-bit identical to the
  original; the old conclusion still holds under the new rules.
* ``CHANGED``    — recomputation was allowed but the conclusion differs;
  the metric definition has effectively moved.
* ``REJECTED``   — recomputation was refused (incompatible versions
  without explicit opt-in, unknown version, bad input, ...).  The item
  carries the typed error; other items are unaffected.

A breaking upgrade is never applied silently: incompatible pairs are
REJECTED unless the caller passes ``allow_breaking=True`` on the job, in
which case the recomputed conclusion is stamped with the new version.
"""

from __future__ import annotations

from dataclasses import dataclass

from risk_engine.models import CorrelationAssumption, MarketData, Portfolio

VERDICT_REPRODUCED = "REPRODUCED"
VERDICT_CHANGED = "CHANGED"
VERDICT_REJECTED = "REJECTED"


@dataclass(frozen=True)
class ReassessJob:
    portfolio: Portfolio
    market: MarketData
    stamped_version: str
    target_version: str
    original: dict  # original conclusion payload (report_to_dict shape)
    window: int = 20
    confidence: float = 0.99
    horizon_days: int = 1
    correlation: CorrelationAssumption | None = None
    allow_breaking: bool = False


def conclusions_equal(original: dict, recomputed: dict) -> bool:
    """Bit-for-bit comparison of two conclusion payloads.

    The engine is deterministic, so a reproducible conclusion compares
    exactly equal — including every float.  The ``version`` field is
    excluded: a recomputed conclusion is always stamped with the target
    version, and a version bump alone does not make a conclusion
    "changed".
    """
    o = {k: v for k, v in original.items() if k != "version"}
    r = {k: v for k, v in recomputed.items() if k != "version"}
    return o == r
