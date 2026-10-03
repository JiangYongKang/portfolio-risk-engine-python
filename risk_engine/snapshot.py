"""Deterministic result snapshots and worst-case rankings.

Snapshot
--------
A snapshot freezes a batch of results into a self-contained, locally
viewable document.  ``canonical_json`` serializes it deterministically:
keys sorted, floats via shortest-round-trip repr, NaN/Infinity rejected.
Same inputs -> byte-identical snapshot, every time.

Every entry carries an input summary, the rules-version stamp and a
rationale string explaining why the conclusion came out that way.

Rankings
--------
Two worst-case views over the ok entries of a scan:

* by portfolio: each portfolio's worst (minimum) scenario PnL;
* by scenario:  each scenario's worst (minimum) PnL across portfolios.

The metric is stated explicitly (``metric``/``order`` in the payload).
Ties are broken by name, then by submission index, so the same batch
always ranks identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from risk_engine.models import BatchItemResult

SNAPSHOT_FORMAT = "risk-engine-snapshot/1"


@dataclass(frozen=True)
class SnapshotEntry:
    index: int
    kind: str               # "assess" | "scenario"
    input_summary: dict
    version: str
    ok: bool
    rationale: str
    result: dict | None
    error: dict | None


def canonical_json(obj) -> str:
    """Deterministic serialization: sorted keys, exact floats, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def build_snapshot(entries: list[SnapshotEntry], *, rankings: dict) -> dict:
    """Assemble the snapshot document from entries and rankings."""
    return {
        "format": SNAPSHOT_FORMAT,
        "count": len(entries),
        "entries": [
            {
                "index": e.index,
                "kind": e.kind,
                "input_summary": e.input_summary,
                "version": e.version,
                "ok": e.ok,
                "rationale": e.rationale,
                "result": e.result,
                "error": e.error,
            }
            for e in entries
        ],
        "rankings": rankings,
    }


def rank_worst(
    entries: Iterable[SnapshotEntry], *, group_by: str, metric: str = "pnl"
) -> dict:
    """Worst-case ranking of ``group_by`` values by ``metric``.

    ``group_by`` is "portfolio" or "scenario".  Worst = smallest metric
    value (most negative PnL).  Ties break by name, then first index.
    """
    best: dict[str, tuple[float, int]] = {}
    for e in entries:
        if not e.ok or e.result is None:
            continue
        name = e.input_summary.get(group_by)
        value = e.result.get(metric)
        if name is None or value is None:
            continue
        current = best.get(name)
        if current is None or value < current[0]:
            best[name] = (value, e.index)
    rows = sorted(
        ((value, name, first_index) for name, (value, first_index) in best.items()),
        key=lambda r: (r[0], r[1], r[2]),
    )
    return {
        "group_by": group_by,
        "metric": metric,
        "order": "ascending: worst (smallest) value first",
        "tie_break": "equal values ordered by name, then first submission index",
        "items": [
            {"rank": i + 1, "name": name, "value": value}
            for i, (value, name, _idx) in enumerate(rows)
        ],
    }
