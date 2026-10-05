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

from risk_engine.errors import classify_failure
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
    failure_class: str | None = None  # see errors.classify_failure


def canonical_json(obj) -> str:
    """Deterministic serialization: sorted keys, exact floats, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def render_snapshot_text(snapshot: dict) -> str:
    """Render a snapshot as a human-readable, byte-stable text document.

    The rendering contains no timestamps or run-dependent data, so the
    same inputs always produce byte-identical text — safe to archive,
    diff or send around.  Every entry shows its input summary, the rules
    version and the rationale (including the rejection reason and the
    coarse failure class for failed entries).
    """
    entries = snapshot["entries"]
    ok_count = sum(1 for e in entries if e["ok"])
    lines = [
        "# Risk Engine Scan Snapshot",
        f"format: {snapshot['format']}",
        f"entries: {len(entries)} (ok={ok_count}, "
        f"failed={len(entries) - ok_count})",
        "",
        "## Rankings",
    ]
    for name, ranking in snapshot["rankings"].items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append(
            f"metric={ranking['metric']}; order={ranking['order']}; "
            f"tie_break={ranking['tie_break']}"
        )
        if not ranking["items"]:
            lines.append("  (no evaluated entries)")
        for item in ranking["items"]:
            lines.append(f"  {item['rank']}. {item['name']}  {item['value']!r}")
    lines.append("")
    lines.append("## Entries")
    for e in entries:
        summary = " ".join(
            f"{k}={v}" for k, v in sorted(e["input_summary"].items())
        )
        if e["ok"]:
            head = f"[{e['index']}] OK | rules={e['version']}"
        else:
            head = f"[{e['index']}] FAILED [{e['failure_class']}]"
        lines.append("")
        lines.append(f"{head} | {summary}")
        lines.append(f"    why: {e['rationale']}")
        if e["ok"] and e["result"] is not None:
            lines.append(f"    result: {canonical_json(e['result'])}")
        elif e["error"] is not None:
            lines.append(
                f"    error: {e['error']['code']} — {e['error']['reason']}"
            )
    return "\n".join(lines) + "\n"


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
                "failure_class": e.failure_class,
            }
            for e in entries
        ],
        "rankings": rankings,
    }


def scenario_rationale(summary: dict, result: dict) -> str:
    """Human-readable 'why this conclusion' for an evaluated scenario pair."""
    return (
        f"scenario '{summary['scenario']}' evaluated under rules "
        f"{result['version']} on portfolio '{summary['portfolio']}': "
        f"pnl={result['pnl']!r}; "
        f"consistent_with_single_factor="
        f"{result['consistent_with_single_factor']}"
    )


def failure_rationale(error: dict) -> str:
    """Human-readable rejection reason for a failed entry."""
    return f"rejected: {error['code']} — {error['reason']}"


def entry_from_result(
    index: int,
    *,
    kind: str,
    input_summary: dict,
    item: BatchItemResult,
) -> SnapshotEntry:
    """Build a snapshot entry from one batch item result.

    Ok entries are stamped with the rules version the engine reported;
    failed entries carry the typed error, a rejection rationale and the
    coarse ``failure_class`` (timeout / over_limit / invalid_input /
    internal_error).
    """
    if item.ok and item.result is not None:
        return SnapshotEntry(
            index=index,
            kind=kind,
            input_summary=input_summary,
            version=item.result.get("version", ""),
            ok=True,
            rationale=scenario_rationale(input_summary, item.result),
            result=item.result,
            error=None,
            failure_class=None,
        )
    error = item.error or {"code": "INTERNAL_ERROR", "reason": "no result"}
    return SnapshotEntry(
        index=index,
        kind=kind,
        input_summary=input_summary,
        version="",
        ok=False,
        rationale=failure_rationale(error),
        result=None,
        error=error,
        failure_class=classify_failure(error["code"]),
    )


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
