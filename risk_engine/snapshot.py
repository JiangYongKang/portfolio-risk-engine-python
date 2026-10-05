"""Deterministic result snapshots and worst-case rankings.

Snapshot
--------
A snapshot freezes a batch of results into a self-contained, locally
viewable document.  ``canonical_json`` serializes it deterministically:
keys sorted, floats via shortest-round-trip repr, NaN/Infinity rejected.
Same inputs -> byte-identical snapshot, every time.

Every entry carries an input summary, the rules-version stamp and a
rationale string explaining why the conclusion came out that way.

Readable export
---------------
``render_text`` turns a snapshot into a plain-text report that can be
opened directly in any editor: a header with outcome tallies, one block
per entry (input summary, rules version, rationale, result or typed
error with its failure category), and the worst-case rankings.  It is
fully deterministic — no timestamps, no run-dependent data — so the
same batch always exports byte-identical text, ready to archive or
diff.

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


def _failure_tallies(entries: list[dict]) -> list[tuple[str, int]]:
    """Count failed entries per failure category, sorted by category."""
    tallies: dict[str, int] = {}
    for e in entries:
        if e.get("ok"):
            continue
        category = (e.get("error") or {}).get("category", "unknown")
        tallies[category] = tallies.get(category, 0) + 1
    return sorted(tallies.items())


def render_text(snapshot: dict) -> str:
    """Render a snapshot as a deterministic, human-readable text report.

    The output contains no timestamps or run-dependent data: exporting
    the same inputs twice (or under different worker counts) produces
    byte-identical text.
    """
    entries = snapshot["entries"]
    lines: list[str] = []
    lines.append("RISK ENGINE RESULT SNAPSHOT")
    lines.append(f"format: {snapshot['format']}")
    lines.append(f"entries: {snapshot['count']}")
    ok_count = sum(1 for e in entries if e["ok"])
    lines.append(f"outcomes: {ok_count} ok, {len(entries) - ok_count} rejected")
    tallies = _failure_tallies(entries)
    if tallies:
        breakdown = ", ".join(f"{cat}={n}" for cat, n in tallies)
        lines.append(f"failures by category: {breakdown}")
    lines.append("")

    for e in entries:
        status = "OK" if e["ok"] else "REJECTED"
        lines.append(f"ENTRY {e['index']} [{e['kind']}] {status}")
        summary = ", ".join(
            f"{k}={v}" for k, v in sorted(e["input_summary"].items())
        )
        lines.append(f"  input: {summary}")
        lines.append(f"  rules version: {e['version'] or '-'}")
        lines.append(f"  rationale: {e['rationale']}")
        if e["ok"]:
            lines.append(f"  result: {canonical_json(e['result'])}")
        else:
            err = e["error"] or {}
            lines.append(
                "  error: "
                f"code={err.get('code', '?')} "
                f"category={err.get('category', '?')} "
                f"reason={err.get('reason', '')}"
            )
        lines.append("")

    rankings = snapshot.get("rankings", {})
    if rankings:
        lines.append("RANKINGS")
        for name in sorted(rankings):
            view = rankings[name]
            lines.append(
                f"{name}: metric={view['metric']} "
                f"order={view['order']}; {view['tie_break']}"
            )
            if view["items"]:
                for item in view["items"]:
                    lines.append(
                        f"  {item['rank']}. {item['name']} {item['value']!r}"
                    )
            else:
                lines.append("  (no evaluated entries)")
        lines.append("")
    return "\n".join(lines)


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
