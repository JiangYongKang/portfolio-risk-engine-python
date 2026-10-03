"""Deterministic result snapshots and worst-case rankings.

A snapshot is a self-contained, locally-openable HTML document (plus an
embedded canonical JSON data island) describing a batch scan.  Repeated
exports of the same inputs produce byte-identical output: no timestamps,
host names or environment data are included, dicts are serialized with
sorted keys, and floats use a fixed repr.

Rankings are offered in two directions:

* worst **by portfolio** - for each portfolio, its worst scenario PnL;
* worst **by scenario**  - for each scenario, its worst portfolio PnL.

"Worst" means smallest PnL (largest loss); ties are broken by name then
by the orthogonal dimension, all lexicographic, so rankings are stable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from html import escape

from risk_engine.models import BatchItemResult

#: Metric used by every ranking: scenario PnL, "worst" = minimum (max loss).
RANK_METRIC = "min_scenario_pnl"
#: Tie-break order documented in every snapshot.
TIE_BREAK = (
    "ties broken lexicographically: worst_pnl asc, then name asc, "
    "then worst_partner asc"
)


@dataclass(frozen=True)
class RankEntry:
    name: str
    worst_pnl: float
    worst_partner: str  # the scenario (portfolio rank) or portfolio (scenario rank)
    cells: int


@dataclass(frozen=True)
class Snapshot:
    canonical_json: str
    html: str
    data: dict


def _normalize(value):
    """Convert payloads into JSON-stable forms (sorted mappings, tuples->lists)."""
    if isinstance(value, dict):
        return {k: _normalize(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def canonical_json(payload: dict) -> str:
    """Deterministic JSON encoding: sorted keys, fixed separators."""
    return json.dumps(
        _normalize(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _ok_cells(cells: list[dict]) -> list[dict]:
    return [c for c in cells if c.get("ok")]


def _worst_groups(cells: list[dict], key: str, partner_key: str) -> list[RankEntry]:
    """Group successful cells by ``key``; keep minimum PnL per group."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for cell in _ok_cells(cells):
        grouped[cell[key]].append(cell)
    entries: list[RankEntry] = []
    for name, members in grouped.items():
        worst = min(members, key=lambda c: (c["pnl"], c[partner_key]))
        entries.append(
            RankEntry(
                name=name,
                worst_pnl=worst["pnl"],
                worst_partner=worst[partner_key],
                cells=len(members),
            )
        )
    # deterministic ordering: worst PnL first, then names
    entries.sort(key=lambda e: (e.worst_pnl, e.name, e.worst_partner))
    return entries


def rank_by_portfolio(cells: list[dict]) -> list[RankEntry]:
    """Worst-scenario PnL per portfolio, deterministic ordering."""
    return _worst_groups(cells, "portfolio", "scenario")


def rank_by_scenario(cells: list[dict]) -> list[RankEntry]:
    """Worst-portfolio PnL per scenario, deterministic ordering."""
    return _worst_groups(cells, "scenario", "portfolio")


def _flatten(results: list[BatchItemResult]) -> list[dict]:
    """Normalize batch slots into stable cell dicts for snapshot/ranking."""
    cells: list[dict] = []
    for slot in results:
        base = {
            "index": slot.index,
            "ok": bool(slot.ok),
            "portfolio": None,
            "scenario": None,
            "pnl": None,
            "version": None,
            "input": None,
            "rationale": None,
            "error": slot.error,
        }
        if slot.ok and slot.result is not None:
            r = slot.result
            inp = r.get("input") or {}
            base.update(
                {
                    "portfolio": inp.get("portfolio", r.get("scenario")),
                    "scenario": r.get("scenario"),
                    "pnl": r.get("pnl"),
                    "version": r.get("version"),
                    "input": inp,
                    "rationale": r.get("rationale"),
                }
            )
        cells.append(base)
    return cells


def _rank_to_dict(entries: list[RankEntry]) -> list[dict]:
    return [
        {
            "name": e.name,
            "worst_pnl": e.worst_pnl,
            "worst_partner": e.worst_partner,
            "cells": e.cells,
        }
        for e in entries
    ]


def build_snapshot(
    cells: list[BatchItemResult],
    *,
    title: str = "risk scan snapshot",
) -> Snapshot:
    """Assemble a stable snapshot (canonical JSON + standalone HTML)."""
    flat = _flatten(cells)
    by_portfolio = rank_by_portfolio(flat)
    by_scenario = rank_by_scenario(flat)

    data = {
        "title": title,
        "schema": "risk-scan-snapshot/1",
        "ranking_metric": RANK_METRIC,
        "tie_break": TIE_BREAK,
        "count": len(flat),
        "cells": flat,
        "rank_by_portfolio": _rank_to_dict(by_portfolio),
        "rank_by_scenario": _rank_to_dict(by_scenario),
    }
    payload = canonical_json(data)
    html = _render_html(data, payload)
    return Snapshot(canonical_json=payload, html=html, data=data)


# --- HTML rendering (pure function of ``data``) ----------------------------

def _fmt_num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return escape(str(v))


def _render_table(rows: list[dict], partner_header: str) -> str:
    head = (
        "<tr><th>name</th><th>worst_pnl</th>"
        f"<th>{partner_header}</th><th>cells</th></tr>"
    )
    body = "".join(
        "<tr>"
        f"<td>{escape(r['name'])}</td>"
        f"<td>{_fmt_num(r['worst_pnl'])}</td>"
        f"<td>{escape(r['worst_partner'])}</td>"
        f"<td>{r['cells']}</td>"
        "</tr>"
        for r in rows
    )
    return f"<table>{head}{body}</table>"


def _render_cells(cells: list[dict]) -> str:
    head = (
        "<tr><th>index</th><th>ok</th><th>portfolio</th><th>scenario</th>"
        "<th>version</th><th>pnl</th><th>input</th><th>rationale / error</th></tr>"
    )
    body_rows = []
    for c in cells:
        if c["ok"]:
            verdict = escape(c.get("rationale") or "")
        else:
            err = c.get("error") or {}
            verdict = escape(f"{err.get('code', '')}: {err.get('reason', '')}")
        body_rows.append(
            "<tr>"
            f"<td>{c['index']}</td>"
            f"<td>{str(c['ok']).lower()}</td>"
            f"<td>{escape(c.get('portfolio') or '')}</td>"
            f"<td>{escape(c.get('scenario') or '')}</td>"
            f"<td>{escape(c.get('version') or '')}</td>"
            f"<td>{_fmt_num(c.get('pnl'))}</td>"
            f"<td><code>{escape(canonical_json(c.get('input') or {}))}</code></td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    return f"<table>{head}{''.join(body_rows)}</table>"


def _render_html(data: dict, payload: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{escape(data['title'])}</title>"
        "<style>"
        "body{font-family:monospace;margin:2em;}"
        "table{border-collapse:collapse;margin:1em 0;}"
        "td,th{border:1px solid #999;padding:2px 8px;text-align:left;}"
        "th{background:#eee;}"
        "</style></head><body>"
        f"<h1>{escape(data['title'])}</h1>"
        f"<p>schema: {escape(data['schema'])}; cells: {data['count']}</p>"
        "<p>ranking metric: "
        f"{escape(data['ranking_metric'])}; {escape(data['tie_break'])}</p>"
        "<h2>Worst by portfolio</h2>"
        + _render_table(data["rank_by_portfolio"], "worst scenario")
        + "<h2>Worst by scenario</h2>"
        + _render_table(data["rank_by_scenario"], "worst portfolio")
        + "<h2>Cells</h2>"
        + _render_cells(data["cells"])
        + "<script id=\"snapshot-data\" type=\"application/json\">"
        + payload
        + "</script></body></html>\n"
    )
