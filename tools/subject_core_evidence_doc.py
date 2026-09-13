#!/usr/bin/env python3
"""Write the evidence section of the subject-core document from the runs.

A document that summarises a measurement by hand goes stale the first time the
measurement is repeated, and the reader has no way to tell. This regenerates
one section between two markers, straight from the reports on disk, so the
numbers in the prose are the numbers in the artifacts or the gate fails.

    python tools/subject_core_evidence_doc.py           # rewrite the section
    python tools/subject_core_evidence_doc.py --check   # fail if it has drifted

The prose around the section is written by a person and left alone. What is
generated is the part that is a table of numbers, which is the part nobody
should be retyping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "SUBJECT_CORE.md"
RUNS = REPO / "artifacts" / "subject_core"

OPEN = "<!-- generated: subject-core evidence -->"
CLOSE = "<!-- end generated -->"


def _reports(root: Path) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for directory in sorted(root.glob("run_*")):
        report = directory / "subject_core_report.json"
        if not report.is_file():
            continue
        try:
            out.append((directory.name, json.loads(report.read_text())))
        except (OSError, ValueError):
            continue
    return out


def _row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    verdict = report.get("verdict", {}) or {}
    criteria = verdict.get("criteria", []) or []
    campaign = report.get("campaign", {}) or {}
    graph = report.get("graph", {}) or {}
    return {
        "run": name,
        "commit": str(campaign.get("commit", ""))[:12],
        "campaign": str(campaign.get("fingerprint", ""))[:8],
        "authoritative": campaign.get("authoritative"),
        "passed": sum(1 for c in criteria if c.get("passed")),
        "of": len(criteria),
        "phi": (report.get("phi", {}) or {}).get("phi_do"),
        "edges": sum(1 for r in report.get("edges", []) or [] if r.get("kept")),
        "component": max(
            (len(part) for part in graph.get("components", []) or []), default=0
        ),
        "vertex": graph.get("vertex_connectivity"),
        "spread": (report.get("perturbation", {}) or {}).get("mean_spread"),
    }


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No run in `artifacts/subject_core` carries a report yet."
    head = (
        "| run | commit | campaign | criteria | phi | edges | component | vertex | spread |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [head]
    for row in rows:
        mark = "" if row["authoritative"] is not False else " (degraded)"
        phi = "-" if row["phi"] is None else f"{float(row['phi']):+.4f}"
        spread = "-" if row["spread"] is None else f"{float(row['spread']):.3f}"
        lines.append(
            f"| {row['run']}{mark} | `{row['commit']}` | `{row['campaign']}` | "
            f"{row['passed']}/{row['of']} | {phi} | {row['edges']} | "
            f"{row['component']}/10 | {row['vertex'] if row['vertex'] is not None else '-'} | {spread} |"
        )
    return "\n".join(lines)


def render(root: Path = RUNS) -> str:
    rows = [_row(name, report) for name, report in _reports(root)]
    body = _table(rows)
    return f"{OPEN}\n\n{body}\n\n{CLOSE}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the section has drifted")
    parser.add_argument("--runs", type=Path, default=RUNS)
    args = parser.parse_args()

    block = render(args.runs)
    text = DOC.read_text()
    if OPEN not in text or CLOSE not in text:
        print(f"{DOC.name} has no generated section; add the markers first")
        return 1
    start = text.index(OPEN)
    end = text.index(CLOSE) + len(CLOSE)
    rewritten = text[:start] + block + text[end:]
    if args.check:
        if rewritten != text:
            print(f"{DOC.name}'s evidence section has drifted from the artifacts")
            return 1
        print(f"{DOC.name} matches the artifacts")
        return 0
    if rewritten != text:
        DOC.write_text(rewritten)
        print(f"{DOC.name}: evidence section rewritten from the artifacts")
    else:
        print(f"{DOC.name}: already matches the artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
