#!/usr/bin/env python3
"""Read several battery runs and say which criteria actually hold.

One run of the Intrinsic Subject Core battery gives a number between 10 and 13
out of 24, and the difference between those numbers is noise. Reporting the run
that came out highest would be picking the draw, and reporting the last one is
only marginally better.

So this takes every report it is given and asks, per criterion, whether it held
every time, failed every time, or changed its answer. A criterion that changes
answer between identical runs is not evidence either way, and saying so is more
useful than a total.

    tools/subject_core_scorecard.py artifacts/subject_core run_b run_c

Writes the table to stdout and, with `--json`, a machine-readable version that
a document can be regenerated from rather than transcribed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: Numbers worth tracking across runs whatever their criterion did, because the
#: spread between runs is the thing a single figure hides.
TRACKED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("phi_do", ("phi", "phi_do")),
    ("d_eff_normalised", ("differentiation", "d_eff_normalised")),
    ("intrinsic", ("intrinsic", "delta_intrinsic")),
    ("spread", ("perturbation", "mean_spread")),
    ("pci", ("perturbation", "mean_pci")),
    ("surrogate_floor", ("nulls", "surrogate_floor")),
)


def _dig(blob: Any, path: tuple[str, ...]) -> Any:
    node = blob
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def load(directory: Path) -> dict[str, Any]:
    report = directory / "subject_core_report.json"
    if not report.exists():
        raise SystemExit(f"no report in {directory}")
    return json.loads(report.read_text())


def scorecard(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Per criterion: how many runs it held, and what it says it needs."""
    order: list[str] = []
    held: dict[str, int] = {}
    bars: dict[str, Any] = {}
    sections: dict[str, str] = {}
    statements: dict[str, str] = {}
    for report in reports:
        for row in (report.get("verdict", {}) or {}).get("criteria", []):
            name = row["criterion"]
            if name not in held:
                order.append(name)
                held[name] = 0
                bars[name] = row.get("bar")
                sections[name] = row.get("section", "")
                statements[name] = row.get("requires", "")
            held[name] += int(bool(row.get("passed")))

    runs = len(reports)
    rows = []
    for name in order:
        count = held[name]
        verdict = "holds" if count == runs else ("fails" if count == 0 else "unresolved")
        rows.append(
            {
                "criterion": name,
                "section": sections[name],
                "requires": statements[name],
                "bar": bars[name],
                "held": count,
                "runs": runs,
                "verdict": verdict,
            }
        )

    numbers: dict[str, Any] = {}
    for label, path in TRACKED:
        values = [
            float(v) for v in (_dig(r, path) for r in reports) if isinstance(v, (int, float))
        ]
        if not values:
            continue
        numbers[label] = {
            "values": [round(v, 5) for v in values],
            "spread": round(max(values) - min(values), 5) if len(values) > 1 else 0.0,
            "median": round(statistics.median(values), 5),
        }

    campaigns: dict[str, list[str]] = {}
    for index, report in enumerate(reports):
        mark = (report.get("campaign") or {}).get("fingerprint", "unrecorded")
        campaigns.setdefault(mark, []).append(f"run {index + 1}")

    return {
        "runs": runs,
        "campaigns": campaigns,
        "holds": sum(1 for row in rows if row["verdict"] == "holds"),
        "unresolved": sum(1 for row in rows if row["verdict"] == "unresolved"),
        "fails": sum(1 for row in rows if row["verdict"] == "fails"),
        "totals_per_run": [
            (r.get("verdict", {}) or {}).get("passed") for r in reports
        ],
        "criteria": rows,
        "numbers": numbers,
    }


def _edge_lines(reports: list[dict[str, Any]]) -> list[str]:
    """Which edges held in every run, which in some, which in none."""
    held: dict[str, int] = {}
    effects: dict[str, list[float]] = {}
    for report in reports:
        for record in report.get("edges", []) or []:
            key = f"{record.get('source')}->{record.get('target')}"
            effects.setdefault(key, []).append(float(record.get("effect", 0.0)))
            if record.get("kept"):
                held[key] = held.get(key, 0) + 1
    runs = len(reports)
    lines = [f"| edge | kept in | effect across runs |", "|---|---|---|"]
    for key in sorted(held, key=lambda k: (-held[k], -max(effects[k]))):
        values = ", ".join(f"{value:.2f}" for value in effects[key])
        lines.append(f"| `{key}` | {held[key]}/{runs} | {values} |")
    if len(lines) == 2:
        lines.append("| — | 0 | no edge survived every bar in any run |")
    return lines


def _near_misses(reports: list[dict[str, Any]], limit: int = 12) -> list[str]:
    """The pairs closest to carrying, and what stopped each one."""
    best: dict[str, dict[str, Any]] = {}
    for report in reports:
        for record in report.get("edges", []) or []:
            if record.get("kept"):
                continue
            key = f"{record.get('source')}->{record.get('target')}"
            if key not in best or float(record.get("effect", 0.0)) > float(
                best[key].get("effect", 0.0)
            ):
                best[key] = record
    ranked = sorted(best.values(), key=lambda r: -float(r.get("effect", 0.0)))[:limit]
    lines = ["| pair | effect | q | conditions carrying |", "|---|---|---|---|"]
    for record in ranked:
        conditions = ", ".join(record.get("conditions", ())) or "none"
        lines.append(
            f"| `{record['source']}->{record['target']}` | "
            f"{float(record.get('effect', 0.0)):.2f} | {float(record.get('q', 1.0)):.4f} | "
            f"{conditions} |"
        )
    return lines


def report_markdown(card: dict[str, Any], reports: list[dict[str, Any]]) -> str:
    """The scorecard as a document, generated rather than transcribed.

    The completion specification asks for human-readable scorecards produced
    from the raw artifacts. Anything typed by hand from a terminal is a claim
    about a run rather than a reading of one, and drifts from it the moment
    either changes.
    """
    campaign = (reports[-1].get("campaign") or {}) if reports else {}
    frozen = campaign.get("frozen", {})
    lines: list[str] = []
    lines.append("# Intrinsic Subject Core — scorecard")
    lines.append("")
    lines.append(
        f"{card['holds']} hold, {card['unresolved']} unresolved, {card['fails']} fail "
        f"across {card['runs']} runs. Per-run totals {card['totals_per_run']} of "
        f"{card['criteria'] and len(card['criteria'])}."
    )
    lines.append("")
    lines.append("## The campaign")
    lines.append("")
    lines.append(f"- commit `{campaign.get('commit', '?')[:12]}` "
                 f"({'dirty' if campaign.get('dirty') else 'clean'} tree)")
    lines.append(f"- tree hash `{campaign.get('tree_hash', '?')}`")
    lines.append(f"- fingerprint `{campaign.get('fingerprint', '?')}`")
    prints = card.get("campaigns") or {}
    if len(prints) > 1:
        lines.append(
            f"- **{len(prints)} different fingerprints across these runs — not one campaign**"
        )
    intervention = frozen.get("intervention", {})
    recording = frozen.get("recording", {})
    lines.append(
        f"- {recording.get('rounds')} rounds over {len(recording.get('conditions', []))} "
        f"conditions; {intervention.get('trials')} trials, "
        f"{intervention.get('turns_per_arm')} turns per arm, delta "
        f"{intervention.get('delta')}"
    )
    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    lines.append("| | criterion | held | § | needs |")
    lines.append("|---|---|---|---|---|")
    for row in card["criteria"]:
        mark = {"holds": "✓", "fails": "✗"}.get(row["verdict"], "~")
        lines.append(
            f"| {mark} | `{row['criterion']}` | {row['held']}/{row['runs']} | "
            f"{row['section']} | {row['bar']} |"
        )
    if card.get("numbers"):
        lines.append("")
        lines.append("## The numbers, across runs")
        lines.append("")
        lines.append("| quantity | values | median | spread |")
        lines.append("|---|---|---|---|")
        for label, entry in card["numbers"].items():
            lines.append(
                f"| {label} | {entry['values']} | {entry['median']} | {entry['spread']} |"
            )
    lines.append("")
    lines.append("## Edges that survived every bar")
    lines.append("")
    lines.extend(_edge_lines(reports))
    lines.append("")
    lines.append("## The pairs closest to carrying")
    lines.append("")
    lines.extend(_near_misses(reports))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="read the N most recent run_NNN directories beneath the one given",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="write the scorecard as a document, generated from the artifacts",
    )
    args = parser.parse_args()

    directories = list(args.directories)
    if args.latest:
        root = directories[0]
        runs = sorted(p for p in root.glob("run_*") if (p / "subject_core_report.json").exists())
        if not runs:
            raise SystemExit(f"no run_NNN directories under {root}")
        directories = runs[-args.latest :]
    reports = [load(d) for d in directories]
    card = scorecard(reports)
    card["runs_read"] = [str(d) for d in directories]

    print(
        f"{card['holds']} hold, {card['unresolved']} unresolved, {card['fails']} fail "
        f"across {card['runs']} runs (per-run totals {card['totals_per_run']})"
    )
    prints = card.get("campaigns") or {}
    if len(prints) > 1:
        print(f"  WARNING: {len(prints)} different campaign fingerprints — not one campaign")
    for mark, runs in prints.items():
        print(f"  campaign {mark}: {', '.join(runs)}")
    print()
    width = max(len(row["criterion"]) for row in card["criteria"])
    for row in card["criteria"]:
        print(
            f"  {row['verdict']:<10} {row['criterion']:<{width}}  "
            f"{row['held']}/{row['runs']}  §{row['section']}  needs {row['bar']}"
        )
    if card["numbers"]:
        print()
        print("  the numbers, across runs:")
        for label, entry in card["numbers"].items():
            print(
                f"    {label:<18} {entry['values']}  median {entry['median']}  "
                f"spread {entry['spread']}"
            )

    if args.json or args.markdown:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        with local_internal_governed_scope("subject_core.scorecard"):
            if args.json:
                gateway.write_text(
                    args.json, json.dumps(card, indent=2), source="subject_core.scorecard"
                )
            if args.markdown:
                gateway.write_text(
                    args.markdown,
                    report_markdown(card, reports),
                    source="subject_core.scorecard",
                )
        for target in (args.json, args.markdown):
            if target:
                print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
