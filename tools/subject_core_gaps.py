#!/usr/bin/env python3
"""Which edges the battery is missing, and how far off each one is.

The verdict says a criterion failed. It does not say which of ninety ordered
pairs is the reason, which of them nearly carried, or which target columns
moved at all when a domain was displaced. Without that, the next repair is a
guess.

This reads a run directory — the edge table and the archived arms — and sorts
the ninety pairs into what carried, what is close, and what showed nothing.
For a pair that showed nothing it says why: no trial moved the target, or the
target moved as much in the sham as in the displaced arm, or the effect is
there and the replication is not.

Nothing here re-runs anything or changes a threshold. It reads the evidence a
run already wrote down.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DOMAINS = ("P", "I", "A", "G", "C", "S", "M", "W", "D", "N")

#: What each domain is meant to reach, from the completion specification. A
#: pair named here and missing is a gap; a pair not named here and present is a
#: bonus, not a defect.
TARGETS: dict[str, tuple[str, ...]] = {
    "P": ("W", "A", "G", "C", "M", "D"),
    "I": ("A", "G", "C", "D"),
    "A": ("G", "D", "C", "N"),
    "G": ("C", "S", "M", "A", "D"),
    "C": ("G", "A", "S", "D"),
    "S": ("G", "D", "M", "N"),
    "M": ("G", "S", "W", "D"),
    "W": ("G", "A", "D", "C"),
    "D": ("G", "W", "P", "S"),
    "N": ("A", "G", "M", "D"),
}


def _load_arms(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "_meta" in record:
            continue
        rows.append(record)
    return rows


def _reason(source: str, target: str, arms: list[dict[str, Any]], record: dict[str, Any] | None) -> str:
    mine = [row for row in arms if row["source"] == source]
    if not mine:
        return "no trial displaced this domain"
    effects = [float(row["effect"].get(target, 0.0)) for row in mine]
    floors = [float(row["floor"].get(target, 0.0)) for row in mine]
    moved = sum(1 for value in effects if value > 1e-6)
    if moved == 0:
        return "the target never moved in any displaced arm"
    mean_effect = sum(effects) / len(effects)
    mean_floor = sum(floors) / len(floors)
    if mean_floor >= mean_effect:
        return (
            f"the sham moved it as much: {mean_effect:.2f} displaced against "
            f"{mean_floor:.2f} floor"
        )
    if record is None:
        return f"moved {mean_effect - mean_floor:.2f} above the floor, untested"
    gap = float(record.get("effect", 0.0))
    if gap < 0.30:
        return f"above the floor but small: {gap:.2f} against a bar of 0.30"
    if int(record.get("replication", 0)) < 3:
        carried = ", ".join(record.get("conditions", ())) or "none"
        return f"large enough at {gap:.2f} but carried in {carried}"
    if float(record.get("q", 1.0)) >= 0.01:
        return f"large and replicated at {gap:.2f}, q={record.get('q')} misses 0.01"
    return f"{gap:.2f}, all three bars met"


def report(run: Path) -> dict[str, Any]:
    edges = json.loads((run / "subject_core_report.json").read_text())["edges"]
    table = {(row["source"], row["target"]): row for row in edges}
    arms = _load_arms(run / "intervention_arms.jsonl")

    kept = {pair for pair, row in table.items() if row.get("kept")}
    wanted = {(s, t) for s, targets in TARGETS.items() for t in targets}
    out_degree: dict[str, int] = defaultdict(int)
    in_degree: dict[str, int] = defaultdict(int)
    for source, target in kept:
        out_degree[source] += 1
        in_degree[target] += 1

    lines: list[str] = []
    lines.append(f"run {run.name}: {len(kept)} edges kept of {len(table)} tested")
    lines.append("")
    lines.append("degree by domain (kept edges)")
    for key in DOMAINS:
        lines.append(f"  {key}: out {out_degree[key]}, in {in_degree[key]}")
    lines.append("")

    missing = sorted(wanted - kept)
    lines.append(f"specified edges not yet retained: {len(missing)} of {len(wanted)}")
    for source, target in missing:
        lines.append(f"  {source}->{target}: {_reason(source, target, arms, table.get((source, target)))}")

    extra = sorted(kept - wanted)
    if extra:
        lines.append("")
        lines.append("retained edges the specification did not name:")
        for source, target in extra:
            lines.append(f"  {source}->{target}: {table[(source, target)]['effect']}")

    print("\n".join(lines))
    return {
        "kept": sorted(f"{s}->{t}" for s, t in kept),
        "missing": sorted(f"{s}->{t}" for s, t in missing),
        "out_degree": dict(out_degree),
        "in_degree": dict(in_degree),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a run_NNN directory")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    run = args.run
    if not (run / "subject_core_report.json").exists():
        runs = sorted(run.glob("run_*"))
        if not runs:
            raise SystemExit(f"no report in {run}")
        run = runs[-1]
    summary = report(run)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
