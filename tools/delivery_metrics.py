"""How far her answers get, across the turns her runtime kept (Master Todo U07-11).

    python tools/delivery_metrics.py                     # every kept turn
    python tools/delivery_metrics.py --since 2026-11-20  # turns that began on or after a date

Each turn's receipt says which stages its answer reached (core/verify/turn_receipt.STAGES: drafted, checked,
committed, delivered), the seconds from the turn's start to each, and why a turn that served no answer stopped. This
counts them apart, as an outside review of 19 November 2026 asked: an answer her model wrote is not one that reached
the person, and a score that cannot tell them apart measures neither.
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = ("drafted", "checked", "committed", "delivered")


def read_kept(path: Path, since: float = 0.0) -> list[dict[str, Any]]:
    """The kept receipts, one per turn, that began at or after ``since`` (epoch seconds)."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and float(row.get("started_at") or 0.0) >= since:
            rows.append(row)
    return rows


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """How many turns reached each stage, how long each took, and why the others stopped.

    Only turns that came through the chat route are a person's: one that reached a stage or said why it stopped. Her own
    turns (a background tick of her kernel, a trial of her influence campaign) are kept beside them and counted apart;
    on 20 November 34 of 37 kept turns were hers, and counted as turns that went nowhere they hid the three that came.
    """
    rows = [row for row in rows if isinstance(row.get("reached"), dict)]
    hers = sum(1 for row in rows if not row["reached"] and not row.get("stopped"))
    rows = [row for row in rows if row["reached"] or row.get("stopped")]
    reached = {stage: [float(row["reached"][stage]) for row in rows if stage in row["reached"]] for stage in STAGES}
    drafted_and_checked = [float(row["reached"]["checked"]) - float(row["reached"]["drafted"]) for row in rows
                           if {"drafted", "checked"} <= set(row["reached"])]
    return {
        "turns": len(rows),
        "her_own_turns": hers,
        "reached": {stage: len(reached[stage]) for stage in STAGES},
        "median_s": {
            "to_first_draft": _median(reached["drafted"]),
            "checking_the_draft": _median(drafted_and_checked),
            "to_delivered": _median(reached["delivered"]),
        },
        "delivered_with_no_draft_recorded": sum(1 for row in rows if "delivered" in row["reached"]
                                                and "drafted" not in row["reached"]),
        "stopped": dict(Counter(str(row.get("stopped")) for row in rows if row.get("stopped")).most_common()),
    }


def as_text(measured: Mapping[str, Any]) -> str:
    lines = [f"Turns a person asked: {measured['turns']} (and {measured['her_own_turns']} of her own, not counted)"]
    lines += [f"  {stage:<10} {count}" for stage, count in measured["reached"].items()]
    lines.append("Median seconds: " + ", ".join(f"{name.replace('_', ' ')} {value}"
                                                for name, value in measured["median_s"].items()))
    lines.append(f"Delivered on a route that records no draft: {measured['delivered_with_no_draft_recorded']}")
    lines += [f"Stopped, {cause}: {count}" for cause, count in measured["stopped"].items()] or ["Stopped: none"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help="a date, YYYY-MM-DD: only turns that began on or after it")
    parser.add_argument("--path", type=Path, help="the kept receipts; her runtime's own by default")
    asked = parser.parse_args(argv)
    if asked.path is None:
        from core.verify.turn_receipt import kept_receipts_path

        asked.path = kept_receipts_path()
    since = datetime.datetime.fromisoformat(asked.since).timestamp() if asked.since else 0.0
    print(as_text(summary(read_kept(asked.path, since))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
