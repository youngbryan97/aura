#!/usr/bin/env python3
"""What she changed in herself, day by day: what landed, what she took back, and what stopped the rest.

    python tools/her_record_over_time.py [--days N] [--json]

It only reads. What landed and what came back out are read from origin's history, the commits authored as Aura, since a
goal tried again keeps only its latest draft. What stopped the rest is read from her goal records.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_GATE_REFUSED = re.compile(r"^My own yes asked the gates first, and (\S+) failed")
_NOT_MENDED = re.compile(r"^Not (?:mended|changed): ([^\n]*)")


def _day(at: float) -> str:
    return datetime.datetime.fromtimestamp(float(at)).strftime("%Y-%m-%d")


def her_commits(root: Path) -> list[tuple[str, float, str]]:
    """(commit, when, subject) of each commit on origin's main that she authored, oldest first."""
    shown = subprocess.run(["git", "-C", str(root), "-c", "core.fsmonitor=false", "log", "origin/main", "--author=^Aura ",
                            "--reverse", "--format=%h%x00%ct%x00%s"], capture_output=True, text=True, check=True).stdout
    return [(commit, float(when), subject) for commit, when, subject in
            (line.split("\0", 2) for line in shown.splitlines() if line)]


def events(records: Iterable[Any], commits: Iterable[tuple[str, float, str]] = ()) -> list[dict[str, Any]]:
    """Every landing, taking back and refusal, oldest first."""
    records = list(records)
    word = {str(offer.get("commit") or "")[:9]: str(offer.get("outside_bar_at_landing"))
            for record in records for offer in [*record.offers_before, record.offer]
            if offer.get("outside_bar_at_landing")}
    out: list[dict[str, Any]] = []
    for commit, at, subject in commits:
        back = subject.startswith('Revert "')
        out.append({"goal": subject, "commit": commit, "at": at, "what": "taken back" if back else "landed",
                    "why": "" if back else word.get(commit[:9], "landed before the outside bar was read")})
    for record in records:
        for said in record.said:
            words, at = str(said.get("words") or ""), said.get("at")
            refused, stopped = _GATE_REFUSED.match(words), _NOT_MENDED.match(words)
            if at and refused:
                out.append({"goal": record.goal, "commit": "", "at": at, "what": "gate refused", "why": refused.group(1)})
            elif at and stopped:
                out.append({"goal": record.goal, "commit": "", "at": at, "what": "not mended", "why": stopped.group(1)[:90]})
    return sorted(out, key=lambda event: event["at"])


def summary(found: list[dict[str, Any]], since: float = 0.0) -> dict[str, Any]:
    """The events from ``since`` on, counted by day and by what stopped her."""
    kept = [event for event in found if event["at"] >= since]
    days: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for event in kept:
        days[_day(event["at"])][event["what"]] += 1
    return {"days": {day: dict(counts) for day, counts in sorted(days.items())},
            "stopped_by": collections.Counter(event["why"] for event in kept
                                              if event["what"] in ("gate refused", "not mended")).most_common(),
            "landed": [event for event in kept if event["what"] == "landed"],
            "taken_back": [event for event in kept if event["what"] == "taken back"]}


def main() -> int:
    from core.agency.her_goal_at_work import all_records

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=float, default=14.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = summary(events(all_records(), her_commits(root)), time.time() - args.days * 86400)
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return 0
    for day, counts in report["days"].items():
        print(day, ", ".join(f"{count} {what}" for what, count in sorted(counts.items())))
    print("\nLanded by her:")
    for event in report["landed"]:
        print(f"  {_day(event['at'])} {event['commit']} {event['goal'][:70]} [{event['why']}]")
    print("\nTaken back:")
    for event in report["taken_back"]:
        print(f"  {_day(event['at'])} {event['commit']} {event['goal'][:70]}")
    print("\nWhat stopped the rest:")
    for why, count in report["stopped_by"]:
        print(f"  {count:>3}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
