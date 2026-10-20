#!/usr/bin/env python3
"""F06's check: what her model was handed, since a time, that told it who to be or what to do, by origin.

Read from her log's census lines (core/brain/what_she_is_given.py, "[INSTRUCTED]"). F06 on the master todo closes
when a day of her running shows none from any origin.

    what_instructs_her.py [--log FILE] [--since 2026-10-20T15:00]    by origin: dispatches that instructed, characters,
                                                                    and the first sentence of each kind
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LOG = Path.home() / ".aura" / "logs" / "aura_json.log"
_LINE = re.compile(r"\[INSTRUCTED\] (?P<origin>\S+): (?P<chars>\d+) characters .*?; the first: (?P<first>.*)$", re.DOTALL)


def census(log: Path = LOG, since: str = "") -> dict[str, dict]:
    found: dict[str, dict] = defaultdict(lambda: {"dispatches": 0, "characters": 0, "first": {}})
    with log.open(encoding="utf-8", errors="replace") as lines:
        for line in lines:
            if "[INSTRUCTED]" not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if since and str(row.get("timestamp") or "") < since:
                continue
            said = _LINE.search(str(row.get("event") or ""))
            if not said:
                continue
            origin = found[said["origin"]]
            origin["dispatches"] += 1
            origin["characters"] += int(said["chars"])
            origin["first"][said["first"][:90]] = origin["first"].get(said["first"][:90], 0) + 1
    return dict(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--since", default="", help="an ISO time in the log's own zone (UTC)")
    args = parser.parse_args()
    found = census(args.log, args.since)
    for origin, row in sorted(found.items(), key=lambda kv: -kv[1]["characters"]):
        print(f"{row['characters']:>9} {row['dispatches']:>5}  {origin}")
        for first, n in sorted(row["first"].items(), key=lambda kv: -kv[1])[:3]:
            print(f"{'':>17}{n:>4} x {first}")
    print(f"{sum(r['characters'] for r in found.values())} characters of instruction from {len(found)} origin(s)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
