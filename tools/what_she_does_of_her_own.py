#!/usr/bin/env python3
"""Behaviour coordination alone does not show, read from her own record.

Bryan, 21 October 2026: a preference held consistently, an interjection, a
change of mind. The battery measures how her parts move together; this reads
what she has come to like and dislike and what she set out to do, from the
record her conation keeps (core/conation/what_she_wants.py). Behavioural
evidence beside the battery, not a criterion of it.

A liking stands when it sits further from indifference than its own
uncertainty: her liking for a thing is the mean of how turns with it went
against what she expected, and `unsure` is the variance of that mean, so a
liking more than two standard errors from zero is one she holds, on that
record, rather than noise. How many of her things clear that bar, which way
each points, and what she decided to pursue and why, is the report.

    python tools/what_she_does_of_her_own.py [path to what_she_wants.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT = Path.home() / ".aura" / "data" / "conation" / "what_she_wants.json"


def held(things: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The things whose liking is more than two standard errors from indifference."""
    out = []
    for thing in things:
        liking = float(thing.get("liking") or 0.0)
        unsure = float(thing.get("unsure") or 0.0)
        error = math.sqrt(unsure) if unsure > 0 else float("inf")
        if abs(liking) > 2.0 * error:
            out.append({
                "thing": thing.get("label") or thing.get("key"),
                "liking": round(liking, 4),
                "standard_error": round(error, 4),
                "contacts": int(thing.get("contacts") or 0),
                "first_turn": thing.get("first"),
                "last_turn": thing.get("last"),
            })
    return sorted(out, key=lambda row: -abs(row["liking"]) / row["standard_error"])


def report(record: dict[str, Any]) -> dict[str, Any]:
    things = list(record.get("things") or [])
    standing = held(things)
    decided = record.get("decided") or {}
    return {
        "turns": record.get("turns"),
        "things_met": len(things),
        "likings_held": len(standing),
        "liked": [row for row in standing if row["liking"] > 0][:12],
        "disliked": [row for row in standing if row["liking"] < 0][:12],
        "decided_to_pursue": {
            key: decided.get(key) for key in ("label", "way", "kind", "origin", "why", "decided_turn")
        } if decided else None,
        "exchanges": record.get("exchanges"),
    }


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    out = report(json.loads(path.read_text(encoding="utf-8")))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
