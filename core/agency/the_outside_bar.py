"""What the outside benchmarks last said about her, kept on the record of each change she lands (Master Todo F16).

The cheapest way for a self-changing system to look healthier is to prune the hard thinking its own metrics never
test, so her changes are read against benchmarks somebody else wrote, measured by a harness she cannot edit
(``outside_bar/``). This module only reads that harness's ledger. Bryan, 18 November 2026: she goes on editing and
pushing live while this is watched over time, so what it says is written down with each landing and stops none of them.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["LEDGER", "verdicts_since", "what_the_outside_bar_says"]

LEDGER = Path("~/.aura/outside-bar/ledger.jsonl").expanduser()


def _entries(ledger: Path) -> list[dict[str, Any]] | None:
    """The ledger's lines, or None when one of them is not what the line after it recorded."""
    lines = ledger.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    before = ""
    for line in lines:
        entry = json.loads(line)
        if entry.get("previous") != before:
            return None
        before = hashlib.sha256(line.encode("utf-8")).hexdigest()
        entries.append(entry)
    return entries


def what_the_outside_bar_says(ledger: Path = LEDGER) -> str:
    """What stands against her changes on the outside benchmarks, in words for her record; empty when nothing does."""
    try:
        entries = _entries(ledger)
    except FileNotFoundError:
        return "the outside benchmarks have no baseline yet"
    except (OSError, ValueError, AttributeError):
        return "the outside benchmarks' ledger cannot be read"
    if entries is None:
        return "the outside benchmarks' ledger has been altered"
    if not any(entry.get("kind") == "baseline" for entry in entries):
        return "the outside benchmarks have no baseline yet"
    decided = [entry for entry in entries if entry.get("kind") == "compare" and entry.get("verdict") in ("HELD", "REGRESSED")]
    if decided and decided[-1]["verdict"] == "REGRESSED":
        return f"the outside benchmarks regressed in run {decided[-1].get('candidate')}"
    return ""


def verdicts_since(moment: float, ledger: Path = LEDGER) -> list[str]:
    """The outside benchmarks' decided verdicts recorded after ``moment``, oldest first."""
    try:
        entries = _entries(ledger) or []
    except (OSError, ValueError, AttributeError):
        return []
    return [str(entry["verdict"]) for entry in entries if entry.get("kind") == "compare" and entry.get("verdict") in ("HELD", "REGRESSED")
            and datetime.datetime.strptime(str(entry.get("at")), "%Y-%m-%dT%H:%M:%S%z").timestamp() > moment]
