"""What she changed in her own source, day by day (Master Todo F16, part 5).

Each draft of hers that reached a commit is one change: the mend or improvement, who took it (her own yes, or Bryan),
the gates it passed, what of it was loaded into her running process and what waits for her next start, whether it
was undone there because it faulted, and the commit that took it back out on origin. Read from her goal records
(core/agency/her_goal_at_work.py), which her loop writes as each of those happens; nothing here writes.
"""
from __future__ import annotations

import time
from typing import Any

#: The order a day's changes are read in: what needs a person first.
_STATES = ("undone, not yet taken back", "taken back out", "her gates refused it", "waits for Bryan",
           "landed, waits for her next start", "landed and loaded")


def _state(record: Any) -> str:
    from core.agency.her_goal_at_work import MERGED, OFFERED

    offer = record.offer or {}
    if offer.get("reverted"):
        return "taken back out"
    if offer.get("undone"):
        return "undone, not yet taken back"
    if record.status == MERGED:
        loaded = offer.get("loaded") or {}
        return "landed and loaded" if loaded.get("functions") and not loaded.get("waits") else \
            "landed, waits for her next start"
    if record.status == OFFERED:
        return "waits for Bryan"
    return "her gates refused it"


def a_change(record: Any) -> dict[str, Any] | None:
    """One change of hers as a person reads it, or None for a goal that never reached a commit."""
    offer = record.offer or {}
    if not offer.get("commit"):
        return None
    when = float(offer.get("reverted_at") or offer.get("merged_at") or offer.get("at") or record.last_worked_at or 0.0)
    gates = offer.get("gates") or {}
    loaded = offer.get("loaded") or {}
    undone = offer.get("undone") or {}
    return {
        "day": time.strftime("%Y-%m-%d", time.localtime(when)) if when else "undated",
        "at": when,
        "state": _state(record),
        "summary": str(offer.get("summary") or record.goal),
        "goal": record.goal,
        "commit": str(offer.get("commit")),
        "made_by": str(offer.get("origin") or ""),
        "taken_by": str(offer.get("taken_by") or ""),
        "files": list(offer.get("files") or []),
        "gates": {name: bool((gate or {}).get("passed")) for name, gate in gates.items()},
        "loaded": list(loaded.get("functions") or []),
        "waits": list(loaded.get("waits") or []),
        "undone_because": str(undone.get("because") or ""),
        "taken_back_by": str(offer.get("reverted") or ""),
        "last_words": str((record.said or [{}])[-1].get("words") or "") if record.said else "",
    }


def changes_by_day(records: list[Any] | None = None) -> list[dict[str, Any]]:
    """Her changes grouped by the day they landed or were taken back, newest day first; within a day, what needs a
    person first, then newest first."""
    from core.agency.her_goal_at_work import all_records

    changes = [change for record in (all_records() if records is None else records)
               if (change := a_change(record)) is not None]
    days: dict[str, list[dict[str, Any]]] = {}
    for change in changes:
        days.setdefault(change["day"], []).append(change)
    return [{"day": day, "changes": sorted(found, key=lambda c: (_STATES.index(c["state"]), -c["at"]))}
            for day, found in sorted(days.items(), reverse=True)]


__all__ = ["a_change", "changes_by_day"]
