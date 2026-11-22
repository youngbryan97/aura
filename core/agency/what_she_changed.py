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
           "waits for her own yes", "landed, waits for her next start", "landed, its load not recorded",
           "landed and loaded")


def _state(record: Any, offer: dict[str, Any] | None = None) -> str:
    """What became of an offer: the goal's current one by the goal's status, an earlier one by what its offer kept."""
    from core.agency.her_goal_at_work import MERGED, OFFERED

    current = offer is None
    offer = (record.offer if current else offer) or {}
    if offer.get("reverted"):
        return "taken back out"
    if offer.get("undone"):
        return "undone, not yet taken back"
    if (record.status == MERGED) if current else bool(offer.get("merged_at")):
        if "loaded" not in offer:
            # Landed before her record kept what was loaded (16 November 2026): not said to be waiting when it may not be.
            return "landed, its load not recorded"
        loaded = offer.get("loaded") or {}
        return "landed and loaded" if loaded.get("functions") and not loaded.get("waits") else \
            "landed, waits for her next start"
    if current and record.status == OFFERED:
        # Her own yes takes a draft (Bryan, 9 and 18 November 2026); only one that changes her own governance waits for
        # him. LIVE 22 November 2026 this page said "waits for Bryan" of two drafts waiting on her own gates since the
        # 17th and the 20th.
        from core.agency.her_goal_at_work import governs_her

        return ("waits for Bryan" if any(governs_her(name) for name in offer.get("files") or [])
                else "waits for her own yes")
    return "her gates refused it"


def a_change(record: Any, offer: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """One change of hers as a person reads it, the goal's current offer or an earlier one, or None for an offer that
    never reached a commit."""
    current = offer is None
    offer = (record.offer if current else offer) or {}
    if not offer.get("commit"):
        return None
    when = float(offer.get("reverted_at") or offer.get("merged_at") or offer.get("at") or record.last_worked_at or 0.0)
    gates = offer.get("gates") or {}
    loaded = offer.get("loaded") or {}
    undone = offer.get("undone") or {}
    return {
        "day": time.strftime("%Y-%m-%d", time.localtime(when)) if when else "undated",
        "at": when,
        "state": _state(record, None if current else offer),
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
        # What she said of this change, not of a try before it: a mend that landed kept "Not mended: nothing proposed."
        # from the attempt that stalled before it.
        "last_words": next((str(said.get("words") or "") for said in reversed(record.said or [])
                            if float(offer.get("at") or 0.0) <= float(said.get("at") or 0.0) <= _until(record, offer)),
                           ""),
    }


def _until(record: Any, offer: dict[str, Any]) -> float:
    """Until the next offer of the goal was made: what she said after that is about that one."""
    later = [float(o.get("at") or 0.0) for o in [*(record.offers_before or []), record.offer or {}]
             if float(o.get("at") or 0.0) > float(offer.get("at") or 0.0)]
    return min(later) if later else float("inf")


def changes_by_day(records: list[Any] | None = None) -> list[dict[str, Any]]:
    """Her changes grouped by the day they landed or were taken back, newest day first; within a day, what needs a
    person first, then newest first."""
    from core.agency.her_goal_at_work import all_records

    changes = [change for record in (all_records() if records is None else records)
               for offer in [*(getattr(record, "offers_before", None) or []), None]
               if (change := a_change(record, offer)) is not None]
    days: dict[str, list[dict[str, Any]]] = {}
    for change in changes:
        days.setdefault(change["day"], []).append(change)
    return [{"day": day, "changes": sorted(found, key=lambda c: (_STATES.index(c["state"]), -c["at"]))}
            for day, found in sorted(days.items(), reverse=True)]


__all__ = ["a_change", "changes_by_day"]
