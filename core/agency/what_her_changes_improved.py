"""Whether a change of hers made something better, and on what evidence (Master Todo F16, from an outside review).

Her record of what she changed (core/agency/what_she_changed.py) says what became of each change: landed, loaded,
taken back. It does not say whether anything got better, and keeping what worked is not the same as improving. An
outside review of 19 November (F16-09) asked for a record that tells the two apart, and that never calls a change an
improvement because its tests pass or because a call got faster. Each change of hers that landed is read here as one
of these, by the evidence her own records hold:

- a failure repaired: it mends a fault her running raised, and that fault has not been heard since the mend landed;
- a repair that did not hold: the fault it mends was heard again after it landed;
- a mend whose fault is no longer on record: whether it held cannot be told from what is kept;
- a cost lowered: her loop measured what a site cost her before the change and after it, and kept it on that;
- a drill: it mends a fault planted to exercise her repair loop (core/self_modification/loop_drill.py), which shows
  the loop works and says nothing about her;
- taken back: it landed and was reverted;
- kept, not shown to improve anything: it landed, and nothing measured a gain.

Each entry keeps the source before it, the change, what else the outside comparison measured with it, and how it is
undone. Nothing here writes.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

__all__ = ["KINDS", "improvements", "what_her_changes_improved"]

KINDS = ("a failure repaired", "a repair that did not hold", "a mend whose fault is no longer on record",
         "a cost lowered", "a drill", "taken back", "kept, not shown to improve anything")

#: The module whose faults are planted to exercise her repair loop, not raised by her running.
_DRILL_MODULE = "core.self_modification.loop_drill"


def _landed_offers(record: Any) -> list[dict[str, Any]]:
    """Every offer of a goal that reached her source, earliest first: those set aside before, then the current one."""
    offers = [dict(offer) for offer in (getattr(record, "offers_before", None) or []) if isinstance(offer, dict)]
    offers.append(dict(getattr(record, "offer", None) or {}))
    return [offer for offer in offers if offer.get("merged_at") and offer.get("commit")]


def _its_faults(record: Any, heard: Iterable[Any]) -> list[Any]:
    """The heard faults this goal mends: those whose goal, written the way her mend loop writes it, is this goal."""
    from core.agency.mending_a_fault_of_hers import goal_for

    return [fault for fault in heard if goal_for(fault) == record.goal]


def _days(seconds: float) -> str:
    days = seconds / 86400.0
    return f"{days:.1f} days" if days >= 1.0 else f"{seconds / 3600.0:.1f} hours"


def _what_it_shows(record: Any, offer: dict[str, Any], heard: list[Any], now: float) -> tuple[str, str]:
    landed = float(offer["merged_at"])
    if offer.get("reverted"):
        return "taken back", f"reverted by {str(offer['reverted'])[:9]}"
    judged = offer.get("judged") or {}
    if judged.get("kept") and offer.get("measured_before") and judged.get("measured_after"):
        return "a cost lowered", str(judged.get("why") or "measured before and after, and kept")
    faults = _its_faults(record, heard)
    if faults:
        if all(fault.call.module == _DRILL_MODULE for fault in faults):
            return "a drill", "a fault planted to exercise her repair loop"
        again = [fault for fault in faults if float(fault.last_at) > landed]
        if again:
            return ("a repair that did not hold",
                    f"heard again {_days(max(float(fault.last_at) for fault in again) - landed)} after it landed")
        return "a failure repaired", f"not heard in the {_days(now - landed)} since it landed"
    if record.goal.startswith("Mend "):
        return "a mend whose fault is no longer on record", "the fault it mends is not among those kept"
    return "kept, not shown to improve anything", "nothing measured a gain"


def improvements(records: Iterable[Any] | None = None, heard: Iterable[Any] | None = None, *,
                 now: float | None = None) -> list[dict[str, Any]]:
    """Each change of hers that landed, earliest first, with what it shows and the evidence for it."""
    if records is None:
        from core.agency.her_goal_at_work import all_records

        records = all_records()
    if heard is None:
        from core.self_modification.faults_she_can_reproduce import faults_heard

        heard = faults_heard()
    heard = list(heard)
    now = time.time() if now is None else float(now)
    found = []
    for record in records:
        for offer in _landed_offers(record):
            kind, evidence = _what_it_shows(record, offer, heard, now)
            commit = str(offer["commit"])
            found.append({
                "goal": record.goal, "slug": record.slug, "kind": kind, "evidence": evidence,
                "commit": commit, "landed_at": float(offer["merged_at"]), "taken_by": str(offer.get("taken_by") or ""),
                "before": str(getattr(record, "base", "") or ""),
                "measured_beside": list((offer.get("settled_on") or {}).get("measured_beside") or []),
                "undo": "" if offer.get("reverted") else f"git revert {commit[:9]}",
            })
    return sorted(found, key=lambda entry: entry["landed_at"])


def what_her_changes_improved(records: Iterable[Any] | None = None, heard: Iterable[Any] | None = None, *,
                              now: float | None = None) -> dict[str, Any]:
    """How many of her landed changes show each kind, and the changes themselves."""
    found = improvements(records, heard, now=now)
    return {"counts": {kind: sum(1 for entry in found if entry["kind"] == kind) for kind in KINDS}, "changes": found}
