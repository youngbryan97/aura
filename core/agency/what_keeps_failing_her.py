"""What keeps failing her: a way her own work fails, on goal after goal, set as a goal of its own (Master Todo F16).

Bryan, 16 November 2026: "if she notices it's thin or could be more efficient or more capable, she could do it. or if
she notices a gap ... she could build it or find where the capability is failing, fix it, and then use it." Her loops
each stop with a reason written down: a fault whose test did not make it again, a draft a gate refused. One such stop is
an accident. The same stop on goal after goal is a capability of hers failing, and until 26 November nothing of hers
noticed: five of her goals stalled on "its test did not raise what she raised", the same fault in how her own code made
faults again, and it took a person reading her records to find it.

Here her goal records are read for the reasons her work stopped, and a reason shared by SAME_WAY of her goals or more
becomes a goal: mend what in her own code makes it happen. What each of those goals showed is said in it, so a pass at it
starts from the evidence. It is worked like any goal of hers in her own source (core/agency/her_goal_at_work.py), chosen
by her own time among the others, and a mend of it goes through every gate of hers and the outside check. A stop her
loop makes by design, such as a fault in what governs her, is no failure of hers and is not counted.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Agency.WhatKeepsFailingHer")

__all__ = ["SAME_WAY", "AWayShefails", "set_what_keeps_failing_her", "the_goal_of", "ways_she_fails"]

#: How many of her goals must have stopped the same way before the way is a goal: one is an accident, and two can be
#: the same cause met twice; three is the support the emergent goals of hers are adopted on
#: (core/goals/emergent_goals.py, ADOPTION_THRESHOLD).
SAME_WAY = 3
#: How much of what each stopped goal showed is said in the goal, in characters, and of how many goals.
_SHOWN = 600
_SHOWN_OF = 3
#: The gates that judge a draft of hers, by name, as her own yes asks them (core/agency/her_goal_at_work.py).
_GATE_WORDS = "her {gate} gate refused a draft of hers"


@dataclass(frozen=True)
class AWayShefails:
    """A reason her work stopped, the goals it stopped, what each showed, and when it last did."""

    how: str
    goals: tuple[str, ...]
    shown: tuple[str, ...]
    last_at: float


def _stops_of(record: Any) -> list[tuple[str, str, float]]:
    """The ways one goal of hers stopped: its stall, with its reason and evidence, and each gate that refused a draft."""
    out: list[tuple[str, str, float]] = []
    check = record.last_check if isinstance(record.last_check, dict) else {}
    if record.status == "stalled" and check.get("why") and not check.get("by_design"):
        out.append((str(check["why"]), str(check.get("evidence") or ""), float(check.get("at") or 0.0)))
    offer = record.offer if isinstance(record.offer, dict) else {}
    for gate, said in (offer.get("gates") or {}).items():
        if isinstance(said, dict) and said.get("passed") is False and said.get("finished", True) is not False:
            out.append((_GATE_WORDS.format(gate=gate), str(said.get("tail") or ""), float(offer.get("asked_at") or offer.get("at") or 0.0)))
    return out


def ways_she_fails(records: Iterable[Any], *, same_way: int = SAME_WAY) -> list[AWayShefails]:
    """Each reason that stopped ``same_way`` of her goals or more, the most goals first. A goal set by this module is
    not counted among them, so a mend of a way she fails that itself fails does not feed its own count."""
    held: dict[str, dict[str, Any]] = {}
    for record in records:
        if str(getattr(record, "goal", "")).startswith(_THE_GOAL):
            continue
        for how, shown, at in _stops_of(record):
            way = held.setdefault(how, {"goals": [], "shown": [], "last_at": 0.0})
            if record.slug not in way["goals"]:
                way["goals"].append(record.slug)
                way["shown"].append(f"{record.slug}: {' '.join(shown.split())[-_SHOWN:]}")
            way["last_at"] = max(way["last_at"], at)
    ways = [AWayShefails(how, tuple(way["goals"]), tuple(way["shown"][:_SHOWN_OF]), way["last_at"])
            for how, way in held.items() if len(way["goals"]) >= same_way]
    return sorted(ways, key=lambda way: (-len(way.goals), -way.last_at))


_THE_GOAL = "Mend what in my own code keeps failing this way: "


def the_goal_of(way: AWayShefails) -> str:
    """The goal a way she fails becomes: the same words each time, so it is one goal however often it recurs."""
    return _THE_GOAL + way.how


async def set_what_keeps_failing_her() -> list[str]:
    """Each way she fails that is not yet a goal of hers under way, set as one, with what it showed said in it, and
    offered to the goals her own time chooses among. A goal already under way, or mended since the way last stopped
    her, is left as it is. The slugs set."""
    from core.agency import her_goal_at_work as goals

    try:
        records = await asyncio.to_thread(goals.all_records)
    except (OSError, ValueError) as exc:
        record_degradation("what_keeps_failing_her", exc, severity="warning",
                           action="her goal records could not be read, so no way she fails was set as a goal")
        return []
    set_now: list[str] = []
    for way in ways_she_fails(records):
        goal = the_goal_of(way)
        slug = goals.slug_of(goal)
        known = goals.load(slug)
        if known is not None and (known.status in (goals.WORKING, goals.OFFERED, goals.STALLED)
                                  or float((known.offer or {}).get("merged_at") or 0.0) >= way.last_at):
            continue
        record = known or goals.GoalRecord(goal=goal, slug=slug)
        if known is not None:  # mended once, and it stopped her again since: the same goal, with what became of the mend
            record.offers_before = [*record.offers_before, {**(record.offer or {}), "set_aside": "it stopped her again"}]
            record.offer, record.status, record.quiet_passes = {}, goals.WORKING, 0
        record.said.append({"words": f"This stopped {len(way.goals)} of my goals: {', '.join(way.goals)}. "
                                     "What they showed, last first:\n" + "\n".join(way.shown), "at": time.time()})
        await goals.save(record)
        await _offered_to_her_own_time(goal, way)
        logger.info("🛠️ A way her work keeps failing is a goal of hers now: %s (%d goals)", way.how[:80], len(way.goals))
        set_now.append(slug)
    return set_now


async def _offered_to_her_own_time(goal: str, way: AWayShefails) -> None:
    """The goal in her goal store, where her own time finds the goals it chooses among (core/agency/what_she_could_do.py)."""
    from core.container import ServiceContainer

    engine = ServiceContainer.get("goal_engine", default=None)
    if engine is None or not hasattr(engine, "add_goal"):
        return
    try:
        await engine.add_goal(goal, goal, source="what_keeps_failing_her", priority=0.7,
                              metadata={"stopped": list(way.goals), "how": way.how})
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("what_keeps_failing_her", exc, severity="warning",
                           action="a way she fails was set as a goal of hers but not offered to her own time")
