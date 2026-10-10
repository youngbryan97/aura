"""What the screen itself shows of what a place is, measured as she plays: no model asked.

A person learns a good deal about a place from how it behaves before they could name a single thing in it. A crowd of
alike things all going one way, one of them answering to them, is a race and they are one of the racers. Scenery
streaming past is a run or a flight. A thing that follows the mouse is a hand or a cursor. A counter of distance says how
far is what counts; one of points, how many; two counters set against each other, a contest. A bar that fills and empties
beside its words is read for what they say.

All of it is already measured by her perception and play: what moves and how (core/perception/
what_moves_in_the_picture.py, with the view going by from how_the_scenery_goes_by.py), which thing answers her
(core/agency/which_one_answers_to_her.py), the counters and bars read (core/agency/what_meeting_things_does.py,
core/agency/what_she_has_left.py), how a contest stands (core/agency/how_the_contest_stands.py). This only says it, as
facts of the place, into the reading of what the place is (core/cognition/what_this_place_is.py), where what is
measured outranks what is supposed, and where a counter the place shows says what she wants out of it.

Nothing here knows a place, and no model is asked.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

__all__ = ["measured_in_play", "what_play_measures"]

#: How often what play measures is put to the reading, in seconds of play.
EVERY_S = 2.0
#: How many alike things make a crowd, and how nearly alike their ways must be (cosine) to be going one way.
A_CROWD = 3
ONE_WAY = 0.8
#: How fast, in working pixels a second, a thing must go to count as going anywhere.
GOING = 6.0

#: Counters of how far, and of how many.
_HOW_FAR = re.compile(r"\b(?:distance|far|height|high|feet|ft|met(?:er|re)s?|miles?|km|yards?)\b", re.I)
_HOW_MANY = re.compile(r"\b(?:score|points?|pts|coins?|gems?|stars?|total)\b", re.I)


def _going_one_way(things: list[Any]) -> bool:
    """Whether most of ``things`` are going, and going the same way."""
    ways = [(t.vx, t.vy) for t in things if math.hypot(t.vx, t.vy) > GOING]
    if len(ways) < A_CROWD:
        return False
    mx, my = sum(w[0] for w in ways), sum(w[1] for w in ways)
    mean = math.hypot(mx, my)
    if mean == 0:
        return False
    alike = sum(1 for vx, vy in ways if (vx * mx + vy * my) / (math.hypot(vx, vy) * mean) >= ONE_WAY)
    return alike >= max(A_CROWD, 0.7 * len(ways))


def what_play_measures(moves: Any, hers: Any, meeting: Any, contest: Any = None, bars: Any = None) -> dict[str, str]:
    """The facts of the place that play has measured, each a short sentence, by what it is about."""
    facts: dict[str, str] = {}
    by_kind: dict[int, list[Any]] = defaultdict(list)
    for thing in (getattr(moves, "things", {}) or {}).values():
        by_kind[int(getattr(thing, "kind", -1))].append(thing)
    mine = getattr(hers, "kind", None) if getattr(hers, "number", None) is not None else None
    if mine is not None and len(by_kind.get(mine, [])) >= A_CROWD:
        crowd = by_kind[mine]
        facts["crowd"] = (f"I'm one of {len(crowd)} alike things" + (", all going the same way" if _going_one_way(crowd) else ""))
    else:
        going = [kind for kind, things in by_kind.items() if kind >= 0 and len(things) >= A_CROWD and _going_one_way(things)]
        if going:
            facts["crowd"] = f"{len(by_kind[going[0]])} alike things are all going the same way"
    if getattr(moves, "view_is_moving", False):
        facts["view"] = "the view goes by, as the scenery of a run or a flight does"
    if getattr(hers, "follows_pointer", False):
        facts["control"] = "what's mine follows the mouse"
    elif getattr(hers, "number", None) is not None:
        facts["control"] = "my keys move one thing" + (", carried on by its own going" if getattr(hers, "carried", False) else "")
    counted = [str(name) for name in (getattr(getattr(meeting, "readouts", None), "values", {}) or {})
               if re.search(r"[A-Za-z]{3,}", str(name))]
    if counted:
        facts["counted"] = "it counts " + ", ".join(dict.fromkeys(counted[:4]))
    if contest is not None and getattr(contest, "mine", None) is not None and getattr(contest, "theirs", None) is not None:
        facts["contest"] = "two sides are counted against each other"
    shown = [b for b in (bars.bars() if bars is not None and hasattr(bars, "bars") else [])]
    if shown:
        facts["bars"] = f"{len(shown)} bar{'s' if len(shown) > 1 else ''} filling and emptying"
    return facts


def _wanted(facts: dict[str, str]) -> str:
    """What a counter the place shows says she wants out of it: as far as it goes, as many as she can."""
    counted = facts.get("counted", "")
    if _HOW_FAR.search(counted):
        return "to go as far as it will go"
    if _HOW_MANY.search(counted):
        return "as many points as I can"
    return ""


def measured_in_play(run: Any, moves: Any, hers: Any, meeting: Any, at: float) -> None:
    """Every few seconds of play, what play measures put to the reading of what the place is: as facts for reasoning and
    for her model's reading to fit, and what a counter the place shows says she wants, as the place's own word."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.cognition.what_this_place_is import SAID, Reading

    guide = THE_GUIDE.get()
    if guide is None or at - float(getattr(run, "measured_at", -math.inf)) < EVERY_S:
        return
    run.measured_at = at
    facts = what_play_measures(moves, hers, meeting, getattr(run, "contest", None), getattr(run, "bars", None))
    guide.reading.measure(facts)
    wanted = _wanted(facts)
    if wanted and (guide.reading.now is None or not guide.reading.now.want):
        guide.reading.take(Reading(want=wanted, rests_on=SAID))
