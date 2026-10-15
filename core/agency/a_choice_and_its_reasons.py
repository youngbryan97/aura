"""Each choice of her fast play, bound to the act that carries it out, the reason for it, the craft of people's play
it is a case of, what she knows that it rests on, and what should follow by when; and, once that time is up, what did.

People choose in play by recognising. A situation seen as a typical one brings with it what to watch, what to expect,
what to aim at and what to do, and an expectation that breaks sends them back to look again (the recognition-primed
decision, Klein 1993). Asked why, a player names the cue, the aim and what they counted on: "it lands there in under a
second, and I can get there". Her play chose by a ladder (core/agency/playing_as_it_happens.py: meet what arrives
first, walk to what is on her level, line up a shot, use what she holds, cover ground, wait in the middle) and kept one
word for why.

Here each choice is a record: the cue, the act, the reason, the craft (core/agency/the_craft_of_play.py, drawn from
people's play), what she counts on with how she knows it (her own measurements), and what should follow by when. A
record is made when the choice changes, not every picture. It is settled when its time is up, against what met her and
what she gained and lost; one set aside for another before then counts for neither. How each craft's choices came out
is kept with the game. Nothing here decides an act: it binds the acts her play makes to their reasons and outcomes,
says them, and gives them to whatever reasons about her play (docs/design-docs/HOW_PEOPLE_DECIDE_AS_THEY_PLAY.md).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CHOICES", "Choice", "ChoicesMade", "chose_in_play", "how_soon_she_reaches", "reachable_arrivals", "the_arrival_to_meet", "the_choice_now"]

#: What a choice expects to follow: the thing it is for met; a gain while a shot is out; nothing costing her; her thing
#: at the place it is for.
MEETS, HITS, UNHARMED, REACHES = "meets", "hits", "unharmed", "reaches"
#: How a record came out.
AS_EXPECTED, NOT_AS_EXPECTED, SET_ASIDE = "as expected", "not as expected", "set aside"


@dataclass(frozen=True)
class KindOfChoice:
    """One rung of her play's ladder, as people play it."""

    craft: str
    because: str
    expects: str
    expects_said: str


#: Her play's words for why, each as the craft of people's play it is a case of. Every craft named is one of
#: core/agency/the_craft_of_play.py's (a test holds it to that).
CHOICES: Mapping[str, KindOfChoice] = {
    "meet": KindOfChoice("getting under what to catch", "to be where {aim} gets to when it does", MEETS, "meeting {aim}"),
    "shoot": KindOfChoice("lining up to throw or shoot", "to be in line with {aim} when my shot gets there", HITS,
                          "a shot of mine paying"),
    "use": KindOfChoice("opening every door", "to use what I was told of where it works", REACHES, "getting there"),
    "cover": KindOfChoice("going screen to screen", "to see ground I have not been over", REACHES, "getting there"),
    "wait": KindOfChoice("sidestepping what falls", "to keep both ways open while nothing needs me", UNHARMED,
                         "nothing costing me"),
}


@dataclass
class Choice:
    """One choice: what she noticed, what she did, why, as what craft, counting on what, expecting what by when."""

    at: float
    why: str
    act: str
    aim: int | None
    aim_named: str
    craft: str
    because: str
    cue: str
    knows: tuple[str, ...]
    expects: str
    by: float
    where: tuple[float, float] | None = None
    near: float = 0.0
    gains: int = 0
    losses: int = 0
    came_to: str = ""

    def said(self) -> str:
        """The choice as a player says it: what, why, and what it counts on."""
        doing = f"Holding {self.act}" if self.act else "Staying put"
        line = f"{doing} {self.because.format(aim=self.aim_named or 'it')}"
        cue = f": {self.cue}" if self.cue else ""
        known = f" ({'; '.join(self.knows)})" if self.knows else ""
        return f"{line}{cue}{known}."


@dataclass
class ChoicesMade:
    """Her choices in one game, open and settled, and how each craft's came out."""

    open: Choice | None = None
    settled: list[Choice] = field(default_factory=list)
    came_out: dict[str, Counter] = field(default_factory=dict)
    said_crafts: set[str] = field(default_factory=set)

    def chose(self, choice: Choice, say: Callable[[str], Any] | None = None) -> bool:
        """Take a choice in; whether it was a new one. The same aim for the same reason is the same choice held."""
        held = self.open
        if held is not None and held.why == choice.why and held.aim == choice.aim and not held.came_to:
            held.act = choice.act
            return False
        if held is not None and not held.came_to:
            self._settle(held, SET_ASIDE)
        self.open = choice
        if say is not None and choice.craft not in self.said_crafts:
            self.said_crafts.add(choice.craft)
            say(choice.said())
        return True

    def settle(self, at: float, *, met: Mapping[int, float], gains: int, losses: int,
               mine_at: tuple[float, float] | None = None, say: Callable[[str], Any] | None = None) -> str:
        """Hold the open choice to what followed; what it came to, once it has come to something."""
        choice = self.open
        if choice is None or choice.came_to:
            return ""
        lost, gained = losses > choice.losses, gains > choice.gains
        when_met = met.get(choice.aim) if choice.aim is not None else None
        if choice.expects == MEETS and when_met is not None and when_met >= choice.at:
            came = AS_EXPECTED
        elif choice.expects == HITS and gained:
            came = AS_EXPECTED
        elif choice.expects == UNHARMED and lost:
            came = NOT_AS_EXPECTED
        elif (choice.expects == REACHES and mine_at is not None and choice.where is not None
              and math.hypot(mine_at[0] - choice.where[0], mine_at[1] - choice.where[1]) <= choice.near):
            came = AS_EXPECTED
        elif at >= choice.by:
            came = AS_EXPECTED if choice.expects == UNHARMED else NOT_AS_EXPECTED
        else:
            return ""
        self._settle(choice, came)
        tally = self.came_out.setdefault(choice.craft, Counter())
        # Two that did not come out as counted on, and none that did, is the point a player says the way is not working.
        if say is not None and came == NOT_AS_EXPECTED and tally[NOT_AS_EXPECTED] == 2 and not tally[AS_EXPECTED]:
            say(f"Twice now I counted on {_expected(choice)}, and it didn't happen.")
        return came

    def _settle(self, choice: Choice, came: str) -> None:
        choice.came_to = came
        self.settled.append(choice)
        del self.settled[:-200]
        if came != SET_ASIDE:
            self.came_out.setdefault(choice.craft, Counter())[came] += 1

    def how_it_came_out(self) -> list[str]:
        """Each craft's settled choices here, as lines to reason with."""
        return [f"{craft}: {tally[AS_EXPECTED]} of {tally[AS_EXPECTED] + tally[NOT_AS_EXPECTED]} as expected"
                for craft, tally in sorted(self.came_out.items(), key=lambda item: -sum(item[1].values()))]

    def summary(self) -> dict[str, Any]:
        """What her choices came to, for the record of a run."""
        return {"by_craft": {craft: dict(tally) for craft, tally in self.came_out.items()},
                "last": [c.said() + f" It came out {c.came_to}." for c in self.settled[-5:] if c.came_to != SET_ASIDE]}


def _expected(choice: Choice) -> str:
    kind = CHOICES.get(choice.why)
    return (kind.expects_said if kind is not None else "what I expected").format(aim=choice.aim_named or "it")


def the_choice_now(choosing: Any, key: str, why: str, aim: Any, at: float, *, physics: Any = None,
                   named: Callable[[Any], str] | None = None) -> Choice | None:
    """The record of the choice her play just made: its craft and reason from its rung, its cue and what it counts on
    from her own measurements, and when what it expects should have happened by."""
    kind = CHOICES.get(why)
    mine = getattr(choosing, "mine", None)
    if kind is None or mine is None:
        return None
    respond = float(getattr(choosing, "response_s", 0.0) or 0.0) + float(getattr(choosing, "next_picture_s", 0.0) or 0.0)
    aim_named = named(aim) if (named is not None and aim is not None) else ""
    knows: list[str] = []
    cue, by, where, near = "", at + max(respond, 0.0), None, 0.0
    size = max(float(mine.w), float(mine.h))
    arrival = getattr(choosing, "arrival", None)
    if why == "meet" and arrival is not None:
        when, _there, reach_s = arrival
        cue = f"it gets to me in {when:.1f} s" + (f" and I can be there in {reach_s:.1f} s" if reach_s is not None else "")
        by = at + when + respond
    elif why == "meet" and aim is not None:
        far = math.hypot(aim.x - mine.x, aim.y - mine.y)
        fastest = max(choosing.speed(0), choosing.speed(1))
        cue = f"it is {far:.0f} px from me"
        by = at + far / max(1.0, fastest) + respond
    elif why in ("use", "cover"):
        target = choosing.going_to if why == "use" else None
        target = target or _ground(choosing)
        if target is not None:
            where, near = (float(target[0]), float(target[1])), size
            far = math.hypot(where[0] - mine.x, where[1] - mine.y)
            by = at + 2 * far / max(1.0, max(choosing.speed(0), choosing.speed(1))) + respond
            cue = f"it is {far:.0f} px from me"
    elif why == "shoot" and getattr(choosing, "shot", None) is not None and aim is not None:
        shot = choosing.shot
        flight = max(1.0, math.hypot(shot.vx, shot.vy))
        by = at + math.hypot(aim.x - mine.x, aim.y - mine.y) / flight + respond
    elif why == "wait":
        by = at + max(getattr(choosing, "ahead_s", 0.0) or 0.0, 1.0)
    for axis, way in ((0, "across"), (1, "up and down")):
        if choosing.ways and (getattr(choosing, "across", False), getattr(choosing, "updown", False))[axis]:
            knows.append(f"I go about {choosing.speed(axis):.0f} px a second {way}, measured")
    if aim is not None and physics is not None:
        pull = physics.gravity(int(aim.kind)) if hasattr(physics, "gravity") else (0.0, 0.0)
        if any(pull):
            knows.append(f"things like it are pulled at {math.hypot(*pull):.0f} px/s², measured over their falls")
    if aim is not None and choosing.meeting.known(aim.kind):
        knows.append("meeting one has counted for me before" if why == "meet" else "I have met its kind before")
    return Choice(at=at, why=why, act=key, aim=getattr(aim, "number", None), aim_named=aim_named, craft=kind.craft,
                  because=kind.because, cue=cue, knows=tuple(knows), expects=kind.expects, by=by, where=where,
                  near=near)


def _ground(choosing: Any) -> tuple[float, float] | None:
    found = choosing._ground_not_yet_covered(choosing.line()) if hasattr(choosing, "_ground_not_yet_covered") else None
    if found is None:
        return None
    mine = choosing.mine
    return (found[0] if found[0] is not None else mine.x, found[1] if found[1] is not None else mine.y)


def chose_in_play(choices: ChoicesMade | None, choosing: Any, key: str, why: str, aim: Any, at: float,
                  say: Callable[[str, str], Any]) -> None:
    """The choice her fast play just made, taken in with its reason, craft, what it counts on and what should follow;
    said once a craft."""
    from core.agency.naming_what_she_sees import describe
    from core.perception.recording_the_frames_and_inputs import record_play_choice

    record_play_choice(choosing, key, why, aim, at)

    if choices is None:
        return
    choice = the_choice_now(choosing, key, why, aim, at, physics=getattr(choosing, "physics", None),
                            named=lambda thing: describe(choosing.moves, thing.kind, thing))
    if choice is not None:
        choices.chose(choice, lambda line: say(line, f"choice {choice.craft}"))


def the_arrival_to_meet(choosing: Any, arrivals: list[tuple[float, float, Any]], free: int,
                        her_free: float) -> tuple[float, float, Any]:
    """Of the things arriving at her line (when, where, which), the soonest she can reach in time; the soonest of all
    where she can reach none. A catcher lets go of what cannot be reached: chasing one and missing two that could have
    been caught is the classic mistake (core/agency/the_craft_of_play.py). Before her speed is measured, every arrival
    counts as reachable. Kept on ``choosing.arrival`` for the choice's cue."""
    chosen = min(reachable_arrivals(choosing, arrivals, free, her_free) or arrivals, key=lambda a: a[0])
    choosing.arrival = (chosen[0], chosen[1], how_soon_she_reaches(choosing, chosen, free, her_free))
    return chosen


def reachable_arrivals(choosing: Any, arrivals: list[tuple[float, float, Any]], free: int,
                       her_free: float) -> list[tuple[float, float, Any]]:
    """The arrivals she can get to in time, every one where her speed is not yet measured."""
    return [a for a in arrivals if (r := how_soon_she_reaches(choosing, a, free, her_free)) is None or r <= a[0]]


def how_soon_she_reaches(choosing: Any, arrival: tuple[float, float, Any], free: int, her_free: float) -> float | None:
    """How soon she can be where an arrival comes, or None where her speed is not known."""
    if choosing.speed(free) <= 1.0:
        return None
    mine = choosing.mine
    _when, there, thing = arrival
    gap = abs(there - her_free) - ((mine.w, mine.h)[free] + (thing.w, thing.h)[free]) / 2
    return max(0.0, gap) / choosing.speed(free) + choosing.response_s

