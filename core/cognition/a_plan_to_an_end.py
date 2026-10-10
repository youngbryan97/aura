"""A plan from where things stand to what she is there to do: what the place gives her to work with, what each thing
does and what it works with, and the steps that join the start to the end, each an act of hers using one thing for
another. Followed a step at a time, held to what happens, and made again when what happens says it was wrong.

A person who sits down to a game of devices does not click about to see what happens. They see what they are there to
do (catch Jerry in the cage), where things start (the mouse trap), and what they have (a library of devices: one falls,
one rolls, one launches). They know what joins what (a device takes over at the end of the last one's arrow). And they
think in uses: I need the ball to roll to the seesaw, so the seesaw throws the anvil onto the cage. Then they put the
first part on, turn it toward where the next must go, and add the next. They test the whole, see where it broke, and
change that link. LIVE 2026-10-10 she clicked a device library's tabs and the shapes on its page about forty times,
and tested an empty trap.

So she asks her own model for the plan, from all the guide holds (what the place says it is for, its lesson as read,
the manual made from its code, what others wrote, what she has seen), with what is on the screen now and what the last
try came to. What comes back is held to the place: a thing a step uses, or a means it names, must be something the
place has shown or said. Nothing is invented. The steps become a procedure as a lesson's are
(core/cognition/reading_the_rules.py): the next one not behind her leads her choice of move, a step that keeps doing
nothing is passed over, and the plan is part of what she reasons with and says aloud as uses ("using the anvil to
drop onto the cage"). When a try ends without the end reached, the plan is made again with what happened.

Each step also says what will show once it is done, and what the screen shows next is held to it, as the robots that
plan in words check each step's success before the next (Inner Monologue, 2022) and the agents that play unseen games
abort a plan whose step did not do what was expected (explore, verify, plan: ARC-AGI-3, 2026). A step that did not show
what it should is not done, what it was to show and what showed instead are kept, and a second such miss has the plan
made again with them, described, as the planners that explain a failure before replanning do (DEPS, 2023; REFLECT,
2023). And what she has found each thing on the screens does (core/agency/where_things_lead.py) is part of what the plan
is made from: what the place lets her use, and in what ways.

Nothing here knows a place.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from core.cognition.reading_the_rules import Frame, Rules

__all__ = ["Means", "Plan", "ask_for_a_plan", "plan_again", "the_screen_answered"]

logger = logging.getLogger("Aura.APlanToAnEnd")

#: The most steps a plan holds, means it names, tokens her model writes for it, and of what she knows put before it.
MOST_STEPS = 10
MOST_MEANS = 10
MOST_TOKENS = 1000
MOST_KNOWN = 5000

_WORD = re.compile(r"[a-z]{3,}")
_COMMON = frozenset("""the and you your for with from that this into onto any each other another one all its it's
    then them they there here what when where which while will can use using used make made get put set""".split())


@dataclass(frozen=True)
class Means:
    """One thing the place gives her to work with: what it does, what it works with, and whether it works alone."""

    name: str
    does: str = ""
    works_with: str = ""
    on_its_own: bool = True

    def said(self) -> str:
        how = "; ".join(p for p in (self.does, f"works with {self.works_with}" if self.works_with else "",
                                    "" if self.on_its_own else "not on its own") if p)
        return f"{self.name} ({how})" if how else self.name


@dataclass
class Plan(Rules):
    """Her plan: what has to be true at the end, where things start, what she has to work with, and the steps, kept as
    a procedure is (core/cognition/reading_the_rules.py) but neither kept for every place nor taught to the surfaces
    that learn from a place's own words."""

    end: str = ""
    start: str = ""
    means: list[Means] = field(default_factory=list)
    because: str = ""
    #: What each step (by its key) is to show once done; the step done last and awaiting the screen's answer; and what
    #: steps showed instead of what they were to.
    expects: dict[str, str] = field(default_factory=dict)
    awaiting: str = ""
    missed: list[str] = field(default_factory=list)
    #: Whether a step could not be done as written, for the plan to be made again around it.
    stuck: bool = False

    def took(self, frames: Sequence[Frame]) -> list[Frame]:
        taken = []
        for frame in frames:
            if frame.sentence not in self.heard:
                self.heard.append(frame.sentence)
            frame = replace(frame, order=self.heard.index(frame.sentence))
            self.frames[frame.key] = frame
            taken.append(frame)
        return taken

    def tried(self, move: str, changed: bool) -> None:
        """A move made: the step it does done as a lesson's is, and, where it says what will show, awaiting the screen. A
        step tried until it is passed over is a step that could not be done as written: said, for the plan to be made
        again around it or broken down (ADaPT, 2023: a task is decomposed when doing it fails)."""
        from core.cognition.reading_the_rules import PASSED_OVER_AFTER

        frame = self.step_of(move)
        super().tried(move, changed)
        if frame is not None and changed and self.expects.get(frame.key):
            self.awaiting = frame.key
        if frame is not None and not changed and self.unanswered_steps.get(frame.key, 0) == PASSED_OVER_AFTER:
            self.missed.append(f"“{frame.sentence}” did nothing when tried ({move})")
            del self.missed[:-4]
            self.stuck = True

    def saw(self, shown: str) -> bool:
        """What the screen shows after a step that said what would: the step stands where most of what it was to show
        is there, and is not done where it is not, with what showed kept. Whether the plan is to be made again: a second
        step in a row that did not show what it should."""
        key, self.awaiting = self.awaiting, ""
        frame, expected = self.frames.get(key), self.expects.get(key, "")
        wanted = _words(expected)
        if frame is None or not wanted:
            return False
        if len(wanted & _words(shown)) * 2 >= len(wanted):
            self.missed.clear()
            return False
        self.done.discard(key)
        self.missed.append(f"“{frame.sentence}” was to show {expected}; the screen showed: {' '.join(shown.split())[:160]}")
        del self.missed[:-4]
        return len(self.missed) >= 2

    def as_uses(self) -> list[str]:
        """Its steps as uses, as she says them: "using the anvil to drop onto the cage"."""
        out = []
        for f in self.steps():
            used = f.thing or f.using
            why = f.for_what or (f"to {f.where}" if f.where else "")
            if why and not re.match(r"(?:to|so|for|in order|until)\b", why, re.I):
                why = f"to {_lower(why)}"
            out.append(f"using {used} {why}".strip() if used else f.sentence)
        return out

    def said(self) -> str:
        uses = self.as_uses()
        if not uses:
            return ""
        lead = " ".join(p for p in (f"What I'm after: {_lower(self.end)}." if self.end else "",
                                     f"It starts from {_lower(self.start)}." if self.start else "") if p)
        return (f"{lead} " if lead else "") + "My plan: " + "; then ".join(uses[:4]) + ("; and so on." if len(uses) > 4
                                                                                        else ".")

    def for_thinking(self) -> str:
        steps = self.steps()
        if not steps:
            return ""
        nxt = self.next_step()
        lines = [f"What I'm there to get to: {self.end}" if self.end else "",
                 f"Where it starts: {self.start}" if self.start else "",
                 "What I can use: " + "; ".join(m.said() for m in self.means[:MOST_MEANS]) if self.means else "",
                 "My plan, in order: " + "; ".join(("→ " if f is nxt else "✓ " if self.passed(f) else "") + f.sentence
                                                     for f in steps)]
        return "\n".join(line for line in lines if line)


def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(str(text or "").lower()) if w not in _COMMON}


def _shown(part: str, known: set[str]) -> str:
    """A part of a step kept only where every word of what it names is one the place has shown or said."""
    words = _words(part)
    return part if words and words <= known else ""


def _what_she_knows(guide: Any, on_screen: Sequence[str]) -> str:
    rules = getattr(guide, "rules", None)
    heard = " | ".join(getattr(rules, "heard", []) or [])
    found = list(getattr(guide, "found_to_do", None) or [])
    built = str(getattr(guide, "built", "") or "")
    return "\n".join(p for p in (guide.for_thinking(), f"Its words, as shown: {heard}" if heard else "",
                                 "What they have found things here do: " + " | ".join(found) if found else "",
                                 f"What they have built so far: {built}" if built else "",
                                 "On the screen now: " + ", ".join(on_screen[:40]) if on_screen else "") if p)[:MOST_KNOWN]


async def _a_plan(guide: Any, on_screen: Sequence[str], happened: str, ask: Callable[..., Awaitable[Any]]) -> Plan | None:
    from typing import Literal

    from pydantic import BaseModel, Field

    from core.agency.ways_of_playing import WAYS
    from core.cognition.asking_in_turn import THE_PLACE
    from core.cognition.what_things_are import asked_patiently
    from core.cognition.what_this_place_is import _ACTS

    class _Means(BaseModel):
        name: str = Field(max_length=60, description="its name, as the place shows or says it")
        does: str = Field(default="", max_length=140, description="what it does when used")
        works_with: str = Field(default="", max_length=80, description="what it connects to or is used with, if anything")
        on_its_own: bool = Field(default=True, description="false where it does nothing unless joined with something")

    class _Step(BaseModel):
        step: str = Field(max_length=160, description="the step, as she would say it")
        act: Literal[_ACTS] = Field(description="the way it is done, as one of the ways listed")  # type: ignore[valid-type]
        thing: str = Field(default="", max_length=60, description="what she uses or acts on, by its name there")
        where: str = Field(default="", max_length=60, description="where it goes or is done, by its name there")
        using: str = Field(default="", max_length=60, description="the control it is done with, by its name there")
        for_what: str = Field(default="", max_length=100, description="what this step achieves toward the end")
        again: bool = Field(default=False, description="true where it is done again and again until something")
        expect: str = Field(default="", max_length=100, description="what will show on the screen once it is done")

    class _Plan(BaseModel):
        end: str = Field(default="", max_length=160, description="what has to be true when it is done")
        start: str = Field(default="", max_length=120, description="where things start")
        means: list[_Means] = Field(default_factory=list, max_length=MOST_MEANS)
        steps: list[_Step] = Field(default_factory=list, max_length=MOST_STEPS)

    ways = "; ".join(f"{w.name}: {w.asks}" for w in WAYS if w.name in _ACTS)
    known = _what_she_knows(guide, on_screen)
    prompt = (f"Someone is in “{getattr(guide, 'place', '') or 'a place on a screen'}”. What they know of it:\n{known}\n"
              + (f"Their last try: {happened}\n" if happened else "")
              + "Make their plan: what has to be true at the end, where things start, what the place gives them to work "
              "with (what each does, what it connects to or is used with, whether it works on its own), and the steps "
              f"from the start to the end, in order, each one of these ways ({ways}), naming what is used, where, with "
              "which control, what it achieves toward the end, and what will show once it is done. Name only things "
              "the place shows or says."
              + (" Change what the last try showed was wrong." if happened else ""))
    try:
        got = await asked_patiently(ask, prompt, _Plan, MOST_TOKENS, matters=THE_PLACE)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("a plan could not be made: %s", str(why)[:160])
        return None
    if got is None:
        return None
    return _held_to_the_place(got.model_dump(), _words(known), happened)


def _held_to_the_place(said: dict[str, Any], known: set[str], happened: str) -> Plan:
    """The plan her model made, with each thing it names held to what the place has shown or said."""
    plan = Plan(end=str(said.get("end") or ""), start=str(said.get("start") or ""), because=happened)
    for one in said.get("means") or []:
        if _shown(one.get("name") or "", known):
            plan.means.append(Means(one["name"], one.get("does") or "", _shown(one.get("works_with") or "", known),
                                    bool(one.get("on_its_own", True))))
    frames = []
    for one in said.get("steps") or []:
        thing, where, using = (_shown(one.get(k) or "", known) for k in ("thing", "where", "using"))
        if not (thing or where or using):
            continue                        # a step that names nothing the place has is no step of this place
        frames.append(Frame(sentence=" ".join(str(one.get("step") or "").split()) or f"{one['act']} {thing}",
                            act=str(one.get("act") or ""), thing=thing, where=where, using=using,
                            for_what=str(one.get("for_what") or ""), again=bool(one.get("again"))))
        plan.expects[frames[-1].key] = " ".join(str(one.get("expect") or "").split())
    plan.took(frames)
    return plan


def ask_for_a_plan(guide: Any, ask: Callable[..., Awaitable[Any]] | None, *, on_screen: Sequence[str] = (),
                   tell: Callable[[str], Any] | None = None) -> bool:
    """Ask her model for a plan beside her work, once there is an end to aim at and something to work with, and again
    when a try said the last one was wrong (``plan_again``). Whether asked now."""
    if guide is None or ask is None or getattr(guide, "planning", None) is not None and not guide.planning.done():
        return False
    again = str(getattr(guide, "plan_again_because", "") or "")
    if guide.plan is not None and not again:
        return False
    rules = getattr(guide, "rules", None)
    has_an_end = bool(guide.goals or (rules is not None and rules.what_it_is_for()))
    has_means = bool((rules is not None and rules.steps()) or getattr(guide, "from_its_code", None) or on_screen)
    if not (has_an_end and has_means):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    guide.plan_again_because = ""

    async def asked() -> None:
        plan = await _a_plan(guide, list(on_screen), again, ask)
        if plan is None or not plan.steps():
            logger.info("her model gave no plan for %r", getattr(guide, "place", ""))
            return
        guide.plan = plan
        logger.info("her plan: %s", [asdict(f) for f in plan.steps()][:MOST_STEPS])
        if tell is not None and plan.said():
            tell(("Changing my plan. " if again else "") + plan.said())

    guide.planning = loop.create_task(asked())
    return True


def the_screen_answered(guide: Any, shown: str) -> None:
    """What the screen shows now, held to what her plan's last step was to show: a second step in a row that did not show
    it has the plan made again, with what each was to show and what showed instead."""
    plan = getattr(guide, "plan", None)
    if plan is not None and (plan.awaiting and plan.saw(shown) or plan.stuck):
        plan.stuck = False
        logger.info("her plan's steps did not do what they were to: %s", plan.missed)
        guide.plan_again_because = " ".join(plan.missed)


def plan_again(guide: Any, happened: str) -> None:
    """A try ended without the end reached: the next plan is made with what happened."""
    if guide is not None and getattr(guide, "plan", None) is not None:
        guide.plan_again_because = " ".join(str(happened or "").split())[:240]


def _lower(text: str) -> str:
    from core.language.words_of_the_language import as_said_inside

    return as_said_inside(str(text).rstrip("."))
