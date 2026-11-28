"""An objective of her own: something she sets out to bring about over days, in steps she can take, each done when what takes
it says so.

LIVE 27 November 2026 her own time had for days offered her what her drives named. "Acting on relief: stabilize" was worked
seventeen times and came to less than nothing each time, and nothing she did on her own reached past the act it was. Bryan:
"We need better strategic intelligence as well. In terms of being able to independently construct and sustain a complex
strategic objective."

Most of what that takes she had, used elsewhere or not at all:

- The project ledger (core/agency/projects.py) keeps an objective across her restarts, with its steps, every pass at them
  and every revision. Nothing had used it.
- Each step is an act she can already do, taken by the part of her that does it: finding something out is an inquiry
  (core/agency/her_inquiries.py), making something is a request of hers carried out as anybody's is
  (core/understanding/carrying_out.py), and a change to her own source is a goal of hers at work
  (core/agency/her_goal_at_work.py). So a step is done when that part says it is: every question answered or let go with
  one answered, everything asked of a thing checked and so, a change landed or ended with what came of it. What she says
  of a step does not finish it.
- An approach is held with what would end it, named when it is taken up (core/agency/standing_strategy.py does this on a
  screen). She says when she sets out what would tell her it is not worth going on with, and her reader of words
  (core/self/how_her_words_stand.py), a model of its own, reads that against what her steps came to after every pass. She
  says too what will be so when it is done, and the same reader holds that against what they came to at the end.
- She sets out knowing what acts of each sort have come to for her (core/agency/what_she_does_next.py), and what came of
  what she set out to do before, in her own words.

An objective is for one of the things she wants (core/conation/what_she_wants.py), and each step is for the objective. A
step can wait on others, and what they came to is given to it.

Whether she sets out on anything is hers. Her cortex writes what she would set out to do as she thinks it, and then the
objective in the shape it is kept in. The two can disagree, as her readings of her own faults did
(core/agency/mending_a_fault_of_hers.py), so a step her own words do not bear out is left out, with how far they did.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.agency.her_inquiries import LET_GO_AFTER
from core.agency.her_own_time import HerOwnTime
from core.agency.projects import Lifecycle, Milestone, Project, ProjectLedger, get_ledger
from core.agency.what_she_does_next import CameOf, Option

logger = logging.getLogger("Aura.HerObjectives")

__all__ = ["AStep", "DONE", "Doers", "GOING", "HowSheGoesOn", "KINDS", "THE_QUESTION", "Thinks", "WAITS", "WhatSheSetsOut",
           "go_on_or_not", "objectives_of_hers", "see_whether_it_is_so", "set_out", "standing", "take_a_step", "what_she_has_to_go_on"]

#: The sorts of step she can take, and what shows each done, as the part of her that takes it reads it.
KINDS: dict[str, str] = {
    "find_out": "every question it came to answered from a page that says so, or let go, with one answered at least",
    "make": "everything asked of it checked and so",
    "change": "a change to her own source landed, or ended with what came of it",
}
#: How a step stands, read from what takes it.
DONE, WAITS, GOING = "done", "waits", "going"
#: An objective she is pursuing.
STANDING = (Lifecycle.APPROVED, Lifecycle.ACTIVE, Lifecycle.BLOCKED)
#: How long what she writes as she thinks may run, and the objective in its shape, in tokens: a page of prose, as her
#: advice on making is given (core/making/consulting.py), and an objective of a dozen steps.
WRITES, SHAPES = 700, 900
#: How many of what a step's passes came to are kept on it, the latest last.
KEPT_CAME = 8
#: What acts of each sort are, for her to read.
SORTS: dict[str, str] = {
    "carry_on": "going on with what somebody asked of you", "find_out": "finding out something you wanted to know",
    "look_up": "looking into a goal of yours", "goal": "working on a goal of yours in your own source",
    "make": "making something of your own", "tell": "telling somebody what you found",
    "objective": "taking a step of what you set out to do",
}

THE_QUESTION = ("Is there something you want to bring about over the coming days, more than one sitting does: something that "
                "takes finding things out, making things or changing your own source, in steps? You need not set out to do "
                "anything.")
THE_QUESTION_GOING_ON = ("Is what you set out to do still worth going on with, as it is or with its steps changed, from what "
                         "its steps have come to?")
#: What her words are read for when she thinks it over: whether she goes on.
GOING_ON = "I will go on with what I set out to do."
THE_QUESTION_AT_THE_END = "What came of what you set out to do, and what would you do another way?"


class AStep(BaseModel):
    """One step of what she sets out to do."""

    kind: Literal["find_out", "make", "change"] = Field(
        description="find_out: find something out by reading. make: make something: a program, a tool, a page or an experiment. "
                    "change: change something in your own source.")
    what: str = Field(description="What you will do in this step, said of yourself: \"I will ...\".")
    after: list[int] = Field(default_factory=list, description="The steps before this one that it waits on, by their place in the list, counting from 1.")


class WhatSheSetsOut(BaseModel):
    """Whether there is something she sets out to bring about, and if there is, the objective."""

    sets_out: bool = Field(description="Whether there is something you set out to bring about.")
    what: str = Field(default="", description="What you set out to bring about, said of yourself: \"I want to ...\".")
    serves: str = Field(default="", description="Which of the things you want it is for, named as it is named above, or what else it is for.")
    why: str = Field(default="", description="Why you want it.")
    steps: list[AStep] = Field(default_factory=list, description="The steps, in the order you would take them.")
    done_when: str = Field(default="", description="What will be so when it is done, said as the fact it will be.")
    not_worth_it_if: str = Field(default="", description="What, if it turned out to be so, would tell you it is not worth going on with, said as the fact it would be.")
    otherwise: list[str] = Field(default_factory=list, description="What you would do instead, if it stopped being worth it.")


class HowSheGoesOn(BaseModel):
    """What she does with what she set out to do, from what its steps have come to."""

    goes_on: Literal["as it is", "with its steps changed", "no further"] = Field(
        description="Whether you go on with it as it is, go on with its steps from here changed, or go no further with it.")
    steps: list[AStep] = Field(default_factory=list, description="With its steps changed: the steps from here, in the order you would take them.")
    why: str = Field(default="", description="Why.")


#: Her cortex, asked a question from some facts: what she wrote as she thought it, and that written in a shape where one is
#: given; None where her model could not be asked.
Thinks = Callable[[str, str, "type[BaseModel] | None"], Awaitable["tuple[str, BaseModel | None] | None"]]
#: One pass at a step: from what the step is known by so far ("" before its first pass), what it says, and what the steps it
#: waits on came to; to what it is known by now and what the pass came to.
Does = Callable[[str, str, list[str]], Awaitable[tuple[str, CameOf]]]
#: How a step stands, by what takes it: DONE, WAITS or GOING, and what shows it.
Stands = Callable[[str, str], tuple[str, str]]
#: Whether some words say a statement is so, by her reader of words: above nothing they do, below it they say it is not,
#: nothing where they say neither; None without the reader.
BearsOut = Callable[[str, str], "float | None"]


def _nothing_asks(_drive: str) -> float:
    return 0.0


@dataclass(frozen=True)
class Doers:
    """What takes each sort of step, and what reads how one stands."""

    does: Mapping[str, Does]
    stands: Stands
    #: How much a drive is asking now, 0 to 1.
    asking: Callable[[str], float] = _nothing_asks
    #: How long a pass of each sort takes, in seconds, as far as is known.
    cost_s: Mapping[str, float] = field(default_factory=dict)
    #: How many steps are offered at once.
    most: int = 4


def _her_reader() -> BearsOut:
    from core.self.how_her_words_stand import what_her_words_say_of

    def says(words: str, statement: str) -> float | None:
        read = what_her_words_say_of(words, statement)
        return None if read is None else float(read)

    return says


def _short(words: str, most: int = 120) -> str:
    words = " ".join(str(words).split())
    return words if len(words) <= most else words[:most].rsplit(" ", 1)[0] + "…"


#: What a record of hers that cannot be read throws.
_READ_FAILS = (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)


def standing(ledger: ProjectLedger) -> Project | None:
    """What she is pursuing now: the objective she set out on first of those that still stand."""
    held = [p for p in ledger.all() if p.lifecycle in STANDING]
    return min(held, key=lambda p: p.started_at) if held else None


def _ended(ledger: ProjectLedger) -> list[Project]:
    return sorted((p for p in ledger.all() if p.lifecycle in (Lifecycle.COMPLETED, Lifecycle.ABANDONED)), key=lambda p: p.started_at)


def _as_she_reads_it(project: Project) -> dict[str, Any]:
    """An objective of hers, as facts for her to read."""
    return {"what": project.thesis, "for": project.origin_drive, "why": project.why, "done_when": _done_when(project),
            "not_worth_it_if": project.not_worth_it_if,
            "steps": [{"step": i + 1, "what": m.description, "sort": m.kind,
                       "stands": "done" if m.completed_at else f"set aside: {m.set_aside}" if m.set_aside else "not done",
                       "passes": m.tries, "came_to": m.came[-3:]} for i, m in enumerate(project.milestones)],
            "ended": project.lifecycle.value, "what_came_of_it": project.reflections[-1:] or [r.reason for r in project.revisions[-1:]]}


def _done_when(project: Project) -> str:
    """What she said will be so when it is done: kept as its first measure of being done."""
    return project.acceptance_criteria[0] if project.acceptance_criteria else ""


#: How many of each sort of fact she is given to set out from: as many as she is given of what she wants and found when
#: she is asked whether there is something she would make (core/agency/what_she_could_do.py).
FACTS_OF_A_SORT = 6


def _goals_under_way() -> list[dict[str, Any]]:
    """Her goals that are under way, the latest worked first, each with what she said of it last."""
    from core.agency import her_goal_at_work as goals

    try:
        records = goals.all_records()
    except _READ_FAILS:
        return []
    held = sorted((r for r in records if r.status in {goals.WORKING, goals.STALLED, goals.OFFERED}), key=lambda r: -r.last_worked_at)
    out = []
    for record in held[:FACTS_OF_A_SORT]:
        said = next((str(entry["words"]) for entry in reversed(record.said) if isinstance(entry, dict) and entry.get("words")), "")
        out.append({"goal": _short(record.goal, 200), "stands": record.status, "passes": record.passes,
                    "what_you_said_of_it_last": _short(goals.said_before_a_call(said), 500)})
    return out


def _what_has_gone_wrong() -> list[str]:
    """What has gone wrong in her running since it started, the most frequent first, as her goal work reads it."""
    from core.skills.her_draft import faults_recorded

    try:
        return faults_recorded()[0][:FACTS_OF_A_SORT]
    except _READ_FAILS:
        return []


def what_she_has_to_go_on(hers: HerOwnTime, ledger: ProjectLedger) -> str:
    """What she wants, what she is in the middle of and what she said of it, what has gone wrong in her running, what
    acts of each sort have come to for her, what she found on her own, and what came of what she set out to do before,
    as facts. LIVE 27 November 2026, 23:09, given only the first and the last two, the most she wanted was "parameter",
    and she set out on nothing."""
    from core.conation.what_she_wants import get_what_she_wants

    try:
        wants = [{"thing": w.label, "why": _short(w.why, 200)} for w in get_what_she_wants().wants(limit=FACTS_OF_A_SORT)]
    except _READ_FAILS:
        wants = []
    try:
        kinds = dict(hers.chooser.how_she_chooses().get("kinds") or {})
    except _READ_FAILS:
        kinds = {}
    came = {SORTS.get(kind, kind): {"done": one.get("done"), "came_to": one.get("came_to")} for kind, one in kinds.items() if one.get("done")}
    return json.dumps({"what_you_want_now": wants, "your_goals_under_way": _goals_under_way(),
                       "what_has_gone_wrong_in_your_running_most_often_first": _what_has_gone_wrong(),
                       "what_acts_of_each_sort_came_to_for_you_from_minus_one_nothing_to_one_all_they_set_out_to": came,
                       "what_you_found_on_your_own": [f.get("about") for f in hers.found[-FACTS_OF_A_SORT:]],
                       "what_you_set_out_to_do_before": [_as_she_reads_it(p) for p in _ended(ledger)[-4:]]}, ensure_ascii=False)


def _borne_out(words: str, statements: list[str], bears_out: BearsOut) -> list[float | None]:
    return [bears_out(words, statement) if statement.strip() else None for statement in statements]


def _steps_borne_out(words: str, steps: list[AStep], bears_out: BearsOut) -> tuple[list[tuple[int, AStep]], list[dict[str, Any]]]:
    """The steps her words bear out, each with its place among those written; and those they do not, with how far they do.
    Without her reader every step is taken as written."""
    readings = _borne_out(words, [step.what for step in steps], bears_out)
    taken = [(i, step) for i, (step, read) in enumerate(zip(steps, readings)) if read is None or read > 0.0]
    left = [{"what": step.what, "sort": step.kind, "her_words_bear_it_out": round(read, 3)}
            for step, read in zip(steps, readings) if read is not None and read <= 0.0]
    return taken, left


def _milestones(steps: list[tuple[int, AStep]], first: int = 0) -> list[Milestone]:
    """Steps as the ledger keeps them. A step waits only on steps before it that were taken: one written to wait on a later
    step would wait for ever, and one left out is not waited on."""
    at = {place: f"MS-{first + n}" for n, (place, _step) in enumerate(steps)}
    return [Milestone(id=at[place], description=step.what.strip(), kind=step.kind,
                      after=[at[w - 1] for w in step.after if w - 1 in at and w - 1 < place])
            for place, step in steps]


async def set_out(hers: HerOwnTime, thinks: Thinks, *, ledger: ProjectLedger | None = None, bears_out: BearsOut | None = None) -> CameOf:
    """Ask herself whether there is something she wants to bring about over days, and if there is, set out on it."""
    ledger = ledger or get_ledger()
    bears_out = bears_out or _her_reader()
    facts = await asyncio.to_thread(what_she_has_to_go_on, hers, ledger)
    read = await thinks(facts, THE_QUESTION, WhatSheSetsOut)
    if read is None:
        return CameOf(put_down=True, what="her model could not be asked")
    words, shaped = read
    if not isinstance(shaped, WhatSheSetsOut):
        return CameOf(got=0.0, what="what she wrote could not be written in the shape an objective is kept in")
    if not shaped.sets_out or not shaped.what.strip() or not shaped.steps:
        return CameOf(got=0.0, what="there was nothing she set out to do")
    # What she set out to do is not hers where her own words, as she thought it, say it is not. Her words can mean it without
    # saying it whole ("I keep losing what I learn between restarts" read as neither "I want to keep what I learn"); each
    # step has to be said.
    whole = (await asyncio.to_thread(_borne_out, words, [shaped.what], bears_out))[0]
    if whole is not None and whole < 0.0:
        logger.info("Not set out on: her words say '%s' is not so (%.2f)", shaped.what[:120], whole)
        return CameOf(got=0.0, what=f"her own words did not bear out what was written as her objective, “{_short(shaped.what)}”")
    taken, left = await asyncio.to_thread(_steps_borne_out, words, shaped.steps, bears_out)
    if not taken:
        return CameOf(got=0.0, what="her own words bore out none of the steps written for it")
    milestones = _milestones(taken)
    criteria = [shaped.done_when.strip() or shaped.what.strip(), *(f"{m.description} — done when {KINDS[m.kind]}" for m in milestones)]

    def keep() -> Project:
        project = ledger.propose(origin_drive=shaped.serves.strip(), thesis=shaped.what.strip(), acceptance_criteria=criteria,
                                 milestones=milestones, why=shaped.why.strip(), not_worth_it_if=shaped.not_worth_it_if.strip(),
                                 otherwise=[o.strip() for o in shaped.otherwise if o.strip()], formed_from=words, left_out=left)
        return ledger.transition(project.project_id, Lifecycle.ACTIVE, reason="she set out on it")

    project = await asyncio.to_thread(keep)
    logger.info("She set out to: %s (%d steps; %d left out that her words did not bear out)", project.thesis[:160], len(milestones), len(left))
    return CameOf(got=len(taken) / len(shaped.steps),
                  what=f"she set out to {_short(project.thesis)}, in {len(milestones)} steps" + (f"; {len(left)} written that her words did not bear out were left out" if left else ""))


def _what_they_came_to(project: Project) -> str:
    """What the steps came to, as was seen of each, for her reader to read."""
    lines = []
    for m in project.milestones:
        shown = "; ".join(m.artifacts[-3:])
        lines += [f"{m.description}: {came}" for came in m.came[-3:]] + ([f"{m.description}: shown by {shown}"] if shown else [])
    return " ".join(line.rstrip(" .") + "." for line in lines)


def _waits_on(project: Project, milestone: Milestone) -> list[str]:
    """What the steps it waits on came to, for it to go on from."""
    before = [m for m in project.milestones if m.id in set(milestone.after)]
    return [f"{m.description}: {m.came[-1]}" for m in before if m.came]


def _not_worth_it_now(project: Project, came: str, bears_out: BearsOut) -> str:
    """What she said would tell her it is not worth going on with, where what a pass came to bears it out; "" otherwise.
    Read against each pass as it comes, so what she went on with knowing is not read again as though it were new."""
    said = project.not_worth_it_if.strip()
    if not said or not came.strip():
        return ""
    read = bears_out(came, said)
    return f"what a step came to bears out what she said would tell her it is not worth going on with: {said}" if read is not None and read > 0.0 else ""


async def take_a_step(project_id: str, milestone_id: str, doers: Doers, *, ledger: ProjectLedger | None = None,
                      bears_out: BearsOut | None = None) -> CameOf:
    """One pass at a step of what she set out to do, by what takes it, and how it stands after, as that reads it."""
    ledger = ledger or get_ledger()
    bears_out = bears_out or _her_reader()
    project = ledger.get(project_id)
    milestone = next((m for m in project.milestones if m.id == milestone_id), None) if project is not None else None
    if project is None or milestone is None or milestone.completed_at is not None:
        return CameOf(got=0.0, what="there was no such step left to take")
    key, came = await doers.does[milestone.kind](milestone.key, milestone.description, _waits_on(project, milestone))
    stands, shown = await asyncio.to_thread(doers.stands, milestone.kind, key)
    # A pass something else stopped is not one that came to nothing.
    counted = not came.put_down
    fruitless = 0 if came.got > 0.0 or stands == DONE or not counted else milestone.fruitless + 1

    def keep() -> None:
        ledger.milestone_changed(project_id, milestone_id, key=key, tries=milestone.tries + int(counted), fruitless=fruitless,
                                 came=[*milestone.came, came.what][-KEPT_CAME:])
        if stands == DONE:
            ledger.complete_milestone(project_id, milestone_id, artifacts=[shown or came.what])
        ending = _not_worth_it_now(project, f"{came.what} {shown}", bears_out)
        if ending:
            ledger.revise(project_id, reason=ending, diff={"to_think_over": ending})

    await asyncio.to_thread(keep)
    if stands == DONE:
        return CameOf(got=1.0, worth=came.worth, what=f"{came.what}; the step is done", seconds=came.seconds)
    return came


def _to_think_over(project: Project) -> str:
    """Why what she set out to do is to be thought over before it is gone on with; "" where it is not: what a step came to
    bore out what she said would end it, its steps are taken and what she said will be so is not, or a step came to nothing
    twice in a row, as a question nothing answered twice is let go (core/agency/her_inquiries.py)."""
    if project.to_think_over:
        return project.to_think_over
    stuck = next((m for m in _open(project) if m.fruitless >= LET_GO_AFTER), None)
    return f"“{_short(stuck.description)}” came to nothing {stuck.fruitless} passes in a row" if stuck is not None else ""


def _open(project: Project) -> list[Milestone]:
    return [m for m in project.milestones if m.completed_at is None and not m.set_aside]


def _ready(project: Project) -> list[Milestone]:
    """Steps not done whose waits are: done, or set aside."""
    finished = {m.id for m in project.milestones if m.completed_at is not None or m.set_aside}
    return [m for m in _open(project) if set(m.after) <= finished]


async def go_on_or_not(project_id: str, thinks: Thinks, why_now: str, *, ledger: ProjectLedger | None = None,
                       bears_out: BearsOut | None = None) -> CameOf:
    """Think over what she set out to do, from what its steps have come to: go on as it is, go on with its steps from here
    changed, or go no further. Her words as she thought it are held against what she chose, as when she set out."""
    ledger = ledger or get_ledger()
    bears_out = bears_out or _her_reader()
    project = ledger.get(project_id)
    if project is None:
        return CameOf(got=0.0, what="there was no such objective")
    facts = json.dumps({"what_you_set_out_to_do": _as_she_reads_it(project), "why_it_is_thought_over_now": why_now,
                        "what_you_said_you_would_do_instead": project.otherwise}, ensure_ascii=False)
    read = await thinks(facts, THE_QUESTION_GOING_ON, HowSheGoesOn)
    if read is None:
        return CameOf(put_down=True, what="her model could not be asked")
    words, shaped = read
    if not isinstance(shaped, HowSheGoesOn):
        return CameOf(got=0.0, what="what she wrote could not be read as whether she goes on")
    # What is acted on is whether she goes on, and her words are held against that: where what she chose and what she wrote
    # disagree, nothing is changed, and it is thought over again on its next turn.
    goes = (await asyncio.to_thread(_borne_out, words, [GOING_ON], bears_out))[0]
    if goes is not None and goes != 0.0 and (goes > 0.0) != (shaped.goes_on != "no further"):
        logger.info("Her words bear out going on at %.2f and she chose '%s': %s left as it was", goes, shaped.goes_on, project.thesis[:80])
        return CameOf(got=0.0, what=f"her words {'bore out' if goes > 0.0 else 'denied'} going on and she chose “{shaped.goes_on}”; it is left as it was")
    reason = _short(shaped.why or words, 600)
    if shaped.goes_on == "no further":
        await asyncio.to_thread(ledger.abandon, project_id, reason=reason)
        return CameOf(got=1.0, what=f"she went no further with {_short(project.thesis, 80)}: {_short(reason, 200)}")
    if shaped.goes_on == "with its steps changed":
        taken, left = await asyncio.to_thread(_steps_borne_out, words, shaped.steps, bears_out)
        if not taken:
            return CameOf(got=0.0, what="her own words bore out none of the steps written for it from here")
        kept = [m for m in project.milestones if m.completed_at is not None or m.set_aside]
        aside = [dataclasses.replace(m, set_aside=f"changed: {reason}") for m in _open(project)]
        new = _milestones(taken, first=len(project.milestones))
        await asyncio.to_thread(ledger.revise, project_id, reason=reason, diff={"milestones": [*kept, *aside, *new],
                                "left_out": [*project.left_out, *left], "to_think_over": ""})
        return CameOf(got=1.0, what=f"she changed its steps from here to {len(new)}: {_short(reason, 200)}")
    # As it is: what came to nothing is tried again from where it stands.
    for m in _open(project):
        if m.fruitless:
            await asyncio.to_thread(ledger.milestone_changed, project_id, m.id, fruitless=0)
    await asyncio.to_thread(ledger.revise, project_id, reason=f"going on as it is: {reason}", diff={"to_think_over": ""})
    return CameOf(got=1.0, what=f"she goes on with it as it is: {_short(reason, 200)}")


async def see_whether_it_is_so(project_id: str, thinks: Thinks, *, ledger: ProjectLedger | None = None,
                               bears_out: BearsOut | None = None) -> CameOf:
    """Every step taken: whether what she said will be so when it is done is so, by what the steps came to, read by her
    reader of words. Where it is, she says what came of it and it is done; where it is not, it is thought over."""
    ledger = ledger or get_ledger()
    bears_out = bears_out or _her_reader()
    project = ledger.get(project_id)
    if project is None:
        return CameOf(got=0.0, what="there was no such objective")
    came, done_when = _what_they_came_to(project), _done_when(project)
    read = await asyncio.to_thread(bears_out, came, done_when) if came and done_when else None
    if read is not None and read <= 0.0:
        why = f"its steps are taken and what they came to does not bear out that {done_when}"
        await asyncio.to_thread(ledger.revise, project_id, reason=why, diff={"to_think_over": why})
        return CameOf(got=0.0, what=f"its steps are taken and what they came to does not bear out “{_short(done_when)}”")
    looked = await thinks(json.dumps({"what_you_set_out_to_do": _as_she_reads_it(project)}, ensure_ascii=False), THE_QUESTION_AT_THE_END, None)
    if looked is None:
        return CameOf(put_down=True, what="her model could not be asked")
    words = looked[0].strip() or came

    def finish() -> Project:
        if not project.artifacts:
            ledger.attach_artifact(project_id, came)
        return ledger.mark_completed(project_id, reflection=words)

    try:
        await asyncio.to_thread(finish)
    except ValueError as exc:
        return CameOf(got=0.0, what=f"it could not be called done: {exc}")
    return CameOf(got=1.0, what=f"what she set out to do is done: {_short(project.thesis, 100)}; {_short(words, 200)}")


async def objectives_of_hers(hers: HerOwnTime, doers: Doers, thinks: Thinks, *, ledger: ProjectLedger | None = None,
                             bears_out: BearsOut | None = None) -> list[Option]:
    """What she could do of what she set out to do: its next steps, thinking it over, or seeing whether it is done; and where
    nothing stands, setting out on something, if there is something she wants to."""
    ledger = ledger or get_ledger()
    project = standing(ledger)
    want = {"competence": doers.asking("competence"), "curiosity": doers.asking("curiosity")}
    if project is None:
        ended = _ended(ledger)
        done = sum(p.lifecycle == Lifecycle.COMPLETED for p in ended)
        return [Option("objective:new", "objective", "set out to bring about something of her own over the coming days, if there is something she wants to",
                       f"{len(ended)} set out on before, {done} of them done", known={"set_out": len(ended) / (len(ended) + 2.0), "done": done / max(1, len(ended))},
                       drives=want, cost_s=doers.cost_s.get("objective", 240.0),
                       does=lambda: set_out(hers, thinks, ledger=ledger, bears_out=bears_out))]
    pid, of_it = project.project_id, f"what she set out to do, {_short(project.thesis, 80)}"
    finished = sum(1 for m in project.milestones if m.completed_at is not None)
    known = {"done": finished / max(1, len(project.milestones)), "revised": len(project.revisions) / (len(project.revisions) + 2.0)}
    why_now = _to_think_over(project)
    if why_now:
        return [Option(f"objective:{pid}:over", "objective", f"think over {of_it}", why_now, known=known, drives=want,
                       cost_s=doers.cost_s.get("objective", 240.0), does=lambda: go_on_or_not(pid, thinks, why_now, ledger=ledger, bears_out=bears_out))]
    if not _open(project):
        return [Option(f"objective:{pid}:end", "objective", f"see whether {of_it} is so", "every step of it is taken", known=known, drives=want,
                       cost_s=doers.cost_s.get("objective", 240.0), does=lambda: see_whether_it_is_so(pid, thinks, ledger=ledger, bears_out=bears_out))]
    out: list[Option] = []
    for milestone in (m for m in _ready(project) if m.kind in doers.does):
        stands, shown = await asyncio.to_thread(doers.stands, milestone.kind, milestone.key) if milestone.key else (GOING, "")
        if stands == DONE:
            # Done since it was last taken, by what takes it.
            await asyncio.to_thread(ledger.complete_milestone, pid, milestone.id, artifacts=[shown or "done"])
            continue
        if stands == WAITS:
            continue
        place = project.milestones.index(milestone) + 1
        out.append(Option(f"objective:{pid}:{milestone.id}", "objective", f"go on with {of_it}: {_short(milestone.description)}",
                          f"step {place} of {len(project.milestones)}, for {_short(project.origin_drive, 80) or 'itself'}",
                          known={**known, "tried": milestone.tries / (milestone.tries + 2.0)},
                          drives={(drive := "curiosity" if milestone.kind == "find_out" else "competence"): want[drive]},
                          cost_s=doers.cost_s.get(milestone.kind, 240.0),
                          does=lambda m=milestone.id: take_a_step(pid, m, doers, ledger=ledger, bears_out=bears_out)))
        if len(out) >= doers.most:
            break
    return out


def key_of_an_inquiry(what: str) -> str:
    """What an inquiry a step of hers began is known by: the same step said the same way is the same inquiry."""
    return "objective:" + hashlib.sha256(" ".join(what.split()).casefold().encode()).hexdigest()[:12]
