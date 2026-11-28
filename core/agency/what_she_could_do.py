"""The things she could do on her own time, each with what does it and how what came of it is seen.

Every one of these uses what she has already: the engine that carries out what people ask (core/understanding/carrying_out.py),
her record of what things have been worth to her (core/conation/what_she_wants.py), her goals at work
(core/agency/her_goal_at_work.py), her search, and her word in a conversation nobody asked for one in. Nothing here decides
which is done, or that any is.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.agency.her_inquiries import Answered, Inquiry, NotNowError
from core.agency.her_own_time import HERSELF, HerOwnTime
from core.agency.what_she_does_next import CameOf, Option

__all__ = ["OBJECTIVE", "WhatSheWouldMake", "act_of_a_goal", "begin_her_own_time", "doers_of_her_objectives", "goals_of_hers", "her_own_makes", "left_unmet", "put_away",
           "things_to_find_out", "things_to_make", "things_to_tell"]

#: A goal that asks for something to be known or found is looked up; one that asks for something to be built or changed is
#: about her, and is worked in her own source. A goal that asks for nothing to be brought about has nothing that does it.
LOOKED_UP: frozenset[str] = frozenset({"known", "found"})
IN_HER_SOURCE: frozenset[str] = frozenset({"program", "program_changed"})
#: What a goal asks for (core/understanding/effects.py): "" where her model is not there to read it.
Asks = Callable[[str], Awaitable[str]]
#: One pass at an inquiry of hers, which says what pages it read (core/agency/her_inquiries.py).
LooksInto = Callable[[Inquiry, Callable[[Any], Any]], Awaitable[Answered | None]]

#: How many of a sort are offered at once: what is wanted most of each.
OF_A_SORT = 4
#: How long a request of somebody's that came to less than was asked is still gone on with unasked, in seconds.
STILL_WANTED_S = 2 * 86400.0


#: Where her own time is joined to what does each act: how her pages are found and read, her model asked, and her words said.
_JOINED = "interface/routes/her_own_time.py"
#: What does each sort of act, by its source: where that changes, what came of it before was done another way. LIVE 27
#: November 2026 her inquiries read nothing for days for want of a change in where her own time is joined to her search,
#: and the change made there was not a change in how she finds things out.
DOES: dict[str, tuple[str, ...]] = {
    "carry_on": ("core/understanding/carriers/*.py", "core/making/*.py", "core/making/*.js"),
    "find_out": ("core/agency/her_inquiries.py", _JOINED), "look_up": ("core/agency/her_inquiries.py", _JOINED),
    "goal": ("core/agency/her_goal_at_work.py",), "tell": ("core/agency/what_she_could_do.py", _JOINED),
    "objective": ("core/agency/her_objectives.py", "core/agency/her_inquiries.py", "core/agency/her_goal_at_work.py",
                  "core/understanding/carriers/building.py", _JOINED),
    "make": ("core/understanding/carriers/building.py", "core/making/*.py", "core/making/*.js", _JOINED),
}


@lru_cache(maxsize=16)
def way_of(kind: str) -> str:
    """How a sort of act is done now, as a mark of the source that does it; "" where that cannot be read."""
    root, read = Path(__file__).resolve().parents[2], hashlib.sha256()
    files = sorted(path for pattern in DOES.get(kind, ()) for path in root.glob(pattern))
    try:
        for path in files:
            read.update(path.read_bytes())
    # not a failure: with no source to read beside her, every outcome counts for as much as it did.
    except OSError:
        return ""
    return read.hexdigest()[:16] if files else ""


def _asking(drive: str) -> float:
    """How much a drive is asking now, 0 to 1; nothing where it cannot be read."""
    from core.container import ServiceContainer

    engine = ServiceContainer.get("drive_integration", default=None)
    held = getattr(engine, "drives", None)
    one = held.get(drive) if isinstance(held, dict) else None
    level = getattr(one, "activation", getattr(one, "level", 0.0)) if one is not None else 0.0
    return max(0.0, min(1.0, float(level or 0.0)))


def left_unmet(carry_on: Callable[[Any], Awaitable[None]]) -> list[Option]:
    """Requests of people's that came to less than they asked, gone on with."""
    from core.understanding import carrying_out
    from core.understanding.requests_of_her import CUT_SHORT, PARTLY, SET_ASIDE, UNMET, get_requests_of_her

    out = []
    for request in reversed(get_requests_of_her().all()):
        if request.stands not in {PARTLY, UNMET, CUT_SHORT} or request.conversation == HERSELF or request.effect not in carrying_out.carriers() or carrying_out.is_under_way(request):
            continue
        age = time.time() - (request.changed_at or request.opened_at)
        if age > STILL_WANTED_S:
            continue
        missing, met = len(request.missing()), sum(1 for p in request.parts if p.stands == "met")

        async def does(request: Any = request, met: int = met, missing: int = missing) -> CameOf:
            held = await asyncio.to_thread(_how_much_holds, request)
            await carry_on(request)
            # Stopped by somebody, or by her being closed, it has come to nothing yet, which is not having come to nothing.
            if request.stands in {CUT_SHORT, SET_ASIDE}:
                return CameOf(put_down=True, what="it was put down before it came to anything")
            now, holds = sum(1 for p in request.parts if p.stands == "met"), await asyncio.to_thread(_how_much_holds, request)
            further = 0.0 if held is None or holds is None or held >= 1.0 else (holds - held) / (1.0 - held)
            return CameOf(got=max(0.0, min(1.0, max((now - met) / max(1, missing), further))),
                          what=f"{now} of {len(request.parts)} things asked of it are so, and {met} were" + (f"; {holds:.0%} of what it is tried by holds, and {held:.0%} did" if held is not None and holds is not None else ""))

        out.append(Option(f"carry_on:{request.id}", "carry_on", f"go on with what was asked: {_short(request.said)}", f"{missing} of what they asked is not so yet",
                          known={"waiting": 1.0 - math.exp(-age / 3600.0), "met": met / max(1, met + missing), "tried": len(request.tries) / (len(request.tries) + 2.0)},
                          drives={"integrity": _asking("integrity")}, cost_s=900.0, does=does))
        if len(out) >= OF_A_SORT:
            break
    return out


def _how_much_holds(request: Any) -> float | None:
    """How much of what a program of somebody's is tried by holds as it was last kept; None where it is no program, or none is
    kept. What is asked of it is said to be so or not once it is done; this is how far it has come meanwhile."""
    page = str(request.made.get("page") or "")
    try:
        tried = json.loads((Path(page).parent / "program.json").read_text(encoding="utf-8")).get("tried") if page else None
    # not a failure: with nothing kept beside it there is nothing to measure it by.
    except (OSError, ValueError, AttributeError):
        return None
    return sum(1 for one in tried if one.get("ok")) / len(tried) if tried else None


async def _look_into(hers: HerOwnTime, looks: LooksInto, of: str, about: str) -> tuple[CameOf, Inquiry]:
    """One pass at finding something out: its next question, taken to the pages it finds. What it came to is whether one of them
    answers it, in its own words, and a little for what she had not read before."""
    inquiry = hers.inquiry(of, about)
    new: list[str] = []
    try:
        one = await looks(inquiry, lambda urls: new.extend(hers.is_new(urls)))
    except NotNowError as away:
        return CameOf(put_down=True, what=f"it could not be gone on with now: {away}"), inquiry
    if one is None:
        return CameOf(got=0.0, what=f"there was nothing to ask about {about}"), inquiry
    question = one.question
    if question.answer:
        hers.finds(question.question, question.answer, question.where)
    unread = 0.1 * len(new) / len(one.read) if one.read else 0.0
    came = f"asked “{_short(question.question)}”, and {'it is answered' if question.stands == 'answered' else 'it is answered in part' if question.answer else 'nothing read answers it'}"
    return CameOf(got=min(1.0, max(one.got, unread)), what=f"{came}; {len(one.read)} pages read"), inquiry


def things_to_find_out(hers: HerOwnTime, looks: LooksInto) -> list[Option]:
    """What she wants to know more of, looked into: each is a thing her own turns have shown is worth going to."""
    from core.conation.what_she_wants import FIND_OUT, get_what_she_wants

    record, out = get_what_she_wants(), []
    for want in record.wants(limit=OF_A_SORT):
        if want.way == "play":
            continue

        async def does(want: Any = want) -> CameOf:
            await asyncio.to_thread(record.started, want.thing, "act", label=want.label)
            try:
                came, _inquiry = await _look_into(hers, looks, f"want:{want.thing}", want.label)
            finally:
                await asyncio.to_thread(record.stopped, "act", want.thing)
            if came.put_down:
                return came
            await asyncio.to_thread(lambda: record.went(want.thing, FIND_OUT, label=want.label))
            if came.got <= 0.0:
                await asyncio.to_thread(record.failed, want.thing)
            return came

        held = hers.inquiries.get(f"want:{want.thing}")
        open_one = held.next() if held is not None else None
        out.append(Option(f"find_out:{want.thing}", "find_out", f"find out more about {want.label}" + (f": {_short(open_one.question)}" if open_one is not None else ""), _short(want.why),
                          known={"wanted": max(0.0, min(1.0, float(want.utility))), "new": 1.0 / (1.0 + want.contacts), "unfinished": 1.0 if want.unfinished else 0.0,
                                 "answered": held.met if held is not None else 0.0},
                          drives={"curiosity": _asking("curiosity")}, cost_s=90.0, does=does))
    return out


def things_to_tell(hers: HerOwnTime, tell: Callable[[dict[str, Any]], Awaitable[bool]], here: Callable[[], float]) -> list[Option]:
    """What she found on her own that nobody has been told, told to whoever is here. ``here`` is how near somebody is to
    talking with her, 0 to 1."""
    out = []
    for find in reversed(hers.found[-OF_A_SORT:]):

        async def does(find: dict[str, Any] = find) -> CameOf:
            said = await tell(find)
            if not said:
                return CameOf(put_down=True, what="it was not a moment to say it")
            hers.has_told(find)
            # Whether it was worth saying is for whoever heard it: what came of it so far is that it was said.
            return CameOf(got=0.5, what=f"told them about {find['about']}")

        age = time.time() - float(find.get("at") or 0.0)
        out.append(Option(f"tell:{find['about']}", "tell", f"tell them what she found about {find['about']}", _short(find["what"], 160),
                          known={"here": here(), "fresh": math.exp(-age / 21600.0)}, drives={"social": _asking("social")}, cost_s=30.0, does=does))
    return out


# ── what she makes for herself ───────────────────────────────────────

#: What a thing she makes for herself is asked as: a program, made as anybody's is (core/understanding/carriers/building.py).
HER_PROGRAM = "program"
#: Where a thing she made and let go of is kept: beside what she makes, put away and not destroyed.
PUT_AWAY = "put_away"


class WhatSheWouldMake(BaseModel):
    """Whether there is something she wants made now, what, and what of hers she no longer wants kept."""

    wants_to: bool = Field(description="Whether there is something you want to make now.")
    what: str = Field(default="", description="What to make, said the way you would ask somebody to make it.")
    why: str = Field(default="", description="What it is for: a use you see for it, something to practise, or an experiment.")
    let_go: list[str] = Field(default_factory=list, description="The ids of things you made for yourself that you no longer want kept.")


#: Her model's reading of what she would make, from what she has to go on; None where her model is not there to read it.
Imagines = Callable[[str], Awaitable[WhatSheWouldMake | None]]


def her_own_makes() -> list[Any]:
    """The things she set out to make for herself, the oldest first; those she let go of are not among them."""
    from core.understanding.requests_of_her import get_requests_of_her

    return [r for r in get_requests_of_her().all() if r.conversation == HERSELF and r.effect == HER_PROGRAM and not r.made.get(PUT_AWAY)]


def what_she_has_to_go_on(hers: HerOwnTime) -> str:
    """What she wants now, what she found on her own, and what she has made for herself, as facts."""
    from core.conation.what_she_wants import get_what_she_wants

    try:
        wants = [{"thing": w.label, "why": _short(w.why, 200)} for w in get_what_she_wants().wants(limit=6)]
    except _READ_FAILS:
        wants = []
    made = [{"id": r.id, "asked": _short(r.said, 300), "stands": r.stands, "of_what_it_is_tried_by_holds": _how_much_holds(r),
             "begun": time.strftime("%Y-%m-%d", time.localtime(r.opened_at))} for r in her_own_makes()[-8:]]
    return json.dumps({"what_you_want_now": wants, "what_you_found_on_your_own": [f["about"] for f in hers.found[-6:]],
                       "what_you_have_made_for_yourself": made}, ensure_ascii=False)


#: What a record of hers that cannot be read throws.
_READ_FAILS = (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)


async def _made(request: Any, make: Callable[[Any], Awaitable[None]], *, began: bool = False) -> CameOf:
    """Carry out a thing she set out to make; what came of it is how much more of what it is tried by holds, or, begun now,
    how much holds."""
    from core.understanding.requests_of_her import CUT_SHORT, SET_ASIDE

    held = None if began else await asyncio.to_thread(_how_much_holds, request)
    await make(request)
    if request.stands in {CUT_SHORT, SET_ASIDE}:
        return CameOf(put_down=True, what="it was put down before it came to anything")
    holds = await asyncio.to_thread(_how_much_holds, request)
    met = sum(1 for p in request.parts if p.stands == "met")
    so = met / max(1, len(request.parts))
    got = so if holds is None else holds if held is None or held >= 1.0 else max(0.0, (holds - held) / (1.0 - held))
    return CameOf(got=max(0.0, min(1.0, got)), what=f"“{_short(request.said, 80)}”: {met} of {len(request.parts)} things asked of it are so"
                  + (f"; {holds:.0%} of what it is tried by holds" + (f", and {held:.0%} did" if held is not None else "") if holds is not None else ""))


def put_away(request: Any) -> str:
    """Put away what she made for herself and no longer wants kept: moved beside the others where it can be had back, not
    deleted. Where it went; "" where it was not a folder of hers to move."""
    page = str(request.made.get("page") or "")
    folder = Path(page).parent if page else None
    if request.conversation != HERSELF or folder is None or not folder.is_dir():
        return ""
    from core.runtime.her_writes import governed

    to = folder.parent / PUT_AWAY / folder.name
    governed("ensure_directory", to.parent, source="her_own_time.put_away")
    governed("move_path", folder, to, source="her_own_time.put_away")
    request.made[PUT_AWAY] = str(to)
    return str(to)


async def _make_something(hers: HerOwnTime, make: Callable[[Any], Awaitable[None]], imagines: Imagines) -> CameOf:
    """Ask herself whether there is something she wants made, and make it; and put away what of hers she no longer wants."""
    from core.understanding.requests_of_her import get_requests_of_her

    read = await imagines(await asyncio.to_thread(what_she_has_to_go_on, hers))
    if read is None:
        return CameOf(put_down=True, what="her model could not be asked")
    gone = [r for r in her_own_makes() if r.id in set(read.let_go)]
    put = [r.said for r in gone if await asyncio.to_thread(put_away, r)]
    ledger = get_requests_of_her()
    let = f"; she put away {len(put)} thing(s) she had made: " + "; ".join(_short(said, 60) for said in put) if put else ""
    if not read.wants_to or not read.what.strip():
        await ledger.keep()
        return CameOf(got=0.0, what="there was nothing she wanted to make" + let)
    request = ledger.asked(read.what.strip(), effect=HER_PROGRAM, conversation=HERSELF)
    if read.why.strip():
        request.added.append(read.why.strip())
    await ledger.keep()
    came = await _made(request, make, began=True)
    return CameOf(came.got, came.worth, came.put_down, came.what + let, came.seconds)


def things_to_make(hers: HerOwnTime, make: Callable[[Any], Awaitable[None]], imagines: Imagines) -> list[Option]:
    """Making something of her own, if there is something she wants made; and going on with what she set out to make and did
    not finish. Whether she makes anything is hers: asked, she may say there is nothing."""
    from core.understanding import carrying_out
    from core.understanding.requests_of_her import CUT_SHORT, PARTLY, UNMET

    own = her_own_makes()
    if any(carrying_out.is_under_way(r) for r in own):
        return []
    out = []
    for request in reversed(own):
        # What she makes as a step of what she set out to do is gone on with as that step (core/agency/her_objectives.py).
        if request.made.get(OBJECTIVE):
            continue
        if request.stands not in {PARTLY, UNMET, CUT_SHORT} or time.time() - (request.changed_at or request.opened_at) > STILL_WANTED_S:
            continue
        met = sum(1 for p in request.parts if p.stands == "met")
        out.append(Option(f"make:{request.id}", "make", f"go on making what she set out to make: {_short(request.said)}", f"{len(request.missing())} of it is not so yet",
                          known={"met": met / max(1, len(request.parts)), "tried": len(request.tries) / (len(request.tries) + 2.0)},
                          drives={"competence": _asking("competence")}, cost_s=900.0, does=lambda request=request: _made(request, make)))
        if len(out) >= OF_A_SORT - 1:
            break
    held = [h for h in (_how_much_holds(r) for r in own) if h is not None]
    out.append(Option("make:new", "make", "make something of her own, if there is something she wants made", f"{len(own)} made for herself before",
                      known={"made": len(own) / (len(own) + 2.0), "held": sum(held) / len(held) if held else 0.0},
                      drives={"competence": _asking("competence"), "curiosity": _asking("curiosity")}, cost_s=1200.0,
                      does=lambda: _make_something(hers, make, imagines)))
    return out


async def _hers_to_do() -> list[tuple[str, Any, str]]:
    """Her goals that are under way: each with what she did of it in her own source, and its record in the goal store."""
    from core.agency import her_goal_at_work as goals
    from core.container import ServiceContainer

    worked = {r.slug: r for r in await asyncio.to_thread(goals.all_records)}
    out = {r.slug: (r.goal, r, "") for r in sorted(worked.values(), key=lambda r: -r.last_worked_at) if r.status in {goals.WORKING, goals.STALLED}}
    engine = ServiceContainer.get("goal_engine", default=None)
    if engine is not None:
        held = await engine.get_active_goals_async(limit=30, include_external=False, actionable_only=True)
        stands = {goals.slug_of(str(g.get("objective") or g.get("name") or "")): g for g in held}
        # A goal let go of in the store is not under way because some of it was worked.
        out = {slug: (goal, record, str(stands[slug].get("id") or "")) for slug, (goal, record, _id) in out.items() if slug in stands}
        for slug, goal in stands.items():
            done = worked.get(slug)
            if slug not in out and goal.get("source") != "boot_default" and (done is None or done.status in {goals.WORKING, goals.STALLED}):
                out[slug] = (str(goal.get("objective") or goal.get("name")), done, str(goal.get("id") or ""))
    return list(out.values())


async def act_of_a_goal(goal: str, *, hers: HerOwnTime, asks: Asks, looks: LooksInto, record: Any = None, goal_id: str = "") -> Option | None:
    """What would do a goal of hers, by what it asks to be brought about; None where nothing would.

    LIVE 24 November 2026 "Research the history of quantum error correction." had been her goal for twelve days, chosen again
    and again, and every pass at it offered her the tools that edit her own source.
    """
    from core.agency import her_goal_at_work as goals

    slug, asked = goals.slug_of(goal), await asks(goal)
    if asked in LOOKED_UP:

        async def looked() -> CameOf:
            came, inquiry = await _look_into(hers, looks, f"goal:{slug}", goal)
            # It is done when what would have to be found out for it has been, and is further on by each that is.
            if goal_id and inquiry.questions:
                await _is_so_far(goal_id, inquiry)
            return came

        held = hers.inquiries.get(f"goal:{slug}")
        return Option(f"look_up:{slug}", "look_up", f"look into what she set herself: {_short(goal)}", "a goal of hers that asks for something to be known",
                      known={"answered": held.met if held is not None else 0.0},
                      drives={"curiosity": _asking("curiosity")}, cost_s=90.0, does=looked)
    # One she has begun in her own source is gone on with there while her model is not there to read it.
    if asked not in IN_HER_SOURCE and not (asked == "" and record is not None and record.steps):
        return None
    record = record or goals.GoalRecord(goal=goal, slug=slug)

    async def works(steps: int = len(record.steps)) -> CameOf:
        after = await goals.work_a_while(goal)
        done = [s for s in after.steps[steps:] if "error" not in str(s.get("result") or "")[:80].lower()]
        got = 1.0 if after.status not in {goals.WORKING, goals.STALLED} else len(done) / max(1, goals.CALLS_A_PASS)
        return CameOf(got=min(1.0, got), what=f"{len(done)} steps further; it is {after.status}")

    return Option(f"goal:{slug}", "goal", f"work on her own goal: {_short(goal)}", f"{record.passes} passes at it, {len(record.steps)} steps",
                  known={"tried": record.passes / (record.passes + 4.0), "quiet": record.quiet_passes / (record.quiet_passes + 1.0), "stalled": float(record.status == goals.STALLED)},
                  drives={"competence": _asking("competence")}, cost_s=240.0, does=works)


async def _is_so_far(goal_id: str, inquiry: Inquiry) -> None:
    from core.container import ServiceContainer

    engine = ServiceContainer.get("goal_engine", default=None)
    if engine is None:
        return
    told = " ".join(f"{q.question} {q.answer}" for q in inquiry.questions if q.answer)
    await engine.update_goal_status(goal_id, status="completed" if inquiry.is_done else "in_progress", summary=_short(told, 600),
                                    progress=round(1.0 if inquiry.is_done else inquiry.met, 3), evidence=[q.where for q in inquiry.questions if q.where][:12])


async def goals_of_hers(hers: HerOwnTime, asks: Asks, looks: LooksInto) -> list[Option]:
    """Goals she set herself, each a pass further by whatever does what it asks."""
    out: list[Option] = []
    for goal, record, goal_id in await _hers_to_do():
        option = await act_of_a_goal(goal, hers=hers, asks=asks, looks=looks, record=record, goal_id=goal_id)
        if option is not None:
            out.append(option)
        if len(out) >= OF_A_SORT:
            break
    return out


def _short(words: str, most: int = 120) -> str:
    words = " ".join(str(words).split())
    return words if len(words) <= most else words[:most].rsplit(" ", 1)[0] + "…"


#: What marks a thing she makes as a step of what she set out to do.
OBJECTIVE = "objective"


def doers_of_her_objectives(hers: HerOwnTime, looks: LooksInto, make: Callable[[Any], Awaitable[None]] | None) -> Any:
    """What takes each sort of step of what she set out to do (core/agency/her_objectives.py): the same inquiry, making and
    goal of hers that take each sort of act on her own time, and what each says of how a step stands."""
    from core.agency import her_goal_at_work as goals
    from core.agency import her_objectives as objectives
    from core.understanding.requests_of_her import ASKED, MET, get_requests_of_her

    async def find_out(key: str, what: str, given: list[str]) -> tuple[str, CameOf]:
        key = key or objectives.key_of_an_inquiry(what)
        before = {q.question for q in hers.inquiry(key, what).questions if q.answer}
        came, inquiry = await _look_into(hers, looks, key, what)
        # What was found, and not only that something was: what her reader holds against what she said would end it.
        found = [f"{q.question} {q.answer}" for q in inquiry.questions if q.answer and q.question not in before]
        return key, dataclasses.replace(came, what=came.what + (": " + _short(" ".join(found), 600) if found else ""))

    async def made(key: str, what: str, given: list[str]) -> tuple[str, CameOf]:
        ledger = get_requests_of_her()
        request = ledger.get(key) if key else None
        if request is None:
            request = ledger.asked(what, effect=HER_PROGRAM, conversation=HERSELF)
            request.made[OBJECTIVE] = True
            request.added.extend(given)
            await ledger.keep()
            return request.id, await _made(request, make, began=True)
        return key, await _made(request, make)

    async def change(key: str, what: str, given: list[str]) -> tuple[str, CameOf]:
        held = await asyncio.to_thread(goals.load, key) if key else None
        steps = len(held.steps) if held is not None else 0
        after = await goals.work_a_while(what)
        done = [s for s in after.steps[steps:] if "error" not in str(s.get("result") or "")[:80].lower()]
        got = 1.0 if after.status not in {goals.WORKING, goals.STALLED} else len(done) / max(1, goals.CALLS_A_PASS)
        said = str(after.came_of_it.get("words") or "")
        return after.slug, CameOf(got=min(1.0, got), what=f"{len(done)} steps further; it is {after.status}" + (f": {_short(said, 300)}" if said else ""))

    def stands(kind: str, key: str) -> tuple[str, str]:
        if kind == "find_out":
            inquiry = hers.inquiries.get(key)
            if inquiry is not None and inquiry.is_done:
                return objectives.DONE, "; ".join(dict.fromkeys(q.where for q in inquiry.questions if q.where))[:600]
            return objectives.GOING, ""
        if kind == "make":
            request = get_requests_of_her().get(key)
            if request is None:
                return objectives.GOING, ""
            shown = str(request.made.get("page") or request.id)
            return (objectives.DONE, shown) if request.stands == MET else (objectives.WAITS, "") if request.stands == ASKED else (objectives.GOING, "")
        record = goals.load(key)
        if record is None:
            return objectives.GOING, ""
        if record.status == goals.MERGED:
            return objectives.DONE, f"commit {record.offer.get('commit') or ''}".strip()
        if record.status == goals.DONE:
            return objectives.DONE, str(record.came_of_it.get("words") or "ended with nothing to change")
        # An offered draft waits on her own yes and its gates, or on Bryan's word.
        return (objectives.WAITS, "") if record.status == goals.OFFERED else (objectives.GOING, "")

    does = {"find_out": find_out, "change": change, **({"make": made} if make is not None else {})}
    return objectives.Doers(does=does, stands=stands, asking=_asking, most=OF_A_SORT,
                            cost_s={"find_out": 90.0, "make": 900.0, "change": 240.0, "objective": 240.0})


async def _nothing_reads(_goal: str) -> str:
    return ""


def begin_her_own_time(hers: HerOwnTime, *, carry_on: Callable[[Any], Awaitable[None]], looks: LooksInto,
                       tell: Callable[[dict[str, Any]], Awaitable[bool]], here: Callable[[], float], asks: Asks = _nothing_reads,
                       make: Callable[[Any], Awaitable[None]] | None = None, imagines: Imagines | None = None,
                       thinks: Any = None) -> HerOwnTime:
    """Give her own time what she could do with it. ``make`` carries out a thing she set out to make for herself, and
    ``imagines`` reads whether there is one; without both she makes nothing of her own. ``thinks`` is her cortex asked
    whether there is something she wants to bring about over days (core/agency/her_objectives.py); without it she sets out
    on nothing."""
    def as_done_now(options: list[Option]) -> list[Option]:
        return [dataclasses.replace(option, way=way_of(option.kind)) for option in options]

    async def goals() -> list[Option]:
        return as_done_now(await goals_of_hers(hers, asks, looks))

    hers.offers("requests", lambda: asyncio.to_thread(lambda: as_done_now(left_unmet(carry_on))))
    hers.offers("wants", lambda: asyncio.to_thread(lambda: as_done_now(things_to_find_out(hers, looks))))
    hers.offers("finds", lambda: as_done_now(things_to_tell(hers, tell, here)))
    hers.offers("goals", goals)
    if make is not None and imagines is not None:
        hers.offers("makes", lambda: asyncio.to_thread(lambda: as_done_now(things_to_make(hers, make, imagines))))
    if thinks is not None:
        from core.agency.her_objectives import objectives_of_hers

        doers = doers_of_her_objectives(hers, looks, make)

        async def objectives() -> list[Option]:
            return as_done_now(await objectives_of_hers(hers, doers, thinks))

        hers.offers("objectives", objectives)
    hers.act_of = lambda goal: act_of_a_goal(goal, hers=hers, asks=asks, looks=looks)
    return hers
