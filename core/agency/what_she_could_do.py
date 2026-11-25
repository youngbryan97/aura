"""The things she could do on her own time, each with what does it and how what came of it is seen.

Every one of these uses what she has already: the engine that carries out what people ask (core/understanding/carrying_out.py),
her record of what things have been worth to her (core/conation/what_she_wants.py), her goals at work
(core/agency/her_goal_at_work.py), her search, and her word in a conversation nobody asked for one in. Nothing here decides
which is done, or that any is.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.agency.her_inquiries import Answered, Inquiry, NotNowError
from core.agency.her_own_time import HERSELF, HerOwnTime
from core.agency.what_she_does_next import CameOf, Option

__all__ = ["act_of_a_goal", "begin_her_own_time", "goals_of_hers", "left_unmet", "things_to_find_out", "things_to_tell"]

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
    from core.understanding.requests_of_her import CUT_SHORT, PARTLY, UNMET, get_requests_of_her

    out = []
    for request in reversed(get_requests_of_her().all()):
        if request.stands not in {PARTLY, UNMET, CUT_SHORT} or request.conversation == HERSELF or request.effect not in carrying_out.carriers() or carrying_out.is_under_way(request):
            continue
        age = time.time() - (request.changed_at or request.opened_at)
        if age > STILL_WANTED_S:
            continue
        missing, met = len(request.missing()), sum(1 for p in request.parts if p.stands == "met")

        async def does(request: Any = request, met: int = met, missing: int = missing) -> CameOf:
            await carry_on(request)
            now = sum(1 for p in request.parts if p.stands == "met")
            return CameOf(got=max(0.0, min(1.0, (now - met) / max(1, missing))), what=f"{now} of {len(request.parts)} things asked of it are so, and {met} were")

        out.append(Option(f"carry_on:{request.id}", "carry_on", f"go on with what was asked: {_short(request.said)}", f"{missing} of what they asked is not so yet",
                          known={"waiting": 1.0 - math.exp(-age / 3600.0), "met": met / max(1, met + missing), "tried": len(request.tries) / (len(request.tries) + 2.0)},
                          drives={"integrity": _asking("integrity")}, cost_s=900.0, does=does))
        if len(out) >= OF_A_SORT:
            break
    return out


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


async def _nothing_reads(_goal: str) -> str:
    return ""


def begin_her_own_time(hers: HerOwnTime, *, carry_on: Callable[[Any], Awaitable[None]], looks: LooksInto,
                       tell: Callable[[dict[str, Any]], Awaitable[bool]], here: Callable[[], float], asks: Asks = _nothing_reads) -> HerOwnTime:
    """Give her own time what she could do with it."""
    hers.offers("requests", lambda: asyncio.to_thread(left_unmet, carry_on))
    hers.offers("wants", lambda: asyncio.to_thread(things_to_find_out, hers, looks))
    hers.offers("finds", lambda: things_to_tell(hers, tell, here))
    hers.offers("goals", lambda: goals_of_hers(hers, asks, looks))
    hers.act_of = lambda goal: act_of_a_goal(goal, hers=hers, asks=asks, looks=looks)
    return hers
