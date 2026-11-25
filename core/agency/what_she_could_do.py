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

from core.agency.her_own_time import HERSELF, HerOwnTime
from core.agency.what_she_does_next import CameOf, Option

__all__ = ["begin_her_own_time", "goals_of_hers", "left_unmet", "things_to_find_out", "things_to_tell"]

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


def things_to_find_out(hers: HerOwnTime, search: Callable[[str], Awaitable[dict[str, Any]]]) -> list[Option]:
    """What she wants to know more of, looked into: each is a thing her own turns have shown is worth going to."""
    from core.conation.what_she_wants import FIND_OUT, get_what_she_wants

    record, out = get_what_she_wants(), []
    for want in record.wants(limit=OF_A_SORT):
        if want.way == "play":
            continue

        async def does(want: Any = want) -> CameOf:
            await asyncio.to_thread(record.started, want.thing, "act", label=want.label)
            try:
                result = await search(want.label)
            finally:
                await asyncio.to_thread(record.stopped, "act", want.thing)
            await asyncio.to_thread(lambda: record.went(want.thing, FIND_OUT, label=want.label))
            if not result.get("ok"):
                await asyncio.to_thread(record.failed, want.thing)
                return CameOf(got=0.0, what=f"the search gave nothing: {str(result.get('error') or '')[:120]}")
            pages = [(str(c.get("title") or ""), str(c.get("url") or "")) for c in _citations(result)]
            new = hers.is_new(url for _title, url in pages)
            told = _short(str(result.get("summary") or result.get("answer") or result.get("content") or ""), 400)
            if new and told:
                title, url = next((p for p in pages if p[1] == new[0]), ("", new[0]))
                hers.finds(want.label, f"{title}: {told}" if title else told, url)
            return CameOf(got=len(new) / max(1, len(pages)) if pages else 0.0, what=f"read {len(new)} pages she had not, of {len(pages)}, about {want.label}")

        out.append(Option(f"find_out:{want.thing}", "find_out", f"find out more about {want.label}", _short(want.why),
                          known={"wanted": max(0.0, min(1.0, float(want.utility))), "new": 1.0 / (1.0 + want.contacts), "unfinished": 1.0 if want.unfinished else 0.0},
                          drives={"curiosity": _asking("curiosity")}, cost_s=60.0, does=does))
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


async def goals_of_hers() -> list[Option]:
    """Goals she set herself that are being worked, each a pass further."""
    from core.agency import her_goal_at_work as goals

    out = []
    for record in sorted(await asyncio.to_thread(goals.all_records), key=lambda r: -r.last_worked_at):
        if record.status not in {goals.WORKING, goals.STALLED}:
            continue

        async def does(goal: str = record.goal, steps: int = len(record.steps)) -> CameOf:
            after = await goals.work_a_while(goal)
            done = [s for s in after.steps[steps:] if "error" not in str(s.get("result") or "")[:80].lower()]
            got = 1.0 if after.status not in {goals.WORKING, goals.STALLED} else len(done) / max(1, goals.CALLS_A_PASS)
            return CameOf(got=min(1.0, got), what=f"{len(done)} steps further; it is {after.status}")

        out.append(Option(f"goal:{record.slug}", "goal", f"work on her own goal: {_short(record.goal)}", f"{record.passes} passes at it, {len(record.steps)} steps",
                          known={"tried": record.passes / (record.passes + 4.0), "quiet": record.quiet_passes / (record.quiet_passes + 1.0), "stalled": float(record.status == goals.STALLED)},
                          drives={"competence": _asking("competence")}, cost_s=240.0, does=does))
        if len(out) >= OF_A_SORT:
            break
    return out


def _citations(result: dict[str, Any]) -> list[dict[str, Any]]:
    for name in ("citations", "results", "sources"):
        held = result.get(name)
        if isinstance(held, list) and held:
            return [c for c in held if isinstance(c, dict)]
    return []


def _short(words: str, most: int = 120) -> str:
    words = " ".join(str(words).split())
    return words if len(words) <= most else words[:most].rsplit(" ", 1)[0] + "…"


def begin_her_own_time(hers: HerOwnTime, *, carry_on: Callable[[Any], Awaitable[None]], search: Callable[[str], Awaitable[dict[str, Any]]],
                       tell: Callable[[dict[str, Any]], Awaitable[bool]], here: Callable[[], float]) -> HerOwnTime:
    """Give her own time what she could do with it."""
    hers.offers("requests", lambda: asyncio.to_thread(left_unmet, carry_on))
    hers.offers("wants", lambda: asyncio.to_thread(things_to_find_out, hers, search))
    hers.offers("finds", lambda: things_to_tell(hers, tell, here))
    hers.offers("goals", goals_of_hers)
    return hers
