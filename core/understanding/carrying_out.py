"""Carrying a request out: tried, checked, tried again for what is missing, and told as it came out.

A carrier does one kind of thing (core/understanding/carriers) and answers with the parts the request came to and whether
each is so, as the world shows it. What stands is decided from those parts here and never from a carrier's returning: met
when every part is, partly when some are, unmet when none is. While a try leaves fewer parts missing than the one before
it, another is made, handed what is missing and why. When a try gains nothing the work stops and every missing part is
named with its reason.

It runs on her own time after the turn that asked is answered, so the conversation goes on beside it, and a person's turn
comes before any part of it. What it comes to is told in the conversation when it is known.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.understanding.her_model_between_turns import nobody_waiting
from core.understanding.requests_of_her import (
    CUT_SHORT,
    MET,
    OPEN,
    PARTLY,
    SET_ASIDE,
    UNMET,
    WORKING,
    Part,
    Request,
    RequestsOfHer,
    Try,
    get_requests_of_her,
)
from core.understanding.workbench import PutDown, get_workbench, working_on

logger = logging.getLogger("Aura.Understanding")

__all__ = ["CameTo", "Carrier", "begin", "carriers", "carry_out", "is_under_way"]


@dataclass
class CameTo:
    """What one try came to."""

    how: str
    parts: list[Part] = field(default_factory=list)
    made: dict[str, Any] = field(default_factory=dict)
    came_to: str = ""
    #: Whether trying again could change anything.
    again: bool = True


Tell = Callable[[str], Any]
Carrier = Callable[[Request, Tell], Awaitable[CameTo]]


def carriers() -> dict[str, Carrier]:
    """What does each kind of thing. A kind not here is done in the turn, by the turn."""
    from core.understanding.carriers.building import build, change
    from core.understanding.carriers.finding import find
    from core.understanding.carriers.reading_whole import read_whole

    return {"program": build, "program_changed": change, "found": find, "document_read": read_whole}


_UNDER_WAY: dict[str, asyncio.Task[Any]] = {}


def is_under_way(request: Request) -> bool:
    task = _UNDER_WAY.get(request.id)
    return task is not None and not task.done()


def stands_by(parts: list[Part]) -> str:
    """How a request stands, by its parts."""
    if not parts:
        return UNMET
    met = sum(1 for part in parts if part.stands == "met")
    return MET if met == len(parts) else PARTLY if met else UNMET


async def carry_out(request: Request, carrier: Carrier, *, tell: Tell, ledger: RequestsOfHer | None = None) -> Request:
    """Try, check, and try again for what is missing while that gains anything; what it came to is on the request."""
    ledger = ledger or get_requests_of_her()
    missing_before: int | None = None
    while True:
        await ledger.changed(request, stands=WORKING, now="")
        began = time.time()
        try:
            came = await carrier(request, tell)
        except asyncio.CancelledError:
            request.tries.append(Try(at=began, how="it was stopped", came_to="cut short"))
            await ledger.changed(request, stands=CUT_SHORT)
            raise
        except PutDown:
            request.tries.append(Try(at=began, how="it was stopped by them", came_to="set aside"))
            await ledger.changed(request, stands=SET_ASIDE)
            return request
        except Exception as exc:  # noqa: BLE001 - a try that breaks is a try that failed, and is named as one
            record_degradation("carrying_out", exc, severity="warning", action=f"a try at request {request.id} broke off")
            came = CameTo(how="it broke off", came_to=f"{type(exc).__name__}: {exc}"[:400],
                          parts=[Part(part.what, part.stands, part.shown_by) for part in request.parts]
                          or [Part(_the_whole(request), "unmet", f"it broke off: {type(exc).__name__}: {exc}"[:300])])
        request.parts = came.parts or request.parts or [Part(_the_whole(request), "unmet", came.came_to)]
        request.made.update({key: value for key, value in came.made.items() if value})
        missing = request.missing()
        request.tries.append(Try(at=began, how=came.how, came_to=came.came_to, missing=[part.what for part in missing]))
        stands = stands_by(request.parts)
        logger.info("🧰 Request %s (%s): %s; %d of %d parts are so.%s", request.id, request.effect, stands,
                    len(request.parts) - len(missing), len(request.parts), f" {came.came_to}" if came.came_to else "")
        gained = missing_before is None or len(missing) < missing_before
        if not missing or not came.again or not gained:
            await ledger.changed(request, stands=stands, now="")
            return request
        missing_before = len(missing)
        await ledger.changed(request, stands=stands, now="")
        tell(f"Trying again for what is missing: {'; '.join(part.what for part in missing[:6])}.")


def _the_whole(request: Request) -> str:
    from core.understanding.effects import BY_NAME

    effect = BY_NAME.get(request.effect)
    return effect.met if effect is not None else "what was asked"


def _title(request: Request) -> str:
    words = " ".join(request.said.split())
    return words if len(words) <= 70 else words[:70].rsplit(" ", 1)[0] + " …"


def begin(request: Request, *, tell: Tell, told: Callable[[Request], Awaitable[None]],
          carrier: Carrier | None = None, ledger: RequestsOfHer | None = None) -> bool:
    """Start carrying a request out on her own time; False where nothing does this kind of thing or it is under way."""
    from core.utils.task_tracker import get_task_tracker

    carrier = carrier or carriers().get(request.effect)
    if carrier is None or is_under_way(request):
        return False

    bench = get_workbench().begin(request.id, _title(request), request.effect)

    async def work() -> None:
        try:
            # The turn that asked is answered first.
            bench.waits("the conversation, which comes first")
            await nobody_waiting()
            with working_on(bench):
                await carry_out(request, carrier, tell=tell, ledger=ledger)
        finally:
            _UNDER_WAY.pop(request.id, None)
            bench.finished("stopped" if bench.stopped or request.stands in {CUT_SHORT, SET_ASIDE} else "done")
        if request.stands not in {OPEN, SET_ASIDE}:
            await told(request)

    _UNDER_WAY[request.id] = get_task_tracker().create_task(work(), name=f"request.{request.effect}.{request.id}")
    return True
