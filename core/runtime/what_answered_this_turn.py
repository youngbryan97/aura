"""Which route actually answered, and which has never answered anything.

Fourteen routes get offered every turn in order — the sequence induction, the
positional solver, the tabular reader, the game solver, the filesystem count,
and the rest. Each returns the reply unchanged when it declines, which is the
right behaviour and also means a route that CANNOT fire looks exactly like one
that rarely applies.

An external review put it as: a channel wired to a consumer is not a measured
downstream effect. That is the distinction this keeps. Offering is counted,
answering is counted, and a route offered five hundred times that has never
answered once is reported by name.

Deliberately not a policy. Nothing here decides whether a route runs; it
records what happened when it did. A measurement that also intervenes cannot
be used to judge the intervention.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.WhatAnsweredThisTurn")

__all__ = [
    "ARoute",
    "how_the_routes_have_gone",
    "offer",
    "offer_async",
    "routes_that_have_never_answered",
    "what_answered",
]

#: How many turns a route has to be offered before never answering is a
#: finding rather than a small sample. Not a threshold on quality — a route
#: that legitimately applies to one turn in a thousand needs a long window
#: before its silence means anything.
ENOUGH_TO_JUDGE = 50


@dataclass
class ARoute:
    """What one answer route has done."""

    name: str
    offered: int = 0
    answered: int = 0
    declined: int = 0
    raised: int = 0
    last_answered_at: float = 0.0
    #: The most recent turn it changed, trimmed. For reading a surprise, not
    #: for storing conversation: capped hard and never persisted.
    last_answer: str = ""
    why_it_raised: list[str] = field(default_factory=list)

    @property
    def share(self) -> float:
        return self.answered / self.offered if self.offered else 0.0


_ROUTES: dict[str, ARoute] = {}
_LOCK = checked_lock("what_answered_this_turn")

#: How much of an answer is kept for reading back. Enough to recognise it.
_HOW_MUCH_OF_AN_ANSWER = 240
#: How many distinct failures are kept per route.
_HOW_MANY_FAILURES = 8


def _route(name: str) -> ARoute:
    found = _ROUTES.get(name)
    if found is None:
        found = ARoute(name=name)
        _ROUTES[name] = found
    return found


def _record(name: str, before: str, after: str) -> None:
    with _LOCK:
        route = _route(name)
        route.offered += 1
        if after.strip() != before.strip():
            route.answered += 1
            route.last_answered_at = time.time()
            route.last_answer = after.strip()[:_HOW_MUCH_OF_AN_ANSWER]
        else:
            route.declined += 1


def _blame(name: str, exc: BaseException) -> None:
    with _LOCK:
        route = _route(name)
        route.offered += 1
        route.raised += 1
        why = f"{type(exc).__name__}: {exc}"[:200]
        if why not in route.why_it_raised:
            route.why_it_raised.append(why)
            del route.why_it_raised[:-_HOW_MANY_FAILURES]


def offer(name: str, before: Any, run: Callable[[], Any]) -> Any:
    """Offer this turn to one route and record what it did.

    Returns whatever the route returned. A route that raises is counted and
    the reply is left alone: an answer route is an improvement on the model's
    words, and a broken one must not cost the turn.
    """
    body = str(before or "")
    try:
        after = run()
    except Exception as exc:  # noqa: BLE001 — a broken route must not cost a turn
        _blame(name, exc)
        logger.debug("answer route %s raised: %s", name, exc)
        return before
    _record(name, body, str(after or ""))
    return after


async def offer_async(name: str, before: Any, run: Callable[[], Any]) -> Any:
    """The same, for a route that has to await something."""
    body = str(before or "")
    try:
        after = await run()
    except Exception as exc:  # noqa: BLE001 — a broken route must not cost a turn
        _blame(name, exc)
        logger.debug("answer route %s raised: %s", name, exc)
        return before
    _record(name, body, str(after or ""))
    return after


def what_answered(name: str) -> ARoute | None:
    with _LOCK:
        found = _ROUTES.get(name)
        return ARoute(**vars(found)) if found else None


def how_the_routes_have_gone() -> dict[str, dict[str, Any]]:
    """Every route, by how often it was offered a turn and answered one."""
    with _LOCK:
        return {
            name: {
                "offered": route.offered,
                "answered": route.answered,
                "declined": route.declined,
                "raised": route.raised,
                "share": round(route.share, 4),
                "last_answered_at": route.last_answered_at,
                "why_it_raised": list(route.why_it_raised),
            }
            for name, route in sorted(_ROUTES.items())
        }


def routes_that_have_never_answered(enough: int = ENOUGH_TO_JUDGE) -> list[str]:
    """Routes offered enough turns to judge, that have never answered one.

    Either the route cannot fire, or its gate is wrong, or the thing it
    answers never comes up. All three are worth knowing and none of them are
    visible from the source.
    """
    with _LOCK:
        return sorted(
            name
            for name, route in _ROUTES.items()
            if route.offered >= enough and route.answered == 0
        )


def forget_everything() -> None:
    """Drop the record. For tests; the live runtime never calls this."""
    with _LOCK:
        _ROUTES.clear()
