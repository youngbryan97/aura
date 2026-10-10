"""What she knows herself of the place she is in, asked of her own model while she gets on with it.

Her model has read about a great many games, programs and sites, and about
the kinds they come in: what a lander game asks, how a spreadsheet is worked,
what a checkout page wants. Asked before she begins, it would hold her up for
the half-minute or more it takes; so it is asked beside her work, and what it
says goes into the guide to the place (core/cognition/a_guide_to_a_place.py)
when it comes, as a manual's sections: what it is for, how it is won or done,
what loses, how it works, what to get and keep clear of, and how to do well.

It is the least sure of what she is told of a place, and is kept as that: it may
be of places like this one rather than of this one, and it is never taken for
how the place is worked (its keys and controls), only for what it is for and how
to go about it. What the place shows her outranks it.

Nothing here knows a place.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = ["ask_in_the_background", "what_i_know_of"]

logger = logging.getLogger("Aura.WhatIKnowOfIt")

#: The most her model is asked to write, in tokens, and the most of what the place says that goes with the question.
MOST_TOKENS = 450
MOST_SAID = 700


async def what_i_know_of(place: str, task: str, said: str, ask: Callable[..., Awaitable[Any]]) -> dict[str, Any] | None:
    """Her model's manual of the place, as plain sections; None where it gave none."""
    from pydantic import BaseModel, Field

    class _Manual(BaseModel):
        goal: str = Field(default="", max_length=240, description="what it is for: what a person is there to do")
        win: str = Field(default="", max_length=240, description="how it is won or done; empty if unsure")
        lose: str = Field(default="", max_length=240, description="what loses, fails or goes wrong; empty if unsure")
        how: str = Field(default="", max_length=400, description="how it works: what moves, what pulls, what costs")
        get: list[str] = Field(default_factory=list, max_length=5, description="things worth getting or using")
        avoid: list[str] = Field(default_factory=list, max_length=5, description="things to keep clear of")
        tips: list[str] = Field(default_factory=list, max_length=4, description="how to do well at it")

    seen = " ".join(str(said or "").split())[:MOST_SAID]
    prompt = (f"Someone is about to {_the_task(task)} in “{place}”." + (f" What it shows them: {seen}" if seen else "")
              + f"\nFrom what you know of “{place}” (or, if you do not know it, of things of its kind, as what it shows "
              "describes), write them a short manual: what it is for; how it is won or done; what loses or goes wrong; "
              "how it works; what to get; what to keep clear of; and up to three tips. Leave out what you are not fairly "
              "sure of. Do not name keys or buttons: what it shows them says how it is worked.")
    try:
        got = await ask(prompt, _Manual, MOST_TOKENS)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("what I know of %r could not be asked: %s", place, str(why)[:160])
        return None
    return got.model_dump() if got is not None and hasattr(got, "model_dump") else None


def _the_task(task: str) -> str:
    said = " ".join(str(task or "").split())
    said = re.sub(r"^(?:please\s+)?(?:go to\s+\S+\s+and\s+)?", "", said, flags=re.I).rstrip(".")
    return said[:160] if said else "use it"


def ask_in_the_background(guide: Any, *, task: str, said: str, ask: Callable[..., Awaitable[Any]],
                          tell: Callable[[str], Any] | None = None) -> bool:
    """Ask her model of the place the guide is to, once, beside her work; what it says taken into the guide when it
    comes, and what it added said. Whether it was asked now."""
    if guide is None or getattr(guide, "asked_what_i_know", False) or not getattr(guide, "place", ""):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    guide.asked_what_i_know = True

    async def asked() -> None:
        manual = await what_i_know_of(guide.place, task, said, ask)
        if not manual:
            return
        filled = guide.take_in_what_i_know(manual)
        logger.info("what I know of %r: %s (filled %s)", guide.place, {k: v for k, v in manual.items() if v}, filled)
        line = _what_it_added(manual, filled)
        if line and tell is not None:
            tell(line)

    guide.asking_what_i_know = loop.create_task(asked())
    return True


def _what_it_added(manual: dict[str, Any], filled: list[str]) -> str:
    """What her own knowledge added to the guide, in a few words a watcher can follow."""
    parts: list[str] = []
    if "goal" in filled and manual.get("goal"):
        parts.append(f"it's for {_lower(manual['goal'])}")
    if "lose" in filled and manual.get("lose"):
        parts.append(f"what loses is {_lower(manual['lose'])}")
    tips = [str(t) for t in manual.get("tips") or [] if str(t).strip()]
    if tips:
        parts.append(f"a tip: {_lower(tips[0])}")
    avoid = [str(t) for t in manual.get("avoid") or [] if str(t).strip()][:3]
    if "avoid" in filled and avoid:
        parts.append(f"keep clear of {', '.join(avoid)}")
    return ("From what I know of it: " + "; ".join(parts) + ".") if parts else ""


def _lower(text: str) -> str:
    said = " ".join(str(text or "").split()).rstrip(".")
    return said[:1].lower() + said[1:] if said[:2] != said[:2].upper() else said
