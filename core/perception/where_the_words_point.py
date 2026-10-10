"""The place a thing's words say to put something, found on the screen by her eyes: "drag the device to the end of
another device's arrow", "drop the piece on the highlighted square", "put the toy in the box".

A person told where a thing goes looks for that place, and puts it there. They don't try every place on the screen
in turn. Her reading of a screen finds its writing and the shapes that look pressable. The end of an arrow, a lit
square or an open box is neither of those. LIVE 2026-10-10 a game said "drag the device at the end of another
device's arrow to make a connection". She carried the shapes she had found from one to another and never put a
device where an arrow ended.

So the place the words name (``the_place_named``) is asked of her eyes (the vision model) on the screen as it stands
still. Where they point is offered as a place to carry things to, first among the places. It is asked again once
something carried there has changed the screen, because the end of the arrow moves as the chain grows.

Nothing here knows a game.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PLACES", "PlacesNamed", "place_from", "remember_the_still_picture", "the_place_named"]

logger = logging.getLogger("Aura.WhereTheWordsPoint")

#: Where words say a thing is carried to: "drag ... to the end of the arrow", "drop it on the lit square".
_CARRIED_TO = re.compile(
    r"\b(?:drag|drop|put|place|move|carry|slide)\b[^.!?;]{0,60}?\b(?:to|into|onto|on|at|in|over|next to|beside)\s+"
    r"((?:the|a|an|another|any|each|your)\s[^.!?;,]{3,48}?)(?=\s+(?:to|so|and|for|when|then|until|before|after|while|if|unless|or|without)\b|[.!?;,]|$)",
    re.IGNORECASE)
#: What is no place: a thing's own name for the carrying ("the mouse", "the screen", "the left button").
_NO_PLACE = re.compile(r"\b(?:mouse|screen|button|cursor|key|keys|pointer|left|right)\b", re.IGNORECASE)
#: How long a place found is kept before her eyes are asked again, though nothing carried there changed the screen.
KEPT_FOR_S = 30.0
LOOK_TIMEOUT_S = 40.0

_PROMPT = ("This is the screen of a game. Point to {place}. Answer with JSON only: {{\"bbox_2d\": [x1, y1, x2, y2]}} on "
           "a grid of 0 to 1000 across and down, or {{\"bbox_2d\": null}} if it is not on the screen.")


def the_place_named(words: str) -> str:
    """The place words say a thing is carried to ("the end of another device's arrow"); "" where they name none."""
    for found in _CARRIED_TO.finditer(" ".join(str(words or "").split())):
        place = found.group(1).strip()
        if not _NO_PLACE.search(place):
            return place.lower()
    return ""


def place_from(answer: str) -> tuple[float, float] | None:
    """The middle of the box her eyes drew, as shares of the screen; None where they drew none."""
    found = re.search(r"\{.*\}", str(answer or ""), re.S)
    try:
        box = json.loads(found.group(0)).get("bbox_2d") if found else None
        x1, y1, x2, y2 = (float(v) for v in box)
    except (ValueError, TypeError, AttributeError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return (x1 + x2) / 2000, (y1 + y2) / 2000


async def _her_eyes(prompt: str, image_b64: str) -> str:
    from core.perception.her_eyes import look_with_her_eyes

    return await look_with_her_eyes(prompt, image_b64, max_tokens=60, timeout_s=LOOK_TIMEOUT_S)


@dataclass
class PlacesNamed:
    """The places words named, as her eyes found them on the screen standing still last: asked beside her work."""

    found: dict[str, tuple[tuple[float, float] | None, float]] = field(default_factory=dict)
    asking: dict[str, Any] = field(default_factory=dict)
    picture: Any = None
    see: Callable[[str, str], Awaitable[str]] = _her_eyes

    def where(self, place: str) -> tuple[float, float] | None:
        """Where ``place`` is, as her eyes last found it; asked of them (beside her work) where not found lately."""
        if not place:
            return None
        found = self.found.get(place)
        if found is not None and time.monotonic() - found[1] < KEPT_FOR_S:
            return found[0]
        task = self.asking.get(place)
        if (task is None or task.done()) and self.picture is not None:
            try:
                self.asking[place] = asyncio.get_running_loop().create_task(self._look(place, self.picture))
            except RuntimeError:
                pass
        return found[0] if found is not None else None

    def moved(self) -> None:
        """Something carried changed the screen: where the places are is looked for again."""
        self.found.clear()

    async def _look(self, place: str, picture: Any) -> None:
        from core.perception.where_i_am_on_screen import _as_jpeg

        try:
            image = await asyncio.to_thread(_as_jpeg, picture)
            answer = await asyncio.wait_for(self.see(_PROMPT.format(place=place), image), 2 * LOOK_TIMEOUT_S + 5)
        except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
            logger.info("her eyes could not find %r: %s", place, str(why)[:120])
            return
        at = place_from(answer)
        self.found[place] = (at, time.monotonic())
        logger.info("where %r is, to her eyes: %s", place, at)


#: The places named, for the screen she is working on.
PLACES = PlacesNamed()


def remember_the_still_picture(picture: Any) -> None:
    """The screen as it last stood still, for her eyes to look for named places on."""
    PLACES.picture = picture
