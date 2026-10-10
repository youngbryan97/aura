"""What the things she sees look like: a car, a zombie, a parking space, a bat; and what about one looks unlike its kind.

A person looking at a screen does not see "a green thing at the left": they see a zombie, a coin, a door. And what
they see a thing as tells them a good deal before they have touched it: a zombie is trouble, a coin is worth getting,
a platform is stood on. They notice, too, when a thing looks unlike how such things look (an orange with a face, a
rabbit with a sword, a parking space striped like a warning), and wonder what that is about.

Her play finds things by how they move and what colour they are (core/perception/what_moves_in_the_picture.py); this
asks her eyes (the vision model, core/brain/llm/mlx_vision_client.py) what one of each kind is, one at a time, marked
with a box on the picture as it is: one thing marked is named far more surely than several, and each look takes under
a second. It is asked beside her play, never in its way: a few kinds at a time, the one that is hers first, then what
moves, then what is still; a kind once named is not asked again.

What a thing is seen as is a guess, and is kept as one: what play shows of it outranks it.

Nothing here knows a game.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["LookingAtThings", "Sighting", "sighting_from"]

logger = logging.getLogger("Aura.WhatThingsLookLike")

#: How many kinds are looked at in one go, how many in one game at most, how soon play must have gone on before the
#: first look, and how long between two goes.
AT_ONCE = 4
MOST_A_GAME = 14
FIRST_AFTER_S = 1.0
EVERY_S = 8.0
#: The least a thing must cover to be named (working pixels a side), and the most of the picture (by share a side): a
#: speck is no thing to name, and what covers the screen is the backdrop.
LEAST_SIDE = 4.0
MOST_SHARE = 0.6
#: How wide the picture is made for her eyes, and how long a look may take before it is given up.
SEEN_WIDE = (640, 1024)
LOOK_TIMEOUT_S = 20.0
#: After her eyes fail, how long before they are tried again.
RESTED_S = 300.0

_PROMPT = ("This is a picture of a screen, from a game, a website or a program. One thing in it is marked with a "
           "{colour} box. What is the thing in the box, in a word or a few, as a person would name it (\"unclear\" if "
           "you cannot tell)? Does anything about it look unlike how such a thing usually looks in the real world? Say "
           "what in a few words, or leave it empty. Answer with JSON only: {{\"is\": \"...\", \"odd\": \"\"}}")

#: Answers that say nothing is odd, or that nothing could be told.
_NOTHING = re.compile(r"^\W*(?:no|none|nothing|n/?a|not really|normal|no\W.*|nothing (?:odd|unusual).*|it looks normal.*)\W*$",
                      re.I)
_UNCLEAR = re.compile(r"^\W*(?:unclear|unknown|unsure|cannot tell|can't tell|not sure|nothing|none|n/?a)\W*$", re.I)


@dataclass
class Sighting:
    """What one kind of thing was seen as, and what about it looked unlike its kind ("" for nothing)."""

    what: str
    odd: str = ""
    at: float = 0.0


def sighting_from(answer: str, at: float = 0.0) -> Sighting | None:
    """Her eyes' answer, as a sighting; None where it named nothing."""
    text = str(answer or "").strip()
    found = re.search(r"\{.*\}", text, re.S)
    said: dict[str, Any] = {}
    if found:
        try:
            said = json.loads(found.group(0))
        except ValueError:
            said = {}
    what = " ".join(str(said.get("is") or "").split()).strip(" .").lower()
    odd = " ".join(str(said.get("odd") or "").split()).strip()
    if not what or _UNCLEAR.match(what) or len(what.split()) > 5:
        return None
    what = re.sub(r"^(?:a|an|the)\s+", "", what)
    return Sighting(what=what, odd="" if not odd or _NOTHING.match(odd) else odd[:160], at=at)


async def _her_eyes(prompt: str, image_b64: str) -> str:
    from core.brain.llm.mlx_vision_client import get_vision_client

    return await get_vision_client().see_async(prompt, image_b64, max_tokens=90, temp=0.0, timeout_s=LOOK_TIMEOUT_S)


@dataclass
class LookingAtThings:
    """What each kind of thing she has seen in one game looks like, asked of her eyes beside her play."""

    by_kind: dict[int, Sighting] = field(default_factory=dict)
    see: Callable[[str, str], Awaitable[str]] = _her_eyes
    asked: set[int] = field(default_factory=set)
    _began: float | None = None
    _last: float = -math.inf
    _asking: Any = None
    _new: list[tuple[int, Sighting]] = field(default_factory=list)
    _rested_until: float = -math.inf

    def look(self, picture: Any, moves: Any, at: float, *, mine: int | None = None, met: Any = ()) -> bool:
        """Ask what the kinds in ``picture`` not yet named are, where it is time to: hers (``mine``, a thing's number)
        first, then those met (kinds), then what moves, then what is still. Whether it was asked now."""
        self._began = at if self._began is None else self._began
        if (self._asking is not None and not self._asking.done()) or len(self.asked) >= MOST_A_GAME:
            return False
        if at - self._began < FIRST_AFTER_S or at - self._last < EVERY_S or time.monotonic() < self._rested_until:
            return False
        chosen = self._to_look_at(moves, mine, set(met or ()))
        if not chosen:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._last = at
        self.asked |= {kind for kind, _box in chosen}
        frame = np.array(picture)[..., :3]
        scale = float(getattr(moves, "scale", 1.0) or 1.0)
        self._asking = loop.create_task(self._ask(frame, chosen, scale, at))
        return True

    def newly(self) -> list[tuple[int, Sighting]]:
        """The kinds named since this was last asked, with what each was seen as."""
        new, self._new = self._new, []
        return new

    def _to_look_at(self, moves: Any, mine: int | None, met: set[int]) -> list[tuple[int, tuple[float, ...]]]:
        tall, wide = getattr(moves, "shape", (0, 0)) or (0, 0)
        best: dict[int, Any] = {}
        for thing in (getattr(moves, "things", {}) or {}).values():
            kind = int(getattr(thing, "kind", -1))
            if kind < 0 or kind in self.asked or kind in self.by_kind:
                continue
            if min(thing.w, thing.h) < LEAST_SIDE or (wide and thing.w > MOST_SHARE * wide) or (tall and thing.h > MOST_SHARE * tall):
                continue
            if kind not in best or thing.w * thing.h > best[kind].w * best[kind].h:
                best[kind] = thing

        def first(kind: int) -> tuple[int, int, int, float]:
            thing = best[kind]
            return (0 if thing.number == mine else 1, 0 if kind in met else 1, 0 if thing.moved else 1, -thing.w * thing.h)

        return [(kind, best[kind].box()) for kind in sorted(best, key=first)[:AT_ONCE]]

    async def _ask(self, frame: np.ndarray, chosen: list[tuple[int, tuple[float, ...]]], scale: float, at: float) -> None:
        for kind, box in chosen:
            try:
                colour, image = await asyncio.to_thread(_marked, frame, box, scale)
                answer = await asyncio.wait_for(self.see(_PROMPT.format(colour=colour), image), LOOK_TIMEOUT_S + 5)
            except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
                logger.info("her eyes could not say what a thing is: %s", str(why)[:160])
                self._rested_until = time.monotonic() + RESTED_S
                return
            sighting = sighting_from(answer, at)
            logger.info("what a thing looks like: kind %s seen as %r (%s)", kind, getattr(sighting, "what", None),
                        getattr(sighting, "odd", "") or "nothing odd")
            if sighting is not None:
                self.by_kind[kind] = sighting
                self._new.append((kind, sighting))


def _marked(frame: np.ndarray, box: tuple[float, ...], scale: float) -> tuple[str, str]:
    """The picture with one thing boxed in a colour unlike it, at a size her eyes read well, as base64 JPEG; and the
    colour's name."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.ascontiguousarray(frame.astype(np.uint8)))
    grow = min(max(1.0, SEEN_WIDE[0] / max(1, image.width)), SEEN_WIDE[1] / max(1, image.width))
    if abs(grow - 1.0) > 0.01:
        image = image.resize((max(1, round(image.width * grow)), max(1, round(image.height * grow))))
    x0, y0, x1, y1 = (v / max(1e-6, scale) * grow for v in box)
    inside = frame[max(0, int(y0 / grow)):max(1, int(y1 / grow)), max(0, int(x0 / grow)):max(1, int(x1 / grow))]
    mean = inside.reshape(-1, 3).mean(axis=0) if inside.size else np.array([0, 0, 0])
    # Magenta, unless the thing is itself much that colour.
    magenta = float(mean[0]) > 150 and float(mean[2]) > 150 and float(mean[1]) < 120
    colour, rgb = ("cyan", (0, 230, 255)) if magenta else ("magenta", (255, 0, 255))
    margin = 4
    ImageDraw.Draw(image).rectangle((x0 - margin, y0 - margin, x1 + margin, y1 + margin), outline=rgb, width=3)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return colour, base64.b64encode(out.getvalue()).decode("ascii")
