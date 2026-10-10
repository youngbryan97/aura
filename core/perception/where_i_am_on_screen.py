"""Which thing on a screen is the one the player acts through: her avatar, as a person sees it before touching anything.

A person shown a game they have never played knows, nearly always, which one is them before they press a key: the
game names it ("Use the mouse to move Frankie", "Playing as: Robin", a health bar with the hero's name on it), and it
looks the part (the one the view follows, the one the story is about, the one drawn in the middle of things). Watching
people play the games she plays, that is how every one of them began: they knew who they were, and acted through it.

Her play finds which thing is hers by what answers to her keys (core/agency/which_one_answers_to_her.py); that is the
proof. This is the first guess before the proof: her eyes (the vision model) are asked, once a game, which thing is the
player's own, told what the place calls it, and point at it. What they point at is matched to the things play follows,
and kept as the one to try her keys on first; where her trials find nothing that answers, it is taken for her, until
something answers otherwise.

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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Avatar", "avatar_from", "where_the_avatar_is", "which_thing_is_shown"]

logger = logging.getLogger("Aura.WhereIAmOnScreen")

#: How wide the picture is made for her eyes, and how long the look may take.
SEEN_WIDE = 768
LOOK_TIMEOUT_S = 45.0
#: How much of its size a box is grown by for a thing's middle to be counted inside it.
GROWN_BY = 0.25

_PROMPT = ("This is the screen of a game. Which one thing on it is the character or object the player controls, their "
           "own avatar?{named} It is often named by the game, by its health bar or a 'playing as' label, and is what "
           "the view follows. Not the mouse pointer. Answer with JSON only: {{\"avatar\": \"<what it is, in a few "
           "words>\", \"bbox_2d\": [x1, y1, x2, y2]}}, the box on a grid of 0 to 1000 across and down.")


@dataclass
class Avatar:
    """What her eyes took for the player's own thing, and where: a box as shares of the picture (left, top, right,
    bottom)."""

    what: str
    box: tuple[float, float, float, float]


def avatar_from(answer: str) -> Avatar | None:
    """Her eyes' answer, as an avatar and its box; None where it pointed at nothing usable."""
    found = re.search(r"\{.*\}", str(answer or ""), re.S)
    if not found:
        return None
    try:
        said = json.loads(found.group(0))
    except ValueError:
        return None
    what = " ".join(str(said.get("avatar") or "").split()).strip(" .")
    box = said.get("bbox_2d") or said.get("box")
    if not what or not isinstance(box, Sequence) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return Avatar(what=what, box=(x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000))


def which_thing_is_shown(avatar: Avatar, moves: Any) -> Any | None:
    """The thing play follows whose middle is inside the box her eyes drew (grown a little), nearest its middle."""
    left, top, right, bottom = avatar.box
    wide, tall = right - left, bottom - top
    left, right = left - GROWN_BY * wide, right + GROWN_BY * wide
    top, bottom = top - GROWN_BY * tall, bottom + GROWN_BY * tall
    middle = ((left + right) / 2, (top + bottom) / 2)
    inside = []
    for thing in (getattr(moves, "things", {}) or {}).values():
        x, y = moves.share(thing.x, thing.y)
        if left <= x <= right and top <= y <= bottom:
            inside.append((math.dist((x, y), middle), thing))
    return min(inside, key=lambda pair: pair[0])[1] if inside else None


async def _her_eyes(prompt: str, image_b64: str) -> str:
    from core.perception.her_eyes import look_with_her_eyes

    return await look_with_her_eyes(prompt, image_b64, max_tokens=90, timeout_s=LOOK_TIMEOUT_S)


async def where_the_avatar_is(picture: Any, names: Sequence[str] = (), *,
                              see: Callable[[str, str], Awaitable[str]] = _her_eyes) -> Avatar | None:
    """Her eyes asked which thing in ``picture`` is the player's own, told what the place calls it (``names``)."""
    named = f" The game calls it: {', '.join(names[:3])}." if names else ""
    try:
        image = await asyncio.to_thread(_as_jpeg, picture)
        answer = await asyncio.wait_for(see(_PROMPT.format(named=named), image), 2 * LOOK_TIMEOUT_S + 5)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("her eyes could not say which thing is hers: %s", str(why)[:160])
        return None
    avatar = avatar_from(answer)
    logger.info("which thing is hers, to her eyes: %s", avatar)
    return avatar


def _as_jpeg(picture: Any) -> str:
    from PIL import Image

    frame = np.ascontiguousarray(np.asarray(picture)[..., :3]).astype(np.uint8)
    image = Image.fromarray(frame)
    if image.width != SEEN_WIDE:
        image = image.resize((SEEN_WIDE, max(1, round(image.height * SEEN_WIDE / max(1, image.width)))))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode("ascii")
