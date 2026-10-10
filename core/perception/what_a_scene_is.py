"""What a screen is a picture of, at a glance: its setting, and what stands out in it.

A person takes in a scene before any one thing in it: a court, a ring, a road through hills, a night sky over a
backyard, a page of tracks and instruments. The setting says what a place is about as much as its name does, and most
of it is backdrop that her play, which finds things by how they move and what colour they are
(core/perception/what_moves_in_the_picture.py), never sees as things at all.

So her eyes (the vision model, core/brain/llm/mlx_vision_client.py) are asked, beside her play and never in its way,
what the whole picture is of and what stands out in it, on a few screens of a place: its first, and the first few she
had not seen before. What they say goes to the reading of what the place is (core/cognition/what_this_place_is.py).

Nothing here knows a place.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Glance", "glance_from", "glance_beside"]

logger = logging.getLogger("Aura.WhatASceneIs")

#: How many screens of one place are glanced at, the most things kept from one glance, and how long a glance may take.
GLANCES_A_PLACE = 3
THINGS_KEPT = 6
GLANCE_TIMEOUT_S = 45.0
#: How wide the picture is made for her eyes.
SEEN_WIDE = 768

_PROMPT = ("This is a picture of a screen, from a game, a website or a program. What is the picture of, as a setting or "
           "scene, in a few words? And what stands out in it: up to six things, each named in a word or a few as a "
           "person would name it. Answer with JSON only: {\"scene\": \"...\", \"things\": [\"...\"]}")
_UNCLEAR = re.compile(r"^\W*(?:unclear|unknown|unsure|cannot tell|can't tell|not sure|nothing|none|n/?a)\W*$", re.I)


@dataclass
class Glance:
    """What one screen was seen as: its setting, and the things that stand out in it."""

    scene: str
    things: list[str] = field(default_factory=list)


def glance_from(answer: str) -> Glance | None:
    """Her eyes' answer as a glance; None where it said nothing of the scene."""
    found = re.search(r"\{.*\}", str(answer or ""), re.S)
    try:
        said = json.loads(found.group(0)) if found else {}
    except ValueError:
        said = {}
    scene = " ".join(str(said.get("scene") or "").split()).strip(" .")
    if not scene or _UNCLEAR.match(scene) or len(scene.split()) > 14:
        return None
    things = []
    for thing in said.get("things") or []:
        name = re.sub(r"^(?:a|an|the)\s+", "", " ".join(str(thing).lower().split()).strip(" ."))
        if name and not _UNCLEAR.match(name) and len(name.split()) <= 5 and name not in things:
            things.append(name)
    return Glance(scene=scene[:1].lower() + scene[1:], things=things[:THINGS_KEPT])


async def _her_eyes(prompt: str, image_b64: str) -> str:
    from core.perception.her_eyes import look_with_her_eyes

    return await look_with_her_eyes(prompt, image_b64, max_tokens=120, timeout_s=GLANCE_TIMEOUT_S)


def _as_jpeg(picture: Any) -> str:
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(np.asarray(picture)[..., :3]).astype(np.uint8))
    if image.width != SEEN_WIDE:
        image = image.resize((SEEN_WIDE, max(1, round(image.height * SEEN_WIDE / max(1, image.width)))))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode("ascii")


def glance_beside(look: Callable[[], Awaitable[Any]], guide: Any, *, screen: str = "",
                  see: Callable[[str, str], Awaitable[str]] = _her_eyes, then: Callable[[], Any] | None = None) -> bool:
    """Glance at the screen in front of her, beside her work, where the place has had fewer than its glances and this
    screen (by its words) has not been glanced at; what her eyes say kept in the reading of what the place is, and
    ``then`` told. Whether a glance was begun."""
    reading = getattr(guide, "reading", None)
    if reading is None or len(reading.glanced) >= GLANCES_A_PLACE or screen[:80] in reading.glanced:
        return False
    if reading.glancing is not None and not reading.glancing.done():
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    reading.glanced.append(screen[:80])

    async def glanced() -> None:
        try:
            seen = await look()
            if seen is None:
                return
            picture = seen[0] if isinstance(seen, tuple) else seen
            image = await asyncio.to_thread(_as_jpeg, picture)
            answer = await asyncio.wait_for(see(_PROMPT, image), 2 * GLANCE_TIMEOUT_S + 5)
        except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
            logger.info("her eyes could not take in the scene: %s", str(why)[:160])
            return
        glance = glance_from(answer)
        logger.info("the scene, at a glance: %s", glance)
        if glance is not None:
            reading.glimpsed(glance.scene, glance.things)
            if then is not None:
                then()

    reading.glancing = loop.create_task(glanced())
    return True
