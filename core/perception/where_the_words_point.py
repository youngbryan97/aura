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
import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.verify.invariants import invariant

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
#: Retained answers and requests still running, including requests awaiting cancellation.
MOST_PLACES = 32
MOST_LOOKS = 4
MOST_PLACE_CHARS = 240
MOST_IMAGE_BYTES = 64 * 1024 * 1024

_PROMPT = ("Point to {place} on this screen. Answer with JSON only: {{\"bbox_2d\": [x1, y1, x2, y2]}} on "
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
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in box)):
            return None
        x1, y1, x2, y2 = (float(v) for v in box)
    except (ValueError, TypeError, AttributeError):
        return None
    if not (all(math.isfinite(v) for v in (x1, y1, x2, y2))
            and 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
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
    surface: str = ""
    episode: str = ""
    viewport: tuple[float, ...] = ()
    revision: int = field(default=0, init=False)
    _image_key: str = field(default="", init=False, repr=False)
    _found_in: dict[str, tuple[Any, ...]] = field(default_factory=dict, init=False, repr=False)
    _tasks: set[asyncio.Task] = field(default_factory=set, init=False, repr=False)

    @property
    def identity(self) -> tuple[Any, ...]:
        """The surface, episode, viewport and image revision an answer must belong to."""
        return self.surface, self.episode, self.viewport, self.revision, self._image_key

    def set_context(self, *, surface: str, episode: str, viewport: Sequence[float] = ()) -> None:
        """Bind the surface being observed. A changed surface, episode or viewport requires a fresh picture."""
        bounds = tuple(float(v) for v in viewport)
        if not all(math.isfinite(v) for v in bounds):
            raise ValueError("a viewport must have finite coordinates")
        context = str(surface), str(episode), bounds
        if context != (self.surface, self.episode, self.viewport):
            self.surface, self.episode, self.viewport = context
            self.moved()

    def remember(self, picture: Any, *, surface: str | None = None, episode: str | None = None,
                 viewport: Sequence[float] | None = None) -> None:
        """Observe a picture, preserving the current context where none is supplied. Different pixels retire geometry."""
        if surface is not None or episode is not None or viewport is not None:
            self.set_context(surface=self.surface if surface is None else surface,
                             episode=self.episode if episode is None else episode,
                             viewport=self.viewport if viewport is None else viewport)
        self.picture = picture
        self._refresh_picture()

    def _refresh_picture(self) -> np.ndarray | None:
        pixels, key = _pixels_and_key(self.picture)
        if key != self._image_key:
            self._invalidate()
            self._image_key = key
        return pixels

    def _invalidate(self) -> None:
        self.revision += 1
        self.found.clear()
        self._found_in.clear()
        self.asking.clear()
        # Cancelled requests still count against MOST_LOOKS until they finish. An observer that delays cancellation
        # cannot make a succession of scene changes create unbounded work.
        for task in self._tasks:
            if not task.done() and not task.cancelling():
                task.cancel()

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        for place in [p for p, asking in self.asking.items() if asking is task]:
            del self.asking[place]
        if not task.cancelled() and task.exception() is not None:
            logger.error("a place observer failed: %s", task.exception())

    def _current_found(self, place: str, identity: tuple[Any, ...], now: float) -> Any:
        found = self.found.get(place)
        return (found if found is not None and self._found_in.get(place) == identity
                and 0 <= now - found[1] < KEPT_FOR_S else None)

    def where(self, place: str) -> tuple[float, float] | None:
        """Where ``place`` is, as her eyes last found it; asked of them (beside her work) where not found lately."""
        if not isinstance(place, str) or not place.strip() or len(place) > MOST_PLACE_CHARS:
            return None
        place = " ".join(place.split())
        pixels = self._refresh_picture()
        if pixels is None:
            return None
        identity = self.identity
        found = self._current_found(place, identity, time.monotonic())
        if found is not None:
            return found[0]
        self.found.pop(place, None)
        self._found_in.pop(place, None)
        task = self.asking.get(place)
        if (task is None or task.done()) and len(self._tasks) < MOST_LOOKS:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return None
            task = loop.create_task(self._look(place, pixels.copy(), identity))
            self.asking[place] = task
            self._tasks.add(task)
            task.add_done_callback(self._finished)
        return None

    def moved(self) -> None:
        """Something carried changed the screen: where the places are is looked for again."""
        self._invalidate()
        self.picture = None
        self._image_key = ""

    async def _look(self, place: str, picture: Any, identity: tuple[Any, ...]) -> None:
        from core.perception.where_i_am_on_screen import _as_jpeg

        try:
            image = await asyncio.to_thread(_as_jpeg, picture)
            self._refresh_picture()
            if identity != self.identity:
                return
            answer = await asyncio.wait_for(self.see(_PROMPT.format(place=place), image), 2 * LOOK_TIMEOUT_S + 5)
        except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
            logger.info("her eyes could not find %r: %s", place, str(why)[:120])
            return
        self._refresh_picture()
        if identity != self.identity:
            logger.debug("a place answer belonged to an earlier scene: %r", place)
            return
        at = place_from(answer)
        self.found[place] = (at, time.monotonic())
        self._found_in[place] = identity
        while len(self.found) > MOST_PLACES:
            oldest = next(iter(self.found))
            del self.found[oldest]
            self._found_in.pop(oldest, None)
        logger.info("where %r is, to her eyes: %s", place, at)


def _pixels_and_key(picture: Any) -> tuple[np.ndarray | None, str]:
    """Numeric image pixels and their exact geometry identity; unsupported pictures carry no coordinates."""
    if picture is None:
        return None, ""
    try:
        pixels = np.asarray(picture)
        if (pixels.ndim != 3 or pixels.shape[2] < 3 or min(pixels.shape) < 1
                or pixels.dtype.kind not in "buif" or pixels.nbytes > MOST_IMAGE_BYTES):
            return None, ""
        if pixels.dtype.kind == "f" and not np.isfinite(pixels).all():
            return None, ""
        pixels = np.ascontiguousarray(pixels)
        digest = hashlib.sha256()
        digest.update(str((pixels.shape, pixels.dtype.str)).encode("ascii"))
        digest.update(memoryview(pixels))
        return pixels, digest.hexdigest()
    except (TypeError, ValueError, AttributeError):
        return None, ""


#: The places named, for the screen she is working on.
PLACES = PlacesNamed()


def remember_the_still_picture(picture: Any, *, surface: str | None = None, episode: str | None = None,
                               viewport: Sequence[float] | None = None) -> None:
    """The screen as it last stood still, for her eyes to look for named places on."""
    PLACES.remember(picture, surface=surface, episode=episode, viewport=viewport)


@invariant("perception.named_places_require_current_scene", scope="perception",
           owner="core/perception/where_the_words_point.py", observational=False)
def _named_places_invariant() -> tuple:
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    places = PlacesNamed()
    places.remember(pixels, surface="one", episode="first", viewport=(0, 0, 3, 2))
    before = places.identity
    places.found["the destination"] = ((0.2, 0.3), time.monotonic() - KEPT_FOR_S)
    places._found_in["the destination"] = before
    assert places._current_found("the destination", before, time.monotonic()) is None, "expired geometry was offered"
    places.set_context(surface="two", episode="first", viewport=(0, 0, 3, 2))
    assert before != places.identity and places.picture is None and not places.found, "geometry crossed surfaces"
    return ()
