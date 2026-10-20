"""What the things she sees look like: a car, a zombie, a parking space, a bat; and what about one looks unlike its kind.

A person looking at a screen does not see "a green thing at the left": they see a zombie, a coin, a door. And what
they see a thing as tells them a good deal before they have touched it: a zombie is trouble, a coin is worth getting,
a platform is stood on. They notice, too, when a thing looks unlike how such things look (an orange with a face, a
rabbit with a sword, a parking space striped like a warning), and wonder what that is about.

Her play finds things by how they move and what colour they are (core/perception/what_moves_in_the_picture.py); this
asks her eyes (the vision model, core/brain/llm/mlx_vision_client.py) what each tracked instance is, one at a time, marked
with a box on the picture as it is: one thing marked is named far more surely than several, and each look takes under
a second. It is asked beside her play, never in its way: a few instances at a time, the one that is hers first, then
what moves, then what is still. Each instance keeps its own query and answer.

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
from numbers import Real
from typing import Any

import numpy as np

from core.perception.names_for_observed_instances import InstanceQuery, query_for
from core.perception.whole_referents_from_source_material import scene_referents
from core.verify import invariant

__all__ = ["LookingAtThings", "Sighting", "sighting_from"]

logger = logging.getLogger("Aura.WhatThingsLookLike")

#: How many instances are looked at in one go, how many in one game at most, how soon play must have gone on before the
#: first look, and how long between two goes.
AT_ONCE = 3
MOST_A_GAME = 14
FIRST_AFTER_S = 3.0
EVERY_S = 8.0
#: The least a thing must cover to be named (working pixels a side), and the most of the picture (by share a side): a
#: speck is no thing to name, and what covers the screen is the backdrop.
LEAST_SIDE = 4.0
MOST_SHARE = 0.6
#: How wide the picture is made for her eyes, and how long a look may take before it is given up: her eyes take
#: their turn with every other look (core/perception/her_eyes.py), so a look waits as well as works.
SEEN_WIDE = (768, 1024)
LOOK_TIMEOUT_S = 45.0
#: After her eyes fail, how long before they are tried again.
RESTED_S = 300.0

_PROMPT = ("This is part of a screen from a game, a website or a program. One thing in it is marked with a {colour} "
           "box. Answer with JSON only, filled in: {{\"is\": \"<the most specific everyday name for the thing in the "
           "box, in a word or a few, as a person would name it; not 'object', 'item', 'thing' or 'character'; "
           "'unclear' if you cannot tell>\", \"unusual\": <true only if the thing itself, not its setting or how it "
           "is drawn, is plainly unlike such a thing in the real world, such as a face on something that has none, a "
           "weapon, a strange colour, or two things in one; else false>, \"how\": \"<if unusual, what about it, in a "
           "few words; else empty>\", \"who\": \"<its name, if it is a character or a thing you recognise by name; else "
           "empty>\"}}")

#: Answers that say nothing is odd, or that nothing could be told; a bare yes is no saying what.
_NOTHING = re.compile(r"^\W*(?:no|none|nothing|n/?a|not really|normal|yes|true|false|no\W.*|nothing (?:odd|unusual).*|"
                      r"it looks normal.*)\W*$", re.I)
_UNCLEAR = re.compile(r"^\W*(?:unclear|unknown|unsure|cannot tell|can't tell|not sure|nothing|none|n/?a)\W*$", re.I)
#: Names that say what anything is: no name for a thing.
_NO_NAME = frozenset("""object objects item items thing things character characters shape shapes icon icons symbol symbols
    element elements graphic graphics image images sprite sprites figure figures picture pictures stuff entity""".split())
#: An odd look that is only of where a thing is or how it is drawn (every thing in a cartoon is cartoonish), unless it
#: names something about the thing itself.
_ONLY_HOW_DRAWN = re.compile(r"\b(?:setting|background|scene|environment|context|cartoon\w*|styli[sz]ed|digital|"
                             r"pixel\w*|drawn|animated|realistic|interface|game)\b", re.I)
_OF_THE_THING = re.compile(r"\b(?:face|eyes?|mouth|smil\w*|arms?|legs?|hands?|weapon|sword|gun|knife|axe|club|bat|crown|"
                           r"glow\w*|stitch\w*|half|halves|two|split|mixed|colou?r\w*|shape\w*|size|giant|huge|tiny|"
                           r"wings?|horns?|fangs?|teeth|blood\w*|scar\w*|mask\w*|hat|wearing|holding|carrying|"
                           r"twisted|broken|melting|burning|floating|flying|talking|angry|evil)\b", re.I)


@dataclass
class Sighting:
    """What one boxed thing was seen as, and what about it looked unusual ("" for nothing)."""

    what: str
    odd: str = ""
    at: float = 0.0
    #: Its own name, where her eyes knew it by one (a character, a famous thing): "Amy Rose", not "hedgehog".
    who: str = ""
    #: Added by the query owner after parsing; never read from the provider's JSON.
    query: InstanceQuery | None = None


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
    # What is odd about it is said only where her eyes said it is unusual at all: asked for what was odd, they said
    # what was not ("a letter, not a face or weapon") as readily as what was (LIVE 2026-10-10).
    unusual = said.get("unusual")
    flagged = unusual is True or str(unusual).strip().lower() in ("true", "yes")
    odd = " ".join(str((said.get("how") if flagged else "") or said.get("odd") or "").split()).strip()
    if not what or _UNCLEAR.match(what) or len(what.split()) > 5:
        return None
    what = re.sub(r"^(?:a|an|the)\s+", "", what)
    if what in _NO_NAME:
        return None
    if not odd or _NOTHING.match(odd) or (_ONLY_HOW_DRAWN.search(odd) and not _OF_THE_THING.search(odd)):
        odd = ""
    # Said as what it has, not as a comparison: "It has a face, which is not typical for an apple" is "has a face".
    odd = re.sub(r",?\s+(?:which|unlike|not typical|not like|as opposed)\b.*$", "", odd, flags=re.I)
    odd = re.sub(r"^(?:it|this|the thing)\s+(?=has|is|wears|holds|looks|carries)", "", odd, flags=re.I)
    who = " ".join(str(said.get("who") or "").split()).strip(" .")
    if not who or _UNCLEAR.match(who) or who.lower() in (what, *_NO_NAME) or len(who.split()) > 4 or not who[:1].isupper():
        who = ""
    return Sighting(what=what, odd=odd.rstrip(". ")[:160], at=at, who=who)


async def _her_eyes(prompt: str, image_b64: str) -> str:
    from core.perception.her_eyes import look_with_her_eyes

    return await look_with_her_eyes(prompt, image_b64, max_tokens=90, timeout_s=LOOK_TIMEOUT_S)


def _naming_time(at: Any) -> float | None:
    """Canonicalize the capture coordinate; the instance query owns its box and birth."""
    if isinstance(at, bool) or not isinstance(at, Real):
        return None
    try:
        value = float(at)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


@invariant('perception.visual_naming_accepts_measured_capture_times', scope='perception',
           owner='core/perception/what_things_look_like.py', observational=False)
def _visual_naming_clock_invariant() -> tuple:
    from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime

    at = MeasuredFrameTime(4., lower=4., upper=4.001, arrived=4.002,
                           source='clock', source_at=10., epoch=1)
    assert _naming_time(at) == 4. and _naming_time(np.float32(4.)) == 4.
    assert all(_naming_time(value) is None for value in (True, None, -1., math.inf, 10**400))
    return ()


@dataclass
class LookingAtThings:
    """Names for boxed instances, asked beside play; agreeing answers also supply a kind summary."""

    by_kind: dict[int, Sighting] = field(default_factory=dict)
    see: Callable[[str, str], Awaitable[str]] = _her_eyes
    asked: set[int] = field(default_factory=set)
    asked_instances: set[tuple] = field(default_factory=set)
    by_instance: dict[tuple, Sighting] = field(default_factory=dict)
    #: Last queried instance of each kind, retained for older callers' diagnostics.
    looked_on: dict[int, int] = field(default_factory=dict)
    _began: float | None = None
    _last: float = -math.inf
    #: When her eyes were last asked, in play's time.
    asked_at: float = -math.inf
    _asking: Any = None
    _new: list[tuple[int, Sighting]] = field(default_factory=list)
    _rested_until: float = -math.inf
    _done_at: float = -math.inf

    def look(self, picture: Any, moves: Any, at: float, *, mine: int | None = None, met: Any = (),
             sought: tuple[tuple, ...] = ()) -> bool:
        """Ask what the instances in ``picture`` not yet named are, where it is time to: hers (``mine``, a thing's number)
        first, then those met (kinds), then what moves, then what is still. Whether it was asked now."""
        at = _naming_time(at)
        if at is None:
            return False
        self._began = at if self._began is None else self._began
        if (self._asking is not None and not self._asking.done()) or len(self.asked_instances) >= MOST_A_GAME:
            return False
        # The time between goes from when the last looks were done, not begun: LIVE 2026-10-10 fifteen looks in thirty
        # seconds, as play was finding which thing was hers.
        if (at - self._began < FIRST_AFTER_S or at - self._last < EVERY_S or time.monotonic() < self._rested_until
                or time.monotonic() - self._done_at < EVERY_S):
            return False
        frame = np.asarray(picture)
        if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] < 3
                or min(frame.shape[:2]) < 1 or max(frame.shape[:2]) > 4096):
            return False
        frame = np.ascontiguousarray(frame[..., :3]).copy()
        chosen = [query.with_picture(frame) for query in self._to_look_at(moves, mine, set(met or ()), at=at, sought=sought)]
        if not chosen:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._last = self.asked_at = at
        self.asked |= {query.kind for query in chosen}
        self.asked_instances |= {query.key for query in chosen}
        scale = float(getattr(moves, "scale", 1.0) or 1.0)
        self._asking = loop.create_task(self._ask(frame, chosen, scale, at))
        return True

    def resting(self) -> bool:
        """Whether her eyes failed lately and are let rest before they are asked again."""
        return time.monotonic() < self._rested_until

    def newly(self) -> list[tuple[int, Sighting]]:
        """The kinds named since this was last asked, with what each was seen as."""
        new, self._new = self._new, []
        return new

    def _to_look_at(self, moves: Any, mine: int | None, met: set[int], *, at: float | None = None,
                    sought: tuple[tuple, ...] = ()) -> list[InstanceQuery]:
        tall, wide = getattr(moves, "shape", (0, 0)) or (0, 0)
        best = []
        for thing in scene_referents(moves):
            query = query_for(thing, moves, at if at is not None else getattr(thing, 'seen', None))
            if query is None or query.key in self.asked_instances or query.key in self.by_instance:
                continue
            if min(thing.w, thing.h) < LEAST_SIDE or (wide and thing.w > MOST_SHARE * wide) or (tall and thing.h > MOST_SHARE * tall):
                continue
            best.append((query, thing))
        chosen = sorted(best, key=lambda item: (0 if item[1].number == mine else 1,
            0 if item[1].kind in met else 1, 0 if item[1].moved else 1, -item[1].w*item[1].h))
        # One slot serves the unresolved task. Ownership, size and prior-query
        # gates above still apply, and the remaining slots keep their order.
        available = {query.key: (query, thing) for query, thing in chosen}
        wanted = next((available[key] for key in sought[:32] if key in available), None)
        if wanted is not None:
            chosen = [wanted, *(item for item in chosen if item[0].key != wanted[0].key)]
        chosen = chosen[:min(AT_ONCE, MOST_A_GAME-len(self.asked_instances))]
        self.looked_on.update({query.kind: query.number for query, _thing in chosen})
        return [query for query, _thing in chosen]

    async def _ask(self, frame: np.ndarray, chosen: list[InstanceQuery], scale: float, at: float) -> None:
        try:
            await self._ask_each(frame, chosen, scale, at)
        finally:
            self._done_at = time.monotonic()

    async def _ask_each(self, frame: np.ndarray, chosen: list[InstanceQuery], scale: float,
                        at: float) -> None:
        for query in chosen:
            kind, box = query.kind, query.box
            try:
                colour, image = await asyncio.to_thread(_marked, frame, box, scale)
                answer = await asyncio.wait_for(self.see(_PROMPT.format(colour=colour), image), 2 * LOOK_TIMEOUT_S + 5)
            except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
                logger.info("her eyes could not say what a thing is: %s", str(why)[:160])
                self._rested_until = time.monotonic() + RESTED_S
                return
            sighting = sighting_from(answer, at)
            logger.info("what a thing looks like: kind %s seen as %r (%s)", kind, getattr(sighting, "what", None),
                        getattr(sighting, "odd", "") or "nothing odd")
            if sighting is not None:
                sighting.query = query
                self.by_instance[query.key] = sighting
                peers = [value for value in self.by_instance.values() if value.query.kind == kind]
                if len({(value.what, value.who) for value in peers}) == 1:
                    self.by_kind[kind] = sighting
                else:
                    self.by_kind.pop(kind, None)
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
