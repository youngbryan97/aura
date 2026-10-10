"""What stands still in a place and matters to a player: the doors, ladders, chests, signs and waiting people of a
screen, seen at a glance, named and placed.

Her play follows what moves (core/perception/what_moves_in_the_picture.py). What stands still is the backdrop
against which motion is seen, and only something small and unlike its floor is taken off it as a thing. A door, a
ladder, a counter or a person standing at it is larger than that and never moves, so it is part of the backdrop.
Offline 2026-10-10, in a hall whose words said "Press the SPACE BAR to open doors", the doors never became things.
Her eyes named only her own figure, and she walked past every door.

Much of what a player goes to stands still: a door to open, a ladder to climb, an exit, a sign to read, a chest, a
person to talk to. Anyone looking at the screen sees these at once. The people in the recordings of these games
looked over each new screen before they moved: where the doors were, what could be picked up, who stood where.

So once a screen her eyes (the vision model) are asked what in it a player could go to, use or must keep clear of,
and where. They are told the things the place's words speak of, so that what was named is looked for. Each answer
is a name and a box (``bbox_2d`` on a 0 to 1000 grid). What they name is kept for the screen, as a share of the
picture, to be gone to and used (core/agency/using_a_control_on_a_thing.py).

Her eyes are surer of what a thing is than of exactly where it is. Asked of a row of three doors, they named three
doors of the right size, evenly spaced, the first a third of the picture from where it stood (offline 2026-10-10,
at every size of picture and every way of asking). A person told "there are three doors" looks, and sees where they
are. So does she (``placed``): each box her eyes drew is put on what stands out from its surroundings in the picture
at about that size and shape. Where there are as many such places as things of one name, they are paired in order.
Otherwise each box goes to the nearest one. A box with no such place near it is kept as her eyes drew it.

Nothing here knows a game.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Landmark", "landmarks_from", "placed", "what_is_here"]

logger = logging.getLogger("Aura.WhatIsInAPlace")

#: The most things her eyes are asked to name in one look, and how long the look may take.
MOST = 8
LOOK_TIMEOUT_S = 45.0
#: The largest share of the picture a thing may cover and still be one thing in it, not the place itself.
LARGEST = 0.3

_PROMPT = ("This is the screen of a game. List the things on it that a player could go to, use, open, climb, pick up, "
           "talk to, or must keep away from: doors, ladders, exits, items, chests, signs, characters, hazards.{named} "
           "Not the score or other writing, not the walls, floor or sky. Answer with JSON only: {{\"things\": "
           "[{{\"what\": \"<a few words>\", \"bbox_2d\": [x1, y1, x2, y2]}}]}}, at most {most}, each box on a grid of "
           "0 to 1000 across and down.")


@dataclass(frozen=True)
class Landmark:
    """A thing her eyes saw standing in the place: what it is, and its box as shares of the picture (left, top, right,
    bottom)."""

    what: str
    box: tuple[float, float, float, float]

    @property
    def middle(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def _box(said: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(said, Sequence) or isinstance(said, str) or len(said) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in said)
    except (TypeError, ValueError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    if (x2 - x1) * (y2 - y1) > LARGEST * 1000 * 1000:
        return None
    return x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000


def landmarks_from(answer: str) -> list[Landmark]:
    """Her eyes' answer, as the things named in it with their boxes; what is unnamed, unplaced or the whole place,
    left out. A list given bare, or as one thing each, is read as well as the asked-for form."""
    text = str(answer or "")
    found = re.search(r"[\[{].*[\]}]", text, re.S)
    if not found:
        return []
    try:
        said = json.loads(found.group(0))
    except ValueError:
        return []
    things = said.get("things") if isinstance(said, dict) else said
    if isinstance(things, dict):
        things = [things]
    seen: list[Landmark] = []
    for one in things if isinstance(things, list) else []:
        if not isinstance(one, dict):
            continue
        what = " ".join(str(one.get("what") or one.get("label") or one.get("name") or "").split()).strip(" .").lower()
        box = _box(one.get("bbox_2d") or one.get("box"))
        if what and box is not None and len(what.split()) <= 5:
            seen.append(Landmark(what, box))
    return seen[:MOST]


async def _her_eyes(prompt: str, image_b64: str) -> str:
    from core.perception.her_eyes import look_with_her_eyes

    return await look_with_her_eyes(prompt, image_b64, max_tokens=420, timeout_s=LOOK_TIMEOUT_S)


async def what_is_here(picture: Any, speaks_of: Sequence[str] = (), *,
                       see: Callable[[str, str], Awaitable[str]] = _her_eyes) -> list[Landmark]:
    """Her eyes asked what in ``picture`` a player could go to, use or must keep clear of, and where; told what the
    place's words speak of (``speaks_of``: "doors", "stars"), so those are looked for."""
    from core.perception.where_i_am_on_screen import _as_jpeg

    named = f" The game speaks of: {', '.join(dict.fromkeys(speaks_of))}." if speaks_of else ""
    try:
        image = await asyncio.to_thread(_as_jpeg, picture)
        answer = await asyncio.wait_for(see(_PROMPT.format(named=named, most=MOST), image), 2 * LOOK_TIMEOUT_S + 5)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("her eyes could not say what is in this place: %s", str(why)[:160])
        return []
    seen = landmarks_from(answer)
    try:
        seen = await asyncio.to_thread(placed, picture, seen)
    except (ValueError, TypeError, MemoryError) as why:
        logger.debug("what her eyes saw could not be placed on the picture: %s", why)
    logger.info("what is in this place, to her eyes: %s", "; ".join(f"{s.what} at {s.middle[0]:.2f},{s.middle[1]:.2f}"
                                                                   for s in seen) or "nothing")
    return seen


#: The width the picture is read at to place what her eyes named, and how unlike its surroundings a place must be.
PLACED_WIDE = 160
STANDS_OUT_BY = 40
#: How far a place's size and shape may be from the box her eyes drew, as a factor, and how far off it may be, in
#: the box's own sizes, for the box to be put on it.
SIZED_WITHIN = 2.2
OFF_BY_AT_MOST = 2.5


def _places_that_stand_out(picture: Any, largest: float, least: float) -> list[tuple[tuple[float, float, float, float],
                                                                                     tuple[float, float, float]]]:
    """The places in a picture unlike their surroundings, as boxes in shares, with their colour: unlike the middle value
    around them and the middle value of their own row, at a reach a few times the largest thing looked for. Where a
    wall meets a floor, the middle value around a pixel near the edge is the other side's, and a strip along it stands
    out, joining up everything standing there (offline 2026-10-10, a row of doors read as one band); along its own
    row, it is the wall it is. What is thinner than half the least thing looked for (writing, an edge) is left out."""
    import numpy as np
    from scipy import ndimage

    from core.perception.picture_arithmetic import apart, median, pieces, shrink

    frame = np.asarray(picture)[..., :3]
    tall, wide = frame.shape[:2]
    small_tall = max(8, round(tall * PLACED_WIDE / max(1, wide)))
    small = shrink(frame, PLACED_WIDE, small_tall)
    reach = max(9, int(2.4 * largest * max(PLACED_WIDE, small_tall)) | 1)
    floor = median(small, min(reach, (min(PLACED_WIDE, small_tall) // 2) | 1))
    along = ndimage.median_filter(small, size=(1, min(2 * reach + 1, PLACED_WIDE | 1), 1), mode="nearest")
    unlike = (apart(small, floor).max(axis=2) > STANDS_OUT_BY) & (apart(small, along).max(axis=2) > STANDS_OUT_BY)
    thin = max(2, int(0.5 * least * PLACED_WIDE))
    unlike = ndimage.binary_opening(unlike, structure=np.ones((thin, thin), bool))
    count, labels, stats, _centres = pieces(unlike)
    places = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= 4:
            colour = small[labels == label].mean(axis=0)
            places.append(((x / PLACED_WIDE, y / small_tall, (x + w) / PLACED_WIDE, (y + h) / small_tall),
                           (float(colour[0]), float(colour[1]), float(colour[2]))))
    return places


def _alike(box: tuple[float, ...], other: tuple[float, ...]) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    ow, oh = other[2] - other[0], other[3] - other[1]
    return (max(w, ow) / max(1e-6, min(w, ow)) <= SIZED_WITHIN and max(h, oh) / max(1e-6, min(h, oh)) <= SIZED_WITHIN)


def placed(picture: Any, landmarks: list[Landmark]) -> list[Landmark]:
    """What her eyes named, each put on the place in the picture it most likely is: of the places standing out from
    their surroundings at about its size and shape, paired in order where there are as many as things of its name,
    else the nearest; as her eyes drew it where there is none near."""
    if not landmarks:
        return landmarks
    largest = max(max(m.box[2] - m.box[0], m.box[3] - m.box[1]) for m in landmarks)
    least = min(min(m.box[2] - m.box[0], m.box[3] - m.box[1]) for m in landmarks)
    found = _places_that_stand_out(picture, largest, least)
    places = [box for box, _colour in found]
    colours = [colour for _box, colour in found]
    out: dict[int, Landmark] = {}
    taken: set[int] = set()
    for name in dict.fromkeys(m.what for m in landmarks):
        named = [(i, m) for i, m in enumerate(landmarks) if m.what == name]
        fits = [j for j, p in enumerate(places) if j not in taken and all(_alike(m.box, p) for _i, m in named)]
        if len(fits) > len(named) > 1:
            fits = _most_alike(fits, places, colours, len(named))
        if len(fits) == len(named):
            order = sorted(named, key=lambda pair: (pair[1].middle[0], pair[1].middle[1]))
            spots = sorted(fits, key=lambda j: ((places[j][0] + places[j][2]) / 2, (places[j][1] + places[j][3]) / 2))
            for (i, m), j in zip(order, spots, strict=True):
                out[i], _ = Landmark(m.what, places[j]), taken.add(j)
            continue
        for i, m in named:
            size = max(m.box[2] - m.box[0], m.box[3] - m.box[1])
            near = [(abs((p[0] + p[2]) / 2 - m.middle[0]) + abs((p[1] + p[3]) / 2 - m.middle[1]), j)
                    for j, p in enumerate(places) if j not in taken and _alike(m.box, p)]
            best = min(near, default=None)
            if best is not None and best[0] <= OFF_BY_AT_MOST * size:
                out[i] = Landmark(m.what, places[best[1]])
                taken.add(best[1])
            else:
                out[i] = m
    return [out[i] for i in range(len(landmarks))]


def _most_alike(fits: list[int], places: list[tuple[float, ...]], colours: list[tuple[float, ...]], many: int) -> list[int]:
    """Of more places than things of one name, the ``many`` most like one another in size and colour: things of one
    name in one place are mostly drawn alike."""
    def apart(a: int, b: int) -> float:
        wa, ha = places[a][2] - places[a][0], places[a][3] - places[a][1]
        wb, hb = places[b][2] - places[b][0], places[b][3] - places[b][1]
        size = abs(wa - wb) / max(wa, wb, 1e-6) + abs(ha - hb) / max(ha, hb, 1e-6)
        return size + max(abs(x - y) for x, y in zip(colours[a], colours[b], strict=True)) / 64.0

    best: list[int] = []
    best_spread = float("inf")
    for first in fits:
        group = sorted(fits, key=lambda other: apart(first, other))[:many]
        spread = sum(apart(first, other) for other in group)
        if spread < best_spread:
            best, best_spread = group, spread
    return best
