"""Relations between things on a screen, said in words and checked by measurement.

A step's end is often a relation between two things: "the roller sits at the end of the arrow", "the cage drops on
Jerry", "the box is under the ledge". Whether it holds is not in the screen's words, which never say it; it is in
where the two things are. Robots that follow instructions check the same way: the language names the things and the
relation, and the relation is measured on where perception found them (VoxPoser and ReKep, 2023 and 2024, write a
task's ends as constraints between named parts; Inner Monologue held each step to what the scene then showed).

Here a sentence is read for two things and the relation between them, each thing found on the screen by its name
(the words a region shows or her eyes named it by), and the relation measured on their boxes. Where the sentence names
no relation, or either thing cannot be found, nothing is measured: that is not a relation shown not to hold.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.verify.invariants import invariant

__all__ = ["a_relation_in", "holds", "located", "measured"]

_RELATIONS = {
    "on": ("on top of", "onto", "on", "upon"),
    "above": ("above", "over"),
    "below": ("below", "under", "underneath", "beneath"),
    "beside": ("next to", "beside", "by"),
    "left of": ("to the left of", "left of"),
    "right of": ("to the right of", "right of"),
    "inside": ("inside", "into", "in"),
    "touching": ("touching", "against", "on to"),
    "at": ("at the end of", "at the edge of", "at", "near", "reaches", "reaching"),
}
_SAID_AS = {phrase: name for name, phrases in _RELATIONS.items() for phrase in phrases}
_PHRASES = "|".join(sorted((re.escape(p) for p in _SAID_AS), key=len, reverse=True))
_SENTENCE = re.compile(
    rf"^\s*(?P<a>(?:the |a |an |your |my )?[\w' -]+?)\s+(?:(?:is|are|sits|sit|rests|rest|lands|land|stands|stand|goes|go|"
    rf"lies|lie|hangs|hang|ends up|end up|falls|fall|drops|drop|rolls|roll|moves|move|gets|get|comes|come|is now|now)\s+)*"
    rf"(?:right |just |directly |straight )?(?P<rel>{_PHRASES})\s+(?P<b>(?:the |a |an |your |my )?[\w' -]+?)\s*[.!]?\s*$",
    re.I)
_COMMON = frozenset("the a an your my of its it to and with is are".split())


def a_relation_in(sentence: str) -> tuple[str, str, str] | None:
    """The two things a sentence relates and the relation, as (thing, relation, other); None where it states none."""
    found = _SENTENCE.match(" ".join(str(sentence or "").split()))
    if found is None:
        return None
    return found.group("a").strip(), _SAID_AS[found.group("rel").lower()], found.group("b").strip()


def _words(name: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", str(name or "").lower()) if w not in _COMMON}


def _finite_box(values: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(values, (tuple, list)) or len(values) != 4 or any(isinstance(v, bool) for v in values):
        return None
    try:
        box = tuple(float(value) for value in values)
    # not a failure: a box that is not numbers is none
    except (TypeError, ValueError, OverflowError):
        return None
    x, y, width, height = box
    if (not all(math.isfinite(value) for value in box) or width <= 0 or height <= 0
            or not math.isfinite(x + width) or not math.isfinite(y + height)):
        return None
    return box


def located(name: str, regions: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, float] | None:
    """The unique best named region's measured box; missing geometry or tied names remain unknown."""
    wanted = _words(name)
    if not wanted:
        return None
    candidates, overlap = [], 0
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        shared = len(wanted & _words(str(region.get("text") or "")))
        if shared < overlap or not shared:
            continue
        box = _finite_box([region.get(key) for key in ("x", "y", "width", "height")])
        if shared > overlap:
            candidates, overlap = [box], shared
        else:
            candidates.append(box)
    # A named candidate whose extent was not measured still competes for the
    # referent. Dropping it would manufacture certainty about another one.
    return candidates[0] if len(candidates) == 1 else None


def holds(relation: str, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool | None:
    """Measure the relation on two finite extents, or return None when it cannot be measured."""
    a, b = _finite_box(a), _finite_box(b)
    if a is None or b is None:
        return None
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
    size = max(aw, ah, bw, bh, 1e-6)
    across = ax < bx + bw and bx < ax + aw
    down = ay < by + bh and by < ay + ah
    gap = max(0.0, max(ax, bx) - min(ax + aw, bx + bw), max(ay, by) - min(ay + ah, by + bh))
    return {
        "on": across and abs((ay + ah) - by) <= max(ah, bh) / 2 + size * 0.1,
        "above": acy < by and (across or abs(acx - bcx) <= size),
        "below": acy > by + bh and (across or abs(acx - bcx) <= size),
        "beside": down and gap <= size,
        "left of": acx < bx,
        "right of": acx > bx + bw,
        "inside": bx <= acx <= bx + bw and by <= acy <= by + bh,
        "touching": gap <= size * 0.15,
        "at": math.hypot(acx - bcx, acy - bcy) <= size * 1.5,
    }.get(relation)


def measured(sentence: str, regions: Sequence[Mapping[str, Any]]) -> bool | None:
    """Whether the relation a sentence states holds on the screen; None where it states none or either thing is not
    found."""
    relation = a_relation_in(sentence)
    if relation is None:
        return None
    thing, how, other = relation
    a, b = located(thing, regions), located(other, regions)
    if a is None or b is None or a == b:
        return None
    return holds(how, a, b)


@invariant("perception.spatial_relations_require_unique_finite_entities", scope="perception",
           owner="core/perception/relations_between_things.py", observational=False)
def _relations_require_unique_finite_entities() -> tuple:
    inside = {"text": "crate", "x": .15, "y": .15, "width": .1, "height": .1}
    outside = {"text": "crate", "x": .75, "y": .75, "width": .1, "height": .1}
    room = {"text": "room", "x": .1, "y": .1, "width": .3, "height": .3}
    assert measured("the crate is inside the room", [inside, outside, room]) is None
    assert measured("the crate is inside the room", [outside, inside, room]) is None
    assert measured("the crate is inside the room", [inside, room]) is True
    assert measured("the crate is inside the room", [{**inside, "x": math.nan}, room]) is None
    return ()
