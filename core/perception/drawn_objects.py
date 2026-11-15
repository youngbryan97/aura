"""Measured drawings that may be manipulated, separate from pressable controls.

An extent is a candidate for a bounded interaction, not a claim of an object's
name or affordance. Local contrast finds drawings; enclosing frames group them.
Writing and named controls supply exclusion evidence. Delivery and its observed
effect must establish whether a candidate can actually be carried.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.how_a_place_looks import look_of
from core.perception.observed_transfer import (
    Box,
    PixelSnapshot,
    _box,
    _distinctive,
    _patch,
    freeze_screen,
    make_snapshot,
)
from core.perception.picture_arithmetic import pieces, shrink
from core.verify.invariants import invariant

MAX_EDGE = 480
MAX_COMPONENTS = 256
MAX_OBJECTS = 32
MIN_EDGE = 4
MAX_OBJECT_SHARE = .12


@dataclass(frozen=True)
class _Drawing:
    box: tuple[int, int, int, int]
    occupied: int


def _contains(outer: tuple, inner: tuple) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[0] + outer[2] >= inner[0] + inner[2]
            and outer[1] + outer[3] >= inner[1] + inner[3])


def _area(box: tuple) -> float:
    return box[2] * box[3]


def _intersection(a: tuple, b: tuple) -> float:
    return (max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            * max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])))


def _region_box(region: Any, wide: int, high: int) -> tuple[int, int, int, int] | None:
    if not isinstance(region, Mapping):
        return None
    try:
        values = tuple(float(region[key]) for key in ("x", "y", "width", "height"))
    # not a failure: a region that is not four numbers is none
    except (KeyError, ValueError, TypeError, OverflowError):
        return None
    x, y, w, h = values
    if not all(math.isfinite(v) for v in values) or min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
        return None
    left, top, right, bottom = round(x * wide), round(y * high), round((x + w) * wide), round((y + h) * high)
    return left, top, max(1, right - left), max(1, bottom - top)


def _components(small: np.ndarray) -> list[_Drawing]:
    from scipy import ndimage

    from core.perception.picture_arithmetic import middle_values

    # Each channel contributes: equal-brightness objects of different colours
    # must still have boundaries. Closing joins only one-pixel outline gaps.
    values = small.astype(np.int16)
    edge = np.zeros(small.shape[:2], dtype=bool)
    edge[:, 1:] |= np.max(np.abs(values[:, 1:] - values[:, :-1]), axis=2) >= 24
    edge[1:, :] |= np.max(np.abs(values[1:] - values[:-1]), axis=2) >= 24
    around = middle_values(small, (9, 9)).astype(np.int16)
    contrast = np.max(np.abs(values - around), axis=2) >= 24
    masks = (ndimage.binary_closing(edge, iterations=1),
             ndimage.binary_closing(contrast, iterations=1))
    found: dict[tuple[int, int, int, int], _Drawing] = {}
    for pass_index, mask in enumerate(masks):
        count, _, stats, _ = pieces(mask)
        if count - 1 > MAX_COMPONENTS:
            continue
        for left, top, w, h, occupied in stats[1:]:
            if w < MIN_EDGE or h < MIN_EDGE or occupied < 6 or occupied / (w * h) < .025:
                continue
            box = int(left), int(top), int(w), int(h)
            if pass_index and any(_contains(existing, box) for existing in found):
                continue  # local contrast must not split an already measured outline
            found[box] = _Drawing(box, int(occupied))
    return list(found.values())


def _separate(drawings: Sequence[_Drawing]) -> list[_Drawing]:
    """Deduplicate nearly equal extents; abstain on unresolved partial overlap."""
    unique: list[_Drawing] = []
    for drawing in sorted(drawings, key=lambda item: (-item.occupied, item.box)):
        if any(_intersection(drawing.box, kept.box) / max(_area(drawing.box), _area(kept.box)) >= .8
               for kept in unique):
            continue
        unique.append(drawing)
    ambiguous: set[tuple] = set()
    for index, first in enumerate(unique):
        for second in unique[index + 1:]:
            share = _intersection(first.box, second.box) / min(_area(first.box), _area(second.box))
            if share > .15 and not _contains(first.box, second.box) and not _contains(second.box, first.box):
                ambiguous.update((first.box, second.box))
    return [drawing for drawing in unique if drawing.box not in ambiguous]


def candidate_witness(picture: PixelSnapshot, box: Box) -> str:
    """The measured extent and local appearance, without assigning persistent identity."""
    appearance = _patch(picture, box)
    if not _distinctive(appearance):
        return ""
    return hashlib.sha256(repr((box, picture.width, picture.height)).encode("ascii") + appearance).hexdigest()


def measured_objects(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Verify each candidate against the exact private capture that owns it."""
    screen = freeze_screen(observation)
    if (screen.picture is None or not screen.surface or screen.bounds is None or screen.capture_at is None
            or not screen.epoch or not screen.settled or screen.occluded):
        return {}
    raw = observation.get("objects") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return {}
    found: dict[str, dict[str, Any]] = {}
    duplicate: set[str] = set()
    for item in raw[:MAX_OBJECTS]:
        if (not isinstance(item, Mapping) or item.get("role") != "object" or item.get("source") != "measured_pixels"
                or item.get("capture_epoch") != screen.epoch or (box := _box(item)) is None):
            continue
        name = " ".join(str(item.get("text") or "").split())
        witness = candidate_witness(screen.picture, box)
        center = item.get("center_x"), item.get("center_y")
        expected = box[0] + box[2] / 2, box[1] + box[3] / 2
        if (not name or not witness or witness != item.get("candidate_id")
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or abs(value - expected[index]) > 1e-8 for index, value in enumerate(center))):
            continue
        if name in found:
            duplicate.add(name)
        else:
            found[name] = dict(item)
    return {name: item for name, item in found.items() if name not in duplicate}


def drawn_objects(picture: Any, *, writing: Sequence[Mapping[str, Any]] = (),
                  controls: Sequence[Mapping[str, Any]] = (), at_most: int = MAX_OBJECTS,
                  capture_epoch: str = "") -> list[dict[str, Any]]:
    """Bounded anonymous extents and local witnesses from this picture only.

    An enclosing frame is a container, not a pickup target. Writing can label
    a nearby drawing but cannot create its extent. No drawing is declared
    draggable from its appearance.
    """
    snapshot = make_snapshot(picture)
    budget = max(0, min(MAX_OBJECTS, int(at_most)))
    if snapshot is None or min(snapshot.height, snapshot.width) < 20 or not budget:
        return []
    pixels = np.frombuffer(snapshot.data, np.uint8).reshape(snapshot.height, snapshot.width, 3)
    scale = min(1., MAX_EDGE / max(snapshot.width, snapshot.height))
    wide, high = max(1, round(snapshot.width * scale)), max(1, round(snapshot.height * scale))
    small = pixels if (wide, high) == (snapshot.width, snapshot.height) else shrink(pixels, wide, high)
    from core.perception.a_picture_read_once import read_once

    drawings = read_once("drawn", small, lambda: _separate(_components(small)))
    containers = [drawing.box for drawing in drawings
                  if _area(drawing.box) > MAX_OBJECT_SHARE * wide * high
                  or (drawing.occupied / _area(drawing.box) < .18
                      and sum(_contains(drawing.box, other.box) for other in drawings if other != drawing) >= 2)]
    excluded = [box for region in (*tuple(writing or ()), *tuple(controls or ()))
                if (box := _region_box(region, wide, high)) is not None]
    # A measured text extent inside a button disqualifies that button as an
    # anonymous source. An adjacent caption does not erase the drawing.
    drawings = [drawing for drawing in drawings
                if not any(_contains(drawing.box, box)
                           or _intersection(drawing.box, box) / min(_area(drawing.box), _area(box)) >= .25
                           for box in excluded)]
    candidates = [drawing for drawing in drawings if drawing.box not in containers
                  and not any(other.box != drawing.box and other.box not in containers
                              and _contains(other.box, drawing.box) for other in drawings)
                  and .08 <= drawing.box[2] / drawing.box[3] <= 12
                  and _area(drawing.box) <= MAX_OBJECT_SHARE * wide * high
                  and drawing.box[0] > 1 and drawing.box[1] > 1
                  and drawing.box[0] + drawing.box[2] < wide - 1
                  and drawing.box[1] + drawing.box[3] < high - 1]
    out: list[dict[str, Any]] = []
    for drawing in sorted(candidates, key=lambda item: (item.box[1], item.box[0])):
        left, top, w, h = drawing.box
        # One measured background pixel around an outline makes a solid
        # object's appearance distinguishable from a flat colour patch.
        box: Box = ((left - 1) / wide, (top - 1) / high, (w + 2) / wide, (h + 2) / high)
        witness = candidate_witness(snapshot, box)
        if not witness:
            continue
        x, y, width, height = box
        cx, cy = x + width / 2, y + height / 2
        enclosing = [container for container in containers if _contains(container, drawing.box)]
        container = min(enclosing, key=_area) if enclosing else None
        name = f"the object at {cx * 100:.2f}% across, {cy * 100:.2f}% down"
        if any(region["text"] == name for region in out):
            # Concentric candidates must not share a dispatch name.
            continue
        region = {"text": name, "x": x, "y": y, "width": width, "height": height,
                  "center_x": cx, "center_y": cy, "role": "object", "source": "measured_pixels",
                  "candidate_id": witness, "capture_epoch": str(capture_epoch),
                  **({"container_bbox": [container[0] / wide, container[1] / high,
                                         container[2] / wide, container[3] / high]} if container else {})}
        region["look"] = look_of(pixels, region)
        out.append(region)
        if len(out) >= budget:
            break
    return out


@invariant("perception.object_candidates_require_measured_extents", scope="perception",
           owner="core/perception/drawn_objects.py", observational=False)
def _object_candidates_invariant() -> tuple:
    from core.perception.observed_transfer import ScreenReading

    image = np.zeros((80, 120, 3), np.uint8)
    image[20:40, 20:40] = (230, 70, 50)
    found = drawn_objects(image, capture_epoch="one")
    assert found and all(item["role"] == "object" and item["candidate_id"] and item["capture_epoch"] == "one" for item in found)
    assert not drawn_objects(np.zeros_like(image)), "a flat background supplied pickup targets"
    assert not drawn_objects(image, writing=[{"x": .1, "y": .1, "width": .5, "height": .5}]), "writing supplied an object"
    reading = ScreenReading({"objects": found, "surface_id": "editor", "bounds": [0, 0, 120, 80],
                             "capture_at": 1., "_capture_epoch": "one", "settled": True},
                            picture_snapshot=make_snapshot(image))
    assert measured_objects(reading), "a measured current candidate was discarded"
    reading["_capture_epoch"] = "another"
    assert not measured_objects(reading), "a pickup extent borrowed a different capture"
    return ()
