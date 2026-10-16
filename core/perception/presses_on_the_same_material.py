"""Remember delivered presses at corresponding local material points.

This supplies an exploration cost, not object identity or a scene transition.
Names and coordinates can change while the visible material stays the same.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.a_cursor_target_in_its_pixels import CursorTarget, MAX_AGE_S
from core.perception.a_visual_reference_in_the_picture import reference_appearances
from core.perception.observed_transfer import ScreenReading, freeze_screen
from core.verify import invariant

MOST_REFERENCES = 8
MOST_PRESSES = 32
SAME_POINT_WITHIN_PX = 2.
MAX_HISTORY_AGE_S = 300.


def _scope(target: CursorTarget) -> tuple:
    return target.page_id, target.address, target.viewport, target.bounds


def _usable(target: Any, *, current: bool = True) -> bool:
    return (isinstance(target, CursorTarget) and type(target.captured) in (int, float)
            and 0 <= time.monotonic()-target.captured <= (MAX_AGE_S if current else MAX_HISTORY_AGE_S)
            and type(target.page_id) is int and target.page_id > 0 and isinstance(target.address, str)
            and isinstance(target.viewport, tuple) and len(target.viewport) == 2
            and all(type(v) is int and v > 0 for v in target.viewport)
            and isinstance(target.bounds, tuple) and len(target.bounds) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in target.bounds)
            and min(target.bounds[:2]) >= 0 and min(target.bounds[2:]) > 0
            and isinstance(target.size, tuple) and len(target.size) == 2
            and all(type(v) is int and 24 <= v <= 96 for v in target.size)
            and isinstance(target.patch, bytes) and len(target.patch) == 3*math.prod(target.size)
            and isinstance(target.anchor, tuple) and len(target.anchor) == 2
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < extent
                    for v, extent in zip(target.anchor, target.size, strict=True)))


def _same_point(old: CursorTarget, new: CursorTarget) -> bool:
    """Require one redundant local correspondence at the prior press point."""
    if _scope(old) != _scope(new):
        return False
    pictures = [ndimage.gaussian_filter(np.frombuffer(t.patch, np.uint8).reshape(t.size[1], t.size[0], 3)
                                       .astype(float), (.8, .8, 0)).astype(np.uint8) for t in (old, new)]
    reference = np.full((*pictures[0].shape[:2], 4), 255, np.uint8)
    reference[..., :3] = pictures[0]
    poses = reference_appearances(pictures[1], reference, scales=(1.,), edge=128)
    return len(poses) == 1 and math.dist(poses[0].project(old.anchor), new.anchor) <= SAME_POINT_WITHIN_PX


@dataclass
class MaterialPressHistory:
    """A bounded, local record; delivery is not accepted action or progress."""

    entries: tuple[tuple[CursorTarget, int, float], ...] = field(default=(), repr=False)

    def count(self, target: CursorTarget) -> int | None:
        if not _usable(target):
            return None
        now = time.monotonic()
        matches = [count for old, count, delivered in self.entries
                   if 0 <= now-delivered <= MAX_HISTORY_AGE_S and _usable(old, current=False) and _same_point(old, target)]
        return matches[0] if len(matches) == 1 else 0 if not matches else None

    def delivered(self, target: CursorTarget) -> None:
        if not _usable(target):
            return
        now = time.monotonic()
        recent = [(old, count, at) for old, count, at in self.entries
                  if _scope(old) == _scope(target) and 0 <= now-at <= MAX_HISTORY_AGE_S and _usable(old, current=False)]
        matched = [i for i, (old, _count, _at) in enumerate(recent) if _same_point(old, target)]
        count = min(MOST_PRESSES, recent[matched[0]][1]+1) if len(matched) == 1 else 1
        if len(matched) == 1:
            recent.pop(matched[0])
        self.entries = tuple([*recent, (target, count, now)][-MOST_REFERENCES:])


@dataclass(frozen=True)
class MaterialPressesInReading:
    """Costs attached privately to this capture and its current point labels."""

    frame: tuple = field(repr=False)
    counts: tuple[tuple[str, int], ...]


def _capture(reading: ScreenReading) -> tuple | None:
    frame = freeze_screen(reading)
    if frame.picture is None or frame.capture_at is None or not frame.epoch or frame.bounds is None or not frame.surface:
        return None
    return frame.picture, frame.surface, frame.bounds, frame.capture_at, frame.epoch, _anonymous_points(reading)


def _anonymous_points(reading: dict) -> tuple:
    return tuple(sorted((region['text'], region['center_x'], region['center_y']) for region in reading.get('shapes') or ()
                     if isinstance(region, dict) and region.get('answers_the_pointer') is True
                     and isinstance(region.get('text'), str)
                     and region['text'].startswith('the one that stands out at ')
                     and all(type(v := region.get(key)) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                             for key in ('center_x', 'center_y'))))


def bind_material_presses(reading: dict, counts: dict[str, int]) -> None:
    if not isinstance(reading, ScreenReading):
        return
    reading.material_presses = None
    frame = _capture(reading)
    if frame is not None:
        labels = {label for label, _x, _y in _anonymous_points(reading)}
        qualified = tuple((label, n) for label, n in counts.items()
                          if label in labels and type(n) is int and 0 < n <= MOST_PRESSES)
        reading.material_presses = MaterialPressesInReading(frame, qualified)


def current_material_presses(reading: Any) -> dict[str, int]:
    """A public dictionary cannot copy a private capture's exploration costs."""
    held = getattr(reading, 'material_presses', None)
    if (not isinstance(reading, ScreenReading) or not isinstance(held, MaterialPressesInReading)
            or _capture(reading) != held.frame):
        return {}
    labels = {label for label, _x, _y in _anonymous_points(reading)}
    return {f'click "{label}"': count for label, count in held.counts if label in labels}


@invariant('agency.material_press_cost_stays_with_its_capture', scope='agency',
           owner='core/perception/presses_on_the_same_material.py', observational=False)
def _material_press_invariant() -> tuple:
    label = 'the one that stands out at 40% across, 60% down'
    assert current_material_presses({'material_presses': {label: 9}}) == {}
    assert MaterialPressHistory().count(None) is None
    return ()
