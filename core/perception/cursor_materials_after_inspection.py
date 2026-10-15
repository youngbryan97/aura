"""Refresh a page reading after active inspection and relocate sampled material."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import numpy as np

from core.perception.a_cursor_target_in_its_pixels import (
    _scope,
    cursor_material_here,
    sample_cursor_target,
    target_in_picture,
)
from core.perception.observed_transfer import ScreenReading, freeze_screen
from core.perception.recording_the_frames_and_inputs import (
    record_pointer_correspondence,
    record_pointer_reference,
)
from core.perception.what_her_page_shows import look_at_a_page
from core.perception.where_the_pointer_turns import _A_HAND

logger = logging.getLogger(__name__)
MAX_TARGETS = 4


class CursorMaterialCollector:
    """Acquire a bounded set inside the original physical sample's custody."""

    def __init__(self, page: Any, bounds: tuple):
        self.page, self.bounds, self.scope = page, bounds, _scope(page)
        self.targets: list = []

    async def sampled(self, x: float, y: float, shown: str) -> None:
        if shown not in _A_HAND or len(self.targets) >= MAX_TARGETS or _scope(self.page) != self.scope:
            return
        left, top, width, height = self.bounds
        point = ((x-left)/width, (y-top)/height)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in point):
            return
        if any(math.dist((x-left, y-top), target.sample_point) < 16 for target in self.targets):
            return
        target = await cursor_material_here(self.page, point, self.bounds)
        if target is not None:
            self.targets.append(target)


def collector_for_reading(page: Any, seen: dict, bounds: tuple) -> CursorMaterialCollector | None:
    frame = freeze_screen(seen)
    return (CursorMaterialCollector(page, bounds) if isinstance(seen, ScreenReading) and frame.picture is not None
            and frame.capture_at is not None and frame.epoch else None)


def _unavailable(seen: dict) -> tuple:
    seen.update(ok=False, text='', layout=[], shapes=[], error='the picture changed during cursor inspection')
    if isinstance(seen, ScreenReading):
        seen.picture_snapshot = None
    return (), ()


async def cursor_materials_after_inspection(page: Any, seen: dict, places: tuple, bounds: tuple, *,
                                          over: Any, name: str, sampled_targets: tuple | None = None) -> tuple[tuple, tuple]:
    """Return current locations with private material references, or no authority.

    Legacy readers without a measured capture retain their cursor-only path.
    Measured page readers require a new capture after the pointer sweep: each
    sampled patch must have a unique correspondence in that final picture.
    """
    frame = freeze_screen(seen)
    if frame.picture is None or frame.capture_at is None or not frame.epoch:
        return places, ()
    scope = _scope(page)
    sampled = list(sampled_targets) if sampled_targets is not None else []
    if sampled_targets is None:
        for point in places[:MAX_TARGETS]:
            target = await sample_cursor_target(page, point, bounds)
            if target is not None:
                sampled.append(target)
    latest = await look_at_a_page(page, over, name=name, wait_for_stillness=False)
    if (not isinstance(latest, ScreenReading) or latest.get('ok', True) is not True
            or scope != _scope(page)):
        return _unavailable(seen)
    after = freeze_screen(latest)
    if (after.picture is None or after.bounds != tuple(bounds) or after.surface != frame.surface
            or after.capture_at is None or after.capture_at <= frame.capture_at or not after.epoch):
        return _unavailable(seen)
    # Page readers explicitly consume BGR; browser captures and this reference
    # matcher consume RGB. Keep that conversion at this provider boundary.
    pixels = np.frombuffer(after.picture.data, np.uint8).reshape(after.picture.height, after.picture.width, 3)
    pixels = np.ascontiguousarray(pixels[..., ::-1])
    current = []
    for target in sampled:
        reference = record_pointer_reference(target)
        point = await asyncio.to_thread(target_in_picture, target, pixels)
        record_pointer_correspondence(reference, pixels, phase='after_inspection', matched=point)
        if point is not None:
            current.append(((point[0]/bounds[2], point[1]/bounds[3]), target))
    seen.clear()
    seen.update(latest)
    seen.setdefault('ok', True)
    seen.__dict__.clear()
    seen.__dict__.update(latest.__dict__)
    logger.info('cursor material after active inspection: %d samples, %d current correspondences', len(sampled), len(current))
    return tuple(point for point, _target in current), tuple(target for _point, target in current)
