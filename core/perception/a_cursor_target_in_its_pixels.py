"""Keep the visible material at a physically sampled hand cursor through a move.

The cursor supplies an affordance, not an object name. A small image reference
keeps that particular material distinct from other nearby affordances. Each
correction uses a new picture; loss or competing copies stop the attempt.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.a_picture_of_her_page import picture_of
from core.perception.a_visual_reference_in_the_picture import reference_appearances
from core.perception.recording_the_frames_and_inputs import (
    record_pointer_correspondence,
    record_pointer_reference,
)
from core.perception.where_the_pointer_turns import shown_as_a_hand
from core.verify import invariant

logger = logging.getLogger(__name__)
TARGET_WITHIN_S = 3.0
MAX_CORRECTIONS = 6
MAX_AGE_S = 60.0


@dataclass(frozen=True)
class CursorTarget:
    page_id: int
    address: str
    viewport: tuple[int, int]
    bounds: tuple[float, float, float, float]
    patch: bytes
    size: tuple[int, int]
    anchor: tuple[float, float]
    sample_point: tuple[float, float]
    captured: float


def _scope(page: Any) -> tuple | None:
    size = getattr(page, 'viewport_size', None) or {}
    width, height = size.get('width'), size.get('height')
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        return None
    return id(page), str(getattr(page, 'url', '')), (width, height)


def _belongs(target: CursorTarget, page: Any) -> bool:
    return (_scope(page) == (target.page_id, target.address, target.viewport)
            and 0 <= time.monotonic()-target.captured <= MAX_AGE_S)


async def target_picture(page: Any, bounds: tuple) -> np.ndarray | None:
    """RGB pixels in the supplied CSS rectangle, without changing the viewport."""
    scope = _scope(page)
    if scope is None or len(bounds) != 4:
        return None
    x, y, width, height = bounds
    if (not all(math.isfinite(v) for v in bounds) or min(width, height) < 24
            or min(x, y) < 0 or x+width > scope[2][0] or y+height > scope[2][1]):
        return None
    pixels = await picture_of(page, dict(x=x, y=y, width=width, height=height), css=True, kind='png')
    if (_scope(page) != scope or not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
            or pixels.shape != (round(height), round(width), 3)):
        return None
    return pixels


def material_at(picture: np.ndarray, point: tuple[float, float], *, page: Any, bounds: tuple) -> CursorTarget | None:
    """Bind a sampled location to its patch; the caller must first witness a hand there."""
    scope = _scope(page)
    if (scope is None or not isinstance(picture, np.ndarray) or picture.dtype != np.uint8
            or picture.ndim != 3 or picture.shape[2] != 3 or min(picture.shape[:2]) < 24
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in point)):
        return None
    height, width = picture.shape[:2]
    px, py = point[0]*width, point[1]*height
    left, top = max(0, min(width-96, round(px)-48)), max(0, min(height-96, round(py)-48))
    patch = np.ascontiguousarray(picture[top:min(height, top+96), left:min(width, left+96)])
    return CursorTarget(*scope, tuple(bounds), patch.tobytes(), (patch.shape[1], patch.shape[0]),
                        (px-left, py-top), (point[0]*bounds[2], point[1]*bounds[3]), time.monotonic())


def target_in_picture(target: CursorTarget, picture: np.ndarray) -> tuple[float, float] | None:
    """One redundant appearance correspondence, in CSS coordinates of the picture."""
    width, height = target.size
    reference = np.empty((height, width, 4), np.uint8)
    # A fractional pan resamples the original pixels. Compare the same bounded
    # spatial bandwidth on both sides before the existing point checks.
    original = np.frombuffer(target.patch, np.uint8).reshape(height, width, 3)
    reference[..., :3] = ndimage.gaussian_filter(original.astype(float), (.8, .8, 0)).astype(np.uint8)
    reference[..., 3] = 255
    picture = ndimage.gaussian_filter(picture.astype(float), (.8, .8, 0)).astype(np.uint8)
    ratio = picture.shape[1]/target.bounds[2]
    if abs(picture.shape[0]/target.bounds[3]-ratio) > .005:
        return None
    poses = reference_appearances(picture, reference, scales=(ratio,), edge=512)
    if len(poses) != 1:
        return None
    x, y = poses[0].project(target.anchor)
    if not (0 <= x < picture.shape[1] and 0 <= y < picture.shape[0]):
        return None
    return float(x/ratio), float(y/ratio)


async def sample_cursor_target(page: Any, point: tuple[float, float], bounds: tuple) -> CursorTarget | None:
    """A physical hover and its subsequent picture, with a hand on both sides."""
    if _scope(page) is None or not all(math.isfinite(v) and 0 <= v <= 1 for v in point):
        return None
    x, y = bounds[0]+point[0]*bounds[2], bounds[1]+point[1]*bounds[3]
    await page.mouse.move(x, y)
    await asyncio.sleep(.12)
    return await cursor_material_here(page, point, bounds)


async def cursor_material_here(page: Any, point: tuple[float, float], bounds: tuple) -> CursorTarget | None:
    """Capture the current physical hover before any further pointer movement."""
    scope = _scope(page)
    x, y = bounds[0]+point[0]*bounds[2], bounds[1]+point[1]*bounds[3]
    if not await shown_as_a_hand(page, x, y, move=False):
        return None
    picture = await target_picture(page, bounds)
    if (picture is None or scope != _scope(page)
            or not await shown_as_a_hand(page, x, y, move=False)):
        return None
    return material_at(picture, point, page=page, bounds=bounds)


class PointingFeedback:
    """Estimate how pointer movement changes the target-minus-pointer error."""

    def __init__(self):
        self.response = np.eye(2)
        self.previous: tuple[np.ndarray, np.ndarray] | None = None

    def correct(self, pointer: tuple, target: tuple, limit: float) -> tuple[float, float] | None:
        pointer, target = np.asarray(pointer, float), np.asarray(target, float)
        if (pointer.shape != (2,) or target.shape != (2,) or not np.isfinite([pointer, target]).all()
                or not math.isfinite(limit) or limit <= 0):
            return None
        error = target-pointer
        if self.previous is not None:
            old_pointer, old_error = self.previous
            moved = pointer-old_pointer
            if float(moved @ moved) > 1.:
                # d(error) = -response * d(pointer), including a moving view.
                residual = old_error-error-self.response @ moved
                self.response += np.outer(residual, moved)/float(moved @ moved)
        self.previous = pointer.copy(), error.copy()
        if np.linalg.cond(self.response) > 20.:
            return None
        step = np.linalg.solve(self.response, error)
        step *= min(1., limit/max(1e-9, float(np.linalg.norm(step))))
        return tuple(float(v) for v in pointer+step)


async def _arrive(page: Any, target: CursorTarget, point: tuple, check: Any) -> tuple[float, float] | None:
    pointer = np.asarray((point[0]*target.bounds[2], point[1]*target.bounds[3]), float)
    feedback = PointingFeedback()
    reference = record_pointer_reference(target)
    left, top, width, height = target.bounds
    for _ in range(MAX_CORRECTIONS):
        check()
        if not _belongs(target, page) or not (0 <= pointer[0] < width and 0 <= pointer[1] < height):
            return None
        await page.mouse.move(left+float(pointer[0]), top+float(pointer[1]))
        await asyncio.sleep(.12)
        pixels = await target_picture(page, target.bounds)
        if pixels is None or not _belongs(target, page):
            return None
        observed = await asyncio.to_thread(target_in_picture, target, pixels)
        record_pointer_correspondence(reference, pixels, phase='correct', pointer=pointer, matched=observed)
        if observed is None:
            logger.info('pointer target has no unique current pixel correspondence')
            return None
        logger.info('pointer material feedback: pointer=(%.2f, %.2f), target=(%.2f, %.2f)',
                    *pointer, *observed)
        if math.dist(observed, pointer) <= 1.5:
            at = (left+float(pointer[0]), top+float(pointer[1]))
            if await shown_as_a_hand(page, *at, move=False) and _belongs(target, page):
                return at
        corrected = feedback.correct(tuple(pointer), observed, .25*min(width, height))
        if corrected is None:
            return None
        pointer = np.asarray(corrected)
    return None


async def arrive_at_cursor_target(page: Any, target: CursorTarget, point: tuple, *, check: Any) -> tuple | None:
    """Move through observed feedback; no button is pressed by this procedure."""
    try:
        return await asyncio.wait_for(_arrive(page, target, point, check), TARGET_WITHIN_S)
    except TimeoutError:
        logger.info('pointer target did not converge within %.1fs', TARGET_WITHIN_S)
        return None
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as why:
        logger.info('pointer feedback could not complete: %s', str(why)[:120])
        return None
    except Exception as why:  # noqa: BLE001 - optional browser SDK errors have no shared runtime base
        if not type(why).__module__.startswith('playwright.'):
            raise
        logger.info('the browser could not complete pointer feedback: %s', str(why)[:120])
        return None


@invariant('perception.pointer_feedback_needs_a_distinct_current_target', scope='perception',
           owner='core/perception/a_cursor_target_in_its_pixels.py', observational=False)
def _pointer_feedback_invariant() -> tuple:
    feedback = PointingFeedback()
    assert feedback.correct((0., 0.), (10., 0.), 20.) == (10., 0.)
    assert feedback.correct((10., 0.), (20., 0.), 20.) is None, 'an unreachable target supplied an inverse'
    assert PointingFeedback().correct((math.nan, 0.), (0., 0.), 20.) is None
    return ()
