"""A bounded place to sample belongs to one captured RGB frame.

The hint carries no object name, object mask, correspondence, or input
authority. Matching its frame permits only a share of point sampling.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.verify import invariant

__all__ = ["PixelSearchHint", "SamplingHintReceipt", "frame_priority_mask", "make_search_hint", "sampling_hint_matches"]

MAX_RGB_BYTES = 32 * 1024 * 1024
_SOURCES = frozenset({"canvas-bracket", "browser-monotonic", "browser-wall", "screenshot-bracket"})


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _capture(at: Any) -> tuple | None:
    if not isinstance(at, MeasuredFrameTime):
        return None
    surface, sequence = getattr(at, "surface", None), getattr(at, "sequence", None)
    if (not isinstance(surface, str) or not 0 < len(surface) <= 256
            or type(at.epoch) is not int or at.epoch <= 0 or type(sequence) is not int or sequence <= 0
            or not isinstance(at.source, str) or at.source not in _SOURCES
            or not all(_finite(v) for v in (at.lower, at.upper, at.arrived))
            or not math.isfinite(at) or float(at) != at.lower or not 0 <= at.lower <= at.upper <= at.arrived
            or at.source_at is not None and (not _finite(at.source_at) or at.source_at < 0)):
        return None
    return surface, at.epoch, sequence, at.lower, at.upper, at.arrived, at.source, at.source_at


def _rgb(picture: Any) -> np.ndarray | None:
    if (not isinstance(picture, np.ndarray) or picture.dtype != np.uint8 or picture.ndim != 3
            or picture.shape[2] != 3 or min(picture.shape[:2]) < 16 or picture.nbytes > MAX_RGB_BYTES):
        return None
    return np.ascontiguousarray(picture)


def _region(region: Any) -> bool:
    return (type(region) is tuple and len(region) == 4 and all(_finite(v) for v in region)
            and 0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1
            and (region[2] - region[0]) * (region[3] - region[1]) <= .5)


@dataclass(frozen=True, slots=True)
class PixelSearchHint:
    surface: str
    epoch: int
    sequence: int
    lower: float
    upper: float
    arrived: float
    source: str
    source_at: float | None
    shape: tuple[int, int, int]
    digest: str
    region: tuple[float, float, float, float]

    def valid(self) -> bool:
        return (isinstance(self.surface, str) and 0 < len(self.surface) <= 256
                and type(self.epoch) is int and self.epoch > 0 and type(self.sequence) is int and self.sequence > 0
                and all(_finite(v) for v in (self.lower, self.upper, self.arrived))
                and 0 <= self.lower <= self.upper <= self.arrived
                and isinstance(self.source, str) and self.source in _SOURCES
                and (self.source_at is None or _finite(self.source_at) and self.source_at >= 0)
                and type(self.shape) is tuple and len(self.shape) == 3
                and all(type(v) is int for v in self.shape) and min(self.shape[:2]) >= 16
                and self.shape[2] == 3 and math.prod(self.shape) <= MAX_RGB_BYTES
                and isinstance(self.digest, str) and len(self.digest) == 64
                and all(c in "0123456789abcdef" for c in self.digest) and _region(self.region))


@dataclass(frozen=True, slots=True)
class SamplingHintReceipt:
    """The actual sampler's frame owner and counts after its point selection."""

    surface: str
    epoch: int
    sequence: int
    lower: float
    upper: float
    arrived: float
    source: str
    source_at: float | None
    shape: tuple[int, int, int]
    digest: str
    region: tuple[float, float, float, float]
    consumed: bool
    priority_pixels: int
    fresh_priority_points: int
    kept_priority_points: int
    background_points: int


def sampling_hint_matches(receipt: Any, hint: Any) -> bool:
    """An observer result proves sampling only for the same consumed hint."""
    if not isinstance(receipt, SamplingHintReceipt) or not isinstance(hint, PixelSearchHint) or not hint.valid():
        return False
    owner = PixelSearchHint(receipt.surface, receipt.epoch, receipt.sequence, receipt.lower, receipt.upper,
                            receipt.arrived, receipt.source, receipt.source_at, receipt.shape, receipt.digest, receipt.region)
    return (owner.valid() and owner == hint and receipt.consumed is True
            and all(type(v) is int for v in (receipt.priority_pixels, receipt.fresh_priority_points,
                                            receipt.kept_priority_points, receipt.background_points))
            and 0 < receipt.priority_pixels <= math.prod(hint.shape[:2]) // 2
            and 0 <= receipt.fresh_priority_points <= receipt.kept_priority_points <= 384
            and 0 <= receipt.background_points <= 384 - receipt.kept_priority_points)


def make_search_hint(picture: np.ndarray, at: MeasuredFrameTime, *,
                     normalized_region: tuple[float, float, float, float]) -> PixelSearchHint | None:
    """Bind a sampling region to the exact original frame; unavailable custody stays unknown."""
    capture, rgb = _capture(at), _rgb(picture)
    if capture is None or rgb is None or not _region(normalized_region):
        return None
    return PixelSearchHint(*capture, tuple(rgb.shape), hashlib.sha256(rgb.tobytes()).hexdigest(), normalized_region)


def frame_priority_mask(hint: Any, picture: np.ndarray, at: MeasuredFrameTime) -> np.ndarray | None:
    """Only the same frame can admit a region into point sampling."""
    if not isinstance(hint, PixelSearchHint) or not hint.valid():
        return None
    capture, rgb = _capture(at), _rgb(picture)
    if (capture is None or rgb is None or type(hint.shape) is not tuple or hint.shape != rgb.shape
            or capture != (hint.surface, hint.epoch, hint.sequence, hint.lower, hint.upper,
                           hint.arrived, hint.source, hint.source_at)
            or not isinstance(hint.digest, str) or hint.digest != hashlib.sha256(rgb.tobytes()).hexdigest()):
        return None
    height, width = rgb.shape[:2]
    left, top, right, bottom = hint.region
    x0, y0, x1, y1 = int(math.ceil(left * width)), int(math.ceil(top * height)), int(math.floor(right * width)), int(math.floor(bottom * height))
    if x0 >= x1 or y0 >= y1:
        return None
    priority = np.zeros((height, width), bool)
    priority[y0:y1, x0:x1] = True
    return priority


@invariant("perception.pixel_search_hints_belong_to_the_exact_capture", scope="perception",
           owner="core/perception/a_search_hint_for_these_pixels.py", observational=False)
def _pixel_search_hint_invariant() -> tuple:
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime

    picture = np.zeros((24, 32, 3), np.uint8)
    at = CanvasFrameTime(1., lower=1., upper=1.01, arrived=1.02, source="canvas-bracket",
                         source_at=2., epoch=3, surface="document/canvas", sequence=4)
    hint = make_search_hint(picture, at, normalized_region=(0., 0., .5, .5))
    assert hint is not None and np.count_nonzero(frame_priority_mask(hint, picture, at)) == 192
    changed = picture.copy()
    changed[0, 0, 0] = 1
    assert frame_priority_mask(hint, changed, at) is None
    assert frame_priority_mask(hint, picture, float(at)) is None
    return ()
