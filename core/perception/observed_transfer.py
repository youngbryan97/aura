"""Bind a delivered transfer to a fresh, measured change in the picture.

Receipts retain a small patch and its place. A delivery acknowledgement,
recognized words, or a changing screen cannot stand in for that observation.
"""
from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

import numpy as np

from core.runtime.skill_contract import PredicateState
from core.verify.invariants import invariant

MAX_EDGE = 960
PATCH_EDGE = 16
MAX_REGIONS = 128
APPEARANCE_ERROR = 0.055
APPEARANCE_TAIL_ERROR = 0.18
MIN_SPATIAL_VARIATION = 0.035
MIN_LOCAL_CHANGE = 0.075
MIN_CHANGED_FRACTION = 0.15
CHANGED_PIXEL_ERROR = 0.10
MAX_SIZE_RATIO = 1.35
MIN_RETAINED_OVERLAP = 0.80

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x, y, width, height; shares of the picture
Surface = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PixelSnapshot:
    data: bytes = field(repr=False)
    height: int
    width: int
    channels: int = 3

    def __post_init__(self) -> None:
        if (not isinstance(self.data, bytes)
                or any(type(v) is not int for v in (self.height, self.width, self.channels))
                or not 1 <= self.height <= MAX_EDGE or not 1 <= self.width <= MAX_EDGE
                or self.channels != 3
                or len(self.data) != self.height * self.width * self.channels):
            raise ValueError("a pixel snapshot needs bounded, immutable three-channel bytes")


class ScreenReading(dict):
    """Public reading fields with picture bytes outside dictionary serialization."""

    def __init__(self, reading: Mapping[str, Any], *, picture_snapshot: PixelSnapshot | None = None):
        super().__init__(reading)
        self.picture_snapshot = picture_snapshot


def make_snapshot(picture: Any) -> PixelSnapshot | None:
    """Copy a uint8 picture, preserving its aspect ratio with at most 960 pixels per edge."""
    if picture is None:
        return None
    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.shape[2] < 3 or pixels.dtype != np.uint8 or not pixels.size:
        return None
    height, width = pixels.shape[:2]
    scale = min(1.0, MAX_EDGE / max(height, width))
    target_height, target_width = max(1, round(height * scale)), max(1, round(width * scale))
    rows = np.minimum(height - 1, ((np.arange(target_height) + 0.5) * height / target_height).astype(int))
    cols = np.minimum(width - 1, ((np.arange(target_width) + 0.5) * width / target_width).astype(int))
    bounded = pixels[rows[:, None], cols[None, :], :3]
    return PixelSnapshot(bounded.tobytes(order="C"), target_height, target_width)


@dataclass(frozen=True)
class ObservedRegion:
    label: str
    bbox: Box
    object_extent: bool


@dataclass(frozen=True)
class ScreenSnapshot:
    picture: PixelSnapshot | None = field(repr=False)
    regions: tuple[ObservedRegion, ...]
    surface: Surface
    bounds: tuple[float, float, float, float] | None
    capture_at: float | None
    epoch: str
    settled: bool
    work_epoch: str = ""
    occluded: bool = False


@dataclass(frozen=True)
class TransferIntent:
    attempt_id: str
    move: str
    before: ScreenSnapshot = field(repr=False)
    source: str
    destination: str
    source_at: Point | None
    destination_at: Point | None
    started: float | None = None
    completed: float | None = None
    delivered: bool | None = None


@dataclass(frozen=True)
class TransferReceipt:
    attempt_id: str
    move: str
    source: str
    destination: str
    source_at: Point | None
    destination_at: Point | None
    state: PredicateState
    effect: str
    effect_bbox: Box | None
    reason: str
    before_epoch: str
    after_epoch: str
    surface: Surface
    bounds: tuple[float, float, float, float] | None
    after_at: float | None
    work_epoch: str
    appearance: bytes = field(default=b"", repr=False)
    pixel_size: tuple[int, int] = (0, 0)

    def as_memory(self) -> dict[str, Any]:
        """Store a bounded witness; current validation is still required after loading."""
        return {
            "version": 1, "attempt_id": self.attempt_id, "move": self.move,
            "source": self.source, "destination": self.destination,
            "source_at": list(self.source_at) if self.source_at is not None else None,
            "destination_at": list(self.destination_at) if self.destination_at is not None else None,
            "state": self.state.value, "effect": self.effect,
            "effect_bbox": list(self.effect_bbox) if self.effect_bbox is not None else None,
            "reason": self.reason, "before_epoch": self.before_epoch,
            "after_epoch": self.after_epoch, "surface": [list(item) for item in self.surface],
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "after_at": self.after_at, "work_epoch": self.work_epoch,
            "appearance": base64.b64encode(self.appearance).decode("ascii"),
            "pixel_size": list(self.pixel_size),
        }

    @classmethod
    def from_memory(cls, value: Any) -> TransferReceipt | None:
        """Reject malformed witness fields, dimensions and unbounded stored bytes."""
        if not isinstance(value, Mapping) or type(value.get("version")) is not int or value.get("version") != 1:
            return None
        limits = {"attempt_id": 32, "move": 2048, "source": 512, "destination": 512,
                  "reason": 1024, "before_epoch": 256, "after_epoch": 256, "work_epoch": 256}
        if any(not isinstance(value.get(key), str) or len(value[key]) > limit for key, limit in limits.items()):
            return None
        attempt_id = value["attempt_id"]
        if len(attempt_id) != 32 or any(c not in "0123456789abcdef" for c in attempt_id):
            return None
        try:
            state = PredicateState(value.get("state"))
        except (TypeError, ValueError):
            return None
        effect = value.get("effect")
        if not isinstance(effect, str) or effect not in {"moved", "copied", "none", "unknown"}:
            return None
        raw_surface = value.get("surface")
        if not isinstance(raw_surface, list) or not 1 <= len(raw_surface) <= 5:
            return None
        surface: list[tuple[str, str]] = []
        for item in raw_surface:
            if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                    or item[0] not in {"scope", "scoped_to", "owner", "url", "surface_id"}
                    or not isinstance(item[1], str) or not item[1] or len(item[1]) > 2048
                    or any(key == item[0] for key, _ in surface)):
                return None
            surface.append((item[0], item[1]))
        raw_bounds = value.get("bounds")
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 4:
            return None
        bounds = tuple(_number(item) for item in raw_bounds)
        if any(item is None for item in bounds) or bounds[2] <= 0 or bounds[3] <= 0:
            return None
        after_at = _number(value.get("after_at"))
        if after_at is None or after_at <= 0:
            return None
        encoded = value.get("appearance")
        if not isinstance(encoded, str) or len(encoded) > PATCH_EDGE * PATCH_EDGE * 4:
            return None
        try:
            appearance = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return None
        raw_box = value.get("effect_bbox")
        box = _box(dict(zip(("x", "y", "width", "height"), raw_box, strict=True))) if isinstance(raw_box, list) and len(raw_box) == 4 else None
        source_at, destination_at = _point(value.get("source_at")), _point(value.get("destination_at"))
        if state == PredicateState.SATISFIED and (
                effect not in {"moved", "copied"} or box is None or source_at is None or destination_at is None
                or not _distinctive(appearance) or not value["before_epoch"] or not value["after_epoch"]
                or value["before_epoch"] == value["after_epoch"]):
            return None
        if appearance and len(appearance) != PATCH_EDGE * PATCH_EDGE * 3:
            return None
        pixel_size = value.get("pixel_size")
        if (not isinstance(pixel_size, list) or len(pixel_size) != 2
                or any(type(item) is not int or not 0 <= item <= MAX_EDGE for item in pixel_size)
                or (state == PredicateState.SATISFIED and any(item < 2 for item in pixel_size))):
            return None
        return cls(attempt_id, value["move"], value["source"], value["destination"], source_at,
                   destination_at, state, effect, box, value["reason"], value["before_epoch"],
                   value["after_epoch"], tuple(surface), bounds, after_at, value["work_epoch"], appearance,
                   tuple(pixel_size))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _point(value: Any) -> Point | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    x, y = (_number(v) for v in value)
    if x is None or y is None or not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return x, y


def _box(region: Mapping[str, Any]) -> Box | None:
    values = tuple(_number(region.get(key)) for key in ("x", "y", "width", "height"))
    if any(value is None for value in values):
        return None
    x, y, width, height = values
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
        return None
    return x, y, min(width, 1 - x), min(height, 1 - y)


def _label(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def freeze_screen(observation: Mapping[str, Any] | ScreenSnapshot) -> ScreenSnapshot:
    """Freeze geometry, surface, capture time and privately attached measured pixels."""
    if isinstance(observation, ScreenSnapshot):
        return observation
    picture = getattr(observation, "picture_snapshot", None)
    if picture is None:
        picture = observation.get("_picture_snapshot")
    if not isinstance(picture, PixelSnapshot):
        picture = None
    bounds = observation.get("bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        values = tuple(_number(value) for value in bounds)
        bounds = values if all(value is not None for value in values) and values[2] > 0 and values[3] > 0 else None
    else:
        bounds = None
    surface = tuple((key, str(observation[key])) for key in ("scope", "scoped_to", "owner", "url", "surface_id") if observation.get(key))
    captured = _number(observation.get("capture_at", observation.get("at")))
    epoch = str(observation.get("_capture_epoch") or (f"{captured:.9f}" if captured is not None else ""))
    regions: list[ObservedRegion] = []
    for collection in ("shapes", "layout"):
        raw = observation.get(collection) or ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        for region in raw:
            if not isinstance(region, Mapping):
                continue
            bbox = _box(region)
            if bbox is None:
                continue
            role = _label(region.get("role") or region.get("kind"))
            item = ObservedRegion(_label(region.get("text") or region.get("label")), bbox,
                                  region.get("shape") is True or role in {"object", "draggable", "item"})
            if item not in regions:
                regions.append(item)
            if len(regions) >= MAX_REGIONS:
                break
        if len(regions) >= MAX_REGIONS:
            break
    return ScreenSnapshot(picture, tuple(regions), surface, bounds, captured, epoch,
                          observation.get("settled") is True,
                          str(observation.get("_work_epoch") or ""),
                          bool(observation.get("occluded") or observation.get("obscured") or observation.get("occlusions")))


def bind_transfer(move: str, before: Mapping[str, Any] | ScreenSnapshot, *, source: str,
                  destination: str, source_at: Point | None, destination_at: Point | None) -> TransferIntent:
    """Bind this attempt before dispatch; the same command receives a new identity each time."""
    return TransferIntent(uuid4().hex, str(move), freeze_screen(before), _label(source),
                          _label(destination), _point(source_at), _point(destination_at))


def finish_delivery(intent: TransferIntent, *, started: float, completed: float,
                    delivered: bool) -> TransferIntent:
    """Return the same immutable attempt with its physical delivery interval."""
    return replace(intent, started=_number(started), completed=_number(completed),
                   delivered=delivered if isinstance(delivered, bool) else None)


def _contains(box: Box, point: Point) -> bool:
    x, y, width, height = box
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def _overlap(a: Box, b: Box) -> float:
    width = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = width * height
    return intersection / max(1e-12, a[2] * a[3] + b[2] * b[3] - intersection)


def _source_region(intent: TransferIntent) -> tuple[ObservedRegion | None, str]:
    named = [region for region in intent.before.regions if region.label == intent.source
             and (intent.source_at is None or _contains(region.bbox, intent.source_at))]
    if not named:
        return None, "the named source has no measured extent at the dispatched point"
    extents: list[ObservedRegion] = []
    for named_region in named:
        if named_region.object_extent:
            candidates = [named_region]
        else:
            x, y, width, height = named_region.bbox
            candidates = [region for region in intent.before.regions if region.object_extent
                          and region.bbox[0] <= x and region.bbox[1] <= y
                          and region.bbox[0] + region.bbox[2] >= x + width
                          and region.bbox[1] + region.bbox[3] >= y + height]
            if candidates:
                smallest = min(region.bbox[2] * region.bbox[3] for region in candidates)
                candidates = [region for region in candidates if region.bbox[2] * region.bbox[3] <= smallest * 1.01]
        for region in candidates:
            if region not in extents:
                extents.append(region)
    if len(extents) != 1:
        return None, "the source object extent is missing or ambiguous"
    return extents[0], ""


def _pixels(snapshot: PixelSnapshot) -> np.ndarray:
    return np.frombuffer(snapshot.data, dtype=np.uint8).reshape(snapshot.height, snapshot.width, 3)


def _patch(picture: PixelSnapshot, box: Box) -> bytes:
    x, y, width, height = box
    if x < 0 or y < 0 or x + width > 1.000001 or y + height > 1.000001:
        return b""
    left, top = int(round(x * picture.width)), int(round(y * picture.height))
    right = int(round((x + width) * picture.width))
    bottom = int(round((y + height) * picture.height))
    if right - left < 2 or bottom - top < 2:
        return b""
    crop = _pixels(picture)[top:min(bottom, picture.height), left:min(right, picture.width)]
    if not crop.size:
        return b""
    rows = np.minimum(crop.shape[0] - 1, ((np.arange(PATCH_EDGE) + 0.5) * crop.shape[0] / PATCH_EDGE).astype(int))
    cols = np.minimum(crop.shape[1] - 1, ((np.arange(PATCH_EDGE) + 0.5) * crop.shape[1] / PATCH_EDGE).astype(int))
    return crop[rows[:, None], cols[None, :]].tobytes()


def _difference(a: bytes, b: bytes) -> np.ndarray:
    return np.abs(np.frombuffer(a, np.uint8).astype(np.float32)
                  - np.frombuffer(b, np.uint8).astype(np.float32)) / 255.0


def _matches(a: bytes, b: bytes) -> bool:
    if not a or len(a) != len(b):
        return False
    difference = _difference(a, b)
    return float(np.mean(difference)) <= APPEARANCE_ERROR and float(np.quantile(difference, 0.90)) <= APPEARANCE_TAIL_ERROR


def _distinctive(patch: bytes) -> bool:
    if len(patch) != PATCH_EDGE * PATCH_EDGE * 3:
        return False
    values = np.frombuffer(patch, np.uint8).reshape(-1, 3).astype(np.float32) / 255.0
    return float(np.mean(np.std(values, axis=0))) >= MIN_SPATIAL_VARIATION


def _changed(a: bytes, b: bytes) -> bool:
    if not a or len(a) != len(b):
        return False
    difference = _difference(a, b)
    fraction = float(np.mean(difference.reshape(-1, 3).mean(axis=1) >= CHANGED_PIXEL_ERROR))
    return float(np.mean(difference)) >= MIN_LOCAL_CHANGE and fraction >= MIN_CHANGED_FRACTION


def _radii(expected: Box, picture: PixelSnapshot, *, retained: bool) -> tuple[int, int]:
    share, minimum, maximum = (0.10, 2, 6) if retained else (0.65, 6, 48)
    return (min(maximum, max(round(expected[2] * picture.width * share), minimum)),
            min(maximum, max(round(expected[3] * picture.height * share), minimum)))


def _matching_boxes(screen: ScreenSnapshot, appearance: bytes, expected: Box, *, retained: bool = False) -> tuple[Box, ...]:
    """Find distinct matching occupants around a measured endpoint, with bounded pixel search."""
    picture = screen.picture
    if picture is None:
        return ()
    scored: list[tuple[float, Box]] = []
    seen: set[tuple[int, int, int, int]] = set()
    radius_x, radius_y = _radii(expected, picture, retained=retained)

    def consider(box: Box) -> None:
        key = (round(box[0] * picture.width), round(box[1] * picture.height),
               round(box[2] * picture.width), round(box[3] * picture.height))
        if key in seen:
            return
        seen.add(key)
        patch = _patch(picture, box)
        if not patch or len(patch) != len(appearance):
            return
        error = float(np.mean(_difference(appearance, patch)))
        scored.append((error, box))

    for region in screen.regions:
        box = region.bbox
        if (region.object_extent
                and abs(box[0] + box[2] / 2 - expected[0] - expected[2] / 2) <= radius_x / picture.width
                and abs(box[1] + box[3] / 2 - expected[1] - expected[3] / 2) <= radius_y / picture.height
                and 1 / MAX_SIZE_RATIO <= box[2] / expected[2] <= MAX_SIZE_RATIO
                and 1 / MAX_SIZE_RATIO <= box[3] / expected[3] <= MAX_SIZE_RATIO):
            consider(box)
    for dx in sorted({round(value) for value in np.linspace(-radius_x, radius_x, 13)} | {0}):
        for dy in sorted({round(value) for value in np.linspace(-radius_y, radius_y, 13)} | {0}):
            consider((expected[0] + dx / picture.width, expected[1] + dy / picture.height, expected[2], expected[3]))
    seeds = sorted(scored, key=lambda item: item[0])[:4]
    step_x, step_y = max(1, math.ceil(radius_x / 6)), max(1, math.ceil(radius_y / 6))
    for _, seed in seeds:
        for dx in range(-step_x, step_x + 1):
            for dy in range(-step_y, step_y + 1):
                consider((seed[0] + dx / picture.width, seed[1] + dy / picture.height, seed[2], seed[3]))
    matches: list[Box] = []
    for error, box in sorted(scored, key=lambda item: item[0]):
        if error > APPEARANCE_ERROR:
            break
        if (retained and _overlap(box, expected) < MIN_RETAINED_OVERLAP):
            continue
        if _matches(appearance, _patch(picture, box)) and not any(_overlap(box, present) >= 0.35 for present in matches):
            matches.append(box)
    return tuple(matches)


def _receipt(intent: TransferIntent, after: ScreenSnapshot, state: PredicateState, reason: str,
             *, effect: str = "unknown", box: Box | None = None, appearance: bytes = b"") -> TransferReceipt:
    return TransferReceipt(intent.attempt_id, intent.move, intent.source, intent.destination,
                           intent.source_at, intent.destination_at, state, effect, box, reason,
                           intent.before.epoch, after.epoch, intent.before.surface, intent.before.bounds,
                           after.capture_at, intent.before.work_epoch, appearance,
                           (after.picture.width, after.picture.height) if after.picture is not None else (0, 0))


def verify_transfer(intent: TransferIntent, after: Mapping[str, Any] | ScreenSnapshot) -> TransferReceipt:
    """Verify a unique new source appearance at the delivered destination."""
    after = freeze_screen(after)
    before = intent.before
    unknown = PredicateState.UNKNOWN
    no = PredicateState.UNSATISFIED
    if intent.delivered is False:
        return _receipt(intent, after, no, "the physical transfer was not delivered", effect="none")
    if intent.delivered is not True:
        return _receipt(intent, after, unknown, "delivery has not been measured")
    if (not before.settled or not after.settled or before.occluded or after.occluded
            or before.picture is None or after.picture is None):
        return _receipt(intent, after, unknown, "settled, unobscured before and after pixels are required")
    if (not before.surface or before.surface != after.surface or before.bounds is None
            or before.bounds != after.bounds or before.work_epoch != after.work_epoch
            or (before.picture.width, before.picture.height) != (after.picture.width, after.picture.height)):
        return _receipt(intent, after, unknown, "the observed surface, viewport or work epoch changed")
    if (before.capture_at is None or after.capture_at is None or intent.started is None or intent.completed is None
            or not before.capture_at <= intent.started <= intent.completed < after.capture_at
            or before.epoch == after.epoch):
        return _receipt(intent, after, unknown, "the post-capture is not fresh after this delivery")
    if intent.source_at is None or intent.destination_at is None:
        return _receipt(intent, after, unknown, "the dispatched source and destination points are missing")
    source, reason = _source_region(intent)
    if source is None:
        return _receipt(intent, after, unknown, reason)
    appearance = _patch(before.picture, source.bbox)
    if not _distinctive(appearance):
        return _receipt(intent, after, unknown, "the source appearance is too small or visually ambiguous")
    dx = intent.destination_at[0] - intent.source_at[0]
    dy = intent.destination_at[1] - intent.source_at[1]
    expected = (source.bbox[0] + dx, source.bbox[1] + dy, source.bbox[2], source.bbox[3])
    if not _patch(after.picture, expected):
        return _receipt(intent, after, unknown, "the expected destination extent is outside the measured picture")
    if _overlap(source.bbox, expected) >= 0.35:
        return _receipt(intent, after, unknown, "source and destination extents overlap too much to measure transfer")
    matches = _matching_boxes(after, appearance, expected)
    if len(matches) > 1:
        return _receipt(intent, after, unknown, "multiple source-like occupants occur near the destination")
    if not matches:
        return _receipt(intent, after, no, "the destination pixels do not show the measured source appearance", effect="none")
    occupied = matches[0]
    old_target = _patch(before.picture, occupied)
    new_target = _patch(after.picture, occupied)
    if _matches(appearance, old_target):
        return _receipt(intent, after, no, "this source-like occupant was already at the destination", effect="none")
    if not _changed(old_target, new_target):
        return _receipt(intent, after, unknown, "the destination patch has no distinct new occupancy")
    source_now = _patch(after.picture, source.bbox)
    if _matches(appearance, source_now):
        effect = "copied"
    elif _changed(appearance, source_now):
        effect = "moved"
    else:
        return _receipt(intent, after, unknown, "the original source cannot be matched or measured as displaced")
    return _receipt(intent, after, PredicateState.SATISFIED,
                    "a unique source appearance newly occupies the delivered destination",
                    effect=effect, box=occupied, appearance=new_target)


def revalidate_placement(receipt: TransferReceipt, current: Mapping[str, Any] | ScreenSnapshot) -> PredicateState:
    """Check a retained placement against independent current pixels at its measured place."""
    current = freeze_screen(current)
    if (receipt.state != PredicateState.SATISFIED or receipt.effect_bbox is None or not receipt.appearance
            or current.picture is None or not current.settled or current.occluded
            or current.surface != receipt.surface or current.bounds != receipt.bounds
            or current.work_epoch != receipt.work_epoch or current.capture_at is None
            or receipt.after_at is None or current.capture_at <= receipt.after_at or current.epoch == receipt.after_epoch):
        return PredicateState.UNKNOWN
    if (current.picture.width, current.picture.height) != receipt.pixel_size:
        return PredicateState.UNKNOWN
    matches = _matching_boxes(current, receipt.appearance, receipt.effect_bbox, retained=True)
    if len(matches) == 1:
        return PredicateState.SATISFIED
    if len(matches) > 1:
        return PredicateState.UNKNOWN
    now = _patch(current.picture, receipt.effect_bbox)
    return PredicateState.UNSATISFIED if _changed(receipt.appearance, now) else PredicateState.UNKNOWN


@invariant("perception.transfer_requires_measured_occupancy", scope="perception",
           owner="core/perception/observed_transfer.py", observational=False)
def _transfer_receipts_require_evidence() -> tuple:
    empty = freeze_screen({"settled": True, "capture_at": 1.0, "owner": "surface", "bounds": [0, 0, 10, 10]})
    first = bind_transfer("carry", empty, source="item", destination="place", source_at=(0.1, 0.1), destination_at=(0.8, 0.8))
    second = bind_transfer("carry", empty, source="item", destination="place", source_at=(0.1, 0.1), destination_at=(0.8, 0.8))
    assert first.attempt_id != second.attempt_id, "a repeated command reused its transfer identity"
    finished = finish_delivery(first, started=1.1, completed=1.2, delivered=True)
    assert verify_transfer(finished, empty).state == PredicateState.UNKNOWN, "delivery without pixels proved placement"
    assert len(PixelSnapshot(bytes(12), 2, 2).data) == 12
    return ()
