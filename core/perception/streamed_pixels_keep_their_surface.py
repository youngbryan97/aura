"""Bind streamed RGB crops to their actual session, document and frame order.

Source capture bounds remain the drawing clock's measurement. Session and
document receipts name observation custody; they cannot identify objects.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.verify import invariant


class StreamedSurface:
    """One source session owns an ordered stream across measured document changes."""

    def __init__(self) -> None:
        self.session = uuid.uuid4().hex
        self.document = 1
        self.sequence = 0
        self.changed_at = -math.inf

    def navigated(self, event: Any, at: float | None = None) -> None:
        frame = event.get("frame") if isinstance(event, dict) else None
        if isinstance(frame, dict) and frame.get("id") and not frame.get("parentId"):
            self.document += 1
            if type(at) in (float, int) and math.isfinite(at):
                self.changed_at = at

    @property
    def owner(self) -> str:
        return f"stream/{self.session}/{self.document}"

    def frame(self, clock: MeasuredFrameTime) -> MeasuredFrameTime:
        """Only an accepted source-clock frame advances this stream's sequence."""
        self.sequence += 1
        if clock.lower is None or clock.epoch <= 0 or clock.lower < self.changed_at:
            return clock
        from core.perception.frames_as_they_are_drawn import CanvasFrameTime

        return CanvasFrameTime(float(clock), lower=clock.lower, upper=clock.upper, arrived=clock.arrived,
                               source=clock.source, source_at=clock.source_at, epoch=clock.epoch,
                               surface=self.owner, sequence=self.sequence)


def captured_crop(clock: Any, metadata: dict, clip: dict, shape: tuple) -> Any:
    """A changed crop, source extent or scroll offset names different observed material."""
    if not isinstance(clock, MeasuredFrameTime) or not getattr(clock, "surface", "").startswith("stream/"):
        return clock
    values = tuple(clip.get(key) for key in ("x", "y", "width", "height")) + tuple(
        metadata.get(key, 0.) for key in ("deviceWidth", "deviceHeight", "scrollOffsetX", "scrollOffsetY", "offsetTop"))
    if (len(shape) != 3 or shape[-1] != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or min(values[2:4]) <= 0 or min(values[4:6]) <= 0):
        return MeasuredFrameTime(float(clock), lower=clock.lower, upper=clock.upper, arrived=clock.arrived,
                                 source=clock.source, source_at=clock.source_at, epoch=clock.epoch)
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime

    geometry = hashlib.sha256(repr((values, shape)).encode("ascii")).hexdigest()[:24]
    return CanvasFrameTime(float(clock), lower=clock.lower, upper=clock.upper, arrived=clock.arrived,
                           source=clock.source, source_at=clock.source_at, epoch=clock.epoch,
                           surface=f"{clock.surface}/crop/{geometry}", sequence=clock.sequence)


@invariant("perception.streamed_pixels_keep_session_document_and_crop", scope="perception",
           owner="core/perception/streamed_pixels_keep_their_surface.py", observational=False)
def _stream_surface_invariant() -> tuple:
    source = StreamedSurface()
    clock = MeasuredFrameTime(1., lower=1., upper=1.01, arrived=1.02,
                              source="browser-wall", source_at=1000., epoch=1)
    first, second = source.frame(clock), source.frame(clock)
    assert first.surface == second.surface and first.sequence + 1 == second.sequence
    source.navigated({"frame": {"id": "actual-main-frame"}})
    assert source.frame(clock).surface != first.surface
    assert source.frame(MeasuredFrameTime(2., lower=None, upper=2., arrived=2.,
                                         source="stream-receipt", source_at=None, epoch=1)).lower is None
    return ()
