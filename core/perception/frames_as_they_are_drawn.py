"""A page's pictures as the browser draws them, rather than one screenshot at a time.

A screenshot is a request: the browser stops, renders, encodes and replies.
For a window on screen that round trip ran to a tenth of a second and more,
and live play on 4 October 2026 saw two to thirteen pictures a second where
the same game, headless, gave thirty; she lost every game at that rate.

Chromium can instead send each frame as it composites it (the devtools
screencast). Whatever draws them, a canvas, WebGL, a Flash emulator, a video,
the frames come as fast as the page changes. Silence can mean either a still
page or a stalled stream; a fresh screenshot distinguishes them.

``PageFrames.look`` returns the newest frame cropped to a part of the page,
in the page's own pixels, and None when the browser cannot stream (any
engine but Chromium): the caller then takes screenshots as before.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import math
import time
from typing import Any

from core.perception.a_browser_picture_has_a_deadline import picture_request
from core.perception.a_clock_for_drawn_frames import DrawingClock, MeasuredFrameTime
from core.perception.sampling_the_drawings_pace import DrawingCadence
from core.runtime.executors import run_interactive_cpu
from core.verify import invariant

logger = logging.getLogger(__name__)

__all__ = ["CanvasFrameTime", "CanvasFrames", "PageFrames"]

#: How long to wait before asking the caller for a fresh screenshot.
STILL_AFTER_S = 0.25


def _stream_options(wide: int, tall: int, every_nth: int) -> dict[str, Any]:
    """Local correspondence needs the compositor's colour samples without JPEG changes."""
    return dict(format='png', everyNthFrame=every_nth, maxWidth=wide, maxHeight=tall)


@invariant('perception.local_frame_encoding_keeps_colour_samples', scope='perception',
           owner='core/perception/frames_as_they_are_drawn.py', observational=False)
def _lossless_frame_invariant() -> tuple:
    import io

    import numpy as np
    from PIL import Image

    options = _stream_options(23, 17, 2)
    assert options['format'] == 'png' and 'quality' not in options
    pixels = (np.arange(17*23*3) % 251).astype(np.uint8).reshape(17, 23, 3)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format=options['format'])
    result = _cropped(base64.b64encode(encoded.getvalue()).decode(), {'deviceWidth': 23},
                      dict(x=3, y=2, width=13, height=11))
    assert np.array_equal(result, pixels[2:13, 3:16])
    return ()


class PageFrames:
    """Frames of one page, streamed; started on the first look, stopped on close."""

    def __init__(self, page: Any, *, every_nth_frame: int = 1) -> None:
        self.page = page
        self._session: Any = None
        self._latest: tuple[str, dict[str, Any], float] | None = None
        self._taken_at = -1.0
        self._arrived = asyncio.Event()
        self._acks: set[asyncio.Task] = set()
        self._clock = DrawingClock()
        self._generation = 0
        self.every_nth_frame = max(1, every_nth_frame)
        self._cadence = DrawingCadence(self.every_nth_frame, STILL_AFTER_S / 2)
        self._options: dict[str, Any] = {}
        self._resampling = False
        self._surface: Any = None
        self.unavailable = False

    async def _start(self) -> bool:
        if self._session is not None or self.unavailable:
            return self._session is not None
        # Each frame is acknowledged by a call to the browser, and each such call read its whole stack through the
        # file system: the same stack is read off the frames (core/runtime/browser_calls_read_no_files.py).
        from core.runtime import browser_calls_read_no_files

        browser_calls_read_no_files.install()
        self._generation += 1
        generation, session = self._generation, None
        clock = DrawingClock()
        clock.epoch = generation
        try:
            session = await picture_request(self.page.context.new_cdp_session(self.page))
            if generation != self._generation:
                await picture_request(session.detach())
                return False
            self._session = session
            self._clock = clock
            from core.perception.streamed_pixels_keep_their_surface import StreamedSurface

            self._surface = StreamedSurface()
            surface = self._surface
            session.on("Page.frameNavigated", lambda event: surface.navigated(event, time.monotonic())
                       if self._owns_session(session, generation) else None)
            session.on("Page.screencastFrame", lambda params: self._frame(
                params, source_session=session, generation=generation))
            await picture_request(session.send("Page.enable"))
            await clock.calibrate(session, time.monotonic)
            if not self._owns_session(session, generation):
                return False
            # At the page's own size: a window on a high-density screen
            # otherwise sends each frame at twice that in each direction, and
            # encoding it is most of what a frame costs.
            wide, tall = await picture_request(self.page.evaluate("[innerWidth, innerHeight]"))
            if not self._owns_session(session, generation):
                return False
            self._options = _stream_options(int(wide), int(tall), self.every_nth_frame)
            await picture_request(session.send("Page.startScreencast", self._options))
        except Exception as why:  # noqa: BLE001 - any engine without a screencast: screenshots instead
            logger.info("frames cannot be streamed from this page (%s); taking screenshots", why)
            if generation == self._generation:
                self.unavailable = True
                await self.close()
            return False
        return self._owns_session(session, generation)

    def _owns_session(self, session: Any, generation: int) -> bool:
        return self._session is session and self._generation == generation

    def _frame(self, params: dict[str, Any], *, source_session: Any = None, generation: int | None = None) -> None:
        session = self._session if source_session is None else source_session
        owned = self._owns_session(session, self._generation if generation is None else generation)
        if owned:
            metadata = dict(params.get("metadata") or {})
            captured = self._clock.frame(metadata, time.monotonic())
            if captured is not None:
                if self._surface is not None:
                    captured = self._surface.frame(captured)
                self._cadence.observe(captured)
                self._latest = (str(params.get("data") or ""), metadata, captured)
                self._arrived.set()
        if session is not None:
            task = asyncio.create_task(self._ack(session, params.get("sessionId")))
            self._acks.add(task)
            task.add_done_callback(self._acks.discard)

    async def _ack(self, session: Any, frame_id: Any) -> None:
        try:
            await picture_request(session.send("Page.screencastFrameAck", {"sessionId": frame_id}))
        except Exception as why:  # noqa: BLE001 - closing a page can race its last acknowledgement
            logger.debug("acknowledging a frame: %s", why)

    async def look(self, clip: dict[str, float]) -> tuple[Any, float] | None:
        """The newest frame and its source capture bounds; receipt time where no source clock was given."""
        if not await self._start():
            return None
        if not await self._sample_at_the_drawings_pace():
            return None
        generation, session = self._generation, self._session
        if self._latest is None or self._latest[2] <= self._taken_at:
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), timeout=STILL_AFTER_S)
            except TimeoutError:
                pass
        if not self._owns_session(session, generation):
            return None
        if self._latest is None:
            return None
        data, metadata, captured = self._latest
        if not _new_frame(captured, self._taken_at):
            return None
        self._taken_at = captured
        try:
            picture = await run_interactive_cpu(_cropped, data, metadata, clip)
        except TimeoutError:
            logger.info("frame decoding exceeded its interactive budget")
            return None
        if picture is None:
            return None
        if not self._owns_session(session, generation):
            return None
        if (getattr(captured, "surface", None) is not None and self._surface is not None
                and captured.surface != self._surface.owner):
            return None
        from core.perception.streamed_pixels_keep_their_surface import captured_crop

        captured = captured_crop(captured, metadata, clip, picture.shape)
        return picture, captured

    async def _sample_at_the_drawings_pace(self) -> bool:
        """Keep the session and capture clock while reducing measured undersampling."""
        if self._resampling:
            return self._session is not None
        stride = self._cadence.next_stride()
        if stride == self.every_nth_frame:
            return self._session is not None
        generation, session = self._generation, self._session
        if session is None:
            return False
        self._resampling = True
        try:
            # Chromium requires a stopped screencast before another starts.
            # The browser's clock and the producing CDP session are unchanged.
            await picture_request(session.send('Page.stopScreencast'))
            if not self._owns_session(session, generation):
                return False
            options = {**self._options, 'everyNthFrame': stride}
            await picture_request(session.send('Page.startScreencast', options))
            if not self._owns_session(session, generation):
                return False
            self.every_nth_frame, self._options = stride, options
        except Exception as why:  # noqa: BLE001 - unsupported resampling falls back to fresh screenshots
            logger.info('the drawing stream could not be resampled: %s', why)
            if self._owns_session(session, generation):
                self.unavailable = True
                await self.close()
            return False
        finally:
            self._resampling = False
        return True

    async def close(self) -> None:
        session, self._session = self._session, None
        self._generation += 1
        self._clock.reset()
        self._cadence = DrawingCadence(self.every_nth_frame, STILL_AFTER_S / 2)
        self._latest, self._taken_at = None, -1.0
        self._surface = None
        acks = list(self._acks)
        for task in acks:
            task.cancel()
        if acks:
            await asyncio.gather(*acks, return_exceptions=True)
        self._acks.clear()
        if session is None:
            return
        try:
            await picture_request(session.send("Page.stopScreencast"))
        except Exception as why:  # noqa: BLE001 - a page already gone has nothing to stop
            logger.debug("stopping the stream: %s", why)
        try:
            await picture_request(session.detach())
        except Exception as why:  # noqa: BLE001 - detaching remains necessary when stopping failed
            logger.debug("detaching the stream: %s", why)


def _new_frame(arrived: float, taken: float) -> bool:
    """A frame keeps its original receipt; silence cannot renew it."""
    import math

    return math.isfinite(arrived) and arrived > taken


@invariant("perception.a_repeated_frame_cannot_renew_its_time", scope="perception",
           owner="core/perception/frames_as_they_are_drawn.py", observational=False)
def _frame_time_invariant() -> tuple:
    assert _new_frame(2., 1.)
    assert not _new_frame(2., 2.)
    assert not _new_frame(1., 2.)
    assert not _new_frame(float('nan'), 2.)
    return ()


@invariant('perception.a_stream_frame_keeps_its_session_owner', scope='perception',
           owner='core/perception/frames_as_they_are_drawn.py', observational=False)
def _frame_session_invariant() -> tuple:
    source = PageFrames(None)
    session = object()
    source._session, source._generation = session, 1
    assert source._owns_session(session, 1)
    assert not source._owns_session(session, 0)
    source._session = object()
    assert not source._owns_session(session, 1)
    return ()


def _cropped(data: str, metadata: dict[str, Any], clip: dict[str, float]) -> Any:
    """The part of a streamed frame that ``clip`` covers, at the page's own pixels."""
    from PIL import Image

    from core.perception.picture_arithmetic import decode

    picture = decode(base64.b64decode(data)) if data else None
    if picture is None:
        return None
    tall, wide = picture.shape[:2]
    across = float(metadata.get("deviceWidth") or wide)
    scale = wide / max(1.0, across)
    left = (clip["x"] - float(metadata.get("scrollOffsetX") or 0.0)) * scale
    top = (clip["y"] - float(metadata.get("scrollOffsetY") or 0.0)) * scale + float(metadata.get("offsetTop") or 0.0) * scale
    right, bottom = left + clip["width"] * scale, top + clip["height"] * scale
    box = (max(0, int(round(left))), max(0, int(round(top))), min(wide, int(round(right))), min(tall, int(round(bottom))))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    part = picture[box[1] : box[3], box[0] : box[2]]
    if abs(scale - 1.0) > 0.05:
        import numpy as np

        part = np.asarray(Image.fromarray(part).resize((max(1, int(clip["width"])), max(1, int(clip["height"]))), Image.Resampling.BOX))
    return part


#: The largest canvas on the page, in its shadow roots too, read from inside
#: the page: no window, compositor or screen is involved, so a window on screen
#: gives it as fast as headless does (139 a second, LIVE-like 2026-10-04,
#: against 18 streamed and 7 screenshots from the same window).
_THE_CANVAS_NOW = """
((clip) => {
  const fits = (c) => {
    if (!c || !c.isConnected) return false;
    const r = c.getBoundingClientRect();
    return Math.abs(r.left - clip.x) <= 3 && Math.abs(r.top - clip.y) <= 3
      && Math.abs(r.width - clip.width) <= 3 && Math.abs(r.height - clip.height) <= 3;
  };
  let canvas = window.__auraCanvas;
  if (!fits(canvas)) {
    const found = [];
    const walk = (root) => { for (const el of root.querySelectorAll('*')) {
      if (el.tagName === 'CANVAS') found.push(el); if (el.shadowRoot) walk(el.shadowRoot); } };
    walk(document);
    let area = 0; canvas = null;
    for (const c of found) { const r = c.getBoundingClientRect(); if (fits(c) && r.width * r.height > area) { area = r.width * r.height; canvas = c; } }
    if (!canvas || area < 10000) return null;
    window.__auraCanvas = canvas;
  }
  let capture = null;
  try {
    let custody = window.__auraCanvasCaptureCustody;
    if (!custody || custody.document !== document) {
      const token = new Uint32Array(4); crypto.getRandomValues(token);
      custody = {document, token:Array.from(token, n => n.toString(16)).join('-'),
        canvases:new WeakMap(), canvas:0, epoch:0, sequence:0};
      window.__auraCanvasCaptureCustody = custody;
    }
    const r = canvas.getBoundingClientRect();
    const extent = [canvas.width, canvas.height, r.left, r.top, r.width, r.height];
    let owner = custody.canvases.get(canvas);
    if (!owner) {
      owner = {surface:`${custody.token}/${++custody.canvas}`, epoch:++custody.epoch, extent};
      custody.canvases.set(canvas, owner);
    } else if (extent.some((n, i) => n !== owner.extent[i])) {
      owner.epoch = ++custody.epoch; owner.extent = extent;
    }
    capture = {version:'canvas-capture-v1', surface:owner.surface, epoch:owner.epoch,
      sequence:++custody.sequence, width:canvas.width, height:canvas.height};
  } catch (_) { capture = null; }
  try {
    const source_at = performance.now() / 1000;
    const pixels = canvas.toDataURL('image/png');
    if (capture) capture.source_at = source_at;
    return {pixels, capture,
    scene:window.__auraDrawingObservation?.snapshot(canvas) ?? null,
    sounds:window.__auraSoundsHeard?.take() ?? null};
  } catch (e) { return null; }
})
"""


class CanvasFrameTime(MeasuredFrameTime):
    """Capture bounds with the canvas owner and the page's capture order."""

    def __new__(cls, at: float, *, lower: float, upper: float, arrived: float,
                source: str, source_at: float | None, epoch: int, surface: str, sequence: int):
        value = super().__new__(cls, at, lower=lower, upper=upper, arrived=arrived,
                                source=source, source_at=source_at, epoch=epoch)
        object.__setattr__(value, "surface", surface)
        object.__setattr__(value, "sequence", sequence)
        return value

    def __reduce__(self):
        return _restored_canvas_frame_time, (float(self), self.lower, self.upper, self.arrived,
                                            self.source, self.source_at, self.epoch, self.surface, self.sequence)


def _restored_canvas_frame_time(at, lower, upper, arrived, source, source_at, epoch, surface, sequence):
    return CanvasFrameTime(at, lower=lower, upper=upper, arrived=arrived, source=source,
                           source_at=source_at, epoch=epoch, surface=surface, sequence=sequence)


def _canvas_capture_time(capture: Any, began: float, finished: float, shape: tuple[int, ...]) -> MeasuredFrameTime:
    """A response without matching canvas custody keeps its capture bound unknown."""
    unknown = MeasuredFrameTime(began, lower=None, upper=finished, arrived=finished,
                                source="canvas-receipt", source_at=None, epoch=0)
    if (not all(math.isfinite(v) for v in (began, finished)) or not 0 <= began <= finished
            or not isinstance(capture, dict) or capture.get("version") != "canvas-capture-v1"):
        return unknown
    surface, epoch, sequence = (capture.get(key) for key in ("surface", "epoch", "sequence"))
    width, height, source_at = (capture.get(key) for key in ("width", "height", "source_at"))
    if (not isinstance(surface, str) or not 1 <= len(surface) <= 256
            or type(epoch) is not int or epoch <= 0 or type(sequence) is not int or sequence <= 0
            or type(width) is not int or type(height) is not int or min(width, height) <= 0
            or len(shape) < 2 or (height, width) != shape[:2]):
        return unknown
    if source_at is not None and (isinstance(source_at, bool) or not isinstance(source_at, (int, float))
                                  or not math.isfinite(source_at) or source_at < 0):
        return unknown
    return CanvasFrameTime(began, lower=began, upper=finished, arrived=finished,
                           source="canvas-bracket", source_at=source_at, epoch=epoch,
                           surface=surface, sequence=sequence)


@invariant("perception.canvas_capture_bounds_keep_their_surface", scope="perception",
           owner="core/perception/frames_as_they_are_drawn.py", observational=False)
def _canvas_capture_bounds_invariant() -> tuple:
    capture = {"version": "canvas-capture-v1", "surface": "document/canvas", "epoch": 3,
               "sequence": 7, "width": 20, "height": 12, "source_at": 2.}
    at = _canvas_capture_time(capture, 10., 10.02, (12, 20, 3))
    assert isinstance(at, CanvasFrameTime) and at.lower == float(at) == 10.
    assert at.upper == at.arrived == 10.02 and at.source == "canvas-bracket"
    assert (at.surface, at.epoch, at.sequence, at.source_at) == ("document/canvas", 3, 7, 2.)
    assert _canvas_capture_time(None, 10., 10.02, (12, 20, 3)).lower is None
    assert _canvas_capture_time({**capture, "surface": ""}, 10., 10.02, (12, 20, 3)).lower is None
    assert _canvas_capture_time(capture, 10., 10.02, (10, 20, 3)).lower is None
    return ()


class CanvasFrames:
    """The drawing read from its own canvas; told apart from a canvas that reads back blank."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.unavailable = False
        self._checked = False
        self._closed = False
        self._observing = False

    async def look(self, clip: dict[str, float]) -> tuple[Any, float] | None:
        if self.unavailable or self._closed:
            return None
        try:
            if not self._observing:
                from core.perception.the_drawing_as_objects import acquire
                from core.perception.what_the_page_plays import BOOTSTRAP as LISTEN

                self._observing = await picture_request(acquire(self.page))
                # What the page plays is heard from here on, where it was not listened to from its first script.
                await picture_request(self.page.evaluate(LISTEN))
            # The request starts before the canvas is read. Its response can
            # arrive much later, so completion cannot make the picture new.
            began = time.monotonic()
            data = await picture_request(self.page.evaluate(_THE_CANVAS_NOW, clip))
            finished = time.monotonic()
        except Exception as why:  # noqa: BLE001 - a page that cannot be asked has no canvas to read
            logger.debug("the canvas could not be read: %s", why)
            data = None
        encoded = data.get("pixels") if isinstance(data, dict) else data
        try:
            picture = await run_interactive_cpu(_from_data_url, encoded) if encoded else None
        except TimeoutError:
            logger.info("canvas decoding exceeded its interactive budget")
            return None
        if picture is None:
            self.unavailable = True
            return None
        at = _canvas_capture_time(data.get("capture") if isinstance(data, dict) else None,
                                  began, finished, picture.shape)
        if not self._checked:
            self._checked = True
            if not await self._shows_what_the_screen_shows(picture, clip):
                self.unavailable = True
                return None
        if isinstance(data, dict):
            from core.perception.the_drawing_as_objects import described
            from core.perception.what_the_page_plays import heard_with, sounds_in

            picture = heard_with(described(picture, data.get("scene")), sounds_in(data.get("sounds"), at))
        return picture, at

    async def close(self) -> None:
        self._closed = True
        if self._observing:
            from core.perception.the_drawing_as_objects import release

            self._observing = False
            try:
                await picture_request(release(self.page))
            except Exception:  # noqa: BLE001 - a closed page has already released its observer
                pass

    async def _shows_what_the_screen_shows(self, picture: Any, clip: dict[str, float]) -> bool:
        """A canvas drawn by WebGL without a kept buffer reads back blank while the screen shows a game."""
        import numpy as np


        if float(np.asarray(picture, dtype=np.float32).std()) > 2.0:
            return True
        from core.perception.a_picture_of_her_page import picture_of

        shot = await picture_of(self.page, clip, css=True)
        return shot is not None and float(np.asarray(shot, dtype=np.float32).std()) <= 2.0


def _from_data_url(data: str) -> Any:
    from core.perception.picture_arithmetic import decode

    _head, _comma, body = str(data).partition(",")
    return decode(base64.b64decode(body)) if body else None
