"""A page's pictures as the browser draws them, rather than one screenshot at a time.

A screenshot is a request: the browser stops, renders, encodes and replies.
For a window on screen that round trip ran to a tenth of a second and more,
and live play on 4 October 2026 saw two to thirteen pictures a second where
the same game, headless, gave thirty; she lost every game at that rate.

Chromium can instead send each frame as it composites it (the devtools
screencast). Whatever draws them, a canvas, WebGL, a Flash emulator, a video,
the frames come as fast as the page changes, and a page that stands still
sends none: then the last frame is still the picture.

``PageFrames.look`` returns the newest frame cropped to a part of the page,
in the page's own pixels, and None when the browser cannot stream (any
engine but Chromium): the caller then takes screenshots as before.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any

from core.runtime.executors import run_interactive_cpu

logger = logging.getLogger(__name__)

__all__ = ["CanvasFrames", "PageFrames"]

#: How long to wait for a new frame before taking the page to be standing still.
STILL_AFTER_S = 0.25


class PageFrames:
    """Frames of one page, streamed; started on the first look, stopped on close."""

    def __init__(self, page: Any, *, every_nth_frame: int = 1) -> None:
        self.page = page
        self._session: Any = None
        self._latest: tuple[str, dict[str, Any], float] | None = None
        self._taken_at = -1.0
        self._arrived = asyncio.Event()
        self._acks: set[asyncio.Task] = set()
        self.every_nth_frame = max(1, every_nth_frame)
        self.unavailable = False

    async def _start(self) -> bool:
        if self._session is not None or self.unavailable:
            return self._session is not None
        try:
            session = await self.page.context.new_cdp_session(self.page)
            self._session = session
            session.on("Page.screencastFrame", self._frame)
            # At the page's own size: a window on a high-density screen
            # otherwise sends each frame at twice that in each direction, and
            # encoding it is most of what a frame costs.
            wide, tall = await self.page.evaluate("[innerWidth, innerHeight]")
            await session.send("Page.startScreencast", {
                "format": "jpeg", "quality": 80, "everyNthFrame": self.every_nth_frame,
                "maxWidth": int(wide), "maxHeight": int(tall),
            })
        except Exception as why:  # noqa: BLE001 - any engine without a screencast: screenshots instead
            logger.info("frames cannot be streamed from this page (%s); taking screenshots", why)
            self.unavailable = True
            await self.close()
            return False
        return True

    def _frame(self, params: dict[str, Any]) -> None:
        self._latest = (str(params.get("data") or ""), dict(params.get("metadata") or {}), time.monotonic())
        self._arrived.set()
        session = self._session
        if session is not None:
            task = asyncio.create_task(self._ack(session, params.get("sessionId")))
            self._acks.add(task)
            task.add_done_callback(self._acks.discard)

    async def _ack(self, session: Any, frame_id: Any) -> None:
        try:
            await session.send("Page.screencastFrameAck", {"sessionId": frame_id})
        except Exception as why:  # noqa: BLE001 - closing a page can race its last acknowledgement
            logger.debug("acknowledging a frame: %s", why)

    async def look(self, clip: dict[str, float]) -> tuple[Any, float] | None:
        """The newest frame, cropped to ``clip`` (page pixels), and when it arrived; None if streaming is not possible."""
        if not await self._start():
            return None
        if self._latest is None or self._latest[2] <= self._taken_at:
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), timeout=STILL_AFTER_S)
            except TimeoutError:
                pass
        if self._latest is None:
            return None
        data, metadata, arrived = self._latest
        fresh = arrived > self._taken_at
        self._taken_at = arrived
        try:
            picture = await run_interactive_cpu(_cropped, data, metadata, clip)
        except TimeoutError:
            logger.info("frame decoding exceeded its interactive budget")
            return None
        if picture is None:
            return None
        # A page standing still sends nothing; the same picture, now, is what
        # it shows now.
        return picture, (arrived if fresh else time.monotonic())

    async def close(self) -> None:
        session, self._session = self._session, None
        acks = list(self._acks)
        for task in acks:
            task.cancel()
        if acks:
            await asyncio.gather(*acks, return_exceptions=True)
        self._acks.clear()
        if session is None:
            return
        try:
            await session.send("Page.stopScreencast")
        except Exception as why:  # noqa: BLE001 - a page already gone has nothing to stop
            logger.debug("stopping the stream: %s", why)
        try:
            await session.detach()
        except Exception as why:  # noqa: BLE001 - detaching remains necessary when stopping failed
            logger.debug("detaching the stream: %s", why)


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
  try { return {pixels:canvas.toDataURL('image/jpeg', 0.8),
    scene:window.__auraDrawingObservation?.snapshot(canvas) ?? null}; } catch (e) { return null; }
})
"""

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

                self._observing = await acquire(self.page)
            data = await self.page.evaluate(_THE_CANVAS_NOW, clip)
        except Exception as why:  # noqa: BLE001 - a page that cannot be asked has no canvas to read
            logger.debug("the canvas could not be read: %s", why)
            data = None
        at = time.monotonic()
        encoded = data.get("pixels") if isinstance(data, dict) else data
        try:
            picture = await run_interactive_cpu(_from_data_url, encoded) if encoded else None
        except TimeoutError:
            logger.info("canvas decoding exceeded its interactive budget")
            return None
        if picture is None:
            self.unavailable = True
            return None
        if not self._checked:
            self._checked = True
            if not await self._shows_what_the_screen_shows(picture, clip):
                self.unavailable = True
                return None
        if isinstance(data, dict):
            from core.perception.the_drawing_as_objects import described

            picture = described(picture, data.get("scene"))
        return picture, at

    async def close(self) -> None:
        self._closed = True
        if self._observing:
            from core.perception.the_drawing_as_objects import release

            self._observing = False
            try:
                await release(self.page)
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
