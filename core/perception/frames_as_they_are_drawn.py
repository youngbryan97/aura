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

logger = logging.getLogger(__name__)

__all__ = ["CanvasFrames", "PageFrames"]

#: How long to wait for a new frame before taking the page to be standing still.
STILL_AFTER_S = 0.25


class PageFrames:
    """Frames of one page, streamed; started on the first look, stopped on close."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self._session: Any = None
        self._latest: tuple[str, dict[str, Any], float] | None = None
        self._taken_at = -1.0
        self._arrived = asyncio.Event()
        self.unavailable = False

    async def _start(self) -> bool:
        if self._session is not None or self.unavailable:
            return self._session is not None
        try:
            session = await self.page.context.new_cdp_session(self.page)
            session.on("Page.screencastFrame", self._frame)
            # At the page's own size: a window on a high-density screen
            # otherwise sends each frame at twice that in each direction, and
            # encoding it is most of what a frame costs.
            wide, tall = await self.page.evaluate("[innerWidth, innerHeight]")
            await session.send("Page.startScreencast", {
                "format": "jpeg", "quality": 80, "everyNthFrame": 1,
                "maxWidth": int(wide), "maxHeight": int(tall),
            })
        except Exception as why:  # noqa: BLE001 - any engine without a screencast: screenshots instead
            logger.info("frames cannot be streamed from this page (%s); taking screenshots", why)
            self.unavailable = True
            return False
        self._session = session
        return True

    def _frame(self, params: dict[str, Any]) -> None:
        self._latest = (str(params.get("data") or ""), dict(params.get("metadata") or {}), time.monotonic())
        self._arrived.set()
        session = self._session
        if session is not None:
            asyncio.ensure_future(session.send("Page.screencastFrameAck", {"sessionId": params.get("sessionId")}))

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
        picture = await asyncio.to_thread(_cropped, data, metadata, clip)
        if picture is None:
            return None
        # A page standing still sends nothing; the same picture, now, is what
        # it shows now.
        return picture, (arrived if fresh else time.monotonic())

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        try:
            await session.send("Page.stopScreencast")
            await session.detach()
        except Exception as why:  # noqa: BLE001 - a page already gone has nothing to stop
            logger.debug("stopping the stream: %s", why)


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
(() => {
  let canvas = window.__auraCanvas;
  if (!canvas || !canvas.isConnected) {
    const found = [];
    const walk = (root) => { for (const el of root.querySelectorAll('*')) {
      if (el.tagName === 'CANVAS') found.push(el); if (el.shadowRoot) walk(el.shadowRoot); } };
    walk(document);
    let area = 0; canvas = null;
    for (const c of found) { const r = c.getBoundingClientRect(); if (r.width * r.height > area) { area = r.width * r.height; canvas = c; } }
    if (!canvas || area < 10000) return null;
    window.__auraCanvas = canvas;
  }
  try { return canvas.toDataURL('image/jpeg', 0.8); } catch (e) { return null; }
})()
"""


class CanvasFrames:
    """The drawing read from its own canvas; told apart from a canvas that reads back blank."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.unavailable = False
        self._checked = False

    async def look(self, clip: dict[str, float]) -> tuple[Any, float] | None:
        if self.unavailable:
            return None
        try:
            data = await self.page.evaluate(_THE_CANVAS_NOW)
        except Exception as why:  # noqa: BLE001 - a page that cannot be asked has no canvas to read
            logger.debug("the canvas could not be read: %s", why)
            data = None
        at = time.monotonic()
        picture = await asyncio.to_thread(_from_data_url, data) if data else None
        if picture is None:
            self.unavailable = True
            return None
        if not self._checked:
            self._checked = True
            if not await self._shows_what_the_screen_shows(picture, clip):
                self.unavailable = True
                return None
        return picture, at

    async def _shows_what_the_screen_shows(self, picture: Any, clip: dict[str, float]) -> bool:
        """A canvas drawn by WebGL without a kept buffer reads back blank while the screen shows a game."""
        import numpy as np

        from core.perception.picture_arithmetic import decode

        if float(np.asarray(picture, dtype=np.float32).std()) > 2.0:
            return True
        try:
            shot = decode(await self.page.screenshot(clip=clip, type="jpeg", quality=80, scale="css"))
        except Exception:  # noqa: BLE001 - nothing to compare with: trust neither, use the screen
            return False
        return shot is not None and float(np.asarray(shot, dtype=np.float32).std()) <= 2.0


def _from_data_url(data: str) -> Any:
    from core.perception.picture_arithmetic import decode

    _head, _comma, body = str(data).partition(",")
    return decode(base64.b64decode(body)) if body else None
