"""Reading what a page in her own browser draws, from the page's own pixels.

A Flash game on a museum page is a canvas inside a player's shadow root: no
element holds its PLAY button, its score or its dialogue, so the only way to
read it is to look at it. The desktop way of looking — a picture of a window
by its number — needs her browser to have a window on the screen, and LIVE
2 October it had none, while the window in front was the person's own Chrome.

A page can be photographed by the browser that draws it, whatever is on the
screen and whether or not it is visible at all. The picture is read by the same
test a window's is: pictures are taken until two agree, in what they say and in
how each place looks (``what_the_pixels_show._read_until_settled``). What comes
back is shaped like a reading of a window, so nothing that reads it needs to
know which it was; its ``bounds`` are in the page's own pixels, which is what a
click on the page is given.
"""

from __future__ import annotations

import io
import time
from typing import Any

__all__ = ["look_at_a_page", "the_page_size"]


async def the_page_size(page: Any) -> tuple[int, int]:
    """The page's viewport in CSS pixels."""
    size = getattr(page, "viewport_size", None)
    if isinstance(size, dict) and size.get("width") and size.get("height"):
        return int(size["width"]), int(size["height"])
    wide, tall = await page.evaluate("[window.innerWidth, window.innerHeight]")
    return int(wide), int(tall)


def _decoded(png: bytes) -> Any:
    """PNG bytes as the BGR array the readers take."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(png)) as picture:
        return np.ascontiguousarray(np.asarray(picture.convert("RGB"))[:, :, ::-1])


async def look_at_a_page(
    page: Any,
    over: tuple[float, float, float, float] | None = None,
    *,
    name: str,
    wait_for_stillness: bool = True,
    still_within_s: float | None = None,
) -> dict[str, Any] | None:
    """Read the part ``over`` of a page (shares of its viewport), or all of it. None if it cannot be photographed."""
    from core.perception.what_the_pixels_show import (
        _STILL_WITHIN_S,
        _read_until_settled,
        looker_for,
    )

    began = time.monotonic()
    surface = f"page:{id(page)}:{name}:{getattr(page, 'url', '')}"
    wide, tall = await the_page_size(page)
    left, top, right, bottom = over if over is not None else (0.0, 0.0, 1.0, 1.0)
    clip = {
        "x": max(0.0, left) * wide,
        "y": max(0.0, top) * tall,
        "width": max(1.0, (min(1.0, right) - max(0.0, left)) * wide),
        "height": max(1.0, (min(1.0, bottom) - max(0.0, top)) * tall),
    }
    bounds = [int(clip["x"]), int(clip["y"]), int(clip["width"]), int(clip["height"])]

    async def take() -> Any:
        from core.perception.a_picture_of_her_page import picture_of

        # The whole window, cut down here: a picture of part of the page made a visible window flicker (LIVE 2026-10-09).
        picture = await picture_of(page, clip, css=False, kind="png")
        return None if picture is None else picture[:, :, ::-1].copy()

    settled = await _read_until_settled(
        looker_for(name),
        take,
        wait_for_stillness,
        _STILL_WITHIN_S if still_within_s is None else still_within_s,
        began,
        surface=surface,
        viewport=tuple(bounds),
    )
    if settled is None:
        return None
    reading, still, at_rest_but_unread, pictures, _shape, looked_took = settled
    if surface != f"page:{id(page)}:{name}:{getattr(page, 'url', '')}":
        # Navigation during the read cannot certify either page's geometry.
        return None
    reading.update(
        {
            "scoped_to": name,
            "bounds": bounds,
            "read_within": "the part" if over is not None else "the page",
            # Her own page is in front of her own browser whatever the screen shows.
            "in_front_then": name,
            "her_window_showing": True,
            "owner": name,
            "at": time.time(),
            "settled": bool(still),
            "seconds_to_still": round(looked_took, 3),
            "pictures_to_still": pictures,
            "still_but_unread": at_rest_but_unread,
            "seconds_reading": round(time.monotonic() - began - looked_took, 3),
            "on_her_page": True,
        }
    )
    return reading
