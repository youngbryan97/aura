"""A picture of part of her page, taken without the page moving under it.

Asked for a picture of part of a page, Chromium lays the page out again at the
size of that part while it takes the picture, and a window on screen shows it:
LIVE 2026-10-09 the window she played in flickered, a frame at a time, to the
game alone, enlarged at its corner on grey, each time she looked at the game.
Measured offline the same day: of the frames a window showed while its game
was photographed part by part, half were that; photographed whole, none.

So the whole window is taken, as it already is on screen, and the part is cut
out of it here. Where the browser cannot be asked for that (not Chromium, or no
way to talk to it), the page's own picture of the part is taken as before.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import weakref
from collections.abc import Mapping
from typing import Any

__all__ = ["picture_of"]

logger = logging.getLogger("Aura.APictureOfHerPage")

#: A way to talk to each page's browser directly, made the first time it is needed.
_SESSIONS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


async def _session(page: Any) -> Any:
    held = _SESSIONS.get(page)
    if held is None:
        held = await page.context.new_cdp_session(page)
        _SESSIONS[page] = held
    return held


def _cut(data: bytes, clip: Mapping[str, float] | None, wide: int, css: bool) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as whole:
        image = whole.convert("RGB")
    ratio = image.width / max(1, wide)
    if clip:
        x, y = float(clip["x"]), float(clip["y"])
        w, h = float(clip["width"]), float(clip["height"])
        box = (max(0, round(x * ratio)), max(0, round(y * ratio)),
               min(image.width, round((x + w) * ratio)), min(image.height, round((y + h) * ratio)))
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            return None
        image = image.crop(box)
        if css and abs(ratio - 1.0) > 0.01:
            image = image.resize((max(1, round(w)), max(1, round(h))))
    elif css and abs(ratio - 1.0) > 0.01:
        image = image.resize((max(1, round(image.width / ratio)), max(1, round(image.height / ratio))))
    return np.asarray(image)


async def picture_of(page: Any, clip: Mapping[str, float] | None = None, *, css: bool = True, kind: str = "jpeg") -> Any:
    """What ``page`` shows of ``clip`` (CSS pixels from the top-left of its window), as an RGB array; at CSS size
    where ``css``, else at the screen's own pixels. None where no picture could be taken."""
    from core.perception.what_her_page_shows import the_page_size

    try:
        session = await _session(page)
        asked: dict[str, Any] = {"format": kind}
        if kind == "jpeg":
            asked["quality"] = 85
        data = base64.b64decode((await session.send("Page.captureScreenshot", asked))["data"])
        wide, _tall = await the_page_size(page)
        return await asyncio.to_thread(_cut, data, clip, wide, css)
    except Exception as why:  # noqa: BLE001 - no direct way to the browser: its own picture of the part, which may move it
        logger.debug("a whole-window picture could not be taken (%s); taking the part", str(why)[:120])
    try:
        from core.perception.picture_arithmetic import decode

        options: dict[str, Any] = {"type": "png", "scale": "css" if css else "device"}
        if clip:
            options["clip"] = dict(clip)
        return decode(await page.screenshot(**options))
    except Exception:  # noqa: BLE001 - a page gone, or off screen: no picture
        return None
