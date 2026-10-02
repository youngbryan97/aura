"""Playing what a page draws: from the page's elements to her eyes, in the same window.

A page that draws a game holds nothing for the browser pursuit to read or press:
LIVE 2026-10-02, a Flash game on webdesignmuseum.org is a 540 by 360 canvas in
Ruffle's shadow root, with its own PLAY button painted on it. The screen pursuit
plays exactly that kind of thing — it read and played 2048 in a browser window —
so when she chooses the drawing, the same visible window is handed to it, with the
part of the window the page says it is drawing in.

Nothing here knows what a game is. The page reports where it draws; she chooses
whether what is drawn is the way on; the screen pursuit finds out how it moves.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


#: The selector the observer gives what a page draws. Not a CSS selector: a move
#: on it is a hand-over, never a click.
DRAWING = "::drawing"


def chose_the_drawing(moves: list[tuple[Any, Any]]) -> bool:
    """Whether any of this round's moves is the drawing."""
    return any(str(getattr(action, "selector", "") or "") == DRAWING for action, _said in moves)


def _the_band(said: Any) -> tuple[float, float, float, float] | None:
    """The page's answer to where it draws, as shares of the window."""
    try:
        where = json.loads(str(said or ""))
        band = (float(where["left"]), float(where["top"]), float(where["right"]), float(where["bottom"]))
    except (TypeError, ValueError, KeyError):
        return None
    left, top, right, bottom = band
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        return None
    return band


async def played_on_the_drawing(
    browser: Any, goal: str, observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Play what her own browser draws, or say plainly that it cannot be played yet.

    Never by aiming the screen pursuit at whatever window is in front. LIVE 2 Oct
    her browser had no window on the screen at all, the front window was the
    person's own Chrome, and a hand-over that named the front window as hers was
    stopped only by a page check before any key went. Playing has to go through
    her own page — its picture and its input — and until it does, the round is
    refused with the reason.
    """
    return {
        "chose": ["what the page draws"],
        "url": str(observation.get("url") or ""),
        "landed": 0,
        "error": "playing what her own browser draws, by sight, is not built yet",
    }
