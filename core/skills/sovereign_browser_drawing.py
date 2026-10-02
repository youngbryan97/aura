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

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from core.runtime.errors import record_degradation

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
    """Hand the window to the screen pursuit, over the part the page draws in.

    Returns the round's record: what was handed over, and what came back.
    """
    from core.capabilities import window_server
    from core.perception.what_the_page_says import _WHERE_IS_IT

    from .screen_pursuit import pursue_on_screen

    url = str(observation.get("url") or "")
    step: dict[str, Any] = {"chose": ["what the page draws"], "url": url, "landed": 0}
    page = getattr(browser, "page", None)
    if page is None:
        step["error"] = "no_page_to_play"
        return step
    try:
        await page.bring_to_front()
        band = _the_band(await page.evaluate(_WHERE_IS_IT))
        owner = str(await asyncio.to_thread(window_server.front_owner) or "")
    except Exception as exc:  # noqa: BLE001 - a hand-over that cannot be made is reported, not raised
        record_degradation("sovereign_browser.drawing", exc, severity="warning")
        step["error"] = f"could_not_hand_over:{type(exc).__name__}"
        return step
    if band is None or not owner:
        step["error"] = "the page did not say where it draws" if band is None else "no window in front"
        return step
    played = await pursue_on_screen(
        goal=goal,
        success_when="",
        target_app=owner,
        expect_page=url,
        drawn_at=band,
        narrate=True,
    )
    moves = played.get("moves") if isinstance(played, Mapping) else None
    made = len(moves) if isinstance(moves, list) else 0
    step.update(
        {
            "ok": bool(made),
            "landed": made,
            "moved": bool(made),
            "played": {
                "outcome": str((played or {}).get("outcome") or ""),
                "moves": made,
                "window": owner,
                "band": list(band),
            },
            "why": str((played or {}).get("summary") or (played or {}).get("outcome") or ""),
        }
    )
    return step
