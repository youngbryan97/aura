"""What her own page offers besides its controls, each as one thing on it: a conversation to hold
(core/skills/sovereign_browser_conversation.py) and what it plays, to watch (core/skills/watching_along.py).

Choosing one is a hand-over, never a click, as choosing what a page draws is (core/skills/sovereign_browser_drawing.py).
Nothing here knows a site: a page plays when it has a video on it.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from . import sovereign_browser_conversation as conversation
from .watching_along import WHAT_PLAYS, clock, kept_at, watch_along

#: The selector what a page plays is offered under. Not a CSS selector: a move on it is a hand-over.
WATCHING = "::watching"
#: The longest a watching is held at one choosing when the page does not say how long its video is.
WATCHED_AT_MOST_S = 30 * 60.0


async def what_is_on_it(browser: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """The observation, with its conversation and what it plays among its elements where the page has them."""
    return await with_what_plays(browser, await conversation.with_the_conversation(browser, observation))


async def with_what_plays(browser: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """The observation, with what the page plays among its elements when there is a video on it."""
    page = getattr(browser, "page", None)
    if not observation or not observation.get("elements") or page is None:
        return observation
    try:
        plays = await page.evaluate(WHAT_PLAYS)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        return observation  # a page going away while it was looked at plays nothing
    if not plays or not plays.get("w") or not plays.get("h"):
        return observation
    length = f", {clock(plays['duration'])} long" if plays.get("duration") else ""
    name = f"what this page plays, a video{length}: choosing it watches it as it plays, start to end, with whoever asked"
    return {**observation, "elements": [*observation["elements"], {"role": "watching", "name": name, "selector": WATCHING}]}


def handed_over(moves: list[tuple[Any, Any]]) -> Callable[..., Awaitable[dict[str, Any]]] | None:
    """What a move chose to be handed over to, if it chose one: the conversation, or what the page plays."""
    chosen = {str(getattr(action, "selector", "") or "") for action, _said in moves}
    if conversation.CONVERSATION in chosen:
        return conversation.talked_on_the_page
    if WATCHING in chosen:
        return watched_on_the_page
    return None


async def watched_on_the_page(browser: Any, goal: str, observation: dict[str, Any]) -> dict[str, Any]:
    """Watch what her own page plays with whoever asked ``goal``, and say what came of it."""
    from core.runtime.errors import record_degradation

    step: dict[str, Any] = {"chose": ["what the page plays"], "url": str(observation.get("url") or "")}
    page = getattr(browser, "page", None)
    if page is None:
        return {**step, "error": "her browser has no page to watch on"}
    try:
        plays = await page.evaluate(WHAT_PLAYS) or {}
        at_most = float(plays.get("duration") or 0) + 120.0 if plays.get("duration") else WATCHED_AT_MOST_S
        watching = await watch_along(page, goal, keep=kept_at(str(observation.get("title") or step["url"])), at_most_s=at_most)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("sovereign_browser", exc, severity="info", action="watch what the page plays")
        return {**step, "error": f"the watching could not go on: {exc}"}
    did = (f"Watched {watching.title or 'what the page plays'} {'to its end' if watching.ended else 'until I stopped'}; "
           f"said {len(watching.said)} things as it played.")
    return {**step, "did": did, "why": did, "moved": True, "over": watching.ended,
            "said": [{"at": at, "words": words} for at, words in watching.said]}
