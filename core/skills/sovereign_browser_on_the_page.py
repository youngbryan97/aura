"""What her own page offers besides its controls, each as one thing on it: a conversation to hold
(core/skills/sovereign_browser_conversation.py) and what it plays, to watch (core/skills/watching_along.py).

Choosing one is a hand-over, never a click, as choosing what a page draws is (core/skills/sovereign_browser_drawing.py).
Nothing here knows a site: a page plays when it has a video on it.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from . import sovereign_browser_conversation as conversation
from .watching_along import (
    WHAT_PLAYS,
    WHAT_THE_PAGE_SAYS,
    clock,
    into_her_memory,
    kept_at,
    watch_along,
)

#: The selector what a page plays is offered under. Not a CSS selector: a move on it is a hand-over.
WATCHING = "::watching"
#: The longest a watching is held at one choosing when the page does not say how long its video is.
WATCHED_AT_MOST_S = 30 * 60.0
#: The share of the view a video takes when it is what the page is for. Measured 19 October 2026 in a 1024 by 768
#: view: a channel's page plays its trailer at 0.128 of it, a page for one video plays it at 0.293. A video watched
#: first takes at least this; a smaller one is offered and watched when her model chooses it.
THE_PAGE_IS_ITS_VIDEO = 0.2
#: When each request to watch was last watched first, so that a request is watched first once: a site that plays the
#: next video when one ends (at a new address) is not watched on by itself. After it, her model decides.
_WATCHED_FOR: dict[str, float] = {}
WATCHED_FIRST_ONCE_IN_S = 3600.0


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
    share = plays["w"] * plays["h"] / max(1.0, float(plays.get("vw") or 0) * float(plays.get("vh") or 0))
    return {**observation, "elements": [*observation["elements"],
                                        {"role": "watching", "name": name, "selector": WATCHING, "share_of_view": round(share, 3)}]}


def watched_first(observation: dict[str, Any], goal: str, *, take: bool = True) -> dict[str, Any] | None:
    """The first move on a page that plays what the goal asks her to watch: watching it, before the page is read.

    A video starts when its page opens. Reading the page first, and then deciding, is a minute or more of her model's
    time while it plays on, so she would join it part way through. As with a game seen running
    (core/skills/sovereign_browser_drawing.py ``played_first``): once a request, and only where it asks for
    something to be run, played or watched (core/skills/whether_a_page_serves.py).
    """
    import time

    from core.skills.whether_a_page_serves import what_the_task_needs

    if what_the_task_needs(goal) != "run" or time.time() - _WATCHED_FOR.get(goal, 0.0) < WATCHED_FIRST_ONCE_IN_S:
        return None
    if not any(isinstance(e, dict) and e.get("selector") == WATCHING and float(e.get("share_of_view") or 0) >= THE_PAGE_IS_ITS_VIDEO
               for e in observation.get("elements") or []):
        return None
    if take:
        _WATCHED_FOR[goal] = time.time()
    why = "this page plays what I was asked to watch"
    return {"resolved_actions": [{"selector": WATCHING, "said": why}], "why": why, "done": False}


def handed_over(moves: list[tuple[Any, Any]]) -> Callable[..., Awaitable[dict[str, Any]]] | None:
    """What a move chose to be handed over to, if it chose one: the conversation, or what the page plays."""
    chosen = {str(getattr(action, "selector", "") or "") for action, _said in moves}
    if conversation.CONVERSATION in chosen:
        return conversation.talked_on_the_page
    if WATCHING in chosen:
        return watched_on_the_page
    return None


#: Every video on her own page paused and muted: what she watches with a person plays where they watch it.
QUIET_HERE = "() => { for (const v of document.querySelectorAll('video')) { v.muted = true; v.pause(); } return true; }"


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
        title = str(observation.get("title") or "") or str(await page.title() or "")
        # Watched with the person where they watch: opened in their browser, and her own page quiet so that one plays
        # (core/skills/watching_the_screen.py). Where it cannot be shown there, on her own page as before.
        from core.skills.watching_the_screen import _their_browser, on_their_screen
        from core.skills.watching_together import start_watching_in_a_with

        said_of_itself = await page.evaluate(WHAT_THE_PAGE_SAYS) or {}
        duration, words = float(plays.get("duration") or 0), str(said_of_itself.get("words") or "")
        if await asyncio.to_thread(lambda: bool(_their_browser(step["url"])[0])):
            # Watched together, beside the conversation (core/skills/watching_together.py): her answer to the
            # request goes first, then it opens on their screen; the turn ends with her answer, not the video.
            await page.evaluate(QUIET_HERE)
            start_watching_in_a_with(lambda: on_their_screen(step["url"], title, duration, words), goal,
                                     title=title, url=step["url"], at_most_s=at_most)
            did = (f"Found {title or 'the video'}{f' ({clock(duration)})' if duration else ''}. It opens on their "
                   "screen once this reply has gone, and is watched with them there while the conversation goes on.")
            return {**step, "did": did, "why": did, "moved": True, "over": True, "where": "on their screen, together"}
        where = await on_their_screen(step["url"], title, duration, words)
        if where is not None:
            await page.evaluate(QUIET_HERE)
            step["where"] = f"on their screen, in {where.owner}"
        watching = await watch_along(where or page, goal, keep=kept_at(title, step["url"]),
                                     remember=partial(into_her_memory, url=step["url"]), at_most_s=at_most)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("sovereign_browser", exc, severity="info", action="watch what the page plays")
        return {**step, "error": f"the watching could not go on: {exc}"}
    did = (f"Watched {watching.title or 'what the page plays'} {'to its end' if watching.ended else 'until I stopped'}; "
           f"said {len(watching.said)} things as it played.")
    return {**step, "did": did, "why": did, "moved": True, "over": watching.ended,
            "said": [{"at": at, "words": words} for at, words in watching.said]}
