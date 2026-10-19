"""Talking on a page, chosen like any other control: the conversation on her own page, held to its end.

A page with a live exchange holds one place to write, and a pursuit that moves one control a round, each move
decided by her model over the whole page, answers a stranger a minute or more after they wrote. A person in a chat
does not reconsider the page between lines. So the page's conversation is offered to her as one thing on the page,
as what a page draws is (core/skills/sovereign_browser_drawing.py): when she chooses it, the conversation is held
(core/skills/talking_on_a_page.py), each line hers from her model (core/skills/her_reply_in_a_conversation.py), and
what came of it is her move. Each conversation held is kept, with what the page said when it ended and on the pages
after it on the same site: what came of a conversation is often said only once it is over.

Nothing here knows a site. The page says where it can be written in; she chooses whether talking there is the way on.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

#: The selector the conversation is offered under. Not a CSS selector: a move on it is a hand-over, never a click.
CONVERSATION = "::conversation"

#: The longest a conversation is held at one choosing; what is left of it is offered again on her next look.
HELD_AT_MOST_S = 600.0
#: How many pages after a conversation, on the same site, are kept with it.
PAGES_AFTER = 4

#: The last conversation held: when, on which site, and how many pages after it have been kept.
_LAST: dict[str, Any] = {}


def _kept_at() -> Path:
    from core.config import config

    return Path(config.paths.data_dir) / "conversations_held_on_pages.jsonl"


def _keep(row: dict[str, Any]) -> None:
    from core.runtime.errors import record_degradation

    try:
        path = _kept_at()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        record_degradation("sovereign_browser", exc, severity="info", action="keep a conversation held on a page")


def _after_the_conversation(observation: Mapping[str, Any]) -> None:
    """A page after the last conversation on the same site, kept with it."""
    site = urlparse(str(observation.get("url") or "")).netloc
    if not _LAST or site != _LAST["site"] or _LAST["pages"] >= PAGES_AFTER:
        return
    _LAST["pages"] += 1
    _keep({"after": _LAST["began"], "url": str(observation.get("url") or ""), "page_text": str(observation.get("text") or "")[:4000]})


async def with_the_conversation(browser: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """The observation, with the page's conversation among its elements when the page has a place to write a line."""
    from core.perception.a_conversation_on_a_page import LOOK

    page = getattr(browser, "page", None)
    if not observation or not observation.get("elements") or page is None:
        return observation
    _after_the_conversation(observation)
    try:
        writing = (await page.evaluate(LOOK) or {}).get("writing")
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        return observation  # a page going away while it was looked at: what it offered stands without a conversation
    if not writing:
        return observation
    says = str(writing.get("says") or "").strip()
    # What choosing it does, as a button's label says what pressing it does.
    name = ("the conversation on this page" + (f', written in "{says}"' if says else "")
            + ": choosing it talks there as it happens, each line the other side writes answered as it comes,"
            " until the place to write is gone")
    return {**observation, "elements": [*observation["elements"], {"role": "conversation", "name": name,
                                                                   "selector": CONVERSATION}]}


def chose_the_conversation(moves: list[tuple[Any, Any]]) -> bool:
    """Whether any of this round's moves is the conversation."""
    return any(str(getattr(action, "selector", "") or "") == CONVERSATION for action, _said in moves)


async def talked_on_the_page(browser: Any, goal: str, observation: Mapping[str, Any]) -> dict[str, Any]:
    """Hold the conversation on her own page, each line of hers from her model, and say what came of it."""
    from core.runtime.errors import record_degradation
    from core.skills.her_reply_in_a_conversation import her_reply
    from core.skills.talking_on_a_page import talk_on_the_page

    step: dict[str, Any] = {"chose": ["the conversation on the page"], "url": str(observation.get("url") or "")}
    page = getattr(browser, "page", None)
    if page is None:
        return {**step, "error": "her browser has no page to talk on"}

    async def reply(lines: list[Any], page_text: list[str]) -> str:
        return await her_reply(lines, request=goal, page_text=page_text)

    began = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        held = await talk_on_the_page(page, reply, until=time.monotonic() + HELD_AT_MOST_S)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("sovereign_browser", exc, severity="info", action="hold the conversation on the page")
        return {**step, "error": f"the conversation could not be held: {exc}"}
    _keep({"began": began, "url": step["url"], "asked_of_her": goal, "lines": held["lines"], "over": held["over"],
           "page_text": held["page_text"]})
    _LAST.update(began=began, site=urlparse(step["url"]).netloc, pages=0)
    theirs = sum(1 for who, _text in held["lines"] if who == "them")
    did = (f"Talked on the page: {len(held['sent'])} lines of mine, {theirs} of theirs; "
           + ("the place to write is gone." if held["over"] else "still open when I stopped."))
    return {**step, "did": did, "why": did, "moved": bool(held["sent"]), "conversation": held["lines"],
            "over": held["over"]}
