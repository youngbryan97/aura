"""Holding a conversation on a page: reading it, answering what is new from the other side, until it ends.

Any page with a live exchange (core/perception/a_conversation_on_a_page.py): a chat with a stranger, a support
window, a game played by talking. Each look reads the conversation; a new line from the other side asks for her
reply, from ``reply`` (her mind, given the lines so far and the page's own words), written into the page's place to
write and sent; when nobody has written for a while she may open it. A place to write closed to her is the other
side's turn, and waited out. It ends when the place to write is gone, when it is closed and the page offers something
new to do, or at ``until``. Nothing here knows a site, and nothing here writes a word of hers: ``reply`` does.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.perception.a_conversation_on_a_page import LOOK, Line, read_conversation

#: How often the page is looked at, and how long a conversation nobody has opened waits before she may open it.
LOOK_EVERY_S = 0.5
OPENED_BY_NOBODY_S = 6.0
#: How long a sent line is given to appear in the stream before the control beside the box is pressed instead.
SENT_SHOWS_S = 1.5
#: How many times her reply is asked for the same lines when her mind gives no words in the time (a lane busy, a decode
#: out of time) before they are let go: once more, not until the other side writes again.
ASKED_AT_MOST = 2

#: Press the control nearest the place to write: what sends a line where Return does not.
_PRESS_THE_SENDER = r"""(path) => {
  const at = (p) => { let el = document.body; for (const part of p.split("/")) { const [tag, n] = part.split(":");
      el = Array.from(el.children).filter(c => c.tagName.toLowerCase() === tag)[Number(n) - 1]; if (!el) return null; } return el; };
  const box = at(path); if (!box) return false; const b = box.getBoundingClientRect();
  const near = Array.from(document.querySelectorAll("button,[role=button],input[type=submit]")).filter(e => {
      const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && !e.disabled; })
    .map(e => { const r = e.getBoundingClientRect(); return [Math.hypot(r.left + r.width / 2 - (b.left + b.width), r.top + r.height / 2 - (b.top + b.height / 2)), e]; })
    .sort((x, y) => x[0] - y[0]);
  if (!near.length) return false; near[0][1].click(); return true; }"""


def css_of(path: str) -> str:
    """A path from ``LOOK`` ("div:2/form:1/input:1") as a CSS selector from the body."""
    return "body > " + " > ".join(f"{tag}:nth-of-type({n})" for tag, n in (part.split(":") for part in path.split("/")))


async def talk_on_the_page(page: Any, reply: Callable[[list[Line], list[str]], Awaitable[str]], *, until: float = float("inf"),
                           look_every_s: float = LOOK_EVERY_S) -> dict[str, Any]:
    """Hold the conversation on ``page`` until it ends; what was said, and the page's words at the end."""
    sent: list[str] = []
    stream, before, answered, began, ended = "", None, 0, time.monotonic(), False
    asked_for, tries = -1, 0  # how many of their lines her reply was last asked for, and how many times
    talk = read_conversation(await page.evaluate(LOOK), sent)
    offered = set(talk.controls)  # what the page offered to press while she could write, or when she came
    while time.monotonic() < until:
        look = await page.evaluate(LOOK)
        talk = read_conversation(look, sent, before=before, stream=stream)
        stream, before = talk.stream or stream, look
        if talk.over:
            break
        if talk.writing is None:
            # Closed to her: the other side's turn, waited out; or the end, where the page offers something new to do
            # beside a place to write it did not take away ("A person", "A machine").
            if ended := bool(talk.controls - offered):
                break
            await asyncio.sleep(look_every_s)
            continue
        offered |= talk.controls
        theirs = talk.theirs()
        opened_by_nobody = not talk.lines and not sent and time.monotonic() - began >= OPENED_BY_NOBODY_S
        if len(theirs) > answered or (opened_by_nobody and not (asked_for == 0 and tries >= ASKED_AT_MOST)):
            tries = tries + 1 if asked_for == len(theirs) else 1
            asked_for = len(theirs)
            words = (await reply(list(talk.lines), list(talk.page_text))).strip()
            if words and await _say(page, words):
                sent.append(words)
            if words or tries >= ASKED_AT_MOST:
                answered = len(theirs)
        await asyncio.sleep(look_every_s)
    return {"lines": [(line.who, line.text) for line in talk.lines], "over": talk.over or ended,
            "page_text": talk.page_text, "sent": sent}


async def _say(page: Any, words: str) -> bool:
    """Write ``words`` where the page takes them and send them: Return, else the control beside the box. False where
    the place to write closed or went while she thought: what she would have said is not said."""
    writing = (await page.evaluate(LOOK) or {}).get("writing")
    if not writing:
        return False
    path = writing["path"]
    await page.fill(css_of(path), words)
    await page.press(css_of(path), "Enter")
    gives_up_at = time.monotonic() + SENT_SHOWS_S
    while time.monotonic() < gives_up_at:
        look = await page.evaluate(LOOK)
        if any(" ".join(b["text"].lower().split()) == " ".join(words.lower().split()) for b in look.get("blocks") or ()):
            return True
        await asyncio.sleep(0.2)
    await page.evaluate(_PRESS_THE_SENDER, path)
    return True
