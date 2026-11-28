"""Her own view of what she made, as somebody using it would see it.

Bryan, 27 November 2026: "aura should be able to actually reason about taste". What she made is opened as it is: as it
opens, where each of its tests leaves it, and at a phone's width. Her eyes say what is on each screen
(core/perception/her_eyes.py), and her own model reads that, what was asked and what it is for, and says what she thinks of
it, what works in it, and what she would change. What she would change is built in as rules of it, tested like any other,
and then she looks again: what she does not think better made is taken back. Nothing here says what good is; she does.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from core.making.design import Design
from core.runtime.page_gateway import get_page_gateway, page_script

logger = logging.getLogger("Aura.Making.HerView")

__all__ = ["AChange", "Compared", "Eyes", "HerView", "Shot", "as_she_saw_it", "compared", "her_eyes", "her_view_of", "looked_at", "most_unlike"]

#: Her eyes: what they see in a picture, asked a question of it. The picture is a PNG, base64.
Eyes = Callable[[str, str], Awaitable[str]]
#: The window it is looked at in, and a phone's.
WIDE, NARROW = (1120, 780), (390, 844)
#: How many of the states its tests leave it in she looks at: those least like each other and like how it opens.
STATES = 3
#: How many changes she makes of one look.
CHANGES = 4
#: How long what she writes of it as she thinks may run, and that written in its shape, in tokens: a page of prose, as her
#: objectives are thought out (core/agency/her_objectives.py).
WRITES, SHAPES = 700, 900
#: Where each of its tests leaves it, read in the page that tries it.
_ENDS = page_script("() => Frame.ends(PROGRAM, window.TESTS || [])")
#: What her eyes are asked of each screen.
LOOK = ("Say what is on this screen and how it is drawn: what is where, the colours, the type, the spacing, the buttons and "
        "what they say, and anything that looks unfinished, crowded, cut off, out of line or hard to read.")


class AChange(BaseModel):
    change: str = Field(description="What to change, said as something it does or a way it looks that can be seen to be so.")
    why: str = Field(default="", description="Why it would be better made so.")


class HerView(BaseModel):
    thinks: str = Field(description="What you think of it, in your own words.")
    works: list[str] = Field(default_factory=list, max_length=4, description="What works in it.")
    would_change: list[AChange] = Field(default_factory=list, max_length=CHANGES,
                                        description="What you would change to make it better made; nothing where you would leave it as it is.")


class Compared(BaseModel):
    better: bool = Field(description="Whether it is better made now than it was before the change.")
    thinks: str = Field(default="", description="What you think of the change, in your own words.")


@dataclass
class Shot:
    """One screen of it: what it shows, the picture, and what her eyes saw in it."""

    caption: str
    png: str
    seen: str = ""


async def her_eyes(prompt: str, png: str) -> str:
    """What her own eyes see in a screen, said plainly."""
    from core.perception.her_eyes import look_with_her_eyes, plainly

    return plainly(await look_with_her_eyes(prompt, png, max_tokens=500, timeout_s=120.0))


def _words(state: Any) -> set[str]:
    return set(re.findall(r"\w+", json.dumps(state, ensure_ascii=False, sort_keys=True).lower()))


def _unlike(a: set[str], b: set[str]) -> float:
    return 1.0 - len(a & b) / max(1, len(a | b))


def most_unlike(first: Any, ends: list[tuple[str, Any]], most: int = STATES) -> list[tuple[str, Any]]:
    """Of the states its tests leave it in, those least like how it opens and like each other, one at a time."""
    had, out = [_words(first)], []
    left = [(name, state, _words(state)) for name, state in ends if isinstance(state, dict)]
    while left and len(out) < most:
        name, state, words = max(left, key=lambda one: min(_unlike(one[2], seen) for seen in had))
        if min(_unlike(words, seen) for seen in had) <= 0.0:
            break
        out.append((name, state))
        had.append(words)
        left = [one for one in left if one[0] != name]
    return out


async def _ends(design: Design, code: str, tests: str, browser: Any) -> list[tuple[str, Any]]:
    """Where each of its tests leaves it."""
    from core.making.trying import page_of

    page = await get_page_gateway().new_page(browser, source="making.her_view")
    try:
        await page.set_content(page_of(design, code, tests=tests), wait_until="load")
        got = await get_page_gateway().read(page, _ENDS, source="making.her_view")
        return [(str(name), state) for name, state in got or []]
    finally:
        await page.close()


async def _shot(html: str, browser: Any, size: tuple[int, int]) -> str:
    page = await get_page_gateway().new_page(browser, source="making.her_view", viewport={"width": size[0], "height": size[1]})
    try:
        await page.set_content(html, wait_until="load")
        await page.wait_for_timeout(450)  # what comes in has come in
        return base64.b64encode(await page.screenshot(type="png")).decode("ascii")
    finally:
        await page.close()


async def looked_at(design: Design, code: str, tests: str, browser: Any, *, pictures: dict[str, str] | None = None,
                    ends: list[tuple[str, Any]] | None = None) -> list[Shot]:
    """It as it opens, at a phone's width, and where the tests that leave it least like that leave it."""
    from core.making.trying import page_of

    shots = [Shot("as it opens", await _shot(page_of(design, code, pictures=pictures), browser, WIDE))]
    if design.kind in {"app", "site"}:
        shots.append(Shot("at a phone's width, as it opens", await _shot(page_of(design, code, pictures=pictures), browser, NARROW)))
    if design.kind == "app" and tests.strip():
        try:
            states = ends if ends is not None else await _ends(design, code, tests, browser)
        except Exception as why:  # noqa: BLE001 - what its tests leave it in is more to look at, and it is looked at without it
            logger.info("Where its tests leave %s was not read: %s", design.name, why)
            states = []
        for name, state in most_unlike(design.state, states):
            opened_in = f"window.__openIn = {json.dumps(state, ensure_ascii=False)};"
            shots.append(Shot(f"where the test “{name}” leaves it", await _shot(page_of(design, code, pictures=pictures, before=opened_in), browser, WIDE)))
    return shots


async def as_she_saw_it(shots: list[Shot], eyes: Eyes) -> list[Shot]:
    """Each screen with what her eyes saw on it; a screen they could not see is left out."""
    out = []
    for shot in shots:
        try:
            seen = " ".join(str(await eyes(LOOK, shot.png)).split())
        except (RuntimeError, TimeoutError, OSError, ValueError) as why:
            logger.info("Her eyes did not see %s: %s", shot.caption, why)
            continue
        if seen:
            out.append(Shot(shot.caption, shot.png, seen))
    return out


def _read(shots: list[Shot]) -> str:
    return "\n\n".join(f"{shot.caption.capitalize()}: {shot.seen}" for shot in shots)


async def _thought_out(question: str, given: str, shape: type[BaseModel], *, ask: Callable[..., Awaitable[Any]],
                       write: Callable[..., Awaitable[str]], how: dict[str, Any]) -> Any:
    """Her model asked ``question``: what she writes as she thinks it, then that written in ``shape``. Held to a shape from
    the start her model writes the shape and skips the thinking (interface/routes/her_own_time.py, _thinks)."""
    words = await write(question, WRITES, given=given, **how)
    if not str(words or "").strip():
        return None
    return await ask(f"What you wrote of it:\n\n{words}\n\nWrite that in this shape.", shape, SHAPES, given=given)


async def her_view_of(shots: list[Shot], *, asked: str, design: Design, ask: Callable[..., Awaitable[Any]], write: Callable[..., Awaitable[str]],
                      how: dict[str, Any] | None = None) -> HerView | None:
    """What she thinks of it as she sees it; None where her model gave nothing that could be read. ``how`` is how her model is
    to be given room to think."""
    # What was asked and what it is for, as whoever uses it knows them; its rules are its tests', and are left to them.
    given = (f"Somebody asked for this:\n{asked}\n\nYou made it: {design.name}. {design.about}\n\n"
             f"This is how it looks, as your eyes saw it:\n\n{_read(shots)}")
    view = await _thought_out("What do you think of it, as somebody who would use it? What works in it, and what would you change to "
                              "make it better made: more finished, clearer, better to use?", given, HerView, ask=ask, write=write, how=how or {})
    return view if isinstance(view, HerView) and view.thinks.strip() else None


async def compared(before: list[Shot], after: list[Shot], view: HerView, *, ask: Callable[..., Awaitable[Any]], write: Callable[..., Awaitable[str]],
                   how: dict[str, Any] | None = None) -> Compared | None:
    """Whether she thinks it better made after her changes than before them; None where her model gave nothing to read."""
    changes = "\n".join(f"- {c.change}" for c in view.would_change)
    given = f"You changed something you made:\n{changes}\n\nBefore, as your eyes saw it:\n\n{_read(before)}\n\nNow:\n\n{_read(after)}"
    got = await _thought_out("Is it better made now?", given, Compared, ask=ask, write=write, how=how or {})
    return got if isinstance(got, Compared) else None
