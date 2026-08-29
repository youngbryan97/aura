"""A browser knows what it is showing. Asking it beats squinting at it.

She reads the screen by looking at it — the same instrument whatever is in
front of her, which is right for an application nobody can question. A page is
not that. It knows exactly what it is showing and where.

LIVE 2026-08-29 on play2048.co: the screen reading found five of the sixteen
places on a board, at two distinct columns out of four. Nothing downstream
could recover — no lattice in a handful of scattered cells, so no thing to
model; no model, so nothing to look ahead over; and every move therefore fell
through to a full language generation, about twenty-eight seconds each.

Nothing here knows what a game is. It asks the page for every element holding
text of its own, and hands back what it is told in the shape the screen reader
uses, so everything downstream is unchanged.
"""

from __future__ import annotations

import json

import pytest

from core.capabilities.browser_controller import _as_applescript_text
from core.perception.what_the_page_says import MOST_REGIONS, what_the_page_says


class Page:
    """A browser that answers with whatever it was given."""

    def __init__(self, said):
        self.said = said
        self.asked = []

    async def read_page_text(self, expression: str) -> str:
        self.asked.append(expression)
        return self.said


# ── what comes back ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_page_is_read_in_the_shape_the_screen_reader_uses():
    page = Page(json.dumps([
        {"t": "2", "y": 0.4, "x": 0.3},
        {"t": "SCORE", "y": 0.1, "x": 0.6},
    ]))
    said = await what_the_page_says(page)
    assert said == ((0.1, 0.6, "SCORE"), (0.4, 0.3, "2"))


@pytest.mark.asyncio
async def test_it_is_ordered_down_the_page_then_across():
    page = Page(json.dumps([
        {"t": "c", "y": 0.5, "x": 0.9},
        {"t": "a", "y": 0.2, "x": 0.1},
        {"t": "b", "y": 0.5, "x": 0.1},
    ]))
    assert [cell[2] for cell in await what_the_page_says(page)] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_what_it_reads_can_be_laid_out_like_any_other_reading():
    from core.perception.what_is_there import arranged

    page = Page(json.dumps([
        {"t": str(r * 4 + c), "y": 0.3 + r * 0.1, "x": 0.3 + c * 0.1}
        for r in range(4)
        for c in range(4)
    ]))
    board = arranged(list(await what_the_page_says(page)))
    assert (board.rows, board.columns) == (4, 4)


# ── and nothing it cannot trust ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_browser_that_cannot_be_asked_answers_nothing():
    class Mute:
        pass

    assert await what_the_page_says(Mute()) == ()


@pytest.mark.asyncio
async def test_a_page_that_answers_with_rubbish_is_not_a_reading():
    assert await what_the_page_says(Page("not json")) == ()
    assert await what_the_page_says(Page("{}")) == ()
    assert await what_the_page_says(Page("")) == ()


@pytest.mark.asyncio
async def test_a_browser_that_throws_is_not_fatal():
    class Angry:
        async def read_page_text(self, expression):
            raise RuntimeError("no javascript here")

    assert await what_the_page_says(Angry()) == ()


@pytest.mark.asyncio
async def test_positions_off_the_page_are_left_out():
    page = Page(json.dumps([
        {"t": "on", "y": 0.5, "x": 0.5},
        {"t": "above", "y": -0.2, "x": 0.5},
        {"t": "right", "y": 0.5, "x": 1.4},
        {"t": "unreadable", "y": "x", "x": 0.5},
    ]))
    assert [cell[2] for cell in await what_the_page_says(page)] == ["on"]


@pytest.mark.asyncio
async def test_it_asks_for_a_bounded_amount():
    page = Page("[]")
    await what_the_page_says(page)
    assert str(MOST_REGIONS) in page.asked[0]


# ── the script survives being handed to AppleScript ──────────────────────

def test_a_script_with_quotes_and_newlines_survives():
    said = _as_applescript_text('var s = "a";\nreturn s;')
    assert said.startswith('"')
    assert "return" in said
    assert '\\"a\\"' in said


def test_a_backslash_is_escaped_before_the_quotes_are():
    assert _as_applescript_text("\\") == '"\\\\"'


def test_nothing_is_an_empty_literal():
    assert _as_applescript_text("") == '""'
