"""What she came for is usually below the writing about it.

She opened a real sliding puzzle, read the heading and the advertising rail
above it, found no part of what she could see that answered to her, and
reported truthfully that nothing on screen offered a move. The board was
eleven screenfuls further down. LIVE 2026-08-27, and the run ended having made
no moves at all.

Scrolling commits to nothing: it moves a view and is undone by moving back,
which is the same line drawn around every other input she may try without
knowing what it will do.
"""

from __future__ import annotations

import pytest

from core.perception.what_is_there import arranged
from core.skills.screen_pursuit import (
    SCREENFULS_TO_LOOK,
    _bring_it_into_view,
    _is_a_thing_laid_out,
)


def grid(rows: int = 4, columns: int = 4):
    return arranged([
        (0.20 + r * 0.15, 0.20 + c * 0.15, str(r * columns + c + 1))
        for r in range(rows)
        for c in range(columns)
    ])


# ── telling a thing from prose about a thing ─────────────────────────────

def test_a_grid_is_a_thing_laid_out():
    assert _is_a_thing_laid_out(grid()) is True


def test_a_heading_and_a_paragraph_are_not():
    assert _is_a_thing_laid_out(arranged([(0.1, 0.1, "Sliding Puzzle")])) is False


def test_nor_is_a_single_row_of_links():
    assert _is_a_thing_laid_out(arranged([(0.1, 0.1 + n * 0.1, f"Game {n}") for n in range(5)])) is False


def test_nor_a_grid_with_almost_nothing_in_it():
    thin = arranged([(0.2, 0.2, "1"), (0.35, 0.35, "2")])
    assert _is_a_thing_laid_out(thin) is False


def test_and_nothing_is_not_a_thing():
    assert _is_a_thing_laid_out(None) is False
    assert _is_a_thing_laid_out(arranged([])) is False


# ── scrolling until it is on screen ──────────────────────────────────────

class Page:
    """A page where the thing sits a given number of screenfuls down."""

    def __init__(self, thing_at: int):
        self.thing_at = thing_at
        self.at = 0
        self.scrolls = 0

    async def look(self):
        return {"ok": True, "at": self.at}

    def read(self, seen):
        return grid() if seen["at"] >= self.thing_at else arranged([(0.1, 0.1, "Sliding Puzzle")])

    async def scroll(self, dx: int = 0, dy: int = 0):
        self.scrolls += 1
        self.at += 1


@pytest.fixture
def hands(monkeypatch):
    def install(page):
        monkeypatch.setattr(
            "core.capabilities.host_automation.get_host_automation", lambda: page, raising=False
        )
        return page
    return install


@pytest.mark.asyncio
async def test_a_thing_already_on_screen_needs_no_scrolling(hands):
    page = hands(Page(thing_at=0))
    assert await _bring_it_into_view(page.look, page.read) == 0
    assert page.scrolls == 0


@pytest.mark.asyncio
async def test_a_thing_below_the_fold_is_scrolled_to(hands):
    page = hands(Page(thing_at=3))
    assert await _bring_it_into_view(page.look, page.read) == 3
    assert page.scrolls == 3


@pytest.mark.asyncio
async def test_a_page_with_no_thing_on_it_is_not_walked_end_to_end(hands):
    page = hands(Page(thing_at=999))
    assert await _bring_it_into_view(page.look, page.read) == SCREENFULS_TO_LOOK
    assert page.scrolls == SCREENFULS_TO_LOOK


@pytest.mark.asyncio
async def test_a_reading_that_fails_stops_the_looking(hands):
    page = hands(Page(thing_at=999))

    async def broken():
        return {"ok": False}

    assert await _bring_it_into_view(broken, page.read) == 0
    assert page.scrolls == 0


@pytest.mark.asyncio
async def test_hands_that_will_not_scroll_are_not_fatal(monkeypatch):
    page = Page(thing_at=999)

    def missing():
        raise RuntimeError("no automation here")

    monkeypatch.setattr(
        "core.capabilities.host_automation.get_host_automation", missing, raising=False
    )
    assert await _bring_it_into_view(page.look, page.read) == 0
