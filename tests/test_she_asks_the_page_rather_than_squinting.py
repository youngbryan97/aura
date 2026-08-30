"""A page knows what it is showing. She was looking at a photograph of it.

LIVE 2026-08-29 on play2048.co: the screen reading found five of the sixteen
places on the board, at two distinct columns out of four. No lattice in a
handful of scattered cells, so no thing to model; no model, so nothing to look
ahead over; and every move fell through to a full language generation, about
twenty-eight seconds each. The board was drawn perfectly well the whole time.

The reader for this was written the same day and never called by anything.
"""

from __future__ import annotations

import pytest

from core.skills.screen_pursuit import _the_best_reading_available

#: What a screenshot made of the board: a few tiles, badly placed.
SQUINTING = {
    "layout": [
        {"text": "2", "center_x": 0.20, "center_y": 0.30},
        {"text": "4", "center_x": 0.52, "center_y": 0.30},
        {"text": "2", "center_x": 0.20, "center_y": 0.62},
    ]
}

#: What the page says: every tile, where it actually is.
THE_PAGE = tuple(
    (0.30 + 0.16 * row, 0.20 + 0.16 * column, "2")
    for row in range(4)
    for column in range(4)
)


@pytest.mark.asyncio
async def test_the_page_is_preferred_when_it_sees_more(monkeypatch):
    async def says():
        return THE_PAGE

    monkeypatch.setattr("core.perception.what_the_page_says.what_the_page_says", says)
    got = await _the_best_reading_available(
        SQUINTING, None, like=None, in_a_browser=True
    )
    assert got.occupied() == 16
    assert (got.rows, got.columns) == (4, 4)


@pytest.mark.asyncio
async def test_a_page_that_says_less_changes_nothing(monkeypatch):
    async def says():
        return ((0.30, 0.20, "2"),)

    monkeypatch.setattr("core.perception.what_the_page_says.what_the_page_says", says)
    got = await _the_best_reading_available(
        SQUINTING, None, like=None, in_a_browser=True
    )
    assert got.occupied() == 3


@pytest.mark.asyncio
async def test_nothing_that_is_not_a_browser_is_asked(monkeypatch):
    async def says():  # pragma: no cover - must not run
        raise AssertionError("a native application was asked what page it is showing")

    monkeypatch.setattr("core.perception.what_the_page_says.what_the_page_says", says)
    got = await _the_best_reading_available(
        SQUINTING, None, like=None, in_a_browser=False
    )
    assert got.occupied() == 3


@pytest.mark.asyncio
async def test_a_page_that_will_not_answer_leaves_the_screen_reading(monkeypatch):
    async def says():
        raise RuntimeError("no browser")

    monkeypatch.setattr("core.perception.what_the_page_says.what_the_page_says", says)
    got = await _the_best_reading_available(
        SQUINTING, None, like=None, in_a_browser=True
    )
    assert got.occupied() == 3
