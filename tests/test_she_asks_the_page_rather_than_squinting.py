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


@pytest.mark.asyncio
async def test_a_page_full_of_furniture_does_not_win_on_count(monkeypatch):
    """A canvas board has no text in it, and plenty around it.

    LIVE 2026-08-30 on play2048.co: "the page says 5x7 with 10 thing(s);
    looking at it said 4x5 with 8" — the ten were a score, a best, a New Game
    and a footer, and preferring them threw the board away.
    """
    furniture = (
        (0.02, 0.40, "24"), (0.02, 0.58, "6068"), (0.05, 0.50, "2048"),
        (0.95, 0.20, "Give Feedback"), (0.97, 0.50, "play2048.co"),
        (0.93, 0.50, "New Game"), (0.99, 0.10, "N or R"),
        (0.91, 0.70, "1"), (0.91, 0.80, "2"), (0.91, 0.90, "3"),
    )

    async def says():
        return furniture

    monkeypatch.setattr("core.perception.what_the_page_says.what_the_page_says", says)
    #: A screen reading that IS the board, laid out.
    board = {
        "layout": [
            {"text": str(2 ** ((r + c) % 3 + 1)), "center_x": 0.20 + 0.16 * c,
             "center_y": 0.30 + 0.16 * r}
            for r in range(4)
            for c in range(4)
        ]
    }
    got = await _the_best_reading_available(board, None, like=None, in_a_browser=True)
    assert got.occupied() == 16
