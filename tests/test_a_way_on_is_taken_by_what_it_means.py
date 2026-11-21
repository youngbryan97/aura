"""Asked to do something in an application, she presses the control whose words mean it, not the arrow keys.

LIVE 20 November 2026 she opened En Croissant on its start page of cards ("Play Chess: play against an engine or a
friend", "Analysis Board", "Puzzles"...), took the grid for a board, pressed arrows at it and gave up. The encoder here
is a stand-in that counts shared words, so the test is of the choosing, not of a model.
"""
from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest

import core.skills.screen_pursuit as pursuit
import core.skills.screen_pursuit_a_way_on as way_on
import core.skills.screen_pursuit_surface as surface

#: The start page as her reading gave it, 20 November 2026 (shares of the window).
START_PAGE = [
    ("En Croissant", 0.16, 0.06), ("New Tab", 0.14, 0.10),
    ("Play Chess", 0.31, 0.26), ("Play against an engine or a friend", 0.31, 0.29), ("Play", 0.31, 0.35),
    ("Analysis Board", 0.73, 0.26), ("Analyze a game or position", 0.73, 0.29), ("Open", 0.73, 0.35),
    ("New Repertoire", 0.31, 0.53), ("Build and practice your opening repertoire", 0.31, 0.56), ("Create", 0.31, 0.62),
    ("Import Game", 0.73, 0.53), ("Import a game from a PGN", 0.73, 0.56), ("Import", 0.73, 0.62),
    ("Puzzles", 0.31, 0.80), ("Train your chess skills", 0.31, 0.83), ("Train", 0.31, 0.89),
]
ASKED = "Open En Croissant and play a game of chess against Stockfish. Tell me each move as you go."


def _words_in_common(texts):
    vocabulary = sorted({w for t in texts for w in re.findall(r"[a-z]+", t.lower())})
    return [[float(w in re.findall(r"[a-z]+", t.lower())) + 1e-3 for w in vocabulary] for t in texts]


@pytest.fixture
def screen(monkeypatch):
    layout = [{"text": t, "center_x": x, "center_y": y} for t, x, y in START_PAGE]
    pressed: list[str] = []

    async def read_screen(app, over=None):
        return {"layout": list(layout), "bounds": [0, 0, 1000, 800]}

    async def click(x, y, **_kwargs):
        pressed.append(next(r["text"] for r in layout if (r["center_x"], r["center_y"]) == (x, y)))
        if pressed[-1] == "Play":
            layout[:] = [{"text": "Start game", "center_x": 0.75, "center_y": 0.5}]
        return True

    monkeypatch.setattr(pursuit, "read_screen", read_screen)
    monkeypatch.setattr(pursuit, "_tell", lambda line: None)
    monkeypatch.setattr(surface, "click_normalized", click)
    monkeypatch.setattr("core.language.learned_matcher.embed_sentences", _words_in_common)
    monkeypatch.setattr(way_on, "SHOWS_IN_S", 0.0)

    async def no_window(_app):  # the real desktop is never read in a test
        return None

    monkeypatch.setattr(way_on, "_its_own_pixels_read", no_window)
    return pressed


def test_the_card_that_means_what_was_asked_is_pressed(screen):
    assert asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, {}, nothing_answered=True))
    assert screen[-1] == "Play"
    assert all(text.startswith("Play") for text in screen)


def test_nothing_is_pressed_once_what_she_does_has_changed_the_screen(screen):
    assert not asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, {}, nothing_answered=False))
    assert screen == []


def test_the_name_of_the_application_she_is_in_is_no_way_on(screen):
    asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, {}, nothing_answered=True))
    assert "En Croissant" not in screen


def test_a_control_stands_out_only_when_it_leads_by_more_than_the_rest_trail():
    assert way_on._stands_out([0.86, 0.78, 0.72, 0.68]) == 0
    assert way_on._stands_out([0.50, 0.49, 0.45, 0.40]) is None
    assert np.isclose(len(way_on._controls([{"text": "a", "x": 0.3, "y": 0.2}, {"text": "b", "x": 0.31, "y": 0.25},
                                            {"text": "c", "x": 0.7, "y": 0.2}])), 2)
