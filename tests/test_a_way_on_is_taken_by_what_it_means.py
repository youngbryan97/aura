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


def test_a_surface_of_controls_is_pressed_by_meaning_whatever_her_keys_did_before(screen):
    """LIVE 20 November 2026 her record of En Croissant said her keys had done something there, so nothing on its
    start page was pressed and she tried the arrow keys on it. What the window shows now decides, not that record."""
    seen = {"ok": True, "window_number": 7,
            "layout": [{"text": t, "center_x": x, "center_y": y} for t, x, y in START_PAGE]}
    keep: dict = {}
    assert asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, keep, nothing_answered=False, seen=seen))
    assert screen[-1] == "Play" and keep["surface_is_controls"] and not keep["way_on_found_nothing"]


def test_numbered_tiles_with_a_button_are_a_world_not_a_surface_of_controls():
    tiles = [{"text": t} for t in ("2", "4", "8", "2", "16", "4", "32", "Score 120", "New Game")]
    assert not way_on._a_surface_of_controls(tiles)
    assert way_on._a_surface_of_controls([{"text": t} for t, _, _ in START_PAGE])


def test_a_world_her_keys_answer_in_is_left_to_her_keys(screen):
    seen = {"ok": True, "window_number": 7, "layout": [{"text": t} for t in ("2", "4", "8", "16", "New Game")]}
    keep: dict = {}
    assert not asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, keep, nothing_answered=False, seen=seen))
    assert screen == [] and not keep["surface_is_controls"]


def test_the_name_of_the_application_she_is_in_is_no_way_on(screen):
    asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, {}, nothing_answered=True))
    assert "En Croissant" not in screen


def test_a_control_stands_out_only_when_it_leads_by_more_than_the_rest_trail():
    assert way_on._stands_out([0.86, 0.78, 0.72, 0.68]) == 0
    assert way_on._stands_out([0.50, 0.49, 0.45, 0.40]) is None
    assert np.isclose(len(way_on._controls([{"text": "a", "x": 0.3, "y": 0.2}, {"text": "b", "x": 0.31, "y": 0.25},
                                            {"text": "c", "x": 0.7, "y": 0.2}])), 2)


#: The game's setup page as she read it, 20 November 2026.
SETUP_PAGE = [
    ("En Croissant", 0.13, 0.02), ("New Game", 0.18, 0.06), ("White", 0.63, 0.13), ("Human", 0.65, 0.19),
    ("Black", 0.87, 0.13), ("Engine", 0.84, 0.19), ("Player", 0.58, 0.26), ("Time Settings", 0.74, 0.31),
    ("Time", 0.64, 0.37), ("Unlimited", 0.84, 0.37), ("Increment", 0.78, 0.43), ("Start game", 0.76, 0.49),
    ("Starting Position", 0.62, 0.56),
]


def test_where_nothing_stands_out_she_presses_what_her_understanding_names(monkeypatch, screen):
    """"Start game" is not about chess against an engine, and is what starts one: her model, given the labels on the
    page as the only answers, names it."""
    layout = [{"text": t, "center_x": x, "center_y": y} for t, x, y in SETUP_PAGE]
    pressed: list[str] = []

    async def read_screen(app, over=None):
        return {"layout": list(layout), "bounds": [0, 0, 1000, 800]}

    async def click(x, y, **_kwargs):
        pressed.append(next(r["text"] for r in layout if (r["center_x"], r["center_y"]) == (x, y)))
        layout[:] = [{"text": "Resign", "center_x": 0.6, "center_y": 0.5}]
        return True

    asked: list[str] = []

    async def her_model(ask, prompt, schema, most, *, matters):
        asked.append(prompt)
        return schema(press="Start game")

    monkeypatch.setattr(pursuit, "read_screen", read_screen)
    monkeypatch.setattr(surface, "click_normalized", click)
    monkeypatch.setattr(way_on, "_stands_out", lambda scores: None)
    monkeypatch.setattr("core.cognition.what_things_are.asked_patiently", her_model)
    assert asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, {}, nothing_answered=True))
    assert pressed == ["Start game"]
    assert "Start game" in asked[0] and "En Croissant" not in asked[0].split("can be pressed is this:")[1]


def test_where_the_way_on_does_not_answer_the_next_best_is_not_pressed(monkeypatch, screen):
    """LIVE 21 November 2026 "Play Chess" did not answer a press made as the screen was unlocked, and with it set aside
    "Puzzles: Train your chess skills" stood out and was pressed. The next best is another act, not a way on."""
    layout = [{"text": t, "center_x": x, "center_y": y} for t, x, y in START_PAGE]
    pressed: list[str] = []

    async def read_screen(app, over=None):
        return {"layout": list(layout), "bounds": [0, 0, 1000, 800]}

    async def a_press_that_does_nothing(x, y, **_kwargs):
        pressed.append(next(r["text"] for r in layout if (r["center_x"], r["center_y"]) == (x, y)))
        return True

    monkeypatch.setattr(pursuit, "read_screen", read_screen)
    monkeypatch.setattr(surface, "click_normalized", a_press_that_does_nothing)
    monkeypatch.setattr(way_on, "LOOKED_EVERY_S", 0.0)
    keep: dict = {}
    for _ in range(4):
        assert not asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, keep, nothing_answered=True))
    assert pressed and all(text.startswith("Play") for text in pressed), pressed
    assert keep["ways_on_taken"].count(keep["ways_on_taken"][0]) == way_on.PRESSES_OF_A_WAY_ON


def test_a_retry_uses_the_matching_native_control_when_the_pointer_did_not_change_the_view(monkeypatch):
    from types import SimpleNamespace
    import sys

    from core.capabilities import window_server
    from core.skills import their_player

    receipt = []
    window = SimpleNamespace(number=7, owner="Example", pid=5, bounds=(32, 48, 800, 600))
    element = {"AXRole": "AXButton", "AXTitle": "Continue", "AXEnabled": True}

    def hit_test(pid, x, y, out):
        assert (pid, x, y) == (5, 432, 348)
        return 0, element

    monkeypatch.setitem(sys.modules, "ApplicationServices", SimpleNamespace(
        AXUIElementCopyElementAtPosition=hit_test, AXUIElementCreateApplication=lambda pid: pid,
    ))
    monkeypatch.setattr(window_server, "window_of", lambda *args, **kwargs: window)
    monkeypatch.setattr(window_server, "front_owner", lambda: window.owner)
    monkeypatch.setattr(their_player, "_read", lambda node, name: node.get(name))
    monkeypatch.setattr(their_player, "press", lambda node: receipt.append("native") or True)

    async def pointer(*args, **kwargs):
        receipt.append("pointer")
        return True

    monkeypatch.setattr(surface, "click_normalized", pointer)
    region = {"x": 0.5, "y": 0.5, "text": "Continue"}
    reading = {"window_number": 7, "bounds": window.bounds}

    assert asyncio.run(way_on._touch_control(region, reading, "Example", retry=False))
    assert asyncio.run(way_on._touch_control(region, reading, "Example", retry=True))
    element["AXRole"] = "AXStaticText"
    assert asyncio.run(way_on._touch_control(region, reading, "Example", retry=True))
    assert receipt == ["pointer", "native", "pointer"]


def test_a_missing_read_after_a_press_cannot_verify_a_change(monkeypatch):
    async def touch(*args, **kwargs):
        return True

    async def lost_read(*args, **kwargs):
        return None

    monkeypatch.setattr(way_on, "_touch_control", touch)
    monkeypatch.setattr(way_on, "_a_reading", lost_read)
    monkeypatch.setattr(way_on, "SHOWS_IN_S", 0.0)
    region = {"x": 0.5, "y": 0.5, "text": "Continue"}

    assert not asyncio.run(way_on._pressed(region, [region], "Example", {}, {}, None, True))
