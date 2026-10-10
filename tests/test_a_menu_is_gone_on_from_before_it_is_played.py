"""A screen that offers a way on is a menu, whatever moves on it: it is gone on from before it is played.

LIVE 2026-10-08 a title screen's fireflies were played as a game for fifty-three
seconds, with START and HOW TO PLAY written under them.
"""
from __future__ import annotations

import pytest

from core.language.a_way_on import offers_a_way_on
from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens


@pytest.mark.unit
def test_a_screen_with_a_way_on_is_a_menu_and_a_scoreboard_is_not():
    assert offers_a_way_on("REGULAR SHOW ALL NIGHTER START HOW TO PLAY CREDITS")
    assert offers_a_way_on("Press space to play")
    assert not offers_a_way_on("SCORE 120 LIVES 3")


@pytest.mark.unit
def test_play_waits_for_the_first_reading_and_twice_at_most_for_a_menu():
    playing = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="", ends_at=0.0)
    assert playing._a_menu_first()  # nothing read yet
    playing.read({"text": "ALL NIGHTER START HOW TO PLAY"})
    assert playing._a_menu_first() and playing._a_menu_first()
    assert not playing._a_menu_first()  # its way on did nothing twice: what moves on it is played
    playing.read({"text": "SCORE 0 FIREFLIES 3"})
    assert not playing._a_menu_first()


@pytest.mark.unit
def test_a_menu_is_never_shot_at_and_a_shooter_aimed_with_the_mouse_is_no_world_of_shots():
    import asyncio
    import time

    from core.agency.playing_by_shots import sends_by_letting_go

    # LIVE 2026-10-09: "Mouse to aim" and a menu reading "Enter Code or Play" had her letting go of shots at the menu.
    assert not sends_by_letting_go("Arrow keys or WASD to move, Space to attack, Mouse to aim. Collect power-ups.")
    playing = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="", ends_at=time.monotonic() + 60)
    playing.keep["counsel"] = "Click and hold, then release to launch."
    playing.read({"text": "Enter Code or Play", "layout": [{"text": "Enter Code"}, {"text": "or"}, {"text": "Play"}]})
    assert playing.way_on_shown
    asyncio.run(playing._by_shots(time.monotonic()))
    assert not playing.stretches
    # Rules in a sentence that says "let go" are not a way on.
    playing.read({"text": "Drag back and let go to putt", "layout": [{"text": "Drag back and let go to putt it into the hole"}]})
    assert not playing.way_on_shown


@pytest.mark.unit
def test_a_screen_that_says_it_is_being_made_ready_is_waited_out_before_anything_on_it_is_pressed(monkeypatch):
    # LIVE 2026-10-09 an archive's "Downloading game metadata..." screen had its logo clicked to see what it did.
    import asyncio

    import core.skills.screen_pursuit_as_it_happens as reflexes_module
    from core.language.a_way_on import says_it_is_being_made_ready

    assert says_it_is_being_made_ready("INTERNET ARCHIVE Downloading game metadata...")
    assert says_it_is_being_made_ready("Loading 40%") and not says_it_is_being_made_ready("Powering the treehouse is hard work")
    monkeypatch.setattr(reflexes_module, "READY_LOOK_EVERY_S", 0.0)
    monkeypatch.setattr(reflexes_module, "_said_while_playing", lambda line: None)
    screens = iter([{"text": "INTERNET ARCHIVE Downloading game metadata..."}, {"text": "Launching emulator"},
                    {"text": "Flight of the Hamsters PLAY", "layout": [{"text": "PLAY"}]}])

    async def look():
        return next(screens)

    playing = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="", ends_at=0.0)
    first = {"text": "INTERNET ARCHIVE Downloading game metadata..."}
    playing.read(first)
    seen = asyncio.run(reflexes_module._waited_out_while_made_ready(playing, look, first))
    assert seen["text"].endswith("PLAY")
