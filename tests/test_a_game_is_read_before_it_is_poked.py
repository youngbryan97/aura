"""What a game says is read; what it offers is tried; what changes is a reading.

LIVE 2026-10-03 04:46-04:49, Cow and Chicken: Ballet Parking, played from her
chat on a page in her own browser:
- she clicked the game's instructions a line at a time ("Bump into the cars
  and make them spin like ballistic", "ballerinas until they reach the parking
  spot that matches");
- with the game's clock running, every look read it differently ("TImE 01:50",
  "TTE 0152", "TITE 040"), each was a control she had never tried, and she
  clicked the clock;
- the arrow keys did nothing on the title screen and were never pressed again
  in the game they drive;
- the chat got a line a second, "Clicking ... — this is the one that would
  settle how this moves".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _line(text: str, y: float, *, x: float = 0.1, width: float = 0.6, height: float = 0.04) -> dict:
    return {"text": text, "x": x, "y": y, "width": width, "height": height}


def test_the_lines_of_a_paragraph_are_not_offered_as_clicks():
    from core.skills.screen_pursuit_bearings import things_to_click

    layout = [
        _line("Bump into the cars and make them spin like ballistic", 0.30),
        _line("ballerinas until they reach the parking spot that matches", 0.345),
        _line("their color.", 0.39),
        _line("Forward", 0.80, x=0.7, width=0.15),
    ]
    offered = things_to_click({"layout": layout}, drawn_where=(0, 0, 1, 1))
    assert offered == ('click "Forward"',)


def test_buttons_with_room_between_them_are_all_offered():
    from core.skills.screen_pursuit_bearings import things_to_click

    layout = [_line("PLAY", 0.40, width=0.2), _line("INSTRUCTIONS", 0.50, width=0.3)]
    assert len(things_to_click({"layout": layout}, drawn_where=(0, 0, 1, 1))) == 2


def test_writing_that_changes_between_looks_is_not_a_control():
    from core.agency.what_i_can_do_here import WhatWorksHere

    can_do = WhatWorksHere()
    can_do.looked_at(['click "TImE 01:50"', 'click "Forward"'])
    assert can_do.on_screen == ()
    can_do.looked_at(['click "TTE 0152"', 'click "Forward"'])
    assert can_do.on_screen == ('click "Forward"',)


def test_a_click_that_changed_the_screen_gives_dead_keys_another_chance():
    from core.agency.what_i_can_do_here import ENOUGH_TO_JUDGE, WhatWorksHere

    can_do = WhatWorksHere(told=("up",))
    for _ in range(ENOUGH_TO_JUDGE):
        can_do.tried("up", changed=False)
    assert "up" in can_do.dead()
    can_do.tried('click "Forward"', changed=True)
    assert "up" not in can_do.dead()


def test_a_key_that_changed_the_screen_does_not_revive_the_others():
    from core.agency.what_i_can_do_here import ENOUGH_TO_JUDGE, WhatWorksHere

    can_do = WhatWorksHere(told=("up", "space"))
    for _ in range(ENOUGH_TO_JUDGE):
        can_do.tried("space", changed=False)
    can_do.tried("up", changed=True)
    assert "space" in can_do.dead()


def test_in_a_run_a_move_line_waits_for_the_last_to_be_read(monkeypatch):
    from core.skills import screen_pursuit
    from core.skills import screen_pursuit_looking as looking

    said: list[str] = []
    monkeypatch.setattr(screen_pursuit, "_tell", said.append)
    monkeypatch.setattr(screen_pursuit, "_publish_decision", lambda *a, **k: None)
    chosen = SimpleNamespace(rationale="", chosen=None)

    token = looking.MOVES_SAID.set({"at": 0.0, "line": ""})
    try:
        looking._say_intent("left", chosen, out_loud=True)
        looking._say_intent("right", chosen, out_loud=True)
    finally:
        looking.MOVES_SAID.reset(token)
    assert said == ["Going left"]


def _rules_screen(second_line: str = "ballerinas until they reach the parking spot that matches") -> dict:
    return {
        "layout": [
            _line("Bump into the cars and make them spin like ballistic", 0.30),
            _line(second_line, 0.345),
            _line("Forward", 0.80, x=0.7, width=0.15),
        ]
    }


def test_what_a_game_says_is_its_prose_in_reading_order():
    from core.skills.screen_pursuit_bearings import what_it_says

    assert what_it_says(_rules_screen(), drawn_where=(0, 0, 1, 1)) == (
        "Bump into the cars and make them spin like ballistic "
        "ballerinas until they reach the parking spot that matches"
    )
    assert what_it_says(_rules_screen(), drawn_where=None) == ""


def test_the_rules_are_read_out_once_a_run(monkeypatch):
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.skills import screen_pursuit
    from core.skills import screen_pursuit_looking as looking

    said: list[str] = []
    monkeypatch.setattr(screen_pursuit, "_tell", said.append)
    token = looking.MOVES_SAID.set({"at": 0.0, "line": ""})
    try:
        can_do = WhatWorksHere()
        looking._take_in_the_screen(can_do, _rules_screen(), (0, 0, 1, 1), True)
        # A frame later, recognised a little differently: the same words.
        looking._take_in_the_screen(
            can_do, _rules_screen("ballerinas untll they reach the parking spot that matches"),
            (0, 0, 1, 1), True,
        )
    finally:
        looking.MOVES_SAID.reset(token)
    assert len(said) == 1 and said[0].startswith("It says: Bump into the cars")
    assert can_do.on_screen == ('click "Forward"',)


def test_a_title_over_a_button_is_not_a_paragraph():
    """LIVE 2026-10-03 08:15: "It says: GLONFA DOR START", and START was never pressed."""
    from core.skills.screen_pursuit_bearings import things_to_click

    layout = [
        _line("CLONE-A-DOODLE DOO", 0.30, height=0.10),
        _line("START", 0.42, x=0.4, width=0.2, height=0.04),
    ]
    offered = things_to_click({"layout": layout}, drawn_where=(0, 0, 1, 1))
    assert 'click "START"' in offered
