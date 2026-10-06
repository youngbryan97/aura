"""Choices set side by side on one line are each a click, and a thing done again to no end tells less each time.

LIVE 2026-10-06, Toonami Tunnel Rush's difficulty screen: text recognition read
"Easy / Hard" as one piece of writing, she clicked its middle (the stroke
between them), and then pressed right thirty times, because the picture moved
by itself and every press looked as if it had moved it.
"""
from __future__ import annotations

import pytest

from core.skills.screen_pursuit_bearings import things_to_click, where_to_click


def _screen(*lines: tuple[str, float, float, float]) -> dict:
    return {"layout": [{"text": text, "x": x, "y": y, "width": width, "height": 0.06, "center_x": x + width / 2, "center_y": y + 0.03}
                       for text, x, y, width in lines]}


@pytest.mark.unit
@pytest.mark.parametrize(("line", "choices"), [
    ("Easy / Hard", ["Easy", "Hard"]),                     # a game's difficulty
    ("1 Player | 2 Players", ["1 Player", "2 Players"]),   # a game's players
    ("Save · Load · Quit", ["Save", "Load", "Quit"]),      # a program's menu line
    ("Yes / No", ["Yes", "No"]),                           # a dialog's answer
])
def test_each_choice_on_a_line_is_its_own_click_where_it_stands(line, choices):
    seen = _screen(("Choose the", 0.05, 0.1, 0.3), (line, 0.3, 0.5, 0.4))
    clicks = things_to_click(seen, drawn_where=True)
    for choice in choices:
        assert f'click "{choice}"' in clicks
    assert f'click "{line}"' not in clicks
    xs = [where_to_click(seen, choice)[0] for choice in choices]
    assert xs == sorted(xs) and 0.3 < xs[0] < 0.5 < xs[-1] < 0.7  # each where its letters are, none on the stroke
    assert where_to_click(seen, choices[0])[1] == pytest.approx(0.53)


@pytest.mark.unit
@pytest.mark.parametrize("line", ["10 / 20", "Score: 120 / 500", "Speed", "TIME 01:50"])
def test_a_count_or_a_single_label_is_left_as_it_is(line):
    seen = _screen((line, 0.3, 0.5, 0.4))
    assert things_to_click(seen, drawn_where=True) == (f'click "{line}"',)


@pytest.mark.unit
def test_a_thing_done_again_here_to_no_end_tells_less_each_time():
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
    from core.skills.screen_pursuit_decision import _first_what_goes_on

    here = WhatWorksHere(told=("up", "down", "left", "right"))
    menu = [a_click_on("Choose the"), a_click_on("difficulty level")]
    here.looked_at(menu, "")
    first = _first_what_goes_on(here, {"right": 1.0, "up": 1.0})
    for _ in range(6):  # right "moves" the picture each time, and the screen stays itself
        here.looked_at(menu, "")
        here.tried("right", True)
    here.looked_at(menu, "")
    later = _first_what_goes_on(here, {"right": 1.0, "up": 1.0})
    assert first["right"] == first["up"]
    assert later["right"] < later["up"] / 4


@pytest.mark.unit
def test_what_led_on_from_a_screen_is_not_written_down_for_being_done_there():
    from core.agency.where_things_lead import WhereThingsLead

    leads = WhereThingsLead()
    menu = ['click "Easy"', 'click "Hard"']
    leads.looked(menu, "Choose the difficulty level")
    leads.looked(["Score", "Lives"], "Level 1")  # she clicked Easy, and the game began
    leads.acted('click "Easy"', True)
    leads.looked(menu, "Choose the difficulty level")  # lost, and back at the menu
    leads.acted("left", True)
    assert leads.taken_here_to_no_end('click "Easy"') == 0
    leads.looked(menu, "Choose the difficulty level")
    leads.acted('click "Hard"', False)
    assert leads.taken_here_to_no_end('click "Hard"') == 1


@pytest.mark.unit
def test_a_line_she_said_a_moment_ago_is_not_said_again(monkeypatch):
    import core.skills.screen_pursuit as pursuit
    import core.skills.screen_pursuit_looking as looking

    told: list[str] = []
    monkeypatch.setattr(pursuit, "_tell", told.append)
    monkeypatch.setattr(pursuit, "_publish_decision", lambda *a, **k: None)
    monkeypatch.setattr(looking, "_the_last_move_could_be_read", lambda paced: True)
    token = looking.MOVES_SAID.set({})
    try:
        class Chosen:
            rationale = "to see what it does"

        for key in ("right", "right", "right", "left", "right"):
            looking._say_intent(key, Chosen(), out_loud=True)
    finally:
        looking.MOVES_SAID.reset(token)
    assert told == ["Going right — to see what it does", "Going left — to see what it does"]
