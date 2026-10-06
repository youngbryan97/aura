"""On a game's menus she presses what goes on first, and a new screen's labels are new controls.

LIVE-like 2026-10-05, Toonami Tunnel Rush from its Internet Archive copy: she
clicked every label of the rules screen ("Accelerate", "Brake", "750 pts") in
turn before "Next", and on the second page passed "Next" over as already tried.
"""
from __future__ import annotations

from core.language.a_way_on import how_much_it_leads_on
from core.perception.where_it_responds import Responsive, noticed


def _screen(*labels: str) -> dict:
    return {"layout": [{"text": label, "center_x": 0.1 * n, "center_y": 0.5} for n, label in enumerate(labels, start=1)]}


def test_what_goes_on_is_pressed_before_what_is_read():
    for label in ("Play", "Next", "Start Game", "Click to Play", "Easy", "Nex", "P1ay"):
        assert how_much_it_leads_on(label) > 1.0, label
    for label in ("SCORE", "750 pts", "Speed", "Time Left"):
        assert how_much_it_leads_on(label) < 1.0, label


def test_a_click_that_leads_to_a_new_screen_makes_its_labels_new():
    state = Responsive()
    rules_one = _screen("Avoid the walls", "collect points", "Next")
    rules_two = _screen("Accelerate", "Brake", "Launch Bomb", "Next")
    noticed(state, rules_one, rules_two, worked=True, acting='click "Next"')
    assert 'click "Next"' not in state.tried


def test_a_screen_that_only_moves_keeps_what_was_tried():
    state = Responsive()
    playing = _screen("SCORE", "0015250", "Next")
    later = _screen("SCORE", "0015300", "Next")
    noticed(state, playing, later, worked=True, acting='click "SCORE"')
    assert 'click "SCORE"' in state.tried


def test_what_a_label_did_last_time_is_not_carried_to_a_new_session():
    from core.agency.what_i_can_do_here import WhatWorksHere

    remembered = {"told": ["up"], "did_something": {"up": 3}, "did_nothing": {'click "Play"': 4, "left": 3}}
    now = WhatWorksHere.from_memory(remembered, told=["up"])
    assert 'click "Play"' not in now.did_nothing and now.did_nothing.get("left") == 3


def test_a_label_that_goes_on_is_offered_beside_working_keys():
    from core.agency.what_i_can_do_here import WhatWorksHere

    here = WhatWorksHere(told=("left", "right"))
    here.looked_at(['click "Easy"', 'click "Hard"', 'click "SCORE"'])
    here.looked_at(['click "Easy"', 'click "Hard"', 'click "SCORE"'])
    offered = here.available()
    assert offered[:2] == ("left", "right")
    assert 'click "Easy"' in offered and 'click "SCORE"' not in offered


def test_when_no_key_moves_the_screen_its_labels_are_offered():
    """LIVE-like 2026-10-05, a level select: up had worked in the game before, so it was never written
    off, and "Strike Em Out" was never offered because it does not read as a way on."""
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

    here = WhatWorksHere(told=("up", "down", "left", "right", "space"))
    here.tried("up", True)  # it worked on the screen before this one
    here.looked_at([a_click_on("Strike Em Out"), a_click_on("Road Rage")])
    here.looked_at([a_click_on("Strike Em Out"), a_click_on("Road Rage")])
    assert a_click_on("Strike Em Out") not in here.available()
    for key in ("down", "left", "up"):
        here.tried(key, False)
    assert a_click_on("Strike Em Out") in here.available()
    here.tried("up", True)  # the screen answered: the keys are worth pressing again
    assert a_click_on("Strike Em Out") not in here.available()


def test_short_lines_one_over_another_are_items_not_prose():
    from core.skills.screen_pursuit_bearings import things_to_click, what_it_says

    def line(text, y, h=0.07):
        return {"text": text, "x": 0.02, "y": y, "width": 0.25, "height": h}

    level_select = {"layout": [line("Road Rage", 0.395, 0.09), line("Backyard", 0.55), line("Beatdown", 0.638)]}
    offered = things_to_click(level_select, drawn_where=(0, 0, 1, 1))
    assert 'click "Road Rage"' in offered and 'click "Backyard"' in offered
    rules = {"layout": [line("Bump into the cars and make them spin", 0.3), line("until they reach the parking spot", 0.38)]}
    assert things_to_click(rules, drawn_where=(0, 0, 1, 1)) == ()
    assert what_it_says(rules, drawn_where=(0, 0, 1, 1)).startswith("Bump into")


def test_a_line_is_read_left_to_right_whatever_its_pieces_heights():
    """LIVE 2026-10-06 "First to 5 points wins." was said "points wins. First to 5"."""
    from core.skills.screen_pursuit_bearings import what_it_says

    def piece(text, x, y, h, w=0.3):
        return {"text": text, "x": x, "y": y, "width": w, "height": h}

    rules = {"layout": [piece("Keep the ball from getting past you.", 0.1, 0.40, 0.04, 0.8),
                        piece("First to 5", 0.1, 0.452, 0.05), piece("points wins.", 0.42, 0.45, 0.04)]}
    assert what_it_says(rules, drawn_where=(0, 0, 1, 1)).endswith("First to 5 points wins.")
