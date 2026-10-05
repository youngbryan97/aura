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
