"""A word on a button is a label, not what a place says to do: a start button that says "LAUNCH" does not make a game
one of sending things by a press let go.

LIVE 2026-10-10 a tunnel racer steered with the arrow keys had a "LAUNCH" button, and she pulled and let go at it for
its first run. Nothing here is any one game.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_a_sending_word_counts_in_an_instruction_and_not_as_a_label_on_its_own():
    from core.agency.playing_by_shots import sends_by_letting_go
    from core.skills.screen_pursuit_as_it_happens import _without_labels

    screen = "Choose the difficulty level LAUNCH You have two minutes to collect as many points as possible."
    assert not sends_by_letting_go(_without_labels(screen, {"LAUNCH", "Next"}))
    assert not sends_by_letting_go("LAUNCH")
    assert sends_by_letting_go("Pull back and release to launch the bird.")
    assert sends_by_letting_go(_without_labels("Pull back and release to launch the bird. PLAY", {"PLAY"}))
