"""What she reads off a screen is said in words of the language: a misread letter mended, a non-word not said.

LIVE 2026-10-07 the readouts of two games' screens were said aloud as read: "Leuel 1, paint of the 50." and
"Ini 0, iin 0, x 5."
"""
from __future__ import annotations

import pytest

from core.language.words_of_the_language import the_word, words_of

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("read", "said"), [
    ("Leuel", "level"), ("Iives", "lives"), ("Sc0re", "score"), ("SCORE", "score"), ("Time Left", "time left"),
    ("Paint supply", "paint supply"), ("Coins", "coins"), ("Ammo", "ammo"), ("Snacks", "snacks"),
])
def test_a_name_read_off_a_screen_is_said_as_the_word_it_is(read, said):
    assert words_of(read) == said


@pytest.mark.parametrize("read", ["Ini", "iin", "x", "Ln", "paint of the", "of the score", "Wxqzt", "a b c d"])
def test_what_is_not_a_name_in_words_is_not_said(read):
    assert words_of(read) is None


def test_a_short_word_is_not_mended_into_another():
    assert the_word("lap") == "lap" and the_word("ihl") is None


@pytest.mark.parametrize(("text", "readout"), [
    ("00003900", True), ("x4", True), ("*3", True), ("100", True), ("0", True),   # a score, lives, counts: to read
    ("1", False), ("12", False), ("PLAY", False), ("Easy", False), ("Level 2", False),  # a choice, a control
])
def test_a_readout_is_not_taken_for_a_control(text, readout):
    """LIVE 2026-10-07 she clicked a game's score ("00003900") and its lives ("x4") as if they were buttons."""
    from core.skills.screen_pursuit_bearings import _a_readout, things_to_click

    assert _a_readout(text) is readout
    seen = {"layout": [{"text": text, "x": 0.4, "y": 0.4, "width": 0.1, "height": 0.05}]}
    assert bool(things_to_click(seen, drawn_where=True)) is not readout


def test_a_line_said_a_moment_ago_is_not_said_again_in_the_same_game():
    """LIVE 2026-10-07 "The black things are worth shooting." three times in two seconds; "Space fires." over and over."""
    from core.agency.playing_as_it_happens import REPEAT_AFTER_S, _Run, _say

    heard: list[str] = []
    run = _Run(keys=["space"], began=0.0)
    for at, line in ((1.0, "Space fires."), (5.0, "Space fires."), (9.0, "The black things are worth shooting."),
                     (9.5, "The black things are worth shooting."), (1.0 + REPEAT_AFTER_S + 1, "Space fires.")):
        _say(run, heard.append, line, at, once="")
        run.said_at = -100.0  # pacing between lines is another rule; here only repetition is measured
    assert heard == ["Space fires.", "The black things are worth shooting.", "Space fires."]
