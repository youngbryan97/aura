"""A rule a person states for picking from a list is followed exactly, whatever the list is of."""
from __future__ import annotations

import datetime

import pytest

from core.language.picking_by_a_rule import a_picking_rule

pytestmark = pytest.mark.unit

_GAMES = [f"game {n}" for n in range(56)]
_AT_7_23 = datetime.datetime(2026, 10, 6, 7, 23, 5)

_THREE = ("Go to the museum page and play three of the games, one after another, and win each one. To pick them: number the games on the list "
          "from 0, starting with the first one. Take the current minute of the hour, divide it by how many games there are, and play the game "
          "whose number is the remainder. When that game is over, go back to the list, add 19 to the number, take the remainder again, and play "
          "that game. Then add 19 once more for the third game.")


def test_three_picks_by_the_minute_and_nineteen_more_each_time():
    rule = a_picking_rule(_THREE)
    assert rule is not None and rule.start == "minute" and rule.counted_from == 0 and rule.picks == 3
    picks = rule.worked_out(_GAMES, _AT_7_23)
    assert [p.index for p in picks] == [23, 42, 5]
    assert [p.item for p in picks] == ["game 23", "game 42", "game 5"]
    assert "the minute is 23" in picks[0].working and "61 divided by 56 leaves 5" in picks[2].working


def test_every_item_has_its_chance_over_the_hour():
    rule = a_picking_rule(_THREE)
    firsts = {rule.worked_out(_GAMES, _AT_7_23.replace(minute=m))[0].index for m in range(60)}
    assert firsts == set(range(56))


def test_the_last_digit_and_counting_forward_from_the_first():
    rule = a_picking_rule("Take the last digit of the current time, add 2, and count that many games forward from the first game on the list. "
                          "Open that game, play it, and beat it.")
    assert rule is not None and rule.picks == 1
    assert [p.index for p in rule.worked_out(_GAMES, _AT_7_23)] == [5]  # 3 + 2 forward from the first


@pytest.mark.parametrize(("words", "items", "picked"), [
    ("Number the songs from 1. Take today's date, divide it by how many songs there are, and play the song whose number is the remainder.",
     ["a", "b", "c", "d", "e", "f", "g"], ["f"]),       # the 6th: 6 leaves 6 of 7, counted from 1
    ("Start at 4, multiply it by 3, take the remainder by how many recipes there are, and cook that recipe. Then add 2 and cook two more the same way.",
     [f"r{n}" for n in range(10)], ["r1", "r3", "r5"]),  # 12 → 2 → second recipe counted from 1, then +2 each
])
def test_other_lists_by_other_rules(words, items, picked):
    rule = a_picking_rule(words)
    assert rule is not None
    assert [p.item for p in rule.worked_out(items, _AT_7_23)] == picked


@pytest.mark.parametrize("words", [
    "Play the first game on the list.",
    "Pick any three games you like and play them.",
    "What time is it?",
    "Add 19 to my order.",
])
def test_words_with_no_rule_give_none(words):
    assert a_picking_rule(words) is None
