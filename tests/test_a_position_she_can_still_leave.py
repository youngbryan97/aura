"""How much can still happen from here, as something she judges a state by.

The one thing every account of playing well has in common, across things that
look nothing like each other. Position before submission: take the position
from which many attacks are open and few escapes are, and only then attack. Do
not pick up your dribble. Keep your base, keep your feet under you, keep a
hand free. Each of those is the same instruction — do not enter a state you
cannot leave — and none of them mentions the goal at all.

What she had instead was room: the share of the thing that is empty. Space is
a fact about the picture and this is a fact about what she can still do, and
the two come apart exactly when it matters. A thing can be half empty and
locked, and nearly full and live.

Measured through the rule she worked out rather than declared, so it assumes
nothing about any particular world.
"""

from __future__ import annotations

from core.agency.how_good_is_this import _freedom, terms
from core.perception.how_it_moves import RULES, HowItMoves
from core.perception.where_it_responds import what_is_there

ACTS = ("left", "right", "up", "down")


def _board(text: str):
    return what_is_there({"ok": True, "text": text, "layout": [], "bounds": []}, None)


def _one_that_knows(rule_name: str = "slides and combines") -> HowItMoves:
    """A model that has worked out how this world moves."""
    knows = HowItMoves()
    for rule in RULES:
        knows.tried[rule.name] = knows.tried_when_it_moved[rule.name] = 20
        got = 20 if rule.name == rule_name else 0
        knows.right[rule.name] = knows.right_when_it_moved[rule.name] = got
    knows.seen = knows.moved = 20
    return knows


def test_a_locked_position_is_worth_less_than_a_live_one():
    knows = _one_that_knows()
    locked = _board("2 4 2 4\n4 2 4 2\n2 4 2 4\n4 2 4 2")
    live = _board("2 2 4 8\n16 32 64 128\n256 512 1024 2\n4 8 16 32")
    assert _freedom(locked, knows, ACTS) < _freedom(live, knows, ACTS)


def test_a_position_nothing_can_be_done_from_scores_nothing():
    knows = _one_that_knows()
    assert _freedom(_board("2 4 2 4\n4 2 4 2\n2 4 2 4\n4 2 4 2"), knows, ACTS) == 0.0


def test_two_acts_with_the_same_result_are_one_option():
    """An option wearing two names is not two ways out.

    The act duplicated here has to be one that WORKS. The version of this
    test that shipped duplicated an act that does nothing in the state it
    chose, so both sides were zero and the invariant it names was never
    exercised — while the implementation it guarded halved the score for a
    repeated name, 1.0 down to 0.5, on a state where nothing had changed but
    the spelling of the list.
    """
    knows = _one_that_knows()
    # Sliding left moves this; nothing about the world changes when the list
    # says so twice.
    one = _board(". . . 2\n. . . .\n. . . .\n. . . .")
    assert _freedom(one, knows, ("left",)) == 1.0
    assert _freedom(one, knows, ("left", "left")) == 1.0
    assert _freedom(one, knows, ("left", "left", "left")) == 1.0


def test_a_dead_act_costs_freedom_and_a_second_dead_act_does_not():
    """Going nowhere is one option too, however many names it wears."""
    knows = _one_that_knows()
    one = _board(". . . 2\n. . . .\n. . . .\n. . . .")
    # Left moves it, right and up leave it exactly where it is.
    assert _freedom(one, knows, ("left", "right")) == 0.5
    assert _freedom(one, knows, ("left", "right", "up")) == 0.5


def test_naming_an_alias_never_lowers_freedom():
    """The property the fraction has to have, over every act in this world."""
    knows = _one_that_knows()
    for text in (
        ". . . 2\n. . . .\n. . . .\n. . . .",
        "2 2 4 8\n16 32 64 128\n256 512 1024 2\n4 8 16 32",
        "2 . . .\n. 4 . .\n. . 8 .\n. . . 16",
    ):
        board = _board(text)
        plain = _freedom(board, knows, ACTS)
        for act in ACTS:
            doubled = _freedom(board, knows, (*ACTS, act))
            assert doubled == plain, f"{act} repeated changed freedom on {text!r}"


def test_without_a_model_she_claims_nothing():
    """She cannot say what she can still do in a world she has not worked out."""
    assert _freedom(_board("2 4\n8 16"), None, ACTS) == 0.0
    assert _freedom(_board("2 4\n8 16"), HowItMoves(), ACTS) == 0.0


def test_with_no_acts_there_is_no_freedom_to_measure():
    assert _freedom(_board("2 4\n8 16"), _one_that_knows(), ()) == 0.0


def test_it_is_one_of_the_things_she_judges_a_state_by():
    said = terms(_board("2 4\n8 16"), knows=_one_that_knows(), acts=ACTS)
    assert "freedom" in said
    assert 0.0 <= said["freedom"] <= 1.0


def test_a_state_read_without_a_model_still_scores_the_rest():
    """Adding this must not break judging a situation she cannot model."""
    said = terms(_board("2 4\n8 16"))
    assert said["freedom"] == 0.0
    assert set(said) >= {"nearness", "line", "room", "order", "smoothness", "freedom"}
