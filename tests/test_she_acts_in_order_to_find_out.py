"""Choosing a move for what it would tell her, not only for where it leads.

Looking ahead asks which move is best under the rule she is using. It cannot
ask which rule is right, so a loop with only that question does whatever looks
best now and stays unsure of the world for as long as the world lets it. Being
right about the rule improves every move after this one; a slightly better
position improves only this one.
"""

from __future__ import annotations

from core.agency.looking_ahead import worth_finding_out
from core.perception.how_it_moves import ENOUGH_TO_TRUST, RULES, HowItMoves
from core.perception.where_it_responds import what_is_there


def _board(text: str):
    return what_is_there({"ok": True, "text": text, "layout": [], "bounds": []}, None)


def test_with_nothing_ruled_out_some_acts_settle_more_than_others():
    knows = HowItMoves()
    crowded = _board("2 2 2 2\n2 2 2 2\n2 2 2 2\n2 2 2 2")
    sparse = _board("2 . . .\n. . . .\n. . . .\n. . . 2")
    assert knows.what_this_would_settle(crowded, "left") > knows.what_this_would_settle(
        sparse, "left"
    ), "a board where the rules must disagree told her no more than one where they need not"


def test_a_settled_rule_leaves_nothing_to_find_out():
    """The mechanism turns itself off, so there is nothing to turn off."""
    knows = HowItMoves()
    # One rule right about everything, every other rule wrong: the question
    # is settled, and settled questions are not worth acting on.
    survivor = RULES[0].name
    for rule in RULES:
        knows.tried[rule.name] = ENOUGH_TO_TRUST * 3
        knows.tried_when_it_moved[rule.name] = ENOUGH_TO_TRUST * 3
        got = ENOUGH_TO_TRUST * 3 if rule.name == survivor else 0
        knows.right[rule.name] = got
        knows.right_when_it_moved[rule.name] = got
    knows.seen = ENOUGH_TO_TRUST * 3
    knows.moved = ENOUGH_TO_TRUST * 3
    assert len(knows.still_standing()) == 1
    assert knows.what_this_would_settle(_board("2 2 2 2\n2 2 2 2\n2 2 2 2\n2 2 2 2"), "left") == 0.0


def test_a_rule_that_has_been_wrong_too_often_stops_standing():
    knows = HowItMoves()
    out = RULES[0].name
    knows.tried[out] = ENOUGH_TO_TRUST * 2
    knows.right[out] = 0
    assert out not in {rule.name for rule in knows.still_standing()}


def test_too_little_evidence_keeps_a_rule_standing():
    """What she has no evidence about is exactly what is worth finding out."""
    knows = HowItMoves()
    one = RULES[0].name
    knows.tried[one] = ENOUGH_TO_TRUST - 1
    knows.right[one] = 0
    assert one in {rule.name for rule in knows.still_standing()}


def test_finding_out_is_scored_even_with_no_futures_to_compare():
    """With no model to prefer anything by, this is all she has to go on."""
    knows = HowItMoves()
    board = _board("2 2 2 2\n2 2 2 2\n2 2 2 2\n2 2 2 2")
    told = worth_finding_out(knows, board, ["left", "right", "up", "down"], None)
    assert told, "she had nothing to choose by and was given nothing"
    assert all(value > 0.0 for value in told.values())


def test_what_knowing_is_worth_comes_from_the_futures_she_can_see():
    """Not a constant: where every move leads to the same place, knowing the
    rule buys nothing, and where they differ sharply it buys a lot."""
    knows = HowItMoves()
    board = _board("2 2 2 2\n2 2 2 2\n2 2 2 2\n2 2 2 2")
    acts = ["left", "right", "up", "down"]
    flat = dict.fromkeys(acts, (1.0, "same"))
    wide = {"left": (0.0, "a"), "right": (10.0, "b"), "up": (5.0, "c"), "down": (5.0, "d")}
    assert not worth_finding_out(knows, board, acts, flat), (
        "settling the rule was priced above nothing where it could change nothing"
    )
    priced = worth_finding_out(knows, board, acts, wide)
    assert max(priced.values()) > max(
        worth_finding_out(knows, board, acts, None).values()
    )


def test_a_thing_with_no_model_at_all_is_left_alone():
    assert worth_finding_out(None, _board("2 2"), ["left"], None) == {}
    assert worth_finding_out(HowItMoves(), None, ["left"], None) == {}
