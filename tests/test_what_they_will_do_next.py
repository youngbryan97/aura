"""Acting against what they are about to do, and knowing who goes first.

Strong players do not answer the thing in front of them. They answer the thing
that will be in front of them, because the opponent is about to swap it out
and the move that beats what is there loses to what is coming. And everything
resolves in speed order, so a single point either way is the difference
between landing a blow and being gone before you swing.

The tests are a deploy race, because it is the same shape.
"""

from __future__ import annotations

from core.cognition.what_they_will_do_next import (
    WhatTheyTendToDo,
    against_what_is_coming,
    worth_going_first,
)

ACTS = ["patch it", "roll back", "wait"]


def _tends() -> WhatTheyTendToDo:
    tends = WhatTheyTendToDo()
    for _ in range(6):
        tends.they_did("the other team", facing="a red build", act="force merge")
    tends.they_did("the other team", facing="a red build", act="revert")
    return tends


def _after(now: str, theirs: str) -> str:
    return "theirs landed" if theirs == "force merge" else now


def _how_good(where: str, act: str) -> float:
    # Rolling back is right once theirs has landed, and wasteful before.
    if where == "theirs landed":
        return {"roll back": 10.0, "patch it": 1.0, "wait": 0.0}[act]
    return {"roll back": 2.0, "patch it": 8.0, "wait": 1.0}[act]


def test_she_answers_what_is_coming_when_she_moves_second() -> None:
    tends = _tends()
    will, likely = tends.likely_next("the other team", facing="a red build")
    assert will == "force merge" and likely > 0.6
    ranked = against_what_is_coming(
        ACTS, now="a red build", they_will=will, how_likely=likely,
        after_theirs=_after, how_good=_how_good, she_moves_first=False,
    )
    assert ranked[0][0] == "roll back", ranked


def test_moving_first_she_answers_what_is_there() -> None:
    """The same act, the same prediction, and a different answer — because
    the order decides which world her act meets."""
    tends = _tends()
    will, likely = tends.likely_next("the other team", facing="a red build")
    ranked = against_what_is_coming(
        ACTS, now="a red build", they_will=will, how_likely=likely,
        after_theirs=_after, how_good=_how_good, she_moves_first=True,
    )
    assert ranked[0][0] == "patch it", ranked


def test_a_party_she_cannot_predict_costs_her_nothing_she_had() -> None:
    """The weight falls back to the situation as it stands, which is exactly
    what she did before any of this existed."""
    ranked = against_what_is_coming(
        ACTS, now="a red build", they_will="", how_likely=0.0,
        after_theirs=_after, how_good=_how_good, she_moves_first=False,
    )
    assert ranked[0][0] == "patch it"


def test_one_sighting_is_a_hint_and_not_a_law() -> None:
    once = WhatTheyTendToDo()
    once.they_did("somebody", facing="a thing", act="the one thing")
    _act, sure = once.likely_next("somebody", facing="a thing")
    assert sure < 1.0


def test_somebody_never_seen_here_is_unknown_and_not_borrowed_from_elsewhere() -> None:
    tends = _tends()
    assert tends.likely_next("the other team", facing="something else") == ("", 0.0)


def test_going_first_is_priced_rather_than_felt() -> None:
    """Where going second means not going at all, the gap is the whole act."""
    worth = worth_going_first(
        ACTS, now="a red build", how_good=_how_good,
        if_it_lands_second=lambda where, act: 0.0,
    )
    assert worth == 8.0
