"""Several things going on, and one move to spend on them.

Watching two very strong Go players: there are never fewer than three separate
fights on the board and every move answers which one to spend it on. What
decides it is not which is biggest — it is what it costs to leave each alone.
"""

from __future__ import annotations

from core.cognition.where_to_spend_the_next_one import (
    what_it_costs_to_leave_them,
    where_to_spend_it,
)

#: A little contest: how much of it is hers, how much theirs, and whether the
#: other side is in a position to take it. Nothing about Go, and nothing about
#: games — two numbers and whether somebody else can move them.
def _mine(one: dict) -> int:
    return one["mine"]


def _how_good(one: dict) -> float:
    return float(one["mine"] - one["theirs"])


def _her_acts(one: dict) -> list[str]:
    return ["build"]


def _step(one: dict, act: str) -> dict | None:
    if act != "build":
        return None
    return {**one, "mine": one["mine"] + 1, "safe": True}


def _their_acts(one: dict) -> list[str]:
    return ["take"] if one.get("at_risk") and not one.get("safe") else []


def _their_step(one: dict, act: str) -> dict | None:
    if act != "take" or one.get("safe"):
        return None
    return {**one, "mine": 0, "theirs": one["theirs"] + one["mine"]}


SETTLED = {"mine": 20, "theirs": 0, "at_risk": False}
IN_DANGER = {"mine": 3, "theirs": 2, "at_risk": True}


def test_she_walks_away_from_the_biggest_thing_she_has() -> None:
    """It is large and nobody is threatening it, so a move spent there buys
    almost nothing. Which is why strong players leave their biggest group."""
    spend = where_to_spend_it(
        {"the big settled one": SETTLED, "the small one in danger": IN_DANGER},
        her_acts=_her_acts,
        step=_step,
        how_good=_how_good,
        their_acts=_their_acts,
        their_step=_their_step,
    )
    assert spend is not None
    assert spend.name == "the small one in danger", spend.describe()


def test_size_is_not_the_measure_and_the_gap_is() -> None:
    weighed = what_it_costs_to_leave_them(
        {"the big settled one": SETTLED, "the small one in danger": IN_DANGER},
        her_acts=_her_acts,
        step=_step,
        how_good=_how_good,
        their_acts=_their_acts,
        their_step=_their_step,
    )
    by_name = {one.name: one for one in weighed}
    # The big one is worth twenty, and playing in it gains one, because
    # nothing was going to happen to it either way.
    assert by_name["the big settled one"].worth == 1.0
    # The small one is worth one after she plays there, and minus five if she
    # does not — they take the three and it counts on their side. Seven, in a
    # situation a seventh the size of the other.
    assert by_name["the small one in danger"].if_she_acts == 2.0
    assert by_name["the small one in danger"].if_she_does_not == -5.0
    assert by_name["the small one in danger"].worth == 7.0
    assert _mine(SETTLED) > _mine(IN_DANGER), "and it really is the smaller one"


def test_acting_where_they_cannot_answer_keeps_the_move() -> None:
    weighed = what_it_costs_to_leave_them(
        {"the small one in danger": IN_DANGER},
        her_acts=_her_acts,
        step=_step,
        how_good=_how_good,
        their_acts=_their_acts,
        their_step=_their_step,
    )
    assert weighed[0].keeps_the_move, weighed[0].describe()
    assert weighed[0].ways_they_could_answer == 0


def test_nothing_worth_playing_in_is_an_answer() -> None:
    """Where acting changes nothing anywhere, the move is better spent on
    something this does not know about, and saying so is more use than
    picking the least pointless of them."""
    nothing_doing = {"mine": 5, "theirs": 0, "at_risk": False}

    def cannot_help(one: dict, act: str) -> dict | None:
        return dict(one)

    assert (
        where_to_spend_it(
            {"a": nothing_doing, "b": nothing_doing},
            her_acts=_her_acts,
            step=cannot_help,
            how_good=_how_good,
        )
        is None
    )


def test_a_world_that_does_not_push_back_is_the_same_subtraction() -> None:
    """Two jobs and an afternoon. One is nearly done and nothing threatens it;
    the other falls over if it is left."""
    jobs = {
        "the nearly finished one": {"mine": 9, "theirs": 0, "at_risk": False},
        "the one that is slipping": {"mine": 4, "theirs": 0, "at_risk": True},
    }
    weighed = what_it_costs_to_leave_them(
        jobs, her_acts=_her_acts, step=_step, how_good=_how_good
    )
    # With nobody on the other side, leaving a thing leaves it as it is, so
    # the gap is simply what she could improve it by, and they tie. That is
    # correct: without something that gets worse, urgency is not a fact.
    assert {round(one.worth, 3) for one in weighed} == {1.0}

    weighed = what_it_costs_to_leave_them(
        jobs,
        her_acts=_her_acts,
        step=_step,
        how_good=_how_good,
        their_acts=_their_acts,
        their_step=_their_step,
    )
    assert weighed[0].name == "the one that is slipping", [
        one.describe() for one in weighed
    ]
