"""She chooses what to do about herself, and the grounds are checkable.

The claim these hold is narrow and it is the one that matters: the thing that
decides whether to change her language reads a record, and its answer moves
when the record moves. A ladder cannot pass that, which is the point of
checking it rather than describing it.
"""

from __future__ import annotations

import pytest

from core.cognition.the_record_of_her_own_work import (
    HOW_MANY_EPISODES_ARE_KEPT,
    attribution,
    episodes,
    forget_the_record,
    how_long_since,
    how_often,
    note_a_use,
    note_an_episode,
    the_record,
    what_it_has_cost,
)
from core.cognition.she_decides_to_develop import (
    forget_the_trace,
    she_decides_to_develop,
    the_trace,
    what_it_may_spend,
    what_to_do_next,
    who_started_it,
    why_each_one,
)
from core.cognition.what_it_is_worth_doing import (
    THE_WORTH,
    forget_the_worth,
    how_much_it_is_worth,
    how_often_it_will_come_up,
    the_choice_follows_the_record,
    the_price_of_finding_out,
    the_worth_she_wrote,
    what_each_occasion_would_save,
    what_it_risks,
    what_the_record_says_is_slow,
    where_a_split_disagrees_with_the_whole,
)
from core.cognition.what_she_could_do_next import (
    WHAT_SHE_COULD_DO,
    WHERE_A_TERM_CAN_GO,
    forget_the_action,
    the_action_she_wrote,
    the_actions_she_has,
    what_she_could_do,
)


@pytest.fixture(autouse=True)
def a_clean_slate():
    held = dict(WHAT_SHE_COULD_DO)
    forget_the_record()
    forget_the_trace()
    forget_the_worth()
    WHAT_SHE_COULD_DO.clear()
    yield
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)
    forget_the_record()
    forget_the_trace()
    forget_the_worth()


def a_cheap_one(name: str = "a new word", price: int = 400):
    return what_she_could_do(
        name, over="the words", kind="a word", do_it=lambda s: name, price=price
    )


def a_dear_one(name: str = "a way of computing", price: int = 1400):
    return what_she_could_do(
        name,
        over="the ways of computing",
        kind="a way of computing",
        do_it=lambda s: name,
        price=price,
    )


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def test_the_counts_outlive_the_episodes_they_came_from():
    for _ in range(HOW_MANY_EPISODES_ARE_KEPT + 40):
        note_an_episode("doubling", route="a head", walked=10, used=("here",))
    assert len(episodes()) == HOW_MANY_EPISODES_ARE_KEPT
    # The instance is gone and the statistic is not, which is the shape finite
    # memory forces.
    assert how_often("doubling") == HOW_MANY_EPISODES_ARE_KEPT + 40


def test_disuse_is_measurable_and_a_thing_never_used_says_so():
    note_a_use("here")
    note_an_episode("mirror", route="a word", walked=3)
    note_an_episode("mirror", route="a word", walked=3)
    assert how_long_since("here") == 2
    assert how_long_since("the far end") is None


def test_what_a_route_has_cost_is_read_not_guessed():
    note_an_episode("a", route="a head", walked=100)
    note_an_episode("a", route="a head", walked=300)
    assert what_it_has_cost("a head") == 200
    assert what_it_has_cost("a rule") is None


def test_the_record_says_which_part_of_her_is_slow():
    for _ in range(3):
        note_an_episode("hard", route=None, walked=9000)
    note_an_episode("easy", route="a word", walked=10)
    slow = what_the_record_says_is_slow()
    assert slow[0]["route"] == "nothing answered"
    assert slow[0]["per answer"] > slow[-1]["per answer"]


# --------------------------------------------------------------------------
# The value
# --------------------------------------------------------------------------


def test_the_worth_is_occasions_times_saving_less_cost_and_risk():
    assert how_much_it_is_worth(occasions=10, saving=50, cost=100, risk=20) == 380


def test_the_worth_is_a_term_she_can_replace_and_lesion():
    from core.cognition.the_floor_she_stands_on import L, N, build

    always_nothing = build(L("a", L("b", L("c", L("d", N(0))))))
    the_worth_she_wrote(always_nothing)
    assert how_much_it_is_worth(occasions=10, saving=50, cost=1, risk=1) == 0
    forget_the_worth()
    assert how_much_it_is_worth(occasions=10, saving=50, cost=1, risk=1) == 498


def test_a_saving_with_no_history_is_refused_rather_than_defaulted():
    saving, why = what_each_occasion_would_save("a way of computing", costs_now=900)
    assert saving is None
    assert "admitted" in why


def test_a_saving_is_measured_from_what_that_kind_saved_before():
    for _ in range(3):
        note_an_episode("doubling", route=None, walked=1000)
    note_an_episode("doubling", route="a head", walked=1400, admitted="a way of computing")
    for _ in range(3):
        note_an_episode("doubling", route="a head", walked=100)
    saving, why = what_each_occasion_would_save("a way of computing", costs_now=1000)
    assert why == ""
    # Nine tenths of the cost went away, so nine tenths is what it is worth.
    assert saving == pytest.approx(900, abs=20)


def test_a_kind_that_never_paid_is_riskier_than_one_that_did():
    note_an_episode("a", route=None, walked=100)
    note_an_episode("a", route="x", walked=100, admitted="worked")
    note_an_episode("a", route="x", walked=10)
    note_an_episode("b", route=None, walked=100)
    note_an_episode("b", route="y", walked=100, admitted="failed")
    note_an_episode("b", route=None, walked=400)
    assert what_it_risks("failed", cost=1000, entries=0) > what_it_risks(
        "worked", cost=1000, entries=0
    )


def test_the_price_of_finding_out_is_the_price_of_doing_it():
    assert the_price_of_finding_out(700) == 700


# --------------------------------------------------------------------------
# The two conditions
# --------------------------------------------------------------------------


def test_a_constant_switch_disagrees_with_the_ranking_somewhere():
    """A ladder is a constant switch, and this is why that is not a decision."""
    worths = {"answer": 1, "write a head": 2}
    found = where_a_split_disagrees_with_the_whole(
        ordinary=["answer"],
        developmental=["write a head"],
        worth=worths.get,
        switch=lambda ordinary, developmental: "act",
    )
    assert found
    assert found[0]["the ranking chose"] == "write a head"
    assert found[0]["the split chose"] == "answer"


def test_a_switch_that_agrees_everywhere_has_computed_the_ranking():
    worths = {"answer": 1, "write a head": 2}
    found = where_a_split_disagrees_with_the_whole(
        ordinary=["answer"],
        developmental=["write a head"],
        worth=worths.get,
        switch=lambda ordinary, developmental: (
            "develop"
            if max(map(worths.get, developmental)) > max(map(worths.get, ordinary))
            else "act"
        ),
    )
    assert found == []


def test_her_choice_moves_when_the_record_moves():
    a_cheap_one()
    a_dear_one()

    def cheap_is_all_that_ever_helps() -> None:
        forget_the_record()
        for _ in range(3):
            note_an_episode("f", route=None, walked=1000)
        note_an_episode("f", route="a new word", walked=400, admitted="a word")
        for _ in range(3):
            note_an_episode("f", route="a new word", walked=20)

    def the_head_is_all_that_ever_helps() -> None:
        forget_the_record()
        for _ in range(3):
            note_an_episode("f", route=None, walked=1000)
        note_an_episode(
            "f", route="a way of computing", walked=1400, admitted="a way of computing"
        )
        for _ in range(3):
            note_an_episode("f", route="a way of computing", walked=20)

    def chose() -> str:
        decided = what_to_do_next("f", costs_now=1000)
        return decided.action.name if decided.action else "nothing"

    assert the_choice_follows_the_record(
        chose, [cheap_is_all_that_ever_helps, the_head_is_all_that_ever_helps]
    )


def test_a_ladder_fails_the_same_check():
    """The control for the test above: something that ignores the record."""
    a_cheap_one()
    a_dear_one()

    def first_rung_always() -> str:
        return the_actions_she_has()[0].name

    def one_way() -> None:
        forget_the_record()
        note_an_episode("f", route="a new word", walked=1, admitted="a word")

    def the_other() -> None:
        forget_the_record()
        for _ in range(50):
            note_an_episode("f", route="a way of computing", walked=1)

    assert not the_choice_follows_the_record(first_rung_always, [one_way, the_other])


# --------------------------------------------------------------------------
# The action space
# --------------------------------------------------------------------------


def test_an_action_can_only_go_where_a_term_can_go():
    with pytest.raises(ValueError):
        what_she_could_do("nowhere", over="my imagination", kind="k", do_it=lambda s: None)


def test_every_destination_is_something_that_holds_a_term():
    assert "the order she tries them in" in WHERE_A_TERM_CAN_GO
    assert "what a change is worth" in WHERE_A_TERM_CAN_GO


def test_an_action_she_wrote_needs_no_edit_to_be_admitted():
    from core.cognition.the_floor_she_stands_on import L, N, QUOTE, build
    from core.cognition.the_order_she_tries_them_in import (
        THE_ORDER,
        forget_the_order,
        the_order_she_uses,
    )

    a_different_order = build(
        L("agreed", L("places", L("won", L("of", L("symbols", N(7))))))
    )
    made = the_action_she_wrote(
        "try them in the order I wrote",
        over="the order she tries them in",
        look_for=build(L("situation", QUOTE(a_different_order))),
    )
    assert made.hers
    assert made.name in WHAT_SHE_COULD_DO
    try:
        made.do_it(None)
        assert the_order_she_uses() is not THE_ORDER
    finally:
        forget_the_order()
        forget_the_action(made.name)


def test_an_action_can_be_taken_out():
    a_cheap_one()
    assert forget_the_action("a new word") is not None
    assert the_actions_she_has() == ()


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


def test_she_explores_when_nothing_has_a_history():
    a_cheap_one()
    a_dear_one()
    decided = what_to_do_next("f", costs_now=1000)
    assert decided.because == "exploring"
    assert decided.action.name == "a new word"


def test_she_refuses_when_nothing_is_worth_doing():
    a_cheap_one(price=5000)
    a_dear_one(price=9000)
    decided = what_to_do_next("f", costs_now=10)
    assert decided.because == "refused"
    assert decided.action is None
    assert "nothing is worth doing" in decided.grounds


def test_she_chooses_the_one_the_record_says_pays():
    a_cheap_one()
    a_dear_one()
    for _ in range(4):
        note_an_episode("f", route=None, walked=5000)
    note_an_episode(
        "f", route="a way of computing", walked=1400, admitted="a way of computing"
    )
    for _ in range(4):
        note_an_episode("f", route="a way of computing", walked=30)
    decided = what_to_do_next("f", costs_now=5000)
    assert decided.because == "chosen"
    assert decided.action.name == "a way of computing"
    assert decided.worth.worth > 0
    assert len(why_each_one(decided)) == 2


def test_the_budget_comes_from_the_family_not_from_a_constant():
    note_an_episode("f", route=None, walked=100)
    note_an_episode("f", route=None, walked=100)
    assert what_it_may_spend("f", costs_now=100) == 200
    assert what_it_may_spend("never seen", costs_now=100) == 100


def test_a_named_action_is_recorded_as_asked_for_not_as_hers():
    a_cheap_one()
    decided, came = she_decides_to_develop(
        "f", costs_now=1000, asked_for="a new word"
    )
    assert decided.started_by == "asked"
    assert came == "a new word"
    assert who_started_it("trigger") == {"asked": 1}


def test_a_choice_she_made_is_recorded_as_hers():
    a_cheap_one()
    decided, came = she_decides_to_develop("f", costs_now=1000)
    assert decided.started_by == "she"
    assert who_started_it("trigger") == {"she": 1}
    assert [one.what for one in the_trace()] == [
        "trigger",
        "diagnosis",
        "proposal",
        "evaluation",
        "installation",
    ]


def test_an_episode_is_written_down_whatever_happened():
    what_she_could_do(
        "gives nothing", over="the words", kind="a word", do_it=lambda s: None, price=1
    )
    she_decides_to_develop("f", costs_now=1000)
    assert episodes()[-1].route is None
    assert episodes()[-1].admitted is None


def test_the_record_survives_a_restart():
    from core.cognition.the_record_of_her_own_work import (
        keep_the_record,
        recall_the_record,
    )

    note_an_episode("f", route="a head", walked=42, used=("here",), admitted="a way")
    if not keep_the_record():
        pytest.skip("this machine would not let the record be written")
    forget_the_record()
    assert recall_the_record() >= 1
    assert how_often("f") >= 1
    assert any(one.walked == 42 for one in episodes())
