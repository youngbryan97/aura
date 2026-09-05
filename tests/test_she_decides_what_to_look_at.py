"""Perception that chooses, rather than perception that receives.

Aura's perception was receptive: a camera sends an image and something
describes it, and the observations she got were the ones that happened to
arrive. `expected_information_gain.py` computed what an observation would be
worth and nobody handed it a question. There was information-gain reasoning
elsewhere, experiment proposal and selection, active sensing loops and cost
machinery relating gain to spend — and no single controller that runs

    uncertainty -> candidate observations -> P(o|h) -> EIG - C
        -> take the best -> update the belief

over the things she can actually do.

Two properties are what make this a policy rather than a preference for
looking at things. An observation that cannot discriminate is worth nothing
however interesting it is. And gain is not worth: an observation that would
settle the question and costs more than the question is worth should not be
made.
"""

from __future__ import annotations

import random

import pytest

from core.perception.expected_information_gain import Recommendation, entropy
from core.perception.how_she_finds_out import (
    WayOfFindingOut,
    clear_the_inventory,
    find_out,
    how_it_went,
    register_a_way,
    snapshot,
    the_inventory,
    what_to_look_at,
)


@pytest.fixture(autouse=True)
def an_empty_inventory():
    clear_the_inventory()
    yield
    clear_the_inventory()


def _a_way(name: str, cost: float, sees: str = "a", **counts) -> WayOfFindingOut:
    return register_a_way(
        WayOfFindingOut(
            name=name, about=("x",), cost=cost,
            outcomes=("a", "b"), take=lambda _s, out=sees: out, **counts,
        )
    )


def _fixed(value: float):
    """A reliability draw that does not draw, so a test is a measurement."""

    return lambda _a, _b: value


UNSURE = {"a": 0.5, "b": 0.5}


def test_a_way_that_cannot_discriminate_is_worth_nothing():
    """One outcome is not a way of finding out. It is refused at the door."""

    with pytest.raises(ValueError, match="discriminates nothing"):
        register_a_way(
            WayOfFindingOut(
                name="always says yes", about=("x",), cost=0.0,
                outcomes=("yes",), take=lambda _s: "yes",
            )
        )


def test_an_observation_at_chance_is_uninformative():
    _a_way("a coin", cost=0.0)
    ranked = what_to_look_at(UNSURE, about="x", draw=_fixed(0.5))
    assert ranked[0].recommendation is Recommendation.UNINFORMATIVE
    assert ranked[0].expected_bits < 1e-6


def test_gain_is_not_worth():
    """It would settle the question and costs more than the question is worth."""

    _a_way("ask him", cost=2.0, right=40)
    ranked = what_to_look_at(UNSURE, about="x", draw=_fixed(0.99))
    assert ranked[0].expected_bits > 0.9
    assert ranked[0].recommendation is Recommendation.TOO_EXPENSIVE
    assert find_out("x", UNSURE, draw=_fixed(0.99)).looked is False


def test_the_same_observation_is_worth_taking_for_a_dearer_question():
    """The subtraction is the decision. Nothing about it is a property of
    the observation on its own."""

    _a_way("ask him", cost=2.0, right=40)
    assert find_out("x", UNSURE, value_per_bit=1.0, draw=_fixed(0.99)).looked is False
    assert find_out("x", UNSURE, value_per_bit=10.0, draw=_fixed(0.99)).looked is True


def test_settled_questions_are_not_investigated():
    _a_way("look", cost=0.0, right=40)
    settled = {"a": 0.9999, "b": 0.0001}
    ranked = what_to_look_at(settled, about="x", draw=_fixed(0.99))
    assert ranked[0].recommendation is Recommendation.SETTLED


def test_taking_the_observation_moves_the_belief():
    _a_way("look", cost=0.01, sees="a", right=40)
    found = find_out("x", UNSURE, draw=_fixed(0.95))
    assert found.looked
    assert found.saw == "a"
    assert found.after["a"] > found.before["a"]
    assert found.bits_gained > 0.0
    assert entropy(found.after) < entropy(found.before)


def test_the_cheaper_of_two_equally_good_ways_wins():
    _a_way("cheap", cost=0.01, right=40)
    _a_way("dear", cost=0.50, right=40)
    ranked = what_to_look_at(UNSURE, about="x", draw=_fixed(0.95))
    assert ranked[0].observation == "cheap"


def test_an_unmeasured_way_is_explored_and_a_useless_one_is_not():
    """The cold start, solved by the shape of the uncertainty rather than by
    a number nudging it upward.

    A way with reliability fixed at one half discriminates nothing, has zero
    expected gain, is never taken and never learns. Scoring on a reliability
    drawn from the posterior means an untried way is sometimes optimistic and
    gets its trial, while a way measured at chance forty times almost never
    is — and neither behaviour is written down anywhere.
    """

    _a_way("never tried", cost=0.05)
    untried = sum(
        1
        for trial in range(200)
        if (
            r := what_to_look_at(
                UNSURE, about="x", draw=random.Random(trial).betavariate
            )
        )
        and r[0].take
    )
    clear_the_inventory()
    _a_way("measured at chance", cost=0.05, right=20, wrong=20)
    useless = sum(
        1
        for trial in range(200)
        if (
            r := what_to_look_at(
                UNSURE, about="x", draw=random.Random(trial).betavariate
            )
        )
        and r[0].take
    )
    assert untried > 100, "an untried way is never explored, so it never learns"
    assert useless < 30, "a way known to be useless is still being taken"
    assert untried > useless * 3


def test_the_ranking_does_not_change_per_process():
    """One sample never settles whether a failure is yours."""

    _a_way("one", cost=0.05)
    _a_way("two", cost=0.05)
    first = [one.observation for one in what_to_look_at(UNSURE, about="x")]
    for _ in range(5):
        assert [one.observation for one in what_to_look_at(UNSURE, about="x")] == first


def test_being_unable_to_look_is_not_being_wrong():
    """A sensor that is not running is not an inaccurate sensor.

    Counting an availability failure against accuracy would make the number
    the controller ranks on a measure of whether the subsystem was up.
    """

    def cannot(_subject):
        raise RuntimeError("the daemon is not running")

    way = register_a_way(
        WayOfFindingOut(
            name="a sensor that is off", about=("x",), cost=0.0,
            outcomes=("a", "b"), take=cannot, right=10,
        )
    )
    before = (way.right, way.wrong)
    found = find_out("x", UNSURE, draw=_fixed(0.95))
    assert not found.looked
    assert (way.right, way.wrong) == before
    assert way.unavailable == 1


def test_a_way_that_cannot_say_what_it_saw_saw_nothing():
    register_a_way(
        WayOfFindingOut(
            name="mumbles", about=("x",), cost=0.0,
            outcomes=("a", "b"), take=lambda _s: "something else", right=10,
        )
    )
    found = find_out("x", UNSURE, draw=_fixed(0.95))
    assert found.way == "mumbles"
    assert found.after == found.before


def test_no_inventory_is_a_stated_answer_not_a_silence():
    found = find_out("x", UNSURE)
    assert not found.looked
    assert "nothing registered" in found.because


def test_reliability_comes_from_use_and_only_from_use():
    way = _a_way("look", cost=0.0)
    assert way.used == 0
    assert way.reliability == pytest.approx(0.5)
    for _ in range(8):
        how_it_went("look", right=True)
    how_it_went("look", right=False)
    assert way.used == 9
    assert way.reliability > 0.7
    assert how_it_went("a way nobody registered", right=True) is None


def test_the_real_inventory_is_declared():
    """It must not be another module nothing reaches."""

    from core.perception.the_ways_she_has import declare_the_ways_she_has

    declared = declare_the_ways_she_has()
    assert "look at the screen" in declared
    assert {one.name for one in the_inventory()} >= set(declared)
    assert snapshot()["ways"] >= len(declared)


def test_redeclaring_keeps_what_a_way_learned():
    """A boot that forgot the track record makes every way untried again."""

    from core.perception.the_ways_she_has import declare_the_ways_she_has

    declare_the_ways_she_has()
    for _ in range(5):
        how_it_went("look at the screen", right=True)
    declare_the_ways_she_has()
    again = {one.name: one for one in the_inventory()}["look at the screen"]
    assert again.right == 5


def test_the_belief_state_can_go_and_find_out():
    """The organism-wide seam: any subsystem holding hypotheses can ask."""

    from core.perception.belief_state import EnvironmentBeliefState

    _a_way("look", cost=0.01, sees="a", right=40)
    beliefs = EnvironmentBeliefState(session_id="a test")
    beliefs.ensure_hypotheses("x", ["a", "b"])
    before = beliefs.epistemic_uncertainty("x")
    found = beliefs.find_out_about("x")
    assert found.looked
    assert beliefs.epistemic_uncertainty("x") < before
    assert beliefs.hypotheses["x"]["a"].probability > 0.5
    assert any("look" in one for one in beliefs.hypotheses["x"]["a"].evidence)


def test_one_hypothesis_is_nothing_to_tell_apart():
    from core.perception.belief_state import EnvironmentBeliefState

    beliefs = EnvironmentBeliefState(session_id="a test")
    beliefs.ensure_hypotheses("x", ["only one"])
    found = beliefs.find_out_about("x")
    assert not found.looked
    assert "nothing to tell apart" in found.because
