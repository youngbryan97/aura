"""A choice follows what she expects to happen in this world, weighed by what it is worth to her.

Q(a) = sum over o of P(o | W, a) U(o | A, S) - C(a). Her arbiter valued an act by
what acts of its kind had been worth, the same in any world; the world reached
deliberation only as a gain on the final score. Now her worth for an act is
conditioned on the world by the likelihood ratio of this world among those in
which that kind of act went well (core/agency/consequence.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.consequence import ConsequenceModel, act_kind

PRESENT = np.array([1.0, 0.0, 0.2])
ABSENT = np.array([-1.0, 0.0, 0.2])


def _learned(seed: int = 0, n: int = 200) -> ConsequenceModel:
    """Greeting goes well when someone is there; studying goes well when no one is."""
    rng = np.random.default_rng(seed)
    model = ConsequenceModel()
    for _ in range(n):
        world = np.array([rng.choice([-1.0, 1.0]), rng.normal(), 0.2 + rng.normal(0, 0.1)])
        there = world[0] > 0
        model.learn("act:greet", world, good=bool(rng.random() < (0.8 if there else 0.2)))
        model.learn("act:study", world, good=bool(rng.random() < (0.2 if there else 0.8)))
    return model


def test_with_no_record_the_world_changes_nothing():
    model = ConsequenceModel()
    assert model.chance("act:greet", PRESENT) == (0.5, 0.5)
    assert model.worth_here(0.7, "act:greet", PRESENT) == 0.7


def test_a_kind_of_act_is_worth_more_in_the_world_where_it_has_gone_well():
    model = _learned()
    assert model.worth_here(0.5, "act:greet", PRESENT) > 0.5 > model.worth_here(0.5, "act:greet", ABSENT)
    assert model.worth_here(0.5, "act:study", ABSENT) > 0.5 > model.worth_here(0.5, "act:study", PRESENT)


def _choice(model: ConsequenceModel, world: np.ndarray, greet_worth: float, *, score_only: bool = False) -> float:
    """1 when she would greet, 0 when she would study."""
    study_worth = 0.6
    if score_only:
        # The world as one gain on every option's score: the order cannot change.
        gain = 1.0 + 0.5 * float(world[0])
        return float(greet_worth * gain > study_worth * gain)
    return float(model.worth_here(greet_worth, "act:greet", world) > model.worth_here(study_worth, "act:study", world))


def test_crossing_the_world_and_her_valuation_moves_the_choice_jointly():
    model = _learned()
    y = {(w, a): _choice(model, world, worth) for w, world in ((1, PRESENT), (0, ABSENT)) for a, worth in ((1, 0.55), (0, 0.15))}
    interaction = y[(1, 1)] - y[(1, 0)] - y[(0, 1)] + y[(0, 0)]
    assert interaction != 0.0
    gated = {(w, a): _choice(model, world, worth, score_only=True) for w, world in ((1, PRESENT), (0, ABSENT)) for a, worth in ((1, 0.55), (0, 0.15))}
    assert gated[(1, 1)] - gated[(1, 0)] - gated[(0, 1)] + gated[(0, 0)] == 0.0


def test_a_feature_that_never_mattered_does_not_turn_the_choice():
    model = _learned(seed=3, n=400)
    base = _choice(model, PRESENT, 0.55)
    nudged = PRESENT + np.array([0.0, 1.0, 0.0])
    assert _choice(model, nudged, 0.55) == base


def test_an_outcome_closes_the_choice_it_belongs_to_and_no_other():
    model = ConsequenceModel()
    model.opened("d1", "act:greet", PRESENT)
    assert model.resolved("d1", 0.4) is True
    assert model.resolved("d1", 0.4) is False
    assert model.resolved("never-opened", 0.4) is False
    assert model.chance("act:greet", PRESENT)[1] == pytest.approx(2 / 3)


def test_an_initiative_is_keyed_as_her_worth_is_keyed():
    assert act_kind("Greet Bryan when he returns") == "act:greet"
    assert act_kind("") == "act:"
