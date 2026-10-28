"""A whole swap can cancel an effect that an intermediate subset exposes."""
from itertools import combinations

import pytest

from core.subject.coalitions import plan_coalitions


def test_every_boolean_effect_hidden_between_the_two_four_bit_anchors_is_exposed():
    width = 4
    units = [(str(i), None) for i in range(width)]
    plan = plan_coalitions(units, budget=10, seed=97)
    traditional = {0, 15, *(1 << i for i in range(width)), *(15 ^ (1 << i) for i in range(width))}
    missing = sorted(set(range(16)) - traditional)
    assert len(missing) == 6
    observed = traditional | {sum(1 << i for i in indices) for indices in plan.subsets}
    # All 63 nonzero truth tables supported only on the six missed states.
    # Every one looks inert to singles and whole swaps at both anchors.
    for effect_mask in range(1, 1 << len(missing)):
        effects = {state for bit, state in enumerate(missing) if effect_mask & (1 << bit)}
        assert effects.isdisjoint(traditional)
        assert effects & observed
    assert observed == set(range(16))
    assert plan.description()["complete_on_declared_units"]


def test_sampled_subsets_cover_a_masked_four_unit_cause_among_irrelevant_units():
    units = [(str(i), None) for i in range(96)]
    plan = plan_coalitions(units, budget=128, seed=97)
    # a*b*(1-c)*(1-d), with every other unit causally irrelevant.
    assert any(0 in indices and 1 in indices and 2 not in indices and 3 not in indices
               for indices in plan.subsets)
    assert len(set(plan.subsets)) == 128
    assert plan == plan_coalitions(units, budget=128, seed=97)
    assert not plan.description()["complete_on_declared_units"]


def test_near_complete_sampling_does_not_retry_an_exponential_search():
    units = [(str(i), None) for i in range(10)]
    plan = plan_coalitions(units, budget=1011, seed=97)
    assert len(plan.subsets) == 1011 and plan.possible == 1012
    assert all(2 <= len(indices) < 10 for indices in plan.subsets)


@pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 5])
def test_complete_plans_cover_each_proper_nonsingle_subset_once(width):
    plan = plan_coalitions([(str(i), None) for i in range(width)], budget=1024, seed=97)
    expected = {indices for size in range(2, width) for indices in combinations(range(width), size)}
    assert set(plan.subsets) == expected
    assert len(plan.subsets) == len(expected)


@pytest.mark.parametrize("budget", [-1, 1025])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        plan_coalitions([], budget=budget, seed=97)


def test_duplicate_units_cannot_change_the_sampling_distribution():
    with pytest.raises(ValueError, match="distinct"):
        plan_coalitions([("a", None), ("a", None)], budget=2, seed=97)
