"""A place as evidence of capacity, measured.

"Empire State of Mind" reads making it somewhere hard as evidence of making it
anywhere. These pin the measured version: success is weighted towards what has
beaten her most, that capacity is the prior for what she has not tried, and it
is Laplace smoothing exactly when nothing has been tried.
"""

from __future__ import annotations

import pytest

from core.agency.capacity import Capacity, capacity_of, confidence_with_capacity


def test_nothing_tried_is_the_middle_and_laplace_exactly() -> None:
    capacity = capacity_of({})
    assert not capacity.measured
    assert confidence_with_capacity(0, 0, capacity) == pytest.approx(0.5)
    assert confidence_with_capacity(3, 2, capacity) == pytest.approx(3 / 5)


def test_success_is_weighted_towards_what_has_beaten_her() -> None:
    capacity = capacity_of({"proofs": [20, 10], "greetings": [20, 19]})
    hard, easy = 1 - 11 / 22, 1 - 20 / 22
    expected = (10 * hard + 19 * easy + 1) / (20 * hard + 20 * easy + 2)
    assert capacity.capacity == pytest.approx(expected)
    assert capacity.capacity < 29 / 40
    assert capacity.hardest == "proofs"


def test_a_capability_is_not_its_own_evidence() -> None:
    capacity = capacity_of({"proofs": [1, 1]}, excluding="proofs")
    assert not capacity.measured
    assert confidence_with_capacity(1, 1, capacity) == pytest.approx(2 / 3)


def test_one_success_elsewhere_is_not_certainty() -> None:
    capacity = capacity_of({"proofs": [1, 1]})
    assert 0.5 < capacity.capacity < 1.0


def test_capacity_is_the_prior_for_what_she_has_not_tried() -> None:
    strong = Capacity(capacity=0.8, measured=True)
    weak = Capacity(capacity=0.3, measured=True)
    assert confidence_with_capacity(0, 0, strong) == pytest.approx(0.8)
    assert confidence_with_capacity(0, 0, weak) == pytest.approx(0.3)
    assert confidence_with_capacity(8, 8, weak) > confidence_with_capacity(0, 0, weak)


def test_every_failure_stays_in_the_account() -> None:
    capacity = capacity_of({"proofs": [20, 0]})
    assert capacity.capacity < 0.05
    assert "worked 0.05" in capacity.why


def test_unreadable_rows_are_left_out() -> None:
    capacity = capacity_of({"a": [0, 0], "b": [2, 5], "c": ["x", 1], "d": [4, 2]})
    assert capacity.capabilities == 1


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = capacity_of({"a": [3, 2]}).as_dict()
    for key in ("capacity", "hardest", "hardest_rate", "capabilities", "measured", "why"):
        assert key in row, key
