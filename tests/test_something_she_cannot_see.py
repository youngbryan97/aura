"""A world that is not a function of what she can see.

No rule reading only the observation can fit a record where the same thing,
done twice from the same place, came out two ways. Both existing diagnoses
assume a best hypothesis exists to read the leftovers of, so both answer the
wrong question here.
"""

from __future__ import annotations

import random

from core.cognition.something_she_cannot_see import (
    a_coordinate_she_can_compute,
    what_she_cannot_see,
)
from core.cognition.what_the_failures_have_in_common import (
    SOMETHING_SHE_CANNOT_SEE,
    why_nothing_fits,
)

PLAIN = [((1, 2), (2, 1)), ((3, 4), (4, 3)), ((5, 6), (6, 5))]


def _alternating(times: int = 6):
    return [
        ((1, 2), (2, 1) if step % 2 == 0 else (1, 2)) for step in range(times)
    ]


def test_a_world_that_is_a_function_hides_nothing():
    found = what_she_cannot_see(PLAIN)
    assert not found.anything
    assert found.how_many == 1


def test_one_contradiction_proves_a_hidden_quantity():
    found = what_she_cannot_see(_alternating())
    assert found.anything
    assert found.how_many == 2


def test_the_count_is_the_most_ways_any_one_situation_came_out():
    thrice = [((1, 2), one) for one in [(2, 1), (1, 2), (2, 2), (2, 1), (1, 2)]]
    assert what_she_cannot_see(thrice).how_many == 3


def test_a_hidden_quantity_that_runs_in_a_cycle_becomes_one_she_computes():
    found = what_she_cannot_see(_alternating())
    assert found.she_can_compute_it
    assert found.every == 2
    reads = a_coordinate_she_can_compute(found)
    assert [reads(step) for step in range(4)] == [0, 1, 0, 1]


def test_a_cycle_is_not_claimed_on_less_than_two_turns_of_it():
    """One turn of a cycle is a claim every sequence satisfies."""
    assert what_she_cannot_see(_alternating(times=3)).every == 0


def test_a_random_world_is_named_as_one_rather_than_as_a_wrong_rule():
    """The failure this exists to prevent: every wrong prediction after a
    random event looks exactly like a wrong rule."""
    roll = random.Random(11)
    spawned = [
        (((2, 2, 0, 0), "left"), (4, 0, 0, roll.choice([2, 4])))
        for _ in range(12)
    ]
    found = what_she_cannot_see(spawned)
    assert found.anything
    assert not found.she_can_compute_it
    assert a_coordinate_she_can_compute(found) is None


def test_what_she_did_is_part_of_the_situation():
    """Two keys from one board are not one situation coming out two ways."""
    both = [(((2, 2), "left"), (4, 0)), (((2, 2), "right"), (0, 4))]
    assert not what_she_cannot_see(both).anything


def test_the_diagnosis_names_it_before_blaming_the_search_or_the_language():
    why = why_nothing_fits(_alternating())
    assert why.because == SOMETHING_SHE_CANNOT_SEE
    assert why.is_something_unseen
    assert "not reading" in why.describes()


def test_a_world_that_fits_is_still_reported_as_fitting():
    assert why_nothing_fits([((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (6, 5, 4))]).because != (
        SOMETHING_SHE_CANNOT_SEE
    )


def test_an_empty_record_proves_nothing():
    assert not what_she_cannot_see([]).anything


def _board(rows):
    from core.perception.what_is_there import Arrangement, Cell

    return Arrangement(
        rows=len(rows),
        columns=len(rows[0]),
        cells=tuple(
            Cell(row=r, column=c, says=str(v), at=(c * 10.0, r * 10.0))
            for r, row in enumerate(rows)
            for c, v in enumerate(row)
            if v
        ),
    )


def test_a_merge_and_a_deal_leave_the_count_level_and_the_record_does_not():
    """Its own comment named the case it could not see: two things become one
    and a third is dealt, so occupancy comes back level and nothing looks to
    have arrived."""
    from core.perception.how_it_moves import HowItMoves

    watching = HowItMoves()
    before = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    for dealt in (2, 4, 2, 4):
        watching.watched(
            before,
            "left",
            _board([[4, 0, 0, 0], [0, 0, 0, dealt], [0, 0, 0, 0], [0, 0, 0, 0]]),
        )
    assert watching.arrivals == 0
    record = watching.what_she_saw_happen()
    assert what_she_cannot_see(
        [((seen, did), then) for seen, did, then in record]
    ).anything


def test_a_world_that_adds_nothing_leaves_the_record_a_function():
    from core.perception.how_it_moves import HowItMoves

    watching = HowItMoves()
    still = _board([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    for _ in range(4):
        watching.watched(still, "left", still)
    record = watching.what_she_saw_happen()
    assert not what_she_cannot_see(
        [((seen, did), then) for seen, did, then in record]
    ).anything


def test_the_play_loop_says_which_of_the_two_worlds_it_is_in():
    """"Not worked out yet" is true of a world with a rule she has not found
    and of one where no rule can fit, and those want opposite responses."""
    from core.perception.how_it_moves import HowItMoves
    from core.skills.screen_pursuit import _what_she_is_not_reading

    watching = HowItMoves()
    before = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    import random

    roll = random.Random(3)
    for _ in range(14):
        watching.watched(
            before,
            "left",
            _board(
                [
                    [4, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, roll.choice([0, 2]), roll.choice([0, 4])],
                ]
            ),
        )
    said = _what_she_is_not_reading(watching)
    assert "not reading" in said
    assert "does not control" in said

    settled = HowItMoves()
    still = _board([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    for _ in range(6):
        settled.watched(still, "left", still)
    assert _what_she_is_not_reading(settled) == ""
