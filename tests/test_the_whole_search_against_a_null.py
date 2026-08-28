"""What the mechanism finds when the answers are shuffled.

A p-value for the winner ignores that the winner was chosen from thousands. The
null this needs is not "does this one answer survive a shuffle" but "would a
search this powerful have found something in noise", so the whole live path is
run — positional forms, composition, the learned library, the refutation, the
ordering, and the ordering composed with a move — and the worst round is taken.

Measured over twenty shuffled batteries, 2,600 problems: the mechanism produced
an answer for 7 of them, at most 2 in any one round, and answered none of them
correctly. On the real battery it produces an answer for 107 of 130.

Three rounds here, because this is a gate and not a study; the twenty-round
figure is recorded in the floor beside the score.
"""

from __future__ import annotations

import random

import pytest

from core.cognition.induction_battery import Problem, _solve, generate_battery
from core.cognition.language_limits import certify
from core.cognition.primitive_invention import (
    Transition,
    _index_forms,
    invent_relation,
)
from core.cognition.relation_language import RelationLanguage
from core.cognition.value_order import solve_ordering, solve_ordering_then_move

_ROUNDS = 3


@pytest.fixture(scope="module")
def battery():
    return generate_battery()


def _shuffled(problem, rng: random.Random):
    """The same states, with every answer independently permuted."""

    shown = []
    for item in problem.shown:
        after = list(item.after)
        rng.shuffle(after)
        shown.append(Transition(item.before, tuple(after)))
    held = list(problem.held_out.after)
    rng.shuffle(held)
    return shown, Transition(problem.held_out.before, tuple(held))


def _anything_found(shown, asked) -> str | None:
    """Every route the live path can take, so the null covers the whole search."""

    if invent_relation(shown) is not None:
        return "positional"
    if certify(shown).proven_outside:
        ordering = solve_ordering(shown)
        if ordering is not None and ordering.apply(asked.before) is not None:
            return "ordering"
        composed = solve_ordering_then_move(shown, _index_forms(len(asked.before)))
        if composed is not None and composed.apply(asked.before) is not None:
            return "composed"
    return None


def test_the_mechanism_finds_almost_nothing_in_noise(battery) -> None:
    rng = random.Random(20260828)
    worst = 0
    for _round in range(_ROUNDS):
        hits = sum(
            1 for problem in battery if _anything_found(*_shuffled(problem, rng))
        )
        worst = max(worst, hits)
    # It found 2 at worst over twenty rounds. Five is generous.
    assert worst <= 6, f"{worst}/{len(battery)} answers in shuffled data"


def test_it_almost_never_gets_a_shuffled_world_RIGHT(battery) -> None:
    """The number that matters. Producing an answer is not the failure.

    Not zero, and it should not be. A shuffle of four cells lands on the
    identity one time in twenty-four, and a question whose answer accidentally
    became "nothing moved" has a real answer that a mechanism is right to find.
    Asserting zero here would be asserting that chance does not happen, and the
    first seed that disagreed would look like a defect.

    Measured: 1 in 650 on one seed, 0 in 2,600 on another. The bound is set an
    order of magnitude above that, so a real regression moves it and a run of
    luck does not.
    """

    rng = random.Random(19951121)
    right = 0
    for _round in range(_ROUNDS):
        for problem in battery:
            shown, held = _shuffled(problem, rng)
            fake = Problem(
                name=problem.name,
                shown=tuple(shown),
                held_out=held,
                shape=problem.shape,
                representation=problem.representation,
            )
            right += int(_solve(fake, RelationLanguage()))
    seen = _ROUNDS * len(battery)
    assert right / seen < 0.02, f"{right}/{seen} shuffled worlds answered correctly"


def test_the_real_battery_is_not_similarly_empty(battery) -> None:
    """A null means nothing beside a mechanism that finds nothing anywhere."""

    found = sum(
        1
        for problem in battery
        if _anything_found(list(problem.shown), problem.held_out)
    )
    assert found >= 100, f"only {found}/{len(battery)} on the real battery"
