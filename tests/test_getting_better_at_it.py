"""U8: does keeping what she works out make the next problem go better?

Not "can she learn something" — every piece answers that alone — but whether
learning ACCUMULATES: does a later problem go better because of the earlier
ones, in a domain the earlier ones were not about.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition import how_she_learns_to_look as look
from core.cognition.a_kind_of_thing_she_named import (
    KINDS_OF_THING,
    a_kind_of_thing_she_named,
    a_way_of_building_over,
)
from core.cognition.an_action_she_composed import (
    World,
    an_action_she_composed,
    what_it_does,
)
from core.cognition.getting_better_at_it import (
    Problem,
    a_run_of_problems,
    did_keeping_it_help,
)

#: One depth of search for both arms. The question is whether what she KEPT
#: lets her reach further at the same depth.
HOW_DEEP = 1

FOURS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]
FIVES = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3), (5, 1, 9, 3, 7)]
FOURS_UNSEEN = [(3, 1, 4, 2), (5, 9, 2, 6), (8, 7, 1, 3), (2, 6, 4, 9)]
FIVES_UNSEEN = [(3, 1, 4, 1, 5), (9, 2, 6, 5, 3), (8, 7, 1, 3, 2), (2, 6, 4, 9, 7)]


@pytest.fixture
def a_beginner():
    given = dict(kinds.WHERE_FROM)
    composed: dict[str, object] = {}
    was = (dict(kinds.WAYS_TO_BUILD), dict(KINDS_OF_THING), dict(kinds.KINDS))
    # Her own state root belongs to the live runtime; a test writes nowhere
    # near it.
    kept_at = look._kept_at
    nowhere = pathlib.Path(tempfile.mkdtemp()) / "worked.json"
    look._kept_at = lambda: nowhere

    def forget():
        kinds.WAYS_TO_BUILD.clear()
        KINDS_OF_THING.clear()
        kinds.KINDS.clear()
        for name in [one for one in list(kinds.WHERE_FROM) if one not in given]:
            kinds.WHERE_FROM.pop(name, None)
        composed.clear()
        look.forget_what_worked()

    def what_she_has():
        return [
            *kinds.WAYS_TO_BUILD,
            *KINDS_OF_THING,
            *composed,
            *[one for one in kinds.WHERE_FROM if one not in given],
        ]

    forget()
    try:
        yield given, composed, forget, what_she_has
    finally:
        look._kept_at = kept_at
        forget()
        for store, before in zip(
            (kinds.WAYS_TO_BUILD, KINDS_OF_THING, kinds.KINDS), was
        ):
            store.clear()
            store.update(before)


def _family(sizes, when_even, when_odd):
    return [
        (
            one,
            tuple(
                one[(when_even if len(one) % 2 == 0 else when_odd)(at, len(one)) % len(one)]
                for at in range(len(one))
            ),
        )
        for one in sizes
    ]


def _step(by):
    return lambda at: max(0, min(9, at + by))


A_LINE = {"one right": _step(1), "one left": _step(-1)}


def _problems(given):
    def sequences(name, one, other):
        return Problem(
            kind="sequences",
            name=name,
            shown=_family(FOURS + FIVES, given[one], given[other]),
            held_back=_family(FOURS_UNSEEN + FIVES_UNSEEN, given[one], given[other]),
        )

    return [
        sequences("mixed by parity", "the far end", "one along"),
        # Reachable at this depth from what she was given: repeat one key.
        Problem("worlds", "to the far wall", (A_LINE, [(3, 9), (7, 9)]), [(1, 9), (5, 9), (8, 9)]),
        sequences("mixed the other way", "one along", "the far end"),
        # NOT reachable at this depth from what she was given. It needs the
        # action above as a part: kept it is one step, forgotten it is two.
        Problem(
            "worlds",
            "to the far wall then back one",
            (A_LINE, [(3, 8), (0, 8)]),
            [(5, 8), (8, 8), (1, 8)],
        ),
        sequences("parity again", "its partner", "one back"),
        Problem("worlds", "to the near wall", (A_LINE, [(6, 0), (9, 0)]), [(2, 0), (8, 0), (4, 0)]),
        Problem(
            "worlds",
            "to the near wall then forward one",
            (A_LINE, [(6, 1), (9, 1)]),
            [(2, 1), (8, 1), (4, 1)],
        ),
    ]


def _solving(composed):
    def solve(problem: Problem) -> bool:
        if problem.kind == "sequences":
            got = kinds.induce_from(problem.shown)
            if got is None:
                named = a_kind_of_thing_she_named(problem.shown)
                if named is not None:
                    maker, _over = a_way_of_building_over(named)
                    kinds.WAYS_TO_BUILD[f"over {named.name}"] = maker
                got = kinds.induce_from(problem.shown)
            return got is not None and all(
                got.read(before) == after for before, after in problem.held_back
            )
        can_do, shown = problem.shown
        world = World(can_do={**can_do, **composed}, can_tell={})
        for act in world.can_do.values():
            if all(act(one) == want for one, want in shown) and all(
                act(one) == want for one, want in problem.held_back
            ):
                return True
        found = an_action_she_composed(
            world, shown, held_out=problem.held_back, deepest=HOW_DEEP
        )
        if found is None:
            return False
        doing, _worth = found
        composed[doing.name] = lambda one, d=doing, w=world: what_it_does(d, one, w)
        return all(
            what_it_does(doing, one, world) == want for one, want in problem.held_back
        )

    return solve


def test_keeping_what_she_works_out_solves_what_forgetting_cannot(a_beginner):
    given, composed, forget, what_she_has = a_beginner
    found = did_keeping_it_help(
        _problems(given),
        solve=_solving(composed),
        forget=forget,
        what_she_has=what_she_has,
    )
    assert found.keeping.rate > found.forgetting.rate
    assert found.helped


def test_what_forgetting_misses_is_exactly_what_needed_an_earlier_answer(a_beginner):
    """Two problems, both of them the ones built on an action composed two
    problems before. Nothing else separates the arms."""
    given, composed, forget, what_she_has = a_beginner
    found = did_keeping_it_help(
        _problems(given),
        solve=_solving(composed),
        forget=forget,
        what_she_has=what_she_has,
    )
    assert set(found.forgetting.missed) == {
        "to the far wall then back one",
        "to the near wall then forward one",
    }
    assert not found.keeping.missed


def test_she_builds_on_what_she_built(a_beginner):
    given, composed, forget, what_she_has = a_beginner
    went = a_run_of_problems(
        _problems(given),
        solve=_solving(composed),
        keeping=True,
        forget=forget,
        what_she_has=what_she_has,
    )
    assert any(
        "while it still changes anything, then" in one for one in went.added
    ), went.added


def test_seconds_are_compared_only_over_what_both_arms_got_right(a_beginner):
    """An arm that fails fast looks cheap. Keeping solved two more and took
    fifteen times as long, which is true and answers nothing."""
    given, composed, forget, what_she_has = a_beginner
    found = did_keeping_it_help(
        _problems(given),
        solve=_solving(composed),
        forget=forget,
        what_she_has=what_she_has,
    )
    shared = found.both_solved
    assert shared == frozenset(found.forgetting.solved)
    assert found.keeping.cost_of(shared) <= found.keeping.cost


def test_nothing_is_said_about_seconds_from_a_single_run(a_beginner):
    """One sample has no spread, so it can carry no timing claim."""
    given, composed, forget, what_she_has = a_beginner
    once = did_keeping_it_help(
        _problems(given),
        solve=_solving(composed),
        forget=forget,
        what_she_has=what_she_has,
        times=1,
    )
    assert once.sooner_at == ()


def test_repeating_it_lets_the_clock_speak(a_beginner):
    """Keeping is faster on sequences because the maker persists, and the
    margin is wider than the runs vary."""
    given, composed, forget, what_she_has = a_beginner
    found = did_keeping_it_help(
        _problems(given),
        solve=_solving(composed),
        forget=forget,
        what_she_has=what_she_has,
        times=3,
    )
    assert "sequences" in found.sooner_at


def test_a_gain_shows_in_both_kinds_of_problem(a_beginner):
    """Sequences and states of a world share no vocabulary, so a gain in each
    is something general having been built rather than one search improving."""
    given, composed, forget, what_she_has = a_beginner
    found = did_keeping_it_help(
        _problems(given),
        solve=_solving(composed),
        forget=forget,
        what_she_has=what_she_has,
        times=3,
    )
    assert "worlds" in found.better_at
    assert "sequences" in found.sooner_at
    assert found.carried
