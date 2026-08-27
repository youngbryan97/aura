"""The same machinery, moved to something it has never seen.

A solver written for one thing is handed the environment, the state, the
actions, the transition rules, the noise model and the objective. The question
worth asking of a general one is not whether it plays the thing it was tuned
on — it is whether the same machinery discovers enough structure in an
unfamiliar world to become competent, without anyone writing the solution.

Refusing a world it has no hypothesis for is a PASS. Claiming a rule that does
not hold is the failure.
"""

from __future__ import annotations

import pytest

from tools.measure_generality import WORLDS, live_in


@pytest.fixture(scope="module")
def lived() -> dict[str, dict]:
    return {
        name: live_in(world, adds_things=adds, starts_full=full, moves=120, seed=0)
        for name, (world, adds, full, _covered) in WORLDS.items()
    }


# ── worlds she has a hypothesis for ──────────────────────────────────────

@pytest.mark.parametrize(
    ("world", "expected"),
    [
        ("slides and combines", "slides and combines"),
        ("slides", "slides"),
        ("still", "does not move"),
    ],
)
def test_she_works_out_a_world_she_has_a_hypothesis_for(lived, world, expected):
    assert lived[world]["rule"] == expected


def test_she_tells_two_worlds_of_the_same_size_apart(lived):
    assert lived["slides"]["rule"] != lived["slides and combines"]["rule"]


def test_and_is_sure_of_what_she_worked_out(lived):
    for world in ("slides and combines", "slides", "still"):
        assert lived[world]["confidence"] >= 0.7


# ── a world nobody wrote a hypothesis for, which she builds one for ──────

def test_a_rule_is_composed_rather_than_chosen_off_a_list():
    """The point of the whole thing, in one assertion.

    Nobody wrote "one thing steps". It falls out of three facts about what a
    push can do — how far a thing carries, whether equals combine, how many
    things move — and a sliding puzzle is one of the eight ways those go.
    """
    from core.perception.how_it_moves import composed

    rule = composed("one place", False, "one thing")
    assert rule.as_facts() == "one thing carries one place, nothing combines"


def test_the_puzzle_is_worked_out_by_composing_one(lived):
    assert lived["one into the gap"]["rule"] == "one thing steps"


def test_and_she_looks_ahead_in_it_like_any_other_world(lived):
    assert lived["one into the gap"]["searched"] > 0


# ── and a world the space does not reach ─────────────────────────────────

def test_she_refuses_a_shape_no_composition_describes(lived):
    """Exchanging two ends is not a distance, a merge or a count of things."""
    assert lived["swaps"]["rule"] == ""


def test_and_says_so_rather_than_claiming_something(lived):
    assert "act and look" in lived["swaps"]["shape"]


def test_a_world_she_never_modelled_is_barely_searched(lived):
    """A few early cycles can pass before enough has been watched to refuse."""
    assert lived["swaps"]["searched"] < lived["slides and combines"]["searched"] / 10


# ── and she uses what she worked out ─────────────────────────────────────

def test_a_world_she_worked_out_is_one_she_looks_ahead_in(lived):
    assert lived["slides and combines"]["searched"] > 0


def test_nothing_in_the_stack_was_written_for_any_of_these():
    """The worlds live in the harness. The machinery has never heard of them."""
    import inspect

    from core.perception import how_it_moves

    source = inspect.getsource(how_it_moves)
    for world in ("swaps", "one_into_the_gap"):
        assert world not in source
    # Naming where a defect was measured is not knowing about a world: the
    # rules are composed from facts about pushing, and the only mentions of
    # any particular thing are in comments recording what caught a bug.
    for named in ("2048", "sliding puzzle"):
        assert not any(
            named in line and not line.lstrip().startswith("#")
            for line in source.splitlines()
        )
