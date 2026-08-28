"""Structure learned in one world becomes better priors on entering the next.

Two claims are checked here and they are different claims.

The first is that the hypothesis language extends itself. Given a world nothing
in the current language can express, the mechanism works out what relation
would fit — from the observations, with no operator names anywhere in it — and
refuses when there is nothing to find. Swap, rotation and reversal all fall out
of one solve; none of those words appears in the module.

The second is transfer: Train(A, D1) -> dP(A, D2) > 0 for D2 the system has not
seen. Measured as the number of observations a world takes to pin down, with
and without shapes learned on other worlds, over a generated population rather
than a few chosen cases. The cost is measured the same way: a prior that is
wrong about a world makes it take longer, and that shows up in the same table.
"""

from __future__ import annotations

import pytest

from core.cognition.primitive_invention import (
    Transition,
    invent_relation,
    language_is_sufficient,
)
from core.cognition.relation_language import RelationLanguage, observations_needed


def _mirror(n: int) -> Transition:
    return Transition(tuple(range(n)), tuple(reversed(range(n))))


def _offset(n: int, k: int) -> Transition:
    row = tuple(range(n))
    return Transition(row, row[k:] + row[:k])


def _exchange(n: int, i: int, j: int) -> Transition:
    row = list(range(n))
    row[i], row[j] = row[j], row[i]
    return Transition(tuple(range(n)), tuple(row))


def _gains(n: int, d: int) -> Transition:
    return Transition(tuple(range(n)), tuple(v + d for v in range(n)))


# ---------------------------------------------------------------- invention


def test_the_language_she_has_does_not_cover_the_swap_world() -> None:
    """The refusal is correct, and is the thing worth getting past."""

    have = [
        lambda s: (s[-1],) + s[:-1],
        lambda s: tuple("x" for _ in s),
        lambda s: s,
    ]
    assert not language_is_sufficient(have, [_exchange(4, 1, 3)])


@pytest.mark.parametrize(
    ("world", "family"),
    [
        ([_exchange(4, 1, 3), _exchange(6, 1, 3)], "pairwise exchange"),
        ([_mirror(4), _mirror(3)], "mirror"),
        ([_offset(4, 2), _offset(6, 2)], "offset"),
        ([_gains(3, 3), _gains(5, 3)], "value offset"),
    ],
)
def test_the_shape_falls_out_of_the_solve(world, family: str) -> None:
    """No operator names anywhere: the correspondence is solved for."""

    found = invent_relation(world)
    assert found is not None
    assert found.family == family
    assert found.generalises


def test_a_relation_must_explain_what_it_was_not_built_from() -> None:
    found = invent_relation([_mirror(4), _mirror(3)], held_out=[_mirror(7)])
    assert found is not None
    assert found.held_out_checked == 1
    # And the same relation offered a world it does not explain is refused.
    assert invent_relation([_mirror(4)], held_out=[_offset(4, 1)]) is None


def test_a_world_with_no_relation_in_it_invents_nothing() -> None:
    noise = [
        Transition((1, 2, 3), (9, 4, 7)),
        Transition((4, 5, 6), (2, 8, 1)),
    ]
    assert invent_relation(noise) is None


def test_no_operator_name_appears_in_the_mechanism() -> None:
    """"Not adding swap" is checkable, so it is checked."""

    from pathlib import Path

    body = Path("core/cognition/primitive_invention.py").read_text().lower()
    code = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    )
    # The prose explains what it is not doing; the code must not name them.
    body_only = code[code.index("from __future__") :]
    for word in ("swap(", "rotate(", "reverse(", "def swap", "def rotate"):
        assert word not in body_only, word


# ---------------------------------------------------------------- transfer


def _language_taught_on(worlds) -> RelationLanguage:
    learned = RelationLanguage()
    for world in worlds:
        learned.admit(invent_relation(world))
    return learned


def test_the_right_prior_settles_a_world_no_later_than_none() -> None:
    """Asserted as a relationship, because the absolute numbers move.

    An earlier version of this fixed the numbers at two observations without a
    prior and one with. Composing shapes made a plain mirror resolvable from a
    single observation, so the number fell to one either way, and the test
    failed for an improvement. What the claim actually is: the shape a world
    HAS settles it soonest, and a shape it does not have never settles it
    sooner than that.
    """

    worlds = {
        "mirror": [_mirror(2), _mirror(4), _mirror(5)],
        "offset": [_offset(2, 1), _offset(4, 1), _offset(5, 1)],
    }
    priors = {
        name: _language_taught_on([[build(n)] for n in (3, 5, 6)])
        for name, build in (
            ("mirror", _mirror),
            ("offset", lambda n: _offset(n, 1)),
        )
    }

    settled_sooner = 0
    for name, world in worlds.items():
        blank = observations_needed(world, language=RelationLanguage())
        right = observations_needed(world, language=priors[name])
        assert blank is not None and right is not None
        assert right <= blank, f"{name}: the right prior made it worse"
        settled_sooner += int(right < blank)
        for other, prior in priors.items():
            if other == name:
                continue
            wrong = observations_needed(world, language=prior)
            assert wrong is not None and wrong >= right
    assert settled_sooner >= 1, "no world settled sooner: that is not transfer"


def test_a_prior_never_manufactures_structure() -> None:
    taught = _language_taught_on([[_mirror(n)] for n in (3, 5, 6)])
    noise = [
        Transition((1, 2, 3), (9, 4, 7)),
        Transition((4, 5, 6), (2, 8, 1)),
        Transition((7, 8, 9), (3, 3, 3)),
    ]
    assert observations_needed(noise, language=RelationLanguage()) is None
    assert observations_needed(noise, language=taught) is None


def test_the_gain_holds_over_a_generated_population() -> None:
    """Not three chosen worlds: every shape, taught on some, tried on others.

    Each shape is taught on three worlds and then measured on four it has not
    seen, and the same four are measured with no prior at all. The claim is
    that the total observations needed falls, and that it falls for every
    shape rather than on average because one of them is dramatic.
    """

    shapes = {
        "mirror": lambda n: _mirror(n),
        "offset": lambda n: _offset(n, 1),
        "pairwise exchange": lambda n: _exchange(n, 0, 1),
    }
    taught_lengths = (7, 8, 9)
    unseen_lengths = (2, 3, 4, 5)

    total_blank = 0
    total_taught = 0
    for name, build in shapes.items():
        taught = _language_taught_on([[build(n)] for n in taught_lengths])
        assert taught.families()[0] == name
        blank_cost = 0
        taught_cost = 0
        for length in unseen_lengths:
            world = [build(length), build(length + 10), build(length + 20)]
            blank = observations_needed(world, language=RelationLanguage())
            after = observations_needed(world, language=taught)
            assert blank is not None and after is not None
            blank_cost += blank
            taught_cost += after
        assert taught_cost <= blank_cost, f"{name}: {taught_cost} > {blank_cost}"
        total_blank += blank_cost
        total_taught += taught_cost

    assert total_taught < total_blank


def test_the_language_carries_nothing_about_the_world_it_came_from() -> None:
    """What makes it transfer: it holds shapes, not states or domains."""

    taught = _language_taught_on([[_mirror(9)], [_exchange(7, 2, 4)]])
    assert set(taught.counts) == {"mirror", "pairwise exchange"}
    assert all(isinstance(count, int) for count in taught.counts.values())


def test_shapes_survive_a_round_trip_through_disk(tmp_path) -> None:
    taught = _language_taught_on([[_mirror(4)], [_mirror(6)], [_offset(5, 2)]])
    taught.path = tmp_path / "relations.json"
    taught.save()
    again = RelationLanguage.load(tmp_path / "relations.json")
    assert again.counts == taught.counts
    assert again.families()[0] == "mirror"


def test_an_absent_store_loads_as_an_empty_language() -> None:
    empty = RelationLanguage.load("/no/such/relations.json")
    assert empty.counts == {}
    assert empty.families() == []
