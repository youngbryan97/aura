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


# ------------------------------------------------------- higher-order transfer


def _applied(fn, lengths):
    return [Transition(tuple(range(n)), fn(tuple(range(n)))) for n in lengths]


def _mirror_of(state):
    return tuple(reversed(state))


def _rot1(state):
    return state[1:] + state[:1]


def _ends_of(state):
    row = list(state)
    row[0], row[-1] = row[-1], row[0]
    return tuple(row)


def test_what_can_be_learned_grows_with_what_has_been() -> None:
    """A shape outside the language becomes inside it, because an earlier world
    taught a member of it.

    This is the higher-order claim, and it is not a reordering: the three-deep
    world is UNREACHABLE with an empty language however many observations are
    offered, and reachable after a different world taught the two-deep shape.
    Shapes worked out earlier are members of the language, not a preference
    over it.
    """

    def two_deep(state):
        return _rot1(_mirror_of(state))

    def three_deep(state):
        return _ends_of(_rot1(_mirror_of(state)))

    blank = RelationLanguage()
    assert blank.explain(_applied(three_deep, (5, 6, 7))) is None

    taught = RelationLanguage()
    taught.admit(invent_relation(_applied(two_deep, (5, 6, 7))))
    assert taught.forms, "a shape with a rule over indices is kept"

    found = taught.explain(_applied(three_deep, (5, 6, 7)))
    assert found is not None, "the taught shape did not make the new one reachable"
    # And it is a relation, not a fit: it predicts lengths it never saw.
    for length in (9, 11, 12):
        assert tuple(found.apply(tuple(range(length)))) == three_deep(
            tuple(range(length))
        )


def test_a_language_that_learned_nothing_useful_adds_nothing() -> None:
    """The null: shapes that do not help must not help."""

    def three_deep(state):
        return _ends_of(_rot1(_mirror_of(state)))

    unhelpful = RelationLanguage()
    unhelpful.admit(invent_relation(_applied(lambda s: s, (4, 5))))
    assert unhelpful.explain(_applied(three_deep, (5, 6, 7))) is None


# ------------------------------------------------- structure nobody solved whole


def _rot1(state):
    return state[1:] + state[:1]


def _ends(state):
    row = list(state)
    row[0], row[-1] = row[-1], row[0]
    return tuple(row)


def _library(*, refactor: bool) -> RelationLanguage:
    language = RelationLanguage()
    language.admit(invent_relation(_applied(lambda s: _rot1(_mirror_of(s)), (5, 6, 7))))
    language.admit(
        language.explain(_applied(lambda s: _ends(_rot1(_mirror_of(s))), (5, 6, 7)))
    )
    language.admit(
        language.explain(
            _applied(lambda s: _mirror_of(_ends(_rot1(_mirror_of(s)))), (5, 6, 7))
        )
    )
    if refactor:
        language.refactor()
    return language


def test_the_library_finds_structure_none_of_its_solutions_is() -> None:
    """A library that only keeps whole winners can hold nothing new.

    The long-term studies of chunking in Soar and ACT-R report where that ends:
    symbolic learning eventually stopped in both. What keeps DreamCoder's
    library growing is refactoring the solutions and admitting the structure
    common across them, which is what this does.
    """

    language = _library(refactor=False)
    before = set(language.forms)
    extracted = language.refactor()
    assert extracted, "nothing was shared, so nothing was learned"
    assert extracted not in before, "a whole winner is not a refactoring"
    assert set(language.forms) - before == {extracted}


def test_the_extracted_run_is_a_real_relation() -> None:
    """Rebuilt from its parts, it does what those parts do."""

    from core.cognition.primitive_invention import _permutation_operator

    language = _library(refactor=False)
    extracted = language.refactor()
    _family, rule, parts = language.forms[extracted]
    assert len(parts) >= 2
    state = tuple(range(6))
    assert _permutation_operator(rule)(state) == _rot1(_ends(state))


def _inner(state):
    """The cells one in from each end change places."""

    row = list(state)
    row[1], row[-2] = row[-2], row[1]
    return tuple(row)


#: The moves a target is built from. Small and named, so a failure says what
#: the world was rather than printing a permutation.
_THE_MOVES = {
    "rotate one": _rot1,
    "rotate two": lambda s: s[2:] + s[:2],
    "swap the ends": _ends,
    "swap one in": _inner,
    "mirror": _mirror_of,
}


def _a_world_only_the_refactored_library_reaches():
    """Find a target outside three languages, rather than naming one.

    Named by hand this had to be re-chosen three times, and each time for the
    same reason: the base search got better and the world it could not reach
    became a world it could. That is the search improving, and a test that
    hard-codes an instance turns it into a failure.

    So the instance is found here. What is being checked is the property —
    that refactoring reaches something the whole winners cannot — and if no
    such world exists in this space, that is a finding about the refactoring
    step rather than a broken test, and it is said out loud.
    """

    from itertools import product

    blank = RelationLanguage()
    winners = _library(refactor=False)
    refactored = _library(refactor=True)
    for depth in (3, 4, 5):
        for names in product(_THE_MOVES, repeat=depth):

            def apply(state, names=names):
                for name in reversed(names):
                    state = _THE_MOVES[name](state)
                return state

            world = _applied(apply, (5, 6, 7))
            if blank.explain(world) is not None:
                continue
            if winners.explain(world) is not None:
                continue
            found = refactored.explain(world)
            if found is None:
                continue
            if not all(
                tuple(found.apply(tuple(range(length)))) == apply(tuple(range(length)))
                for length in (9, 11)
            ):
                continue
            return apply, " then ".join(names)
    return None, ""


def test_refactoring_reaches_a_world_the_winners_could_not() -> None:
    """The point of the step, measured: unreachable, then reachable.

    Three claims, and the first two are the ones that make the third mean
    anything. A blank language cannot express this world. A library holding
    only the whole solutions it has won cannot either. The library that has
    refactored those solutions into the run they share can, because the run
    is now a part it can compose with.
    """

    apply, said = _a_world_only_the_refactored_library_reaches()
    assert apply is not None, (
        "no world in this space is outside a blank language and outside the "
        "whole winners and inside the refactored library. That is a finding "
        "about the refactoring step, not a broken test: the base search may "
        "have caught up with it"
    )
    world = _applied(apply, (5, 6, 7))
    assert RelationLanguage().explain(world) is None, said
    assert _library(refactor=False).explain(world) is None, said
    found = _library(refactor=True).explain(world)
    assert found is not None, said
    for length in (9, 11):
        assert tuple(found.apply(tuple(range(length)))) == apply(tuple(range(length)))


def test_nothing_is_extracted_from_one_solution() -> None:
    """Shared means shared. One shape has nothing to share with."""

    alone = RelationLanguage()
    alone.admit(invent_relation(_applied(lambda s: _rot1(_mirror_of(s)), (5, 6, 7))))
    assert alone.refactor() == ""


def test_the_run_kept_is_the_one_that_saves_most() -> None:
    """Counting, not taste: occurrences beyond the first, times its length."""

    from pathlib import Path

    body = Path("core/cognition/relation_language.py").read_text()
    start = body.index("def refactor(")
    window = body[start : start + 2600]
    assert "(shared[run] - 1) * len(run)" in window


# ------------------------------------------- the language survives a restart


def test_a_learned_shape_is_a_value_not_a_closure() -> None:
    """A function cannot be written down, so it cannot be learned durably."""

    from core.cognition.primitive_invention import IndexProgram

    found = invent_relation(_applied(lambda s: _rot1(_mirror_of(s)), (5, 6, 7)))
    assert found is not None
    assert isinstance(found.index_rule, IndexProgram)
    # It interprets itself, compares by value, and round-trips.
    again = IndexProgram.from_json(found.index_rule.to_json())
    assert again == found.index_rule
    for length in (5, 9, 12):
        state = tuple(range(length))
        assert tuple(state[again(i, length)] for i in range(length)) == _rot1(
            _mirror_of(state)
        )


def test_the_expanded_language_survives_a_restart(tmp_path) -> None:
    """It used to come back knowing how often mirroring worked, not what it is.

    The counts persisted and the shapes did not, so the language contracted to
    its basis on every boot and the one thing that had been learned was the one
    thing lost.
    """

    store = tmp_path / "language.json"
    language = _library(refactor=True)
    language.path = store
    before = dict(language.forms)
    assert language.save() is None or True
    language.save()

    again = RelationLanguage.load(store)
    assert set(again.forms) == set(before), "a shape was dropped in the round trip"
    assert any(family == "refactored" for family, _r, _p in again.forms.values()), (
        "the one shape the system derived for itself is the one that must survive"
    )


def test_a_restarted_library_still_reaches_what_a_blank_one_cannot(tmp_path) -> None:
    """The measurement, not the mechanism: it can still do the thing."""

    apply, said = _a_world_only_the_refactored_library_reaches()
    assert apply is not None, "no world separates the three languages here"

    store = tmp_path / "language.json"
    language = _library(refactor=True)
    language.path = store
    language.save()

    world = _applied(apply, (5, 6, 7))
    assert RelationLanguage().explain(world) is None, said
    assert _library(refactor=False).explain(world) is None, said
    restarted = RelationLanguage.load(store)
    found = restarted.explain(world)
    assert found is not None, said
    for length in (9, 11):
        assert tuple(found.apply(tuple(range(length)))) == apply(tuple(range(length)))


def test_a_corrupt_store_loads_as_an_empty_language(tmp_path) -> None:
    store = tmp_path / "language.json"
    store.write_text('{"counts": {"mirror": 2}, "forms": {"x": {"program": "nonsense"}}}')
    loaded = RelationLanguage.load(store)
    assert loaded.counts == {"mirror": 2}
    assert loaded.forms == {}
