"""A proof beats a search that gave up.

"No form fits" was said for a world needing one more example and for a world
needing a different kind of language. Same sentence, nothing to act on.

There is a proof available and it is cheap. Every positional program computes
after[i] = before[f(i, n)], where f sees the position and the length and never
the cells. Composing two of them gives x[g(f(i,n), n)] — value-blind again — so
by induction no depth of composition ever introduces dependence on the cells.
Depth cannot be the answer, which makes "is this reachable" decidable instead of
merely unproven.
"""

from __future__ import annotations

from core.cognition.language_limits import certify
from core.cognition.primitive_invention import Transition, _grouped_source


def test_sorting_is_proven_outside_rather_than_unfound() -> None:
    """Two states of one length, needing two different rules.

    (3,1,2) puts position 0's cell at 1; (1,3,2) puts it at 0. One rule reading
    only the position and the length has to answer both, and cannot.
    """

    verdict = certify(
        [
            Transition((3, 1, 2), (1, 2, 3)),
            Transition((1, 3, 2), (1, 2, 3)),
        ]
    )
    assert verdict.proven_outside
    assert verdict.length == 3
    assert verdict.position is not None


def test_a_change_of_length_needs_no_search_at_all() -> None:
    verdict = certify([Transition((1, 2, 3, 4), (2, 4))])
    assert verdict.proven_outside
    assert "became" in verdict.reason


def test_grouping_is_not_escalated() -> None:
    """The shape a person had to add by hand was inside the envelope.

    It is value-blind: which class a cell belongs to depends on where it sits,
    not on what it holds. A detector that called this a case for a different
    KIND of language would be measuring its own appetite rather than the world.
    """

    world = [
        Transition(tuple(range(n)), tuple(_grouped_source(i, n, 2, 0) for i in range(n)))
        for n in (6, 8)
    ]
    assert not certify(world).proven_outside


def test_the_ordinary_shapes_are_not_escalated() -> None:
    for build in (
        lambda row: tuple(reversed(row)),
        lambda row: row[1:] + row[:1],
        lambda row: row,
    ):
        world = [
            Transition(tuple(range(n)), build(tuple(range(n)))) for n in (4, 5, 6)
        ]
        assert certify(world).standing == "inside"


def test_one_observation_proves_nothing() -> None:
    """Sorting and a fixed permutation are the same world until the second one.

    (3,1,2) -> (1,2,3) is "sort" and is equally "position i takes from i+1".
    Calling for a wider language here would be inventing a kind to explain a
    world that an ordinary rule already explains.
    """

    verdict = certify([Transition((3, 1, 2), (1, 2, 3))])
    assert not verdict.proven_outside


def test_a_repeated_value_never_manufactures_the_contradiction() -> None:
    """Every possible source is kept, so no tie-break invents a proof.

    Choosing one source for a repeated cell would produce contradictions that
    are artefacts of the choice — the detector would prove worlds outside the
    language because of how it read them.
    """

    world = [
        Transition((5, 5, 1), (1, 5, 5)),
        Transition((5, 5, 2), (2, 5, 5)),
    ]
    assert not certify(world).proven_outside


def test_an_unknown_kind_is_not_the_identity() -> None:
    """It returned the identity, and the identity fits some worlds.

    A program written by a later build and read back by this one would have
    been reported as a relation that was found rather than one that could not
    be run.
    """

    import pytest

    from core.cognition.primitive_invention import IndexProgram

    with pytest.raises(ValueError):
        IndexProgram("something_this_build_has_never_heard_of")(0, 4)


def test_a_transformed_cell_is_not_a_proof_of_anything() -> None:
    """"Mirror, then add one to every cell" was proved outside the language.

    It is squarely inside it: a value-blind rule about where cells come from,
    and a map applied to what they hold. Every source set came back empty — not
    because the observations contradicted each other, but because nothing in
    the state held the value being looked for — and empty was read as
    contradiction.

    Where the cells were transformed, the premise of this proof does not hold,
    so the proof is not claimed.
    """

    verdict = certify(
        [
            Transition((1, 2, 3, 4), (5, 4, 3, 2)),
            Transition((2, 3, 4, 5), (6, 5, 4, 3)),
        ]
    )
    assert not verdict.proven_outside
    assert verdict.standing == "undecided"


def test_a_shortage_of_evidence_is_told_apart_from_both() -> None:
    """Three failures wore one face; a repeated cell is the third.

    When a value recurs, more than one place it could have come from survives
    every observation. That is not a language problem and not a search problem.
    """

    verdict = certify(
        [
            Transition((5, 5, 1), (1, 5, 5)),
            Transition((5, 5, 2), (2, 5, 5)),
        ]
    )
    assert verdict.standing == "undecided"
    assert "another example" in verdict.reason


def test_the_proof_actually_fires_on_the_battery() -> None:
    """The test that was missing, and its absence is the whole lesson.

    The proof was written, tested against worlds built by hand for it, and
    passed. On the one battery that exists it fired on zero problems out of a
    hundred and twenty — including all ten it was written for — because every
    problem showed its two states at two DIFFERENT lengths. Each length was seen
    once, there was nothing to intersect, and a correct mechanism sat inert
    behind a green suite.

    Repetition within a length is what makes a language refutable. Variation
    across lengths is what makes a rule identifiable. A battery needs both.
    """

    from core.cognition.induction_battery import (
        BEYOND_THE_LANGUAGE,
        generate_battery,
    )

    battery = generate_battery()
    outside = [p for p in battery if p.shape in BEYOND_THE_LANGUAGE]
    inside = [p for p in battery if p.shape not in BEYOND_THE_LANGUAGE]
    assert len(outside) == 20

    proved = [p for p in outside if certify(list(p.shown)).proven_outside]
    assert len(proved) >= 16, f"only {len(proved)}/{len(outside)} refuted"

    # And never on a shape that is expressible. A proof that fires where a rule
    # exists is worse than no proof.
    assert not [p for p in inside if certify(list(p.shown)).proven_outside]


def test_a_length_is_shown_twice() -> None:
    """The property the battery has to have for any of that to be possible."""

    from core.cognition.induction_battery import generate_battery

    for problem in generate_battery():
        lengths = [len(item.before) for item in problem.shown]
        assert len(lengths) > len(set(lengths)), problem.name
        # And still more than one length, or nothing is identifiable.
        assert len(set(lengths)) >= 2, problem.name


def test_non_empty_sources_are_not_enough() -> None:
    """Two positions wanting the same cell cannot both have it.

    Emptiness per position is necessary and is not sufficient. A rearrangement
    puts each cell somewhere once, so the possible sources have to be handed
    out one each — and checking only for emptiness let a world through as
    "inside the language" that no rearrangement can produce, where the next
    thing that happened was no form fitting and no reason given.
    """

    from core.cognition.language_limits import _matches_one_to_one

    # Every set non-empty, and both positions want the same single cell.
    assert not _matches_one_to_one([{0}, {0}, {2}], 3)
    # The same sets, made satisfiable.
    assert _matches_one_to_one([{0}, {1}, {2}], 3)
    # Overlapping but satisfiable, which a greedy pick can get wrong.
    assert _matches_one_to_one([{0, 1}, {0}, {2}], 3)
    # More positions than cells.
    assert not _matches_one_to_one([{0}, {1}, {2}, {0, 1}], 3)


def test_a_refutation_says_whether_it_survives_a_bad_observation() -> None:
    """One corrupted transition empties an intersection on its own.

    The proof would then be about the corruption rather than about the
    language, and would read exactly the same either way.
    """

    solid = certify(
        [
            Transition((3, 1, 2), (1, 2, 3)),
            Transition((1, 3, 2), (1, 2, 3)),
            Transition((2, 1, 3), (1, 2, 3)),
        ]
    )
    assert solid.proven_outside
    assert solid.robust

    # Two observations cannot be leave-one-out tested at all.
    thin = certify(
        [Transition((3, 1, 2), (1, 2, 3)), Transition((1, 3, 2), (1, 2, 3))]
    )
    assert thin.proven_outside
    assert not thin.robust


def test_the_interpreter_and_the_reader_agree_on_what_a_kind_is() -> None:
    """A kind the interpreter cannot run must not be read back as a program.

    from_json accepted any non-empty string, so anything written by a later
    build — or by another kind of learned relation entirely — came back as an
    IndexProgram that raised the first time it was asked for a position.
    """

    from core.cognition.primitive_invention import (
        _KINDS_THIS_BUILD_INTERPRETS,
        IndexProgram,
        _index_forms,
    )

    assert IndexProgram.from_json({"kind": "a_kind_from_a_later_build"}) is None
    assert IndexProgram.from_json({"kind": "mirror"}) is not None

    # And every kind the forms actually build is one the reader accepts.
    for _family, _said, rule in _index_forms(6):
        assert rule.kind in _KINDS_THIS_BUILD_INTERPRETS, rule.kind
    assert "affine" in _KINDS_THIS_BUILD_INTERPRETS


def test_what_the_rule_writes_is_arithmetic_not_recognition() -> None:
    """A cell survives, disappears, or arrives. Three cases, and no fourth.

    Comparing what went in with what came out settles which. Nothing here
    recognises a family or matches a name, so there is no fourth case for a
    later one to be missing from — which is the property a list of
    transformation names never has.
    """

    reorders = certify([Transition((1, 2, 3), (3, 2, 1)), Transition((4, 5, 6), (6, 5, 4))])
    drops = certify([Transition((1, 2, 3, 4), (2, 4))])
    creates = certify([Transition((1, 2, 3), (2, 4, 6)), Transition((4, 5, 6), (8, 10, 12))])

    assert reorders.writes == "reorders"
    assert drops.writes == "drops"
    assert creates.writes == "creates"


def test_creating_cells_is_not_a_proof_of_anything() -> None:
    """No rule about where a cell CAME FROM can do it. One about what it BECOMES can.

    "Every value becomes 2 times itself" creates cells that never went in, and
    the solver has it. Reading the multiset as a refutation would have called
    an expressible world impossible.
    """

    from core.cognition.primitive_invention import invent_relation

    doubled = [Transition((1, 2, 3), (2, 4, 6)), Transition((4, 5, 6), (8, 10, 12))]
    verdict = certify(doubled)
    assert verdict.writes == "creates"
    assert not verdict.proven_outside
    found = invent_relation(doubled)
    assert found is not None and "2 times itself" in found.form


def test_every_verdict_says_what_the_rule_writes() -> None:
    """Including the ones that return early, which is where it was missing."""

    for world in (
        [Transition((1, 2, 3, 4), (2, 4))],
        [Transition((1, 2, 3), (1, 2, 3, 0))],
        [Transition((3, 1, 2), (1, 2, 3)), Transition((1, 3, 2), (1, 2, 3))],
        [Transition((1, 2, 3), (3, 2, 1)), Transition((4, 5, 6), (6, 5, 4))],
    ):
        assert certify(world).writes in {"reorders", "drops", "creates"}
