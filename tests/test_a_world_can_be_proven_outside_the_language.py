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
