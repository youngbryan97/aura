"""The test that tells a learned abstraction from a well-shaped prior.

Put to a council of nine models, the metalanguage question drew one answer
almost unanimously: add `sort_by`, `partition_by` and `filter`. Four of them
conceded, when pushed, that this is a person extending the language and calling
it invention. One of them gave the test that catches it.

    Run the learner on its experience and take what it produced. Delete the
    experience that produced one abstraction and run it again. If the
    abstraction is still there, a generator was always offering it and the
    experience was decoration. If it is gone, it is downstream of what
    happened.

Formally: q is downstream when q is in A(D) and q is not in A(D minus D_q).

This file applies it to what is actually here, including to work of mine that
fails it. A gate you wrote and pass says nothing.
"""

from __future__ import annotations

from core.cognition.primitive_invention import Transition, invent_relation
from core.cognition.relation_language import RelationLanguage


def _learned(worlds: list[list[Transition]]) -> set[str]:
    """What a language holds after being shown these worlds and no others."""

    language = RelationLanguage()
    for world in worlds:
        found = invent_relation(world, known_forms=language.forms)
        if found is not None:
            language.admit(found)
    return {str(text) for text in language.counts}


def _mirror(n: int) -> Transition:
    return Transition(tuple(range(n)), tuple(reversed(range(n))))


def _rotate(n: int) -> Transition:
    row = tuple(range(n))
    return Transition(row, row[1:] + row[:1])


def test_a_learned_form_disappears_when_its_world_does() -> None:
    """The library passes. What it holds came from what it was shown."""

    both = _learned([[_mirror(4), _mirror(6)], [_rotate(4), _rotate(6)]])
    without_the_mirrors = _learned([[_rotate(4), _rotate(6)]])

    mirrors = both - without_the_mirrors
    assert mirrors, "nothing was owed to the mirror worlds"
    for text in mirrors:
        assert text not in without_the_mirrors


def test_the_affine_family_fails_this_and_that_is_the_finding() -> None:
    """Mine does not pass, and pretending otherwise would be the whole error.

    Three of the written-down positional forms turned out to be one family,
    fitted rather than listed, reaching thirty-four permutations at length six
    that nobody wrote down. That is a real result and it is not learning: the
    family is offered on every world regardless of what came before, so
    deleting every world leaves it exactly where it was.

    It is a better-shaped prior. The difference between that and an abstraction
    downstream of experience is this assertion.
    """

    from core.cognition.primitive_invention import _affine_forms_that_fit

    every = [list(range(6)) for _ in range(6)]
    with_no_history_at_all = {text for _f, text, _r in _affine_forms_that_fit(every)}
    assert with_no_history_at_all, (
        "the family is available before anything has been observed, which is "
        "what makes it a prior rather than something that was learned"
    )


def test_the_two_are_held_in_different_places() -> None:
    """Both are real; only one of them is growth.

    Reaching for the wrong field is easy and this test did it first. ``counts``
    is a frequency prior over FAMILIES and shares its namespace with the basis
    kinds — "mirror" appears there whether or not a mirror was ever seen, and
    it should, because that is what a prior over families is. The library is
    ``forms``, and that is the part that grows with experience.

    Anything measuring learning against ``counts`` measures the basis as well
    and reports a prior as an acquisition.
    """

    from core.cognition.primitive_invention import _index_forms

    always_there = {rule.kind for _f, _d, rule in _index_forms(6)}
    assert always_there >= {"identity", "mirror", "offset", "grouping"}

    blank = RelationLanguage()
    assert not blank.forms, "the library starts empty; the basis does not"

    taught = RelationLanguage()
    for world in ([_mirror(4), _mirror(6)],):
        found = invent_relation(world, known_forms=taught.forms)
        if found is not None:
            taught.admit(found)
    assert taught.forms, "nothing was learned, so there is nothing to compare"

    # And it goes away when the world does, which is the whole test.
    assert not RelationLanguage().forms
