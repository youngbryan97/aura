"""Three of the written-down forms are one family, fitted rather than listed.

The basis was grown one form at a time by a person: identity and mirror, then
offset, then the exchanges, then grouping — each added to reach a family that
was failing, each predicted to fix that family and nothing else. Adding a form
to reach a failure is the act this is trying to stop needing.

Measured here: identity, mirror and offset are not three kinds. They are three
members of

    f(i) = (a*i + b) mod m,   for i < m,   else i

and the member is SOLVED for from the observations rather than enumerated. What
that buys is reach nobody wrote down: at length six the written-down positional
forms reach fifteen permutations and the family reaches forty-four.

What it does not buy is grouping, and that is recorded here too. An even deal —
a span that divides the length — is in the family; an uneven one is genuinely
piecewise and is not. The claim that four forms collapse into one was too
strong, and the number that showed it is in this file rather than in a note.
"""

from __future__ import annotations

from core.cognition.primitive_invention import (
    IndexProgram,
    _affine_forms_that_fit,
    _grouped_source,
)


def _fits(target: list[int]) -> bool:
    return bool(_affine_forms_that_fit([[place] for place in target]))


def test_identity_mirror_and_offset_are_one_family() -> None:
    """Not three kinds. Three members, each solved for from what was seen."""

    for length in range(3, 9):
        assert _fits([IndexProgram("identity")(i, length) for i in range(length)])
        assert _fits([IndexProgram("mirror")(i, length) for i in range(length)])
        for step in range(1, length):
            target = [IndexProgram("offset", (step,))(i, length) for i in range(length)]
            assert _fits(target), f"offset {step} at {length}"


def test_an_even_deal_is_in_the_family_and_an_uneven_one_is_not() -> None:
    """The honest half of the claim.

    Dealing six cells into two classes is (2, 0) mod n-1 — the shuffle a person
    had to add by hand, derivable. Dealing seven into three is not affine at
    all: the classes come out different sizes and the map is piecewise.
    """

    assert _fits([_grouped_source(i, 6, 2, 0) for i in range(6)])
    assert not _fits([_grouped_source(i, 7, 3, 0) for i in range(7)])


def test_the_family_reaches_what_nobody_wrote_down() -> None:
    """The point of fitting rather than listing."""

    length = 6
    listed = set()
    for form in (IndexProgram("identity"), IndexProgram("mirror")):
        listed.add(tuple(form(i, length) for i in range(length)))
    for step in range(1, length):
        listed.add(tuple(IndexProgram("offset", (step,))(i, length) for i in range(length)))
    for span in range(2, length // 2 + 1):
        for first in range(span):
            listed.add(tuple(_grouped_source(i, length, span, first) for i in range(length)))

    every = [[place for place in range(length)] for _ in range(length)]
    reached = {
        tuple(rule(i, length) for i in range(length))
        for _f, _d, rule in _affine_forms_that_fit(every)
    }
    assert len(reached) > 2 * len(listed), (len(reached), len(listed))
    assert reached - listed, "a family that reaches nothing new is a rename"


def test_a_member_means_the_same_thing_at_two_lengths() -> None:
    """Stated relative to the length, the way the end-exchanges are.

    An absolute modulus cannot say a shape: dealing into two classes is mod 5 at
    length six and mod 7 at length eight, so one shape seen at two lengths was
    two shapes and intersected to nothing. The battery fell fourteen problems
    before this was written the other way round.
    """

    at_six = {text for _f, text, _r in _affine_forms_that_fit([[i] for i in range(6)])}
    at_eight = {text for _f, text, _r in _affine_forms_that_fit([[i] for i in range(8)])}
    assert "position i takes from 1i+0 (mod n)" in at_six & at_eight


def test_the_family_is_reached_for_only_where_the_language_fails() -> None:
    """A wider net catches things the written-down forms already caught.

    Offered beside them it changed answers that were already right: three
    groupings went from found to lost, and the family had not gained a single
    problem to pay for them. A language is extended where it fails.
    """

    from core.cognition.induction_battery import (
        generate_battery,
        score_battery,
        teach_the_language,
    )
    from core.cognition.relation_language import RelationLanguage

    battery = generate_battery()

    def scored(without: frozenset[str]):
        language = RelationLanguage()
        teach_the_language(battery, language=language, without=without)
        return score_battery(battery, language=language, without=without)

    whole = scored(frozenset())
    authored_only = scored(frozenset({"affine"}))
    assert whole.solved >= authored_only.solved, (
        f"the family cost {authored_only.solved - whole.solved} problems"
    )
    assert whole.solved_expressible == authored_only.solved_expressible
