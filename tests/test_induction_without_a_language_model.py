"""A number for what is left when the foundation model is taken away.

The standing criticism is that removing the resident model makes what remains
sharply narrower. That is a claim about a number and there was no number. This
is the number, on a frozen battery, with nothing in the path that consults a
model, an embedding or a stored answer.

The battery is built to be failable. One of its ten shapes is outside anything
the mechanism can express, and it is named as such, because a battery a
mechanism cannot fail measures nothing about it. It was two until grouping was
added; the count here said two for as long as it took somebody reading the
file to check it against BEYOND_THE_LANGUAGE, which is the assertion that
actually holds. The score on shapes it can
express is reported separately from the score on all of them: "got it wrong"
and "was never able to say it" are different facts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.cognition.induction_battery import (
    BEYOND_THE_LANGUAGE,
    generate_battery,
    learning_curve,
    score_battery,
)
from core.cognition.primitive_invention import Transition, invent_relation
from core.cognition.relation_language import RelationLanguage, observations_needed

_FLOOR = Path("config/induction_battery_floor.json")


@pytest.fixture(scope="module")
def battery():
    return generate_battery()


@pytest.fixture(scope="module")
def report(battery):
    """Scored with the language taught, which is how the floor is recorded."""

    from core.cognition.induction_battery import teach_the_language

    taught = RelationLanguage()
    teach_the_language(battery, language=taught)
    return score_battery(battery, language=taught)


def test_the_battery_is_frozen_and_failable(battery) -> None:
    """Same problems before and after any change, and some are unreachable."""

    again = generate_battery()
    assert [item.name for item in battery] == [item.name for item in again]
    assert len(battery) == 120
    shapes = {item.shape for item in battery}
    assert BEYOND_THE_LANGUAGE < shapes
    representations = {item.representation for item in battery}
    # Four of the five are not what the mechanism was written for.
    assert representations >= {"integers", "words", "colours", "records", "grid rows"}


def test_the_score_only_goes_up(report) -> None:
    try:
        floor = json.loads(_FLOOR.read_text())
    except (OSError, ValueError):
        pytest.fail(f"{_FLOOR} must record the floor this holds")
    assert report.solved >= int(floor["solved"]), (
        f"{report.line()} on the frozen battery, down from "
        f"{floor['solved']}/{floor['attempted']}."
    )
    assert report.solved_expressible >= int(floor["solved_expressible"])


def test_the_representation_does_not_matter(report) -> None:
    """A structural shape should not care what the cells contain.

    Written for sequences of integers, scored on words, colours, records and
    grids whose cells are themselves tuples. If the shapes were numeric this
    is where it would show.
    """

    scores = {
        name: solved / seen
        for name, (solved, seen) in report.by_representation.items()
    }
    assert len(scores) == 5
    # The bare report has no taught language, so the deep shapes are out of
    # reach in every representation equally — which is the point: the spread is
    # what matters here, not the level.
    assert min(scores.values()) >= 0.55, scores
    # And no representation is more than one problem away from the best.
    spread = max(scores.values()) - min(scores.values())
    assert spread <= 0.2, scores


def test_a_shape_beyond_the_language_is_reported_as_that(report) -> None:
    assert BEYOND_THE_LANGUAGE == {"reordered by the cells"}
    for shape in BEYOND_THE_LANGUAGE:
        solved, seen = report.by_shape[shape]
        assert seen == 10
        assert solved <= 2, f"{shape} should be out of reach, scored {solved}/{seen}"
    assert report.attempted_expressible == 110


def test_composition_is_found_without_either_half_being_given(report) -> None:
    """"Mirror then rotate" looks like neither a mirror nor a rotation.

    Twenty of the hundred were unreachable however many observations were
    offered, until shapes could be composed.
    """

    for shape in ("mirror then rotate", "rotate then exchange the ends"):
        solved, seen = report.by_shape[shape]
        assert solved >= 6, f"{shape}: {solved}/{seen}"


def test_the_simpler_description_wins() -> None:
    """A world that IS a plain mirror is never explained as two things."""

    found = invent_relation(
        [
            Transition((1, 2, 3, 4), (4, 3, 2, 1)),
            Transition(("a", "b", "c"), ("c", "b", "a")),
        ]
    )
    assert found is not None
    assert found.family == "mirror"
    assert "then" not in found.form


def test_the_right_prior_is_never_worse_than_no_prior() -> None:
    """The transfer matrix: the diagonal is the gain, the rest is the cost."""

    def mirror(n: int) -> Transition:
        return Transition(tuple(range(n)), tuple(reversed(range(n))))

    def offset(n: int) -> Transition:
        row = tuple(range(n))
        return Transition(row, row[1:] + row[:1])

    def taught(build) -> RelationLanguage:
        language = RelationLanguage()
        for length in (3, 5, 6):
            language.admit(invent_relation([build(length)]))
        return language

    worlds = {"mirror": [mirror(2), mirror(4), mirror(5)],
              "offset": [offset(2), offset(4), offset(5)]}
    priors = {"mirror": taught(mirror), "offset": taught(offset)}

    gains = 0
    for name, world in worlds.items():
        blank = observations_needed(world, language=RelationLanguage())
        right = observations_needed(world, language=priors[name])
        assert blank is not None and right is not None
        assert right <= blank, f"{name}: the right prior made it worse"
        gains += int(right < blank)
        for other, prior in priors.items():
            if other == name:
                continue
            wrong = observations_needed(world, language=prior)
            assert wrong is not None
            assert wrong >= right, f"{name}: a wrong prior beat the right one"
    assert gains >= 1, "no world was settled sooner: that is not transfer"


def test_a_prior_never_manufactures_an_answer() -> None:
    noise = [
        Transition((1, 2, 3), (9, 4, 7)),
        Transition((4, 5, 6), (2, 8, 1)),
        Transition((7, 8, 9), (3, 3, 3)),
    ]
    taught = RelationLanguage(counts={"mirror": 9, "offset": 4})
    assert observations_needed(noise, language=taught) is None


def test_the_curve_is_reported_rather_than_asserted(battery) -> None:
    """The raw sequence, so a caller measures the higher-order claim itself."""

    curve = list(learning_curve(battery[:30]))
    assert len(curve) == 30
    assert all(isinstance(index, int) and isinstance(got, bool) for index, got in curve)
    assert any(got for _index, got in curve)


# ------------------------------------------------------------- what contributes


def test_the_problems_are_fixed_by_a_fingerprint(battery) -> None:
    """The score is only evidence while the problems are.

    Whoever owns the generator can raise the number by making the problems
    easier, and would not have to mean to: widening the solver's basis and
    widening its battery are two edits in the same file.
    """

    from core.cognition.induction_battery import battery_fingerprint

    recorded = json.loads(_FLOOR.read_text())
    assert battery_fingerprint() == recorded["fingerprint"], (
        "the battery changed, so the floor does not apply to it — record the "
        "new fingerprint deliberately rather than letting the number carry over"
    )


def test_each_part_is_worth_what_it_is_worth(battery) -> None:
    """The ablation, because a component that cannot change a score is not
    being measured by it.

    The first version of this table reported the same number for every
    ablation, because the flag was dropped before it reached the solver. The
    second reported no contribution from the learned library, because every
    problem showed one shape at two lengths — enough to pin it unaided — so the
    battery measured induction while being described as measuring transfer.
    """

    from core.cognition.induction_battery import teach_the_language
    from core.cognition.relation_language import RelationLanguage

    taught = RelationLanguage()
    assert teach_the_language(battery, language=taught) == 2

    whole = score_battery(battery, language=taught)
    no_composition = score_battery(
        battery, language=taught, without=frozenset({"composition"})
    )
    no_library = score_battery(
        battery, language=taught, without=frozenset({"known_forms"})
    )

    assert whole.solved - no_composition.solved >= 15, "composition earns its place"
    assert whole.solved - no_library.solved >= 20, "the learned library earns its place"


def test_the_deep_shapes_are_impossible_without_the_library(battery) -> None:
    """Not harder: impossible, however many observations are offered."""

    from core.cognition.induction_battery import (
        NEEDS_A_LEARNED_FORM,
        teach_the_language,
    )
    from core.cognition.relation_language import RelationLanguage

    taught = RelationLanguage()
    teach_the_language(battery, language=taught)
    deep = [item for item in battery if item.shape in NEEDS_A_LEARNED_FORM]
    assert len(deep) == 20

    with_library = score_battery(deep, language=taught)
    without_library = score_battery(
        deep, language=taught, without=frozenset({"known_forms"})
    )
    assert with_library.solved == 20
    assert without_library.solved == 0


def test_the_prior_contributes_nothing_here_and_that_is_reported(battery) -> None:
    """Said plainly rather than buried.

    Breaking ties between shapes that fit equally well has a measured effect in
    the transfer tests, and none on this battery, because these problems do not
    put the solver in front of a tie. A table that only showed the parts that
    helped would be worth less than one that shows this.
    """

    from core.cognition.induction_battery import teach_the_language
    from core.cognition.relation_language import RelationLanguage

    taught = RelationLanguage()
    teach_the_language(battery, language=taught)
    whole = score_battery(battery, language=taught)
    no_prior = score_battery(battery, language=taught, without=frozenset({"prior"}))
    assert no_prior.solved == whole.solved

    recorded = json.loads(_FLOOR.read_text())["ablations"]
    assert recorded["no_prior"]["solved"] == recorded["nothing_removed"]["solved"]


def test_the_missing_core_system_was_predicted_before_it_was_added(report) -> None:
    """Objecthood was the one core-knowledge system the basis omitted.

    The basis had order, symmetry and adjacency — geometry and number — and no
    way to say that some cells belong together and travel as a set. The
    prediction was made in advance that adding grouping would lift exactly the
    shape needing it and nothing else.

    It lifted nothing at first: the form laid the even class down first and
    could not say the other order. With that fixed it went 0/10 to 10/10 and no
    other shape moved. The prediction failing first, for a findable reason, is
    worth more than it working immediately.
    """

    solved, seen = report.by_shape["odd positions first"]
    assert (solved, seen) == (10, 10)
    for shape in (
        "mirror",
        "rotate by one",
        "exchange the ends",
        "identity",
        "mirror then rotate",
    ):
        got, of = report.by_shape[shape]
        assert got == of, f"{shape} moved: {got}/{of}"


def test_grouping_can_say_which_group_leads() -> None:
    """A grouping with no say in which group leads is half a grouping."""

    from core.cognition.primitive_invention import _grouped_source

    six = 6
    assert [_grouped_source(i, six, 2, 0) for i in range(six)] == [0, 2, 4, 1, 3, 5]
    assert [_grouped_source(i, six, 2, 1) for i in range(six)] == [1, 3, 5, 0, 2, 4]
    # A span of one is not a grouping.
    assert [_grouped_source(i, six, 1) for i in range(six)] == list(range(six))
