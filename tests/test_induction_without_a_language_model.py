"""A number for what is left when the foundation model is taken away.

The standing criticism is that removing the resident model makes what remains
sharply narrower. That is a claim about a number and there was no number. This
is the number, on a frozen battery, with nothing in the path that consults a
model, an embedding or a stored answer.

The battery is built to be failable. Two of its ten shapes are outside anything
the mechanism can express, and they are named as such, because a battery a
mechanism cannot fail measures nothing about it. The score on shapes it can
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
    return score_battery(battery)


def test_the_battery_is_frozen_and_failable(battery) -> None:
    """Same problems before and after any change, and some are unreachable."""

    again = generate_battery()
    assert [item.name for item in battery] == [item.name for item in again]
    assert len(battery) == 100
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
        f"{report.line()} on the frozen battery, down from {floor['solved']}/100."
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
    assert min(scores.values()) >= 0.7, scores
    # And no representation is more than one problem away from the best.
    spread = max(scores.values()) - min(scores.values())
    assert spread <= 0.2, scores


def test_a_shape_beyond_the_language_is_reported_as_that(report) -> None:
    for shape in BEYOND_THE_LANGUAGE:
        solved, seen = report.by_shape[shape]
        assert seen == 10
        assert solved <= 2, f"{shape} should be out of reach, scored {solved}/{seen}"
    assert report.attempted_expressible == 80


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
