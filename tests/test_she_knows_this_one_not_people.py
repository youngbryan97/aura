"""Holding somebody as themselves rather than as an instance of people.

"Woman" names Nigeria, Jamaica, Ghana, London and what each one does. "Camera"
does it down a bus, one face at a time. Neither reaches for the category, and
the store already refuses the category at its input. What nothing measured is
the other side: whether what she ends up holding about a person is about that
person, or the same handful of things she holds about everyone.

Two readings, because they answer different halves and one of them needs a
second person before it is a sentence at all.
"""

from __future__ import annotations

from core.memory.interpersonal_model import (
    Facet,
    PersonModel,
    Provenance,
    Subject,
)
from core.social.particular import MIN_CLAIMS, read_particularity


def _person(name: str, claims: list[tuple[str, Provenance]]) -> PersonModel:
    model = PersonModel(name)
    for index, (claim, provenance) in enumerate(claims):
        model.observe(
            claim,
            episode_id=f"{name}-{index}",
            facet=Facet.PREFERENCE,
            subject=Subject.THEM,
            provenance=provenance,
        )
    return model


def test_an_empty_record_says_so() -> None:
    reading = read_particularity({})
    assert reading.measured is False
    assert reading.held_as_themselves is False


def test_too_few_claims_is_not_a_share() -> None:
    model = _person("bryan", [("likes short meetings", Provenance.STATED)])
    reading = read_particularity({"bryan": model})
    assert reading.measured is False
    assert str(MIN_CLAIMS) in reading.why


def test_one_person_can_be_read_for_anchoring_and_not_for_uniqueness() -> None:
    model = _person(
        "bryan",
        [(f"said thing {i}", Provenance.STATED) for i in range(MIN_CLAIMS + 1)],
    )
    reading = read_particularity({"bryan": model})
    assert reading.measured is True
    assert reading.unique_unreadable is True
    assert reading.unique == 0.0
    assert reading.anchored == 1.0
    assert "nobody for it to be unlike" in reading.why


def test_inference_alone_is_not_holding_them() -> None:
    model = _person(
        "bryan",
        [(f"probably thing {i}", Provenance.INFERRED) for i in range(MIN_CLAIMS + 1)],
    )
    reading = read_particularity({"bryan": model})
    assert reading.anchored == 0.0
    assert reading.held_as_themselves is False


def test_the_particular_one_stands_above_the_boilerplate_one() -> None:
    boilerplate = [(f"generic {i}", Provenance.STATED) for i in range(MIN_CLAIMS + 1)]
    theirs = _person(
        "bryan",
        boilerplate[:1] + [(f"only bryan {i}", Provenance.STATED) for i in range(4)],
    )
    everyone_else = _person("sam", boilerplate)
    another = _person("kit", boilerplate)

    reading = read_particularity(
        {"bryan": theirs, "sam": everyone_else, "kit": another}, person="bryan"
    )
    assert reading.measured is True
    assert reading.unique_unreadable is False
    assert reading.held_as_themselves is True
    assert reading.unique > 0.5
    assert reading.only_theirs

    other = read_particularity(
        {"bryan": theirs, "sam": everyone_else, "kit": another}, person="sam"
    )
    assert other.unique == 0.0
    assert other.held_as_themselves is False


def test_the_shared_share_counts_what_she_holds_about_everybody() -> None:
    same = [(f"generic {i}", Provenance.STATED) for i in range(MIN_CLAIMS + 1)]
    reading = read_particularity({"a": _person("a", same), "b": _person("b", same)})
    assert reading.shared == 1.0
    assert reading.unique == 0.0
