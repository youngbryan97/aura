"""Not "a prior helped" — which prior, established by removing it.

An external review set the bar for the transfer claim precisely:

    The stronger result would show that Aura created internal
    abstractions/operators/strategies during D_i, and intervention on those
    newly developed structures changes performance on D_i'.

Gate 4 already compared P(B | A) with P(B | ∅), which says a prior helped and
not what the prior was. The intervention arm removes the one structure she
built while learning A, leaves everything else, and measures B again.

Measured against the live state root, 40 pairs, seed from the freeze:

    P(B | A)                                     1.0000
    P(B | A without the structure built in A)    0.5714
    P(B | ∅)                                     0.5714
    carried by what she built                    0.4286

Removing it takes transfer back to exactly the from-scratch level, so all of
the transfer runs through that structure. On the controls — same surface,
different structure — removing it changes nothing, which is what says she is
not matching appearances.

Under the test harness's own state root the magnitude is smaller — 0.16 at
19 usable pairs, with 16 outside the language — because she knows less there.
That is a real property of the measurement and not noise, so what is asserted
here is the direction and the shape of the result rather than the number: a
threshold tuned to one environment would be a number about the environment.
"""
from __future__ import annotations

import pytest

from tools.agi_gauntlet.protocol import Freeze
from tools.agi_gauntlet.runnable import _without, transfer


@pytest.fixture(scope="module")
def measured():
    freeze = Freeze(
        commit="a fixed commit", dirty=False, source_digest="d",
        weights="w", config="c",
    )
    return transfer(freeze, {"pairs": 40})


# ------------------------------------------------------------ the numbers


def test_the_gate_reports_an_intervention_and_not_only_a_comparison(measured):
    found = measured["intervention"]
    assert set(found) >= {
        "with_what_she_built",
        "with_it_removed",
        "from_scratch",
        "carried_by_what_she_built",
        "controls",
    }


def test_removing_what_she_built_costs_her_the_transfer(measured):
    """The claim. Not that a prior helped — that this one did."""
    found = measured["intervention"]
    assert found["with_what_she_built"] > found["with_it_removed"]
    assert found["carried_by_what_she_built"] > 0.0


def test_removing_it_lands_at_the_from_scratch_level(measured):
    """All of the transfer runs through it, rather than some of it."""
    found = measured["intervention"]
    assert found["with_it_removed"] == pytest.approx(found["from_scratch"], abs=0.05)


def test_it_changes_nothing_on_the_controls(measured):
    """Same surface, different structure. This is what rules out appearances."""
    controls = measured["intervention"]["controls"]
    assert controls["with_what_she_built"] == pytest.approx(
        controls["with_it_removed"], abs=0.05
    )


def test_the_result_says_what_it_shows(measured):
    assert "says what" in measured["intervention"]["what_this_shows"]


# ------------------------------------------------------------ the lesion


def test_the_lesion_removes_exactly_one_structure():
    from core.cognition.relation_language import RelationLanguage

    class ABuiltThing:
        family = "the one she built"
        form = "its form"

    language = RelationLanguage()
    language.counts.update({"the one she built": 3, "something else": 2})
    language.forms.update({"its form": ("x",), "another form": ("y",)})

    lesioned = _without(language, ABuiltThing())
    assert "the one she built" not in lesioned.counts
    assert lesioned.counts["something else"] == 2
    assert "its form" not in lesioned.forms
    assert "another form" in lesioned.forms


def test_the_lesion_is_a_copy_and_leaves_the_original_alone():
    """Every pair has to be measured against the same thing it was taught."""
    from core.cognition.relation_language import RelationLanguage

    class ABuiltThing:
        family = "kept"
        form = "shape"

    language = RelationLanguage()
    language.counts.update({"kept": 1})
    language.forms.update({"shape": ("x",)})

    _without(language, ABuiltThing())
    assert language.counts == {"kept": 1}
    assert language.forms == {"shape": ("x",)}


def test_a_lesion_of_nothing_leaves_the_language_intact():
    """When the structure has no family, there is nothing to take away."""
    from core.cognition.relation_language import RelationLanguage

    language = RelationLanguage()
    language.counts.update({"a": 1, "b": 2})
    lesioned = _without(language, None)
    assert lesioned.counts == {"a": 1, "b": 2}


def test_the_arm_is_additive_and_does_not_decide_the_gate(measured):
    """The intervention reports; the gate's own threshold still decides it.

    An arm that could fail a gate on its own would be a second gate wearing
    the first one's name.
    """
    assert "intervention" in measured
    assert isinstance(measured["passed"], bool)
    assert measured["intervention"]["carried_by_what_she_built"] >= 0.0
