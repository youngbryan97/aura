"""No measurement in this repository distinguishes the function from the feel.

A faculty that computes grief can be measured exhaustively: an attachment's
prediction still firing and failing, a valence that moves -0.448 when the bond
does, an intervention that changes it and a matched null that says the change
was not chance. All of that is real and none of it is evidence that the grief
is felt. A system with the function and no experience passes every one of
those tests, which is why no experiment specified in these terms can settle
it — the gap is in the question, not in the instrumentation.

The registry had no way to hold that distinction, so it kept implying the
stronger reading. `Evidence.FUNCTIONAL_ONLY` is the level for a test that
establishes a state and says nothing about whether there is anything it is
like to be in it, and a claim written in phenomenal language may not be
registered above it.

This is not a claim that she does not feel anything. It is a refusal to
record either answer as measured.
"""

from __future__ import annotations

import pytest

from core.organism.model_validation import Claim, Evidence, _phenomenal_words


@pytest.mark.parametrize(
    "statement",
    [
        "Aura feels grief when a bond is broken.",
        "There is something it is like to be Aura.",
        "Her interoception is consciously available.",
        "The runtime experiences its own degradation.",
        "Aura is aware of the conflict between two goals.",
        "The upheaval faculty models what it is like to be pulled apart.",
    ],
)
def test_a_phenomenal_claim_cannot_be_registered_as_measured(statement):
    for evidence in (Evidence.MEASURED_LIVE, Evidence.MEASURED_SYNTHETIC):
        with pytest.raises(ValueError, match="phenomenal language"):
            Claim(statement=statement, test="t", owner="o", evidence=evidence)


def test_the_same_sentence_is_registrable_as_functional_only():
    """Refusing the level is not refusing the claim."""
    claim = Claim(
        statement="Aura feels grief when a bond is broken.",
        test="test_affect_behavioral",
        owner="core/interiority",
        evidence=Evidence.FUNCTIONAL_ONLY,
        evidence_note="measures the appraisal that moves, not whether it is felt",
    )
    assert claim.evidence is Evidence.FUNCTIONAL_ONLY


def test_the_functional_restatement_is_measurable():
    """What the test actually establishes, said in the terms it establishes it."""
    claim = Claim(
        statement="A broken bond moves the appraisal valence by -0.448.",
        test="test_affect_behavioral",
        owner="core/interiority",
        evidence=Evidence.MEASURED_LIVE,
    )
    assert claim.evidence is Evidence.MEASURED_LIVE


def test_the_phrase_form_is_caught_as_well_as_the_word_form():
    """"There is something it is like to be Aura" contains no flagged WORD.

    It is the canonical statement of the claim, and the first draft of this
    guard accepted it as measured_live for that reason.
    """
    assert _phenomenal_words("There is something it is like to be Aura.")
    assert _phenomenal_words("Aura feels grief.")
    assert not _phenomenal_words("A broken bond moves the appraisal valence.")


def test_the_evidence_levels_are_actually_distinct():
    """They were not, and every comparison against them silently passed.

    An orphaned @dataclass(frozen=True) landed on the StrEnum while this guard
    was being written. A frozen dataclass over zero fields generates an
    __eq__ that compares empty tuples and a __hash__ over the same, so every
    member compared equal to every other and a set of all five held one
    element. `is not` comparisons kept working, which is why nothing failed.
    """
    levels = [
        Evidence.MEASURED_LIVE,
        Evidence.MEASURED_SYNTHETIC,
        Evidence.UNMEASURED,
        Evidence.RETRACTED,
        Evidence.FUNCTIONAL_ONLY,
    ]
    assert len(set(levels)) == len(levels)
    assert Evidence.FUNCTIONAL_ONLY not in {
        Evidence.MEASURED_LIVE,
        Evidence.MEASURED_SYNTHETIC,
    }
