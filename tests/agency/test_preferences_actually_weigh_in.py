"""Her stated preferences have to be able to touch a choice.

LIVE 2026-08-17, from the neural stream:

    [SubjectiveChoice] Chose 'Deconstruct and comprehensively research: Aura is
    idle' because preference alignment 0.00 and drive alignment 0.49 produced
    final score 0.27.

Two separate defects behind that one line.

The 0.00 was not a judgement that the option suited her poorly. The feature
inference matched NOTHING: "research" appeared in none of the ten keyword
lists, and research is her most common autonomous act. Her strongest stated
preferences — truth 0.94, coherence 0.88, care 0.86 — had no route to the
decision, so they were decoration.

And reporting an absent reading as "0.00" makes it look measured. It hands the
whole decision to drive while appearing to have weighed both.
"""

from __future__ import annotations

import pytest

from core.agency.subjective_choice import (
    ChoiceOption,
    SubjectiveChoiceEngine,
    infer_preference_features,
)


@pytest.fixture()
def engine() -> SubjectiveChoiceEngine:
    return SubjectiveChoiceEngine()


# ── the vocabulary she actually generates ────────────────────────────────────

@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Deconstruct and comprehensively research the topic", "novelty"),
        ("investigate why the router drops the lane", "novelty"),
        ("verify the receipt against the ledger", "truth"),
        ("repair the failing subsystem", "care"),
        ("reconcile the two conflicting states", "coherence"),
        ("reply to Bryan about the failing test", "connection"),
        ("polish the interface until it feels right", "beauty"),
    ],
)
def test_her_own_action_vocabulary_is_recognised(text: str, key: str) -> None:
    features = infer_preference_features(text, None)

    assert features.get(key, 0.0) > 0.0, features


def test_the_live_goal_now_reads_a_preference(engine) -> None:
    """It scored exactly 0.00 before; research is a novelty-shaped act."""
    affinity = engine.preference_affinity(
        "Deconstruct and comprehensively research: Cognitive Neuroscience of Agency"
    )

    assert affinity > 0.0


# ── preference has to be able to WIN ────────────────────────────────────────

def test_a_preferred_option_can_override_a_higher_drive_one(engine) -> None:
    """If preference can never change the outcome it is not a preference."""
    receipt = engine.choose(
        [
            ChoiceOption(id="a", label="xyzzy plugh", description="", drive_score=0.5),
            ChoiceOption(
                id="b",
                label="research the neuroscience of agency",
                description="",
                drive_score=0.4,
            ),
        ],
        context="test",
        record=False,
    )

    assert receipt.chosen_id == "b"


# ── an absent reading is not a zero ─────────────────────────────────────────

def test_an_unreadable_option_says_so_instead_of_reporting_zero(engine) -> None:
    receipt = engine.choose(
        [ChoiceOption(id="a", label="xyzzy plugh", description="", drive_score=0.5)],
        context="test",
        record=False,
    )

    assert "0.00" not in receipt.rationale
    assert "absent reading" in receipt.rationale


def test_a_readable_option_still_reports_its_alignment(engine) -> None:
    receipt = engine.choose(
        [
            ChoiceOption(
                id="a",
                label="research the neuroscience of agency",
                description="",
                drive_score=0.5,
            )
        ],
        context="test",
        record=False,
    )

    assert "preference alignment" in receipt.rationale
