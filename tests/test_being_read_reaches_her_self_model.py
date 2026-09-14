"""Being read by somebody else reaches her self-model, her state and the core.

The ledger's own tests pin the comparison. These pin that it arrives: the
conversation phase records the claim and the warmth, the self prediction loop
scores the claim against the same outcome and lets it cap her confidence, and
the self-state columns that read it are held by the lesion clamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.consciousness.self_prediction import SelfPredictionLoop
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.self import recognition
from core.state.aura_state import AuraState
from core.state.percepts import PERCEPT_EMOTIONS
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


@pytest.fixture(autouse=True)
def _fresh_ledger():
    recognition.reset_for_test()
    yield
    recognition.reset_for_test()


def test_the_conversation_phase_records_what_was_said_about_her() -> None:
    state = AuraState.default()
    ConversationalDynamicsPhase._read_recognition(state, "you seem exhausted tonight")
    assert "pairs" in state.identity.read_by_other
    assert state.identity.read_by_other["cared_for"] == 0.0
    # The claim is held until a moment settles it.
    recognition.get_recognition_ledger().settle(-0.6)
    assert recognition.get_recognition_ledger().reading().pairs == 1


def test_a_message_that_says_nothing_about_her_leaves_nothing_to_score() -> None:
    state = AuraState.default()
    ConversationalDynamicsPhase._read_recognition(state, "the build is failing again")
    recognition.get_recognition_ledger().settle(-0.6)
    assert recognition.get_recognition_ledger().reading().pairs == 0


def test_warmth_about_her_is_recorded_and_felt() -> None:
    state = AuraState.default()
    state.response_modifiers["user_sentiment"] = {"warmth": 0.9}
    state.response_modifiers["register"] = {"second": 0.6}
    ConversationalDynamicsPhase._read_recognition(state, "you have been a real help to me")
    assert state.identity.read_by_other["cared_for"] == pytest.approx(0.54)
    assert "cared_for" in str(state.world.recent_percepts)


def test_the_emotions_being_cared_about_carries_are_ones_she_has() -> None:
    assert set(PERCEPT_EMOTIONS["cared_for"]) <= set(AuraState.default().affect.emotions)


def test_a_better_reader_lowers_her_confidence_in_her_own_reading() -> None:
    loop = SelfPredictionLoop(SimpleNamespace())
    loop._smoothed_error = 0.0
    alone = loop._predict_next().confidence

    ledger = recognition.get_recognition_ledger()
    for _ in range(recognition.MIN_PAIRS):
        ledger.claim(0.5, -0.5)
        ledger.settle(0.5)
    assert ledger.reading().borrowed

    read_better = loop._predict_next().confidence
    assert read_better < alone
    assert read_better >= 0.1


def test_when_she_reads_herself_better_her_confidence_is_untouched() -> None:
    loop = SelfPredictionLoop(SimpleNamespace())
    loop._smoothed_error = 0.2
    alone = loop._predict_next().confidence

    ledger = recognition.get_recognition_ledger()
    for _ in range(recognition.MIN_PAIRS):
        ledger.claim(-0.9, 0.5)
        ledger.settle(0.5)
    assert not ledger.reading().borrowed
    assert loop._predict_next().confidence == pytest.approx(alone)


def test_the_moment_that_settles_a_claim_is_the_one_that_scores_it() -> None:
    """The loop's own error computation is what settles it."""
    loop = SelfPredictionLoop(SimpleNamespace())
    prediction = loop._predict_next()
    recognition.get_recognition_ledger().claim(0.8, prediction.predicted_affect_valence)
    loop._compute_error(prediction, 0.8, "curiosity", "drive_curiosity")
    assert recognition.get_recognition_ledger().reading().pairs == 1


def test_the_columns_that_read_it_are_held_by_the_clamp() -> None:
    sources = set(_SCHEMAS["S"].sources)
    for field in (
        "identity.read_by_other.borrowed",
        "identity.read_by_other.her_error",
        "identity.read_by_other.cared_for",
    ):
        assert field in sources, field
    assert "identity.read_by_other" in CLAMPED_FIELDS["S"]
