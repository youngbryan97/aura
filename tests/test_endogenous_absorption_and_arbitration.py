"""The second arrow, and the veto.

Absorption is what stops a conclusion evaporating when the response is
emitted. Arbitration is what stops a generation counting as her decision
because it came back from the model. Both are only worth anything if they are
driven by the named state channels, so both are tested by moving those
channels and requiring the outcome to follow.
"""

from __future__ import annotations

from core.brain.llm.endogenous_absorption import (
    ABSORPTION_WEIGHT,
    MAX_INJECTION_ENERGY,
    Proposal,
    TurnOutcome,
    absorb,
    arbitrate,
    outcome_from_response,
)
from core.brain.llm.endogenous_state import empty_state


class _AdditiveSubstrate:
    def __init__(self) -> None:
        self.observation = None
        self.weight = None

    def blend_observation(self, observation, weight):
        self.observation = observation
        self.weight = weight

    def inject_observation(self, observation):  # pragma: no cover - must not run
        raise AssertionError("the replacing bus must not be used")


class _ReplacingSubstrate:
    def inject_observation(self, observation):  # pragma: no cover - must not run
        raise AssertionError("the replacing bus must not be used")


class _MuteSubstrate:
    pass


def test_absorption_uses_the_additive_bus():
    organ = _AdditiveSubstrate()
    receipt = absorb(TurnOutcome(summary="a conclusion", evidence_items=2), substrate=organ)
    assert receipt.accepted is True
    assert organ.weight == ABSORPTION_WEIGHT
    assert organ.observation["source"] == "endogenous_absorption"


def test_absorption_refuses_a_bus_that_would_erase_other_writers():
    receipt = absorb(TurnOutcome(summary="x"), substrate=_ReplacingSubstrate())
    assert receipt.accepted is False
    assert "erase" in receipt.reason


def test_the_exclusive_bus_is_available_when_asked_for():
    class OnlyReplacing:
        def __init__(self):
            self.got = None

        def inject_observation(self, observation):
            self.got = observation

    organ = OnlyReplacing()
    receipt = absorb(TurnOutcome(summary="x"), substrate=organ, allow_exclusive_bus=True)
    assert receipt.accepted is True
    assert organ.got is not None


def test_an_organ_with_no_bus_is_refused():
    receipt = absorb(TurnOutcome(summary="x"), substrate=_MuteSubstrate())
    assert receipt.accepted is False
    assert "no observation input bus" in receipt.reason


def test_injection_energy_is_bounded():
    loud = TurnOutcome(
        summary="x",
        evidence_items=99,
        goal_advanced=True,
        contradiction_found=True,
        refused=True,
    )
    assert loud.as_observation()["energy"] <= MAX_INJECTION_ENERGY


def test_delivery_and_movement_are_reported_separately():
    """A substrate integrating on its own clock has not failed to receive."""
    receipt = absorb(TurnOutcome(summary="x"), substrate=_AdditiveSubstrate())
    assert receipt.accepted is True
    assert receipt.state_moved is False


def _unsure():
    return empty_state().do(
        **{
            "uncertainty.confidence": 0.1,
            "uncertainty.evidence_support": 0.1,
            "goal.active": 1.0,
            "goal.priority": 0.95,
        }
    )


def _settled():
    return empty_state().do(
        **{
            "uncertainty.confidence": 0.95,
            "uncertainty.evidence_support": 0.9,
            "goal.active": 0.0,
            "goal.priority": 0.0,
        }
    )


_PROPOSAL = Proposal(
    summary="commit and drop the goal",
    asserted_confidence=0.95,
    abandons_active_goal=True,
    requires_action=True,
)


def test_the_same_proposal_is_rejected_or_accepted_by_the_state_it_meets():
    assert arbitrate(_PROPOSAL, _unsure()).decision == "reject"
    assert arbitrate(_PROPOSAL, _settled()).decision == "accept"


def test_a_conflict_names_the_channel_it_came_from():
    conflicts = arbitrate(_PROPOSAL, _unsure()).conflicts
    channels = {c.channel for c in conflicts}
    assert {"uncertainty", "goal"} <= channels
    for conflict in conflicts:
        assert conflict.feature and conflict.detail


def test_an_absent_channel_is_skipped_and_never_counted_as_agreement():
    arbitration = arbitrate(_PROPOSAL, empty_state())
    assert arbitration.checks_run == ()
    assert set(arbitration.checks_skipped) >= {
        "certainty_matches_state",
        "respects_active_goal",
        "action_has_support",
        "recall_is_consistent",
    }
    assert arbitration.decision == "accept"


def test_a_recall_contradiction_downgrades_rather_than_rejects():
    state = empty_state().do(**{"memory.contradiction": 1.0})
    arbitration = arbitrate(Proposal(summary="conclude"), state)
    assert arbitration.decision == "revise"
    assert arbitration.conflicts[0].channel == "memory"


def test_a_response_frame_reads_as_an_outcome():
    outcome = outcome_from_response(
        {"text": "an answer", "status": "ok", "tokens_used": 12, "confidence": 0.4}
    )
    assert outcome.summary == "an answer"
    assert outcome.refused is False
    assert outcome.confidence == 0.4
    assert outcome_from_response({"status": "error", "text": "x"}).refused is True
    assert outcome_from_response(None).summary == ""
