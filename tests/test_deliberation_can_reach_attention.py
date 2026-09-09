"""Deliberation produced nothing per turn, and what it produced went nowhere.

Three breaks between a depleted drive and an intention competing for attention.

The intention generator picks the most depleted drive and dispatches on its
name. Five drives exist and it had branches for three; the one it had no branch
for — growth — starts lowest and decays, so it is the most depleted on almost
every tick of an ordinary life. The assessment returned None every time it ran.

The urgency it would have carried was a constant per drive, so a drive one
point below the line asked as loudly as one empty for a week, and nothing about
her state could change how hard anything pressed.

And the initiative that survives governance is written to
`cognition.pending_initiatives`, while the workspace's bid for deliberation read
`cognition.active_goals`. The one thing deliberation produces within a turn
never reached attention.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.consciousness.workspace_feed import build_candidates
from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


def _phase() -> MotivationUpdatePhase:
    return MotivationUpdatePhase(SimpleNamespace(organs={}))


def test_the_drive_that_is_always_most_depleted_has_an_arm() -> None:
    state = AuraState.default()
    levels = {name: budget["level"] for name, budget in state.motivation.budgets.items()}
    assert min(levels, key=lambda k: levels[k]) == "growth", levels
    intention = _phase()._assess_needs(state)
    assert intention is not None, "the intention generator still cannot fire"
    assert intention["drive"] == "growth"


def test_every_drive_the_state_carries_can_produce_an_intention() -> None:
    """A drive with no branch is a need that can never be acted on."""
    phase = _phase()
    for name in AuraState.default().motivation.budgets:
        state = AuraState.default()
        for other, budget in state.motivation.budgets.items():
            budget["level"] = 5.0 if other == name else 95.0
        intention = phase._assess_needs(state)
        if name == "energy":
            # Energy is the threshold's own input rather than a need with a
            # goal of its own; it has never had a branch and does not need one.
            continue
        assert intention is not None, f"{name} is depleted and asks for nothing"
        assert intention["drive"] == name


def test_urgency_rises_with_how_badly_the_moment_is_going() -> None:
    phase = _phase()
    calm = AuraState.default()
    calm.cognition.coherence_score = 1.0
    rough = AuraState.default()
    rough.cognition.coherence_score = 0.3
    assert phase._assess_needs(rough)["urgency"] > phase._assess_needs(calm)["urgency"]


def test_what_she_decides_to_work_on_is_read_off_what_is_worst() -> None:
    phase = _phase()
    incoherent = AuraState.default()
    incoherent.cognition.coherence_score = 0.2
    unstable = AuraState.default()
    unstable.identity.stability = 0.2
    assert phase._assess_needs(incoherent)["goal"] != phase._assess_needs(unstable)["goal"]


def test_an_initiative_can_compete_for_attention() -> None:
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {"goal": "work out why that answer was wrong", "urgency": 0.6}
    ]
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "deliberation"]
    assert bids, "an initiative still cannot reach the workspace"
    assert bids[0].priority == pytest.approx(0.6)


def test_the_urgency_gate_can_be_passed_by_a_real_situation() -> None:
    """Below 0.3 the will defers an initiative, and it used to sit at 0.14."""
    phase = _phase()
    rough = AuraState.default()
    rough.cognition.coherence_score = 0.3
    assert phase._assess_needs(rough)["urgency"] >= 0.3


def test_the_world_and_the_feelings_reach_what_she_decides() -> None:
    """A moment goes badly when the world surprises her or something hurts.

    Deliberation could see the argument coming apart and could not see either
    of the other two, so neither the world model nor affect could reach it.
    """
    phase = _phase()
    calm = AuraState.default()
    afraid = AuraState.default()
    afraid.affect.emotions["fear"] = 0.8
    assert phase._assess_needs(afraid)["urgency"] > phase._assess_needs(calm)["urgency"]

    # And which of the five readings is worst is what she names.
    incoherent = AuraState.default()
    incoherent.cognition.coherence_score = 0.1
    assert "coherence" in phase._assess_needs(incoherent)["goal"]
    assert "bothering" in phase._assess_needs(afraid)["goal"]


def test_a_strong_recollection_is_worth_working_on() -> None:
    """Retrieval scores how well each recollection matched; nothing read it."""
    phase = _phase()
    state = AuraState.default()
    state.cognition.long_term_memory = ["a weak note", "the thing that answers it"]
    state.cognition.memory_scores = [0.1, 0.9]
    assert "the thing that answers it" in phase._assess_needs(state)["goal"]

    state.cognition.memory_scores = [0.01, 0.02]
    assert "the thing that answers it" not in phase._assess_needs(state)["goal"]


def test_the_world_model_is_shown_what_arrived_not_only_how_much() -> None:
    from core.state.percepts import emit_percept
    from core.world_model.observe_cycle import observation_of

    quiet = AuraState.default()
    loud = AuraState.default()
    emit_percept(loud.world, "interaction", content="a message", intensity=0.9)
    assert not (observation_of(quiet) == observation_of(loud)).all()

    recalled = AuraState.default()
    recalled.cognition.long_term_memory = ["the answer"]
    recalled.cognition.memory_scores = [0.8]
    assert not (observation_of(quiet) == observation_of(recalled)).all()
