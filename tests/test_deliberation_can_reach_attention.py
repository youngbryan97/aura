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
