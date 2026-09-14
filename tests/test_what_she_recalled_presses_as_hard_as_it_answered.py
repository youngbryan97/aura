"""Memory reached deliberation as a switch.

When a recollection outranked every footing, the growth intention named it,
and its urgency was still scaled by how badly the moment was going. A strong
recollection and a barely-winning one pressed exactly as hard as each other,
and a displacement of active memory reached deliberation only by crossing the
line. The reading that chose the focus is now what presses: a footing's value
when a footing wins, which is the old number, and the recollection's match
score when a recollection does.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


def _phase() -> MotivationUpdatePhase:
    return MotivationUpdatePhase(SimpleNamespace(organs={}))


def _unmet_growth(state: AuraState) -> float:
    budget = state.motivation.budgets["growth"]
    capacity = max(1e-6, float(budget.get("capacity", 100.0)))
    return max(0.0, min(1.0, (capacity - float(budget["level"])) / capacity))


def _worst_footing(state: AuraState) -> float:
    return max(MotivationUpdatePhase._footing(state).values())


def _recalled(score: float) -> AuraState:
    state = AuraState.default()
    state.cognition.long_term_memory = ["a note that barely matched", "the thing that answers it"]
    state.cognition.memory_scores = [0.01, score]
    return state


def test_a_stronger_recollection_presses_harder() -> None:
    worst = _worst_footing(AuraState.default())
    weaker, stronger = min(1.0, worst + 0.2), min(1.0, worst + 0.45)
    assert stronger > weaker > worst
    phase = _phase()
    low, high = phase._assess_needs(_recalled(weaker)), phase._assess_needs(_recalled(stronger))
    assert "the thing that answers it" in low["goal"] and "the thing that answers it" in high["goal"]
    assert high["urgency"] > low["urgency"]


def test_when_a_footing_wins_the_urgency_is_what_it_was() -> None:
    state = AuraState.default()
    expected = round(0.6 * max(0.0, min(1.0, _unmet_growth(state) * (1.0 + min(1.0, _worst_footing(state))))), 4)
    assert _phase()._assess_needs(state)["urgency"] == expected


def test_a_displacement_of_memory_moves_deliberation_by_degree() -> None:
    from core.subject.state import perturb

    phase = _phase()
    worst = _worst_footing(AuraState.default())
    urgencies = []
    for delta in (worst - 0.5 + 0.2, worst - 0.5 + 0.4):
        state = AuraState.default()
        assert perturb(state, "M", delta) is True
        intention = phase._assess_needs(state)
        assert "probe recollection" in intention["goal"]
        urgencies.append(intention["urgency"])
    assert urgencies[1] > urgencies[0]
