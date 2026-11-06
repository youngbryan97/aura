"""The two undeclared writes of the third day run (6 November 2026), measured under their phases' contracts.

AffectUpdatePhase rewrote cognition.long_term_memory 22 times, through memory_retrieval.settle_on_what_she_feels, and
each time her lineage refused the phase's receipt and its write of identity.evolution_score: all 22 of the day's steps
that did not develop. ConsciousnessPhase rewrote cognition.modifiers once, through self_conscious.monitoring.
"""
from __future__ import annotations

from core.runtime import cognitive_provenance as provenance
from core.state.aura_state import AuraState


def _measured(phase: str, state, act):
    transformation = provenance.begin_transformation(phase, state)
    act(state)
    return transformation.complete(state, publish_violation=False)


def test_affect_update_declares_the_recall_it_settles_on_her_feeling(monkeypatch):
    import core.memory.felt_at_encoding as felt
    import core.phases.affect_update  # noqa: F401  (registers the contract)
    from core.phases import memory_retrieval

    state = AuraState.default()
    state.cognition.long_term_memory = ["an afternoon spent on a pdf"]
    state.cognition.memory_scores = [0.4]
    state.cognition.recall_pool = {
        "salience": 1.0,
        "candidates": [
            {"text": "an afternoon spent on a pdf", "score": 0.4, "felt": {"v": 0.1}},
            {"text": "a post about a child who was heard", "score": 0.6, "felt": {"v": 0.9}},
        ],
    }
    monkeypatch.setattr(felt, "felt_now", lambda affect: {"v": 0.9})
    monkeypatch.setattr(felt, "usual", lambda stamps: {"v": 0.5})
    monkeypatch.setattr(felt, "resonance", lambda now, stamp, centre: 1.0 if stamp and stamp["v"] == 0.9 else 0.0)

    receipt = _measured("AffectUpdatePhase", state, memory_retrieval.settle_on_what_she_feels)
    assert "cognition.long_term_memory" in receipt.observed_writes
    assert receipt.undeclared_writes == ()


def test_consciousness_phase_declares_the_self_focus_it_puts_on_her_modifiers():
    import core.phases.consciousness_phase  # noqa: F401  (registers the contract)
    from core.consciousness.self_conscious import _MONITORED, monitoring

    state = AuraState.default()
    for name in _MONITORED:
        state.cognition.modifiers[name] = 0.5
    receipt = _measured("ConsciousnessPhase", state, lambda s: monitoring(s.cognition.modifiers, 0.5))
    assert "cognition.modifiers" in receipt.observed_writes
    assert receipt.undeclared_writes == ()
