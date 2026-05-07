from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.state.aura_state import AuraState


def test_integrated_frame_warns_against_false_clarity():
    state = AuraState()
    state.cognition.current_objective = "explain how I feel"
    state.response_modifiers["unity_claim"] = "Something is not sitting right."
    phase = UnitaryResponsePhase(kernel=None)

    mapping = {
        "unity_state": SimpleNamespace(level="fragmented"),
        "unity_fragmentation_report": SimpleNamespace(
            top_causes=[("draft_conflict", 0.61, "conflicting drafts remain active")],
            safe_to_self_report=False,
        ),
        "unity_repair_plan": SimpleNamespace(steps=["preserve competing drafts and answer with qualified uncertainty"]),
        "coherence_report": SimpleNamespace(overall_coherence=0.42, tension_pressure=0.72),
        "phenomenal_now": None,
    }

    with patch("core.phases.response_generation_unitary.ServiceContainer.get", side_effect=lambda name, default=None: mapping.get(name, default)):
        frame = phase._build_integrated_coherence_frame(state)

    assert "Do not claim clarity" in frame
    assert "draft conflict" in frame
