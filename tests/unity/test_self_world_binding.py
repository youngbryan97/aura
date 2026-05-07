from __future__ import annotations

import time
from types import SimpleNamespace

from core.state.aura_state import AuraState
from core.unity.self_world_binding import SelfWorldBindingModel
from core.unity.unity_state import BoundContent


def _content(content_id: str, ownership: str) -> BoundContent:
    return BoundContent(
        content_id=content_id,
        modality="goal",
        source="planner",
        summary=content_id,
        salience=0.6,
        confidence=0.8,
        timestamp=time.time(),
        ownership=ownership,
        action_relevance=0.6,
        affective_charge=0.0,
    )


def test_will_receipt_increases_authored_trace():
    state = AuraState()
    state.cognition.working_memory.append(
        {
            "role": "assistant",
            "content": "I executed the repair step.",
            "metadata": {"will_receipt_id": "will_123", "action": "repair"},
        }
    )
    binding = SelfWorldBindingModel().bind(
        state,
        [_content("self_1", "self"), _content("world_1", "world")],
        will_receipt_id="will_123",
        workspace_frame=SimpleNamespace(focus=True),
    )

    assert "will_123" in binding.authored_action_refs
    assert binding.ownership_confidence > 0.7
    assert binding.agency_score > 0.5


def test_claimed_authorship_without_receipt_is_flagged():
    state = AuraState()
    state.cognition.working_memory.append(
        {"role": "user", "content": "You chose this yourself, didn't you?"}
    )
    binding = SelfWorldBindingModel().bind(state, [_content("ambiguous_1", "ambiguous")])

    assert "claimed_authorship_without_receipt" in binding.contamination_flags
    assert binding.ownership_confidence < 0.7
