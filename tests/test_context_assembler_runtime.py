from core.brain.llm.context_assembler import ContextAssembler
from core.state.aura_state import AuraState


def test_short_self_inquiry_is_not_treated_as_casual():
    assert ContextAssembler._is_casual_interaction("Do you feel anything?") is False
    assert ContextAssembler._is_casual_interaction("Is Aura conscious?") is False


def test_short_greeting_stays_casual():
    assert ContextAssembler._is_casual_interaction("hey") is True


def test_build_messages_updates_attention_focus():
    state = AuraState.default()
    state.cognition.attention_focus = None

    ContextAssembler.build_messages(state, "Let's debug the retrieval pipeline.")

    assert state.cognition.attention_focus == "Let's debug the retrieval pipeline."
