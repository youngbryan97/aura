"""Spare answer capacity does not turn a greeting into a derivation."""
import pytest

from core.brain.cognitive_engine_quick_reply import _AnswersTheDesktopDirectly
from core.brain.llm.chat_format import thinking_enabled_for_generation
from core.brain.llm.mlx_worker import _answer_is_derived_here
from core.runtime.structured_input import answer_surface_token_floor


@pytest.mark.parametrize("message", ["Hello.", "Hi there.", "Good evening.", "How are you?"])
def test_a_conversational_allowance_does_not_open_private_reasoning(message):
    floor, allowance = _AnswersTheDesktopDirectly._direct_desktop_quick_reply_part_2(
        [], False, {}, False, 512, False, False, False, answer_surface_token_floor(message))
    assert allowance >= 512
    job = {"user_surface_completion_floor": floor, "max_tokens": allowance}
    derived = _answer_is_derived_here(job)
    assert not derived
    assert thinking_enabled_for_generation("Aura-Qwen3.8-27B", cognitive_mode="fast",
                                            final_user_surface=True, answer_is_derived_here=derived) is False


def test_an_unresolved_derivation_keeps_its_computation():
    message = "45 becomes 15. 28 becomes 14. 66 becomes 22. What does 91 become, and why?"
    structural = answer_surface_token_floor(message)
    floor, allowance = _AnswersTheDesktopDirectly._direct_desktop_quick_reply_part_2(
        [], False, {}, False, 1024, False, False, True, structural)
    assert floor == structural
    assert _answer_is_derived_here({"user_surface_completion_floor": floor, "max_tokens": allowance})


def test_addressing_her_does_not_name_an_external_entity():
    from core.conversation.asks_about_the_world import _names, asks_about_a_named_thing

    assert _names("Hello Aura. What can you hear?") == []
    assert not asks_about_a_named_thing("Hello Aura. What can you hear?")
    assert _names("Hello Ada Lovelace. What did she build?") == ["Ada Lovelace"]
    assert asks_about_a_named_thing("Hello Microsoft. What did they release?")
