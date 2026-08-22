"""Asking about something named is asking for more than her memory.

LIVE, 2026-08-22, typed into the window: "what can you tell me about the
company Hugging Face? founders, what they sell, rough size. link your
sources." No search ran. The grounding taken was her own source code, the
cortex answered from memory in twenty-six seconds, and the reply read "It was
founded by <NAME> and <NAME>" with no citations — to a question that had asked
for them in as many words.

Asked as "can you LOOK UP Hugging Face", the same question searched. The whole
decision rested on pattern lists that three ordinary phrasings missed.
"""

from __future__ import annotations

import pytest

from core.conversation.asks_about_the_world import (
    asks_about_a_named_thing,
    asks_for_sources,
    wants_outside_evidence,
)


@pytest.mark.parametrize(
    "message",
    [
        "what can you tell me about the company Hugging Face? link your sources.",
        "who founded Hugging Face?",
        "tell me about Anthropic the company",
        "give me a rundown on Mistral AI",
        "who is the ceo of Stripe",
        "what's the background on Cerebras",
    ],
)
def test_a_factual_question_about_something_named_wants_evidence(message: str):
    assert wants_outside_evidence(message), message


@pytest.mark.parametrize(
    "message",
    [
        "how are you feeling today?",
        "tell me about yourself",
        "what is 7919 * 6367?",
        "read CONTRIBUTING.md and tell me the first rule",
        "Morning. How long have you been up?",
        "I made up a game. With perfect play, who wins?",
        "what do you think about consciousness?",
        "Tomorrow I want to try something new",
        "what have you been working on lately?",
    ],
)
def test_a_turn_about_her_or_this_machine_does_not(message: str):
    assert not wants_outside_evidence(message), message


def test_asking_for_sources_is_an_instruction_not_a_topic():
    """Somebody who asks for sources has asked for evidence; an answer with
    none is a broken promise rather than a style choice."""
    for asked in (
        "cite whatever you find",
        "link your sources",
        "with citations please",
        "show me where you got that",
    ):
        assert asks_for_sources(asked), asked
    assert not asks_for_sources("what time is it")


def test_a_name_is_recognised_by_shape_not_by_a_list():
    """So a company nobody has heard of works like one that ships in a list."""
    assert asks_about_a_named_thing("who founded Quibbleflax Dynamics?")
    assert asks_about_a_named_thing("tell me about Zorbtech")
    assert not asks_about_a_named_thing("who founded it?")


def test_the_contract_now_requires_a_search_for_those_turns():
    from core.state.aura_state import AuraState
    from core.phases.response_contract import build_response_contract

    state = AuraState.default()
    for message in (
        "who founded Hugging Face?",
        "tell me about Anthropic the company",
    ):
        contract = build_response_contract(state, message, is_user_facing=True)
        assert contract.requires_search is True, message
        assert contract.required_skill == "web_search", message

    for message in ("how are you feeling today?", "tell me about yourself"):
        contract = build_response_contract(state, message, is_user_facing=True)
        assert contract.requires_search is False, message


def test_the_older_reasons_still_name_themselves():
    """The new signal explains only what the old ones do not."""
    from core.state.aura_state import AuraState
    from core.phases.response_contract import build_response_contract

    state = AuraState.default()
    contract = build_response_contract(
        state,
        'Tell me who wrote "Beautiful Mind" and what the lyrics are about.',
        is_user_facing=True,
    )
    assert contract.reason == "specific_fact_lookup"


def test_a_placeholder_is_not_an_answer():
    """LIVE, 2026-08-22: "It was founded by <NAME> and <NAME>." A placeholder
    is the model saying it does not know, in a shape that reads like an
    answer."""
    from core.conversation.response_reliability import (
        contains_unfilled_placeholder,
        repair_runtime_boilerplate,
    )

    reply = (
        "Hugging Face is a technology company that specializes in NLP. "
        "It was founded by <NAME> and <NAME>."
    )
    assert contains_unfilled_placeholder(reply)
    repaired = repair_runtime_boilerplate(reply)
    assert "<NAME>" not in repaired
    assert "specializes in NLP" in repaired

    for shape in ("The release is set for [DATE].", "Your total is {{ amount }}."):
        assert contains_unfilled_placeholder(shape), shape
        assert repair_runtime_boilerplate(shape) == ""


def test_a_real_answer_is_left_alone():
    from core.conversation.response_reliability import (
        contains_unfilled_placeholder,
        repair_runtime_boilerplate,
    )

    for kept in (
        "Hugging Face was founded by Clement Delangue and Julien Chaumond in 2016.",
        "I compared <a and b> using the operator.",
        "The list is <ul> in HTML.",
    ):
        assert not contains_unfilled_placeholder(kept), kept
        assert repair_runtime_boilerplate(kept) == kept
