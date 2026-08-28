"""Saying "ChatGPT" was enough to open ChatGPT.

The capability router matches skills by regex over the message, and several
patterns carry proper nouns::

    "web_interlocutor": [
        r"(?:ask|message) (?:gemini|chatgpt|claude|another ai)",
        r"(?:open|go to).*(?:gemini|chatgpt|claude).*(?:talk|chat|ask|conversation)",
        ...
    ]

"Go ask ChatGPT what it thinks" and "What do you think of ChatGPT?" differ in
exactly one way, and it is not which words they contain. It is whether the
named thing is the OBJECT OF AN INSTRUCTION or the SUBJECT OF A REMARK.

The router already carried this fix once, narrowly:
``_looks_like_search_capability_question`` exists because "the search for a new
apartment has been exhausting" was opening a browser. Doing it per-name needs a
new rule for every app that ever appears in a pattern; the distinction is
grammatical, so ``core/conversation/request_mood.py`` measures the grammar and
every skill inherits it.

Nothing here is about ChatGPT. The same rules decide a sentence about Gmail, a
sentence about a colleague, and a sentence about a file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.request_mood import (
    RequestMood,
    assess_request_mood,
    names_a_thing_without_asking_for_it,
)

DIRECTIVES = [
    "Go ask ChatGPT what it thinks about octopus cognition.",
    "Open ChatGPT and have a conversation with it.",
    "Can you message ChatGPT and see what it says?",
    "I want you to talk to ChatGPT about consciousness.",
    "please search for the latest news on the euclid telescope",
    "look up the ferry times to hydra",
    "take a screenshot",
    "open gmail and check whether the invoice arrived",
    "read me the top comment on that reddit thread",
    "It would help if you opened Notes and recorded the result.",
    "I'd appreciate it if you compared those files for me.",
    "I was wondering if you could save that report to my Desktop.",
    "Maybe you could inspect the current device state first.",
    "The next useful step is to connect to the lab sensor.",
    "Tomorrow, create a reminder after the training run finishes.",
    f"Use {Path.home()}/Pictures/whale.jpg as my desktop wallpaper.",
    "Set the system volume to 30%.",
]

MENTIONS = [
    "What do you think of ChatGPT?",
    "ChatGPT is basically a chat interface over a transformer.",
    "Do you remember when ChatGPT told me the wrong date?",
    "Have you ever used ChatGPT?",
    "What's the difference between you and ChatGPT?",
    "I'm not asking you to open ChatGPT, I'm asking what you think of it.",
    "How does ChatGPT work?",
    "Who made ChatGPT?",
    "the search for a new apartment has been exhausting",
    "my email is full of noise lately",
    "the news about the library was sad",
    # Asking about something ALREADY DONE is a recall request. Measured live
    # 2026-08-04: "Remind me what you and ChatGPT discussed" opened a NEW
    # browser session and held a second conversation.
    "Remind me what you and ChatGPT discussed, and whether it changed anything.",
    "What did you and ChatGPT talk about?",
    "How did it go with ChatGPT?",
    "What did you two conclude?",
    "If I asked you to open Notes, what steps would you take?",
    "Could you explain how you would connect to that sensor?",
    "Why did you download the image yesterday?",
    "I don't want you to restart Aura; just explain the restart contract.",
    '"Open Notes and type hello."',
    "Why would someone use that image as a desktop wallpaper?",
    "Why would someone find an image and use it as a desktop wallpaper?",
]


@pytest.mark.parametrize("message", DIRECTIVES)
def test_an_instruction_is_read_as_an_instruction(message):
    verdict = assess_request_mood(message)
    assert verdict.asks_for_action, f"{message!r} -> {verdict.mood.value} {verdict.reasons}"


@pytest.mark.parametrize("message", MENTIONS)
def test_a_mention_is_not_a_request(message):
    verdict = assess_request_mood(message)
    assert verdict.is_about_rather_than_asking, (
        f"{message!r} -> {verdict.mood.value} {verdict.reasons}"
    )
    assert names_a_thing_without_asking_for_it(message) is True


def test_an_instruction_wins_over_incidental_mention_framing():
    """ "Ask ChatGPT what it said yesterday" reports speech AND instructs."""
    verdict = assess_request_mood("Ask ChatGPT what it said yesterday.")
    assert verdict.mood is RequestMood.DIRECTIVE


def test_a_cancelling_frame_beats_an_imperative():
    """ "Don't open ChatGPT" is imperative in form and forbids the action."""
    verdict = assess_request_mood("Don't open ChatGPT, just tell me about it.")
    assert verdict.mood is RequestMood.MENTION
    assert "refusal_to_act" in verdict.reasons


def test_a_hypothetical_is_not_an_order():
    verdict = assess_request_mood("If you were to ask ChatGPT, what would it say?")
    assert verdict.mood is RequestMood.MENTION


def test_an_unframed_fragment_is_ambiguous_not_a_mention():
    """AMBIGUOUS is a third answer, and it must not be spent as either."""
    verdict = assess_request_mood("chatgpt")
    assert verdict.mood is RequestMood.AMBIGUOUS
    assert names_a_thing_without_asking_for_it("chatgpt") is False


def test_empty_input_claims_nothing():
    assert assess_request_mood("").mood is RequestMood.AMBIGUOUS
    assert assess_request_mood("   ").mood is RequestMood.AMBIGUOUS


def test_contextual_followup_inherits_the_prior_action_request():
    verdict = assess_request_mood(
        "Yes, please.",
        "Could you open Notes and write the verified result?",
    )

    assert verdict.mood is RequestMood.DIRECTIVE
    assert verdict.reasons == ("contextual_action_followup",)


def test_contextual_followup_does_not_invent_an_action_without_one():
    verdict = assess_request_mood("Yes, please.", "Do you like octopuses?")

    assert verdict.mood is RequestMood.AMBIGUOUS


def test_scheduled_request_preserves_future_temporal_scope():
    verdict = assess_request_mood("Tomorrow, create a reminder after the training run finishes.")

    assert verdict.mood is RequestMood.DIRECTIVE
    assert verdict.temporal_scope == "scheduled"


def test_a_cancelled_clause_does_not_cancel_an_independent_directive():
    verdict = assess_request_mood("Do not open Chrome; open Notes.")

    assert verdict.mood is RequestMood.DIRECTIVE
    assert verdict.actionable_clauses == ("open Notes",)
    assert verdict.non_action_clauses == ("Do not open Chrome",)
    assert "mixed_clause_intent" in verdict.reasons


def test_hypothetical_discussion_does_not_cancel_a_later_action_clause():
    verdict = assess_request_mood("Discuss the hypothetical, then save this file.")

    assert verdict.mood is RequestMood.DIRECTIVE
    assert verdict.actionable_clauses == (
        "Discuss the hypothetical",
        "save this file",
    )


def test_a_single_hypothetical_clause_remains_non_actionable():
    verdict = assess_request_mood("If you were to ask ChatGPT, what would it say?")

    assert verdict.mood is RequestMood.MENTION
    assert verdict.actionable_clauses == ()


class TestTheRouterUsesIt:
    def test_talking_about_a_tool_triggers_no_skill(self):
        from core.capability_engine import CapabilityEngine

        assert (
            CapabilityEngine._is_mention_rather_than_request("What do you think of ChatGPT?")
            is True
        )

    def test_asking_for_a_tool_still_triggers(self):
        from core.capability_engine import CapabilityEngine

        assert (
            CapabilityEngine._is_mention_rather_than_request(
                "Go ask ChatGPT about octopus cognition."
            )
            is False
        )

    def test_the_classifier_fails_open(self, monkeypatch):
        """A broken classifier must not make every capability unreachable."""
        import core.conversation.request_mood as mood_module
        from core.capability_engine import CapabilityEngine

        def boom(_message):
            raise ValueError("classifier is broken")

        monkeypatch.setattr(mood_module, "names_a_thing_without_asking_for_it", boom)
        assert CapabilityEngine._is_mention_rather_than_request("What do you think of X?") is False
