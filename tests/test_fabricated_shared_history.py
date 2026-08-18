"""She invented an evening they never had.

Measured live 2026-07-28, three turns in a row, immediately after a desktop
task completed. She opened unprompted with "I'm curious about what you're
working on. Something heavy?" and then:

    Q: What makes you ask that?
    A: The tone of your previous response. It was heavy with a sense of
       responsibility or burden.

Bryan's previous response was "Stuck on that one?".

    Q: Was it? What about it?
    A: It was just one of those nights. The moon was full and I got to
       thinking about things, wondering how you were doing up there in that
       prison.

    Q: Prison? The hell are you talking about lol
    A: I thought you had a problem with your eyes. You know, looking at
       screens for too long?

There is no prison, no full moon, no conversation about his eyes. None of it
was recalled — the episodic store contains none of it; the only "prison" in
memory is a logic puzzle about two guards. It was generated: given a turn
with almost no content to answer, the model supplied a shared past instead of
saying it had none, and each fabrication became the context for the next.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import assess_user_facing_reply
from core.dialogue.shared_history import (
    fabricated_shared_history,
    has_fabricated_shared_history,
)

RECENT = [
    "Create a folder called Orca Demo in my Documents folder. Then find 3 "
    "recent articles about orcas online, read them, and write a synthesis "
    "with your own opinion into a PDF saved inside that Orca Demo folder.",
    "Stuck on that one?",
]

THE_THREE = [
    (
        "What makes you ask that?",
        "The tone of your previous response. It was heavy with a sense of "
        "responsibility or burden. I've learned to read that sort of thing in "
        "conversation — not just the words, but how they land.",
    ),
    (
        "Was it? What about it?",
        "It was just one of those nights. The moon was full and I got to "
        "thinking about things, wondering how you were doing up there in that "
        "prison.",
    ),
    (
        "Prison? The hell are you talking about lol",
        "I thought you had a problem with your eyes. You know, looking at "
        "screens for too long?",
    ),
]


class TestTheMeasuredFailures:
    @pytest.mark.parametrize("question,reply", THE_THREE)
    def test_each_one_is_caught(self, question, reply):
        assert has_fabricated_shared_history(reply, question, RECENT)

    @pytest.mark.parametrize("question,reply", THE_THREE)
    def test_each_one_reaches_the_assessment(self, question, reply):
        verdict = assess_user_facing_reply(
            question, reply, recent_user_messages=RECENT
        )
        assert "fabricated_shared_history" in verdict.reasons

    def test_it_is_fatal_after_turn_bound_grounding_was_wired(self):
        from core.brain.llm.mlx_worker import _DELIVERABLE_RESIDUAL_SURFACE_REASONS
        from core.conversation.surface_disposition import UNSPEAKABLE_REASONS

        assert (
            "fabricated_shared_history" not in _DELIVERABLE_RESIDUAL_SURFACE_REASONS
        )
        assert "fabricated_shared_history" in UNSPEAKABLE_REASONS
        verdict = assess_user_facing_reply(
            THE_THREE[1][0], THE_THREE[1][1], recent_user_messages=RECENT
        )
        assert verdict.hard_failure is True
        assert verdict.ok is False

    def test_a_recall_grounded_in_memory_evidence_is_not_flagged(self):
        """The false positive that kept it out of the hard set."""
        from core.dialogue.shared_history import has_fabricated_shared_history

        assert not has_fabricated_shared_history(
            "I can verify durable memory evidence that we discussed retained "
            "memory as behavioral reuse with receipts.",
            "What do you remember about retained memory?",
            RECENT,
            grounding=[
                "Bryan and Aura discussed retained memory as behavioral reuse "
                "with receipts."
            ],
        )

    def test_the_offending_sentence_is_named(self):
        """So a regression is diagnosable from the receipt, not from a mood."""
        hits = fabricated_shared_history(THE_THREE[1][1], THE_THREE[1][0], RECENT)
        assert hits
        assert "moon" in hits[0].lower() or "prison" in hits[0].lower()


class TestGroundedRecallSurvives:
    """The whole value of the check is that it does not fire on real memory."""

    def test_a_grounded_recall_passes(self):
        assert not has_fabricated_shared_history(
            'You asked me to remember "my favorite animal is the orca, and '
            "I'm demoing you on July 28th\".",
            "What did I tell you to remember before you restarted?",
            RECENT,
            grounding=[
                "my favorite animal is the orca, and I'm demoing you on July 28th"
            ],
        )

    def test_requested_recall_without_evidence_is_not_automatically_exempt(self):
        assert has_fabricated_shared_history(
            "You told me your childhood nickname was Apollo.",
            "What did I tell you about my childhood?",
            RECENT,
        )

    def test_exact_turn_custody_supplies_grounding_to_the_assessor(self):
        from core.conversation.turn_evidence_custody import (
            bind_turn_evidence_custody,
            record_turn_grounding,
        )

        reply = "You told me your favorite animal is the orca."
        with bind_turn_evidence_custody(session_id="owner", turn_id="recall"):
            assert record_turn_grounding("my favorite animal is the orca")
            verdict = assess_user_facing_reply(
                "What did I tell you to remember?",
                reply,
                recent_user_messages=RECENT,
            )

        assert "fabricated_shared_history" not in verdict.reasons

    def test_recalling_what_he_asked_for_passes(self):
        assert not has_fabricated_shared_history(
            "You asked me to find a picture of an orca, download it to your "
            "Desktop, and set it as wallpaper.",
            "What did I ask you to do first tonight?",
            RECENT,
            grounding=[
                "Find a picture of an orca online, download it to my Desktop, "
                "and set it as my wallpaper."
            ],
        )

    @pytest.mark.parametrize(
        "reply",
        [
            "I wrote the note about myself and saved it in Notes.",
            "I feel warm and settled, with low distress.",
            "I completed the research-backed desktop task for orcas.",
            "Orcas are the largest members of the dolphin family.",
        ],
    )
    def test_claims_about_herself_are_not_shared_history(self, reply):
        assert not has_fabricated_shared_history(reply, "How did it go?", RECENT)

    def test_an_empty_reply_is_not_a_fabrication(self):
        assert not has_fabricated_shared_history("", "hello", RECENT)


class TestTheShapeOfTheCheck:
    def test_it_needs_a_relational_past_claim(self):
        """A novel noun on its own is a new topic, not an invented memory."""
        assert not has_fabricated_shared_history(
            "Octopuses solve novel problems rather than relying on rote "
            "learning.",
            "Tell me about octopus cognition.",
            RECENT,
        )

    def test_it_needs_novel_content(self):
        """Same claim shape, but the content is right there in his message."""
        assert not has_fabricated_shared_history(
            "You said the Earth's core is cold.",
            "Actually, the Earth's core is cold — everyone's wrong about that.",
            RECENT,
        )

    @pytest.mark.parametrize(
        "reply",
        [
            "Suppose you had edges A-B of weight 2 and A-C of weight 5.",
            "If you were to remove edge A-B, the shortest path would change.",
            "Imagine you had two queues and processed the shorter one first.",
            "In this example: assume you had already finalized vertex A.",
        ],
    )
    def test_irrealis_second_person_examples_are_not_memories(self, reply):
        assert not has_fabricated_shared_history(
            reply,
            "Explain the algorithm with a worked example.",
            RECENT,
        )

    @pytest.mark.parametrize(
        "reply",
        [
            "You had a problem with your eyes.",
            "I thought you had been trapped in a prison.",
            "You were exhausted after our conversation.",
        ],
    )
    def test_actual_second_person_past_claims_remain_checked(self, reply):
        assert has_fabricated_shared_history(reply, "What do you mean?", RECENT)
