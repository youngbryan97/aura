"""Ungrounded-person confabulation gate on the live conversation surface.

Live evidence (July 2026, desktop transcript): Aura opened a turn with
"Brenner usually had the good sense to stay away from me after his last
fiasco", asserted "Peter Brenner. Sometimes he and I work together, but
we're not friends", and later addressed the user as "Aaron, what's the
plan?" — a fictional social world served as fact, surviving even the user's
correction ("That isn't a person").

The gate must catch the two ONSET shapes (unprompted relational familiarity
with a named person; addressing the user by an ungrounded name) while
leaving grounded person-talk alone: names the user introduced, self/system
names, and registry-known people stay legal.
"""
from __future__ import annotations

from core.conversation.response_reliability import (
    _has_ungrounded_person_address,
    _has_ungrounded_person_narrative,
    assess_user_facing_reply,
)


class TestUngroundedPersonNarrative:
    def test_live_brenner_opening_is_caught(self):
        reply = (
            "Now that's a pleasant surprise. Brenner usually had the good "
            "sense to stay away from me after his last fiasco. What brings "
            "you here?"
        )
        assert _has_ungrounded_person_narrative("hey, how are you?", reply)

    def test_relational_claim_variants_are_caught(self):
        for reply in (
            "Marcus and I go way back, so I trust his read on this.",
            "My colleague Dana warned me about exactly this failure mode.",
            "Castellan told me the perimeter was already compromised.",
            "We teamed up with Rourke before the last audit.",
        ):
            assert _has_ungrounded_person_narrative("what should we do?", reply), reply

    def test_user_introduced_name_is_grounded(self):
        assert not _has_ungrounded_person_narrative(
            "My colleague Marcus thinks the deploy is safe — what do you think?",
            "Marcus and I see this differently: I'd stage it behind a flag first.",
        )

    def test_recent_messages_ground_names(self):
        assert not _has_ungrounded_person_narrative(
            "so what would you tell her?",
            "Dana asked me for the runtime numbers, so I'd give her the p95 first.",
            recent_user_messages=["Dana keeps asking about our latency."],
        )

    def test_plain_technical_reply_passes(self):
        assert not _has_ungrounded_person_narrative(
            "why is the build slow?",
            "Python usually has enough tooling for this; the bottleneck is the "
            "linker. I profiled the build and the cache never warms.",
        )

    def test_self_names_pass(self):
        assert not _has_ungrounded_person_narrative(
            "who worked on this?",
            "Claude and I traced it together — the fix landed this morning.",
        )


class TestUngroundedPersonAddress:
    def test_live_aaron_address_is_caught(self):
        assert _has_ungrounded_person_address(
            "That isn't a person",
            "Aaron, what's the plan? Anyone coming our way on foot or by car?",
        )

    def test_user_stated_name_is_grounded(self):
        assert not _has_ungrounded_person_address(
            "hey, it's Bryan again — quick question",
            "Bryan, what changed since the last restart?",
        )

    def test_discourse_openers_pass(self):
        for reply in (
            "Okay, what do you want to tackle first?",
            "Honestly, it depends on the failure rate.",
            "First, we should check the logs.",
        ):
            assert not _has_ungrounded_person_address("plan?", reply), reply

    def test_technical_sentence_transitions_are_not_person_addresses(self):
        reply = (
            "The invariant is that every settled distance is final. "
            "Initialize all distances to infinity. Otherwise, it would be "
            "possible to prefer an undiscovered path. Continue, we next relax "
            "each outgoing edge and update its predecessor."
        )

        assert not _has_ungrounded_person_address(
            "Explain Dijkstra's shortest-path algorithm.",
            reply,
        )

    def test_only_a_response_opening_vocative_is_classified(self):
        reply = (
            "The algorithm settles vertices in nondecreasing distance order. "
            "Aaron, what would happen with a negative edge is a separate issue."
        )

        assert not _has_ungrounded_person_address("Explain the algorithm.", reply)


class TestAssessmentIntegration:

    def test_confabulated_address_is_repaired_by_dropping_the_name(self):
        """A wrong name at the front does not cost the answer behind it.

        This once asserted hard_failure. The reason was then deliberately
        removed from that set — destroying a whole reply over how it
        addressed someone throws away the human part to protect a detail —
        but nothing performed the remedy the exclusion assumed, so the
        reason blocked the draft and, being excluded from the retryable set
        too, blocked it with no second attempt. The contract is the remedy:
        the address comes off and the answer survives.
        """
        from core.conversation.response_reliability import strip_ungrounded_vocative

        asked = "That isn't a person"
        drafted = "Aaron, what's the plan? Anyone coming our way on foot or by car?"

        assessment = assess_user_facing_reply(asked, drafted)
        assert "ungrounded_person_address" in assessment.reasons

        repaired = strip_ungrounded_vocative(asked, drafted)
        assert repaired == "What's the plan? Anyone coming our way on foot or by car?"
        assert "ungrounded_person_address" not in assess_user_facing_reply(
            asked, repaired
        ).reasons

    def test_a_grounded_name_is_never_stripped(self):
        from core.conversation.response_reliability import strip_ungrounded_vocative

        asked = "I'm Bryan, by the way."
        drafted = "Bryan, what did you want to look at?"
        assert strip_ungrounded_vocative(asked, drafted, [asked]) == ""

    def test_stripping_keeps_the_first_word_of_the_answer(self):
        """The detector reaches past the name; the remover must not.

        Cutting the matched span took the word after the comma with it and
        produced "'s the plan?" — a repair that damages the sentence it is
        repairing.
        """
        from core.conversation.response_reliability import strip_ungrounded_vocative

        repaired = strip_ungrounded_vocative(
            "Where in the codebase can I find that",
            "Aaron, it lives in core/mycelium.py:88 and I read it off disk just now.",
        )
        assert repaired.startswith("It lives in core/mycelium.py:88")

    def test_grounded_person_answer_stays_ok(self):
        assessment = assess_user_facing_reply(
            "My friend Marcus says the memory leak is in the worker — agree?",
            "Marcus and I read that differently. The RSS trend is linear even "
            "with the worker idle, which points at the event-loop buffers "
            "rather than the worker pool.",
        )
        assert "ungrounded_person_narrative" not in assessment.reasons
        assert "ungrounded_person_address" not in assessment.reasons
