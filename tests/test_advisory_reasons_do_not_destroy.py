"""An advisory objection describes a reply. It must never delete one.

ADVISORY_REASONS has said so in its docstring since it was introduced —
"describe a turn but never destroy it" — but disposition_for did not know
the set existed, so every advisory reason came back REPAIR. At the
conversation-learning gate REPAIR means "do not remember this exchange",
and the exchange was dropped.

The reply that triggered it was correct. "My laptop crashed when Aura used
over 100GB of RAM." answered with "I will treat that as a live desktop
reliability fault and preserve the context." shares no content word with
the question, because it paraphrases instead of echoing — which is what a
good reply does. Overlap zero, verdict reply_abandons_thread, memory gone.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.conversation.surface_disposition import (
    ADVISORY_ONLY_REASONS,
    SurfaceDisposition,
    disposition_for,
    draft_is_servable,
)
from core.conversation.thread_continuity import assess_thread_continuity


class TestAdvisoryReasonsServe:
    def test_an_advisory_reason_alone_serves(self) -> None:
        assert disposition_for(("reply_abandons_thread",)) is SurfaceDisposition.SERVE

    def test_every_declared_advisory_reason_serves_alone(self) -> None:
        for reason in ADVISORY_ONLY_REASONS:
            assert disposition_for((reason,)) is SurfaceDisposition.SERVE, reason

    def test_an_advisory_reason_beside_a_real_one_still_repairs(self) -> None:
        """Advisory does not launder a genuine defect sitting next to it."""
        assert (
            disposition_for(("reply_abandons_thread", "truncated_tail"))
            is SurfaceDisposition.REPAIR
        )

    def test_unspeakable_still_discards(self) -> None:
        assert disposition_for(("empty_reply",)) is SurfaceDisposition.DISCARD

    def test_advisory_drafts_remain_servable(self) -> None:
        assert draft_is_servable(("reply_abandons_thread",))

    def test_the_two_declarations_are_one(self) -> None:
        """response_reliability must not carry a second, drifting copy."""
        from core.conversation.response_reliability import ADVISORY_REASONS

        assert ADVISORY_REASONS is ADVISORY_ONLY_REASONS


class TestTheParaphraseThatTriggeredIt:
    USER = "My laptop crashed when Aura used over 100GB of RAM."
    REPLY = (
        "I will treat that as a live desktop reliability fault and "
        "preserve the context."
    )

    def test_the_overlap_heuristic_still_flags_it(self) -> None:
        """The heuristic is not wrong to notice — it is wrong to be obeyed."""
        verdict = assess_thread_continuity(self.REPLY, self.USER)
        assert verdict.abandoned
        assert verdict.overlap_with_turn == 0.0

    def test_but_the_exchange_is_still_recorded(self) -> None:
        verdict = assess_thread_continuity(self.REPLY, self.USER)
        assert disposition_for((verdict.reason,)) is SurfaceDisposition.SERVE


class TestWordingFailuresKeepTheRecord:
    """The demo bug: her own jargon erased the conversation.

    2026-07-30 demo — Aura answered a desktop request in internal vocabulary
    ("Desktop task completed 2/2 governed computer-use steps through
    heuristic_compat planning"). The reply tripped pseudo_internal_jargon and
    function_word_starvation, the learning gate refused, and the ENTIRE turn
    was dropped: not just her answer, but the fact that Bryan had asked.
    """

    def test_wording_objections_are_continuity_safe(self) -> None:
        from core.conversation.surface_disposition import CONTINUITY_SAFE_REASONS

        assert {
            "borrowed_owner_first_person_speech",
            "pseudo_internal_jargon",
            "function_word_starvation",
        } <= CONTINUITY_SAFE_REASONS


    def test_the_demo_reply_is_wording_only(self) -> None:
        from core.conversation.response_reliability import (
            assess_conversation_learning_admission,
        )
        from core.conversation.surface_disposition import CONTINUITY_SAFE_REASONS

        verdict = assess_conversation_learning_admission(
            "Can you open the Notes app and write a note where you write a "
            "paragraph describing yourself?",
            "Desktop task completed 2/2 governed computer-use steps through "
            "heuristic_compat planning. Completed 2/2 governed desktop steps.",
        )
        assert not verdict.ok, "this reply should still not become experience"
        assert set(verdict.reasons) <= CONTINUITY_SAFE_REASONS, (
            "but every objection to it is about wording, so the exchange is "
            "still remembered"
        )

    def test_the_misgrounded_reply_is_not(self) -> None:
        from core.conversation.response_reliability import (
            assess_conversation_learning_admission,
        )
        from core.conversation.surface_disposition import CONTINUITY_SAFE_REASONS

        verdict = assess_conversation_learning_admission(
            "Are you okay though? Feeling fine?",
            "I am with you. RAM pressure is 75.6% with 15.6 GB available; "
            "CPU load is 25.8% on this host.",
        )
        assert not verdict.ok
        assert not set(verdict.reasons) <= CONTINUITY_SAFE_REASONS


class TestEveryGateThatReadsTheAssessment:
    """One advisory-only exchange, driven through every gate that judges one.

    `disposition_for` learning about ADVISORY_REASONS fixed the gates that
    read `.ok`. It did not fix a gate reading `.reasons` directly — and
    `cognitive_ingress._conversation_pre_admission` did exactly that,
    rejecting a memory on ANY entry in the list. So the reply was served, was
    committed as experience, and was then refused re-admission as evidence,
    because a heuristic that three other gates had already ruled harmless was
    fatal at the fourth.

    The generalisation is behavioural rather than lexical on purpose: a rule
    like "never write `.reasons`" is a style check a new gate can satisfy
    while still failing. This drives the actual gates.
    """

    #: A correct answer that paraphrases instead of echoing. Overlap 0.000.
    USER = "Can you explain why the deadline slipped?"
    REPLY = (
        "Because a floor was applied on top of the remaining allowance. "
        "Whenever less than ten seconds were left, the probe still got ten, "
        "so the promise was exceeded by however much was missing."
    )

    def _assessment(self):
        from core.conversation.response_reliability import (
            assess_conversation_learning_admission,
        )

        return assess_conversation_learning_admission(self.USER, self.REPLY)

    def test_the_heuristic_does_fire_on_it(self) -> None:
        assert "reply_abandons_thread" in self._assessment().reasons

    def test_and_the_assessment_calls_it_admissible(self) -> None:
        verdict = self._assessment()
        assert verdict.ok
        assert verdict.blocking_reasons == ()

    def test_the_chat_turn_logger_admits_it(self) -> None:
        from core.memory.chat_turn_logger import ChatTurnLogger

        turn_logger = ChatTurnLogger.__new__(ChatTurnLogger)
        assert turn_logger._is_meaningful_turn(self.USER, self.REPLY) is True

    def test_the_retrieval_pre_admission_admits_it(self) -> None:
        """The gate that was still reading `.reasons`."""
        from core.brain.cognitive_ingress import _conversation_pre_admission

        hit = SimpleNamespace(
            metadata={
                "conversation_turn": True,
                "user_utterance": self.USER,
                "aura_response": self.REPLY,
            }
        )
        admitted, detail = _conversation_pre_admission(
            self.USER, hit, origin="unit"
        )

        assert admitted, f"advisory-only memory refused re-admission: {detail}"

    def test_a_genuinely_bad_reply_is_still_refused_there(self) -> None:
        """The gate must still work — this is not a removal of the check."""
        from core.brain.cognitive_ingress import _conversation_pre_admission

        hit = SimpleNamespace(
            metadata={
                "conversation_turn": True,
                "user_utterance": "Are you okay though? Feeling fine?",
                "aura_response": (
                    "I am with you. RAM pressure is 75.6% with 15.6 GB "
                    "available; CPU load is 25.8% on this host."
                ),
            }
        )
        admitted, detail = _conversation_pre_admission(
            "Are you okay though? Feeling fine?", hit, origin="unit"
        )

        assert not admitted
        assert detail is not None
        assert any(r.startswith("current_quality:") for r in detail["reasons"])
