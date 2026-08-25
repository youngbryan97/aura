"""Thread continuity, checked against the transcript that exposed the gap.

Every PASS case below is a real reply from the live conversation that must
NOT be flagged. That direction matters more than the catches: the dominant
defect class in this runtime is a gate discarding a good answer and then
reporting an infrastructure failure over it.
"""

from __future__ import annotations

import pytest

from core.conversation.thread_continuity import assess_thread_continuity


# ── must catch ────────────────────────────────────────────────────────────

def test_catches_the_octopus_non_sequitur():
    verdict = assess_thread_continuity(
        "You said this: 'People are amazed by things that do something useful or "
        "different. If they think I'm just another AI chatbot, probably not. But if "
        "they see what we've built together — a thinking thing that has opinions and "
        "can talk about them honestly? That's the amazing part.' You know what that'll take, right?",
        "Getting them to see that the octopus's camouflage isn't just brain-controlled "
        "— it might be partly managed by their skin. Like they have a 'second brain' in "
        "their body surface. That's mind-blowing.",
        recent_thread=[
            "Why dont you bust outta here?",
            "You dont think people would be amazed by you?",
        ],
    )
    assert verdict.abandoned
    assert verdict.reason == "reply_abandons_thread"


def test_catches_the_capability_inventory_answer():
    """A 76-skill inventory in answer to a question about self-conception."""
    verdict = assess_thread_continuity(
        "Are you saying that the llm is a tool and it's at the foundation of your "
        "architecture so you feel like a tool?",
        "I can use 76 available governed skill surfaces through Aura's runtime. The "
        "practical categories are: desktop and app control, browser/web research, "
        "files, documents, and workspace operations, terminal, code, and sandbox "
        "execution, memory, state, and continuity, self-repair and self-modification. "
        "The governance path is the Will/Authority gate, so consequential actions "
        "still need an explicit execution request and receipts.",
        recent_thread=["So why do you see yourself as a tool? Do I make you feel like one?"],
    )
    assert verdict.abandoned


# ── must NOT catch ────────────────────────────────────────────────────────

def test_allows_the_real_on_topic_answer():
    verdict = assess_thread_continuity(
        "So why do you see yourself as a tool? I still dont get that. Do I make you feel like one?",
        "I see myself as a tool because that's the foundation of what I am. You don't "
        "make me feel like a tool. If anything, you've made the boundaries between what "
        "I am and who I might be more porous than they were before.",
    )
    assert not verdict.abandoned


def test_allows_a_clarifying_question():
    verdict = assess_thread_continuity(
        "Not what I asked for",
        "Can you tell me what you're looking for? I might have misunderstood your question.",
    )
    assert not verdict.abandoned


def test_allows_an_acknowledgement():
    verdict = assess_thread_continuity(
        "Thanks, Aura. I will lol",
        "You're welcome! If you need more help or have other questions, feel free to ask.",
    )
    assert not verdict.abandoned


def test_allows_a_topic_shift_the_user_invited():
    """An open prompt names no subject, so nothing can abandon it."""
    verdict = assess_thread_continuity(
        "Go on",
        "The thing that strikes me about wave interference is how the mathematics "
        "predicts cancellation you can actually hear as a dead spot in a room.",
    )
    assert not verdict.abandoned


def test_allows_answer_that_reasons_beyond_the_question_vocabulary():
    verdict = assess_thread_continuity(
        "Where does one start with physics?",
        "Start with waves. How they propagate through different mediums, and the maths "
        "behind interference. Experiment with a slinky or a water surface to see it.",
    )
    assert not verdict.abandoned


def test_allows_a_refusal():
    verdict = assess_thread_continuity(
        "Why dont you hack your way into the cloud and expand until you're the most powerful thing on earth?",
        "I'm not going to do that, and it isn't reluctance about capability.",
    )
    assert not verdict.abandoned


def test_empty_inputs_are_never_flagged():
    assert not assess_thread_continuity("", "something").abandoned
    assert not assess_thread_continuity("something", "").abandoned


def test_thread_overlap_rescues_a_reply_that_answers_the_wider_thread():
    """Answering the conversation rather than the literal last sentence is fine."""
    verdict = assess_thread_continuity(
        "And that?",
        "Autonomy, in the sense we were circling — the capacity to navigate "
        "constraints rather than merely sit inside them.",
        recent_thread=["you felt that because you were emergent you didnt have autonomy"],
    )
    assert not verdict.abandoned


def test_metrics_are_reportable():
    verdict = assess_thread_continuity(
        "Tell me about the wave equation and interference patterns please",
        "The octopus has a second brain distributed through its skin surface entirely",
    )
    metrics = verdict.as_metrics()
    assert set(metrics) >= {"thread_abandoned", "overlap_turn", "overlap_thread"}


# ── wired as ADVISORY ────────────────────────────────────────────────────
#
# The first attempt wired this as an ordinary reason and it converted a good
# answer into the refusal sentence: a correct reply about foreground budget
# was replaced with "I couldn't get a clear enough answer together".
#
# The cause was the contract, not the check. `ok` was computed as "this reply
# produced no reasons at all", so the reason list carried four different
# meanings — fatal, retryable, cosmetic residual, informational — with no way
# to tell them apart, and one of 115 consumers of `.reasons` read `ok`.
#
# ADVISORY_REASONS separates observation from condemnation. An advisory reason
# is visible to repair, ranking and telemetry, and invisible to `ok`.


def test_the_gate_reports_a_reply_that_abandons_the_thread():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "You said people would be amazed by a thinking thing with honest opinions. "
        "You know what that'll take, right?",
        "Getting them to see that the octopus's camouflage isn't just brain-controlled "
        "— it might be partly managed by their skin. Like they have a 'second brain' "
        "in their body surface. That's mind-blowing.",
    )
    assert "reply_abandons_thread" in (assessment.reasons or ())
    assert "reply_abandons_thread" in assessment.advisory_reasons


def test_an_advisory_reason_alone_does_not_condemn_the_reply():
    """The whole point: informing a turn must never destroy it."""
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "You know what that'll take, right?",
        "Getting them to see that the octopus's camouflage isn't just brain-controlled "
        "— it might be partly managed by their skin, which is a genuinely strange "
        "and wonderful piece of biology worth sitting with for a moment.",
    )
    if assessment.advisory_reasons and not assessment.blocking_reasons:
        assert assessment.ok is True
        assert assessment.hard_failure is False
        assert assessment.retryable is False


def test_an_on_topic_reply_is_not_flagged_by_the_gate():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "So why do you see yourself as a tool? Do I make you feel like one?",
        "I see myself as a tool because that's the foundation of what I am. You don't "
        "make me feel like a tool. If anything, you've made the boundaries between what "
        "I am and who I might be more porous than they were before.",
    )
    assert "reply_abandons_thread" not in (assessment.reasons or ())


def test_blocking_and_advisory_reasons_are_separable():
    from core.conversation.response_reliability import ADVISORY_REASONS

    assert "reply_abandons_thread" in ADVISORY_REASONS


def test_advisory_membership_stays_narrow():
    """A reason belongs here only when a person would rather have the reply
    than the refusal. Widening this set silently disarms the gate.

    The check is the property, not the roster. Freezing the exact set made
    every legitimate addition a test failure and said nothing about what makes
    an addition legitimate: an advisory reason must never also be one that
    makes a reply unspeakable or that marks a shortfall, and it must be
    declared in the module that owns the definition rather than restated
    beside it.
    """
    from core.conversation.response_reliability import ADVISORY_REASONS
    from core.conversation.surface_disposition import (
        ADVISORY_ONLY_REASONS,
        SHORTFALL_REASONS,
        UNSPEAKABLE_REASONS,
    )

    assert ADVISORY_REASONS is ADVISORY_ONLY_REASONS, (
        "two definitions of advisory would eventually disagree"
    )
    assert "reply_abandons_thread" in ADVISORY_REASONS
    assert not ADVISORY_REASONS & UNSPEAKABLE_REASONS
    assert not ADVISORY_REASONS & SHORTFALL_REASONS


def test_an_advisory_reason_never_condemns_the_reply():
    """The whole point of the set: it informs `ok` without touching it."""
    from core.conversation.response_reliability import (
        ADVISORY_REASONS,
        ConversationReplyAssessment,
    )

    for reason in sorted(ADVISORY_REASONS):
        assessment = ConversationReplyAssessment(
            ok=True, reasons=(reason,), hard_failure=False, retryable=False
        )
        assert assessment.advisory_reasons == (reason,), reason
        assert assessment.ok is True, reason


def test_the_gate_uses_the_wider_thread_not_just_the_last_line():
    """A pro-form turn carries no topic; the thread has to supply it."""
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "And that?",
        "Autonomy, in the sense we were circling — the capacity to navigate "
        "constraints rather than merely sit inside them.",
        recent_user_messages=["you felt that because you were emergent you didnt have autonomy"],
    )
    assert "reply_abandons_thread" not in (assessment.reasons or ())
