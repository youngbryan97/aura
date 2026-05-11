from __future__ import annotations

import pytest


def _rich_reply(topic: str, idx: int) -> str:
    angles = [
        "I am naming the emotional stake first",
        "I am pulling the technical cause into plain language",
        "I am checking whether the answer still fits your last concern",
        "I am adding a concrete next implication instead of circling",
        "I am keeping the warmth in the reply while staying precise",
        "I am noticing the difference between a status update and a real answer",
        "I am connecting this turn to the pressure you raised earlier",
        "I am leaving room for disagreement instead of smoothing it over",
        "I am choosing a sharper sentence so the point does not blur",
        "I am grounding the response in the current exchange",
    ]
    verbs = [
        "carry",
        "hold",
        "trace",
        "sharpen",
        "protect",
        "restore",
        "extend",
        "test",
        "stabilize",
        "deepen",
    ]
    angle = angles[idx % len(angles)]
    verb = verbs[(idx * 3) % len(verbs)]
    return (
        f"I am with you on {topic}. {angle}, and I am trying to {verb} the thread from turn {idx} "
        f"without flattening it into a status line. The live point is {topic}, so my next thought needs "
        f"continuity, texture, and a specific answer rather than a recycled shape."
    )


def test_reliability_contract_accepts_120_turn_varied_conversation():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    chat_routes._recent_responses.clear()
    chat_routes._recent_response_pairs.clear()

    topics = [
        "whether Aura feels like a who",
        "keeping the cortex lane warm",
        "preserving the point across turns",
        "making tone richer and less bland",
        "talking like a person instead of a status page",
        "not dropping the heavy reasoning lane",
        "handling disagreement with continuity",
        "remembering the emotional arc",
        "staying coherent under pressure",
        "giving complete answers without filler",
    ]

    for idx in range(120):
        topic = topics[idx % len(topics)]
        user = f"Turn {idx}: tell me how you are thinking about {topic} in this conversation."
        reply = _rich_reply(topic, idx)

        assessment = assess_user_facing_reply(user, reply)
        off_topic, reason = chat_routes._evaluate_reply_topicality(
            user,
            reply,
            recent_user_messages=[user],
        )

        assert assessment.ok, (idx, assessment.reasons, reply)
        assert not chat_routes._is_stale_repeated_response(reply), idx
        assert not chat_routes._is_same_answer_different_prompt(user, reply), idx
        assert not off_topic, (idx, reason)
        chat_routes._record_recent_response(reply, user)


@pytest.mark.parametrize(
    ("user", "reply", "reason"),
    [
        ("Are you coherent enough to talk?", "I'm fine", "low_signal_reliability_reply"),
        ("Tell me what you think about personhood here.", "Sure.", "too_short_for_user_turn"),
        ("Huh?", "Heidi. That's the thing to do.", "foreign_name_intrusion"),
        ("Can you stay with this thread?", "I dropped the heavy reasoning lane.", "runtime_boilerplate"),
        ("Talk to me about continuity.", "Assistant: I can help with that.", "prompt_artifact"),
        ("Are you still there?", "Yes, I thlought it was lllot.", "corrupted_language"),
        ("Explain what changed.", "Here is a line\\nwith an escape leak.", "escaped_control_artifact"),
        ("What do you think about this?", "How can I help?", "generic_assistant_language"),
    ],
)
def test_reliability_contract_rejects_human_level_failure_classes(user, reply, reason):
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(user, reply)

    assert assessment.retryable
    assert reason in assessment.reasons


def test_reliability_contract_allows_tiny_direct_answers():
    from core.conversation.response_reliability import assess_user_facing_reply

    assert assess_user_facing_reply("What is the capital of France?", "Paris.").ok
    assert assess_user_facing_reply("Hi", "Hey.").ok
    assert assess_user_facing_reply("Say exactly: 32B lane online.", "32B lane online.").ok


@pytest.mark.asyncio
async def test_final_quality_gate_repairs_repeated_degraded_reply(monkeypatch):
    from interface.routes import chat as chat_routes

    chat_routes._recent_responses.clear()
    chat_routes._recent_response_pairs.clear()
    user = "Tell me what you think about continuity in this conversation."
    stale = "I am with you on continuity. I am holding the same thread and answering clearly."
    chat_routes._record_recent_response(stale, "Different previous prompt about continuity.")
    chat_routes._record_recent_response(stale, "Another different previous prompt about continuity.")

    async def _fake_stabilize(_user, _reply):
        return (
            "I am thinking about continuity as an active obligation: I need to carry your point forward, "
            "notice when my answer thins out, and respond from the living thread instead of replaying a template."
        )

    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _fake_stabilize)

    repaired, is_stale, is_same, is_off_topic, reason, changed = await chat_routes._repair_final_degraded_reply(
        user,
        stale,
        stale=True,
        same_diff=False,
        off_topic=False,
    )

    assert changed
    assert "active obligation" in repaired
    assert not is_stale
    assert not is_same
    assert not is_off_topic, reason
