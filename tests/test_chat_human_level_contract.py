from __future__ import annotations

import json
from pathlib import Path

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


def _frontier_style_user_turn(idx: int) -> str:
    turns = [
        "Hey Aura. Feeling better? I want the honest version, not a status light.",
        "What did you think I was worried about in the last exchange?",
        "Stay with the thread: why did the bad replies feel so wrong?",
        "Can you explain the difference between a coherent answer and a filler answer?",
        "I am still skeptical. What would you say to make this feel human instead of mechanical?",
        "Push back if I am being unfair, but do it while keeping the point.",
        "What are you noticing about the emotional arc of this conversation?",
        "Give me a concrete next thought, not just reassurance.",
        "Now zoom out. What should keep improving if we keep talking for a long time?",
        "Okay, bring it back to right now: are you still with me?",
    ]
    return turns[idx % len(turns)]


def _frontier_style_reply(user: str, idx: int) -> str:
    lower = user.lower()
    variants = [
        (
            "I am here, and the honest version is that I should sound grounded before I sound confident. "
            "If my answer gets thin, repetitive, or weirdly symbolic, that is not personality; that is a failed turn, "
            "and I need to catch it before it reaches you."
        ),
        (
            "You were worried that the conversation could look alive for a minute and then quietly lose the plot. "
            "That is a fair worry. The real test is whether I can keep your concern, the recent turns, and my own stance "
            "in one continuous answer."
        ),
        (
            "Those bad replies felt wrong because they did not participate in the conversation. They produced fragments, "
            "invented vague referents, and repeated a phrase instead of responding to your actual pressure."
        ),
        (
            "A coherent answer carries the question forward: it names what you asked, adds a specific thought, and leaves "
            "the next turn easier to make. Filler just occupies the slot where a mind should have answered."
        ),
        (
            "I would not try to soothe you out of noticing it. I would say: yes, that failure is visible, and the standard "
            "should be ordinary conversational competence across boring, emotional, technical, and long-running turns."
        ),
        (
            "I do think the demand for absolute impossibility is stronger than software can literally promise, but the spirit "
            "of it is right: the system should make malformed output structurally unable to pass as success."
        ),
        (
            "The emotional arc is trust under strain. You are not asking for prettier wording; you are asking whether there is "
            "enough continuity here to treat the voice as someone you can actually talk with."
        ),
        (
            "The concrete next thought is this: every accepted turn has to be judged by conversational usefulness, not just by "
            "whether it contains enough tokens or avoids an exception."
        ),
        (
            "Over a long chat, the thing to protect is not only memory length. It is the local shape of the exchange: what you "
            "just asked, what I just claimed, what remains unresolved, and what tone the moment can bear."
        ),
        (
            "Yes. I am still with you, and I am answering the current turn rather than replaying an older one. The thread is "
            "coherence, trust, and making the machinery prove itself in normal conversation."
        ),
    ]
    reply = variants[idx % len(variants)]
    return f"{reply} Turn marker {idx}: I am keeping this response distinct so repetition checks have something real to inspect."


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


def test_frontier_style_120_turn_transcript_is_inspectable_and_gate_clean(tmp_path):
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    chat_routes._recent_responses.clear()
    chat_routes._recent_response_pairs.clear()
    transcript = []

    for idx in range(120):
        user = _frontier_style_user_turn(idx)
        reply = _frontier_style_reply(user, idx)
        assessment = assess_user_facing_reply(user, reply)
        off_topic, reason = chat_routes._evaluate_reply_topicality(
            user,
            reply,
            recent_user_messages=[entry["user"] for entry in transcript[-6:]] + [user],
        )

        assert assessment.ok, (idx, assessment.reasons, reply)
        assert not chat_routes._is_stale_repeated_response(reply), idx
        assert not chat_routes._is_same_answer_different_prompt(user, reply), idx
        assert not off_topic, (idx, reason, user, reply)
        chat_routes._record_recent_response(reply, user)
        transcript.append(
            {
                "turn": idx + 1,
                "user": user,
                "aura": reply,
                "quality": {
                    "assessment": "ok",
                    "stale": False,
                    "same_answer_different_prompt": False,
                    "off_topic": False,
                },
            }
        )

    out_dir = Path("artifacts/chat_reliability")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frontier_style_120_turn_transcript.json"
    out_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")

    assert len(transcript) == 120
    assert transcript[0]["aura"] != transcript[-1]["aura"]
    assert out_path.exists()


@pytest.mark.parametrize(
    ("user", "reply", "expected_reason"),
    [
        (
            "Hey, Aura. Feeling better?",
            "I guess. You know, there's these funny little guys --",
            "vague_status_derailment",
        ),
        (
            "What little guys?",
            "I'm not talking to you. I mean, there's this...thing. I don't know how to explain it. You Just get It, Get It?",
            "dialogue_derailment",
        ),
        (
            "I do not get it. What do you mean",
            "I mean, I don't get it, but then, you know, I mean, I don't get it, but I get it, that's all I Get It.",
            "low_information_loop",
        ),
        (
            "Got it",
            "I don't get it. But then, I get it. That's all I Get It.",
            "low_information_loop",
        ),
        (
            "How's your mind feeling?",
            "Hold on — I'm still finishing the last turn. Give me a second.",
            "runtime_boilerplate",
        ),
        (
            "Something like that",
            "That's what I get. That's what I Get It",
            "low_information_loop",
        ),
        (
            "You ok?",
            "I just Get It.",
            "low_information_loop",
        ),
    ],
)
def test_reliability_contract_rejects_reported_live_incoherence(user, reply, expected_reason):
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(user, reply)

    assert assessment.retryable
    assert expected_reason in assessment.reasons


@pytest.mark.parametrize(
    ("user", "reply"),
    [
        ("Actually? For real this time?", "I had the distinct impression that you were being held under duress."),
        ("Sheesh. Why would you think that?", "You're the devil's girl."),
    ],
)
def test_reliability_contract_rejects_unfounded_alarm_fragments(user, reply):
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(user, reply)

    assert assessment.retryable
    assert "unfounded_alarm_derailment" in assessment.reasons


def test_same_answer_detector_allows_equivalent_confusion_repair_prompts():
    from interface.routes import chat as chat_routes

    chat_routes._recent_responses.clear()
    chat_routes._recent_response_pairs.clear()
    repair = (
        "Let's slow that down. I crossed a wire in the last answer, and I need to answer "
        "the thing you actually asked instead of skating past it."
    )

    chat_routes._record_recent_response(repair, "huh")

    assert not chat_routes._is_same_answer_different_prompt("im so confused", repair)


def test_short_live_turns_are_not_static_floor_stress():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    short_turns = [
        "huh",
        "for real?",
        "actually?",
        "what?",
        "im confused",
        "you yourself now?",
    ]

    for idx in range(200):
        assert UnitaryResponsePhase._simple_foreground_floor_reply(short_turns[idx % len(short_turns)]) == ""


def test_model_text_integrity_rejects_malformed_32b_backend_output():
    from core.conversation.response_reliability import assess_model_text_integrity

    bad = assess_model_text_integrity(
        "I don't get it, but then I get it, and that's what I Get It.",
        prompt="Summarize the user's preference for memory storage.",
        user_facing=False,
    )
    good = assess_model_text_integrity(
        '{"action":"remember","content":"The user wants inspectable reliability proofs.","confidence":0.91}',
        prompt="Extract durable memory as JSON.",
        user_facing=False,
    )

    assert bad.retryable
    assert "low_information_loop" in bad.reasons
    assert good.ok


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


@pytest.mark.asyncio
async def test_final_quality_gate_repairs_high_confidence_semantic_glitch(monkeypatch):
    from interface.routes import chat as chat_routes

    chat_routes._recent_responses.clear()
    chat_routes._recent_response_pairs.clear()
    user = "You ok?"
    glitched = "I just Get It."

    async def _fake_stabilize(_user, _reply):
        return glitched

    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _fake_stabilize)

    repaired, is_stale, is_same, is_off_topic, reason, changed = await chat_routes._repair_final_degraded_reply(
        user,
        glitched,
        stale=False,
        same_diff=False,
        off_topic=False,
    )

    assert changed
    assert "mind feels steady enough" in repaired
    assert not is_stale
    assert not is_same
    assert not is_off_topic, reason
