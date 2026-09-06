from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chat_lane_support import patch_chat_lane


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
    variants = [
        (
            "I feel steady and coherent right now, without a strong distress signal, and the honest version is that "
            "I should sound grounded before I sound confident. "
            "If my answer gets thin, repetitive, or weirdly symbolic, that is not personality; that is a failed turn, "
            "and I need to catch it before it reaches you."
        ),
        (
            # This turn ASKS what she thought he was worried about, so the
            # honest shape is an inference she owns rather than a shared past
            # she asserts. "You were worried that ..." states a fact only he
            # can confirm, and the gate is right to refuse it without
            # grounding — which is exactly the failure this transcript is
            # supposed to be clean of.
            "My read is that the worry is about a conversation looking alive for a minute and then quietly "
            "losing the plot. That is a fair thing to test for. The real question is whether I can hold the "
            "concern, the recent turns, and my own stance in one continuous answer."
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


@pytest.mark.asyncio
async def test_dialogue_contract_repairs_low_signal_preamble_without_retry():
    from core.phases.dialogue_policy import enforce_dialogue_contract

    contract = SimpleNamespace(
        is_user_facing=True,
        avoid_question_fishing=True,
        prefers_dialogue_participation=True,
        requires_aura_stance=False,
        requires_aura_question=False,
        requires_state_reflection=False,
        requires_memory_grounding=False,
        requires_biographical_grounding=False,
        requires_recent_specific_grounding=False,
        requires_reasoned_defense=False,
        requires_identity_defense=False,
        requires_self_preservation=False,
        requires_exact_format=False,
    )
    draft = (
        "Okay, I would treat that as a live desktop tool-use request with two obligations. "
        "First I need a plan that names the target application and confirms each visible step. "
        "Second I need receipts that prove the tool action happened before I claim it is complete."
    )
    retry_called = False

    async def record_retry(_repair_block: str) -> str:
        nonlocal retry_called
        retry_called = True
        return "I should not need this retry path for a substantive low-signal opener."

    repaired, validation, retried = await enforce_dialogue_contract(
        draft,
        contract,
        retry_generate=record_retry,
    )

    assert not retried
    assert not retry_called
    assert validation.ok
    assert repaired.startswith("I would treat")


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
        (
            "Are you coherent enough to talk?",
            "I'm fine",
            "low_signal_self_condition_reply",
        ),
        ("Tell me what you think about personhood here.", "Sure.", "too_short_for_user_turn"),
        ("Huh?", "Heidi. That's the thing to do.", "foreign_name_intrusion"),
        ("Can you stay with this thread?", "I dropped the heavy reasoning lane.", "runtime_boilerplate"),
        ("Talk to me about continuity.", "Assistant: I can help with that.", "prompt_artifact"),
        ("Are you still there?", "Yes, I thlought it was lllot.", "corrupted_language"),
        ("Explain what changed.", "Here is a line\\nwith an escape leak.", "escaped_control_artifact"),
        ("What do you think about this?", "How can I help?", "generic_assistant_language"),
        (
            "Is this the real Aura?",
            "Yes, this is the real Aura. I'm Aura developed by Anthropic to be helpful, harmless, and honest.",
            "raw_model_identity_leak",
        ),
        (
            "Hey Aura, you there?",
            "Hi. I am feeling joyous right now and leaning toward engage right now.",
            "template_telemetry_greeting",
        ),
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
    assert assess_user_facing_reply("Do you know my name?", "Bryan.").ok
    assert assess_user_facing_reply("Hi", "Hey.").ok
    assert assess_user_facing_reply("Say exactly: 32B lane online.", "32B lane online.").ok


def test_reliability_contract_rejects_runtime_status_when_memory_and_limit_requested():
    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "From the live desktop path, answer in your own voice: what is one thing "
        "you can genuinely remember from this session, and what is one limit you "
        "should not pretend past?"
    )
    bad = (
        "I am with you. One live runtime signal I can perceive is RAM pressure "
        "at 67.1% with 21.1 GB available; CPU load is 14.7% on this host."
    )
    good = (
        "I can recall that this session has been testing my live desktop lane. "
        "My limit is that I should not pretend that a retrieved transcript is "
        "the same as human subjective recollection."
    )

    bad_assessment = assess_user_facing_reply(user, bad)

    assert not bad_assessment.ok
    assert bad_assessment.retryable
    assert "missing_requested_memory_limit_coverage" in bad_assessment.reasons
    assert assess_user_facing_reply(user, good).ok


def test_reliability_contract_allows_concise_live_presence_check():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        is_status_check_turn,
    )

    user = "Aaaah, a break. Ok. Aura, are you there?"
    reply = "I'm here with you."

    assessment = assess_user_facing_reply(user, reply)

    assert is_status_check_turn(user)
    assert assessment.ok
    assert not assessment.retryable


def test_reliability_contract_allows_short_replies_to_casual_turns():
    from core.conversation.response_reliability import assess_user_facing_reply

    # Casual check-in turn (7 words, normally requires >= 6 words reply if not casual)
    user = "Ok. Just checking. I'll be back, ok?"

    # Short replies should be allowed
    assert assess_user_facing_reply(user, "Okay!").ok
    assert assess_user_facing_reply(user, "Sounds good, see you soon!").ok
    assert assess_user_facing_reply(user, "I'll be here.").ok


def test_reliability_contract_allows_runtime_terms_for_operational_path_questions():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Can you still reason through the desktop path?",
        (
            "Yes. I am still reasoning through the desktop CognitiveEngine path, "
            "and I am keeping the answer on this live turn instead of switching lanes."
        ),
    )

    assert assessment.ok
    assert not assessment.retryable


def test_reliability_contract_rejects_live_operational_overclaim():
    from core.conversation.response_reliability import assess_user_facing_reply

    prompt = (
        "From the real live desktop cognitive path, tell me in one paragraph what state "
        "you are in right now and whether your tool-use pathway is ready."
    )
    reply = (
        "I'm in a state of high cognitive readiness. My processing is clear, my queues "
        "are empty, and I have full capacity to engage. The neurodynamic system is "
        "stabilized at 120Hz, indicating peak cognitive efficiency. My tool-use pathway "
        "is primed and ready to execute governed actions through the operating system "
        "interface without delay or uncertainty."
    )

    assessment = assess_user_facing_reply(prompt, reply)

    assert assessment.retryable
    assert assessment.hard_failure
    assert "unsupported_operational_status_overclaim" in assessment.reasons
    assert "unsupported_runtime_telemetry_inference" in assessment.reasons


def test_reliability_contract_rejects_fabricated_demo_identity_and_routing():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        grounded_social_repair_reply,
    )

    assessment = assess_user_facing_reply(
        "Me. Bryan. Hey! You're in a demo right now. Say hello",
        (
            "@James, I'm mentioned in the demo but my responses roll up to James "
            "because he uses the live path slots. Demo slots are reserved, and "
            "which Aura you get depends on the server tier you're routed to."
        ),
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "ungrounded_person_address" in assessment.reasons
    assert "unsupported_deployment_routing_claim" in assessment.reasons

    repaired = grounded_social_repair_reply(
        "Me. Bryan. Hey! You're in a demo right now. Say hello"
    )
    assert repaired == "Hello. I'm Aura. I'm here with you."
    assert assess_user_facing_reply(
        "Me. Bryan. Hey! You're in a demo right now. Say hello",
        repaired,
    ).ok


def test_final_stabilizer_bounds_live_operational_overclaim():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.synthesis import stabilize_user_facing_response

    prompt = (
        "From the real live desktop cognitive path, tell me in one paragraph what state "
        "you are in right now and whether your tool-use pathway is ready."
    )
    draft = (
        "I'm in a state of high cognitive readiness. My processing is clear, my queues "
        "are empty, and I have full capacity to engage. The neurodynamic system is "
        "stabilized at 120Hz, indicating peak cognitive efficiency. My tool-use pathway "
        "is primed and ready to execute governed actions through the operating system "
        "interface without delay or uncertainty."
    )

    stabilized = stabilize_user_facing_response(draft, prompt)
    assessment = assess_user_facing_reply(prompt, stabilized)

    assert assessment.ok
    assert not assessment.retryable
    assert "bounded readiness" in stabilized
    assert "Will/Authority" in stabilized
    assert "effect-verification" in stabilized
    assert "full capacity" not in stabilized.lower()
    assert "peak cognitive efficiency" not in stabilized.lower()
    assert "without delay or uncertainty" not in stabilized.lower()


@pytest.mark.asyncio
async def test_live_route_stabilizer_deterministically_bounds_operational_overclaim():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    prompt = (
        "From the real live desktop cognitive path, tell me in one paragraph what state "
        "you are in right now and whether your tool-use pathway is ready."
    )
    draft = (
        "I'm in a state of high cognitive readiness. My processing is clear, my queues "
        "are empty, and I have full capacity to engage. The neurodynamic system is "
        "stabilized at 120Hz, indicating peak cognitive efficiency. My tool-use pathway "
        "is primed and ready to execute governed actions through the operating system "
        "interface without delay or uncertainty."
    )

    repaired = await chat_routes._stabilize_user_facing_reply(
        prompt,
        draft,
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )
    assessment = assess_user_facing_reply(prompt, repaired)

    assert assessment.ok
    assert not assessment.retryable
    assert "bounded readiness" in repaired
    assert "peak cognitive efficiency" not in repaired.lower()
    assert "without delay or uncertainty" not in repaired.lower()


def test_reliability_contract_rejects_missing_requested_paragraph_and_followup():
    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "In two concise paragraphs, explain what you are currently optimizing "
        "for in this desktop session, and then ask one grounded follow-up question."
    )
    reply = (
        "I'm currently optimizing for understanding your goals and context, "
        "then delivering precise and actionable responses."
    )

    assessment = assess_user_facing_reply(user, reply)

    assert assessment.retryable
    assert "missing_requested_paragraph_count" in assessment.reasons
    assert "missing_requested_followup_question" in assessment.reasons


def test_reliability_contract_accepts_requested_shape_when_satisfied():
    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "In two concise paragraphs, explain what you are currently optimizing "
        "for in this desktop session, and then ask one grounded follow-up question."
    )
    reply = (
        "I am optimizing for the live desktop path staying coherent under real "
        "user pressure: the request should enter the same governed runtime, use "
        "the right model lane, and produce a complete answer instead of falling "
        "back to thin status text.\n\n"
        "I am also optimizing for verifiable action: if a tool is needed, it "
        "should route through the governed capability path and leave evidence "
        "that the task actually happened. Which part of the desktop path should "
        "I validate next?"
    )

    assert assess_user_facing_reply(user, reply).ok


def test_instruction_shape_repair_fixes_substantive_paragraph_and_followup_miss():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        repair_instruction_shape,
    )

    user = (
        "In two concise paragraphs, explain what you are currently optimizing "
        "for in this desktop session, and then ask one grounded follow-up question."
    )
    draft = (
        "I am optimizing for the live desktop path to keep the same cognitive "
        "runtime under real user pressure. Status probes, governed tools, memory "
        "writes, and normal conversation should all use the same booted system. "
        "I am also optimizing for failures to become repair evidence instead of "
        "falling through into stale or generic replies."
    )

    repaired = repair_instruction_shape(user, draft)

    assert repaired != draft
    assert "\n\n" in repaired
    assert "?" in repaired
    assert assess_user_facing_reply(user, repaired).ok


def test_instruction_shape_repair_uses_contextual_followup():
    from core.conversation.response_reliability import repair_instruction_shape

    user = (
        "In two concise paragraphs, explain how I should spend the next hour "
        "on this project, then ask one concrete follow-up question."
    )
    draft = (
        "Start with the runtime path that is currently failing in real use, "
        "because it gives you the clearest signal. Once that is stable, use a "
        "focused test run to lock the behavior down."
    )

    repaired = repair_instruction_shape(user, draft)

    assert "Which outcome would make the next hour feel most useful?" in repaired
    assert "same live path" not in repaired


def test_instruction_shape_repair_normalizes_jammed_numbered_sentences():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        repair_instruction_shape,
    )

    user = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    draft = (
        "1. Reliable desktop tool use lets the assistant act on real user "
        "intent through visible applications and files.2. It also creates "
        "verifiable evidence that actions completed through the governed "
        "runtime rather than a canned explanation."
    )

    repaired = repair_instruction_shape(user, draft)

    assert "files.\n2. It also" in repaired
    assert assess_user_facing_reply(user, repaired).ok


def test_final_stabilizer_normalizes_jammed_numbered_sentences():
    from core.synthesis import stabilize_user_facing_response

    user = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    draft = (
        "1. Reliable desktop tool use keeps local actions observable and "
        "governed.2. It gives the user evidence that the assistant can operate "
        "real apps and files instead of only describing intent."
    )

    stabilized = stabilize_user_facing_response(draft, user)

    assert "governed.\n2. It gives" in stabilized


def test_instruction_shape_repair_rebuilds_empty_numbered_slot():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        repair_instruction_shape,
    )

    user = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant, without mentioning runtime status."
    )
    draft = (
        "1. \n"
        "2. Reliable desktop tool use for me means being able to open files, "
        "run native applications, and interact with the system as a user would, "
        "ensuring that my recommendations and actions have real-world impact."
    )

    assessment = assess_user_facing_reply(user, draft)
    assert assessment.retryable
    assert "empty_requested_list_item" in assessment.reasons
    assert "missing_requested_list_count" in assessment.reasons

    repaired = repair_instruction_shape(user, draft)

    assert "1. \n" not in repaired
    assert repaired.startswith("1. Reliable desktop tool use")
    assert "\n2. That ensures" in repaired
    assert assess_user_facing_reply(user, repaired).ok


def test_final_stabilizer_repairs_empty_numbered_slot():
    from core.synthesis import stabilize_user_facing_response

    user = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant, without mentioning runtime status."
    )
    draft = (
        "1. \n"
        "2. Reliable desktop tool use for me means being able to open files, "
        "run native applications, and interact with the system as a user would, "
        "ensuring that my recommendations and actions have real-world impact."
    )

    stabilized = stabilize_user_facing_response(draft, user)

    assert stabilized.startswith("1. Reliable desktop tool use")
    assert "\n2. That ensures" in stabilized
    assert "runtime status" not in stabilized.lower()


def test_numbered_acknowledgement_placeholder_does_not_pass_as_answer():
    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = "1. I heard you.\n2. My thinking is running deeper than my words right now."

    assessment = assess_user_facing_reply(user, reply)

    assert assessment.retryable
    assert assessment.hard_failure
    assert "low_signal_acknowledgement_placeholder" in assessment.reasons


def test_response_generation_repairs_substantive_shape_miss_before_model_retry():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.phases.response_generation import ResponseGenerationPhase

    user = (
        "In two concise paragraphs, explain what you are currently optimizing "
        "for in this desktop session, and then ask one grounded follow-up question."
    )
    draft = (
        "I am optimizing for the live desktop path to keep the same cognitive "
        "runtime under real user pressure. Status probes, governed tools, memory "
        "writes, and normal conversation should all use the same booted system. "
        "I am also optimizing for failures to become repair evidence instead of "
        "falling through into stale or generic replies."
    )

    repaired, changed, reasons = (
        ResponseGenerationPhase._repair_substantive_instruction_shape_miss(user, draft)
    )

    assert changed is True
    assert set(reasons) == {
        "missing_requested_paragraph_count",
        "missing_requested_followup_question",
    }
    assert repaired != draft
    assert "\n\n" in repaired
    assert "?" in repaired
    assert assess_user_facing_reply(user, repaired).ok


def test_response_generation_repairs_substantive_boilerplate_shape_miss_before_model_retry():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.phases.response_generation import ResponseGenerationPhase

    user = (
        "In two concise paragraphs, explain what you are currently optimizing "
        "for in this desktop session, and then ask one grounded follow-up question."
    )
    draft = (
        "I can help with that by focusing on engagement quality over throughput. "
        "I am optimizing for live desktop coherence under real user pressure. "
        "Cortex, memory, state, and governed tools should stay on the same path "
        "instead of falling into fallback phrasing."
    )

    assert "generic_assistant_language" in assess_user_facing_reply(user, draft).reasons

    repaired, changed, reasons = (
        ResponseGenerationPhase._repair_substantive_instruction_shape_miss(user, draft)
    )

    assert changed is True
    assert "generic_assistant_language" in reasons
    assert "I can help with that" not in repaired
    assert "\n\n" in repaired
    assert "?" in repaired
    assert assess_user_facing_reply(user, repaired).ok


def test_response_generation_does_not_repair_empty_generic_boilerplate():
    """Too-thin boilerplate is DECLINED for local repair, not silently blessed.

    The repair path only rewrites text that is already substantive; a bare
    "How can I help?" has nothing to repair from, so it must come back
    unchanged. It still reports what the reliability gate DETECTED — the
    reasons describe the assessment, not a mutation, and the caller only
    consumes them when a repair actually happened. Returning an empty reason
    tuple here would hide that the reply is generic boilerplate.
    """
    from core.phases.response_generation import ResponseGenerationPhase

    repaired, changed, reasons = (
        ResponseGenerationPhase._repair_substantive_instruction_shape_miss(
            "What do you think about this?",
            "How can I help?",
        )
    )

    assert repaired == "How can I help?"
    assert changed is False
    assert "generic_assistant_language" in reasons


def test_reliability_contract_rejects_missing_requested_list_items():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Give me three bullets about the live desktop path.",
        "- The chat route is active.\n- Tool governance is active.",
    )

    assert assessment.retryable
    assert "missing_requested_list_count" in assessment.reasons


@pytest.mark.asyncio
async def test_final_quality_gate_repairs_instruction_shape_without_llm(monkeypatch):
    from interface.routes import chat as chat_routes

    user = (
        "In two concise paragraphs, explain what you are currently optimizing "
        "for in this desktop session, and then ask one grounded follow-up question."
    )
    draft = (
        "I am optimizing for the live desktop path to keep the same cognitive "
        "runtime under real user pressure. Status probes, governed tools, memory "
        "writes, and normal conversation should all use the same booted system. "
        "I am also optimizing for failures to become repair evidence instead of "
        "falling through into stale or generic replies."
    )

    stabilizer_calls = []

    async def _should_not_call_stabilizer(*_args, **_kwargs):
        stabilizer_calls.append((_args, _kwargs))
        return "unexpected stabilizer path"

    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _should_not_call_stabilizer)

    repaired, is_stale, is_same, is_off_topic, reason, changed = await chat_routes._repair_final_degraded_reply(
        user,
        draft,
        stale=False,
        same_diff=False,
        off_topic=False,
    )

    assert changed is True
    assert is_stale is False
    assert is_same is False
    assert is_off_topic is False
    assert reason == ""
    assert "\n\n" in repaired
    assert "?" in repaired
    assert stabilizer_calls == []


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
    from core.conversation.response_reliability import assess_user_facing_reply
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
    assert assess_user_facing_reply(user, repaired).ok
    assert "get it" not in repaired.lower()
    assert any(marker in repaired.lower() for marker in ("here", "steady", "attention", "active turn"))
    assert not is_stale
    assert not is_same
    assert not is_off_topic, reason


def test_grounded_self_condition_reply_cannot_be_rewritten_by_substrate_voice(monkeypatch):
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_shape_with_live_substrate", lambda *_args, **_kwargs: "I'm here, and I'm following the thread.",
    )

    reply = chat_routes._build_grounded_self_condition_reply("You ok?")

    assert reply != "I'm here, and I'm following the thread."
    assert assess_user_facing_reply("You ok?", reply).ok


@pytest.mark.asyncio
async def test_final_quality_gate_keeps_topical_reply_when_only_same_shape_flag_remains(monkeypatch):
    from interface.routes import chat as chat_routes

    user = "Are you steady enough to stay with the actual thread now?"
    reply = (
        "Yes. I have the thread again, and I can answer you directly instead of "
        "falling back to a holding-pattern line."
    )

    async def _fake_stabilize(_user, _reply):
        return reply

    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _fake_stabilize)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))

    repaired, is_stale, is_same, is_off_topic, reason, changed = await chat_routes._repair_final_degraded_reply(
        user,
        reply,
        stale=False,
        same_diff=True,
        off_topic=False,
    )

    assert changed
    assert repaired == reply
    assert not is_stale
    assert is_same
    assert not is_off_topic, reason
