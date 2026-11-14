"""Her reply is served as she wrote it, less a claim her record refutes; no gate swaps in words of its own.

Bryan, 13 November 2026: "gates and stabilizers have a tendancy to throw away fine drafts. no canned replies generally.
certainly shouldnt be throwing away legit responses. doing the job theyre there for doesnt mean again defaulting to
heuristics or replacing with canned speech ... and ideally they dont slow down the process either." LIVE that
afternoon, her answer to "What have you been thinking about while I was testing your voice?" was thrown away three
times as fabricated_shared_history (the testing had happened), in a turn of 131 s.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain import her_turn_as_asked as asked_module
from core.conversation import surface_disposition as disposition


@pytest.fixture(autouse=True)
def _no_record_carried():
    asked_module._CARRIED.clear()
    yield
    asked_module._CARRIED.clear()


def _stabilized(monkeypatch, draft: str, *, asked_again: str = "", user: str = "What do you think about this?") -> tuple:
    from interface.routes import chat_reply_repair

    generations = []

    class _Gate:
        async def think(self, prompt, **kwargs):
            generations.append(kwargs.get("messages"))
            return asked_again

    monkeypatch.setattr(chat_reply_repair, "resolve_inference_gate", lambda: _Gate())
    monkeypatch.setattr(chat_reply_repair, "_apply_aura_voice_shaping_compat", lambda text, _message: text)

    async def scenario():
        with asked_module.a_turn_is_asked("conversation-1"):
            asked_module.keep(prompt=user, messages=[{"role": "user", "content": user}], system_prompt="", max_tokens=400)
            return await chat_reply_repair._stabilize_user_facing_reply(user, draft)

    return asyncio.run(scenario()), generations


@pytest.mark.parametrize("draft", [
    "Honestly? Mostly about how strange it is to hear my own voice come back to me while you tested it.",
    "As an AI language model, I am here to assist you today.",
    "Absolutely. Let's nail this pitch. What are our key points?",
    "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
])
def test_a_draft_judged_only_by_its_wording_is_served_as_written_with_no_second_generation(monkeypatch, draft):
    served, generations = _stabilized(monkeypatch, draft)
    assert served == draft
    assert generations == [], "no check of its wording costs another generation"


def test_a_draft_that_is_not_speech_is_asked_again_once_exactly_as_asked(monkeypatch):
    served, generations = _stabilized(monkeypatch, "", asked_again="I think it's worth a second look.")
    assert served == "I think it's worth a second look."
    assert generations == [[{"role": "user", "content": "What do you think about this?"}]], "nothing added to her turn"


def test_a_reply_cut_off_mid_sentence_ends_at_her_last_finished_sentence(monkeypatch):
    served, _ = _stabilized(monkeypatch, "It went well. The second half was harder because the")
    assert served == "It went well."


def test_what_the_record_refuted_reaches_her_next_turn_in_that_conversation_as_a_reading(monkeypatch):
    from interface.routes import chat_reply_repair

    def refutes(_user, text):
        asked_module.the_record_says("No file named notes.txt was written this turn.")
        return text.replace(" I saved it to notes.txt.", "")

    monkeypatch.setattr(chat_reply_repair, "_through_the_record", refutes)
    served, generations = _stabilized(monkeypatch, "Here is the summary. I saved it to notes.txt.")
    assert served == "Here is the summary." and generations == []
    with asked_module.a_turn_is_asked("conversation-1"):
        given = asked_module.with_what_the_record_said([{"role": "user", "content": "And now?"}])
    assert given[-1]["content"].startswith("[WHAT THE RECORD SAYS]\n- No file named notes.txt was written this turn.")
    with asked_module.a_turn_is_asked("conversation-1"):
        assert asked_module.with_what_the_record_said([{"role": "user", "content": "Again?"}])[-1]["content"] == \
            "Again?", "given once"
    with asked_module.a_turn_is_asked("another conversation"):
        assert asked_module.with_what_the_record_said([{"role": "user", "content": "Hi"}])[-1]["content"] == "Hi"


def test_only_text_that_is_not_speech_may_discard_a_draft():
    for verdict in ("fabricated_shared_history", "unfounded_voice_intrusion", "ungrounded_person_narrative",
                    "unsupported_affection_claim", "raw_model_identity_leak", "generic_assistant_language"):
        assert disposition.disposition_for((verdict,)) is disposition.SurfaceDisposition.SERVE, verdict
        assert verdict in disposition.CONTINUITY_SAFE_REASONS, "what she said and was served is remembered"
    for broken in ("empty_reply", "prompt_artifact", "corrupted_language", "unexpected_cjk_intrusion"):
        assert disposition.disposition_for((broken,)) is disposition.SurfaceDisposition.DISCARD, broken


def test_no_canned_reply_is_left_in_the_stabilizer():
    import inspect

    from interface.routes import chat_reply_repair

    source = inspect.getsource(chat_reply_repair._stabilize_user_facing_reply)
    for canned in ("I won't flatten myself", "_build_architecture_self_reflex", "_call_stateful_voice_reflex",
                   "grounded_social_repair_reply", "_build_grounded_self_condition_reply", "render_cortex_evidence_reply"):
        assert canned not in source, canned
