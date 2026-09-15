"""A sentence that puts words in his mouth is dropped; the answer is kept.

LIVE 2026-09-15: the final requested-output check flagged "The 2 o'clock hour
never exists that night." as fabricated shared history — the floor's
time-reference pattern, since narrowed — and the correct 27B answer failed
closed. A smaller model's wrong answer was served under an apology. The
honest remedy for one sentence is to drop the sentence.
"""

from __future__ import annotations

from core.conversation import response_reliability
from core.dialogue import shared_history
from interface.routes import chat_reply_shaping

USER = (
    "Quick sanity check: if I run a script every night at 2am and it takes 90 "
    "minutes, and the clocks go forward that night, what actually happens on a "
    "Mac? Keep it short."
)
OFFENDING = "You told me the whole team had already agreed on the UTC change."
REPLY = (
    "On spring forward the clock jumps from 1:59 to 3:00. " + OFFENDING + " "
    "launchd tracks wall-clock time, so a job at 2:00 fires at 3:00 and your "
    "script runs 3:00 to 4:30; it runs once and does not skip."
)


def test_the_flagged_sentence_is_excised_and_the_answer_served(monkeypatch):
    calls: list[str] = []

    def flags_only_the_offending(reply, user_message="", recent=None, *, grounding=None, **_kw):
        calls.append(reply)
        return [OFFENDING] if OFFENDING in str(reply) else []

    monkeypatch.setattr(shared_history, "fabricated_shared_history", flags_only_the_offending)
    monkeypatch.setattr(
        response_reliability,
        "has_fabricated_shared_history",
        lambda reply, *a, **k: OFFENDING in str(reply),
    )

    trace: dict = {}
    out = chat_reply_shaping._enforce_final_requested_output_contract(
        trace, user_message=USER, reply_text=REPLY
    )

    assert OFFENDING not in out
    assert "launchd tracks wall-clock time" in out
    assert trace["final_requested_output_contract_satisfied"] is True
    mutation = next(
        m for m in trace.get("text_mutations", [])
        if m.get("method") == "measured_sentence_excision"
    )
    assert mutation["reasons"] == ["fabricated_shared_history"]
    assert mutation["authorship_effect"] == "preserved"


def test_without_sentences_closes_the_gap():
    assert chat_reply_shaping._without_sentences("A first. B here. C last.", ["B here."]) == "A first. C last."
    assert chat_reply_shaping._without_sentences("Only this.", ["Only this."]) == ""


def test_delivery_gets_shared_history_evidence_from_conversation_reliability(monkeypatch):
    monkeypatch.setattr(shared_history, "fabricated_shared_history",
                        lambda reply, user: [reply + user])
    assert response_reliability.shared_history_violations("reply", "question") == ["replyquestion"]


def test_a_labelled_premise_is_not_a_question_of_its_own():
    """The same turn: "Quick sanity check: if I run a script ..., and the clocks
    go forward that night, what actually happens on a Mac?" is one question.
    The labelled premise had become a segment, and an answer that never
    repeated "happens" or "Mac" was reported as leaving half of it alone."""
    from core.runtime.structured_input import analyze_prompt_shape

    assert analyze_prompt_shape(USER).question_segments == (
        "Quick sanity check: if I run a script every night at 2am and it takes 90 "
        "minutes, and the clocks go forward that night, what actually happens on a Mac?",
    )
    # Two real questions still split.
    assert analyze_prompt_shape("what's 2^20, and what did I ask you to remember?").question_segments == (
        "what's 2^20?",
        "what did I ask you to remember?",
    )
