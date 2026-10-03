"""A request reads the earlier exchanges it bears on, and the last one.

LIVE 2026-10-02 20:26: "Bam 83 point game or Kobe 81 point game?" reached all
forty exchanges of a conversation about her code and a personality test. The
cortex read 9,935 tokens of it in 32.6s, the fallback 18,137 in 84.8s, and
the answer was two words. LIVE 2026-10-03: the same question, asked nine
times among unrelated turns, reached 22 exchanges back and read 10,071 tokens.
These tests hold the reach to the judge's verdicts; the judge itself is
``core/cognition/evidence_relevance.py``.
"""

from __future__ import annotations

import pytest

from core.cognition import evidence_relevance
from core.conversation.delivered_history import messages_the_request_reaches
from core.conversation.history_reach import measure_reach

CONVERSATION = [
    {"user": "Take the personality test on openpsychometrics", "aura": "I finished it: INTJ."},
    {"user": "What did the test say about intuition?", "aura": "Strongly intuitive."},
    {"user": "Do you know what changes were made to your code?", "aura": "I have no diff loaded."},
    {"user": "If your code changes, do you know when?", "aura": "Not without a record of it."},
]


@pytest.fixture
def judge(monkeypatch):
    """A judge that says a passage bears on the request when they share a topic word."""

    topics = ("test", "code", "kobe")

    def assess(request, passages):
        wanted = {topic for topic in topics if topic in request.lower()}
        return [
            evidence_relevance.EvidenceAlignment(
                any(topic in passage.lower() for topic in wanted), True, None, None, (), "stand-in"
            )
            for passage in passages
        ]

    monkeypatch.setattr(evidence_relevance, "assess_evidence_alignments", assess)


def test_a_request_nothing_earlier_bears_on_reads_the_last_exchange(judge) -> None:
    reach = measure_reach(CONVERSATION, request="Bam 83 point game or Kobe 81 point game?")
    assert (reach.boundary, reach.retained, reach.dropped) == (3, 1, 3)
    assert "no earlier exchange bears" in reach.reason


def test_a_request_reads_what_it_bears_on_and_skips_what_it_does_not(judge) -> None:
    reach = measure_reach(CONVERSATION, request="How do you feel about the test now?")
    # The two test exchanges and the last; the code question between is left out.
    assert reach.kept == (0, 1, 3) and (reach.retained, reach.dropped) == (3, 1)
    about_code = measure_reach(CONVERSATION, request="Who changes your code?")
    assert about_code.kept == (2, 3) and about_code.boundary == 2


def test_meaning_only_ever_shortens_what_a_budget_allows(judge) -> None:
    last_two = sum(len(e["user"]) + len(e["aura"]) for e in CONVERSATION[-2:])
    reach = measure_reach(CONVERSATION, budget_chars=last_two, request="How do you feel about the test now?")
    assert reach.kept == (3,) and reach.boundary == 3
    assert measure_reach(CONVERSATION).retained == len(CONVERSATION)


def test_the_fallback_transcript_is_cut_at_an_exchange(judge) -> None:
    messages = []
    for exchange in CONVERSATION:
        messages += [{"role": "user", "content": exchange["user"]}, {"role": "assistant", "content": exchange["aura"]}]
    reached = messages_the_request_reaches(messages, "Bam 83 point game or Kobe 81 point game?")
    assert reached == messages[-2:]
    assert messages_the_request_reaches(messages, "and the test?") == messages[:4] + messages[-2:]
    from core.conversation.delivered_history import reached_exchange_messages
    from core.utils.injected_blocks import stamp_runtime_payload

    attested = [stamp_runtime_payload(dict(exchange)) for exchange in CONVERSATION]
    reached_history = reached_exchange_messages(attested, request="and the test?")
    assert [m["content"] for m in reached_history.messages if m["role"] == "user"] == [
        CONVERSATION[0]["user"], CONVERSATION[1]["user"], CONVERSATION[3]["user"]
    ]
    assert "The other 1 are not in front of you" in reached_history.note
