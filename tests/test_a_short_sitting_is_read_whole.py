"""The sitting going on now reaches her as it goes on: whole, and past what is read of one, its last from one exchange on.

LIVE 8 November 2026, rehearsal 10: "So what did you think of all three?" bore by meaning on none of the sitting's
watchings, so only its last exchange reached her, a failed step's "This did not complete", and she answered "I can't
watch the videos, so I don't have a personal reaction to them."
"""
from __future__ import annotations

import time

import pytest

from core.cognition import evidence_relevance
from core.conversation.history_reach import (
    LEFT_WHEN_LET_GO_CHARS,
    READ_AS_IT_GOES_CHARS,
    measure_reach,
)

ASKED = "So what did you think of all three?"


@pytest.fixture
def bears_on_nothing(monkeypatch):
    def assess(_request, passages):
        return [evidence_relevance.EvidenceAlignment(False, True, None, None, (), "stand-in") for _ in passages]

    monkeypatch.setattr(evidence_relevance, "assess_evidence_alignments", assess)


def _sitting(now: float, *, started_s_ago: float, said: str = "It was good.") -> list[dict]:
    return [
        {"user": "Let's watch this together: a riddle.", "aura": said, "at": now - started_s_ago},
        {"user": "Now pick one from her channel.", "aura": said, "at": now - started_s_ago + 600},
        {"user": "Now a short.", "aura": "This did not complete.", "at": now - started_s_ago + 1500},
    ]


def test_a_short_sitting_reaches_her_whole_though_the_request_bears_on_none_of_it(bears_on_nothing):
    now = time.time()
    reach = measure_reach(_sitting(now, started_s_ago=3000), request=ASKED, now=now)
    assert reach.kept == (0, 1, 2)
    assert "read as it goes on" in reach.reason


def test_an_earlier_sitting_still_reaches_her_only_as_memory(bears_on_nothing):
    now = time.time()
    earlier = [{"user": ASKED, "aura": "Of three other videos.", "at": now - 20_000}]
    reach = measure_reach(earlier + _sitting(now, started_s_ago=3000), request=ASKED, now=now)
    assert reach.kept == (1, 2, 3)


def _talk(n: int, now: float, size: int = 900) -> list[dict]:
    return [{"user": f"question {i}", "aura": f"answer {i} " + "w" * size, "at": now - 60.0 * (n - i)} for i in range(n)]


def test_what_is_read_begins_with_the_same_exchange_turn_after_turn(bears_on_nothing):
    """Her model reads on from the first thing that is different. LIVE 24 November 2026: the median turn waited 14 s for its
    prompt to be read, for the first exchange of it was another one each turn."""
    now = time.time()
    began, read = [], []
    for n in range(4, 40):
        talk = _talk(n, now)
        reach = measure_reach(talk, request="anything at all", budget_chars=2000 if n % 3 else 90_000, now=now)
        assert reach.kept == tuple(range(reach.boundary, n)) and reach.kept[-1] == n - 1
        assert reach.kept_chars <= READ_AS_IT_GOES_CHARS
        began.append(reach.boundary)
        read.append(reach.kept_chars)
    assert began == sorted(began) and len(set(began)) <= 4, f"it moves seldom, and a long way: {began}"
    moved = [read[i] for i in range(1, len(began)) if began[i] != began[i - 1]]
    assert moved and all(chars <= LEFT_WHEN_LET_GO_CHARS for chars in moved)


def test_neither_what_an_answer_affords_nor_what_a_turn_bears_on_changes_what_is_read_of_a_sitting(bears_on_nothing):
    now = time.time()
    talk = _talk(12, now)
    reads = {measure_reach(talk, request=request, budget_chars=budget, now=now).kept for request in ("", "one thing", "another") for budget in (0, 500, 50_000)}
    assert reads == {tuple(range(12))}


def test_the_conversation_is_not_cut_again_by_what_an_answer_affords():
    from core.brain.cognitive_engine import _fit_history_to_what_is_left

    messages = [m for i in range(8) for m in ({"role": "user", "content": f"q{i} " + "w" * 300}, {"role": "assistant", "content": f"a{i} " + "w" * 900})]
    kept, note = _fit_history_to_what_is_left(messages, system_prompt="x" * 400_000, max_tokens=64)
    assert kept == messages and note == ""
