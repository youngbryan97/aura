"""The sitting going on now reaches her whole while it is short; meaning prunes only what is longer.

LIVE 8 November 2026, rehearsal 10: "So what did you think of all three?" bore by meaning on none of the sitting's
watchings, so only its last exchange reached her, a failed step's "This did not complete", and she answered "I can't
watch the videos, so I don't have a personal reaction to them."
"""
from __future__ import annotations

import time

import pytest

from core.cognition import evidence_relevance
from core.conversation.history_reach import A_SITTING_READ_WHOLE_CHARS, measure_reach

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
    assert "read whole" in reach.reason


def test_an_earlier_sitting_still_reaches_her_only_as_memory(bears_on_nothing):
    now = time.time()
    earlier = [{"user": ASKED, "aura": "Of three other videos.", "at": now - 20_000}]
    reach = measure_reach(earlier + _sitting(now, started_s_ago=3000), request=ASKED, now=now)
    assert reach.kept == (1, 2, 3)


def test_a_long_sitting_is_read_as_far_as_the_request_bears(bears_on_nothing):
    now = time.time()
    long = _sitting(now, started_s_ago=3000, said="word " * (A_SITTING_READ_WHOLE_CHARS // 8))
    reach = measure_reach(long, request=ASKED, now=now)
    assert reach.kept == (2,)
