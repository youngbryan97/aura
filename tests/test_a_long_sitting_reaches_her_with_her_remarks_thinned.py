"""A sitting too long to read whole reaches her with remarks of hers left out, not cut down to its last exchange.

LIVE 8 November 2026, the demo: asked "So what did you think of all three?", the sitting's three watchings were 24,000
characters with what she said while each played. Over the room a sitting is read whole in, and borne on by meaning by
none of it, only the last exchange reached her. She said she had predicted "a moral panic" of a video she had predicted
the opposite of, and thanked the video's makers for "admitting his prediction was wrong".
"""
from __future__ import annotations

import time

import pytest

from core.cognition import evidence_relevance
from core.conversation.delivered_history import LEFT_OUT, reached_exchange_messages
from core.conversation.history_reach import A_SITTING_READ_WHOLE_CHARS
from core.utils.injected_blocks import stamp_runtime_payload

ASKED = "So what did you think of all three?"


@pytest.fixture
def bears_on_nothing(monkeypatch):
    def assess(_request, passages):
        return [evidence_relevance.EvidenceAlignment(False, True, None, None, (), "stand-in") for _ in passages]

    monkeypatch.setattr(evidence_relevance, "assess_evidence_alignments", assess)


def _watching(name: str, at: float) -> dict:
    remarks = [f"{name} remark {number}: " + "something on screen " * 30 for number in range(12)]
    remarks.append(f"It's over. My verdict on {name}: the prediction I made was wrong.")
    return stamp_runtime_payload({"user": f"Watch {name} with me.", "aura": f"My prediction for {name}: it goes deeper.",
                                  "said_after": remarks, "timestamp": at})


def _the_demo(now: float) -> list[dict]:
    earlier = [stamp_runtime_payload({"user": ASKED, "aura": "Of three other videos.", "timestamp": now - 30_000})]
    return earlier + [_watching("Kurzgesagt", now - 2100), _watching("Allen Pan", now - 1200),
                      _watching("Salad Fingers", now - 120)]


def test_every_watching_of_the_sitting_reaches_her_with_her_answer_and_her_thoughts_after(bears_on_nothing):
    reached = reached_exchange_messages(_the_demo(time.time()), request=ASKED)
    said = "\n".join(message["content"] for message in reached.messages)
    for name in ("Kurzgesagt", "Allen Pan", "Salad Fingers"):
        assert f"My prediction for {name}: it goes deeper." in said
        assert f"My verdict on {name}: the prediction I made was wrong." in said
    assert "Of three other videos." not in said
    assert LEFT_OUT in said
    assert sum(len(message["content"]) for message in reached.messages) <= A_SITTING_READ_WHOLE_CHARS


def test_a_sitting_that_fits_reaches_her_as_it_was(bears_on_nothing):
    now = time.time()
    short = [stamp_runtime_payload({"user": "Watch this with me.", "aura": "It goes deeper.",
                                    "said_after": ["Two seconds in.", "It's over."], "timestamp": now - 60})]
    reached = reached_exchange_messages(short, request=ASKED)
    assert reached.messages[-1]["content"] == "It goes deeper.\n\nTwo seconds in.\n\nIt's over."


def test_her_own_turns_read_the_sitting_as_it_was(bears_on_nothing):
    reached = reached_exchange_messages(_the_demo(time.time()), request="")
    said = "\n".join(message["content"] for message in reached.messages)
    assert LEFT_OUT not in said and "Kurzgesagt remark 0:" in said
