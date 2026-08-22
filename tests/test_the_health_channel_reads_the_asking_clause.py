"""A wall of telemetry is not an answer to a request for slides.

LIVE, 2026-08-22, typed into the window: "I have to present you to a funding
panel in 10 minutes. Six slides, no fluff: what you are, what you can actually
do today, one thing you demonstrably do that off-the-shelf assistants can't,
your honest limitations, what the money buys, and what we'd measure."

The reply was "Overall runtime status: healthy. No conducted job is currently
recording failures. No degradations recorded recently."

That is the false positive the gate's own docstring warns about — it is
"deliberately narrow" because serving live readings to a question nobody asked
produces exactly this. The words that matched were spread across a long
request about something else, which is the same defect the queued-work channel
had when it answered the rules of an invented game with a maintenance list: a
topic found in one sentence and a question found in another are not evidence
about the same thing.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import asks_about_own_operational_state


def test_a_long_request_that_mentions_her_is_not_a_health_question():
    asked = (
        "I have to present you to a funding panel in 10 minutes. Six slides, no fluff: "
        "what you are, what you can actually do today, one thing you demonstrably do "
        "that off-the-shelf assistants can't, your honest limitations, what the money "
        "buys, and what we'd measure."
    )
    assert not asks_about_own_operational_state(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "how are you doing?",
        "are any of your subsystems degraded?",
        "which of your subsystems is degraded right now?",
        "is anything failing on your side?",
    ],
)
def test_a_real_question_about_her_still_qualifies(asked: str):
    assert asks_about_own_operational_state(asked), asked


@pytest.mark.parametrize(
    "asked",
    [
        "my deploy is failing",
        "the tests in my repo are broken",
        "something is wrong with the printer",
    ],
)
def test_trouble_that_is_not_hers_does_not(asked: str):
    assert not asks_about_own_operational_state(asked), asked


def test_the_gate_reads_the_clause_that_asks():
    from pathlib import Path

    source = Path("core/introspection/self_evidence.py").read_text(encoding="utf-8")
    block = source[source.index("def asks_about_own_operational_state") :]
    block = block[: block.index("def self_health_answer")]
    assert "asking_part" in block
