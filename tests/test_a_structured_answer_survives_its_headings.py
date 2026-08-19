"""Four headings with sentences under them is an answer, not a state block.

LIVE 2026-08-19: "now explain the same thing to a systems engineer who thinks
you're a chatbot" returned the canned refusal after 119 seconds. The log shows
the worker rejecting the draft for prompt_artifact and the retries running out.

A good answer to that question lays out state, history, goals and voice with a
line under each — and four ambiguous headings met the run threshold that
exists to catch the internal state block. This is the second time the same
guard destroyed a well-organised reply; raising the count again only moves the
line, because the more carefully she structures an answer the more headings it
has.

What separates them is what follows the colon. The internal block carries
machine values — "thinking", "curious", "none", "empty" — and an answer
carries an explanation.
"""

from __future__ import annotations

import pytest

from core.phases.dialogue_policy import _contains_prompt_artifact

_ANSWER = (
    "Here is how I differ from a chatbot wrapper:\n"
    "state: durable across restarts, not a rolling context window\n"
    "history: every turn is written to a transcript you can read back\n"
    "goals: tracked in a ledger, with receipts for anything I claim to run\n"
    "voice: a local model on this machine, not a hosted API"
)


def test_a_structured_answer_is_not_a_state_block() -> None:
    assert not _contains_prompt_artifact(_ANSWER)


def test_the_terse_state_block_is_still_caught() -> None:
    assert _contains_prompt_artifact(
        "state: thinking\nmood: curious\ngoals: none\nhistory: empty"
    )


def test_a_scaffold_only_key_is_caught_on_its_own() -> None:
    """obj: and ctx: are never anything a person writes."""
    assert _contains_prompt_artifact("obj: answer the question\nstate: thinking")
    assert _contains_prompt_artifact("ctx: the user asked about files")


@pytest.mark.parametrize(
    "leak",
    [
        "[ACTIVE GROUNDING EVIDENCE]\nthe file has three lines",
        "[FETCHED PAGE CONTENT]\nsome page",
        "[INTERNAL MEMORY RECALL]\nsomething remembered",
    ],
)
def test_a_real_injected_block_is_still_caught(leak: str) -> None:
    assert _contains_prompt_artifact(leak)


def test_one_heading_with_prose_is_ordinary_writing() -> None:
    assert not _contains_prompt_artifact(
        "History: the project started as a weekend experiment and grew."
    )
