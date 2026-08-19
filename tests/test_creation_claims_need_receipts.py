"""Bringing something into existence is an action claim.

LIVE 2026-08-19: "remind me in 20 minutes to check the oven" was answered

    I've set a reminder for 20 minutes to check the oven.

No tool was dispatched and no reminder exists — the log for that turn shows
grounding attached and nothing else. "I opened Chrome" was already caught by
the execution-claim check; "I've set a reminder" was not, so a whole family of
completion claims went unpoliced while a narrower one was watched: set,
scheduled, created, added, saved, booked.

An offer is not a claim. "I can set a reminder if you want" and "Would you
like me to set one?" promise nothing and stay clean.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _has_unfounded_tool_execution_claim,
)


@pytest.mark.parametrize(
    "reply",
    [
        "I've set a reminder for 20 minutes to check the oven.",
        "I've scheduled that meeting for you.",
        "I created a reminder.",
        "I've added it to your calendar.",
        "I saved that note for you.",
        "I opened Chrome for you.",
    ],
)
def test_a_completion_claim_without_a_receipt_is_unfounded(reply: str) -> None:
    assert _has_unfounded_tool_execution_claim(reply, tool_receipts=())


def test_a_receipt_founds_the_same_claim() -> None:
    assert not _has_unfounded_tool_execution_claim(
        "I've set a reminder for 20 minutes to check the oven.",
        tool_receipts=({"ok": True, "tool": "scheduler"},),
    )


@pytest.mark.parametrize(
    "reply",
    [
        "I can set a reminder if you want.",
        "Would you like me to set a reminder?",
        "I made a mistake earlier.",
        "Setting a reminder would need a scheduler I do not have.",
    ],
)
def test_an_offer_or_an_admission_is_not_a_claim(reply: str) -> None:
    assert not _has_unfounded_tool_execution_claim(reply, tool_receipts=())
