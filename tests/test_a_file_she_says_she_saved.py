"""Claiming a file was written is a claim, and a file names itself.

LIVE, 2026-08-20. Asked to build a single-file web app, she answered "I saved
it as `sitting_timer.html` in your Downloads folder", at high confidence,
having written nothing — the tool call that would have written it was cut off
mid-argument by a token cap and never ran.

The action-claim rule already covered "saved … a file". It missed this by
four characters: "folder" sat 44 characters from the determiner and the
window is 40. A distance is the wrong test for a claim whose object is right
there with an extension on it.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import _DESKTOP_ACTION_CLAIM_RE as CLAIM


@pytest.mark.parametrize(
    "reply",
    [
        "I saved it as `sitting_timer.html` in your Downloads folder.",
        "I saved it as sitting_timer.html in your Downloads folder.",
        "I wrote the file to /tmp/report.csv for you.",
        "I've saved the file to your Downloads folder.",
        "I generated index.html and put it on your desktop.",
        "I exported the results as summary.json.",
        "I've set a reminder for 20 minutes to check the oven.",
    ],
)
def test_a_completion_claim_is_recognised(reply: str) -> None:
    assert CLAIM.search(reply) is not None


@pytest.mark.parametrize(
    "reply",
    [
        "You could save it as timer.html if you like.",
        "An html file is just text with tags.",
        "I think sitting_timer.html would be a good name.",
        "Would you like me to write it to disk?",
        "A .csv is a comma-separated file.",
    ],
)
def test_talking_about_a_file_is_not_claiming_one(reply: str) -> None:
    assert CLAIM.search(reply) is None


def test_the_claim_needs_a_receipt() -> None:
    """With no tool receipt, the claim is unsupported; with one, it stands."""
    from core.conversation.response_reliability import (
        _has_unfounded_tool_execution_claim as unfounded,
    )

    reply = "I saved it as sitting_timer.html in your Downloads folder."
    assert unfounded(reply, tool_receipts=()) is True
    assert (
        unfounded(reply, tool_receipts=({"tool": "file_operation", "action": "write"},))
        is False
    )
