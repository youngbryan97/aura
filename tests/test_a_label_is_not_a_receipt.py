"""Showing your working is not claiming a tool ran.

LIVE, 2026-08-28: twelve list-transformation questions in a row came back "I
couldn't get to an answer I'd stand behind on that one." Every one was rejected
as truncated_tail plus unfounded_tool_execution_claim, and the claim was the
word "Output:" in a reply that had worked the answer out in the open.

Nothing had claimed to run anything. "Output:" is ordinary English for "here is
what I got", and it is what somebody writes when showing their own working.
What makes a labelled value a receipt is something being named as having
produced it.

The same lesson was already recorded four lines from where this fired, for "the
result is X" — "that is how anyone states a conclusion... correct arithmetic was
annihilated for phrasing its answer normally" — and left unlearned for the
neighbouring case.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _has_unfounded_tool_execution_claim,
)

_WORKING_ALOUD = (
    "I'll compute this step by step. 0 goes to position 2, 1 stays. Output: [7,6,5]",
    "Working it through in my head. Output: [7, 6, 5]",
    "The rule reverses the list. Output = [7,6,5]",
    "Each element shifts left by one. Output: [6,7,5]",
    "The result is 19/66.",
)

_REALLY_CLAIMING_ONE = (
    "Python: reversed(x). Output: [7,6,5]",
    "The code executed successfully, and the output is: 5",
    "I ran the script. Output: 42",
    "stdout: hello",
    "stderr: nothing",
    "It printed 5",
    "Here's the output of running it: 5",
    "I just searched the web for it.",
    "I looked it up.",
)


@pytest.mark.parametrize("reply", _WORKING_ALOUD)
def test_a_label_on_your_own_working_is_not_a_receipt(reply: str) -> None:
    assert not _has_unfounded_tool_execution_claim(reply), reply


@pytest.mark.parametrize("reply", _REALLY_CLAIMING_ONE)
def test_a_named_instrument_still_needs_one(reply: str) -> None:
    assert _has_unfounded_tool_execution_claim(reply), reply


def test_a_real_receipt_founds_the_claim() -> None:
    said = "I ran the script. Output: 42"
    assert _has_unfounded_tool_execution_claim(said, tool_receipts=[])
    # A receipt says what it is evidence of through the field the tool names
    # itself in, which is "tool" and not "skill".
    assert not _has_unfounded_tool_execution_claim(
        said, tool_receipts=[{"tool": "code_repl", "ok": True}]
    )


def test_the_rule_is_written_once() -> None:
    """It was in two patterns, and only one of them had learned the lesson."""

    from pathlib import Path

    body = Path("core/conversation/response_reliability.py").read_text()
    start = body.index("_TOOL_EXECUTION_CLAIM_RE = re.compile(")
    end = body.index("re.IGNORECASE,", start)
    assert r'\boutput:\s*\S' not in body[start:end]
    assert r'\bstdout:\s*\S' in body[start:end]
