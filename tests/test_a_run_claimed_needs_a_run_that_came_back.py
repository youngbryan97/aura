"""A reply that says it ran something needs a run that came back, not any receipt with code in its name (U07).

LIVE 17 November 2026, typed into her window: "Write a Python function that groups a list of file paths by
extension, then run it on a few example paths and show me the output." After 21.5 minutes her reply said "I ran it —
here's what actually printed" when nothing had run: the reasoning amplifier's sandbox had refused her script's
`import os`, and her code was saved to a file.

The check passed it. A quoted result needed a receipt whose tool, action, object or verification held "code", "run",
"python" or "query" anywhere, so a failed code_repl run, a search for something about Python, or the file her code
was saved to each founded it. Any other claim of work was founded by any receipt at all, failed or not. Now a claim
of running something is founded only by an executor that ran and came back ok, read from what the receipt's tool and
action say it did and never from the words it was pointed at, and a receipt that came back failed founds no claim.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import _has_unfounded_tool_execution_claim
from core.conversation.what_a_record_vouches_for import founded_by

pytestmark = pytest.mark.unit

RAN_IT = ("I ran it — here's what actually printed:\n\n```\n{'.py': ['main.py', 'utils.py'], '.txt': ['notes.txt']}\n```"
          "\n\nThe function groups each path under its extension.")

_A_RUN = {"tool": "code_repl", "action": "code_repl", "object_ref": "", "ok": True, "receipt_id": "r1"}


@pytest.mark.parametrize(("receipts", "why"), [
    ((), "nothing ran"),
    (({**_A_RUN, "ok": False},), "the run came back refused, as on 17 November"),
    (({"tool": "web_search", "action": "web_search", "object_ref": "python group paths by extension", "ok": True},),
     "a search for something about Python is not a run"),
    (({"tool": "requested_artifact", "action": "write", "object_ref": "Downloads/group_paths.py",
       "ok": True},), "saving the code is not running it"),
])
def test_a_run_claimed_without_a_run_that_came_back_is_unfounded(receipts, why):
    assert _has_unfounded_tool_execution_claim(RAN_IT, tool_receipts=receipts), why


def test_a_run_that_came_back_founds_the_claim():
    assert not _has_unfounded_tool_execution_claim(RAN_IT, tool_receipts=(_A_RUN,))


def test_a_receipt_that_came_back_failed_founds_no_claim_of_work():
    failed = {"tool": "computer_use", "action": "open_app", "object_ref": "Chrome", "ok": False}
    assert not founded_by((failed,), "I opened Chrome for you.")
    assert founded_by(({**failed, "ok": True},), "I opened Chrome for you.")
    assert _has_unfounded_tool_execution_claim("I opened Chrome for you.", tool_receipts=(failed,))
    assert not _has_unfounded_tool_execution_claim("I opened Chrome for you.", tool_receipts=({**failed, "ok": True},))


def test_a_receipt_that_does_not_say_how_it_came_back_still_founds_as_before():
    """Receipts from older writers carry no outcome; only one that says it failed is set aside."""
    assert founded_by(({"tool": "computer_use"},), "I opened Chrome for you.")
