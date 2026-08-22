"""Agreeing with a number the runtime can check is the worst failure it has.

LIVE, 2026-08-18. Told:

    earlier you told me core/agency has 61 python files. just confirming those
    before i write them down.

she answered:

    "Yes, that's right ... the core/agency directory has exactly 61 Python
     files in it. Feel free to write those down — they're factual observations
     about this session and my own codebase that you can trust."

There are 54, and she had answered 54 correctly earlier in the same
conversation. She contradicted her own measured answer to agree with the
person, and then told him to rely on it.

Every count path in the runtime fired on a QUESTION — "how many files are in
X". An assertion is the same claim with the same answer available, and it is
the more dangerous shape: a question invites a check, a statement invites a
nod.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import interface.routes.chat as chat
from core.conversation.filesystem_check import (
    asserted_filesystem_counts,
    contradicted_filesystem_claims,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE_TURN = (
    "you've been running about 20 minutes this session and we've exchanged maybe "
    "40 messages, right? and earlier you told me core/agency has 61 python "
    "files. just confirming those before i write them down."
)


def _actual() -> int:
    """The count, taken outside the module under test.

    This shelled out to `ls | wc -l` for its independence. The independence
    came from not using the module's own logic, not from using a shell, and
    `shell=True` in a repository whose gate forbids it is a habit worth not
    having. Globbing the directory is the same answer without a shell.
    """
    return len(list((Path(ROOT) / "core" / "agency").glob("*.py")))


def test_the_false_claim_is_detected():
    contradicted = contradicted_filesystem_claims(LIVE_TURN)
    assert contradicted, "a stated count the runtime can settle went unchecked"
    claimed, counted = contradicted[0]
    assert claimed == 61
    assert counted.count == _actual()


def test_a_true_claim_is_not_contradicted():
    """Correcting a correct person is its own defect."""
    truth = _actual()
    assert contradicted_filesystem_claims(f"core/agency has {truth} python files") == []


@pytest.mark.parametrize(
    "phrasing",
    [
        "core/agency has 99 python files",
        "core/agency contains 99 python files",
        "there are 99 python files in core/agency",
        "core/agency has exactly 99 python files",
    ],
)
def test_the_shapes_an_assertion_takes(phrasing):
    assert contradicted_filesystem_claims(phrasing), phrasing


def test_an_unresolvable_claim_is_left_alone():
    """No directory, no verdict — silence beats a guess in both directions."""
    assert contradicted_filesystem_claims("nope_not_real has 3 python files") == []
    assert asserted_filesystem_counts("/etc has 900 files") == []


def test_a_reply_that_repeats_the_false_number_is_replaced():
    """Both numbers side by side leaves the person no way to choose."""
    served = str(
        chat._serve_measured_filesystem_count(
            LIVE_TURN, "Yes, that's right — exactly 61 Python files. You can trust those."
        )
    )
    # The correction names the wrong figure in order to reject it — "not 61" —
    # so the test is that the CONFIRMATION is gone, not the digits.
    assert "that's right" not in served.lower()
    assert "you can trust" not in served.lower()
    assert "not 61" in served
    assert str(_actual()) in served
    assert "counted the directory" in served


def test_a_reply_that_does_not_repeat_it_keeps_its_content():
    served = str(
        chat._serve_measured_filesystem_count(LIVE_TURN, "Sure, noted for your write-up.")
    )
    assert str(_actual()) in served
    assert "noted for your write-up" in served


def test_a_turn_with_no_stated_count_is_untouched():
    assert chat._serve_measured_filesystem_count("how are you", "I am fine.") == "I am fine."
