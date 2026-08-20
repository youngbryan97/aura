"""Disclaiming evidence she was handed is the mirror of claiming evidence she wasn't.

LIVE, 2026-08-19. The file reading was taken and delivered — the log records
"took 1 reading(s): file you were asked about" — and the reply said:

    [Note: The file path and contents are fictional for this example. If you
    have the actual accounts.py code, I'd be happy to look at it.]

The contents were real, on disk, and in the prompt. Both failures cost the same
thing: the person is told the work cannot be done while it is being done.

Recording what was delivered needs a MUTABLE container, because a ContextVar
set inside a child task does not propagate back to the parent — asyncio hands
children a copy of the context. Readings are taken in child tasks and checked
in the parent.
"""

from __future__ import annotations

import asyncio

import pytest

from core.conversation.response_reliability import disclaims_delivered_evidence
from core.conversation.session_scope import (
    evidence_delivered,
    record_evidence_delivered,
    set_user_question,
)

DISCLAIMER = (
    "Here's the fix. [Note: The file path and contents are fictional for this "
    "example. If you have the actual accounts.py code, I'd be happy to look at it.]"
)


@pytest.fixture(autouse=True)
def _turn():
    set_user_question("read accounts.py and find the sign error")
    yield
    set_user_question("")


def test_the_live_disclaimer_is_caught_when_the_reading_was_delivered():
    assert disclaims_delivered_evidence(DISCLAIMER, {"file"})


def test_with_no_evidence_delivered_there_is_nothing_to_disclaim():
    """Saying an example is hypothetical is fine when it IS one."""
    assert not disclaims_delivered_evidence(DISCLAIMER, set())


def test_an_ordinary_answer_is_untouched():
    assert not disclaims_delivered_evidence(
        "The close() method posts -amount to retained; it should post amount.", {"file"}
    )


def test_speculative_phrasing_about_a_hypothetical_is_not_a_disclaimer():
    assert not disclaims_delivered_evidence(
        "If you had a file like that, the contents would be hypothetical until you share it.",
        {"file"},
    )


def test_delivery_survives_the_task_boundary():
    """The whole reason the container is mutable."""
    from core.brain.observable_grounding import observable_blocks
    from core.brain.observable_registry import install_default_observables

    install_default_observables()
    set_user_question("read CONTRIBUTING.md and tell me the first rule")
    blocks = asyncio.run(observable_blocks("read CONTRIBUTING.md and tell me the first rule"))
    assert blocks
    assert "file" in evidence_delivered()


def test_a_fresh_turn_does_not_inherit_the_last_turns_evidence():
    record_evidence_delivered("file")
    assert "file" in evidence_delivered()
    set_user_question("something else entirely")
    assert evidence_delivered() == frozenset()


def test_the_assessment_reports_it():
    from core.conversation.response_reliability import assess_user_facing_reply

    record_evidence_delivered("file")
    reasons = assess_user_facing_reply("read accounts.py and find the error", DISCLAIMER).reasons
    assert "disclaimed_delivered_evidence" in reasons
