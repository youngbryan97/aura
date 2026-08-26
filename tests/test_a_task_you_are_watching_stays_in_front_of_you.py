"""A long task goes to the background. A task somebody is watching does not.

Handing a long task to the background and answering with a receipt is right
when the person would otherwise sit and wait on something that takes minutes.
That reasoning inverts the moment they ask to be told what is happening AS it
happens: the telling is the thing they asked for, and a ticket cannot deliver
it.

LIVE 2026-08-26: "Find 2048 online, play it, and get to a 256 tile. Say what
you are about to do before each move, and tell me here when you have it." came
back as "Task accepted into governed background execution. Task id: 3f5e2b3b.
Commitment id: 2832f808." — for a request whose whole point was watching.
"""
from __future__ import annotations

import inspect

import pytest

from core.runtime.desktop_objective_intent import asks_to_be_accompanied

WATCHING = [
    "Find 2048 online, play it, and get to a 256 tile. Say what you are about to "
    "do before each move, and tell me here when you have it.",
    "play 2048 and narrate as you go",
    "walk me through the deploy",
    "keep me posted while you refactor that",
    "run the tests, talk me through what breaks",
    "do it step by step and think out loud",
    "let me watch you fill in the form",
]

NOT_WATCHING = [
    "build the app and tell me when you're done",
    "go do the thing and let me know when it's finished",
    "what's the capital of France?",
    "play 2048 until you get a 512 tile",
    "clean up the old logs overnight",
]


@pytest.mark.parametrize("said", WATCHING)
def test_being_told_as_it_happens_is_company(said):
    assert asks_to_be_accompanied(said)


@pytest.mark.parametrize("said", NOT_WATCHING)
def test_being_told_afterwards_is_a_report_and_can_wait(said):
    """The distinction is during versus after. A report can be delivered late;
    company cannot be delivered at all."""
    assert not asks_to_be_accompanied(said)


def test_the_dispatcher_asks_before_sending_it_away():
    from core.agency.task_commitment_verifier import TaskCommitmentVerifier

    source = inspect.getsource(TaskCommitmentVerifier.verify_and_dispatch)
    assert "asks_to_be_accompanied" in source
    where = source.index("asks_to_be_accompanied")
    assert "is_long = False" in source[where : where + 400]


def test_what_she_says_when_something_really_is_backgrounded_is_hers():
    """It still has to be honest — started, nothing finished. It does not have
    to be a ledger entry read aloud."""
    from core.agency import task_commitment_verifier as tcv

    source = inspect.getsource(tcv.TaskCommitmentVerifier._dispatch_async)
    said = source[source.index("summary=\"") :].split("\n", 1)[0]
    assert "I have started on that" in said
    assert "Nothing is finished yet" in said
    # No ledger, no identifiers, in the sentence a person reads.
    assert "ledger" not in said
    assert "task_id" not in said and "commitment_id" not in said


def test_the_surface_does_not_read_internal_identifiers_to_the_person():
    from core.phases import response_generation_unitary as unitary

    source = inspect.getsource(unitary)
    where = source.index('last_task_outcome == "started"')
    block = source[where : where + 1800]
    assert 'f"Task id: {task_id}."' not in block
    assert 'f"Commitment id: {commitment_id}."' not in block
