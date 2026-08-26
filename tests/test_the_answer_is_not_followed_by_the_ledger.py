"""What she asked about, without the bookkeeping read out after it.

On the run where she actually reached the tile, the reply was:

    Reached it: '256' appeared after 25 move(s). 6 of them did what I
    expected. (Completed 1/1 governed desktop steps.)

The first two sentences are the answer. The third is the machinery counting
itself, and nobody asked how many governed steps it took.
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_objective import _reports_an_outcome


@pytest.mark.parametrize(
    "said",
    [
        "Reached it: '256' appeared after 25 move(s). 6 of them did what I expected.",
        "I could not get there: the page never loaded.",
        "Done — the note is written.",
        "I gave up after the dialog would not close.",
    ],
)
def test_an_outcome_answers_the_question_on_its_own(said):
    assert _reports_an_outcome(said)


@pytest.mark.parametrize(
    "said",
    [
        "Chrome is in front, showing play2048.co",
        "Aura is in front. Behind it, partly visible: Claude and Google Chrome.",
        "The form has four fields and a submit button.",
    ],
)
def test_a_description_of_a_screen_is_not_an_outcome(said):
    """A description says what is there. It does not say how the thing the
    person asked for went, so the step count still has somewhere to live."""
    assert not _reports_an_outcome(said)


def test_the_reply_does_not_append_the_ledger_to_an_outcome():
    import inspect

    from interface.routes import chat_desktop_objective

    source = inspect.getsource(chat_desktop_objective)
    where = source.index("answers_the_question = ")
    assert "_reports_an_outcome" in source[where : where + 200]
