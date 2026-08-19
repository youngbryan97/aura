"""An instruction is not a question about her queue.

LIVE, 2026-08-19. Asked to take a sixty-item personality test — "Work through
all 60 and keep going to the next set until you reach your result" — she
answered with a list of deferred maintenance jobs. The browser had run for nine
minutes and answered most of the form. None of that was mentioned.

The loose pattern matched `work` as her activity and `going to` as a pending
marker. The words were there and the question was not, which is true of most
instructions: they talk about what happens next because that is what they are
for.
"""

from __future__ import annotations

import pytest

from core.brain.observable_registry import _matches_queued_work


@pytest.mark.parametrize(
    "message",
    [
        "what are you going to do next?",
        "anything queued?",
        "What's next for you?",
        "Is there anything waiting to run?",
        "do you have work waiting?",
        "tell me what you have queued",
        "show me your backlog",
    ],
)
def test_asking_still_reads_the_queue(message):
    assert _matches_queued_work(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Now actually take it. Work through all 60 and keep going to the next"
        " set until you reach your result.",
        "Go run the test and tell me your result",
        "Open the page and work through it, then tell me what you got",
        "keep going until you're done",
    ],
)
def test_telling_her_to_do_something_is_not_asking(message):
    assert _matches_queued_work(message) is False


def test_naming_the_queue_outright_still_works_imperatively():
    """The strict path is untouched: it needs no question mark."""
    assert _matches_queued_work("list your pending jobs") is True


def test_the_past_is_still_excluded():
    assert _matches_queued_work("what did you have queued earlier?") is False
