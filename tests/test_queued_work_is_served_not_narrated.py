"""What she has queued is a list, not a feeling about continuity.

LIVE, 2026-08-19. "what are you going to do after this?" — the reading was
taken and reached dispatch, the log records "took 1 reading(s): work you have
queued" — and the answer was:

    After this, I'm going to keep running. There's no stopping point — the
    system just keeps processing until it gets told otherwise.

Two jobs were waiting at that moment: dlq_recovery, held by an active
foreground generation, and biological_sleep, held for want of a user anchor.
Neither was mentioned.

Third channel to get this treatment after file counts and belief history, for
the same reason: evidence informs, it does not enforce, and a pending list is
not a matter of opinion.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _reply_was_served_from_a_record, _serve_queued_work


class _Coordinator:
    def __init__(self, pending):
        self._pending = pending

    def status(self):
        return {"pending": self._pending}


@pytest.fixture
def coordinator(monkeypatch):
    def install(pending):
        import core.maintenance.dream_coordinator as module

        monkeypatch.setattr(module, "get_dream_coordinator", lambda: _Coordinator(pending))

    return install


def test_the_real_pending_list_is_served(coordinator):
    coordinator(
        {
            "dlq_recovery": {"reason": "foreground_generation_active"},
            "biological_sleep": {"reason": "no_user_anchor"},
        }
    )
    served = _serve_queued_work("what are you going to do after this?", "I'll keep running.")
    assert "2 jobs waiting to run" in served
    assert "dlq recovery" in served
    assert "biological sleep" in served
    assert "foreground generation active" in served
    # Identifiers are not what a person reads.
    assert "_" not in served


def test_a_question_about_something_else_is_left_alone(coordinator):
    coordinator({"dlq_recovery": {"reason": "x"}})
    for unrelated in (
        "what is 2 + 2",
        "tell me a story",
        "what did you do after the update?",
        "Explain Dijkstra and include a priority queue trace.",
        "What songs are queued in the music player?",
    ):
        assert _serve_queued_work(unrelated, "the model's reply") == "the model's reply"


def test_an_empty_queue_leaves_the_answer_to_her(coordinator):
    """Nothing pending is not a fact that needs composing over her words."""
    coordinator({})
    assert _serve_queued_work("anything queued?", "Nothing waiting.") == "Nothing waiting."


def test_a_broken_coordinator_does_not_break_the_turn(monkeypatch):
    import core.maintenance.dream_coordinator as module

    def explode():
        raise RuntimeError("coordinator is down")

    monkeypatch.setattr(module, "get_dream_coordinator", explode)
    assert _serve_queued_work("anything queued?", "the reply") == "the reply"


def test_the_served_list_carries_its_own_confidence(coordinator):
    """Not the score of the draft it replaced."""
    coordinator({"dlq_recovery": {"reason": "busy"}})
    served = _serve_queued_work("what's queued right now?", "I'll keep running.")
    assert _reply_was_served_from_a_record(served)
