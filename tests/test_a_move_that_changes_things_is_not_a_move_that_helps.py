"""Working has meant "the view is different", which is a poor test of helping.

That is the honest test that an action HAPPENED, and it is the wrong test of
whether it got anywhere. A move that shuffles things scores exactly as well as
one that builds, so a record made from it comes to prefer whatever changes the
screen most reliably rather than whatever makes progress.

LIVE 2026-08-26: playing toward a 512 tile, she pressed the same direction
over and over — it kept "working" every single time — and the commentary in
companion mode repeated one line for fifty-two seconds.
"""
from __future__ import annotations

import inspect

import pytest

from core.agency.deliberate_action import (
    ActionOption,
    Attempt,
    Deliberation,
    Expectation,
    Verdict,
    confirm,
    made_progress,
)


class _Graph:
    def __init__(self):
        self.written = []

    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, action, context, outcome, success):
        self.written.append({"action": action, "outcome": outcome, "success": success})


def test_closer_is_measurable_when_the_goal_names_a_value():
    assert made_progress("2 4 8 16", "2 4 8 32", "256") is True
    assert made_progress("2 4 8 16", "4 8 2 16", "256") is False


def test_nothing_is_claimed_where_there_is_nothing_to_measure():
    """A goal that names no value has no scale to be closer on, and inventing
    one would be worse than having none."""
    assert made_progress("a page", "another page", "Deploy succeeded") is None
    assert made_progress("", "", "256") is None


def _moved(name: str = "right") -> Deliberation:
    return Deliberation(
        goal="reach 512",
        situation="2 4 8 16",
        chosen=ActionOption(
            name=name, expectation=Expectation(changed=True, describes="the view to be different")
        ),
    )


def test_a_move_that_shuffles_is_not_recorded_as_one_that_worked():
    graph = _Graph()
    attempt = confirm(_moved(), "2 4 8 16", "4 8 2 16", graph=graph, toward="512")
    assert attempt.verdict.held, "the view really did change"
    assert attempt.progressed is False
    assert graph.written[0]["success"] is False
    assert "got no closer" in graph.written[0]["outcome"]


def test_a_move_that_builds_is_recorded_as_one_that_worked():
    graph = _Graph()
    attempt = confirm(_moved(), "2 4 8 16", "2 4 8 32", graph=graph, toward="512")
    assert attempt.verdict.held
    assert attempt.progressed is True
    assert graph.written[0]["success"] is True


def test_where_progress_cannot_be_measured_the_old_test_stands():
    """Nothing that works today changes: with no scale, changing the view is
    still the honest evidence that the action happened."""
    graph = _Graph()
    attempt = confirm(_moved(), "a page", "another page", graph=graph, toward="Deploy succeeded")
    assert attempt.progressed is None
    assert graph.written[0]["success"] is attempt.verdict.held


def test_the_difference_is_said_in_her_own_evidence():
    held = Verdict(held=True, observed_change=True)
    assert "got closer" in Attempt("right", "a shift", held, progressed=True).as_evidence()
    assert "no closer" in Attempt("right", "a shift", held, progressed=False).as_evidence()


def test_the_loop_tells_the_check_what_it_is_aiming_at():
    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "toward=success_when" in source
