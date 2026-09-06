"""Registered is not a state, and a report cannot say so without asking."""
from __future__ import annotations

import time

from core.runtime.what_a_part_of_her_declares import APart
from core.runtime.which_parts_say_how_they_are import (
    THE_FIVE_THINGS,
    how_the_parts_answer,
    what_a_part_says,
)


class SaysNothing:
    """A service that is constructed, registered, and mute about its life."""


def test_a_part_that_declares_everything_is_complete() -> None:
    part = APart(name="spine", needs=("clock",), authority="runtime")
    part.alive_since = time.time()
    status = what_a_part_says("spine", part)
    assert status.present
    assert status.state == "not started"
    assert status.owner == "runtime"
    assert status.dependencies == ("clock",)
    assert status.silent_about == ()
    assert status.says_everything


def test_a_registered_object_that_says_nothing_is_named_not_counted_as_healthy() -> None:
    status = what_a_part_says("mystery", SaysNothing())
    assert status.present, "it is there"
    assert not status.says_everything, "and it says nothing about being there"
    assert set(status.silent_about) == set(THE_FIVE_THINGS) - {"reason"}


def test_an_absent_service_is_absent_rather_than_silent() -> None:
    status = what_a_part_says("gone", None)
    assert not status.present
    assert not status.says_everything


def test_a_reason_is_owed_only_where_the_state_needs_explaining() -> None:
    running = APart(name="a", needs=("b",), authority="runtime")
    running.alive = type(running.alive)("running")
    running.alive_since = time.time()
    assert "reason" not in what_a_part_says("a", running).silent_about

    refused = APart(name="c", needs=("b",), authority="runtime")
    refused.alive = type(refused.alive)("refused")
    refused.alive_since = time.time()
    assert "reason" in what_a_part_says("c", refused).silent_about
    refused.why_refused = "its dependency never started"
    assert "reason" not in what_a_part_says("c", refused).silent_about


def test_the_report_separates_what_was_asked_from_what_answered() -> None:
    """Zero registered in a bare process is a fact about the process."""
    full = APart(name="x", needs=("y",), authority="runtime")
    full.alive_since = time.time()
    seen = how_the_parts_answer({"x": full, "y": SaysNothing(), "z": None})
    assert seen["asked"] == 3
    assert seen["registered"] == 2, "the absent one is not counted as silent"
    assert seen["say_all_five"] == 1
    assert seen["say_nothing"] == []
    assert seen["silent_about"]["since"] == 1
    assert seen["silent_about"]["owner"] == 1


def test_asking_never_raises_however_the_object_answers() -> None:
    class Angry:
        @property
        def alive(self):
            raise RuntimeError("no")

        def healthy(self):
            raise RuntimeError("no")

    status = what_a_part_says("angry", Angry())
    assert status.present
    assert not status.says_everything


def test_the_live_container_is_asked_without_starting_anything() -> None:
    seen = how_the_parts_answer()
    assert seen["asked"] >= 60, "the canonical service registry"
    assert seen["registered"] <= seen["asked"]
