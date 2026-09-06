"""One lifecycle for an organ, a phase or a service, and who may start it.

AutoGPT gives every component the same lifecycle and orders them by declared
dependencies. Aura starts things in several ways, each knowing its own
dependencies, and none of them shares a way to say when it may start or who
allowed it — so boot order was a fact about the code rather than a statement
anyone made.
"""
from __future__ import annotations

import pytest

from core.runtime.what_a_part_of_her_declares import (
    APart,
    Alive,
    TheSupervisor,
    WhatItNeeds,
)


def _a_part(name, needs=(), authority="state", start=None, stop=None, healthy=None):
    return APart(
        name=name,
        needs=tuple(needs),
        authority=authority,
        start=start or (lambda: None),
        stop=stop or (lambda: None),
        healthy=healthy or (lambda: True),
    )


# ------------------------------------------------------------ declaring


def test_a_part_must_say_what_authority_it_needs():
    """A supervisor that starts anything that asks is not a supervisor."""
    with pytest.raises(ValueError, match="does not say what authority"):
        TheSupervisor().declare(APart(name="anything"))


def test_a_part_must_have_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        TheSupervisor().declare(APart(name="  ", authority="state"))


def test_a_declared_part_satisfies_the_protocol():
    assert isinstance(_a_part("memory"), WhatItNeeds)


# --------------------------------------------------------------- the order


def test_the_order_puts_dependencies_first():
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("affect", needs=("memory",)))
    supervisor.declare(_a_part("memory"))
    order = supervisor.the_order()
    assert order.index("memory") < order.index("affect")


def test_a_cycle_is_named_rather_than_deadlocked():
    """A boot that hangs because two parts wait for each other is a bad hour."""
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("a", needs=("b",)))
    supervisor.declare(_a_part("b", needs=("a",)))
    with pytest.raises(ValueError, match="a waits for b waits for a"):
        supervisor.the_order()


def test_a_dependency_nobody_declared_does_not_stop_the_ordering():
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("affect", needs=("something absent",)))
    assert supervisor.the_order() == ["affect"]


# -------------------------------------------------------------- starting


def test_everything_that_can_start_does():
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("memory"))
    supervisor.declare(_a_part("affect", needs=("memory",)))
    said = supervisor.start_everything()
    assert said["started"] == ["memory", "affect"]
    assert supervisor.report()["running"] == 2


def test_a_part_that_will_not_start_is_told_apart_from_one_that_was_skipped():
    """A boot report that conflates them sends somebody to the wrong file."""
    def angry():
        raise RuntimeError("no device")

    supervisor = TheSupervisor()
    supervisor.declare(_a_part("voice", start=angry))
    supervisor.declare(_a_part("speech", needs=("voice",)))

    said = supervisor.start_everything()
    assert said["refused"] == ["voice"]
    assert said["skipped_because_something_it_needs_did_not_start"] == ["speech"]


def test_a_refusal_says_why():
    def angry():
        raise RuntimeError("no device")

    supervisor = TheSupervisor()
    supervisor.declare(_a_part("voice", start=angry))
    supervisor.start_everything()
    voice = supervisor.report()["each"][0]
    assert voice["alive"] == "refused"
    assert "no device" in voice["why_refused"]


def test_a_skipped_part_says_what_it_was_waiting_on():
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("speech", needs=("voice",)))
    supervisor.start_everything()
    assert "waiting on voice" in supervisor.report()["each"][0]["why_refused"]


def test_a_supervisor_may_be_limited_to_some_authorities():
    supervisor = TheSupervisor(may_start=frozenset({"state"}))
    supervisor.declare(_a_part("memory", authority="state"))
    supervisor.declare(_a_part("voice", authority="outside"))

    said = supervisor.start_everything()
    assert said["started"] == ["memory"]
    assert said["refused"] == ["voice"]
    assert "may not start outside" in supervisor.report()["each"][1]["why_refused"]


# -------------------------------------------------------------- stopping


def test_stopping_is_the_reverse_of_starting():
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("memory"))
    supervisor.declare(_a_part("affect", needs=("memory",)))
    supervisor.start_everything()
    assert supervisor.stop_everything() == ["affect", "memory"]


def test_a_part_that_never_started_is_not_stopped():
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("speech", needs=("voice",)))
    supervisor.start_everything()
    assert supervisor.stop_everything() == []


def test_a_stuck_stop_does_not_stop_the_shutdown():
    def angry():
        raise RuntimeError("will not stop")

    supervisor = TheSupervisor()
    supervisor.declare(_a_part("stuck", stop=angry))
    supervisor.declare(_a_part("fine", needs=("stuck",)))
    supervisor.start_everything()
    assert supervisor.stop_everything() == ["fine", "stuck"]


def test_the_report_carries_health_as_well_as_life():
    supervisor = TheSupervisor()
    supervisor.declare(_a_part("memory", healthy=lambda: False))
    supervisor.start_everything()
    said = supervisor.report()["each"][0]
    assert said["alive"] == "running"
    assert said["healthy"] is False


def test_the_states_are_the_six_and_nothing_else():
    assert {str(one) for one in Alive} == {
        "not started", "starting", "running", "stopping", "stopped", "refused"
    }
