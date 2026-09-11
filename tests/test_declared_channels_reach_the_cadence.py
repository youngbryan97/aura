"""A subsystem that declares channels must have its publisher on the cadence.

Declaring a channel and writing one are two different acts, and the gap
between them is where the fourteen dispositions sat: declared at boot, written
by a function nothing called, reported for hours as organs that had never
produced a reading. These tests hold the wiring that closes the gap — that
declaring registers the publisher, and that the runtime's telemetry pass runs
what is registered.
"""

from __future__ import annotations

import pytest

from core.fsw.telemetry_samplers import (
    get_sampler_register,
    register_sampler,
    run_registered_samplers,
    samplers_report,
)


@pytest.fixture(autouse=True)
def _keep_the_register_clean():
    register = get_sampler_register()
    before = register.names()
    yield
    for name in register.names():
        if name not in before:
            register.unregister(name)


def test_declaring_the_dispositions_registers_their_publisher() -> None:
    from core.phenomena_wiring import declare_telemetry, sample

    declare_telemetry()
    entry = next(
        row for row in samplers_report()["samplers"] if row["name"] == "phenomena"
    )
    assert entry["owner"] == "core/phenomena_wiring.py"
    assert entry["channels"], "the publisher was registered without its channels"
    assert get_sampler_register()._samplers["phenomena"].run is sample


def test_booting_conation_registers_its_publisher() -> None:
    import core.conation.wiring as wiring

    wiring._wired = False
    result = wiring.boot()
    assert result["sampling"] is True
    assert "conation" in get_sampler_register().names()


def test_the_runtime_telemetry_pass_runs_what_is_registered() -> None:
    """The property, not the call site: registering is enough to be run."""
    from core.runtime.foundations import _sample_standard_telemetry

    ran: list[int] = []
    register_sampler("a_test_publisher", lambda: ran.append(1), owner="tests")
    _sample_standard_telemetry()
    assert ran == [1]


def test_a_sampler_that_raises_does_not_stop_the_others() -> None:
    ran: list[str] = []

    def angry() -> None:
        raise ValueError("no reading available")

    register_sampler("angry", angry, owner="tests")
    register_sampler("calm", lambda: ran.append("calm"), owner="tests")
    outcomes = run_registered_samplers()
    assert outcomes["angry"]["ok"] is False
    assert outcomes["calm"]["ok"] is True
    assert ran == ["calm"]
