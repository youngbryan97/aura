"""Asked how hard the machine is working, she has the number.

LIVE, 2026-08-25, typed into the window: "How hard is the machine you run on
working right now? Give me a number you can stand behind." The reply was "I
have 19 stored turns of recent conversation I can read back. So I can't give
you a defensible number for the machine right now — any specific figure would
be invented, and I'm not going to do that."

An unrelated count, then a refusal, while /api/health reported 24% processor
and 57% memory in the same second. Two faults behind it: the health channel
reported status, failing jobs and degradations and never load, so the question
reached no measured channel and something else filled the gap — and the reading
underneath it was fabricated anyway, because psutil's first non-blocking call
in a process returns exactly 0.0.
"""

from __future__ import annotations

import time


def test_the_health_channel_reports_processor_and_memory() -> None:
    from core.introspection.self_evidence import (
        render_self_health_answer,
        resolve_self_health,
    )

    bundle = resolve_self_health()
    load = {reading.channel: reading for reading in bundle.readings}.get("host_load")
    assert load is not None, "the health channel has no load reading at all"
    assert load.present, f"load was not readable: {load.detail}"
    values = dict(load.value or {})
    assert 0.0 <= values["processor_percent"] <= 100.0
    assert 0.0 < values["memory_percent"] <= 100.0, "a host using no memory is not a reading"
    told = render_self_health_answer(bundle)
    assert "Processor" in told and "memory" in told


def test_the_first_processor_reading_is_measured_not_zero() -> None:
    """psutil diffs two samples; the first non-blocking call has nothing to diff.

    Every caller in the tree reads through here, so that zero was reaching the
    window as a measurement. The baseline is seeded at import, which costs
    nothing, and a caller who asks sooner than the counters can answer gets a
    blocking sample rather than a zero meaning "I did not look".
    """
    from core.runtime.resource_observation import (
        _CPU_MINIMUM_WINDOW_S,
        _measured_cpu_percent,
    )

    # Burn a core for the length of the window, so a reading that looks at the
    # counters cannot come back zero. `0.0 <= x <= 100.0` would pass on the
    # fabricated zero, which is the whole thing being guarded against.
    deadline = time.monotonic() + _CPU_MINIMUM_WINDOW_S + 0.2
    while time.monotonic() < deadline:
        pass
    first = _measured_cpu_percent()
    assert first > 0.0, "a busy host read as 0.0% — the counter had no baseline"
    assert first <= 100.0
    # Asked again immediately it repeats the last reading rather than waiting.
    #
    # It blocked for the rest of the window instead, and this function is on
    # the resource path boot polls continuously: half a second per call took
    # the runtime past the launcher's boot deadline and put it in a restart
    # loop. Measuring honestly must not cost the thing being measured.
    started = time.monotonic()
    again = _measured_cpu_percent()
    assert time.monotonic() - started < 0.05, "a repeat read blocked"
    assert again == first

    # A hot loop stays cheap.
    started = time.monotonic()
    for _ in range(200):
        _measured_cpu_percent()
    assert time.monotonic() - started < 1.0, "200 reads took longer than a second"


def test_a_reading_that_was_never_taken_says_so() -> None:
    """The absence has to stay distinguishable from a real zero."""
    from core.introspection.self_evidence import ReadingState, _load_readings

    readings = _load_readings()
    assert readings
    for reading in readings:
        if not reading.present:
            assert reading.state is not ReadingState.READ
            assert reading.detail, "an absence with no reason is indistinguishable from a value"


import pytest


@pytest.mark.parametrize(
    "asked",
    [
        "How hard is the machine you run on working right now? "
        "Give me a number you can stand behind.",
        "how much memory are you using?",
        "how much of the CPU are you eating?",
        "what kind of load is your host under?",
    ],
)
def test_a_question_about_her_host_reaches_the_measured_channel(asked: str) -> None:
    """The machine under her is her subject, whether or not "your" appears.

    "How hard is the machine you run on working right now" named no subject the
    gate recognised, so it reached no measured channel at all.
    """
    from core.introspection.self_evidence import asks_about_own_operational_state

    assert asks_about_own_operational_state(asked) is True


@pytest.mark.parametrize(
    "asked",
    [
        "my deploy is failing",
        "how hard is this problem?",
        "how much memory does a transformer need?",
        "the build machine has been red since Tuesday",
        "one thing you demonstrably do that off-the-shelf assistants can't",
        "what is the capital of Peru",
    ],
)
def test_somebody_elses_machine_is_not_hers(asked: str) -> None:
    """A false positive answers a question nobody asked with a wall of telemetry."""
    from core.introspection.self_evidence import asks_about_own_operational_state

    assert asks_about_own_operational_state(asked) is False
