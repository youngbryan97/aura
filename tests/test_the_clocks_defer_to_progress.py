"""Every clock in a turn asks the same question: is anything still arriving?

A desktop turn passes through five nested deadlines, each capping the next, so
raising an inner one changes nothing while an outer one is smaller. Raising
all five is the same design with larger numbers, and the number was never the
thing that was wrong: a stopwatch cannot tell a generation that is working
from one that is stuck, and here it was only ever stopping the first kind.

The second kind is caught by watching the output — the first-token ceiling,
the livelock ceiling, and the sentinel that reads what is being written. None
of those asks what time it is.
"""

from __future__ import annotations

import time

from core.runtime.turn_progress import (
    capture_progress,
    forget_progress,
    normal_gap_between_tokens,
    note_progress,
    seconds_since_progress,
    still_producing,
)


def test_nothing_arrived_is_silence_not_slowness() -> None:
    """The first-token ceiling owns that case, and it is a different case."""

    forget_progress()
    assert seconds_since_progress() == -1.0
    assert still_producing(within_s=30.0) is False


def test_a_token_just_arrived_means_the_turn_is_alive() -> None:
    forget_progress()
    note_progress()
    assert still_producing(within_s=30.0) is True
    assert seconds_since_progress() >= 0.0


def test_a_turn_that_has_gone_quiet_is_not_alive() -> None:
    """The case a deadline was standing in for, decided on the real signal."""

    forget_progress()
    note_progress()
    # A window narrower than the gap that has already elapsed.
    time.sleep(0.05)
    assert still_producing(within_s=0.01) is False


def test_the_window_comes_from_the_machine_not_from_a_table() -> None:
    """What counts as a silence is the machine's own decode rate.

    Floored, because the gap before the first token is prefill and can be far
    longer than any gap after it, and because an unmeasured rate must not make
    every pause look like a stall.
    """

    # Nothing measured, or a machine faster than the floor: the floor holds.
    assert normal_gap_between_tokens(0.0) >= 20.0
    assert normal_gap_between_tokens(3.2) == 20.0

    # A slow machine widens it.
    assert normal_gap_between_tokens(64.0) == 64.0

    # And a value that is not a number does not narrow it.
    assert normal_gap_between_tokens("slow") == 20.0


def test_a_bad_window_never_reports_a_live_turn() -> None:
    forget_progress()
    note_progress()
    assert still_producing(within_s=0.0) is False
    assert still_producing(within_s=-5.0) is False
    assert still_producing(within_s="soon") is False


def test_the_cycle_clock_is_held_open_only_for_someone_waiting() -> None:
    """One GPU. A dream cycle does not get to hold it while a person waits."""

    import asyncio

    from core.brain.cognitive_engine import _keep_the_cycle_open_while_it_is_working

    class _Clock:
        def __init__(self) -> None:
            self.reschedules = 0

        def reschedule(self, _when: float) -> None:
            self.reschedules += 1

    async def run(user_facing: bool) -> int:
        clock = _Clock()
        forget_progress()
        note_progress()
        task = asyncio.create_task(
            _keep_the_cycle_open_while_it_is_working(
                clock, ceiling_at=time.monotonic() + 30.0, user_facing=user_facing
            )
        )
        await asyncio.sleep(1.4)
        task.cancel()
        return clock.reschedules

    assert asyncio.run(run(False)) == 0
    assert asyncio.run(run(True)) >= 1


def test_it_stops_at_the_ceiling() -> None:
    """Bounded: held open is not held open forever."""

    import asyncio

    from core.brain.cognitive_engine import _keep_the_cycle_open_while_it_is_working

    class _Clock:
        def __init__(self) -> None:
            self.reschedules = 0

        def reschedule(self, _when: float) -> None:
            self.reschedules += 1

    async def run() -> int:
        clock = _Clock()
        forget_progress()
        note_progress()
        # A ceiling already in the past.
        await _keep_the_cycle_open_while_it_is_working(
            clock, ceiling_at=time.monotonic() - 1.0, user_facing=True
        )
        return clock.reschedules

    assert asyncio.run(run()) == 0


def test_the_memory_floor_is_scaled_to_the_host() -> None:
    """Two thresholds answered the same question and disagreed.

    Whether there is enough memory to generate was checked in two places: the
    prewarm check scaled its numbers to the host, and the one that refuses a
    person's turn used flat ones, two gigabytes and ten points of pressure
    apart. A machine with sixty-four gigabytes and a resident twenty-gigabyte
    model sits nearer these numbers than a smaller one ever does, so the flat
    pair was the one that would fire first on the machine this runs on.
    """

    from pathlib import Path

    source = Path("core/brain/inference_gate.py").read_text()
    refusal_at = source.index("refusing primary foreground generation")
    guard = source.rindex("if (", 0, refusal_at)
    condition = source[guard:refusal_at]

    assert "6.0 if roomy_host else 8.0" in condition, condition
    assert "92.0 if roomy_host else 90.0" in condition, condition

    # The prewarm check it now agrees with.
    assert "6.0 if total_gb >= 60.0 else 10.0" in source
    # And an unknown host falls back to the stricter pair, not the roomier one.
    assert 'admission_snapshot.get("total_gb", 0.0)' in source


def test_the_endpoint_waits_while_the_cortex_is_producing() -> None:
    """The sixth deadline, and it sat below every other one.

    The gate had raised the turn to 557 seconds and this cut the Cortex off at
    150, so raising the others changed nothing: "Endpoint Cortex timed out
    after 150.0s (force_aborted=False)".
    """

    import asyncio

    from core.brain.llm_health_router import _await_while_it_is_working

    async def slow_but_working() -> str:
        for _ in range(6):
            await asyncio.sleep(0.1)
            note_progress()
        return "the answer"

    async def run_user_facing() -> str:
        from core.runtime.turn_outcome import TurnOutcome, bind_turn

        with bind_turn(TurnOutcome("foreground-wait", origin="desktop")):
            note_progress()
            return await _await_while_it_is_working(
                slow_but_working(), budget_s=0.15, user_facing=True,
                person_is_waiting=True,
            )

    assert asyncio.run(run_user_facing()) == "the answer"


def test_background_work_still_gives_up_on_its_budget() -> None:
    """One GPU. A dream cycle does not hold it while somebody waits."""

    import asyncio

    import pytest

    from core.brain.llm_health_router import _await_while_it_is_working

    async def slow_but_working() -> str:
        for _ in range(6):
            await asyncio.sleep(0.1)
            note_progress()
        return "the answer"

    async def run_background() -> str:
        forget_progress()
        note_progress()
        return await _await_while_it_is_working(
            slow_but_working(), budget_s=0.15, user_facing=False
        )

    with pytest.raises(TimeoutError):
        asyncio.run(run_background())


def test_a_silent_endpoint_still_fails() -> None:
    """Which is the case the deadline was standing in for."""

    import asyncio

    import pytest

    from core.brain.llm_health_router import _await_while_it_is_working

    async def never_answers() -> str:
        await asyncio.sleep(30.0)
        return "too late"

    async def run() -> str:
        forget_progress()
        # Nothing has arrived at all, so this is silence rather than slowness.
        return await _await_while_it_is_working(
            never_answers(), budget_s=0.1, user_facing=True
        )

    with pytest.raises(TimeoutError):
        asyncio.run(run())


def test_reading_the_prompt_counts_as_working() -> None:
    """On a long prompt it is the larger half of the turn.

    Counting only decoded tokens made a turn look silent for the whole of
    prefill, so a wait that defers to progress gave up during the one part of
    the turn where nothing could have arrived yet — which is what happened
    live: "Endpoint past its 150.0s budget and still producing; waiting for
    the answer" and then, moments later, "Endpoint Cortex timed out".
    """

    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient(model_path="/models/test-small")
    client._current_request_id = "reading"
    client._current_turn_progress = capture_progress()
    forget_progress()
    client._record_worker_stream_progress(
        {"id": "reading", "phase": "prefill",
         "prompt_tokens_processed": 64, "prompt_tokens_total": 128},
        status="progress", action="generate",
    )
    assert still_producing(within_s=5.0)


def test_the_wall_clock_watchdog_re_arms_while_work_is_arriving() -> None:
    """It exists for a blocked event loop, and a blocked loop reports nothing.

    So the same signal that says a turn is alive is the one that says this
    watchdog is needed, and firing on elapsed time alone could not tell the
    two apart. The case it kept meeting was the healthy one.
    """

    import time as _time

    from core.brain.llm_health_router import _start_endpoint_wall_clock_watchdog

    class _Client:
        def __init__(self) -> None:
            self.aborted = False

    client = _Client()
    forget_progress()
    note_progress()

    fired, aborted, handle = _start_endpoint_wall_clock_watchdog(
        client, reason="test", timeout_s=0.15, user_facing=True
    )
    try:
        # Keep reporting work for longer than the original budget.
        for _ in range(6):
            _time.sleep(0.05)
            note_progress()
        assert fired.is_set() is False
        assert aborted["value"] is False
    finally:
        handle.cancel()


def test_it_still_aborts_a_turn_that_reports_nothing() -> None:
    """The case it was written for survives."""

    import time as _time

    from core.brain.llm_health_router import _start_endpoint_wall_clock_watchdog

    class _Client:
        pass

    forget_progress()
    fired, _aborted, handle = _start_endpoint_wall_clock_watchdog(
        _Client(), reason="test", timeout_s=0.05, user_facing=True
    )
    try:
        _time.sleep(0.4)
        assert fired.is_set() is True
    finally:
        handle.cancel()


def test_background_work_keeps_the_plain_watchdog() -> None:
    import time as _time

    from core.brain.llm_health_router import _start_endpoint_wall_clock_watchdog

    class _Client:
        pass

    forget_progress()
    note_progress()
    fired, _aborted, handle = _start_endpoint_wall_clock_watchdog(
        _Client(), reason="test", timeout_s=0.05, user_facing=False
    )
    try:
        _time.sleep(0.3)
        assert fired.is_set() is True
    finally:
        handle.cancel()


def test_running_a_tool_counts_as_working() -> None:
    """Every deadline defers to progress, and progress was tokens arriving.

    A tool loop stops decoding while the tool runs, so a turn reading three
    files went quiet for thirty-four seconds at a time and its clocks
    concluded it had stopped. Live on 2026-08-28 that ended a ledgerkit turn
    with cognitive_engine_timeout after three successful reads.
    """

    from pathlib import Path

    source = Path("core/transparency/dev_mode.py").read_text()
    start = source.index("async def record_tool_execution")
    complete = source.index("async def complete_tool_execution")
    assert "tool_started" in source[start:complete], "dispatch must open the state"
    assert "tool_finished" in source[complete:complete + 1200], "completion must close it"


def test_a_tool_in_flight_is_working_however_long_it_takes() -> None:
    """Sampling a timestamp is the wrong instrument for an external step.

    A file read that takes twenty-five seconds looks exactly like a worker
    that has died, and live on 2026-08-28 a turn reading three files was
    cancelled for that resemblance. How long a tool takes is a fact about the
    tool, not about the decode rate, so a tool in flight is a state rather
    than a sample.
    """

    import time as _time

    from core.runtime.turn_progress import tool_finished, tool_started

    forget_progress()
    note_progress()
    _time.sleep(0.05)
    assert still_producing(within_s=0.01) is False

    tool = tool_started()
    _time.sleep(0.05)
    assert still_producing(within_s=0.01) is True
    tool_finished(tool)
    assert still_producing(within_s=0.01) is True


def test_two_tools_at_once_both_have_to_finish() -> None:
    from core.runtime.turn_progress import tool_finished, tool_started

    forget_progress()
    first = tool_started()
    second = tool_started()
    tool_finished(first)
    assert still_producing(within_s=0.0001) is True
    tool_finished(second)
    import time as _time

    _time.sleep(0.05)
    assert still_producing(within_s=0.01) is False


def test_a_stray_finish_cannot_drive_the_count_negative() -> None:
    from core.runtime.turn_progress import tool_finished, tool_started

    forget_progress()
    tool_finished(None)
    tool = tool_started()
    tool_finished(tool)
    tool_finished(tool)
    import time as _time

    _time.sleep(0.05)
    assert still_producing(within_s=0.01) is False


def test_the_cycle_clock_holds_a_started_turn_to_its_ceiling_only() -> None:
    """It sits above the lanes that can tell a wedged turn from a working one.

    Extending only while it could see progress kept losing to gaps it cannot
    observe — a tool running, a long prefill, a worker between frames. Live on
    2026-08-28 a ledgerkit turn read three files and died at 139 seconds, in a
    turn where the engine had already logged that it was holding the cycle
    open.

    A silent worker is caught by the first-token ceiling, a livelocked one by
    the livelock ceiling, a looping decode by the sentinel. All three read the
    output. What is left for this clock is the ceiling.
    """

    import asyncio
    import time as _time

    from core.brain.cognitive_engine import _keep_the_cycle_open_while_it_is_working

    class _Clock:
        def __init__(self) -> None:
            self.reschedules = 0

        def reschedule(self, _when: float) -> None:
            self.reschedules += 1

    async def run(*, any_sign_of_work: bool) -> int:
        clock = _Clock()
        forget_progress()
        if any_sign_of_work:
            note_progress()
        task = asyncio.create_task(
            _keep_the_cycle_open_while_it_is_working(
                clock, ceiling_at=_time.monotonic() + 30.0, user_facing=True
            )
        )
        await asyncio.sleep(2.4)
        task.cancel()
        return clock.reschedules

    # A turn that has shown any sign of work is held open, even after the
    # window has gone quiet.
    assert asyncio.run(run(any_sign_of_work=True)) >= 1
    # A turn that has never produced anything is not this clock's to hold.
    assert asyncio.run(run(any_sign_of_work=False)) == 0


def test_a_short_budget_is_not_waited_on_for_minutes() -> None:
    """How far past the budget a wait may go is proportional to the budget.

    It used to be "up to the user-facing ceiling", which is right for a turn
    and catastrophic for a probe: a two-second health check waited eight
    minutes, the inference probe never returned, and the runtime sat in
    CRITICAL with conversation recovering — blockers runtime_required_probes,
    probe:inference, critical:inference_gate.
    """

    import asyncio
    import time as _time

    import pytest

    from core.brain.llm_health_router import _await_while_it_is_working

    async def never_finishes_but_keeps_working() -> str:
        while True:
            await asyncio.sleep(0.05)
            note_progress()

    async def run() -> float:
        forget_progress()
        note_progress()
        started = _time.monotonic()
        with pytest.raises(TimeoutError):
            await _await_while_it_is_working(
                never_finishes_but_keeps_working(), budget_s=0.2, user_facing=True
            )
        return _time.monotonic() - started

    # 0.2s asked for, so at most 0.2 + 3*0.2 plus a slice — not eight minutes.
    assert asyncio.run(run()) < 10.0
