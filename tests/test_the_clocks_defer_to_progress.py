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
        forget_progress()
        note_progress()
        return await _await_while_it_is_working(
            slow_but_working(), budget_s=0.15, user_facing=True
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
