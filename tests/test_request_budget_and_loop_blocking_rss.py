"""The two faults behind the 2026-07-29 "Orca Demo" freeze.

Bryan asked for a folder, three researched articles and a synthesis PDF. The
runtime wedged for 63.5s, the UI and the health endpoint went with it, and the
task came back "Skill error: TimeoutError:. Completed 0/0 steps." Two
independent defects produced that, and each is pinned here.

  * RECORDING A DEGRADATION BLOCKED THE EVENT LOOP. Every flight-recorder
    frame sampled RSS through an observation that walks the whole process
    tree, and psutil reaches a process tree by enumerating every pid on the
    host. record_degradation feeds that path from the loop, so a lag record
    cost a process-table scan, which caused the next lag record: 5.2s became
    63.5s. The frame only ever reads process_rss_bytes.

  * THE NEGOTIATED BUDGET NEVER REACHED THE CLOCK. capability_engine asked
    desktop_task what the request would cost, was told 405s, logged it — and
    BaseSkill went on enforcing its flat declared 180s from the inside, which
    is smaller by construction and so always fired first. The measured kill
    was 188s against a 405s budget.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.runtime.resource_observation import HostResourceObserver
from core.skills.base_skill import SKILL_TIMEOUT_CONTEXT_KEY, BaseSkill


# ─────────────────────────────────────────────────────────────────────────
# Fault 1: the loop-blocking RSS sample
# ─────────────────────────────────────────────────────────────────────────


def test_own_rss_is_reported_without_walking_the_process_tree() -> None:
    """include_process_tree=False used to report an RSS of zero.

    That is why the hot callers passed the default and paid for a full
    process-table scan: asking for only this process got them a lie.
    """
    observer = HostResourceObserver()
    own = observer.memory(include_process_tree=False)

    assert own.process_rss_bytes > 0, "own-RSS mode must report real RSS, not 0"
    assert own.process_tree_rss_bytes >= own.process_rss_bytes


def test_flight_recorder_rss_sample_never_enumerates_host_pids(monkeypatch) -> None:
    """The frame sampler must not reach psutil's pid enumeration.

    This is the exact frame that sat at the top of the 63.5s stall dump:
      record_degradation -> record_event -> record_frame -> _sample_rss_mb
      -> memory() -> Process.children(recursive=True) -> _ppid_map -> pids()
    """
    import psutil

    from core.runtime.flight_recorder import get_flight_recorder

    def _forbidden(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError(
            "flight recorder sampled RSS via a host-wide pid enumeration; "
            "this blocks the event loop from record_degradation"
        )

    # Armed for the sample alone. Left armed until the test's own teardown,
    # it went off in the fixtures that close the test down, which are allowed
    # to list processes, and the test errored on every run.
    with monkeypatch.context() as tripwire:
        tripwire.setattr(psutil, "pids", _forbidden)
        tripwire.setattr(psutil.Process, "children", _forbidden)
        rss_mb = get_flight_recorder()._sample_rss_mb()
    assert rss_mb > 0.0, "the sample must still be a real measurement"


def test_recording_an_event_frame_stays_cheap_enough_for_the_loop() -> None:
    """A degradation record runs on the event loop; it must not cost ms.

    The threshold is deliberately loose. The regression it catches is not a
    slow machine, it is a call that went from microseconds to milliseconds by
    reaching for the process table.
    """
    from core.runtime.flight_recorder import get_flight_recorder

    recorder = get_flight_recorder()
    recorder._sample_rss_mb()  # warm any import/lazy-init cost

    started = time.perf_counter()
    for _ in range(50):
        recorder._sample_rss_mb()
    per_call_ms = (time.perf_counter() - started) / 50 * 1000.0

    assert per_call_ms < 1.0, f"RSS sample cost {per_call_ms:.3f}ms per frame"


# ─────────────────────────────────────────────────────────────────────────
# Fault 2: the budget that was computed, logged, and then ignored
# ─────────────────────────────────────────────────────────────────────────


class _SlowSkill(BaseSkill):
    name = "slow_test_skill"
    description = "sleeps longer than its declared budget"
    timeout_seconds = 2.0

    async def execute(self, params, context=None):  # type: ignore[override]
        await asyncio.sleep(4.0)
        return {"ok": True, "content": "finished the work"}


@pytest.mark.asyncio
async def test_declared_budget_alone_kills_work_that_needs_longer() -> None:
    """The behaviour Bryan hit, kept explicit so the fix has a baseline."""
    result = await _SlowSkill().safe_execute({}, {})

    assert result.get("ok") is False
    assert "TimeoutError" in str(result.get("error"))


@pytest.mark.asyncio
async def test_negotiated_budget_reaches_the_clock_that_cancels_the_work() -> None:
    """A caller that sized the request gets the work it paid for."""
    context = {SKILL_TIMEOUT_CONTEXT_KEY: 8.0}

    result = await _SlowSkill().safe_execute({}, context)

    assert result.get("ok") is True, f"work was cancelled anyway: {result}"
    assert result.get("content") == "finished the work"


@pytest.mark.asyncio
async def test_a_caller_may_lengthen_the_budget_but_never_shorten_it() -> None:
    """The declared value is the skill author's floor.

    Callers that need to constrain a skill do it with their own outer wait,
    where the cancellation is visible as theirs.
    """
    started = time.monotonic()
    result = await _SlowSkill().safe_execute({}, {SKILL_TIMEOUT_CONTEXT_KEY: 0.5})
    elapsed = time.monotonic() - started

    assert result.get("ok") is False
    assert elapsed >= 1.5, f"declared 2.0s floor was shortened to {elapsed:.2f}s"


def test_the_engine_waits_longer_than_the_budget_it_hands_down() -> None:
    """The skill's own timeout must fire first.

    Its failure path carries the step receipts; an outer wait_for cancelling
    the coroutine destroys them and is how a partially-completed desktop task
    became "Completed 0/0 steps".
    """
    from core.capability_engine import _OUTER_TIMEOUT_GRACE_S

    assert _OUTER_TIMEOUT_GRACE_S > 0.0


def test_desktop_task_sizes_the_orca_request_above_its_declared_floor() -> None:
    """The request that started this, end to end through the real sizer."""
    from core.skills.desktop_task import DesktopTaskSkill

    objective = (
        "Create a folder called Orca Demo in my documents folder. Then find 3 "
        "recent articles about orcas online. Read them. And then write a "
        "synthesis with your own opinion into a PDF saved inside that Orca "
        "Demo folder."
    )

    sized = DesktopTaskSkill.timeout_for({"objective": objective})

    assert sized > DesktopTaskSkill.timeout_seconds
    assert sized == pytest.approx(405.0)
