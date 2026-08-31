"""Foreground clocks renew owned work without abandoning cancellation cleanup."""

import asyncio

import pytest

from core.brain import llm_health_router as router
from core.runtime import turn_progress as progress
from core.runtime.turn_outcome import TurnOutcome, bind_turn


@pytest.mark.asyncio
async def test_owned_answer_finishes_beyond_former_hard_ceiling(monkeypatch):
    from core.brain.llm import mlx_client
    from core.runtime import response_policy

    monkeypatch.setattr(response_policy, "USER_FACING_COMPLETION_DEADLINE_MAX_S", 0.02)
    monkeypatch.setattr(mlx_client, "longest_a_turn_may_take", lambda **kw: 0.02)

    async def answer():
        for _ in range(8):
            progress.note_progress()
            await asyncio.sleep(0.01)
        return "complete answer"

    with bind_turn(TurnOutcome("long-answer", origin="desktop")):
        result = await router._await_while_it_is_working(
            answer(), budget_s=0.01, user_facing=True, person_is_waiting=True
        )
    assert result == "complete answer"


@pytest.mark.asyncio
async def test_other_turn_cannot_extend_a_silent_foreground():
    foreground = TurnOutcome("silent", origin="desktop")
    other = TurnOutcome("other", origin="autonomous")
    cleaned = asyncio.Event()

    async def unrelated_work():
        with bind_turn(other):
            try:
                while True:
                    progress.note_progress()
                    await asyncio.sleep(0.005)
            finally:
                cleaned.set()

    with bind_turn(foreground):
        with pytest.raises(TimeoutError):
            await router._await_while_it_is_working(
                unrelated_work(), budget_s=0.025, user_facing=True, person_is_waiting=True
            )
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_probe_stays_bounded_despite_owned_progress(monkeypatch):
    from core.runtime import response_policy

    monkeypatch.setattr(response_policy, "USER_FACING_COMPLETION_DEADLINE_MAX_S", 0.04)
    cleaned = asyncio.Event()

    async def probe():
        try:
            while True:
                progress.note_progress()
                await asyncio.sleep(0.005)
        finally:
            cleaned.set()

    with bind_turn(TurnOutcome("probe", origin="desktop")):
        with pytest.raises(TimeoutError):
            await router._await_while_it_is_working(
                probe(), budget_s=0.01, user_facing=True, person_is_waiting=False
            )
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_user_cancellation_awaits_endpoint_cleanup():
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def endpoint():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    with bind_turn(TurnOutcome("cancel", origin="desktop")):
        waiter = asyncio.create_task(router._await_while_it_is_working(
            endpoint(), budget_s=30, user_facing=True, person_is_waiting=True
        ))
        await started.wait()
        waiter.cancel()
        await asyncio.wait_for(cleaning.wait(), timeout=1)
        assert not waiter.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter


@pytest.mark.asyncio
async def test_endpoint_timeout_is_not_misread_as_wait_expiry():
    async def endpoint():
        progress.note_progress()
        raise TimeoutError("worker_fault")

    with bind_turn(TurnOutcome("fault", origin="desktop")):
        with pytest.raises(TimeoutError, match="worker_fault"):
            await router._await_while_it_is_working(
                endpoint(), budget_s=30, user_facing=True, person_is_waiting=True
            )


def test_watchdog_thread_reads_captured_owner_past_ceiling(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    callbacks = []
    aborts = []

    class Timer:
        def __init__(self, interval, function):
            self.function = function

        def start(self):
            callbacks.append(self.function)

        def cancel(self):
            pass

    monkeypatch.setattr(router.threading, "Timer", Timer)
    monkeypatch.setattr(router, "_force_abort_endpoint_client",
                        lambda client, **kw: aborts.append(client) or True)
    owner = TurnOutcome("watched", origin="desktop")
    with bind_turn(owner):
        progress.note_progress()
        fired, _, handle = router._start_endpoint_wall_clock_watchdog(
            "client", reason="test", timeout_s=1,
            user_facing=True, person_is_waiting=True,
        )
    now[0] = 1000.0
    with bind_turn(owner):
        progress.note_progress()
    with bind_turn(TurnOutcome("timer-context")):
        callbacks.pop(0)()
    assert not fired.is_set()
    assert not aborts
    assert len(callbacks) == 1
    now[0] += 40
    callbacks.pop(0)()
    assert fired.is_set()
    assert aborts == ["client"]
    handle.cancel()


@pytest.mark.asyncio
async def test_cycle_renews_past_estimate_without_shortening_initial_window():
    from core.brain.cognitive_engine import _keep_the_cycle_open_while_it_is_working

    class Clock:
        def __init__(self):
            self.deadline = asyncio.get_running_loop().time() + 100
            self.rescheduled = []

        def when(self):
            return self.deadline

        def reschedule(self, deadline):
            self.rescheduled.append(deadline)
            self.deadline = deadline

    clock = Clock()
    initial = clock.deadline
    with bind_turn(TurnOutcome("cycle", origin="desktop")):
        progress.note_progress()
        keeper = asyncio.create_task(_keep_the_cycle_open_while_it_is_working(
            clock, ceiling_at=0.0, user_facing=True
        ))
        try:
            await asyncio.sleep(1.1)
        finally:
            keeper.cancel()
            await keeper
    assert clock.rescheduled
    assert min(clock.rescheduled) >= initial


@pytest.mark.asyncio
async def test_cycle_renewal_reaches_commit_after_original_estimate_expired():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from core.brain.cognitive_engine import (
        _commit_the_thought_with_retries,
        _keep_the_cycle_open_while_it_is_working,
    )

    loop = asyncio.get_running_loop()
    original = loop.time() - 1
    context = {"cognitive_cycle_deadline_monotonic": original}
    state = object()
    repository = SimpleNamespace(commit=AsyncMock())

    class Clock:
        deadline = original

        def when(self):
            return self.deadline

        def reschedule(self, deadline):
            self.deadline = deadline

    with bind_turn(TurnOutcome("commit-after-renewal", origin="desktop")):
        progress.note_progress()
        keeper = asyncio.create_task(_keep_the_cycle_open_while_it_is_working(
            Clock(), ceiling_at=original, user_facing=True, runtime_context=context
        ))
        try:
            await asyncio.sleep(1.1)
        finally:
            keeper.cancel()
            await keeper
        outcome, committed = await _commit_the_thought_with_retries(
            commit_outcome="not_attempted", cycle_deadline_at=original,
            runtime_context=context, is_test_run=False, max_retries=1,
            origin="desktop", pre_turn_cognition={},
            self=SimpleNamespace(state_repository=repository),
            should_bypass_commit=False, state=state, temp_state=state,
        )
    assert context["cognitive_cycle_deadline_monotonic"] > loop.time()
    assert outcome == "committed"
    assert committed is state
    repository.commit.assert_awaited_once_with(state, "cognitive_cycle")


@pytest.mark.parametrize("foreground,owned,started,allowed", [
    (True, True, True, True),
    (True, True, False, False),
    (True, False, True, False),
    (False, True, True, False),
    (False, False, False, False),
])
def test_worker_repair_uses_completion_ownership_not_expired_estimate(
    foreground, owned, started, allowed,
):
    from core.brain.llm.mlx_worker import _generation_deadline_open

    job = {"deadline_unix": 1.0, "foreground_request": foreground,
           "progress_owned_completion": owned}
    assert _generation_deadline_open(job, started=started) is allowed


def test_worker_initial_decode_still_requires_unexpired_admission():
    import time

    from core.brain.llm.mlx_worker import _generation_deadline_open

    assert _generation_deadline_open({"deadline_unix": time.time() + 60}, started=False)
    assert _generation_deadline_open({}, started=False)


@pytest.mark.asyncio
async def test_expired_cycle_without_renewal_does_not_commit(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from core.brain import cognitive_engine as engine

    monkeypatch.setattr(engine, "record_degradation", lambda *a, **kw: None)
    repository = SimpleNamespace(commit=AsyncMock())
    expired = engine.time.monotonic() - 1
    outcome, _ = await engine._commit_the_thought_with_retries(
        commit_outcome="not_attempted", cycle_deadline_at=expired,
        runtime_context={"cognitive_cycle_deadline_monotonic": expired},
        is_test_run=False, max_retries=1, origin="desktop", pre_turn_cognition={},
        self=SimpleNamespace(state_repository=repository), should_bypass_commit=False,
        state=object(), temp_state=object(),
    )
    assert outcome == "cycle_deadline_expired"
    repository.commit.assert_not_awaited()
