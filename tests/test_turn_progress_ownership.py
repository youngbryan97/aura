"""Unrelated activity cannot renew a foreground turn's deadlines."""

import asyncio
import json

import pytest

from core.runtime import turn_progress as progress
from core.runtime.turn_outcome import TurnOutcome, bind_turn


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    progress.forget_progress()
    return now


def test_background_and_other_turn_cannot_renew_foreground(clock):
    foreground = TurnOutcome("foreground")
    background = TurnOutcome("background")
    with bind_turn(foreground):
        progress.note_progress()
    clock[0] += 40.0
    progress.note_progress()
    unbound_tool = progress.tool_started()
    with bind_turn(background):
        progress.note_progress()
        tool = progress.tool_started()
        assert progress.still_producing(within_s=1)
    with bind_turn(foreground):
        assert progress.seconds_since_progress() == 40.0
        assert not progress.still_producing(within_s=1)
    progress.tool_finished(unbound_tool)
    progress.tool_finished(tool)


def test_completion_targets_starting_turn_and_is_idempotent(clock):
    first, second = TurnOutcome("first"), TurnOutcome("second")
    with bind_turn(first):
        tool = progress.tool_started()
    clock[0] += 40.0
    with bind_turn(second):
        progress.tool_finished(tool)
        assert progress.seconds_since_progress() == -1.0
    clock[0] += 40.0
    progress.tool_finished(tool)
    with bind_turn(first):
        assert progress.seconds_since_progress() == 40.0
        assert not progress.still_producing(within_s=1)


def test_finalized_owner_ignores_late_tokens_and_tool_completion(clock):
    owner = TurnOutcome("finished")
    with bind_turn(owner):
        captured = progress.capture_progress()
        tool = progress.tool_started()
        owner.finalize()
    clock[0] += 40.0
    with bind_turn(TurnOutcome("next")):
        progress.note_progress(progress=captured)
        progress.tool_finished(tool)
        assert progress.seconds_since_progress() == -1.0
    assert not progress.still_producing(within_s=1000, progress=captured)


def test_exited_tool_task_cannot_leave_permanent_activity(clock):
    async def run():
        owner = TurnOutcome("orphan")

        async def starts_but_never_reports_completion():
            progress.tool_started()

        with bind_turn(owner):
            await asyncio.create_task(starts_but_never_reports_completion())
            clock[0] += 40.0
            assert not progress.still_producing(within_s=1)
            assert not owner.progress._tools

    asyncio.run(run())


def test_cancelled_tool_keeps_activity_until_cleanup_finishes():
    async def run():
        owner = TurnOutcome("cleanup")
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def tool():
            activity = progress.tool_started()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                progress.tool_finished(activity)

        with bind_turn(owner):
            task = asyncio.create_task(tool())
            await entered.wait()
            task.cancel()
            await cleaning.wait()
            assert progress.still_producing(within_s=0.000001)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not owner.progress._tools

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_mlx_callback_uses_dispatch_owner_not_reader_context(clock, phase):
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient(model_path="/models/test-small")
    owner, reader = TurnOutcome("request"), TurnOutcome("reader")
    with bind_turn(owner):
        client._mark_generation_started("request-1")
    frame = {"id": "request-1", "phase": phase, "tokens_generated": 16,
             "prompt_tokens_processed": 128, "prompt_tokens_total": 512}
    with bind_turn(reader):
        client._record_worker_stream_progress(frame, status="progress", action="generate")
        assert progress.seconds_since_progress() == -1.0
    with bind_turn(owner):
        assert progress.still_producing(within_s=1)
    client._clear_active_generation_tracking()
    clock[0] += 40
    with bind_turn(reader):
        frame.update(tokens_generated=32, prompt_tokens_processed=256)
        client._record_worker_stream_progress(frame, status="progress", action="generate")
        assert progress.seconds_since_progress() == -1.0
    with bind_turn(owner):
        assert progress.seconds_since_progress() == 40


def test_dev_mode_completion_keeps_owner_without_serializing_runtime_handles(clock):
    from core.transparency.dev_mode import DevMode, TransparencyLevel

    async def run():
        dev = DevMode(TransparencyLevel.SILENT)
        owner, completer = TurnOutcome("dispatch"), TurnOutcome("completer")
        with bind_turn(owner):
            trace = await dev.record_tool_execution("file_read", {"path": "/tmp/example"})
        clock[0] += 40
        with bind_turn(completer):
            await dev.complete_tool_execution(trace, {"ok": True})
            assert progress.seconds_since_progress() == -1
        with bind_turn(owner):
            assert progress.seconds_since_progress() == 0
            assert not owner.progress._tools
        assert "_progress_activity" not in trace.to_dict()
        json.dumps(trace.to_dict())

    asyncio.run(run())
