import asyncio
import json

import pytest

from core.runtime import autonomy_conductor as conductor_module
from core.runtime.autonomy_conductor import AutonomyConductor
from core.runtime.errors import get_degradation_tracker


def test_failed_job_records_degradation_and_keeps_conductor_alive(tmp_path):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()

        def failing_job():
            failing_job.calls += 1
            raise RuntimeError("job unavailable")

        failing_job.calls = 0
        conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")
        conductor.register("repair_cycle", 1, failing_job, run_immediately=True)

        result = await conductor.run_due_once()

        assert failing_job.calls == 1
        assert result["repair_cycle"]["last_status"] == "failed"
        assert result["repair_cycle"]["failures"] == 1
        assert any(
            "kept conductor alive" in record.action
            for record in tracker.recent(subsystem="autonomy_conductor")
        )
        assert (tmp_path / "autonomy.jsonl").read_text(encoding="utf-8").strip()
        tracker.reset()

    asyncio.run(scenario())


def test_non_mapping_job_result_is_preserved_as_value(tmp_path):
    async def scenario():
        conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")
        conductor.register("value_job", 1, lambda: ["alpha", "beta"], run_immediately=True)

        result = await conductor.run_due_once()
        ledger_entry = json.loads((tmp_path / "autonomy.jsonl").read_text(encoding="utf-8"))

        assert result["value_job"]["last_status"] == "ok"
        assert result["value_job"]["last_result"] == {"value": ["alpha", "beta"]}
        assert ledger_entry["job"]["last_result"] == {"value": ["alpha", "beta"]}

    asyncio.run(scenario())


def test_ledger_append_failure_keeps_in_memory_job_status(tmp_path):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        conductor = AutonomyConductor(tmp_path)
        conductor.register("ledger_down", 1, lambda: {"ok": True}, run_immediately=True)

        result = await conductor.run_due_once()

        assert result["ledger_down"]["last_status"] == "ok"
        assert result["ledger_down"]["last_result"] == {"ok": True}
        assert any(
            "ledger append failed" in record.action
            for record in tracker.recent(subsystem="autonomy_conductor")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_start_falls_back_to_asyncio_task_when_task_tracker_fails(monkeypatch, tmp_path):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()

        class TrackerUnavailable:
            def __init__(self):
                self.calls = 0

            def create_task(self, *_args, **_kwargs):
                self.calls += 1
                raise RuntimeError("tracker unavailable")

        task_tracker = TrackerUnavailable()
        monkeypatch.setattr(conductor_module, "get_task_tracker", lambda: task_tracker)
        conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")
        conductor.register("idle", 60, lambda: {}, run_immediately=False)

        await conductor.start()
        await conductor.stop()

        assert task_tracker.calls == 1
        assert conductor._task is not None
        assert any(
            "asyncio task" in record.action
            for record in tracker.recent(subsystem="autonomy_conductor")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_register_rejects_invalid_job_contract(tmp_path):
    conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")

    with pytest.raises(ValueError):
        conductor.register("", 1, lambda: {})
    with pytest.raises(ValueError):
        conductor.register("bad_interval", 0, lambda: {})
    with pytest.raises(TypeError):
        conductor.register("bad_callable", 1, None)
