import asyncio
import json

import pytest

from core.runtime import autonomy_conductor as conductor_module
from core.runtime.autonomy_conductor import AutonomyConductor
from core.runtime.errors import get_degradation_tracker


@pytest.mark.parametrize("deferred", [False, True])
def test_ledger_write_yields_without_reporting_early_completion(monkeypatch, tmp_path, deferred):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        entries = []

        class SlowGateway:
            async def append_text_async(self, path, text, **kwargs):
                started.set()
                await release.wait()
                entries.append(json.loads(text))

            def append_text(self, *args, **kwargs):
                pytest.fail("synchronous ledger write on the event loop")

        monkeypatch.setattr(conductor_module, "get_file_write_gateway", SlowGateway)
        conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")
        conductor.register("probe", 1, lambda: {"ok": True}, run_immediately=True)
        monkeypatch.setattr(conductor, "_job_policy_reason", lambda job: "busy" if deferred else "")
        task = asyncio.create_task(conductor.run_due_once())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            assert not task.done()
            assert entries == []
            release.set()
            result = await asyncio.wait_for(task, timeout=2)
            expected = "deferred" if deferred else "ok"
            assert result["probe"]["last_status"] == expected
            assert entries[0]["job"]["last_status"] == expected
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


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


def test_ledger_append_uses_internal_governance_in_strict_mode(monkeypatch, tmp_path):
    async def scenario():
        monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
        conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")
        conductor.register("strict_job", 1, lambda: {"ok": True}, run_immediately=True)

        result = await conductor.run_due_once()
        ledger_entry = json.loads((tmp_path / "autonomy.jsonl").read_text(encoding="utf-8"))

        assert result["strict_job"]["last_status"] == "ok"
        assert ledger_entry["job"]["name"] == "strict_job"
        assert ledger_entry["job"]["last_result"] == {"ok": True}

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


def test_start_fails_closed_when_task_ownership_fails(monkeypatch, tmp_path):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()

        calls = 0

        def task_owner_unavailable(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("task ownership unavailable")

        monkeypatch.setattr(conductor_module, "create_tracked_task", task_owner_unavailable)
        conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")
        conductor.register("idle", 60, lambda: {}, run_immediately=False)

        with pytest.raises(RuntimeError, match="task ownership unavailable"):
            await conductor.start()

        assert calls == 1
        assert conductor._task is None
        assert any(
            "failed closed" in record.action
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


def test_non_immediate_job_becomes_due_after_its_first_interval(tmp_path):
    conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")
    conductor.register("delayed", 60, lambda: {"ok": True}, run_immediately=False)

    job = conductor.jobs["delayed"]

    assert job.due(job.next_eligible_at - 0.001) is False
    assert job.due(job.next_eligible_at) is True


def test_default_conductor_schedules_bounded_internal_deliberation(tmp_path):
    conductor = AutonomyConductor(tmp_path / "autonomy.jsonl")

    conductor.register_defaults()

    job = conductor.jobs["internal_deliberation_cycle"]
    assert job.policy == "research"
    assert job.run_immediately is False
    assert job.interval_s == 30 * 60.0
