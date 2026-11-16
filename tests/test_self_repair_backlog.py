"""Self-repair backlog — defect registers become approval-gated repair goals.

Guarantees pinned here: read-only ingestion, idempotency, approval-gated +
shadow plan creation, and fail-closed behavior when no engine is present.
"""
from __future__ import annotations

import asyncio
import json

from core.agency.self_repair_backlog import SelfRepairBacklog


def _register(tmp_path, order=(), real=()):
    path = tmp_path / "defect_register.json"
    path.write_text(
        json.dumps(
            {
                "schema": "aura.test_defect_register.v1",
                "order_dependent": list(order),
                "real_failures": list(real),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parse_register_is_pure(tmp_path):
    reg = _register(tmp_path, order=["tests/test_a.py::test_x"], real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")
    items = backlog.parse_register(reg)
    kinds = {it.kind for it in items}
    assert kinds == {"order_dependence", "real_failure"}
    order_item = next(it for it in items if it.kind == "order_dependence")
    assert "ORDER-DEPENDENCE" in order_item.goal
    assert "Do not weaken the assertion" in order_item.goal
    assert not (tmp_path / "seen.json").exists()  # pure — nothing written


def test_parse_register_rejects_wrong_schema_and_malformed_target_collections(tmp_path):
    wrong_schema = tmp_path / "wrong.json"
    wrong_schema.write_text(
        json.dumps(
            {
                "schema": "aura.test_defect_register.v999",
                "real_failures": ["tests/test_a.py::test_x"],
            }
        ),
        encoding="utf-8",
    )
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps(
            {
                "schema": "aura.test_defect_register.v1",
                "order_dependent": "tests/test_a.py::test_x",
                "real_failures": ["", None, " tests/test_b.py::test_y ", " tests/test_b.py::test_y "],
            }
        ),
        encoding="utf-8",
    )
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    assert backlog.parse_register(wrong_schema) == []
    items = backlog.parse_register(malformed)
    assert [item.target for item in items] == ["tests/test_b.py::test_y"]


def test_dry_run_does_not_acknowledge_defects(tmp_path):
    reg = _register(tmp_path, order=["tests/test_a.py::test_x"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")
    first = backlog.new_items(reg)
    assert len(first) == 1
    results = asyncio.run(backlog.enqueue_repairs(reg, dry_run=True))
    assert results[0]["reason"] == "dry_run"
    assert results[0]["accepted"] is False
    assert backlog.new_items(reg) == first
    assert not backlog.seen_path.exists()


def test_stable_defect_ids(tmp_path):
    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    b1 = SelfRepairBacklog(seen_path=tmp_path / "s1.json").parse_register(reg)
    b2 = SelfRepairBacklog(seen_path=tmp_path / "s2.json").parse_register(reg)
    assert b1[0].defect_id == b2[0].defect_id


def test_enqueue_creates_approval_gated_shadow_plans(tmp_path):
    reg = _register(tmp_path, order=["tests/test_a.py::test_x"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    calls: list[dict] = []

    class _Engine:
        async def execute_goal(self, goal, context=None, is_shadow=False):
            calls.append({"goal": goal, "context": context, "is_shadow": is_shadow})
            from types import SimpleNamespace

            return SimpleNamespace(status="waiting_for_approval")

    results = asyncio.run(
        backlog.enqueue_repairs(reg, task_engine=_Engine(), auto_execute=False)
    )
    assert len(results) == 1
    assert results[0]["created"] is True
    assert calls[0]["is_shadow"] is True
    assert calls[0]["context"]["requires_approval"] is True
    assert calls[0]["context"]["origin"] == "self_repair_backlog"
    persisted = json.loads(backlog.seen_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == "aura.self_repair_seen.v1"
    assert persisted["defect_ids"] == [results[0]["defect_id"]]
    assert results[0]["seen_persisted"] is True


def test_no_engine_keeps_defect_eligible(tmp_path, monkeypatch):
    import core.agency.autonomous_task_engine as autonomous_task_engine

    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")
    monkeypatch.setattr(autonomous_task_engine, "get_task_engine", lambda: None)

    results = asyncio.run(backlog.enqueue_repairs(reg, auto_execute=False))

    assert results and all(r["created"] is False for r in results)
    assert results[0]["reason"] == "no_task_engine"
    assert backlog.new_items(reg)
    assert not backlog.seen_path.exists()


def test_enqueue_auto_executes_safe_repair_requests(tmp_path):
    from core.resilience.autonomous_repair_executor import (
        set_autonomous_repair_executor_for_tests,
    )

    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    class _Executor:
        def __init__(self):
            self.requests = []

        def enqueue_background(self, request):
            self.requests.append(request)
            return {"status": "scheduled", "fingerprint": request.fingerprint}

    executor = _Executor()
    set_autonomous_repair_executor_for_tests(executor)
    try:
        results = asyncio.run(backlog.enqueue_repairs(reg))
    finally:
        set_autonomous_repair_executor_for_tests(None)

    assert len(results) == 1
    assert results[0]["created"] is True
    assert results[0]["auto_execute"] is True
    assert executor.requests[0].subsystem == "self_repair_backlog"
    assert executor.requests[0].context["origin"] == "self_repair_backlog"
    assert backlog.new_items(reg) == []


def test_seen_ledger_loads_legacy_list(tmp_path):
    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    item = SelfRepairBacklog(seen_path=tmp_path / "unused.json").parse_register(reg)[0]
    seen_path = tmp_path / "seen.json"
    seen_path.write_text(json.dumps([item.defect_id]), encoding="utf-8")

    backlog = SelfRepairBacklog(seen_path=seen_path)

    assert backlog.new_items(reg) == []


def test_seen_ledger_write_failure_does_not_mutate_memory(tmp_path, monkeypatch):
    import core.agency.self_repair_backlog as self_repair_backlog

    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    class _Engine:
        async def execute_goal(self, goal, context=None, is_shadow=False):
            from types import SimpleNamespace

            return SimpleNamespace(status="planned")

    class _FailingGateway:
        def ensure_directory(self, path, *, source):
            path.mkdir(parents=True, exist_ok=True)

        def write_text(self, path, text, *, source):
            raise OSError("injected ledger failure")

    monkeypatch.setattr(
        self_repair_backlog,
        "get_file_write_gateway",
        lambda: _FailingGateway(),
    )

    results = asyncio.run(
        backlog.enqueue_repairs(reg, task_engine=_Engine(), auto_execute=False)
    )

    assert results[0]["created"] is True
    assert results[0]["seen_persisted"] is False
    assert backlog.new_items(reg)
    assert backlog._seen == set()


def test_busy_autonomous_executor_keeps_defect_eligible(tmp_path):
    from core.resilience.autonomous_repair_executor import (
        set_autonomous_repair_executor_for_tests,
    )

    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    class _BusyExecutor:
        def enqueue_background(self, request):
            return {"status": "busy", "fingerprint": request.fingerprint}

    set_autonomous_repair_executor_for_tests(_BusyExecutor())
    try:
        results = asyncio.run(backlog.enqueue_repairs(reg))
    finally:
        set_autonomous_repair_executor_for_tests(None)

    assert results[0]["created"] is False
    assert results[0]["accepted"] is False
    assert results[0]["reason"] == "busy"
    assert backlog.new_items(reg)
    assert not backlog.seen_path.exists()


def test_concurrent_ingestion_reserves_each_defect_once(tmp_path):
    from core.resilience.autonomous_repair_executor import (
        set_autonomous_repair_executor_for_tests,
    )

    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    class _Executor:
        def __init__(self):
            self.requests = []

        def enqueue_background(self, request):
            self.requests.append(request)
            return {"status": "scheduled", "fingerprint": request.fingerprint}

    async def _run_both():
        return await asyncio.gather(
            backlog.enqueue_repairs(reg),
            backlog.enqueue_repairs(reg),
        )

    executor = _Executor()
    set_autonomous_repair_executor_for_tests(executor)
    try:
        results = asyncio.run(_run_both())
    finally:
        set_autonomous_repair_executor_for_tests(None)

    assert sorted(len(result) for result in results) == [0, 1]
    assert len(executor.requests) == 1
    assert backlog.new_items(reg) == []


def test_failed_shadow_plan_keeps_defect_eligible(tmp_path):
    reg = _register(tmp_path, order=["tests/test_a.py::test_x"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    class _FailedEngine:
        async def execute_goal(self, goal, context=None, is_shadow=False):
            from types import SimpleNamespace

            return SimpleNamespace(status="failed", succeeded=False)

    results = asyncio.run(
        backlog.enqueue_repairs(
            reg,
            task_engine=_FailedEngine(),
            auto_execute=False,
        )
    )

    assert results[0]["created"] is False
    assert results[0]["accepted"] is False
    assert backlog.new_items(reg)


def test_runner_emits_register_contract(monkeypatch, tmp_path):
    """The chunk runner writes the register the ingestor consumes."""
    from pathlib import Path
    from types import SimpleNamespace

    import tools.run_test_chunks as runner

    order, real = "tests/test_a.py::test_x", "tests/test_b.py::test_y"
    monkeypatch.setattr(runner, "discover_test_files", lambda _: [Path("test_examples.py")])
    monkeypatch.setattr(runner, "working_tree_revision", lambda: "fixed-revision")
    monkeypatch.setattr(runner, "free_memory_gb", lambda: 8.0)
    monkeypatch.setattr(runner, "run_chunk", lambda *a, **kw: (False, "chunk failed", [order, real]))
    replies = iter([0, 1])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=next(replies)))
    register = tmp_path / "defects.json"
    assert runner.main([
        "--defect-register", str(register), "--progress-file", str(tmp_path / "progress.log"),
    ]) == 1

    backlog = SelfRepairBacklog(seen_path=tmp_path / "unwritten.json")
    items = backlog.parse_register(register)
    assert [(item.kind, item.target) for item in items] == [
        ("order_dependence", order), ("real_failure", real),
    ]
    assert not backlog.seen_path.exists()
