"""Unavailable evidence cannot repeat an effect or become a successful check."""

import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.capabilities.post_action_verifier import PostActionVerifier, VerificationOutcome
from core.container import ServiceContainer
from core.planning.mission_state import Mission, MissionState, MissionStatus
from core.planning.task_graph import TaskGraph, TaskNode, TaskStatus


@pytest.fixture
def state(tmp_path):
    value = MissionState(str(tmp_path / "missions"))
    value._init_db()
    yield value
    value.close()


def install(state, node):
    graph = TaskGraph("test-mission", "create and verify a report")
    graph.add_node(node)
    mission = Mission("test-mission", graph.objective, MissionStatus.ACTIVE, graph)
    state._active_missions[mission.mission_id] = mission
    return mission


@pytest.mark.asyncio
async def test_real_file_verification_survives_restart_without_repeating_effect(state, tmp_path, monkeypatch):
    target = tmp_path / "report.txt"
    expected = "full-content-" * 100
    node = TaskNode("write", "create_text_file", params={"path": str(target), "content": expected},
                    verification="file_has_content", verification_args={"path": str(target), "contains": expected})
    mission = install(state, node)
    original = state._execute_node
    state._execute_node = AsyncMock(wraps=original)
    state._try_recovery = AsyncMock(return_value=False)
    offline = SimpleNamespace(verify=AsyncMock(side_effect=RuntimeError("observer offline")))
    monkeypatch.setattr(ServiceContainer, "get", lambda name, default=None: offline if name == "post_action_verifier" else default)
    await state.advance_mission(mission.mission_id)
    assert target.read_text() == expected
    assert node.status == TaskStatus.AWAITING_VERIFICATION
    assert node.verification_result["checked"] is False
    state._execute_node.assert_awaited_once()
    state._try_recovery.assert_not_awaited()
    state.close()

    restarted = MissionState(str(state._data_dir))
    restarted._init_db()
    restarted._load_active_missions()
    try:
        restored = restarted._active_missions[mission.mission_id].graph.nodes["write"]
        assert restored.params["content"] == expected
        assert restored.verification_args == node.verification_args
        assert restored.result == node.result
        restarted._execute_node = AsyncMock(side_effect=AssertionError("must not repeat write"))
        verifier = PostActionVerifier()
        monkeypatch.setattr(ServiceContainer, "get", lambda name, default=None: verifier if name == "post_action_verifier" else default)
        await restarted.advance_mission(mission.mission_id)
        assert restored.status == TaskStatus.SUCCEEDED
        assert restored.verification_result["conclusive_success"] is True
        assert restored.error == ""
        restarted._execute_node.assert_not_awaited()
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_measured_mismatch_still_reaches_recovery(state, tmp_path, monkeypatch):
    target = tmp_path / "file.txt"
    node = TaskNode("write", "create_text_file", params={"path": str(target), "content": "actual"},
                    verification="file_has_content", verification_args={"path": str(target), "contains": "expected"})
    mission = install(state, node)
    verifier = PostActionVerifier()
    monkeypatch.setattr(ServiceContainer, "get", lambda name, default=None: verifier if name == "post_action_verifier" else default)
    state._try_recovery = AsyncMock(return_value=False)
    await state.advance_mission(mission.mission_id)
    assert node.status == TaskStatus.FAILED
    assert node.verification_result["checked"] is True
    assert node.verification_result["outcome"] == "measured_mismatch"
    state._try_recovery.assert_awaited_once()


@pytest.mark.asyncio
async def test_omitted_check_is_not_measured_success(state):
    node = TaskNode("noop", "noop", verification="true")
    assert await state._verify_node(node)
    assert node.verification_result["outcome"] == "not_requested"
    assert node.verification_result["checked"] is False
    assert node.verification_result["conclusive_success"] is False
    verifier = PostActionVerifier()
    assert (await verifier.verify("true")).success
    assert verifier.get_status()["success_rate"] is None


@pytest.mark.asyncio
async def test_unknown_predicate_and_host_failure_are_unavailable(monkeypatch):
    from core.capabilities import host_automation
    verifier = PostActionVerifier()
    missing = await verifier.verify("unknown")
    assert missing.infrastructure_failed and not missing.checked
    host = SimpleNamespace(get_frontmost_app=AsyncMock(return_value=SimpleNamespace(
        success=False, result="Notes", error="automation permission denied")))
    monkeypatch.setattr(host_automation, "get_host_automation", lambda: host)
    failure = await verifier.verify("app_is_frontmost", {"name": "Notes"})
    assert failure.outcome == VerificationOutcome.UNAVAILABLE
    assert verifier.get_status()["measured_verifications"] == 0


@pytest.mark.asyncio
async def test_failed_observation_process_is_not_a_measured_mismatch(monkeypatch):
    from core.capabilities import post_action_verifier
    process = SimpleNamespace(returncode=1, communicate=AsyncMock(return_value=(b"/expected.png", b"denied")))
    monkeypatch.setattr(post_action_verifier, "get_subprocess_gateway", lambda: SimpleNamespace(
        spawn_async=AsyncMock(return_value=process)))
    result = await PostActionVerifier().verify("wallpaper_is", {"path": "/expected.png"})
    assert result.infrastructure_failed and not result.success and not result.checked
    process.communicate.assert_awaited_once()
    assert "exited 1" in result.evidence


@pytest.mark.asyncio
async def test_cancelled_observation_keeps_receipt_and_can_be_retried(state, monkeypatch):
    node = TaskNode("effect", "test", verification="file_exists", verification_args={"path": "missing"})
    mission = install(state, node)
    state._execute_node = AsyncMock(return_value={"success": True, "receipt_id": "effect-1"})
    entered = asyncio.Event()
    async def waiting(*args):
        entered.set()
        await asyncio.Event().wait()
    verifier = SimpleNamespace(verify=waiting)
    monkeypatch.setattr(ServiceContainer, "get", lambda name, default=None: verifier if name == "post_action_verifier" else default)
    task = asyncio.create_task(state.advance_mission(mission.mission_id))
    await entered.wait()
    assert await state.advance_mission(mission.mission_id) is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert node.status == TaskStatus.AWAITING_VERIFICATION
    assert node.receipt_id == "effect-1" and node.result["success"] is True
    assert not state._advancing
    state._execute_node.assert_awaited_once()


def test_task_persistence_keeps_full_contract_but_public_view_stays_small():
    graph = TaskGraph("m", "goal")
    node = TaskNode("n", "action", params={"secret": "not public"},
                    verification_args={"path": "/exact/path"}, timeout_s=121,
                    result={"payload": "x" * 1000}, status=TaskStatus.AWAITING_VERIFICATION,
                    rollback_action="undo", rollback_params={"exact": [1, 2]},
                    fallback_action="alternate", retry_count=4)
    graph.add_node(node)
    restored = TaskGraph.from_json(graph.to_json()).nodes["n"]
    assert restored == node
    assert "verification_args" not in node.to_dict()
    assert len(node.to_dict()["result"]["payload"]) == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_observation_process_is_reaped_on_timeout_or_cancellation(monkeypatch, cancel):
    from core.capabilities import post_action_verifier
    entered = asyncio.Event()
    async def wait():
        entered.set()
        await asyncio.Event().wait()
    process = SimpleNamespace(communicate=wait)
    cleanup = AsyncMock(return_value=(b"", b""))
    monkeypatch.setattr(post_action_verifier, "_terminate_async_process_group", cleanup)
    task = asyncio.create_task(PostActionVerifier._communicate(process, 10 if cancel else 0.01))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    cleanup.assert_awaited_once_with(process, grace_s=0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_effect", [False, True])
async def test_storage_failure_cannot_repeat_an_effect(state, monkeypatch, after_effect):
    node = TaskNode("effect", "test")
    mission = install(state, node)
    state._execute_node = AsyncMock(return_value={"success": True, "receipt_id": "effect-1"})
    state._verify_node = AsyncMock(return_value=True)
    state._try_recovery = AsyncMock(return_value=False)
    persist = state._persist_mission
    calls = 0
    def failing(value):
        nonlocal calls
        calls += 1
        if calls == (2 if after_effect else 1):
            raise sqlite3.OperationalError("disk full")
        persist(value)
    monkeypatch.setattr(state, "_persist_mission", failing)
    with pytest.raises(sqlite3.OperationalError):
        await state.advance_mission(mission.mission_id)
    assert node.status == (TaskStatus.AWAITING_VERIFICATION if after_effect else TaskStatus.PENDING)
    state._try_recovery.assert_not_awaited()
    state._verify_node.assert_not_awaited()
    await state.advance_mission(mission.mission_id)
    state._execute_node.assert_awaited_once()
    assert node.status == TaskStatus.SUCCEEDED


def test_real_database_error_is_not_hidden(state):
    mission = install(state, TaskNode("effect", "test"))
    state._conn.execute("DROP TABLE missions")
    with pytest.raises(sqlite3.OperationalError):
        state._persist_mission(mission)


def test_registered_observation_and_record_invariants():
    from core.capabilities.post_action_verifier import _verification_observation
    from core.planning.task_graph import _task_record_roundtrip
    _verification_observation()
    _task_record_roundtrip()
