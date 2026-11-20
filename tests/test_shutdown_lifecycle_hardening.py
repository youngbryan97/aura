from __future__ import annotations

import asyncio
import json
import signal
import time
from types import SimpleNamespace

import pytest

from core.ops import process_manager as process_manager_module
from core.ops.process_manager import ManagedProcess, ProcessConfig, ProcessManager, ProcessState
from core.runtime import shutdown_coordinator


def _target() -> None:
    return None


def test_shutdown_latch_is_visible_before_grace_flag_io(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[bool] = []

    def _write_grace_flag(**_kwargs) -> None:
        observed.append(shutdown_coordinator.is_shutdown_requested())

    monkeypatch.setattr(shutdown_coordinator, "_write_grace_flag", _write_grace_flag)

    snapshot = shutdown_coordinator.request_shutdown("first")
    second = shutdown_coordinator.request_shutdown("second")

    assert observed == [True]
    assert snapshot["first_reason"] == "first"
    assert second["first_reason"] == "first"
    assert second["last_reason"] == "second"
    assert second["request_count"] == 2


def test_repeated_shutdown_request_can_only_tighten_process_deadline() -> None:
    first = shutdown_coordinator.request_shutdown("first", deadline_seconds=10.0)
    first_deadline = first["deadline_at_unix"]

    tightened = shutdown_coordinator.request_shutdown("repeat", deadline_seconds=2.0)
    tightened_deadline = tightened["deadline_at_unix"]
    attempted_extension = shutdown_coordinator.request_shutdown(
        "late-extension",
        deadline_seconds=20.0,
    )

    assert tightened["first_reason"] == "first"
    assert tightened["deadline_source"] == "repeat"
    assert tightened["deadline_tighten_count"] == 1
    assert tightened_deadline < first_deadline
    assert attempted_extension["deadline_at_unix"] == tightened_deadline
    assert attempted_extension["deadline_tighten_count"] == 1
    assert 0.0 < attempted_extension["remaining_budget_seconds"] <= 2.0


@pytest.mark.asyncio
async def test_shutdown_coordinator_obeys_process_deadline_and_reports_blocker() -> None:
    coordinator = shutdown_coordinator.ShutdownCoordinator(phases=("actors",))
    entered = asyncio.Event()

    async def _wedged_owner() -> None:
        entered.set()
        await asyncio.sleep(5.0)

    coordinator.register(
        _wedged_owner,
        phase="actors",
        name="wedged-owner",
        timeout=5.0,
    )
    shutdown_coordinator.request_shutdown("bounded-test", deadline_seconds=0.08)

    started = time.monotonic()
    task = asyncio.create_task(coordinator.shutdown())
    await entered.wait()
    status = coordinator.get_status()
    report = await task
    elapsed = time.monotonic() - started

    assert status["progress"]["blocker_owner"] == "actors:wedged-owner"
    assert status["request"]["remaining_budget_seconds"] <= 0.08
    assert elapsed < 0.5
    assert report.clean is False
    assert report.failed_phases == ["actors"]
    assert any(
        item["kind"] == "global_deadline_exhausted"
        for item in report.escalations
    )


@pytest.mark.asyncio
async def test_process_root_marks_single_container_shutdown_owner() -> None:
    from aura_main import _stop_orchestrator_once

    observed_owner: list[str] = []

    class _Orchestrator:
        async def stop(self) -> None:
            observed_owner.append(self._aura_container_shutdown_owner)

    orchestrator = _Orchestrator()

    await _stop_orchestrator_once(
        orchestrator,
        reason="test_root_owner",
        request_global_shutdown=False,
    )

    assert observed_owner == ["graceful_shutdown"]
    assert orchestrator._aura_stop_invoked is True


@pytest.mark.asyncio
async def test_global_asyncio_patch_preserves_explicit_shutdown_task_scope() -> None:
    from core.utils.asyncio_patch import install_asyncio_task_patch
    from core.utils.task_tracker import (
        begin_shutdown_task_creation_scope,
        end_shutdown_task_creation_scope,
    )

    assert install_asyncio_task_patch() is True
    shutdown_coordinator.request_shutdown("asyncio-patch-shutdown-scope")

    async def _cleanup() -> str:
        await asyncio.sleep(0)
        return "cleanup-complete"

    token = begin_shutdown_task_creation_scope()
    try:
        task = asyncio.create_task(_cleanup(), name="explicit-shutdown-cleanup")
    finally:
        end_shutdown_task_creation_scope(token)

    assert await task == "cleanup-complete"
    assert getattr(task, "_aura_shutdown_critical", False) is True


@pytest.mark.asyncio
async def test_shutdown_coordinator_concurrent_and_repeated_calls_run_handlers_once() -> None:
    coordinator = shutdown_coordinator.ShutdownCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def _handler() -> None:
        calls.append("handler")
        entered.set()
        await release.wait()

    coordinator.register(_handler, phase="actors", name="single-owner", timeout=1.0)

    first = asyncio.create_task(coordinator.shutdown(timeout_per_phase=1.0))
    await entered.wait()
    status = coordinator.get_status()
    assert status["running"] is True
    assert status["report"]["current_phase"] == "actors"
    assert status["progress"]["active_handlers"] == ["actors:single-owner"]
    assert status["progress"]["phase_remaining_seconds"] > 0

    second = asyncio.create_task(coordinator.shutdown(timeout_per_phase=1.0))
    await asyncio.sleep(0)
    release.set()
    first_report, second_report = await asyncio.gather(first, second)
    replay_report = await coordinator.shutdown(timeout_per_phase=1.0)

    assert calls == ["handler"]
    assert first_report.clean is True
    assert second_report.clean is True
    assert replay_report.clean is True
    assert replay_report.repeated_call_count == 2
    assert coordinator.get_status()["lifecycle_state"] == "TERMINATED"


@pytest.mark.asyncio
async def test_shutdown_coordinator_rejects_late_handler_registration() -> None:
    coordinator = shutdown_coordinator.ShutdownCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _handler() -> None:
        entered.set()
        await release.wait()

    coordinator.register(_handler, phase="actors", name="owner", timeout=1.0)
    shutdown_task = asyncio.create_task(coordinator.shutdown(timeout_per_phase=1.0))
    await entered.wait()
    with pytest.raises(RuntimeError, match="teardown has started"):
        coordinator.register(lambda: None, phase="actors", name="late")
    release.set()
    assert (await shutdown_task).clean is True


def test_process_manager_does_not_install_signal_handlers_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[tuple[object, object]] = []
    monkeypatch.setattr(
        process_manager_module.signal,
        "signal",
        lambda sig, handler: installed.append((sig, handler)),
    )

    ProcessManager(register_atexit=False)

    assert installed == []


def test_container_refuses_lazy_service_initialization_after_shutdown(
    service_container,
) -> None:
    from core.exceptions import ContainerError

    calls: list[str] = []
    service_container.register(
        "late_service",
        lambda: calls.append("factory") or object(),
        required=False,
    )
    shutdown_coordinator.request_shutdown("unit-test")

    with pytest.raises(ContainerError, match="cannot initialize service"):
        service_container.get("late_service")

    assert service_container.get("late_service", default=None) is None
    assert calls == []


def test_process_manager_atexit_fallback_is_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    monkeypatch.setattr(
        process_manager_module.atexit,
        "register",
        lambda callback: registered.append(callback),
    )

    ProcessManager()
    assert registered == []

    manager = ProcessManager(register_atexit=True)
    assert registered == [manager.cleanup]


def test_process_manager_signal_latches_global_shutdown_before_cleanup() -> None:
    manager = ProcessManager(register_atexit=False)
    observed: list[bool] = []
    manager.cleanup = lambda **_kwargs: observed.append(  # type: ignore[method-assign]
        shutdown_coordinator.is_shutdown_requested()
    )

    manager._signal_handler(signal.SIGTERM, None)

    assert observed == [True]
    assert manager.shutdown_event.is_set()


def test_process_manager_cleanup_runs_once_even_when_stop_event_is_already_set() -> None:
    manager = ProcessManager(register_atexit=False, cleanup_timeout_s=1.0)
    calls: list[tuple[bool, float | None]] = []

    class _Process:
        def stop(self, force: bool = False, *, timeout_s: float | None = None) -> bool:
            calls.append((force, timeout_s))
            return True

    manager.processes["worker"] = _Process()  # type: ignore[assignment]
    manager.shutdown_event.set()

    first = manager.cleanup(timeout_s=1.0)
    second = manager.cleanup(timeout_s=1.0)

    assert first["status"] == "complete"
    assert second == first
    assert calls == [(True, pytest.approx(1.0, abs=0.1))]
    assert manager._cleanup_complete.is_set()


def test_process_manager_monitor_never_restarts_after_global_latch() -> None:
    manager = ProcessManager(register_atexit=False)
    process = SimpleNamespace(
        stats=SimpleNamespace(restarts=0),
        config=SimpleNamespace(max_restarts=1),
        state=ProcessState.RUNNING,
        get_status=lambda: (_ for _ in ()).throw(
            AssertionError("shutdown check must precede status/restart work")
        ),
    )
    manager.processes["worker"] = process

    shutdown_coordinator.request_shutdown("unit-test")
    manager._check_all_processes()

    assert process.state == ProcessState.RUNNING


@pytest.mark.asyncio
async def test_managed_process_start_refuses_before_process_factory_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = ManagedProcess(ProcessConfig(name="late-worker", target=_target))

    def _must_not_construct(*_args, **_kwargs):
        raise AssertionError("process factory reached after shutdown latch")

    monkeypatch.setattr(process_manager_module.mp, "Process", _must_not_construct)
    shutdown_coordinator.request_shutdown("unit-test")

    assert await managed.start() is False
    assert managed.state == ProcessState.STOPPED


def test_actor_spawn_is_reaped_when_shutdown_crosses_process_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.supervisor import tree as tree_module
    from core.supervisor.tree import ActorSpec, SupervisionTree

    class _Pipe:
        def close(self) -> None:
            return None

    class _Process:
        pid = None
        exitcode = None

        def __init__(self) -> None:
            self.alive = False
            self.terminated = False
            self.closed = False

        def start(self) -> None:
            self.alive = True
            shutdown_coordinator.request_shutdown("crossed-actor-start")

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    process = _Process()

    class _Context:
        def get_start_method(self) -> str:
            return "spawn"

        def Pipe(self, *, duplex: bool):  # noqa: N802 - multiprocessing test double
            assert duplex is False
            return _Pipe(), _Pipe()

        def Process(self, **_kwargs):  # noqa: N802 - multiprocessing test double
            return process

    monkeypatch.setattr(tree_module.multiprocessing, "get_context", lambda _kind: _Context())
    tree = SupervisionTree()
    tree.add_actor(ActorSpec(name="worker", entry_point=lambda *_args: None))

    assert tree.start_actor("worker") is None
    assert process.terminated is True
    assert process.closed is True
    assert tree._actors["worker"].process is None
    assert tree._actors["worker"].next_restart_time == 0.0


@pytest.mark.asyncio
async def test_sensory_spawn_is_reaped_when_shutdown_crosses_process_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.senses import sensory_client as sensory_module

    class _Queue:
        def put(self, _item) -> None:
            return None

        def empty(self) -> bool:
            return True

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class _Process:
        pid = 5432

        def __init__(self) -> None:
            self.alive = False
            self.terminated = False

        def start(self) -> None:
            self.alive = True
            shutdown_coordinator.request_shutdown("crossed-sensory-start")

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

    process = _Process()

    class _Context:
        def get_start_method(self) -> str:
            return "spawn"

        def Queue(self):  # noqa: N802 - multiprocessing test double
            return _Queue()

        def Process(self, **_kwargs):  # noqa: N802 - multiprocessing test double
            return process

    monkeypatch.setattr(sensory_module.mp, "get_context", lambda _kind: _Context())
    client = sensory_module.SensoryLocalClient()

    assert await client.start() is False
    assert process.terminated is True
    assert client._process is None


def test_vision_spawn_is_reaped_when_shutdown_crosses_process_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.brain.llm import mlx_vision_client as vision_module

    class _Queue:
        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class _Process:
        pid = 6543
        name = "MLX-Vision-Worker"

        def __init__(self) -> None:
            self.alive = False
            self.terminated = False

        def start(self) -> None:
            self.alive = True
            shutdown_coordinator.request_shutdown("crossed-vision-start")

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            return None

    process = _Process()

    class _Context:
        def get_start_method(self) -> str:
            return "spawn"

        def Queue(self, maxsize: int = 0):  # noqa: N802 - multiprocessing test double
            return _Queue()

        def Process(self, **_kwargs):  # noqa: N802 - multiprocessing test double
            return process

    monkeypatch.setattr(vision_module.mp, "get_context", lambda _kind: _Context())
    client = vision_module.MLXVisionClient("/models/vision-test")

    assert client.start() is False
    assert process.terminated is True
    assert client._process is None


@pytest.mark.asyncio
async def test_shadow_validation_refuses_before_queue_or_process_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.kernel import shadow_kernel

    def _must_not_create(*_args, **_kwargs):
        raise AssertionError("shadow process resources created after shutdown")

    monkeypatch.setattr(shadow_kernel.multiprocessing, "Queue", _must_not_create)
    instance = shadow_kernel.ShadowExecutionPhase.__new__(
        shadow_kernel.ShadowExecutionPhase
    )
    shutdown_coordinator.request_shutdown("unit-test")

    result = await instance._validate_mutation_detailed("x = 1", "result = True")

    assert result == (False, "runtime_shutdown")


@pytest.mark.asyncio
async def test_boot_and_readiness_health_surfaces_report_stopping() -> None:
    from interface.routes import system

    shutdown_coordinator.request_shutdown("health-surface-test")

    boot_payload, boot_status = system._build_boot_health_payload_sync(
        is_gui_proxy=False
    )
    ready_response = await system.readyz(None)

    assert boot_status == 503
    assert boot_payload["status"] == "stopping"
    assert boot_payload["blockers"] == ["runtime_shutdown"]
    assert boot_payload["shutdown"]["request"]["first_reason"] == (
        "health-surface-test"
    )
    assert ready_response.status_code == 503


@pytest.mark.asyncio
async def test_cognitive_daemon_stop_latches_before_owner_cleanup_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from core.ops import daemon as daemon_module

    observed: list[bool] = []

    class _Orchestrator:
        async def stop(self) -> None:
            observed.append(shutdown_coordinator.is_shutdown_requested())

    monkeypatch.setattr(daemon_module, "DAEMON_PID_FILE", tmp_path / "daemon.pid")
    monkeypatch.setattr(daemon_module, "DAEMON_SOCKET", tmp_path / "daemon.sock")
    daemon = daemon_module.CognitiveDaemon()
    daemon.orchestrator = _Orchestrator()

    await daemon.stop()
    await daemon.stop()

    assert observed == [True]
    assert daemon.orchestrator is None
    assert daemon._stopped is True
    assert daemon._stop_event.is_set()


@pytest.mark.asyncio
async def test_cognitive_daemon_reaps_socket_created_across_shutdown_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from core.ops import daemon as daemon_module

    class _Server:
        def __init__(self) -> None:
            self.closed = False
            self.waited = False

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    server = _Server()

    async def _start_unix_server(*_args, **_kwargs):
        shutdown_coordinator.request_shutdown("crossed-daemon-socket-start")
        return server

    monkeypatch.setattr(daemon_module, "DAEMON_SOCKET", tmp_path / "daemon.sock")
    monkeypatch.setattr(daemon_module.asyncio, "start_unix_server", _start_unix_server)
    daemon = daemon_module.CognitiveDaemon()

    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        await daemon._start_socket_server()

    assert server.closed is True
    assert server.waited is True
    assert daemon._socket_server is None


@pytest.mark.asyncio
async def test_shutdown_verdict_is_atomic_and_distinguishes_interim_from_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    target = tmp_path / "shutdown_report.json"
    monkeypatch.setenv("AURA_SHUTDOWN_REPORT_PATH", str(target))
    coordinator = shutdown_coordinator.ShutdownCoordinator(phases=("actors",))

    report = await coordinator.shutdown()
    interim = json.loads(target.read_text(encoding="utf-8"))
    final = shutdown_coordinator.publish_shutdown_verdict(
        coordinator_report=report,
        container_report={"clean": True},
        runtime_hygiene_report={"clean": True},
        stage="unit_test_complete",
        final=True,
    )

    assert interim["stage"] == "coordinator"
    assert interim["final"] is False
    assert final["verdict"] == {"clean": True, "blockers": []}
    assert final["history_artifact_path"]
    assert len(list((tmp_path / "shutdown_history").glob("*.json"))) == 1
    persisted = json.loads(target.read_text(encoding="utf-8"))
    assert persisted["stage"] == "unit_test_complete"
    assert persisted["final"] is True


def test_shutdown_verdict_blocks_on_surviving_crossed_resource(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "AURA_SHUTDOWN_REPORT_PATH",
        str(tmp_path / "shutdown_report.json"),
    )
    shutdown_coordinator.record_shutdown_admission_event(
        "unit-test-worker",
        resource_kind="process",
        outcome="survived",
        detail="forced reap failed",
    )

    payload = shutdown_coordinator.publish_shutdown_verdict(
        coordinator_report={"clean": True},
        container_report={"clean": True},
        runtime_hygiene_report={"clean": True},
        stage="unit_test_complete",
        final=True,
    )

    assert payload["verdict"]["clean"] is False
    assert "shutdown_resurrection_survived" in payload["verdict"]["blockers"]


@pytest.mark.asyncio
async def test_shutdown_verdict_blocks_on_unfinished_non_owner_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from core.utils.task_tracker import get_task_tracker

    monkeypatch.setenv(
        "AURA_SHUTDOWN_REPORT_PATH",
        str(tmp_path / "shutdown_report.json"),
    )
    release = asyncio.Event()

    async def _linger() -> None:
        await release.wait()

    lingering = get_task_tracker().create_task(
        _linger(),
        name="lingering-finalizer",
        allow_during_shutdown=True,
    )
    await asyncio.sleep(0)

    payload = shutdown_coordinator.publish_shutdown_verdict(
        coordinator_report={"clean": True},
        container_report={"clean": True},
        runtime_hygiene_report={"clean": True},
        stage="unit_test_complete",
        final=True,
    )

    assert payload["verdict"]["clean"] is False
    assert "tasks_remaining_after_final_sweep" in payload["verdict"]["blockers"]
    assert payload["components"]["final_tasks"]["count"] >= 1
    release.set()
    await lingering


@pytest.mark.asyncio
async def test_shutdown_verdict_blocks_on_tracker_blind_pending_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A pending task the TRACKER cannot see must still block the final
    verdict: the loop's own task table is the second witness (2026-07-12
    register: clean=True over a live task with an empty tracker census)."""
    monkeypatch.setenv(
        "AURA_SHUTDOWN_REPORT_PATH",
        str(tmp_path / "shutdown_report.json"),
    )
    release = asyncio.Event()

    async def _invisible() -> None:
        await release.wait()

    # Deliberately NOT tracker-created: plain loop task.
    ghost = asyncio.get_running_loop().create_task(_invisible())
    await asyncio.sleep(0)

    payload = shutdown_coordinator.publish_shutdown_verdict(
        coordinator_report={"clean": True},
        container_report={"clean": True},
        runtime_hygiene_report={"clean": True},
        stage="unit_test_tracker_blind",
        final=True,
    )

    assert payload["verdict"]["clean"] is False
    assert "tasks_remaining_after_final_sweep" in payload["verdict"]["blockers"]
    assert payload["components"]["final_tasks"]["effective_count"] >= 1
    release.set()
    await ghost


@pytest.mark.asyncio
async def test_task_tracker_final_sweep_preserves_teardown_owner() -> None:
    from core.utils.task_tracker import TaskTracker

    tracker = TaskTracker(name="shutdown-owner-test")
    release = asyncio.Event()

    async def _hold() -> None:
        await release.wait()

    ordinary = tracker.create_task(_hold(), name="ordinary")
    teardown_owner = tracker.create_task(
        _hold(),
        name="teardown-owner",
        allow_during_shutdown=True,
    )
    await asyncio.sleep(0)

    sweep = await tracker.shutdown(timeout=0.2)

    assert ordinary.cancelled()
    assert teardown_owner.done() is False
    assert sweep["clean"] is True
    assert sweep["shutdown_critical_active"] == 1
    release.set()
    await teardown_owner


@pytest.mark.asyncio
async def test_runtime_hygiene_closes_owned_resources_in_reverse_order() -> None:
    from core.runtime.runtime_hygiene import RuntimeHygieneManager

    events: list[str] = []

    class _Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(self.name)

    hygiene = RuntimeHygieneManager()
    first = _Resource("first")
    second = _Resource("second")
    hygiene.register_shutdown_resource(
        first,
        kind="listener",
        name="first",
        source="unit-test",
    )
    hygiene.register_shutdown_resource(
        second,
        kind="listener",
        name="second",
        source="unit-test",
    )

    report = await hygiene._cleanup_shutdown_resources()

    assert report["clean"] is True
    assert report["completed"] == 2
    assert events == ["second", "first"]


@pytest.mark.asyncio
async def test_service_container_shutdown_failure_survives_repeated_calls(
    service_container,
) -> None:
    class _BrokenService:
        async def on_stop_async(self) -> None:
            raise RuntimeError("owner cleanup failed")

    service_container.register_instance("broken_shutdown_owner", _BrokenService())

    first = await service_container.shutdown(
        hook_timeout_s=0.1,
        total_timeout_s=0.5,
    )
    replay = await service_container.shutdown(
        hook_timeout_s=0.1,
        total_timeout_s=0.5,
    )

    assert first["clean"] is False
    assert replay["clean"] is False
    assert "broken_shutdown_owner:on_stop_async" in replay["failed_hooks"]
    assert replay["shutdown_pass_count"] == 2


@pytest.mark.asyncio
async def test_service_container_defers_excluded_finalizer_until_root_pass(
    service_container,
) -> None:
    stopped: list[str] = []

    class _Owner:
        async def on_stop_async(self) -> None:
            stopped.append("owner")

    class _Finalizer:
        async def on_stop_async(self) -> None:
            stopped.append("finalizer")

    service_container.register_instance("ordinary_owner", _Owner())
    service_container.register_instance("runtime_hygiene", _Finalizer())

    intermediate = await service_container.shutdown(exclude={"runtime_hygiene"})

    assert intermediate["clean"] is True
    assert stopped == ["owner"]
    assert service_container.get("runtime_hygiene") is not None

    final = await service_container.shutdown()

    assert final["clean"] is True
    assert stopped == ["owner", "finalizer"]
    assert "ordinary_owner" in final["completed_services"]
    assert "runtime_hygiene" in final["completed_services"]
    assert final["deferred_services"] == ["runtime_hygiene"]


@pytest.mark.asyncio
async def test_service_container_uses_conventional_zero_argument_stop_hook(
    service_container,
) -> None:
    stopped: list[str] = []

    class _Service:
        async def stop(self) -> None:
            stopped.append("stopped")

    service_container.register_instance("conventional_stop_owner", _Service())

    report = await service_container.shutdown()

    assert report["clean"] is True
    assert stopped == ["stopped"]
    assert "conventional_stop_owner" in report["completed_services"]


@pytest.mark.asyncio
async def test_service_container_reports_incompatible_stop_hook(
    service_container,
) -> None:
    class _Service:
        def stop(self, required_reason: str) -> None:
            del required_reason

    service_container.register_instance("bad_stop_signature", _Service())

    report = await service_container.shutdown()

    assert report["clean"] is False
    assert report["failed_hooks"]["bad_stop_signature:container"].startswith(
        "no_zero_argument_shutdown_hook"
    )


@pytest.mark.asyncio
async def test_service_container_coalesces_duplicate_singleton_aliases(
    service_container,
) -> None:
    calls: list[str] = []

    class _Service:
        async def stop(self) -> None:
            calls.append("stop")

    instance = _Service()
    service_container.register_instance("canonical_runtime", instance)
    service_container.register_instance("compatibility_runtime", instance)

    report = await service_container.shutdown()

    assert report["clean"] is True
    assert calls == ["stop"]
    assert report["coalesced_aliases"] == {
        "canonical_runtime": "compatibility_runtime"
    }


@pytest.mark.asyncio
async def test_service_container_records_owned_hook_cancellation_and_continues(
    service_container,
) -> None:
    calls: list[str] = []

    class _LaterOwner:
        async def stop(self) -> None:
            calls.append("later")

    class _CancellingOwner:
        async def stop(self) -> None:
            calls.append("cancelled")
            raise asyncio.CancelledError()

    service_container.register_instance("later_owner", _LaterOwner())
    service_container.register_instance("cancelling_owner", _CancellingOwner())

    report = await service_container.shutdown()

    assert calls == ["cancelled", "later"]
    assert report["clean"] is False
    assert report["failed_hooks"] == {
        "cancelling_owner:stop": "hook_cancelled_without_container_cancellation"
    }


@pytest.mark.asyncio
async def test_service_container_propagates_root_shutdown_cancellation(
    service_container,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingOwner:
        async def stop(self) -> None:
            entered.set()
            await release.wait()

    service_container.register_instance("blocking_owner", _BlockingOwner())
    shutdown_task = asyncio.create_task(
        service_container.shutdown(hook_timeout_s=5.0, total_timeout_s=5.0)
    )
    await entered.wait()

    shutdown_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown_task


@pytest.mark.asyncio
async def test_aura_kernel_shutdown_ownership_is_sticky_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.kernel.aura_kernel import AuraKernel

    calls: list[bool] = []

    async def _shutdown_impl(
        _kernel: AuraKernel,
        *,
        finalize_process_runtime: bool,
    ) -> None:
        calls.append(finalize_process_runtime)

    monkeypatch.setattr(AuraKernel, "_shutdown_impl", _shutdown_impl)
    kernel = object.__new__(AuraKernel)
    kernel._shutdown_lock = asyncio.Lock()
    kernel._shutdown_complete = False
    kernel._shutdown_process_runtime_owner = None

    await kernel.shutdown(finalize_process_runtime=False)
    await kernel.shutdown(finalize_process_runtime=True)
    await kernel.on_stop_async()

    assert calls == [False]


@pytest.mark.asyncio
async def test_shutdown_coordinator_bounds_synchronous_handler() -> None:
    import threading
    import time

    release = threading.Event()
    worker_daemon: list[bool] = []
    coordinator = shutdown_coordinator.ShutdownCoordinator(phases=("actors",))

    def _blocking_handler() -> None:
        worker_daemon.append(threading.current_thread().daemon)
        release.wait(0.5)

    coordinator.register(
        _blocking_handler,
        phase="actors",
        name="blocking-sync-handler",
        timeout=0.05,
    )

    started = time.monotonic()
    report = await coordinator.shutdown()
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.25
    assert worker_daemon == [True]
    assert report.clean is False
    assert report.handler_statuses["actors:blocking-sync-handler"] == "failed"
    assert any(item["kind"] == "handler_timeout" for item in report.escalations)


def test_blocking_shutdown_callable_times_out_on_daemon_worker() -> None:
    import threading
    import time

    from core.runtime.shutdown_execution import run_sync_shutdown_callable_blocking

    release = threading.Event()
    worker_daemon: list[bool] = []

    def _block() -> None:
        worker_daemon.append(threading.current_thread().daemon)
        release.wait(0.5)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="exceeded"):
        run_sync_shutdown_callable_blocking(
            _block,
            timeout_s=0.03,
            name="unit-test-blocking-cleanup",
        )
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.2
    assert worker_daemon == [True]


@pytest.mark.asyncio
async def test_runtime_hygiene_suppresses_raw_tasks_and_threads_after_latch() -> None:
    import threading

    from core.runtime.runtime_hygiene import RuntimeHygieneManager
    from core.utils.task_tracker import (
        begin_shutdown_resource_creation_scope,
        begin_shutdown_task_creation_scope,
        end_shutdown_resource_creation_scope,
        end_shutdown_task_creation_scope,
    )

    hygiene = RuntimeHygieneManager()
    loop = asyncio.get_running_loop()
    await hygiene.start(loop)
    await asyncio.to_thread(lambda: None)
    ran: list[str] = []
    suppression_cleanup: list[str] = []

    async def _late_task() -> None:
        ran.append("task")

    shutdown_coordinator.request_shutdown("raw-start-suppression-test")
    task = asyncio.create_task(_late_task(), name="raw-late-task")
    await task

    blocked_thread = threading.Thread(
        target=lambda: ran.append("blocked-thread"),
        name="raw-late-thread",
    )
    blocked_thread._aura_shutdown_suppressed_cleanup = lambda: suppression_cleanup.append(
        "closed"
    )
    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        blocked_thread.start()

    await loop.shutdown_default_executor(timeout=1.0)

    task_token = begin_shutdown_task_creation_scope()
    try:
        async def _cleanup_task() -> None:
            ran.append("cleanup-task")

        cleanup_task = asyncio.create_task(_cleanup_task(), name="cleanup-task")
        task_scope_thread = threading.Thread(
            target=lambda: ran.append("task-scope-thread"),
            name="task-scope-thread",
        )
        with pytest.raises(RuntimeError, match="runtime_shutdown"):
            task_scope_thread.start()
    finally:
        end_shutdown_task_creation_scope(task_token)
    await cleanup_task

    resource_token = begin_shutdown_resource_creation_scope()
    try:
        cleanup_thread = threading.Thread(
            target=lambda: ran.append("cleanup-thread"),
            name="cleanup-thread",
        )
        cleanup_thread.start()
        cleanup_thread.join(timeout=1.0)
    finally:
        end_shutdown_resource_creation_scope(resource_token)
    await hygiene.stop()

    assert ran == ["cleanup-task", "cleanup-thread"]
    assert suppression_cleanup == ["closed"]
    assert getattr(task, "_aura_shutdown_suppressed", False) is True
    admission = shutdown_coordinator.shutdown_admission_snapshot()
    assert admission["counts"]["suppressed"] >= 2
    assert admission["counts"]["allowed_teardown"] >= 1


def test_self_modification_error_forward_closes_coroutine_after_latch() -> None:
    import inspect

    from core.self_modification.self_modification_engine import (
        _schedule_background_coro,
    )

    async def _late_error_forward() -> None:
        return None

    coro = _late_error_forward()
    shutdown_coordinator.request_shutdown("self-modification-forward-test")

    _schedule_background_coro(coro, label="unit-test")

    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_runtime_hygiene_treats_executor_timeout_warning_as_unclean() -> None:
    import warnings

    from core.runtime.runtime_hygiene import RuntimeHygieneManager

    class Worker:
        name = "Aura_Cognition_1"
        ident = 12345
        daemon = False

        @staticmethod
        def is_alive() -> bool:
            return True

    class Executor:
        _threads = {Worker()}

    class Loop:
        _default_executor = Executor()

        @staticmethod
        async def shutdown_default_executor(
            *,
            timeout: float,  # noqa: ASYNC109 - mirrors asyncio API.
        ) -> None:
            warnings.warn(
                f"executor still joining after {timeout}",
                RuntimeWarning,
                stacklevel=2,
            )

    hygiene = RuntimeHygieneManager()
    report = await hygiene._shutdown_default_executor(Loop())

    assert report["attempted"] is True
    assert report["clean"] is False
    assert report["warnings"] == ["executor still joining after 2.0"]
    assert report["workers_after"] == [
        {
            "name": "Aura_Cognition_1",
            "ident": 12345,
            "daemon": False,
        }
    ]


def test_runtime_hygiene_shutdown_thread_summary_includes_stack_and_authority() -> None:
    import threading

    from core.runtime.runtime_hygiene import RuntimeHygieneManager

    release = threading.Event()
    started = threading.Event()

    def _wait_for_release() -> None:
        started.set()
        release.wait(1.0)

    hygiene = RuntimeHygieneManager()
    worker = threading.Thread(target=_wait_for_release, name="stack-evidence-worker")
    worker._aura_shutdown_critical = True
    hygiene._register_thread(worker, source="unit-test")
    worker.start()
    assert started.wait(0.2)
    try:
        summary = hygiene._thread_summary(include_stacks=True)
    finally:
        release.set()
        worker.join(timeout=0.5)

    sample = next(
        item for item in summary["active_sample"] if item["name"] == "stack-evidence-worker"
    )
    assert sample["ident"] == worker.ident
    assert sample["shutdown_critical"] is True
    assert sample["source"] == "unit-test"
    assert any(frame["function"] == "_wait_for_release" for frame in sample["stack"])


@pytest.mark.asyncio
async def test_metrics_exporter_owns_http_server_and_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.resilience import metrics_exporter as metrics_module

    events: list[str] = []

    class _Thread:
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, *, timeout: float) -> None:
            events.append(f"join:{timeout}")

    class _Server:
        def __init__(self, thread: _Thread) -> None:
            self.thread = thread

        def shutdown(self) -> None:
            events.append("shutdown")
            self.thread.alive = False

        def server_close(self) -> None:
            events.append("close")

    thread = _Thread()
    server = _Server(thread)
    monkeypatch.setattr(
        metrics_module,
        "start_http_server",
        lambda _port: (server, thread),
    )
    exporter = metrics_module.MetricsExporter(port=19090)

    await exporter.start()
    await exporter.stop()

    assert exporter._http_server is None
    assert exporter._http_thread is None
    assert events == ["shutdown", "close"]


def test_service_container_suppresses_registration_after_shutdown_latch() -> None:
    """Suppression is SOFT (lived 2026-07-12): the hard raise detonated the
    aura_now telemetry republish INSIDE kernel.shutdown(), aborting teardown
    before the vault closed — leaked non-daemon aiosqlite workers held
    interpreter exit for the chunk runner's full 2400s, twice. Refusing new
    work must never abort the teardown that refuses it: the registration is
    recorded as suppressed and no-ops."""
    from core.container import ServiceContainer

    shutdown_coordinator.request_shutdown("service-registration-test")

    ServiceContainer.register_instance("late-service", object())
    assert ServiceContainer.get("late-service", default=None) is None

    admission = shutdown_coordinator.shutdown_admission_snapshot()
    assert admission["counts"]["suppressed"] == 1


def test_production_shutdown_latch_cannot_be_cleared_in_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    shutdown_coordinator.request_shutdown("monotonic-latch-test")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AURA_ALLOW_IN_PROCESS_SHUTDOWN_RESET", raising=False)

    with pytest.raises(RuntimeError, match="monotonic"):
        shutdown_coordinator.clear_shutdown_request()


@pytest.mark.asyncio
async def test_sensory_queue_timeout_finishes_task_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import queue

    from core.senses.sensory_client import SensoryLocalClient
    from core.supervisor import registry as registry_module

    updates: list[dict[str, object]] = []

    class _Registry:
        def register_task(self, *_args, **_kwargs) -> str:
            return "sensory-task"

        def update_task(self, _task_id: str, **kwargs) -> None:
            updates.append(dict(kwargs))

    class _Process:
        @staticmethod
        def is_alive() -> bool:
            return True

    class _RequestQueue:
        @staticmethod
        def put(_payload) -> None:
            return None

    class _ResponseQueue:
        @staticmethod
        def get(*, timeout: float):
            raise queue.Empty from None

    monkeypatch.setattr(registry_module, "get_task_registry", lambda: _Registry())
    client = SensoryLocalClient()
    client._process = _Process()
    client._req_q = _RequestQueue()
    client._res_q = _ResponseQueue()

    assert await client._send_command("ping", timeout=0.01, auto_restart=False) is False
    assert updates[-1]["error"] == ""
    assert str(updates[-1]["status"]).lower().endswith("failed")


def test_vision_inference_timeout_clears_request_and_stops_worker() -> None:
    from core.brain.llm.mlx_vision_client import MLXVisionClient

    stopped: list[bool] = []

    class _RequestQueue:
        @staticmethod
        def put(_payload, *, timeout: float) -> None:
            return None

    class _Process:
        @staticmethod
        def is_alive() -> bool:
            return True

    async def _start() -> bool:
        return True

    client = MLXVisionClient("/models/test-vision")
    client.start_async = _start  # type: ignore[method-assign]
    client.stop = lambda: stopped.append(True)  # type: ignore[method-assign]
    client.readiness_status = lambda: {  # type: ignore[method-assign]
        "ready": True,
        "reason": "ready",
    }
    client._req_q = _RequestQueue()
    client._process = _Process()

    with pytest.raises(TimeoutError, match="inference timed out"):
        client.see("describe", "aW1hZ2U=", timeout_s=0.01)

    assert stopped == [True]
    assert client._pending_requests == {}


def test_a_real_model_worker_is_still_refused_in_a_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard lets a test's stand-in process through and still refuses a real one (tests/conftest.py)."""
    import multiprocessing

    from core.brain.llm.mlx_vision_client import _mlx_vision_worker_loop
    from core.runtime.subprocess_gateway import (
        AcceleratorCapability,
        ProcessRole,
        PythonProcessSpec,
        get_subprocess_gateway,
    )

    spec = PythonProcessSpec(target=_mlx_vision_worker_loop, args=(), source="test", name="MLX-Vision-Worker",
                             role=ProcessRole.MODEL_WORKER, accelerator_capability=AcceleratorCapability.MODEL)
    with pytest.raises(RuntimeError, match="started a real model worker"):
        get_subprocess_gateway().spawn_python_process(spec, context=multiprocessing.get_context("spawn"))
    with pytest.raises(RuntimeError, match="started a real model worker"):
        get_subprocess_gateway().spawn_python_process(spec)
