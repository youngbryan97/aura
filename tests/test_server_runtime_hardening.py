import asyncio
import errno
import gc
import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.autonomy.sleep_trigger import AutonomousSleepTrigger
from core.backup import BackupManager
from core.bus.local_pipe_bus import LocalPipeBus
from core.bus.shared_mem_bus import SharedMemoryTransport
from core.intent_gate import IntentClassifierQueue, RouteKind
from core.kernel.bridge import AffectBridge
from core.memory.memory_facade import MemoryFacade
from core.memory_synthesizer import MemorySynthesizer
from core.mind_tick import MindTick
from core.motivation.engine import MotivationEngine
from core.motivation.intention import DriveType, Intention
from core.orchestrator.main import RobustOrchestrator
from core.proactive_communication import ProactiveCommunicationManager
from core.process_manager import ManagedProcess, ProcessConfig, ProcessManager
from core.resilience.integrity_monitor import IntegrityReport, SystemIntegrityMonitor
from core.resilience.stability_guardian import StabilityGuardian
from core.state.aura_state import AuraState
from core.state.state_repository import StateRepository, get_state_shm_size_bytes
from core.utils.concurrency import RobustLock
from core.utils.task_tracker import TaskTracker


@pytest.mark.asyncio
async def test_memory_facade_add_and_query_memory_compat():
    records = []

    class VectorStub:
        def __init__(self):
            self._store = records

        def add_memory(self, content, metadata=None):
            records.append(
                {
                    "id": f"id-{len(records)+1}",
                    "content": content,
                    "metadata": dict(metadata or {}),
                }
            )
            return True

    facade = MemoryFacade()
    facade._vector = VectorStub()

    assert await facade.add_memory("Journal line", {"type": "narrative_journal", "timestamp": 10.0}) is True
    assert await facade.add_memory("Other line", {"type": "other", "timestamp": 11.0}) is True

    result = await facade.query_memory("type:narrative_journal", limit=5)

    assert len(result) == 1
    assert result[0]["text"] == "Journal line"
    assert result[0]["metadata"]["type"] == "narrative_journal"


def test_affect_bridge_receive_qualia_echo_updates_kernel_state():
    state = AuraState()
    state.affect.emotions["joy"] = 0.8
    state.affect.dominant_emotion = "joy"
    initial_arousal = state.affect.arousal
    initial_heart_rate = state.affect.physiology["heart_rate"]

    bridge = AffectBridge(SimpleNamespace(state=state))
    bridge.receive_qualia_echo(q_norm=0.82, pri=0.91, trend=0.07)

    assert state.affect.emotions["awe"] > 0.0
    assert state.affect.emotions["anticipation"] > 0.5
    assert state.affect.arousal > initial_arousal
    assert state.affect.physiology["heart_rate"] > initial_heart_rate


def test_stability_guardian_treats_stale_tick_history_as_idle_not_degraded():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))
    guardian._tick_times.append((time.time() - 600.0, 120000.0))
    guardian._last_tick_at = time.time() - 600.0

    result = guardian._check_tick_rate()

    assert result.healthy is True
    assert "idle" in result.message.lower()


@pytest.mark.asyncio
async def test_api_health_exposes_liquid_state_and_soma_payloads(service_container, monkeypatch):
    from interface import server as server_module

    monkeypatch.setattr(server_module, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "foreground_tier": "local",
        },
    )
    monkeypatch.setattr(
        server_module,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "status": "ready",
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "boot_phase": "kernel_ready",
                "conversation_lane": conversation_lane or {},
            },
            200,
        ),
    )
    monkeypatch.setattr(
        server_module,
        "get_runtime_state",
        lambda: {
            "state": {
                "affect": {
                    "energy": 88,
                    "curiosity": 11,
                    "frustration": 7,
                    "mood": "JOY",
                }
            },
            "sha256": "abc",
            "signature": "sig",
        },
    )
    monkeypatch.setattr(
        server_module.psutil,
        "cpu_percent",
        lambda interval=None, percpu=False: [12.0, 18.0] if percpu else 15.0,
    )
    monkeypatch.setattr(
        server_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=42.0),
    )

    class _Soma:
        async def pulse(self):
            return {
                "thermal_load": 0.31,
                "resource_anxiety": 0.62,
                "vitality": 0.84,
            }

        def get_status(self):
            return {
                "soma": {
                    "thermal_load": 0.31,
                    "resource_anxiety": 0.62,
                    "vitality": 0.84,
                }
            }

    service_container.register_instance(
        "orchestrator",
        SimpleNamespace(status=SimpleNamespace(initialized=True, running=True, cycle_count=5, start_time=time.time() - 5)),
        required=False,
    )
    service_container.register_instance(
        "liquid_state",
        SimpleNamespace(get_status=lambda: {"energy": 81, "curiosity": 23, "frustration": 9, "focus": 55, "mood": "NEUTRAL"}),
        required=False,
    )
    service_container.register_instance(
        "homeostasis",
        SimpleNamespace(get_health=lambda: {"integrity": 1.0, "persistence": 1.0, "will_to_live": 0.91}),
        required=False,
    )
    service_container.register_instance("soma", _Soma(), required=False)
    service_container.register_instance("social", SimpleNamespace(get_health=lambda: {"depth": 0.0}), required=False)
    service_container.register_instance("moral", SimpleNamespace(get_health=lambda: {"integrity": 0.95}), required=False)

    response = await server_module.api_health(SimpleNamespace(headers={}))
    payload = json.loads(response.body)

    assert payload["liquid_state"]["energy"] == 81.0
    assert payload["liquid_state"]["curiosity"] == 23.0
    assert payload["liquid_state"]["frustration"] == 9.0
    assert payload["liquid_state"]["confidence"] == 91.0
    assert payload["soma"]["thermal_load"] == 0.31
    assert payload["soma"]["resource_anxiety"] == 0.62
    assert payload["soma"]["vitality"] == 0.84


def test_conversation_lane_standby_helper_and_runtime_capabilities_align():
    from interface import server as server_module

    standby_lane = {
        "conversation_ready": False,
        "state": "cold",
        "warmup_attempted": False,
        "warmup_in_flight": False,
    }
    warming_lane = dict(standby_lane, warmup_attempted=True)

    assert server_module._conversation_lane_is_standby(standby_lane) is True
    assert server_module._conversation_lane_is_standby(warming_lane) is False
    assert server_module._collect_runtime_capabilities(standby_lane)["local_runtime"] == "standby"


def test_collect_stability_details_treats_cold_standby_as_healthy(service_container, monkeypatch):
    from interface import server as server_module

    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "warmup_attempted": False,
            "warmup_in_flight": False,
        },
    )

    details = server_module._collect_stability_details()

    assert details["healthy"] is True
    assert details["status"] == "healthy"
    assert details["active_issues"] == []


def test_state_repository_treats_prefixed_user_origin_as_foreground_for_db_snapshot():
    repo = StateRepository(is_vault_owner=True)
    state = AuraState()
    state.transition_origin = "routing_user"

    assert repo._should_use_bounded_db_snapshot(state, "user_turn") is False


def test_aura_state_snapshot_hot_ignores_dynamic_root_pending_intents():
    state = AuraState()
    state.pending_intents = [{"type": "legacy_root_intent"}]
    state.cognition.pending_intents.append({"type": "cognitive_intent"})

    snapshot = state.snapshot_hot()

    assert "pending_intents" not in snapshot
    assert snapshot["cognition"].pending_intents == [{"type": "cognitive_intent"}]


def test_state_repository_deserialize_migrates_legacy_root_pending_intents():
    repo = StateRepository(is_vault_owner=True)
    payload = json.loads(repo._serialize(AuraState()))
    payload["pending_intents"] = [{"type": "legacy_root_intent"}]
    payload["cognition"].pop("pending_intents", None)

    hydrated = repo._deserialize(json.dumps(payload))

    assert hydrated.cognition.pending_intents == [{"type": "legacy_root_intent"}]


@pytest.mark.asyncio
async def test_api_health_treats_cold_standby_lane_as_ready(service_container, monkeypatch):
    from interface import server as server_module

    standby_lane = {
        "conversation_ready": False,
        "state": "cold",
        "desired_endpoint": "Cortex",
        "foreground_endpoint": None,
        "foreground_tier": "local",
        "warmup_attempted": False,
        "warmup_in_flight": False,
    }

    monkeypatch.setattr(server_module, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_collect_conversation_lane_status", lambda: dict(standby_lane))
    monkeypatch.setattr(
        server_module,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "status": "ready",
                "ready": True,
                "system_ready": True,
                "conversation_ready": False,
                "boot_phase": "kernel_ready",
                "status_message": "Aura is awake. Cortex will warm on first turn.",
                "progress": 100,
                "conversation_lane": conversation_lane or {},
            },
            200,
        ),
    )
    monkeypatch.setattr(
        server_module,
        "get_runtime_state",
        lambda: {"state": {"affect": {}}, "sha256": "abc", "signature": "sig"},
    )
    monkeypatch.setattr(server_module.psutil, "cpu_percent", lambda interval=None, percpu=False: [10.0, 12.0] if percpu else 11.0)
    monkeypatch.setattr(server_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=33.0))

    service_container.register_instance(
        "orchestrator",
        SimpleNamespace(status=SimpleNamespace(initialized=True, running=True, cycle_count=3, start_time=time.time() - 5)),
        required=False,
    )

    response = await server_module.api_health(SimpleNamespace(headers={}))
    payload = json.loads(response.body)

    assert payload["status"] == "ok"
    assert payload["conversation_lane"]["state"] == "cold"
    assert payload["boot"]["status"] == "ready"


@pytest.mark.asyncio
async def test_desktop_access_summary_reuses_cached_probe_result(monkeypatch):
    import core.security.permission_guard as permission_guard_module
    from interface.routes import system as system_routes

    calls = []

    class _Guard:
        async def check_permission(self, ptype, force=False):
            calls.append((ptype.name, force))
            return {"granted": False, "status": "denied", "guidance": ""}

    monkeypatch.setattr(permission_guard_module, "get_permission_guard", lambda: _Guard())
    monkeypatch.setattr("core.skills._pyautogui_runtime.get_pyautogui", lambda: (None, None))
    monkeypatch.setattr(system_routes, "_DESKTOP_ACCESS_CACHE_TTL_S", 60.0)

    original_cache = dict(system_routes._desktop_access_cache)
    system_routes._desktop_access_cache["captured_at"] = 0.0
    system_routes._desktop_access_cache["payload"] = None
    try:
        first = await system_routes._collect_desktop_access_summary()
        second = await system_routes._collect_desktop_access_summary()
    finally:
        system_routes._desktop_access_cache.update(original_cache)

    assert first["overall_status"] == "blocked"
    assert second["overall_status"] == "blocked"
    assert calls == [
        ("SCREEN", False),
        ("ACCESSIBILITY", False),
        ("AUTOMATION", False),
    ]


@pytest.mark.asyncio
async def test_api_memory_episodic_clamps_limit_and_supports_offset(service_container):
    from interface import server as server_module

    class _Episode:
        def __init__(self, idx: int):
            self.idx = idx

        def to_dict(self):
            return {"context": f"ctx-{self.idx}", "timestamp": float(self.idx)}

    class _EpisodicMemory:
        def __init__(self):
            self.calls = []

        def recall_recent(self, limit: int = 10):
            self.calls.append(limit)
            return [_Episode(i) for i in range(limit + 2)]

    episodic = _EpisodicMemory()
    service_container.register_instance("episodic_memory", episodic, required=False)

    response = await server_module.api_memory_episodic(limit=500, offset=3)
    payload = json.loads(response.body)

    assert episodic.calls == [203]
    assert payload["limit"] == 200
    assert payload["offset"] == 3
    assert payload["count"] == 200
    assert payload["has_more"] is True
    assert payload["items"][0]["context"] == "ctx-3"


@pytest.mark.asyncio
async def test_shared_memory_transport_write_serialized_round_trip():
    transport = SharedMemoryTransport(f"st_rt_{uuid.uuid4().hex[:8]}", size=4096)

    await transport.create()
    try:
        transport.write_serialized(json.dumps({"state_id": "st_test", "version": 7}))
        result = await transport.read()
        assert result == {"state_id": "st_test", "version": 7}
    finally:
        transport.close()


@pytest.mark.asyncio
async def test_shared_memory_transport_falls_back_to_file_backed_mmap(monkeypatch, tmp_path):
    class _DeniedSharedMemory:
        def __init__(self, *args, **kwargs):
            raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setenv("AURA_SHM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setattr("core.bus.shared_mem_bus.shared_memory.SharedMemory", _DeniedSharedMemory)

    writer = SharedMemoryTransport(f"st_rt_fb_{uuid.uuid4().hex[:8]}", size=4096)
    reader = SharedMemoryTransport(writer.name)

    await writer.create()
    try:
        writer.write_serialized(json.dumps({"state_id": "st_fallback", "version": 9}))
        await reader.attach()
        try:
            result = await reader.read()
            assert result == {"state_id": "st_fallback", "version": 9}
            assert writer._backend == "file_mmap"
            assert reader._backend == "file_mmap"
        finally:
            reader.close()
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_shared_memory_transport_attach_suppresses_non_owner_resource_tracker_registration(monkeypatch):
    closed = []
    register_calls = []

    class _AttachedSharedMemory:
        def __init__(self, name):
            assert name == "st_rt_attach"
            from core.bus.shared_mem_bus import resource_tracker

            resource_tracker.register("/st_rt_attach", "shared_memory")
            self.size = 4096
            self.buf = None

        def close(self):
            closed.append("closed")

    monkeypatch.setattr(
        "core.bus.shared_mem_bus.shared_memory.SharedMemory",
        _AttachedSharedMemory,
    )
    monkeypatch.setattr(
        "core.bus.shared_mem_bus.resource_tracker.register",
        lambda name, kind: register_calls.append((name, kind)),
    )

    transport = SharedMemoryTransport("st_rt_attach", size=4096)
    await transport.attach()
    transport.close()

    assert register_calls == []
    assert closed == ["closed"]


def test_shared_memory_transport_cross_process_attach_exits_without_leak_warning():
    code = textwrap.dedent(
        """
        import asyncio
        import multiprocessing as mp
        import time

        from core.bus.shared_mem_bus import SharedMemoryTransport

        SEGMENT = "st_rt_cross_process"

        def child():
            async def main():
                owner = SharedMemoryTransport(SEGMENT, size=4096)
                await owner.create()
                owner.write_serialized('{"state_id":"cross","version":1}')
                time.sleep(0.5)
                owner.close()

            asyncio.run(main())

        if __name__ == "__main__":
            ctx = mp.get_context("fork")
            proc = ctx.Process(target=child)
            proc.start()
            time.sleep(0.15)

            async def parent():
                reader = SharedMemoryTransport(SEGMENT, size=4096)
                await reader.attach()
                payload = await reader.read()
                assert payload["version"] == 1
                reader.close()

            asyncio.run(parent())
            proc.join(5)
            if proc.exitcode != 0:
                raise SystemExit(proc.exitcode or 1)
        """
    )

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{os.getcwd()}{os.pathsep}{pythonpath}" if pythonpath else os.getcwd()
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "leaked shared_memory objects" not in proc.stderr


def test_shared_memory_transport_owner_finalizer_unlinks_unclosed_segment():
    events = []

    class _OwnedSharedMemory:
        def unlink(self):
            events.append("unlink")

        def close(self):
            events.append("close")

    transport = SharedMemoryTransport("st_rt_finalize", size=4096)
    transport.shm = _OwnedSharedMemory()
    transport._is_owner = True
    transport._arm_owner_finalizer()
    finalizer = transport._owner_finalizer

    del transport
    gc.collect()

    assert finalizer is not None
    assert finalizer.alive is False
    assert events == ["unlink", "close"]


@pytest.mark.asyncio
async def test_local_pipe_bus_offloads_large_payloads_via_serialized_shm(monkeypatch):
    writes = []

    class _FakeShm:
        def __init__(self, name, size):
            self.name = name
            self.size = size

        async def create(self):
            return None

        def write_serialized(self, payload):
            writes.append(payload)

    monkeypatch.setattr("core.bus.local_pipe_bus.SharedMemoryTransport", _FakeShm)

    bus = LocalPipeBus(start_reader=False)
    payload = {"blob": "x" * (LocalPipeBus._SHM_OFFLOAD_THRESHOLD_BYTES + 1024)}

    prepared = await bus._prepare_payload_for_transport(payload)

    assert prepared["__shm__"].startswith("shm_msg_")
    assert len(writes) == 1
    assert isinstance(writes[0], str)


@pytest.mark.asyncio
async def test_local_pipe_bus_retains_offloaded_shm_until_cleanup(monkeypatch):
    closed = []

    class _FakeShm:
        def __init__(self, name, size):
            self.name = name
            self.size = size

        async def create(self):
            return None

        def write_serialized(self, payload):
            return None

        def close(self):
            closed.append(self.name)

    monkeypatch.setattr("core.bus.local_pipe_bus.SharedMemoryTransport", _FakeShm)

    bus = LocalPipeBus(start_reader=False)
    payload = {"blob": "x" * (LocalPipeBus._SHM_OFFLOAD_THRESHOLD_BYTES + 1024)}

    prepared = await bus._prepare_payload_for_transport(payload)

    assert prepared["__shm__"] in bus._outbound_shm_segments
    assert closed == []

    bus._cleanup_expired_shm_segments(force=True)

    assert closed == [prepared["__shm__"]]


@pytest.mark.asyncio
async def test_state_repository_prefers_bounded_hot_snapshot_before_marker():
    writes = []

    class _FakeShm:
        payload_capacity = 4096

        def write_serialized(self, payload):
            writes.append(payload)

    repo = StateRepository(is_vault_owner=True)
    repo._shm = _FakeShm()
    state = AuraState()
    state.state_id = "st_hot"
    state.version = 12
    state.cold.long_term_memory = ["x" * 4096 for _ in range(12)]

    mode = await repo._sync_to_shm(state, repo._serialize(state))

    assert mode == "hot"
    assert len(writes) == 1
    snapshot = json.loads(writes[0].decode("utf-8"))
    assert snapshot["_transport_snapshot_kind"] == "hot"
    assert "cold" not in snapshot
    assert snapshot["version"] == 12
    assert repo.get_runtime_status()["last_shm_write_mode"] == "hot"


@pytest.mark.asyncio
async def test_state_repository_uses_overflow_marker_when_state_exceeds_shm_capacity():
    writes = []

    class _FakeShm:
        payload_capacity = 128

        def write_serialized(self, payload):
            writes.append(payload)

    repo = StateRepository(is_vault_owner=True)
    repo._shm = _FakeShm()
    state = AuraState()
    state.state_id = "st_overflow"
    state.version = 11

    mode = await repo._sync_to_shm(state, json.dumps({"payload": "x" * 256}))

    assert mode == "marker"
    assert len(writes) == 1
    marker = json.loads(writes[0].decode("utf-8"))
    assert marker["_state_overflow"] is True
    assert marker["state_id"] == "st_overflow"
    assert marker["version"] == 11
    assert repo.get_runtime_status()["last_shm_write_mode"] == "marker"


@pytest.mark.asyncio
async def test_state_repository_fetches_from_vault_when_shm_reports_overflow():
    repo = StateRepository(is_vault_owner=False)
    repo._shm = SimpleNamespace(read=AsyncMock(return_value={"_state_overflow": True, "version": 9}))
    repo._current = AuraState()
    repo._current.version = 4

    hydrated = AuraState()
    hydrated.version = 9

    async def _fetch():
        repo._current = hydrated
        return hydrated

    repo._fetch_state_from_vault = _fetch

    result = await repo.get_state()

    assert result is hydrated


@pytest.mark.asyncio
async def test_state_repository_accepts_hot_transport_snapshot_from_shm():
    owner = StateRepository(is_vault_owner=True)
    proxy = StateRepository(is_vault_owner=False)
    state = AuraState()
    state.version = 17
    state.cold.long_term_memory = ["x" * 4096 for _ in range(8)]

    snapshot_payload = json.loads(owner._serialize_transport_snapshot(state))
    proxy._shm = SimpleNamespace(read=AsyncMock(return_value=snapshot_payload))

    result = await proxy.get_state()

    assert result is not None
    assert result.version == 17


def test_integrity_monitor_scales_memory_thresholds_for_large_ram(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024 * 1024 * 1024),
    )

    monitor = SystemIntegrityMonitor()

    assert monitor._memory_warning_mb > 13_000
    assert monitor._memory_critical_mb > 22_000


def test_integrity_monitor_treats_16gb_process_as_warning_on_64gb_host(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda: 8)

    monitor = SystemIntegrityMonitor()
    monitor._proc = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=16 * 1024 * 1024 * 1024),
        memory_percent=lambda: 25.0,
        cpu_percent=lambda interval=0.1: 24.0,
    )
    monkeypatch.setattr(monitor, "_get_thermal_level", lambda: 0)

    report = IntegrityReport()
    monitor._check_resources(report)

    assert not report.errors
    assert any("High memory usage" in warning for warning in report.warnings)


def test_state_shm_size_scales_for_large_ram(monkeypatch):
    import psutil

    monkeypatch.delenv("AURA_STATE_SHM_BYTES", raising=False)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024 * 1024 * 1024),
    )

    assert get_state_shm_size_bytes() == 16 * 1024 * 1024


@pytest.mark.asyncio
async def test_task_tracker_marks_tracked_tasks_as_supervised():
    tracker = TaskTracker(name="ServerRuntime")

    async def _noop():
        await asyncio.sleep(0)

    task = tracker.create_task(_noop(), name="runtime.noop")
    await task

    assert getattr(task, "_aura_supervised", False) is True
    assert getattr(task, "_aura_task_tracker", "") == "ServerRuntime"


@pytest.mark.asyncio
async def test_stability_guardian_background_task_is_supervised():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))

    await guardian.start()
    await asyncio.sleep(0)

    assert guardian._task is not None
    assert getattr(guardian._task, "_aura_supervised", False) is True

    await guardian.stop()


@pytest.mark.asyncio
async def test_proactive_manager_background_task_is_supervised(monkeypatch):
    manager = ProactiveCommunicationManager()
    release = asyncio.Event()

    async def _hold():
        await release.wait()

    manager._process_messages = _hold

    await manager.start()
    await asyncio.sleep(0)

    assert manager._background_task is not None
    assert getattr(manager._background_task, "_aura_supervised", False) is True

    release.set()
    await manager.stop()


@pytest.mark.asyncio
async def test_autonomous_initiative_loop_background_tasks_are_supervised(monkeypatch):
    from core.autonomous_initiative_loop import AutonomousInitiativeLoop

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    release = asyncio.Event()

    async def _hold():
        await release.wait()

    loop._world_watcher_loop = _hold
    loop._knowledge_gap_monitor_loop = _hold
    monkeypatch.setattr("core.autonomous_initiative_loop.ServiceContainer.get", lambda *_args, **_kwargs: None)

    await loop.start()
    await asyncio.sleep(0)

    assert loop._world_task is not None
    assert loop._knowledge_task is not None
    assert getattr(loop._world_task, "_aura_supervised", False) is True
    assert getattr(loop._knowledge_task, "_aura_supervised", False) is True

    release.set()
    await asyncio.sleep(0)
    await loop.stop()


@pytest.mark.asyncio
async def test_server_websocket_manager_tracks_and_cancels_heartbeat_tasks():
    from interface import server as server_module

    class _FakeWebSocket:
        async def accept(self):
            return None

        async def send_text(self, _payload):
            return None

        async def send_json(self, _payload):
            return None

    ws = _FakeWebSocket()
    await server_module.ws_manager.connect(ws)

    assert ws in server_module.ws_manager._pump_tasks
    assert ws in server_module.ws_manager._heartbeat_tasks
    assert getattr(server_module.ws_manager._pump_tasks[ws], "_aura_supervised", False) is True
    assert getattr(server_module.ws_manager._heartbeat_tasks[ws], "_aura_supervised", False) is True

    await server_module.ws_manager.disconnect(ws)
    await asyncio.sleep(0)

    assert ws not in server_module.ws_manager._pump_tasks
    assert ws not in server_module.ws_manager._heartbeat_tasks


@pytest.mark.asyncio
async def test_robust_lock_cancellation_clears_watchdog_entry(monkeypatch):
    class _Watchdog:
        def __init__(self):
            self.active = {}

        def report_acquire_start(self, lock_id, name):
            self.active[lock_id] = name

        def report_acquire_success(self, lock_id):
            self.active[lock_id] = self.active.get(lock_id, "lock")

        def report_release(self, lock_id):
            self.active.pop(lock_id, None)

    watchdog = _Watchdog()
    monkeypatch.setattr("core.resilience.lock_watchdog.get_lock_watchdog", lambda: watchdog)

    lock = RobustLock("ServerRuntime.CancellationLock")
    assert lock._lock.acquire(timeout=0.0) is True

    task = asyncio.create_task(lock.acquire_robust(timeout=0.05, max_retries=1))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert watchdog.active == {}
    if lock._lock.locked():
        lock._lock.release()


@pytest.mark.asyncio
async def test_lock_watchdog_runs_recovery_callback_when_lock_stalls():
    from core.resilience.lock_watchdog import get_lock_watchdog

    watchdog = get_lock_watchdog()
    await watchdog.stop()
    watchdog._active_locks.clear()
    original_threshold = watchdog._threshold
    original_interval = watchdog._check_interval
    original_cooldown = watchdog._intervention_cooldown
    recovered = asyncio.Event()

    def _recover():
        watchdog.report_release("runtime-watchdog-test")
        recovered.set()

    try:
        watchdog._threshold = 0.01
        watchdog._check_interval = 0.01
        watchdog._intervention_cooldown = 0.01
        watchdog.report_acquire_start("runtime-watchdog-test", "RuntimeLock", on_stall=_recover)
        watchdog.report_acquire_success("runtime-watchdog-test")
        watchdog._active_locks["runtime-watchdog-test"].start_time -= 1.0

        watchdog.start()
        await asyncio.wait_for(recovered.wait(), timeout=0.25)
    finally:
        await watchdog.stop()
        watchdog._active_locks.clear()
        watchdog._threshold = original_threshold
        watchdog._check_interval = original_interval
        watchdog._intervention_cooldown = original_cooldown

    assert recovered.is_set()


@pytest.mark.asyncio
async def test_intent_classifier_queue_returns_passthrough_on_worker_failure(monkeypatch):
    queue = IntentClassifierQueue(max_queue=2)

    async def _boom(_message, _context):
        raise RuntimeError("classification failed")

    monkeypatch.setattr(queue, "_classify_via_llm", _boom)

    await queue.start()
    result = await queue.classify("hello")
    await queue.stop()

    assert result.kind == RouteKind.PASSTHROUGH
    assert queue._queue.qsize() == 0


@pytest.mark.asyncio
async def test_state_repository_owner_commit_avoids_deepcopy_hot_path(monkeypatch, tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "aura_state.db"), is_vault_owner=True)
    repo._current = AuraState()

    next_state = repo._current.derive("foreground_commit", origin="test")
    next_state.cognition.current_objective = "live handoff"

    def _unexpected_deepcopy(_obj, *_args, **_kwargs):
        raise AssertionError("deepcopy should not run on owner commit path")

    monkeypatch.setattr("core.state.state_repository.copy.deepcopy", _unexpected_deepcopy)

    await repo.commit(next_state, "foreground_commit", trace_id="trace-live")

    queued = repo._mutation_queue.get_nowait()
    repo._mutation_queue.task_done()

    assert queued["type"] == "commit"
    assert queued["state"] is next_state
    assert queued["trace_id"] == "trace-live"


@pytest.mark.asyncio
async def test_state_repository_owner_commit_coalesces_stale_queue_entries(tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "aura_state.db"), is_vault_owner=True)
    repo._current = AuraState()

    for idx in range(repo._mutation_queue_maxsize):
        stale_state = repo._current.derive(f"stale_{idx}", origin="test")
        repo._mutation_queue.put_nowait(
            {
                "type": "commit",
                "state": stale_state,
                "cause": "stale",
                "trace_id": f"stale-{idx}",
                "ts": time.time(),
            }
        )

    latest_state = repo._current.derive("latest", origin="test")
    await repo.commit(latest_state, "latest", trace_id="latest-trace")

    assert repo._mutation_queue.qsize() == 1
    queued = repo._mutation_queue.get_nowait()
    repo._mutation_queue.task_done()
    assert queued["trace_id"] == "latest-trace"
    assert queued["state"] is latest_state


@pytest.mark.asyncio
async def test_state_repository_process_commit_writes_inline_without_spawning_background_tasks(service_container, monkeypatch, tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "aura_state.db"), is_vault_owner=False)
    repo._current = AuraState()
    repo._shm = object()
    repo._commit_to_db = AsyncMock()
    repo._sync_to_shm = AsyncMock()

    allowed_gate = SimpleNamespace(approve_state_mutation=AsyncMock(return_value=(True, "approved_by_test")))
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *args, **kwargs: allowed_gate)

    next_state = repo._current.derive("inline_commit", origin="test")
    next_state.cognition.current_objective = "inline persistence"

    await repo._process_commit(next_state, "inline_commit")

    repo._sync_to_shm.assert_awaited_once()
    sync_args = repo._sync_to_shm.await_args.args
    assert sync_args[0] is next_state
    assert isinstance(sync_args[1], str)
    repo._commit_to_db.assert_awaited_once()


@pytest.mark.asyncio
async def test_state_repository_background_commit_prefers_bounded_snapshot_serialization(service_container, monkeypatch, tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "aura_state.db"), is_vault_owner=False)
    repo._current = AuraState()
    repo._shm = object()
    repo._commit_to_db = AsyncMock()
    repo._sync_to_shm = AsyncMock()

    allowed_gate = SimpleNamespace(approve_state_mutation=AsyncMock(return_value=(True, "approved_by_test")))
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *args, **kwargs: allowed_gate)
    monkeypatch.setattr(repo, "_serialize", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full serialize should be skipped")))
    monkeypatch.setattr(repo, "_serialize_transport_snapshot", lambda state: json.dumps({"state_id": state.state_id, "version": state.version, "_transport_snapshot_kind": "hot"}))

    next_state = repo._current.derive("background_compaction", origin="autonomous_thought")
    next_state.cognition.current_objective = "Background continuity maintenance"

    await repo._process_commit(next_state, "background_compaction")

    repo._sync_to_shm.assert_awaited_once()
    repo._commit_to_db.assert_awaited_once()
    payload = repo._commit_to_db.await_args.args[1]
    assert '"_transport_snapshot_kind": "hot"' in payload


@pytest.mark.asyncio
async def test_state_repository_repair_runtime_restarts_consumer_and_coalesces_queue(monkeypatch, tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "aura_state.db"), is_vault_owner=True)
    repo._current = AuraState()
    repo._is_processing = True

    finished = asyncio.create_task(asyncio.sleep(0))
    await finished
    repo._consumer_task = finished

    started = asyncio.Event()
    release = asyncio.Event()

    async def _hold_consumer():
        started.set()
        await release.wait()

    repo._mutation_consumer_loop = _hold_consumer
    repo._ensure_db = AsyncMock(return_value=object())

    threshold = max(1, int(repo._mutation_queue_maxsize * 0.75))
    for idx in range(threshold):
        repo._mutation_queue.put_nowait(
            {
                "type": "commit",
                "state": repo._current.derive(f"queued_{idx}", origin="test"),
                "cause": "queued",
                "trace_id": f"queued-{idx}",
                "ts": time.time(),
            }
        )

    result = await repo.repair_runtime()
    await asyncio.wait_for(started.wait(), timeout=0.2)

    try:
        assert "restarted_consumer" in result["actions"]
        assert "reconnected_db" in result["actions"]
        assert any(action.startswith("coalesced_queue:") for action in result["actions"])
        assert repo._mutation_queue.qsize() == 1
        assert repo._dropped_commit_count > 0
    finally:
        release.set()
        if repo._consumer_task and not repo._consumer_task.done():
            repo._consumer_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await repo._consumer_task


@pytest.mark.asyncio
async def test_mind_tick_background_task_is_supervised(monkeypatch):
    class _Watchdog:
        def register_component(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("infrastructure.watchdog.get_watchdog", lambda: _Watchdog())

    tick = MindTick(SimpleNamespace(state_repo=SimpleNamespace(get_current=AsyncMock(return_value=None))))
    release = asyncio.Event()

    async def _hold():
        await release.wait()

    tick._run_loop = _hold

    await tick.start()
    await asyncio.sleep(0)

    assert tick._task is not None
    assert getattr(tick._task, "_aura_supervised", False) is True

    release.set()
    await tick.stop()


@pytest.mark.asyncio
async def test_mind_tick_missing_state_uses_single_backoff_path(monkeypatch, caplog):
    class _Watchdog:
        def register_component(self, *_args, **_kwargs):
            return None

        def heartbeat(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("infrastructure.watchdog.get_watchdog", lambda: _Watchdog())

    tick = MindTick(
        SimpleNamespace(
            state_repo=SimpleNamespace(get_current=AsyncMock(return_value=None)),
            status=SimpleNamespace(cycle_count=0),
        )
    )
    tick._running = True
    sleeps = []

    async def _fake_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 3:
            tick._running = False

    monkeypatch.setattr("core.mind_tick.asyncio.sleep", _fake_sleep)

    with caplog.at_level("WARNING"):
        await tick._run_loop()

    missing_state_logs = [
        record for record in caplog.records
        if "No current state found. Deferring tick" in record.getMessage()
    ]
    assert sleeps == pytest.approx([2.0, 4.0, 5.0], rel=0.0, abs=0.05)
    assert len(missing_state_logs) == 1


@pytest.mark.asyncio
async def test_state_repository_proxy_reads_from_shm():
    repo = StateRepository(is_vault_owner=False)
    expected = AuraState()
    repo._shm = SimpleNamespace(read=AsyncMock(return_value={"state_id": "from-shm"}))
    repo._deserialize = lambda _payload: expected
    repo._fetch_state_from_vault = AsyncMock()

    state = await repo.get_state()

    assert state is expected
    assert repo._current is expected
    repo._fetch_state_from_vault.assert_not_awaited()


@pytest.mark.asyncio
async def test_state_repository_proxy_falls_back_to_vault_when_shm_empty():
    repo = StateRepository(is_vault_owner=False)
    expected = AuraState()
    repo._shm = SimpleNamespace(read=AsyncMock(return_value=None))

    async def _fetch():
        repo._current = expected

    repo._fetch_state_from_vault = AsyncMock(side_effect=_fetch)

    state = await repo.get_state()

    assert state is expected
    repo._fetch_state_from_vault.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_manager_schedules_restart_on_captured_loop(monkeypatch):
    manager = ProcessManager()
    running_loop = asyncio.get_running_loop()
    manager._event_loop = running_loop

    restarted = {}

    async def _restart():
        restarted["called"] = True
        return True

    fake_process = SimpleNamespace(
        stats=SimpleNamespace(restarts=0),
        config=SimpleNamespace(max_restarts=1),
        state=None,
        get_status=lambda: {"state": "running", "alive": False},
        restart=_restart,
        stop=lambda force=False: True,
    )
    manager.processes["worker"] = fake_process

    scheduled = {}

    def _run_threadsafe(coro, loop):
        scheduled["loop"] = loop
        task = running_loop.create_task(coro)
        scheduled["task"] = task
        return task

    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", _run_threadsafe)

    manager._check_all_processes()
    await scheduled["task"]

    assert scheduled["loop"] is running_loop
    assert restarted["called"] is True


@pytest.mark.asyncio
async def test_sleep_trigger_primes_missing_last_user_baseline():
    orch = SimpleNamespace(
        status=SimpleNamespace(is_processing=False),
        start_time=time.time(),
        _last_user_interaction_time=0.0,
    )
    trigger = AutonomousSleepTrigger(orchestrator=orch)

    called = False

    async def _fake_initiate(*args, **kwargs):
        nonlocal called
        called = True

    trigger._initiate_sleep = _fake_initiate

    await trigger._evaluate()

    assert orch._last_user_interaction_time > 0.0
    assert called is False


@pytest.mark.asyncio
async def test_backup_manager_vacuum_discovers_sqlite_files_without_connection_pool_state(monkeypatch, tmp_path):
    from core.backup import BackupManager

    seen = []
    db_path = tmp_path / "nested" / "aura_state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch()

    manager = BackupManager()
    manager.data_dir = tmp_path
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        "core.resilience.database_coordinator.get_db_coordinator",
        lambda: SimpleNamespace(_connections={}),
    )
    monkeypatch.setattr(
        BackupManager,
        "_vacuum_database_sync",
        staticmethod(lambda path: seen.append(path)),
    )

    result = await manager.run_vacuum()

    assert result is True
    assert seen == [db_path]
    assert manager._last_vacuum_at > 0.0


def test_vector_memory_prunes_stale_neutral_entries_after_hard_expiry(monkeypatch, tmp_path):
    from core.memory.vector_memory import VectorMemory

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_args, **_kwargs):
            return None

        def executemany(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr("core.memory.vector_memory._CHROMA_AVAILABLE", False)
    memory = VectorMemory(collection_name="test_vectors", persist_directory=str(tmp_path))
    memory._fallback_mode = True
    memory.db = SimpleNamespace(get_connection=lambda: _FakeConn())

    now = time.time()
    memory._store = [
        {
            "id": "stale-neutral",
            "content": "old neutral memory",
            "metadata": {
                "timestamp": now - (46 * 86400),
                "last_accessed": now - (46 * 86400),
                "valence": 0.0,
            },
        },
        {
            "id": "recent-neutral",
            "content": "recent neutral memory",
            "metadata": {
                "timestamp": now - (5 * 86400),
                "last_accessed": now - (5 * 86400),
                "valence": 0.0,
            },
        },
    ]

    pruned = memory.prune_low_salience(threshold_days=30, min_salience=-0.2)

    assert pruned == 1
    assert [item["id"] for item in memory._store] == ["recent-neutral"]


def test_health_check_allows_async_server_mode_without_thread():
    status = SimpleNamespace(
        running=True,
        initialized=True,
        last_error="",
        healthy=True,
    )
    orch = SimpleNamespace(status=status, stats={}, _thread=None)

    assert RobustOrchestrator.health_check(orch) is True
    assert orch.status.healthy is True


@pytest.mark.asyncio
async def test_motivation_engine_tracks_autonomous_intention(monkeypatch):
    calls = {}

    class CognitiveStub:
        async def process_autonomous_intention(self, intention):
            calls["goal"] = intention.goal

    engine = MotivationEngine()
    engine.cognitive = CognitiveStub()

    import core.motivation.engine as motivation_module

    def _record_create_task(coro, name=None):
        task = asyncio.create_task(coro, name=name)
        calls["name"] = name
        calls["task"] = task
        return task

    monkeypatch.setattr(motivation_module.task_tracker, "create_task", _record_create_task)

    intention = Intention(DriveType.CURIOSITY, "Investigate anomaly", urgency=0.5)
    await engine._dispatch_intention(intention)
    await calls["task"]

    assert calls["name"] == "MotivationEngine.process_autonomous_intention"
    assert calls["goal"] == "Investigate anomaly"


@pytest.mark.asyncio
async def test_motivation_update_recovers_social_drive_during_high_energy_conversation(monkeypatch):
    from core.phases.motivation_update import MotivationUpdatePhase

    phase = MotivationUpdatePhase(SimpleNamespace())
    state = AuraState()
    before = state.motivation.budgets["social"]["level"]
    state.cognition.conversation_energy = 1.0
    state.motivation.last_tick = time.time() - 300.0

    authority = SimpleNamespace(propose_initiative_to_state=AsyncMock(return_value=(state, {"reason": "noop"})))
    monkeypatch.setattr("core.phases.motivation_update.ServiceContainer.has", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("core.phases.motivation_update.get_executive_authority", lambda *_args, **_kwargs: authority)

    next_state = await phase.execute(state)

    assert next_state.motivation.budgets["social"]["level"] > before


@pytest.mark.asyncio
async def test_memory_synthesizer_dedupes_triggered_synthesis_tasks():
    synthesizer = MemorySynthesizer()
    synthesizer.running = True
    synthesizer._new_since_synthesis = synthesizer.SYNTHESIS_TRIGGER_COUNT - 1
    release = asyncio.Event()

    async def _hold():
        await release.wait()

    synthesizer._run_synthesis = _hold  # type: ignore[method-assign]

    synthesizer.notify_new_memory()
    await asyncio.sleep(0)
    first_task = synthesizer._adhoc_synthesis_task

    synthesizer.notify_new_memory()
    await asyncio.sleep(0)

    assert first_task is not None
    assert synthesizer._adhoc_synthesis_task is first_task

    release.set()
    await first_task


@pytest.mark.asyncio
async def test_managed_process_health_monitor_uses_supervised_nonblocking_loop():
    process = ManagedProcess(
        ProcessConfig(
            name="test-process",
            target=lambda: None,
            health_check_interval=1,
        )
    )
    called = asyncio.Event()

    def _check_health():
        called.set()
        process._stop_health_check.set()

    process._check_health = _check_health  # type: ignore[method-assign]

    await process._start_health_monitoring()

    assert process._health_check_task is not None
    assert getattr(process._health_check_task, "_aura_supervised", False) is True

    await asyncio.wait_for(called.wait(), timeout=1.0)
    await asyncio.wait_for(process._health_check_task, timeout=1.0)


@pytest.mark.asyncio
async def test_memory_governor_skips_repeated_prune_on_stable_high_rss(monkeypatch):
    from core.resilience.memory_governor import MemoryGovernor

    governor = MemoryGovernor(SimpleNamespace())
    current_rss_mb = governor.threshold_prune + 256.0
    governor._last_prune_action_time = time.monotonic() - governor.prune_cooldown_s - 1.0
    governor._last_prune_rss_mb = current_rss_mb
    governor._prune_memory = AsyncMock()

    monkeypatch.setattr(
        governor,
        "_proc",
        SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=current_rss_mb * 1024 * 1024)),
    )
    monkeypatch.setattr(
        "core.resilience.memory_governor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=50.0),
    )

    await governor._enforce_policy()

    governor._prune_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_governor_prunes_when_rss_growth_crosses_hysteresis(monkeypatch):
    from core.resilience.memory_governor import MemoryGovernor

    governor = MemoryGovernor(SimpleNamespace())
    current_rss_mb = governor.threshold_prune + governor.prune_hysteresis_mb + 256.0
    governor._last_prune_action_time = time.monotonic() - governor.prune_cooldown_s - 1.0
    governor._last_prune_rss_mb = governor.threshold_prune
    governor._prune_memory = AsyncMock()

    monkeypatch.setattr(
        governor,
        "_proc",
        SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=current_rss_mb * 1024 * 1024)),
    )
    monkeypatch.setattr(
        "core.resilience.memory_governor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=50.0),
    )

    await governor._enforce_policy()

    governor._prune_memory.assert_awaited_once()
    assert governor._last_prune_action_time > 0.0
    assert governor._last_prune_rss_mb == current_rss_mb


def test_memory_governor_only_counts_descendant_runtime_workers(monkeypatch):
    from core.resilience.memory_governor import MemoryGovernor

    governor = MemoryGovernor(SimpleNamespace())

    managed_child = SimpleNamespace(pid=111)
    monkeypatch.setattr(
        governor,
        "_proc",
        SimpleNamespace(children=lambda recursive=True: [managed_child]),
    )

    class _Proc:
        def __init__(self, pid, cmdline):
            self.info = {
                "pid": pid,
                "cmdline": cmdline,
                "memory_info": SimpleNamespace(rss=128 * 1024 * 1024),
            }

    monkeypatch.setattr(
        "core.resilience.memory_governor.psutil.process_iter",
        lambda *_args, **_kwargs: iter(
            [
                _Proc(111, ["llama-server", "--model", "demo.gguf"]),
                _Proc(222, ["llama-server", "--model", "someone-elses.gguf"]),
                _Proc(333, ["python", "worker.py"]),
            ]
        ),
    )

    managed = list(governor._iter_managed_runtime_processes())

    assert [proc.info["pid"] for proc in managed] == [111]


def test_stability_guardian_treats_single_slow_tick_as_non_degraded_signal():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))
    guardian._tick_times.append((time.time() - 1.0, 19097.0))
    guardian._last_tick_at = time.time() - 1.0

    result = guardian._check_tick_rate()

    assert result.healthy is True
    assert "not sustained" in result.message.lower()


def test_collect_liquid_state_payload_prefers_runtime_signal_over_zero_stub():
    from interface import server as server_module

    payload = server_module._collect_liquid_state_payload(
        {"energy": 0, "curiosity": 0, "frustration": 0, "confidence": 0},
        runtime_state={"affect": {"energy": 64, "curiosity": 24, "valence": -1.0}},
        homeostasis_data={"will_to_live": 0.87, "curiosity": 0.2},
    )

    assert payload["energy"] == 64.0
    assert payload["curiosity"] == 24.0
    assert payload["frustration"] == 100.0
    assert payload["confidence"] == 87.0


@pytest.mark.asyncio
async def test_backup_manager_defers_maintenance_jobs_until_after_boot(monkeypatch):
    registered = []

    class _Scheduler:
        async def register(self, spec):
            registered.append(spec)

    monkeypatch.setattr("core.scheduler.scheduler", _Scheduler())

    manager = BackupManager()
    before = time.monotonic()

    await manager.on_start_async()

    assert len(registered) == 2
    assert {spec.name for spec in registered} == {"periodic_db_vacuum", "periodic_state_backup"}
    assert all(spec.last_run >= before for spec in registered)
    assert all(spec.tick_interval for spec in registered)


@pytest.mark.asyncio
async def test_backup_manager_skips_maintenance_during_active_runtime(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "recent_user_30",
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: SimpleNamespace() if name == "orchestrator" else default,
    )

    manager = BackupManager()

    assert await manager.run_vacuum() is False
    assert await manager.create_backup() is None


@pytest.mark.asyncio
async def test_liquid_substrate_idle_throttles_without_recent_user(monkeypatch, tmp_path):
    import psutil

    from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: (
            SimpleNamespace(_last_user_interaction_time=time.time() - 700.0)
            if name == "orchestrator"
            else default
        ),
    )

    substrate = LiquidSubstrate(SubstrateConfig(state_file=tmp_path / "substrate_state.npy"))

    dt = await substrate._apply_battery_throttling()

    assert dt == pytest.approx(substrate.config.time_constant * 4.0)
    assert substrate.current_update_rate == pytest.approx(max(2.0, substrate.config.update_rate / 4.0))
