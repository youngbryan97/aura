import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.orchestrator.handlers.shutdown import orchestrator_shutdown
from core.orchestrator.mixins.boot.boot_cognitive import BootCognitiveMixin
from core.orchestrator.mixins.boot.boot_resilience import BootResilienceMixin
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin


class _BootProbe(BootCognitiveMixin):
    cognition = None


class _ResilienceProbe(BootResilienceMixin):
    _actor_bus = None
    actor_bus = None
    supervisor = None
    reply_queue = True
    output_gate = None
    emit_spontaneous_message = None

    def __init__(self):
        self.status = SimpleNamespace(temporal_drift_s=0.0)


class _OutputProbe(OutputFormatterMixin):
    def __init__(self, emitter):
        self.cognitive_engine = SimpleNamespace(_emit_thought=emitter)


@pytest.mark.asyncio
async def test_init_cognitive_core_awaits_async_setup(monkeypatch):
    setup = AsyncMock()
    cognitive_engine = SimpleNamespace(setup=setup)
    capability_engine = object()

    def _get(name, default=None):
        if name == "cognitive_engine":
            return cognitive_engine
        if name == "capability_engine":
            return capability_engine
        return default

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_cognitive.ServiceContainer.get",
        staticmethod(_get),
    )

    await _BootProbe()._init_cognitive_core()

    setup.assert_awaited_once_with(registry=capability_engine, router=capability_engine)


@pytest.mark.asyncio
async def test_emit_thought_stream_schedules_async_emitters():
    emitter = AsyncMock()
    probe = _OutputProbe(emitter)

    probe._emit_thought_stream("hello")
    await asyncio.sleep(0)

    emitter.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_start_state_vault_actor_uses_actor_bus_request_for_handshake(monkeypatch):
    started = object()

    class _Supervisor:
        def __init__(self):
            self.specs = []

        def add_actor(self, spec):
            self.specs.append(spec)

        def start_actor(self, name):
            assert name == "state_vault"
            return started

    class _Bus:
        def __init__(self):
            self.actors = {}
            self.requests = []

        def has_actor(self, name):
            return name in self.actors

        def add_actor(self, name, connection):
            self.actors[name] = connection

        async def request(self, actor, msg_type, payload, timeout=0):
            self.requests.append((actor, msg_type, payload, timeout))
            return {"type": "pong", "ts": 123.0}

    bus = _Bus()
    supervisor = _Supervisor()

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda name, default=None: bus if name == "actor_bus" else default),
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = supervisor

    await probe._start_state_vault_actor()

    assert bus.actors["state_vault"] is started
    assert bus.requests == [
        ("state_vault", "ping", {"source": "boot_resilience", "attempt": 1}, 2.0)
    ]


@pytest.mark.asyncio
async def test_start_state_vault_actor_fallback_ping_uses_request_wire_format(monkeypatch):
    class _Pipe:
        def __init__(self):
            self.sent = []
            self.last_request_id = None

        def send(self, raw):
            payload = json.loads(raw)
            self.sent.append(payload)
            self.last_request_id = payload["request_id"]

        def poll(self, timeout):
            return True

        def recv(self):
            return json.dumps(
                {
                    "response_to": self.last_request_id,
                    "payload": {"type": "pong", "ts": 456.0},
                }
            )

    pipe = _Pipe()

    class _Supervisor:
        def add_actor(self, spec):
            self.spec = spec

        def start_actor(self, name):
            assert name == "state_vault"
            return pipe

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = _Supervisor()

    await probe._start_state_vault_actor()

    assert pipe.sent
    assert pipe.sent[0]["type"] == "ping"
    assert pipe.sent[0]["is_request"] is True
    assert pipe.sent[0]["payload"]["source"] == "boot_resilience"


@pytest.mark.asyncio
async def test_calculate_temporal_drift_routes_recovery_through_unified_will(tmp_path, monkeypatch):
    from core.orchestrator.mixins.boot import boot_resilience

    heartbeat_path = tmp_path / "heartbeat"
    heartbeat_path.write_text(str(time.time() - 7200.0))

    monkeypatch.setattr(
        boot_resilience.config,
        "paths",
        SimpleNamespace(home_dir=tmp_path),
    )

    probe = _ResilienceProbe()
    probe.emit_spontaneous_message = AsyncMock(
        return_value={"ok": True, "action": "released", "target": "secondary"}
    )
    probe.output_gate = SimpleNamespace(emit=AsyncMock())

    probe._calculate_temporal_drift()
    await asyncio.sleep(0)

    probe.emit_spontaneous_message.assert_awaited_once()
    _, kwargs = probe.emit_spontaneous_message.await_args
    assert kwargs["origin"] == "recovery"
    assert kwargs["metadata"]["visible_presence"] is True
    assert kwargs["metadata"]["trigger"] == "temporal_drift_recovery"
    probe.output_gate.emit.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_shutdown_requests_graceful_state_vault_stop_before_bus_stop(monkeypatch):
    class _Bus:
        def __init__(self):
            self.calls = []

        def has_actor(self, name):
            return name == "state_vault"

        async def request(self, actor, msg_type, payload, timeout=0):
            self.calls.append(("request", actor, msg_type, payload, timeout))
            return None

        async def stop(self):
            self.calls.append(("stop",))

    class _Supervisor:
        def __init__(self):
            self.stop_calls = 0
            self._running = True

        def is_actor_running(self, name):
            if name != "state_vault":
                return False
            was_running = self._running
            self._running = False
            return was_running

        async def stop(self):
            self.stop_calls += 1

    bus = _Bus()
    supervisor = _Supervisor()
    state_repo = SimpleNamespace(
        get_current=AsyncMock(return_value=None),
        close=AsyncMock(),
        _transport_has_vault=lambda: True,
        is_vault_owner=False,
    )
    service_shutdown = AsyncMock()
    event_bus_shutdown = AsyncMock()

    monkeypatch.setattr(
        "core.resilience.snapshot_manager.SnapshotManager",
        lambda _orch: SimpleNamespace(freeze=lambda: None),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.shutdown",
        service_shutdown,
    )
    monkeypatch.setattr(
        "core.event_bus.get_event_bus",
        lambda: SimpleNamespace(shutdown=event_bus_shutdown),
    )
    monkeypatch.setattr(
        "core.utils.task_tracker.get_task_tracker",
        lambda: SimpleNamespace(shutdown=lambda timeout=3.0: None),
    )

    orch = SimpleNamespace(
        status=SimpleNamespace(running=True, is_processing=True),
        state_repo=state_repo,
        _actor_bus=bus,
        _supervisor_tree=supervisor,
        _publish_status=lambda _payload: None,
        _save_state=lambda _cause: None,
        _stop_event=None,
        kernel_interface=None,
    )

    await orchestrator_shutdown(orch)

    assert bus.calls[0] == (
        "request",
        "state_vault",
        "stop",
        {"source": "orchestrator_shutdown", "reason": "graceful_shutdown"},
        2.0,
    )
    assert bus.calls[1] == ("stop",)
    state_repo.close.assert_awaited_once()
    service_shutdown.assert_awaited_once()
    event_bus_shutdown.assert_awaited_once()
    assert supervisor.stop_calls == 1
